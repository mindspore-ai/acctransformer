# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
import mindspore
import numpy as np
import pytest
from mindspore import Tensor
from mindspore import ops
from tbe import tik

from flash_attention_test_agent import FlashAttentionTestAgent


class NonNormalizedSoftmaxComputer(FlashAttentionTestAgent):
    def define_ios(self, *args, **kwargs):
        M, dim, N, dtype = (
            kwargs.get("M"),
            kwargs.get("dim"),
            kwargs.get("N"),
            kwargs.get("dtype"),
        )
        self.M = M
        self.N = N
        self.dim = dim
        self.dtype = dtype
        self.Sim_gm = self.tik_instance.Tensor(dtype, (M, N), name="Sim_gm", scope=tik.scope_gm)
        self.m_gm = self.tik_instance.Tensor(dtype, (M,), name="m_gm", scope=tik.scope_gm)
        self.l_gm = self.tik_instance.Tensor(dtype, (M,), name="l_gm", scope=tik.scope_gm)
        self.non_normalized_P_gm = self.tik_instance.Tensor(
            dtype, (M, N), name="non_normalized_P_gm", scope=tik.scope_gm
        )

    def compute(self, *args, **kwargs):
        M_alig = (self.M + 16 - 1) // 16 * 16
        N_alig = (self.N + 16 - 1) // 16 * 16
        Sim_ub = self.tik_instance.Tensor(
            self.dtype, (M_alig, N_alig), name="Sim_ub", scope=tik.scope_ubuf
        )
        self.fa.tik_instance.data_move(Sim_ub, self.Sim_gm[0], 0, 1, self.M * self.N // 16, 0, 0)
        Sim_ub = self.fa.tik_ops_utils.MK_TO_K1MK0(Sim_ub)
        mij_ub = self.tik_instance.Tensor(self.dtype, (M_alig,), scope=tik.scope_ubuf, name="mij_ub")
        lij_ub = self.tik_instance.Tensor(self.dtype, (M_alig,), scope=tik.scope_ubuf, name="lij_ub")
        self.fa.Bc = self.N
        self.fa.precision_type = self.dtype
        ones_ub = self.tik_instance.Tensor(self.dtype, (self.N, 16), name="ones_ub", scope=tik.scope_ubuf)
        self.tik_instance.h_duplicate(ones_ub, 1.0)
        ones_l1 = self.tik_instance.Tensor(self.dtype, (self.N, 16), name="ones_l1", scope=tik.scope_cbuf)
        self.tik_instance.data_move(ones_l1, ones_ub, 0, 1, self.N, 0, 0)
        self.fa.ones_l1 = ones_l1
        non_normalized_P_ub, m_ub, l_ub = self.fa.softmax_compute(
            Sim_ub, mij_ub, lij_ub, self.M, self.N
        )
        non_normalized_P_ub = self.fa.tik_ops_utils.N1MN0_TO_MN(non_normalized_P_ub)
        self.fa.tik_instance.data_move(
            self.non_normalized_P_gm[0], non_normalized_P_ub, 0, 1, self.M * self.N // 16, 0, 0
        )
        self.fa.tik_ops_utils.move_vector_from_ub_to_gm(self.m_gm, m_ub, 0, self.M)
        self.fa.tik_ops_utils.move_vector_from_ub_to_gm(self.l_gm, l_ub, 0, self.M)

        self.tik_instance.BuildCCE(
            kernel_name="mock_flash_attention",
            inputs=[self.Sim_gm],
            outputs=[self.non_normalized_P_gm, self.m_gm, self.l_gm],
        )
        return self.tik_instance


class NormalizedSoftmaxComputer(FlashAttentionTestAgent):
    def define_ios(self, *args, **kwargs):
        M, dim, N, dtype = (
            kwargs.get("M"),
            kwargs.get("dim"),
            kwargs.get("N"),
            kwargs.get("dtype"),
        )
        self.M = M
        self.N = N
        self.dim = dim
        self.dtype = dtype
        self.Sim_gm = self.tik_instance.Tensor(dtype, (M, N), name="Sim_gm", scope=tik.scope_gm)
        self.P_gm = self.tik_instance.Tensor(dtype, (M, N), name="P_gm", scope=tik.scope_gm)

    def compute(self, *args, **kwargs):
        M_alig = (self.M + 16 - 1) // 16 * 16
        N_alig = (self.N + 16 - 1) // 16 * 16
        Sim_ub = self.tik_instance.Tensor(
            self.dtype, (M_alig, N_alig), name="Sim_ub", scope=tik.scope_ubuf
        )
        self.fa.tik_instance.data_move(Sim_ub, self.Sim_gm[0], 0, 1, self.M * self.N // 16, 0, 0)

        # row_max(Sim)
        row_max_ub = self.tik_instance.Tensor(
            "float16", (self.M,), name="row_max_ub", scope=tik.scope_ubuf
        )
        self.tik_instance.h_reduce_max(row_max_ub, Sim_ub[0: self.M, 0: self.N], 1)

        # Sim - row_max(Sim)
        with self.tik_instance.for_range(0, self.M) as i:
            src_scalar = self.tik_instance.Scalar(init_value=row_max_ub[i], dtype="float16")
            self.tik_instance.h_sub(Sim_ub[i, :], Sim_ub[i, :], src_scalar)

        # fp16 -> fp32
        Sim_ub_fp32 = self.tik_instance.Tensor(
            "float32", (M_alig, N_alig), name="Sim_ub", scope=tik.scope_ubuf
        )
        self.tik_instance.h_cast(Sim_ub_fp32, Sim_ub, "none")

        # exp(Sim)
        self.tik_instance.h_exp(
            Sim_ub_fp32[0: self.M, 0: self.N], Sim_ub_fp32[0: self.M, 0: self.N]
        )

        # row_sum(Sim) fp32
        row_sum_ub = self.tik_instance.Tensor(
            "float32", (self.M,), name="row_sum_ub", scope=tik.scope_ubuf
        )
        self.tik_instance.h_reduce_sum(row_sum_ub, Sim_ub_fp32[0: self.M, 0: self.N], 1)

        # P = Sim / row_sum(Sim)
        for i in range(self.M):
            src_scalar = self.tik_instance.Scalar(init_value=row_sum_ub[i], dtype="float32")
            self.tik_instance.h_div(
                Sim_ub_fp32[i, 0: self.N], Sim_ub_fp32[i, 0: self.N], src_scalar
            )
        self.tik_instance.h_cast(
            Sim_ub[0: self.M, 0: self.N], Sim_ub_fp32[0: self.M, 0: self.N], "none"
        )

        self.fa.tik_instance.data_move(self.P_gm[0], Sim_ub, 0, 1, self.M * self.N // 16, 0, 0)

        self.tik_instance.BuildCCE(
            kernel_name="mock_flash_attention", inputs=[self.Sim_gm], outputs=[self.P_gm],
        )
        return self.tik_instance


class TestSoftmax:
    @staticmethod
    def ms_impl_non_normalized_softmax(Sim):
        # scale
        Sim = Tensor(Sim)
        # non normalized softmax
        m = ops.reduce_max(Sim, axis=-1)
        m = m.reshape((-1, 1))
        Sim = Sim - m
        # Sim = ops.cast(Sim, mindspore.float32)
        non_normed_P = ops.exp(Sim)
        l = ops.reduce_sum(non_normed_P, axis=-1)
        non_normed_P = ops.cast(non_normed_P, mindspore.float16)
        l = ops.cast(l, mindspore.float16)
        return non_normed_P, m, l

    @staticmethod
    def ms_impl_normalized_softmax(Sim):
        Sim = Tensor(Sim)
        # ms normalized softmax
        P = ops.softmax(Sim, axis=-1)
        return P

    @pytest.mark.parametrize(
        "M, N, dtype",
        [
            (64, 64, "float16"),
            (64, 48, "float16"),
            (128, 48, "float16"),
            (256, 96, "float16"),
            (256, 128, "float16"),
        ],
    )
    def test_non_normalized_softmax(self, M, N, dtype):
        dim = 16
        mfa = NonNormalizedSoftmaxComputer(**dict(M=M, N=N, dim=dim, dtype=dtype))
        tik_instance = mfa.compute()
        Sim = np.random.random([M, N]).astype(dtype)

        (ms_non_normed_P, ms_m, ms_l) = self.ms_impl_non_normalized_softmax(Sim)
        ms_non_normed_P = ms_non_normed_P.asnumpy()
        ms_m = ms_m.asnumpy().reshape(-1)
        ms_l = ms_l.asnumpy().reshape(-1)

        feed_dict = {"Sim_gm": Sim}
        (tik_non_normed_P, tik_m, tik_l) = tik_instance.tikdb.start_debug(
            feed_dict=feed_dict, interactive=False
        )

        print("\n############## ms result #################")
        print(ms_non_normed_P)
        print(ms_non_normed_P.shape)
        print(ms_non_normed_P.dtype)
        print(ms_m)
        print(ms_m.shape)
        print(ms_m.dtype)
        print(ms_l)
        print(ms_l.shape)
        print(ms_l.dtype)

        print("\n############## custom op result #################")
        print(tik_non_normed_P)
        print(tik_non_normed_P.shape)
        print(tik_non_normed_P.dtype)
        print(tik_m)
        print(tik_m.shape)
        print(tik_m.dtype)
        print(tik_l)
        print(tik_l.shape)
        print(tik_l.dtype)

        rtol = 1e-8
        assert np.allclose(ms_non_normed_P, tik_non_normed_P, rtol)
        assert np.allclose(ms_m, tik_m, rtol)
        assert np.allclose(ms_l, tik_l, rtol)

    @pytest.mark.parametrize(
        "M, N, dtype",
        [
            (64, 64, "float16"),
            (64, 48, "float16"),
            (128, 48, "float16"),
            (255, 96, "float16"),
            (256, 128, "float16"),
        ],
    )
    def test_normalized_softmax(self, M, N, dtype):
        dim = 16
        mfa = NormalizedSoftmaxComputer(**dict(M=M, N=N, dim=dim, dtype=dtype))
        tik_instance = mfa.compute()
        Sim = np.random.random([M, N]).astype(dtype)

        ms_P = self.ms_impl_normalized_softmax(Sim)
        ms_P = ms_P.asnumpy()

        feed_dict = {"Sim_gm": Sim}
        (tik_P,) = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)

        rtol = 1e-3
        print("\n############## ms result #################")
        print(ms_P)

        print(f"\n############## custom op result thresh: {rtol} ###########")
        print(tik_P)

        assert np.allclose(ms_P, tik_P, rtol)
