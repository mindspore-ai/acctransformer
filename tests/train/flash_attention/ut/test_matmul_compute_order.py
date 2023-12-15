# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
import mindspore
import numpy as np
import pytest
from mindspore import Tensor
from mindspore import ops
from tbe import tik

from flash_attention_test_agent import FlashAttentionTestAgent


class MatmulComputerOrder(FlashAttentionTestAgent):
    def define_ios(self, *args, **kwargs):
        M, K, N, dtype = (
            kwargs.get("M"),
            kwargs.get("K"),
            kwargs.get("N"),
            kwargs.get("dtype"),
        )
        self.dtype = dtype
        self.M = M
        self.K = K
        self.N = N
        self.A_gm = self.tik_instance.Tensor(dtype, (M, K), name="A_gm", scope=tik.scope_gm)
        self.B_gm = self.tik_instance.Tensor(dtype, (K, N), name="B_gm", scope=tik.scope_gm)
        self.C_gm = self.tik_instance.Tensor(dtype, (M, N), name="C_gm", scope=tik.scope_gm)
        self.C2_gm = self.tik_instance.Tensor(dtype, (M, N), name="C2_gm", scope=tik.scope_gm)
        self.scale_gm = self.tik_instance.Tensor(dtype, (M,), name="scale_gm", scope=tik.scope_gm)

    def compute(self, *args, **kwargs):
        M_alig = (self.M + 16 - 1) // 16 * 16
        K_alig = (self.K + 16 - 1) // 16 * 16
        N_alig = (self.N + 16 - 1) // 16 * 16
        # scale * (Pij * Vj)
        A_l1 = self.tik_instance.Tensor(
            self.dtype, (M_alig, K_alig), name="A_l1", scope=tik.scope_cbuf
        )
        B_l1 = self.tik_instance.Tensor(
            self.dtype, (K_alig, N_alig), name="A_l1", scope=tik.scope_cbuf
        )
        self.fa.tik_instance.data_move(A_l1, self.A_gm[0], 0, 1, self.M * self.K // 16, 0, 0)
        self.fa.tik_instance.data_move(B_l1, self.B_gm[0], 0, 1, self.K * self.N // 16, 0, 0)
        scale = self.tik_instance.Tensor(
            self.dtype, (M_alig,), name="scale_ub", scope=tik.scope_ubuf
        )
        self.tik_instance.h_data_move(scale, self.scale_gm)
        A_l1 = self.fa.tik_ops_utils.MK_TO_K1MK0(A_l1)
        B_l1 = self.fa.tik_ops_utils.KN_TO_K1NK0(B_l1)
        C_ub = self.fa.tik_ops_utils.matmul_compute(A_l1, B_l1, self.M, self.K, self.N)
        with self.tik_instance.for_range(begint=0, endt=self.M) as i:
            src_scalar = self.tik_instance.Scalar(init_value=scale[i], dtype=self.dtype)
            self.tik_instance.h_mul(C_ub[i, 0 : self.N], src_scalar, C_ub[i, 0 : self.N])
        self.fa.tik_instance.data_move(self.C_gm[0], C_ub, 0, 1, self.M * self.N // 16, 0, 0)

        # (scale * Pij) * Vj
        A2_l1 = self.tik_instance.Tensor(
            self.dtype, (M_alig, K_alig), name="A2_l1", scope=tik.scope_cbuf
        )
        B2_l1 = self.tik_instance.Tensor(
            self.dtype, (K_alig, N_alig), name="B2_l1", scope=tik.scope_cbuf
        )
        self.fa.tik_instance.data_move(A2_l1, self.A_gm[0], 0, 1, self.M * self.K // 16, 0, 0)
        self.fa.tik_instance.data_move(B2_l1, self.B_gm[0], 0, 1, self.K * self.N // 16, 0, 0)

        A2_l1_ub = self.tik_instance.Tensor(
            self.dtype, (M_alig, K_alig), name="A2_ub", scope=tik.scope_ubuf
        )
        self.tik_instance.data_move(A2_l1_ub, A2_l1, 0, 1, M_alig * K_alig // 16, 0, 0)
        with self.tik_instance.for_range(begint=0, endt=self.M) as i:
            src_scalar = self.tik_instance.Scalar(init_value=scale[i], dtype=self.dtype)
            self.tik_instance.h_mul(A2_l1_ub[i, 0 : self.K], src_scalar, A2_l1_ub[i, 0 : self.K])
        self.tik_instance.data_move(A2_l1, A2_l1_ub, 0, 1, M_alig * K_alig // 16, 0, 0)
        A2_l1 = self.fa.tik_ops_utils.MK_TO_K1MK0(A2_l1)
        B2_l1 = self.fa.tik_ops_utils.KN_TO_K1NK0(B2_l1)
        C2_ub = self.fa.tik_ops_utils.matmul_compute(A2_l1, B2_l1, self.M, self.K, self.N)
        self.fa.tik_instance.data_move(self.C2_gm[0], C2_ub, 0, 1, self.M * self.N // 16, 0, 0)

        self.tik_instance.BuildCCE(
            kernel_name="mock_flash_attention",
            inputs=[self.A_gm, self.B_gm, self.scale_gm],
            outputs=[self.C_gm, self.C2_gm],
        )
        return self.tik_instance


class MatmulComputerOrderV2(FlashAttentionTestAgent):
    def define_ios(self, *args, **kwargs):
        M, K, N, dtype = (
            kwargs.get("M"),
            kwargs.get("K"),
            kwargs.get("N"),
            kwargs.get("dtype"),
        )
        self.dtype = dtype
        self.M = M
        self.K = K
        self.N = N
        self.A_gm = self.tik_instance.Tensor(dtype, (M, K), name="A_gm", scope=tik.scope_gm)
        self.B_gm = self.tik_instance.Tensor(dtype, (K, N), name="B_gm", scope=tik.scope_gm)
        self.C_gm = self.tik_instance.Tensor(dtype, (M, N), name="C_gm", scope=tik.scope_gm)
        self.C2_gm = self.tik_instance.Tensor(dtype, (M, N), name="C2_gm", scope=tik.scope_gm)
        self.scale_gm = self.tik_instance.Tensor(
            "float32", (M,), name="scale_gm", scope=tik.scope_gm
        )

    def compute(self, *args, **kwargs):
        FP32 = "float32"
        M_alig = (self.M + 16 - 1) // 16 * 16
        K_alig = (self.K + 16 - 1) // 16 * 16
        N_alig = (self.N + 16 - 1) // 16 * 16
        # scale * (Pij * Vj)
        A_l1 = self.tik_instance.Tensor(
            self.dtype, (M_alig, K_alig), name="A_l1", scope=tik.scope_cbuf
        )
        B_l1 = self.tik_instance.Tensor(
            self.dtype, (K_alig, N_alig), name="A_l1", scope=tik.scope_cbuf
        )
        self.fa.tik_instance.data_move(A_l1, self.A_gm[0], 0, 1, self.M * self.K // 16, 0, 0)
        self.fa.tik_instance.data_move(B_l1, self.B_gm[0], 0, 1, self.K * self.N // 16, 0, 0)
        scale = self.tik_instance.Tensor(FP32, (M_alig,), name="scale_ub", scope=tik.scope_ubuf)
        self.tik_instance.h_data_move(scale, self.scale_gm)
        A_l1 = self.fa.tik_ops_utils.MK_TO_K1MK0(A_l1)
        B_l1 = self.fa.tik_ops_utils.KN_TO_K1NK0(B_l1)
        C_ub = self.fa.tik_ops_utils.matmul_compute(A_l1, B_l1, self.M, self.K, self.N)

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            tmp_ub_fp32 = self.tik_instance.Tensor(
                FP32, (M_alig, N_alig), name="tmp_ub_fp32", scope=tik.scope_ubuf
            )
            self.tik_instance.h_cast(tmp_ub_fp32, C_ub, "none")
            with self.tik_instance.for_range(begint=0, endt=self.M) as i:
                src_scalar = self.tik_instance.Scalar(init_value=scale[i], dtype=FP32)
                self.tik_instance.h_mul(
                    tmp_ub_fp32[i, 0 : self.N], tmp_ub_fp32[i, 0 : self.N], src_scalar,
                )
            self.tik_instance.h_cast(
                C_ub[0 : self.M, 0 : self.N], tmp_ub_fp32[0 : self.M, 0 : self.N], "none"
            )
        self.fa.tik_instance.data_move(self.C_gm[0], C_ub, 0, 1, self.M * self.N // 16, 0, 0)

        # (scale * Pij) * Vj
        A2_l1 = self.tik_instance.Tensor(
            self.dtype, (M_alig, K_alig), name="A2_l1", scope=tik.scope_cbuf
        )
        B2_l1 = self.tik_instance.Tensor(
            self.dtype, (K_alig, N_alig), name="B2_l1", scope=tik.scope_cbuf
        )
        self.fa.tik_instance.data_move(A2_l1, self.A_gm[0], 0, 1, self.M * self.K // 16, 0, 0)
        self.fa.tik_instance.data_move(B2_l1, self.B_gm[0], 0, 1, self.K * self.N // 16, 0, 0)

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            A2_l1_ub = self.tik_instance.Tensor(
                self.dtype, (M_alig, K_alig), name="A2_ub", scope=tik.scope_ubuf
            )
            self.tik_instance.data_move(A2_l1_ub, A2_l1, 0, 1, M_alig * K_alig // 16, 0, 0)

            tmp_ub_fp32 = self.tik_instance.Tensor(
                FP32, (M_alig, K_alig), name="tmp_ub_fp32", scope=tik.scope_ubuf
            )
            self.tik_instance.h_cast(tmp_ub_fp32, A2_l1_ub, "none")
            with self.tik_instance.for_range(begint=0, endt=self.M) as i:
                src_scalar = self.tik_instance.Scalar(init_value=scale[i], dtype=FP32)
                self.tik_instance.h_mul(
                    tmp_ub_fp32[i, 0 : self.K], tmp_ub_fp32[i, 0 : self.K], src_scalar,
                )
            self.tik_instance.h_cast(
                A2_l1_ub[0 : self.M, 0 : self.K], tmp_ub_fp32[0 : self.M, 0 : self.K], "none"
            )
            self.tik_instance.data_move(A2_l1, A2_l1_ub, 0, 1, M_alig * K_alig // 16, 0, 0)
        A2_l1 = self.fa.tik_ops_utils.MK_TO_K1MK0(A2_l1)
        B2_l1 = self.fa.tik_ops_utils.KN_TO_K1NK0(B2_l1)
        C2_ub = self.fa.tik_ops_utils.matmul_compute(A2_l1, B2_l1, self.M, self.K, self.N)
        self.fa.tik_instance.data_move(self.C2_gm[0], C2_ub, 0, 1, self.M * self.N // 16, 0, 0)

        self.tik_instance.BuildCCE(
            kernel_name="mock_flash_attention",
            inputs=[self.A_gm, self.B_gm, self.scale_gm],
            outputs=[self.C_gm, self.C2_gm],
        )
        return self.tik_instance


@pytest.mark.skip(reason="The matrix computing order matters in alg of FlashAttention")
class TestMatmulComputeOrder:
    @pytest.mark.parametrize(
        "M, K, N, dtype",
        [
            (64, 40, 64, "float16"),
            (128, 40, 64, "float16"),
            (64, 40, 128, "float16"),
            (64, 64, 64, "float16"),
            (128, 64, 64, "float16"),
            (64, 64, 128, "float16"),
        ],
    )
    def test_matmul_compute(self, M, K, N, dtype):
        mfa = MatmulComputerOrder(**dict(M=M, K=K, N=N, dtype=dtype))
        # 对比: (scale * Pij) * Vj 与 scale * (Pij * Vj) 的结果是否一致
        tik_instance = mfa.compute()

        A = np.random.random([M, K]).astype(dtype)
        B = np.random.random([K, N]).astype(dtype)
        scale = np.random.random([M]).astype(dtype)
        feed_dict = {"A_gm": A, "B_gm": B, "scale_gm": scale}
        (tik_out1, tik_out2) = tik_instance.tikdb.start_debug(
            feed_dict=feed_dict, interactive=False
        )

        ms_out = ops.matmul(Tensor(A), Tensor(B))
        ms_out = ops.cast(ms_out, mindspore.float16)
        for i in range(ms_out.shape[0]):
            ms_out[i, :] = ms_out[i, :] * scale[i]
        ms_out = ms_out.asnumpy()

        for i in range(M):
            A[i, :] = A[i, :] * scale[i]
        ms_out2 = ops.matmul(Tensor(A), Tensor(B))
        ms_out2 = ops.cast(ms_out2, mindspore.float16)
        ms_out2 = ms_out2.asnumpy()
        print("\n############## order1 result #################")
        print(tik_out1)
        print("\n############## order2 result #################")
        print(tik_out2)
        rtol = 1e-3
        assert np.allclose(tik_out1, tik_out2, rtol)
        assert np.allclose(ms_out, tik_out1, rtol)
        assert np.allclose(ms_out2, tik_out2, rtol)

    @pytest.mark.parametrize(
        "M, K, N, dtype",
        [
            (64, 40, 64, "float16"),
            (128, 40, 64, "float16"),
            (64, 40, 128, "float16"),
            (64, 64, 64, "float16"),
            (128, 64, 64, "float16"),
            (64, 64, 128, "float16"),
        ],
    )
    def test_matmul_compute_v2(self, M, K, N, dtype):
        mfa = MatmulComputerOrderV2(**dict(M=M, K=K, N=N, dtype=dtype))
        # 对比: (scale * Pij) * Vj 与 scale * (Pij * Vj) 的结果是否一致
        tik_instance = mfa.compute()

        A = np.random.random([M, K]).astype(dtype)
        B = np.random.random([K, N]).astype(dtype)
        scale = np.random.random([M]).astype("float32")
        feed_dict = {"A_gm": A, "B_gm": B, "scale_gm": scale}
        (tik_out1, tik_out2) = tik_instance.tikdb.start_debug(
            feed_dict=feed_dict, interactive=False
        )

        ms_out = ops.matmul(Tensor(A), Tensor(B))
        ms_out = ops.cast(ms_out, mindspore.float32)
        for i in range(ms_out.shape[0]):
            ms_out[i, :] = ms_out[i, :] * scale[i]
        ms_out = ms_out.asnumpy()

        A = A.astype("float32")
        for i in range(M):
            A[i, :] = A[i, :] * scale[i]
        A = A.astype("float16")
        ms_out2 = ops.matmul(Tensor(A), Tensor(B))
        ms_out2 = ms_out2.asnumpy()
        print("\n############## order1 result #################")
        print(tik_out1)
        print("\n############## order2 result #################")
        print(tik_out2)
        rtol = 1e-3
        assert np.allclose(tik_out1, tik_out2, rtol)
        assert np.allclose(ms_out, tik_out1, rtol)
        assert np.allclose(ms_out2, tik_out2, rtol)
