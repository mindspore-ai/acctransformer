# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
import os
import shutil

import mindspore.ops as ops
import numpy as np
import pytest
from mindspore import Tensor
from tbe import tik
from tbe.common.platform import set_current_compile_soc_info

from flash_attention_test_agent import FlashAttentionTestAgent


def set_up_module():
    if os.path.exists("./rank_0"):
        shutil.rmtree("./rank_0", ignore_errors=True)
        shutil.rmtree("./kernel_meta", ignore_errors=True)
    set_current_compile_soc_info("Ascend910")


class AttentionComputer(FlashAttentionTestAgent):
    def define_ios(self, *args, **kwargs):
        q_len = kwargs.get("q_len")
        kv_len = kwargs.get("kv_len")
        dim = kwargs.get("dim")
        dtype = kwargs.get("dtype")

        self.kv_len = kv_len
        self.q_len = q_len
        self.dtype = dtype
        self.dim = dim
        # define inputs
        self.q_gm = self.tik_instance.Tensor(dtype, (q_len, dim), name="Q_gm", scope=tik.scope_gm)
        self.k_gm = self.tik_instance.Tensor(dtype, (kv_len, dim), name="K_gm", scope=tik.scope_gm)
        self.v_gm = self.tik_instance.Tensor(dtype, (kv_len, dim), name="V_gm", scope=tik.scope_gm)
        # define outputs
        self.out_gm = self.tik_instance.Tensor(dtype, (q_len, dim), name="out_gm", scope=tik.scope_gm)

    def compute(self, *args, **kwargs):
        # load data and reorder
        q_l1 = self.tik_instance.Tensor(self.dtype, (self.q_len, self.dim), name="q_l1", scope=tik.scope_cbuf)
        self.fa.tik_instance.data_move(q_l1, self.q_gm, 0, 1, self.q_len * self.dim // 16, 0, 0)
        q_l1 = self.fa.tik_ops_utils.MK_TO_K1MK0(q_l1)

        k_l1 = self.tik_instance.Tensor(self.dtype, (self.kv_len, self.dim), name="k_l1", scope=tik.scope_cbuf)
        self.fa.tik_instance.data_move(k_l1, self.k_gm, 0, 1, self.kv_len * self.dim // 16, 0, 0)
        k_l1 = self.fa.tik_ops_utils.MK_TO_K1MK0(k_l1)

        v_l1 = self.tik_instance.Tensor(self.dtype, (self.kv_len, self.dim), name="v_l1", scope=tik.scope_cbuf)
        self.fa.tik_instance.data_move(v_l1, self.v_gm, 0, 1, self.kv_len * self.dim // 16, 0, 0)
        v_l1 = self.fa.tik_ops_utils.KN_TO_K1NK0(v_l1)

        # q * k.T
        sim_ub = self.fa.tik_ops_utils.matmul_compute(q_l1, k_l1, self.q_len, self.dim, self.kv_len, N1MN0_to_MN=False)
        # non-normalized softmax
        mij_ub = self.tik_instance.Tensor(self.dtype, (self.q_len,), scope=tik.scope_ubuf, name="mij_ub")
        lij_ub = self.tik_instance.Tensor(self.dtype, (self.q_len,), scope=tik.scope_ubuf, name="lij_ub")
        self.fa.Bc = self.kv_len
        self.fa.precision_type = self.dtype
        ones_ub = self.tik_instance.Tensor(self.dtype, (self.kv_len, 16), name="ones_ub", scope=tik.scope_ubuf)
        self.tik_instance.h_duplicate(ones_ub, 1.0)
        ones_l1 = self.tik_instance.Tensor(self.dtype, (self.kv_len, 16), name="ones_l1", scope=tik.scope_cbuf)
        self.tik_instance.data_move(ones_l1, ones_ub, 0, 1, self.kv_len, 0, 0)
        self.fa.ones_l1 = ones_l1
        p_ub, m_ub, l_ub = self.fa.softmax_compute(sim_ub, mij_ub, lij_ub, self.q_len, self.kv_len)

        # p * v
        p_l1 = self.tik_instance.Tensor(self.dtype, (self.kv_len // 16, self.q_len // 16, 16, 16), name="p_l1",
                                        scope=tik.scope_cbuf)
        self.tik_instance.data_move(p_l1, p_ub, 0, 1, self.q_len * self.kv_len // 16, 0, 0)
        non_normalized_out_ub = self.fa.tik_ops_utils.matmul_compute(p_l1, v_l1, self.q_len, self.kv_len, self.dim)

        # normalize out
        l_rec_ub = self.fa.tik_ops_utils.calc_vec_rec(l_ub, self.q_len)
        with self.tik_instance.for_range(0, self.q_len) as idx:
            cur_l_rec = self.tik_instance.Scalar(init_value=l_rec_ub[idx], dtype=self.dtype)
            self.tik_instance.h_mul(
                non_normalized_out_ub[idx, :],
                non_normalized_out_ub[idx, :],
                cur_l_rec,
            )
            normalized_out_ub = non_normalized_out_ub
        self.fa.tik_instance.data_move(self.out_gm, normalized_out_ub, 0, 1, self.q_len * self.dim // 16, 0, 0)

        self.tik_instance.BuildCCE(
            kernel_name="mock_flash_attention",
            inputs=[self.q_gm, self.k_gm, self.v_gm],
            outputs=[self.out_gm],
        )
        return self.tik_instance


class TestAttention:
    @pytest.mark.parametrize(
        "q_seq_len, kv_seq_len, dim, dtype",
        [
            (16, 16, 16, "float16"),
            (16, 32, 16, "float16"),
            (64, 64, 32, "float16"),
            (64, 48, 48, "float16"),
            (128, 48, 64, "float16"),
            (256, 96, 32, "float16"),
            (128, 128, 48, "float16"),
            (128, 256, 64, "float16"),
        ],
    )
    def test_attention(self, q_seq_len, kv_seq_len, dim, dtype):
        mfa = AttentionComputer(
            **dict(q_len=q_seq_len, kv_len=kv_seq_len, dim=dim, dtype=dtype)
        )
        tik_instance = mfa.compute()

        q = np.random.random([q_seq_len, dim]).astype(dtype)
        k = np.random.random([kv_seq_len, dim]).astype(dtype)
        v = np.random.random([kv_seq_len, dim]).astype(dtype)

        feed_dict = {"Q_gm": q, "K_gm": k, "V_gm": v}
        tik_out, = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)
        ms_out = self.ms_impl_attention(Tensor(q), Tensor(k), Tensor(v))

        result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = mfa.data_compare(
            ms_out.asnumpy(), tik_out)
        print(
            f"Compare result: {result}, Out error_less_than_thousandth_proportion: "
            f"{error_less_than_thousandth_proportion}, "
            f"dQ error_greater_than_tenth_count: {error_greater_than_tenth_count}")
        rtol = 1e-2
        assert np.allclose(tik_out, ms_out.asnumpy(), rtol)

    def init_ms_attention(self):
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()

    def ms_impl_attention(self, q, k, v):
        self.init_ms_attention()
        k = self.transpose(k, (1, 0))
        sim = ops.matmul(q, k)
        attn = self.softmax(sim)
        out = ops.matmul(attn, v)
        return out
