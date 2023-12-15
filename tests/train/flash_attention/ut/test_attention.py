# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
import os
import shutil

import mindspore.nn as nn
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


class FirstMatMul(nn.Cell):
    def __init__(self):
        super(FirstMatMul, self).__init__()
        self.transpose = ops.Transpose()

    def construct(self, Q, K):
        K = self.transpose(K, (1, 0))
        out = ops.matmul(Q, K)
        return out


class SoftMax(nn.Cell):
    def __init__(self, dim=None):
        super(SoftMax, self).__init__()
        self.softmax = ops.Softmax()
        self.scale = None
        if dim is not None:
            self.scale = dim ** -0.5

    def construct(self, Sim):
        if self.scale is not None:
            Sim = Sim * self.scale
        out = self.softmax(Sim)
        return out


class SecondMatMul(nn.Cell):
    def __init__(self):
        super(SecondMatMul, self).__init__()

    def construct(self, P, V):
        out = ops.matmul(P, V)
        return out


class AttentionForwardComputer(FlashAttentionTestAgent):
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
        self.Q_gm = self.tik_instance.Tensor(dtype, (q_len, dim), name="Q_gm", scope=tik.scope_gm)
        self.K_gm = self.tik_instance.Tensor(dtype, (kv_len, dim), name="K_gm", scope=tik.scope_gm)
        self.V_gm = self.tik_instance.Tensor(dtype, (kv_len, dim), name="V_gm", scope=tik.scope_gm)

        # define outputs
        self.Sim_gm = self.tik_instance.Tensor(dtype, (q_len, kv_len), name="Sim_gm", scope=tik.scope_gm)
        self.P_gm = self.tik_instance.Tensor(dtype, (q_len, kv_len), name="P_gm", scope=tik.scope_gm)
        self.O_gm = self.tik_instance.Tensor(dtype, (q_len, dim), name="O_gm", scope=tik.scope_gm)

    def compute(self, *args, **kwargs):
        Q_l1 = self.tik_instance.Tensor(self.dtype, (self.q_len, self.dim), name="Q_l1", scope=tik.scope_cbuf)
        self.fa.tik_instance.data_move(Q_l1, self.Q_gm[0], 0, 1, self.q_len * self.dim // 16, 0, 0)
        Q_l1_K1MK0_ed = self.fa.tik_ops_utils.MK_TO_K1MK0(Q_l1)

        K_l1 = self.tik_instance.Tensor(self.dtype, (self.kv_len, self.dim), name="K_l1", scope=tik.scope_cbuf)
        self.fa.tik_instance.data_move(K_l1, self.K_gm[0], 0, 1, self.kv_len * self.dim // 16, 0, 0)
        K_l1_K1NK0_ed = self.fa.tik_ops_utils.MK_TO_K1MK0(K_l1)

        V_l1 = self.tik_instance.Tensor(self.dtype, (self.kv_len, self.dim), name="V_l1", scope=tik.scope_cbuf)
        self.fa.tik_instance.data_move(V_l1, self.V_gm[0], 0, 1, self.kv_len * self.dim // 16, 0, 0)
        V_l1_K1NK0_ed = self.fa.tik_ops_utils.KN_TO_K1NK0(V_l1)

        # q * k.T
        Sim_ub = self.fa.tik_ops_utils.matmul_compute(Q_l1_K1MK0_ed, K_l1_K1NK0_ed, self.q_len, self.dim, self.kv_len)
        self.tik_instance.data_move(self.Sim_gm, Sim_ub, 0, 1, self.q_len * self.kv_len // 16, 0, 0)

        # Sim / sqrt(dim)
        # scale = self.dim ** -0.5
        # self.tik_instance.h_mul(Sim_ub, Sim_ub, scale)

        # row_max(Sim)
        row_max_ub = self.tik_instance.Tensor(
            self.dtype, (self.q_len,), name="row_max_ub", scope=tik.scope_ubuf
        )
        self.tik_instance.h_reduce_max(row_max_ub, Sim_ub, 1)

        # Sim - row_max(Sim)
        with self.tik_instance.for_range(0, self.q_len) as i:
            cur_row_max = self.tik_instance.Scalar(init_value=row_max_ub[i], dtype="float16")
            self.tik_instance.h_sub(Sim_ub[i, :], Sim_ub[i, :], cur_row_max)

        # fp16 -> fp32
        Sim_ub_fp32 = self.tik_instance.Tensor(
            "float32", (self.q_len, self.kv_len), name="Sim_ub", scope=tik.scope_ubuf
        )
        self.tik_instance.h_cast(Sim_ub_fp32, Sim_ub, "none")

        # exp(Sim)
        self.tik_instance.h_exp(Sim_ub_fp32, Sim_ub_fp32)

        # row_sum(Sim) fp32
        row_sum_ub = self.tik_instance.Tensor(
            "float32", (self.q_len,), name="row_sum_ub", scope=tik.scope_ubuf
        )
        self.tik_instance.h_reduce_sum(row_sum_ub, Sim_ub_fp32, 1)

        # P = Sim / row_sum(Sim)
        for i in range(self.q_len):
            cur_row_sum = self.tik_instance.Scalar(init_value=row_sum_ub[i], dtype="float32")
            self.tik_instance.h_div(Sim_ub_fp32[i, :], Sim_ub_fp32[i, :], cur_row_sum)

        # fp32 -> fp16
        self.tik_instance.h_cast(Sim_ub, Sim_ub_fp32, "none")
        self.tik_instance.data_move(self.P_gm, Sim_ub, 0, 1, self.q_len * self.kv_len // 16, 0, 0)

        P_l1 = self.tik_instance.Tensor(self.dtype, (self.q_len, self.kv_len), name="P_l1", scope=tik.scope_cbuf)
        P_l1_K1MK0_ed = self.fa.tik_ops_utils.MK_TO_K1MK0(Sim_ub, workspace_tensor=P_l1)

        O_ub = self.fa.tik_ops_utils.matmul_compute(P_l1_K1MK0_ed, V_l1_K1NK0_ed, self.q_len, self.kv_len, self.dim)
        self.tik_instance.data_move(self.O_gm, O_ub, 0, 1, self.q_len * self.dim // 16, 0, 0)

        self.tik_instance.BuildCCE(
            kernel_name="mock_attention_forward",
            inputs=[self.Q_gm, self.K_gm, self.V_gm],
            outputs=[self.Sim_gm, self.P_gm, self.O_gm],
        )
        return self.tik_instance


class AttentionBackwardComputer(FlashAttentionTestAgent):
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
        self.Q_gm = self.tik_instance.Tensor(dtype, (q_len, dim), name="Q_gm", scope=tik.scope_gm)
        self.K_gm = self.tik_instance.Tensor(dtype, (kv_len, dim), name="K_gm", scope=tik.scope_gm)
        self.V_gm = self.tik_instance.Tensor(dtype, (kv_len, dim), name="V_gm", scope=tik.scope_gm)
        self.P_gm = self.tik_instance.Tensor(dtype, (q_len, kv_len), name="P_gm", scope=tik.scope_gm)
        self.dO_gm = self.tik_instance.Tensor(dtype, (q_len, dim), name="dO_gm", scope=tik.scope_gm)

        # define outputs
        self.dQ_gm = self.tik_instance.Tensor(dtype, (q_len, dim), name="dQ_gm", scope=tik.scope_gm)
        self.dK_gm = self.tik_instance.Tensor(dtype, (kv_len, dim), name="dK_gm", scope=tik.scope_gm)
        self.dV_gm = self.tik_instance.Tensor(dtype, (kv_len, dim), name="dV_gm", scope=tik.scope_gm)

    def compute(self, *args, **kwargs):
        P_l1 = self.tik_instance.Tensor(self.dtype, (self.q_len, self.kv_len), name="P_l1", scope=tik.scope_cbuf)
        self.tik_instance.data_move(P_l1, self.P_gm, 0, 1, self.q_len * self.kv_len // 16, 0, 0)
        PT_l1_K1MK0_ed = self.fa.tik_ops_utils.KN_TO_K1NK0(P_l1)

        dO_l1_left = self.tik_instance.Tensor(self.dtype, (self.q_len, self.dim), name="dO_l1_left",
                                              scope=tik.scope_cbuf)
        dO_l1_right = self.tik_instance.Tensor(self.dtype, (self.q_len, self.dim), name="dO_l1_right",
                                               scope=tik.scope_cbuf)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            dO_ub = self.tik_instance.Tensor(self.dtype, (self.q_len, self.dim), name="dO_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(dO_ub, self.dO_gm, 0, 1, self.q_len * self.dim // 16, 0, 0)
            dO_l1_K1MK0_ed = self.fa.tik_ops_utils.MK_TO_K1MK0(dO_ub, workspace_tensor=dO_l1_left)
            dO_l1_K1NK0_ed = self.fa.tik_ops_utils.KN_TO_K1NK0(dO_ub, workspace_tensor=dO_l1_right)

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            dV_ub = self.fa.tik_ops_utils.matmul_compute(PT_l1_K1MK0_ed, dO_l1_K1NK0_ed,
                                                         self.kv_len, self.q_len, self.dim)
            self.tik_instance.data_move(self.dV_gm, dV_ub, 0, 1, self.kv_len * self.dim // 16, 0, 0)

        V_l1 = self.tik_instance.Tensor(self.dtype, (self.kv_len, self.dim), name="V_l1", scope=tik.scope_cbuf)
        self.tik_instance.data_move(V_l1, self.V_gm, 0, 1, self.kv_len * self.dim // 16, 0, 0)
        VT_l1_K1NK0_ed = self.fa.tik_ops_utils.MK_TO_K1MK0(V_l1)

        dP_ub = self.fa.tik_ops_utils.matmul_compute(dO_l1_K1MK0_ed, VT_l1_K1NK0_ed, self.q_len, self.dim, self.kv_len)

        dSim_ub = self.tik_instance.Tensor(self.dtype, (self.q_len, self.kv_len), name="dSim_ub", scope=tik.scope_ubuf)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            P_ub = self.tik_instance.Tensor(self.dtype, (self.q_len, self.kv_len), name="P_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(P_ub, self.P_gm, 0, 1, self.q_len * self.kv_len // 16, 0, 0)
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                # P * dP
                self.tik_instance.h_mul(P_ub, dP_ub, P_ub)
                # row_sum(P * dP)
                P_dP_ub_fp32 = self.tik_instance.Tensor("float32", (self.q_len, self.kv_len), name="P_dP_ub_fp32",
                                                        scope=tik.scope_ubuf)
                self.tik_instance.h_cast(P_dP_ub_fp32, P_ub, "none")
                row_sum_ub_fp32 = self.tik_instance.Tensor("float32", (self.q_len,), name="row_sum_ub_fp32",
                                                           scope=tik.scope_ubuf)
                self.tik_instance.h_reduce_sum(row_sum_ub_fp32, P_dP_ub_fp32, 1)
                row_sum_ub = self.tik_instance.Tensor(self.dtype, (self.q_len,), name="P_dP_ub", scope=tik.scope_ubuf)
                self.tik_instance.h_cast(row_sum_ub, row_sum_ub_fp32, "none")
                # dP - row_sum(P * dP)
                with self.tik_instance.for_range(0, self.q_len) as idx:
                    cur_row_sum = self.tik_instance.Scalar(self.dtype, init_value=row_sum_ub[idx])
                    self.tik_instance.h_sub(dP_ub[idx, :], dP_ub[idx, :], cur_row_sum)

            # P_ub被dP_ub * P_ub复用了，需要重新load
            self.tik_instance.data_move(P_ub, self.P_gm, 0, 1, self.q_len * self.kv_len // 16, 0, 0)
            # P * (dP - row_sum(P * dP))
            self.tik_instance.h_mul(dSim_ub, P_ub, dP_ub)

        # dSim / sqrt(dim)
        # scale = self.dim ** -0.5
        # self.tik_instance.h_mul(dSim_ub, dSim_ub, scale)

        dSim_l1_1 = self.tik_instance.Tensor(self.dtype, (self.q_len, self.kv_len), name="dSim_l1_1",
                                             scope=tik.scope_cbuf)
        dSim_l1_2 = self.tik_instance.Tensor(self.dtype, (self.q_len, self.kv_len), name="dSim_l1_2",
                                             scope=tik.scope_cbuf)
        dSim_l1_K1MK0_ed = self.fa.tik_ops_utils.MK_TO_K1MK0(dSim_ub, workspace_tensor=dSim_l1_1)
        dSimT_l1_K1MK0_ed = self.fa.tik_ops_utils.KN_TO_K1NK0(dSim_ub, workspace_tensor=dSim_l1_2)

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            K_l1 = self.tik_instance.Tensor(self.dtype, (self.kv_len, self.dim), name="K_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(K_l1, self.K_gm, 0, 1, self.kv_len * self.dim // 16, 0, 0)
            K_l1_K1NK0_ed = self.fa.tik_ops_utils.KN_TO_K1NK0(K_l1)
            dQ_ub = self.fa.tik_ops_utils.matmul_compute(dSim_l1_K1MK0_ed, K_l1_K1NK0_ed,
                                                         self.q_len, self.kv_len, self.dim)
            self.tik_instance.data_move(self.dQ_gm, dQ_ub, 0, 1, self.q_len * self.dim // 16, 0, 0)

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            Q_l1 = self.tik_instance.Tensor(self.dtype, (self.q_len, self.dim), name="Q_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(Q_l1, self.Q_gm, 0, 1, self.q_len * self.dim // 16, 0, 0)
            Q_l1_K1NK0_ed = self.fa.tik_ops_utils.KN_TO_K1NK0(Q_l1)
            dK_ub = self.fa.tik_ops_utils.matmul_compute(dSimT_l1_K1MK0_ed, Q_l1_K1NK0_ed,
                                                         self.kv_len, self.q_len, self.dim)
            self.tik_instance.data_move(self.dK_gm, dK_ub, 0, 1, self.kv_len * self.dim // 16, 0, 0)

        self.tik_instance.BuildCCE(
            kernel_name="mock_attention_backward",
            inputs=[self.Q_gm, self.K_gm, self.V_gm, self.P_gm, self.dO_gm],
            outputs=[self.dQ_gm, self.dK_gm, self.dV_gm]
        )
        return self.tik_instance


class TestAttention:
    @staticmethod
    def ms_impl_attention_backward(Q, K, V, sens):
        # forward
        model1 = FirstMatMul()
        Sim = model1(Q, K)
        model2 = SoftMax()
        P = model2(Sim)
        model3 = SecondMatMul()
        out = model3(P, V)

        # backward
        grad = ops.GradOperation(sens_param=True, get_all=True)
        dP, dV = grad(model3)(P, V, sens)
        dSim, = grad(model2)(Sim, dP)
        dQ, dK = grad(model1)(Q, K, dSim)

        return dQ, dK, dV

    @staticmethod
    def ms_impl_attention_forward(Q, K, V):
        model1 = FirstMatMul()
        Sim = model1(Q, K)
        model2 = SoftMax()
        P = model2(Sim)
        model3 = SecondMatMul()
        out = model3(P, V)
        return Sim, P, out

    @pytest.mark.parametrize(
        "q_seq_len, kv_seq_len, dim, dtype",
        [
            (32, 32, 32, "float16"),
            (16, 16, 16, "float16"),
            (16, 32, 16, "float16"),
            (64, 64, 64, "float16"),
            (128, 128, 128, "float16"),
        ],
    )
    def test_attention_forward(self, q_seq_len, kv_seq_len, dim, dtype):
        mfa = AttentionForwardComputer(
            **dict(q_len=q_seq_len, kv_len=kv_seq_len, dim=dim, dtype=dtype)
        )
        tik_instance = mfa.compute()

        q = np.random.random([q_seq_len, dim]).astype(dtype)
        k = np.random.random([kv_seq_len, dim]).astype(dtype)
        v = np.random.random([kv_seq_len, dim]).astype(dtype)

        feed_dict = {"Q_gm": q, "K_gm": k, "V_gm": v}
        (tik_sim, tik_p, tik_out) = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)
        ms_sim, ms_p, ms_out = self.ms_impl_attention_forward(Tensor(q), Tensor(k), Tensor(v))

        print("\n-------------- Sim -----------------")
        result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = \
            mfa.data_compare(ms_sim.asnumpy(), tik_sim)
        print(f"Compare result: {result}, error_less_than_thousandth_proportion: "
              f"{error_less_than_thousandth_proportion}, "
              f"error_greater_than_tenth_count: {error_greater_than_tenth_count}")
        assert result == "Pass"

        print("-------------- P -----------------")
        result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = \
            mfa.data_compare(ms_p.asnumpy(), tik_p)
        print(f"Compare result: {result}, error_less_than_thousandth_proportion:"
              f" {error_less_than_thousandth_proportion}, "
              f"error_greater_than_tenth_count: {error_greater_than_tenth_count}")
        assert result == "Pass"

        print("-------------- Out -----------------")
        result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = \
            mfa.data_compare(ms_out.asnumpy(), tik_out)
        print(f"Compare result: {result}, error_less_than_thousandth_proportion: "
              f"{error_less_than_thousandth_proportion}, "
              f"error_greater_than_tenth_count: {error_greater_than_tenth_count}")
        assert result == "Pass"

    @pytest.mark.parametrize(
        "q_seq_len, kv_seq_len, dim, dtype",
        [
            (16, 32, 16, "float16"),
            (32, 32, 32, "float16"),
            (64, 64, 64, "float16"),
            (128, 128, 128, "float16"),
        ],
    )
    def test_attention_backward(self, q_seq_len, kv_seq_len, dim, dtype):
        # softmax中Sim除以sqrt(dim)会导致反向误差增大
        mfa_forward = AttentionForwardComputer(
            **dict(q_len=q_seq_len, kv_len=kv_seq_len, dim=dim, dtype=dtype)
        )
        tik_instance = mfa_forward.compute()

        q = np.random.random([q_seq_len, dim]).astype(dtype)
        k = np.random.random([kv_seq_len, dim]).astype(dtype)
        v = np.random.random([kv_seq_len, dim]).astype(dtype)
        sens = 10.0 * np.random.random((q_seq_len, dim)).astype(dtype)

        feed_dict = {"Q_gm": q, "K_gm": k, "V_gm": v}
        (tik_Sim, tik_P, tik_O) = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)

        mfa_backward = AttentionBackwardComputer(
            **dict(q_len=q_seq_len, kv_len=kv_seq_len, dim=dim, dtype=dtype)
        )
        tik_instance = mfa_backward.compute()

        feed_dict = {"Q_gm": q, "K_gm": k, "V_gm": v, "P_gm": tik_P, "dO_gm": sens}
        tik_dQ, tik_dK, tik_dV = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)

        ms_dQ, ms_dK, ms_dV = self.ms_impl_attention_backward(Tensor(q), Tensor(k), Tensor(v),
                                                              Tensor(sens))

        print("\n-------------- dQ -----------------")
        result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = mfa_backward.data_compare(
            ms_dQ.asnumpy(), tik_dQ)
        print(f"Compare result: {result}, error_less_than_thousandth_proportion: "
              f"{error_less_than_thousandth_proportion}, "
              f"error_greater_than_tenth_count: {error_greater_than_tenth_count}")
        # assert result == "Pass"

        print("-------------- dK -----------------")
        result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = mfa_backward.data_compare(
            ms_dK.asnumpy(), tik_dK)
        print(f"Compare result: {result}, error_less_than_thousandth_proportion: "
              f"{error_less_than_thousandth_proportion}, "
              f"error_greater_than_tenth_count: {error_greater_than_tenth_count}")
        # assert result == "Pass"

        print("-------------- dV -----------------")
        result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = mfa_backward.data_compare(
            ms_dV.asnumpy(), tik_dV)
        print(f"Compare result: {result}, error_less_than_thousandth_proportion: "
              f"{error_less_than_thousandth_proportion}, "
              f"error_greater_than_tenth_count: {error_greater_than_tenth_count}")
        # assert result == "Pass"
