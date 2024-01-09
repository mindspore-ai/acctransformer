# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import os
import shutil
import numpy as np

import mindspore.context as context
from mindspore import Tensor
from mindspore import nn
from mindspore import ops
from mindspore import dtype as mstype

from acctransformer.flash_attention.ops.flash_attention.flash_attention_impl import get_flash_attention, get_flash_attention_grad


def set_env():
    # 设置环境变量
    os.environ["GLOG_v"] = "3"
    os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
    os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"

    shutil.rmtree("./rank_0", ignore_errors=True)
    shutil.rmtree("./kernel_meta", ignore_errors=True)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0,
                        save_graphs=0, save_graphs_path='./graphs')


class FlashAttention(nn.Cell):
    def __init__(self, next_block_num=0):
        super(FlashAttention, self).__init__()
        self.flash_attention = get_flash_attention(next_block_num=next_block_num)

    def construct(self, q, k, v, attn_mask=None, drop_mask=None):
        alibi_mask = None # 算子预留接口，nz未适配，当前不测试
        o, l = self.flash_attention(q, k, v, attn_mask, drop_mask, alibi_mask)
        return o


class FlashAttentionGrad(nn.Cell):
    def __init__(self, next_block_num=0):
        super(FlashAttentionGrad, self).__init__()
        self.flash_attention_grad = get_flash_attention_grad(next_block_num=next_block_num)

    def construct(self, q, k, v, attn_mask, l, o, douts, drop_mask=None):
        alibi_mask = None # 算子预留接口，nz未适配，当前不测试
        grad = self.flash_attention_grad(q, k, v, attn_mask, drop_mask, alibi_mask, (o, l), douts)
        return grad


class DropGenMask(nn.Cell):
    """
    为测试生成一致的drop_mask
    算子为了性能未对drop_mask做nz处理，暂不开启dropout做精度对齐
    """
    def __init__(self, shape, dropout_rate):
        super(DropGenMask, self).__init__()
        self.shape = shape
        self.dropout_rate = dropout_rate
        self.keep_prob = Tensor(1 - self.dropout_rate, dtype=mstype.float16)
        self.drop_gen_mask = ops.DropoutGenMask()
        self.do_dropout = ops.DropoutDoMask()
        self.ones = Tensor(np.ones(self.shape).astype("float16"))

    def construct(self):
        drop_mask_bits = self.drop_gen_mask(self.shape, self.keep_prob)
        drop_mask = self.do_dropout(self.ones, drop_mask_bits, self.keep_prob)
        return drop_mask


def np_impl_sa_fwd(Q, K, V, attn_mask=None, drop_mask=None):
    # Q * K.T
    K_transposed = np.transpose(K, (0, 1, 3, 2))
    Sim = np.matmul(Q, K_transposed)
    # mask
    if attn_mask is not None:
        attn_mask = -10000.0 * attn_mask
        attn_mask = np.expand_dims(attn_mask, 1)
        Sim = Sim + attn_mask
    # softmax
    row_max = np.max(Sim, axis=-1, keepdims=True)
    Sim -= row_max
    Sim = Sim.astype("float32")
    Sim = np.exp(Sim)
    row_sum = np.sum(Sim, axis=-1, keepdims=True)
    P = Sim / row_sum
    P = P.astype("float16")
    if drop_mask is not None:
        P = P * drop_mask
    # P * V
    O = np.matmul(P, V)
    mid_results = (Sim, P, row_sum, row_max)
    return O, mid_results


def np_impl_sa_grad(Q, K, V, P, O, dO, drop_mask=None):
    # grad of P * V
    P_transposed = np.transpose(P, (0, 1, 3, 2))
    dV = np.matmul(P_transposed, dO)
    V_transposed = np.transpose(V, (0, 1, 3, 2))
    dP = np.matmul(dO, V_transposed)
    if drop_mask is not None:
        dP = dP * drop_mask

    # sa grad of softmax (fa计算公式与sa计算公式存在累计误差，暂时采用fa版本作为ground truth)
    # P_dP = np.multiply(P, dP)
    # P_dP = P_dP.astype("float32")
    # row_sum = np.sum(P_dP, axis=-1, keepdims=True)
    # row_sum = row_sum.astype("float16")
    # dP_sub_rowsum = dP - row_sum
    # dSim = np.multiply(P, dP_sub_rowsum)

    # fa grad of softmax
    dO_O = dO * O
    dO_O = dO_O.astype("float32")
    row_sum = np.sum(dO_O, axis=-1, keepdims=True).astype("float16")
    dP_sub_rowsum = dP - row_sum
    dSim = P * dP_sub_rowsum

    # grad of Q * K.T
    dQ = np.matmul(dSim, K)
    dSim_transposed = np.transpose(dSim, (0, 1, 3, 2))
    dK = np.matmul(dSim_transposed, Q)
    return dQ, dK, dV


def data_compare(ground_truth, predict, rtol=0.001, rtol_pct_thd=0.001, atol=0.001, max_rtol=0.1):
    total_count = np.prod(ground_truth.shape)
    greater_than_diff_thd_count = np.sum(
        np.abs(predict - ground_truth) > rtol * np.abs(ground_truth) + atol
    )
    greater_than_max_diff_thd_count = np.sum(
        np.abs(predict - ground_truth) > max_rtol * np.abs(ground_truth) + atol
    )

    diff_gt_thd_proportion = greater_than_diff_thd_count / total_count
    diff_gt_max_thd_proportion = greater_than_max_diff_thd_count / total_count
    result = "Pass"
    if diff_gt_thd_proportion > rtol_pct_thd or diff_gt_max_thd_proportion > 0:
        result = "Failed"
    return result, diff_gt_thd_proportion, diff_gt_max_thd_proportion


def display(ground_truth, predict, result, proportion1, proportion2, thd1=0.001, thd2=0.1):
    print(
        f"Compare result: {result}, relative diff greater than {thd1} proportion: {proportion1}, "
        f"relative diff greater than {thd2} proportion: {proportion2}"
    )
    diff_max = np.max(np.abs(predict - ground_truth))
    diff = np.abs(predict - ground_truth)
    index = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
    index = tuple(map(int, index))
    print(f"max diff {diff_max}, ground_truth: {ground_truth[index]}, predict: {predict[index]}")

