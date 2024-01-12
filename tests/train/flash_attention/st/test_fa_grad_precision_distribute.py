import os
import pytest

import mindspore as ms
import numpy as np

from mindspore import Tensor, ops, context, nn
from mindspore.communication import init

from acctransformer.triangle_attention.triangle_attention import TriangleAttention
from common import FlashAttention, data_compare, display

os.environ["GLOG_v"] = "3"
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"

context.set_context(mode=ms.GRAPH_MODE, save_graphs=0, save_graphs_path='./graphs')
context.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
                                  dataset_strategy="full_batch")
init()
ms.set_seed(1)


class AttentionGradWrapper(nn.Cell):
    def __init__(self, net):
        super(AttentionGradWrapper, self).__init__()
        self.attention = net
        self.grad = ops.GradOperation(get_all=True)

    def construct(self, q, k, v, attention_mask):
        grad_func = self.grad(self.attention)
        q_grad, k_grad, v_grad, _ = grad_func(q, k, v, attention_mask)
        return q_grad, k_grad, v_grad


@pytest.mark.parametrize(
    "q_npy_path, k_npy_path, v_npy_path, dp, mp",
    [
        ("savepath/q.npy", "savepath/k.npy", "savepath/v.npy", 1, 8),
    ],
)
def test_fa_grad_compare_with_triangle_attn_grad(q_npy_path, k_npy_path, v_npy_path, dp, mp):
    q = np.load(q_npy_path)
    k = np.load(k_npy_path)
    v = np.load(v_npy_path)

    q_shape = q.shape
    kv_shape = k.shape

    batch_size, q_seq_len, k_seq_len = q_shape[0], q_shape[2], kv_shape[2]

    fa_attn_mask = np.triu(np.ones(shape=(1, 128, 128), dtype=np.float16), k=1)
    attn_mask = np.triu(np.ones(shape=(batch_size, q_seq_len, k_seq_len), dtype=np.float16), k=1)

    model1 = FlashAttention()
    q_shard_stgy = (dp, mp, 1, 1)
    k_shard_stgy = (dp, mp, 1, 1)
    v_shard_stgy = (dp, mp, 1, 1)
    attn_mask_shard_stgy = (dp, 1, 1)
    in_stgy = (q_shard_stgy, k_shard_stgy, v_shard_stgy, attn_mask_shard_stgy)
    model1.shard(in_stgy)
    model1.flash_attention.recompute()
    grad_model1 = AttentionGradWrapper(model1)

    model2 = TriangleAttention(block_size=8192, dp=dp, mp=mp)
    grad_model2 = AttentionGradWrapper(model2)
    grad_model2.attention.recompute()

    cus_grad = grad_model1(Tensor(q), Tensor(k), Tensor(v), Tensor(fa_attn_mask))
    cus_dQ, cus_dK, cus_dV = cus_grad[0].asnumpy(), cus_grad[1].asnumpy(), cus_grad[2].asnumpy()
    ta_grad = grad_model2(Tensor(q), Tensor(k), Tensor(v), Tensor(attn_mask))
    triangle_dQ, triangle_dK, triangle_dV = ta_grad[0].asnumpy(), ta_grad[1].asnumpy(), ta_grad[2].asnumpy()

    rtol, rtol_pct_thd, atol = 0.005, 0.005, 0.005
    print(f"\n--------- shape: {q_shape}-{kv_shape} -------------")
    print("--------- dQ -------------")
    result, q_diff_gt_rtol_proportion, q_diff_gt_max_rtol_proprotion = data_compare(triangle_dQ, cus_dQ,
                                                                                    rtol, rtol_pct_thd, atol)
    display(triangle_dQ, cus_dQ, result, q_diff_gt_rtol_proportion, q_diff_gt_max_rtol_proprotion, thd1=rtol)
    print("--------- dK -------------")
    result, k_diff_gt_rtol_proportion, k_diff_gt_max_rtol_proprotion = data_compare(triangle_dK, cus_dK,
                                                                                    rtol, rtol_pct_thd, atol)
    display(triangle_dK, cus_dK, result, k_diff_gt_rtol_proportion, k_diff_gt_max_rtol_proprotion, thd1=rtol)
    print("--------- dV -------------")
    result, v_diff_gt_rtol_proportion, v_diff_gt_max_rtol_proprotion = data_compare(triangle_dV, cus_dV, rtol,
                                                                                    rtol_pct_thd, atol)
    display(triangle_dV, cus_dV, result, v_diff_gt_rtol_proportion, v_diff_gt_max_rtol_proprotion, thd1=rtol)

    assert q_diff_gt_max_rtol_proprotion < 1e-4 \
           and k_diff_gt_max_rtol_proprotion < 1e-4 \
           and v_diff_gt_max_rtol_proprotion < 1e-4
