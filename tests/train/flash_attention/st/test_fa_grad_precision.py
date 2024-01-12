# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import pytest
import numpy as np

from mindspore import Tensor, ops

from acctransformer.triangle_attention.triangle_attention import TriangleAttention
from common import data_compare, display, set_env, FlashAttention
from common import FlashAttentionGrad, np_impl_sa_fwd, np_impl_sa_grad

set_env()


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 1, 128, 80), (1, 1, 128, 80)),
        ((4, 8, 128, 80), (4, 8, 128, 80)),
        ((4, 8, 1024, 128), (4, 8, 1024, 128)),
        ((1, 1, 4096, 128), (1, 1, 4096, 128))
    ],
)
def test_fa_grad_with_triangle_attn_mask(q_shape, kv_shape):
    q = np.random.randn(*q_shape).astype("float16")
    k = np.random.randn(*kv_shape).astype("float16")
    v = np.random.randn(*kv_shape).astype("float16")
    batch_size, q_seq_len, k_seq_len = q_shape[0], q_shape[2], kv_shape[2]

    att_mask = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)

    O, mid_results = np_impl_sa_fwd(q, k, v, att_mask)
    _, P, row_sum, row_max = mid_results
    dO = np.random.randn(*q_shape).astype("float16")
    douts = (Tensor(dO), None)
    l = row_max.astype("float32") + np.log(row_sum)
    l = np.squeeze(l, axis=-1)

    model = FlashAttentionGrad()
    cus_grad = model(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask), Tensor(l), Tensor(O), douts)
    cus_dQ, cus_dK, cus_dV = cus_grad[0].asnumpy(), cus_grad[1].asnumpy(), cus_grad[2].asnumpy()
    np_dQ, np_dK, np_dV = np_impl_sa_grad(q, k, v, P, O, dO)
    rtol, rtol_pct_thd, atol = 0.005, 0.005, 0.005
    print(f"\n--------- shape: {q_shape}-{kv_shape} -------------")
    print("--------- dQ -------------")
    result, q_diff_gt_rtol_proportion, q_diff_gt_max_rtol_proprotion = data_compare(np_dQ, cus_dQ,
                                                                                    rtol, rtol_pct_thd, atol)
    display(np_dQ, cus_dQ, result, q_diff_gt_rtol_proportion, q_diff_gt_max_rtol_proprotion, thd1=rtol)
    print("--------- dK -------------")
    result, k_diff_gt_rtol_proportion, k_diff_gt_max_rtol_proprotion = data_compare(np_dK, cus_dK,
                                                                                    rtol, rtol_pct_thd, atol)
    display(np_dK, cus_dK, result, k_diff_gt_rtol_proportion, k_diff_gt_max_rtol_proprotion, thd1=rtol)
    print("--------- dV -------------")
    result, v_diff_gt_rtol_proportion, v_diff_gt_max_rtol_proprotion = data_compare(np_dV, cus_dV, rtol,
                                                                                    rtol_pct_thd, atol)
    display(np_dV, cus_dV, result, v_diff_gt_rtol_proportion, v_diff_gt_max_rtol_proprotion, thd1=rtol)

    assert q_diff_gt_max_rtol_proprotion < 1e-4 \
           and k_diff_gt_max_rtol_proprotion < 1e-4 \
           and v_diff_gt_max_rtol_proprotion < 1e-4


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 1, 128, 80), (1, 1, 128, 80)),
        ((4, 8, 128, 80), (4, 8, 128, 80)),
        ((4, 8, 1024, 128), (4, 8, 1024, 128)),
        ((1, 1, 4096, 128), (1, 1, 4096, 128))
    ],
)
def test_fa_fwd_with_causal_attn_mask(q_shape, kv_shape):
    q = np.random.randn(*q_shape).astype("float16")
    k = np.random.randn(*kv_shape).astype("float16")
    v = np.random.randn(*kv_shape).astype("float16")
    batch_size, q_seq_len, k_seq_len = q_shape[0], q_shape[2], kv_shape[2]

    fa_attn_mask = np.triu(np.ones(shape=(128, 128), dtype=np.float16), k=1)
    np_att_mask = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)

    O, mid_results = np_impl_sa_fwd(q, k, v, np_att_mask)
    _, P, row_sum, row_max = mid_results
    dO = np.random.randn(*q_shape).astype("float16")
    douts = (Tensor(dO), None)
    l = row_max.astype("float32") + np.log(row_sum)
    l = np.squeeze(l, axis=-1)

    model = FlashAttentionGrad()
    cus_grad = model(Tensor(q), Tensor(k), Tensor(v), Tensor(fa_attn_mask), Tensor(l), Tensor(O), douts)
    cus_dQ, cus_dK, cus_dV = cus_grad[0].asnumpy(), cus_grad[1].asnumpy(), cus_grad[2].asnumpy()
    np_dQ, np_dK, np_dV = np_impl_sa_grad(q, k, v, P, O, dO)

    rtol, rtol_pct_thd, atol = 0.005, 0.005, 0.005
    print(f"\n--------- shape: {q_shape}-{kv_shape} -------------")
    print("--------- dQ -------------")
    result, q_diff_gt_rtol_proportion, q_diff_gt_max_rtol_proprotion = data_compare(np_dQ, cus_dQ,
                                                                                    rtol, rtol_pct_thd, atol)
    display(np_dQ, cus_dQ, result, q_diff_gt_rtol_proportion, q_diff_gt_max_rtol_proprotion, thd1=rtol)
    print("--------- dK -------------")
    result, k_diff_gt_rtol_proportion, k_diff_gt_max_rtol_proprotion = data_compare(np_dK, cus_dK,
                                                                                    rtol, rtol_pct_thd, atol)
    display(np_dK, cus_dK, result, k_diff_gt_rtol_proportion, k_diff_gt_max_rtol_proprotion, thd1=rtol)
    print("--------- dV -------------")
    result, v_diff_gt_rtol_proportion, v_diff_gt_max_rtol_proprotion = data_compare(np_dV, cus_dV, rtol,
                                                                                    rtol_pct_thd, atol)
    display(np_dV, cus_dV, result, v_diff_gt_rtol_proportion, v_diff_gt_max_rtol_proprotion, thd1=rtol)

    assert q_diff_gt_max_rtol_proprotion < 1e-4 \
           and k_diff_gt_max_rtol_proprotion < 1e-4 \
           and v_diff_gt_max_rtol_proprotion < 1e-4


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 1, 128, 80), (1, 1, 128, 80)),
        ((4, 8, 128, 80), (4, 8, 128, 80)),
        ((4, 8, 1024, 128), (4, 8, 1024, 128)),
        ((1, 1, 4096, 128), (1, 1, 4096, 128))
    ],
)
def test_fa_fwd_without_attn_mask(q_shape, kv_shape):
    q = np.random.randn(*q_shape).astype("float16")
    k = np.random.randn(*kv_shape).astype("float16")
    v = np.random.randn(*kv_shape).astype("float16")

    O, mid_results = np_impl_sa_fwd(q, k, v)
    _, P, row_sum, row_max = mid_results
    dO = np.random.randn(*q_shape).astype("float16")
    douts = (Tensor(dO), None)
    l = row_max.astype("float32") + np.log(row_sum)
    l = np.squeeze(l, axis=-1)
    k_seq_len = kv_shape[2]
    model = FlashAttentionGrad(next_block_num=k_seq_len)
    cus_grad = model(Tensor(q), Tensor(k), Tensor(v), None, Tensor(l), Tensor(O), douts)
    cus_dQ, cus_dK, cus_dV = cus_grad[0].asnumpy(), cus_grad[1].asnumpy(), cus_grad[2].asnumpy()
    np_dQ, np_dK, np_dV = np_impl_sa_grad(q, k, v, P, O, dO)

    rtol, rtol_pct_thd, atol = 0.005, 0.005, 0.005
    print(f"\n--------- shape: {q_shape}-{kv_shape} -------------")
    print("--------- dQ -------------")
    result, q_diff_gt_rtol_proportion, q_diff_gt_max_rtol_proprotion = data_compare(np_dQ, cus_dQ,
                                                                                    rtol, rtol_pct_thd, atol)
    display(np_dQ, cus_dQ, result, q_diff_gt_rtol_proportion, q_diff_gt_max_rtol_proprotion, thd1=rtol)
    print("--------- dK -------------")
    result, k_diff_gt_rtol_proportion, k_diff_gt_max_rtol_proprotion = data_compare(np_dK, cus_dK,
                                                                                    rtol, rtol_pct_thd, atol)
    display(np_dK, cus_dK, result, k_diff_gt_rtol_proportion, k_diff_gt_max_rtol_proprotion, thd1=rtol)
    print("--------- dV -------------")
    result, v_diff_gt_rtol_proportion, v_diff_gt_max_rtol_proprotion = data_compare(np_dV, cus_dV, rtol,
                                                                                    rtol_pct_thd, atol)
    display(np_dV, cus_dV, result, v_diff_gt_rtol_proportion, v_diff_gt_max_rtol_proprotion, thd1=rtol)

    assert q_diff_gt_max_rtol_proprotion < 1e-4 \
           and k_diff_gt_max_rtol_proprotion < 1e-4 \
           and v_diff_gt_max_rtol_proprotion < 1e-4


@pytest.mark.parametrize(
    "q_npy_path, k_npy_path, v_npy_path",
    [
        ("savepath/q.npy", "savepath/k.npy", "savepath/v.npy"),
    ],
)
def test_fa_grad_compare_with_triangle_attn_grad(q_npy_path, k_npy_path, v_npy_path):
    q = np.load(q_npy_path)
    k = np.load(k_npy_path)
    v = np.load(v_npy_path)

    q_shape = q.shape
    kv_shape = k.shape

    batch_size, q_seq_len, k_seq_len = q_shape[0], q_shape[2], kv_shape[2]

    att_mask = np.triu(np.ones(shape=(batch_size, q_seq_len, k_seq_len), dtype=np.float16), k=1)

    model1 = FlashAttention()
    model2 = TriangleAttention(block_size=8192)

    sens = 100.0 * (np.random.random(q_shape).astype("float16") - 0.5)
    grad = ops.GradOperation(get_all=True, sens_param=True)

    cus_grad = grad(model1)(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask), Tensor(sens))
    triangle_grad = grad(model2)(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask), Tensor(sens))
    cus_dQ, cus_dK, cus_dV = cus_grad[0].asnumpy(), cus_grad[1].asnumpy(), cus_grad[2].asnumpy()
    triangle_dQ, triangle_dK, triangle_dV = triangle_grad[0].asnumpy(), triangle_grad[1].asnumpy(), \
                                            triangle_grad[2].asnumpy()

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
