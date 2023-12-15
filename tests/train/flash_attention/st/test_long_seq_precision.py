# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import numpy as np
import pytest
from mindspore import Tensor
from mindspore import ops as ops
import mindspore as ms
from mindspore import nn as nn

from common import data_compare, display, set_env
from common import FlashAttention, FlashAttentionGrad
from common import MindsporeAttention
from common import np_impl_attention_forward, np_impl_attention_grad_Vfa, np_impl_attention_grad
from common import ms_impl_attention_forward, ms_impl_attention_grad_Vfa

set_env()


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        # ((1, 1, 128 * 1024, 16), (1, 1, 128, 16)),
        # ((1, 1, 128, 16), (1, 1, 128 * 1024, 16)),
        ((1, 1, 128*1024, 80), (1, 1, 128, 80)),
        ((1, 1, 128, 80), (1, 1, 128*1024, 80)),
        # pytest.param((1, 1, 256 * 1024, 16), (1, 1, 256, 16), marks=pytest.mark.time_consuming),
        # pytest.param((1, 1, 256, 16), (1, 1, 256 * 1024, 16), marks=pytest.mark.time_consuming),
    ],
)
def test_fa_forward_precision(q_shape, kv_shape):
    q = np.random.random(q_shape).astype("float16")
    k = np.random.random(kv_shape).astype("float16")
    v = np.random.random(kv_shape).astype("float16")
    batch_size, q_seq_len, k_seq_len = q.shape[0], q.shape[2], k.shape[2]
    att_mask = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)

    high_precision = True
    d = q.shape[-1]
    model1 = FlashAttention(d, high_precision=high_precision)
    model2 = MindsporeAttention()

    # test accuracy
    # profiler =  Profiler(output_path="/home/fuyong/codes/master/tests/ops/tbe/st/profiler_data/opt_12_4096_80", profile_communication=True)
    cus_out = model1(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask)).asnumpy()
    # profiler.analyse()
    ms_out = model2(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask)).asnumpy()
    np_out, _, _, _, _ = np_impl_attention_forward(q, k, v, att_mask)
    print(f"\n--------- shape: {q_shape} -------------")

    print("-------------- cus vs ms --------------------")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(ms_out, cus_out)
    print(f"Compare result: {result}, diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"diff greater than 10% proportion: {diff_gt_tenth_proportion}")

    print("\n-------------- cus vs np --------------------")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(np_out, cus_out)
    print(f"Compare result: {result}, diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"diff greater than 10% proportion: {diff_gt_tenth_proportion}")


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 1, 128 * 1024, 16), (1, 1, 128, 16)),
        ((1, 1, 128, 16), (1, 1, 128 * 1024, 16)),
        pytest.param((1, 1, 256 * 1024, 16), (1, 1, 256, 16), marks=pytest.mark.time_consuming),
        pytest.param((1, 1, 256, 16), (1, 1, 256 * 1024, 16), marks=pytest.mark.time_consuming),
    ],
)
def test_fa_wo_attn_drop_mask_forward_precision(q_shape, kv_shape):
    q = np.random.random(q_shape).astype("float16")
    k = np.random.random(kv_shape).astype("float16")
    v = np.random.random(kv_shape).astype("float16")

    d = q.shape[-1]
    model1 = FlashAttention(d, next_block_num=65536)  # 没有attn_mask时，算子倒三角特性不能用
    model2 = MindsporeAttention()

    # test accuracy
    cus_out = model1(Tensor(q), Tensor(k), Tensor(v)).asnumpy()
    ms_out = model2(Tensor(q), Tensor(k), Tensor(v)).asnumpy()
    np_out, _, _, _, _ = np_impl_attention_forward(q, k, v)
    print(f"\n--------- shape: {q_shape} -------------")

    print("-------------- cus vs ms --------------------")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(ms_out, cus_out)
    print(f"Compare result: {result}, diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"diff greater than 10% proportion: {diff_gt_tenth_proportion}")

    print("\n-------------- cus vs np --------------------")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(np_out, cus_out)
    print(f"Compare result: {result}, diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"diff greater than 10% proportion: {diff_gt_tenth_proportion}")


@pytest.mark.skip(reason='dropout_mask too large')
@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 1, 128 * 1024, 16), (1, 1, 128, 16)),
        ((1, 1, 128, 16), (1, 1, 128 * 1024, 16)),
        pytest.param((1, 1, 256 * 1024, 16), (1, 1, 256, 16), marks=pytest.mark.time_consuming),
        pytest.param((1, 1, 256, 16), (1, 1, 256 * 1024, 16), marks=pytest.mark.time_consuming),
    ],
)
def test_fa_forward_precision_mask(q_shape, kv_shape):
    q = np.random.random(q_shape).astype("float16")
    k = np.random.random(kv_shape).astype("float16")
    v = np.random.random(kv_shape).astype("float16")
    batch_size, q_seq_len, k_seq_len = q.shape[0], q.shape[2], k.shape[2]
    att_mask = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)

    d = q.shape[-1]
    keep_prob = 1.0
    high_precision = True
    model1 = FlashAttention(d, keep_prob=keep_prob, high_precision=high_precision)

    drop_mask_ops = nn.Dropout(keep_prob)
    drop_mask_ops.set_train()
    drop_mask = drop_mask_ops(ms.numpy.ones((q_shape[0], q_shape[1], q_shape[2], q_shape[2]), dtype=ms.uint8))

    # test accuracy
    cus_out = model1(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask), drop_mask).asnumpy()
    ms_out2 = ms_impl_attention_forward(Tensor(q), Tensor(k), Tensor(v),
                                        Tensor(att_mask), drop_mask)[0]
    ms_out2 = ms_out2.asnumpy()

    print(f"\n--------- shape: {q_shape} -------------")
    print("\n-------------- cus vs ms (ops function) --------------------")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(ms_out2, cus_out)
    display(ms_out2, cus_out, result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion)


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 1, 128 * 1024, 16), (1, 1, 128, 16)),
        ((1, 1, 128, 16), (1, 1, 128 * 1024, 16)),
        pytest.param((1, 1, 256 * 1024, 16), (1, 1, 256, 16), marks=pytest.mark.time_consuming),
        pytest.param((1, 1, 256, 16), (1, 1, 256 * 1024, 16), marks=pytest.mark.time_consuming),
    ],
)
def test_fa_grad_precision(q_shape, kv_shape):
    q = 10.0 * (np.random.random(q_shape).astype("float16") - 0.3)
    k = 10.0 * (np.random.random(kv_shape).astype("float16") - 0.3)
    v = 10.0 * (np.random.random(kv_shape).astype("float16") - 0.3)

    batch_size, q_seq_len, k_seq_len = q.shape[0], q.shape[2], k.shape[2]
    att_mask = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)

    d = q.shape[-1]
    model1 = FlashAttention(d)
    model2 = MindsporeAttention()

    sens = 100.0 * (np.random.random(q_shape).astype("float16") - 0.5)
    grad = ops.GradOperation(sens_param=True, get_all=True)

    cus_dQ, cus_dK, cus_dV, cus_dMask = grad(model1)(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask), Tensor(sens))
    ms_dQ, ms_dK, ms_dV, ms_dMask = grad(model2)(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask), Tensor(sens))
    np_O, np_Sim, np_P, row_sum, row_max = np_impl_attention_forward(q, k, v, att_mask)
    np_dQ, np_dK, np_dV, = np_impl_attention_grad(q, k, v, np_P, sens)
    np_dQ2, np_dK2, np_dV2 = np_impl_attention_grad_Vfa(q, k, v, att_mask, row_sum, row_max, np_O, sens)

    print(f"\n--------- shape: {q_shape} -------------")

    print("-------------- cus vs ms --------------------")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(ms_dQ.asnumpy(), cus_dQ.asnumpy())
    print(f"Compare result: {result}, dQ diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"dQ diff greater than 10% proportion: {diff_gt_tenth_proportion}")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(ms_dK.asnumpy(), cus_dK.asnumpy())
    print(f"Compare result: {result}, dK diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"dK diff greater than 10% proportion: {diff_gt_tenth_proportion}")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(ms_dV.asnumpy(), cus_dV.asnumpy())
    print(f"Compare result: {result}, dV diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"dV diff greater than 10% proportion: {diff_gt_tenth_proportion}")

    print(f"\n-------------- cus vs np_sa --------------------")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(np_dQ, cus_dQ.asnumpy())
    print(f"Compare result: {result}, dQ diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"dQ diff greater than 10% proportion: {diff_gt_tenth_proportion}")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(np_dK, cus_dK.asnumpy())
    print(f"Compare result: {result}, dK diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"dK diff greater than 10% proportion: {diff_gt_tenth_proportion}")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(np_dV, cus_dV.asnumpy())
    print(f"Compare result: {result}, dV diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"dV diff greater than 10% proportion: {diff_gt_tenth_proportion}")

    print(f"\n-------------- cus vs np_fa --------------------")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(np_dQ2, cus_dQ.asnumpy())
    print(f"Compare result: {result}, dQ diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"dQ diff greater than 10% proportion: {diff_gt_tenth_proportion}")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(np_dK2, cus_dK.asnumpy())
    print(f"Compare result: {result}, dK diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"dK diff greater than 10% proportion: {diff_gt_tenth_proportion}")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(np_dV2, cus_dV.asnumpy())
    print(f"Compare result: {result}, dV diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"dV diff greater than 10% proportion: {diff_gt_tenth_proportion}")


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 1, 128 * 1024, 16), (1, 1, 128, 16)),
        ((1, 1, 128, 16), (1, 1, 128 * 1024, 16)),
        pytest.param((1, 1, 256 * 1024, 16), (1, 1, 256, 16), marks=pytest.mark.time_consuming),
        pytest.param((1, 1, 256, 16), (1, 1, 256 * 1024, 16), marks=pytest.mark.time_consuming),
    ],
)
def test_fa_grad_wo_attn_drop_precision(q_shape, kv_shape):
    q = 10.0 * (np.random.random(q_shape).astype("float16") - 0.3)
    k = 10.0 * (np.random.random(kv_shape).astype("float16") - 0.3)
    v = 10.0 * (np.random.random(kv_shape).astype("float16") - 0.3)

    d = q.shape[-1]
    model1 = FlashAttention(d, next_block_num=65536)  # 没有attn_mask时，算子倒三角特性不能用
    model2 = MindsporeAttention()

    sens = 100.0 * (np.random.random(q_shape).astype("float16") - 0.5)
    grad = ops.GradOperation(sens_param=True, get_all=True)

    cus_dQ, cus_dK, cus_dV = grad(model1)(Tensor(q), Tensor(k), Tensor(v), Tensor(sens))
    ms_dQ, ms_dK, ms_dV = grad(model2)(Tensor(q), Tensor(k), Tensor(v), Tensor(sens))

    print(f"\n--------- shape: {q_shape} -------------")

    print("-------------- cus vs ms --------------------")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(ms_dQ.asnumpy(), cus_dQ.asnumpy())
    print(f"Compare result: {result}, dQ diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"dQ diff greater than 10% proportion: {diff_gt_tenth_proportion}")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(ms_dK.asnumpy(), cus_dK.asnumpy())
    print(f"Compare result: {result}, dK diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"dK diff greater than 10% proportion: {diff_gt_tenth_proportion}")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(ms_dV.asnumpy(), cus_dV.asnumpy())
    print(f"Compare result: {result}, dV diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"dV diff greater than 10% proportion: {diff_gt_tenth_proportion}")


@pytest.mark.skip(reason='dropout_mask too large')
@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 1, 128 * 1024, 16), (1, 1, 128, 16)),
        ((1, 1, 128, 16), (1, 1, 128 * 1024, 16)),
        pytest.param((1, 1, 256 * 1024, 16), (1, 1, 256, 16), marks=pytest.mark.time_consuming),
        pytest.param((1, 1, 256, 16), (1, 1, 256 * 1024, 16), marks=pytest.mark.time_consuming),
    ],
)
def test_fa_grad_accuracy_mask(q_shape, kv_shape):
    q = 10 * np.random.random(q_shape).astype("float16")
    k = 10 * np.random.random(kv_shape).astype("float16")
    v = 10 * np.random.random(kv_shape).astype("float16")
    batch_size, q_seq_len, k_seq_len = q.shape[0], q.shape[2], k.shape[2]
    att_mask = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)
    sens = 100.0 * (np.random.random(q_shape).astype("float16") - 0.5)

    d = q.shape[-1]
    keep_prob = 1.0
    high_precision = True
    model = FlashAttentionGrad(d, high_precision=high_precision)

    drop_mask_ops = nn.Dropout(keep_prob)
    drop_mask_ops.set_train()
    drop_mask = drop_mask_ops(ms.numpy.ones((q_shape[0], q_shape[1], q_shape[2], q_shape[2]), dtype=ms.uint8))
    # print(f"drop_mask: {drop_mask[0, 0, :, :]}")

    # cal O, rowsum, rowmax
    O, Sim, P, row_sum, row_max = ms_impl_attention_forward(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask)
                                                            , drop_mask, )
    O, Sim, P, row_sum, row_max = O.asnumpy(), Sim.asnumpy(), P.asnumpy(), \
                                  row_sum.asnumpy(), row_max.asnumpy()
    # test accuracy
    if not high_precision:
        row_sum = row_sum.astype("float16")
    # profile = Profiler(output_path='/home/l30029977/MindSpeed_NZ/train/profiling')
    cus_dQ, cus_dK, cus_dV = model(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask), Tensor(row_sum), Tensor(row_max),
                                   Tensor(O), Tensor(sens), drop_mask, )
    ms_dQ, ms_dK, ms_dV = ms_impl_attention_grad_Vfa(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask),
                                                     Tensor(row_sum), Tensor(row_max), Tensor(O),
                                                     Tensor(sens), drop_mask)

    print(f"\n--------- shape: {q_shape} -------------")

    print("-------------- cus vs ms (ops function) --------------------")
    print("==========>dQ:")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(ms_dQ.asnumpy(), cus_dQ.asnumpy())
    display(ms_dQ.asnumpy(), cus_dQ.asnumpy(), result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion)
    print("==========>dK:")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(ms_dK.asnumpy(), cus_dK.asnumpy())
    display(ms_dK.asnumpy(), cus_dK.asnumpy(), result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion)
    print("==========>dV:")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(ms_dV.asnumpy(), cus_dV.asnumpy())
    display(ms_dV.asnumpy(), cus_dV.asnumpy(), result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion)


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 1, 128 * 1024, 16), (1, 1, 128, 16)),
        ((1, 1, 128, 16), (1, 1, 128 * 1024, 16)),
        pytest.param((1, 1, 256 * 1024, 16), (1, 1, 256, 16), marks=pytest.mark.time_consuming),
        pytest.param((1, 1, 256, 16), (1, 1, 256 * 1024, 16), marks=pytest.mark.time_consuming),
    ],
)
def test_fa_grad_attn_mask_shape(q_shape, kv_shape):
    q = np.random.random(q_shape).astype("float16")
    k = np.random.random(kv_shape).astype("float16")
    v = np.random.random(kv_shape).astype("float16")

    # sequence_mask
    batch_size, q_seq_len, k_seq_len = q.shape[0], q.shape[2], k.shape[2]
    att_mask1 = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)
    att_mask2 = np.triu(np.ones(shape=(batch_size, q_seq_len, k_seq_len), dtype=np.float16), k=1)

    d = q.shape[-1]
    model = FlashAttention(d)

    sens = np.random.random(q_shape).astype("float16")
    grad = ops.GradOperation(sens_param=True, get_all=True)

    dQ1, dK1, dV1, dMask1 = grad(model)(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask1), Tensor(sens))
    dQ2, dK2, dV2, dMask2 = grad(model)(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask2), Tensor(sens))

    dQ1, dQ2, dK1, dK2, dV1, dV2 = dQ1.asnumpy(), dQ2.asnumpy(), dK1.asnumpy(), dK2.asnumpy(), dV1.asnumpy(), dV2.asnumpy()

    print(f"\ndQ_max_diff: {np.max(np.abs(dQ1- dQ2))}")
    print(f"dK_max_diff: {np.max(np.abs(dK1 - dK2))}")
    print(f"dV_max_diff: {np.max(np.abs(dV1 - dV2))}")

    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(dK1, dK2)
    print(f"Compare result: {result}, dK diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"dK diff greater than 10% proportion: {diff_gt_tenth_proportion}")
    result, diff_gt_thousandth_proportion, diff_gt_tenth_proportion = data_compare(dV1, dV2)
    print(f"Compare result: {result}, dV diff greater than 0.1% proportion: {diff_gt_thousandth_proportion}, "
          f"dV diff greater than 10% proportion: {diff_gt_tenth_proportion}")

    # indices = np.where(dK1 != dK2)
    #
    # for row, col in zip(indices[2], indices[3]):
    #     print(f"dK在位置 ({row}, {col})不同，值分别为：{dK1[0,0,row, col]} 和 {dK2[0,0,row, col]}")
    # print('='*30)
    # indices = np.where(dV1 != dV2)
    # for row, col in zip(indices[2], indices[3]):
    #     print(f"dV在位置 ({row}, {col})不同，值分别为：{dV1[0,0,row, col]} 和 {dV2[0,0,row, col]}")

    rtol = 1e-8
    # kv_rtol = 1e-2
    assert np.allclose(dQ1, dQ2, rtol=rtol)
    assert np.allclose(dK1, dK2, rtol=rtol)
    assert np.allclose(dV1, dV2, rtol=rtol)
