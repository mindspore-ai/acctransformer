# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import pytest
import numpy as np

from mindspore import Tensor

from common import data_compare, display, set_env
from common import FlashAttention, np_impl_sa_fwd

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
def test_fa_fwd_with_triangle_attn_mask(q_shape, kv_shape):
    q = np.random.random(q_shape).astype("float16")
    k = np.random.random(kv_shape).astype("float16")
    v = np.random.random(kv_shape).astype("float16")
    batch_size, q_seq_len, k_seq_len = q_shape[0], q_shape[2], kv_shape[2]

    att_mask = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)

    model = FlashAttention()
    cus_out = model(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask)).asnumpy()
    np_out, _ = np_impl_sa_fwd(q, k, v, att_mask)

    print(f"\n--------- shape: {q_shape}-{kv_shape} -------------")
    result, diff_gt_rtol_proportion, diff_gt_max_rtol_proportion = data_compare(np_out, cus_out)
    display(np_out, cus_out, result, diff_gt_rtol_proportion, diff_gt_max_rtol_proportion)

    assert result == "Pass"


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
    q = np.random.random(q_shape).astype("float16")
    k = np.random.random(kv_shape).astype("float16")
    v = np.random.random(kv_shape).astype("float16")
    batch_size, q_seq_len, k_seq_len = q_shape[0], q_shape[2], kv_shape[2]

    fa_attn_mask = np.triu(np.ones(shape=(128, 128), dtype=np.float16), k=1)
    np_att_mask = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)

    model = FlashAttention()
    cus_out = model(Tensor(q), Tensor(k), Tensor(v), Tensor(fa_attn_mask)).asnumpy()
    np_out, _ = np_impl_sa_fwd(q, k, v, np_att_mask)

    print(f"\n--------- shape: {q_shape}-{kv_shape} -------------")
    result, diff_gt_rtol_proportion, diff_gt_max_rtol_proportion = data_compare(np_out, cus_out)
    display(np_out, cus_out, result, diff_gt_rtol_proportion, diff_gt_max_rtol_proportion)

    assert result == "Pass"


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 1, 128, 80), (1, 1, 128, 80)),
        ((4, 8, 128, 80), (4, 8, 128, 80)),
        ((4, 8, 1024, 128), (4, 8, 1024, 128)),
        ((1, 1, 256, 128), (1, 1, 128*1024, 128)),
        ((1, 1, 128*1024, 128), (1, 1, 256, 128))
    ],
)
def test_fa_fwd_without_attn_mask(q_shape, kv_shape):
    q = np.random.random(q_shape).astype("float16")
    k = np.random.random(kv_shape).astype("float16")
    v = np.random.random(kv_shape).astype("float16")
    k_seq_len = kv_shape[2]

    model = FlashAttention(next_block_num=k_seq_len)
    cus_out = model(Tensor(q), Tensor(k), Tensor(v)).asnumpy()
    np_out, _ = np_impl_sa_fwd(q, k, v)

    print(f"\n--------- shape: {q_shape}-{kv_shape} -------------")
    result, diff_gt_rtol_proportion, diff_gt_max_rtol_proportion = data_compare(np_out, cus_out)
    display(np_out, cus_out, result, diff_gt_rtol_proportion, diff_gt_max_rtol_proportion)

    assert result == "Pass"
