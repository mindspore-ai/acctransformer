# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import time
import numpy as np
import pytest
from mindspore import Tensor
from mindspore import ops as ops

from common import set_env
from common import FlashAttention
from common import MindsporeAttention

set_env()


@pytest.mark.time_consuming
@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 1, 128 * 1024, 16), (1, 1, 128, 16)),
        ((1, 1, 128, 16), (1, 1, 128 * 1024, 16)),
        ((1, 1, 256 * 1024, 16), (1, 1, 256, 16)),
        ((1, 1, 256, 16), (1, 1, 256 * 1024, 16)),
    ],
)
def test_fa_forward_perf(q_shape, kv_shape):
    q = np.random.random(q_shape).astype("float16")
    k = np.random.random(kv_shape).astype("float16")
    v = np.random.random(kv_shape).astype("float16")
    batch_size, q_seq_len, k_seq_len = q.shape[0], q.shape[2], k.shape[2]
    att_mask = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)
    d = q.shape[-1]
    model1 = FlashAttention(d)
    model2 = MindsporeAttention()

    warm_up = 10
    for _ in range(warm_up):
        model1(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask))
        model2(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask))

    repeat_time = 50
    start_time = time.perf_counter()
    for _ in range(repeat_time):
        model1(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask))
    end_time = time.perf_counter()
    cust_avg_time = (end_time - start_time) / repeat_time

    start_time = time.perf_counter()
    for _ in range(repeat_time):
        model2(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask))
    end_time = time.perf_counter()
    ms_avg_time = (end_time - start_time) / repeat_time

    print(f"\ncustom avg cost time: {cust_avg_time}")
    print(f"mindspore avg cost time: {ms_avg_time}")
    print(f"time reduction rate: {(ms_avg_time - cust_avg_time) / ms_avg_time * 100}%")


@pytest.mark.time_consuming
@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 1, 128 * 1024, 16), (1, 1, 128, 16)),
        ((1, 1, 128, 16), (1, 1, 128 * 1024, 16)),
        ((1, 1, 256 * 1024, 16), (1, 1, 256, 16)),
        ((1, 1, 256, 16), (1, 1, 256 * 1024, 16)),
    ],
)
def test_fa_grad_perf(q_shape, kv_shape):
    q = Tensor(np.random.random(q_shape).astype("float16"))
    k = Tensor(np.random.random(kv_shape).astype("float16"))
    v = Tensor(np.random.random(kv_shape).astype("float16"))
    batch_size, q_seq_len, k_seq_len = q.shape[0], q.shape[2], k.shape[2]
    att_mask = Tensor(np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1))

    dout = Tensor(np.ones(q_shape, dtype="float16"))

    d = q.shape[-1]
    model1 = FlashAttention(d)
    model2 = MindsporeAttention()
    model1(q, k, v, att_mask)
    model2(q, k, v, att_mask)
    grad = ops.GradOperation(sens_param=True)

    warm_up = 10
    for _ in range(warm_up):
        grad(model1)(q, k, v, att_mask, dout)
        grad(model2)(q, k, v, att_mask, dout)

    repeat_time = 50
    start_time = time.perf_counter()
    for _ in range(repeat_time):
        grad(model1)(q, k, v, att_mask, dout)
    end_time = time.perf_counter()
    cust_avg_time = (end_time - start_time) / repeat_time

    start_time = time.perf_counter()
    for _ in range(repeat_time):
        grad(model2)(q, k, v, att_mask, dout)
    end_time = time.perf_counter()
    ms_avg_time = (end_time - start_time) / repeat_time

    print(f"\ncustom avg cost time: {cust_avg_time}")
    print(f"mindspore avg cost time: {ms_avg_time}")
    print(f"time reduction rate: {(ms_avg_time - cust_avg_time) / ms_avg_time * 100}%")
