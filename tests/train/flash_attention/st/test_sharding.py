# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import numpy as np
import pytest
from mindspore import Tensor
from mindspore import ops as ops
from common import set_env
from common import FlashAttention

set_env()


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((4, 8, 1024, 80), (4, 8, 1024, 80)),
        pytest.param((4, 8, 4096, 128), (4, 8, 4096, 128), marks=pytest.mark.time_consuming),
    ],
)
def test_fa_shard(q_shape, kv_shape):
    q = np.random.random(q_shape).astype("float16")
    k = np.random.random(kv_shape).astype("float16")
    v = np.random.random(kv_shape).astype("float16")
    # sequence_mask
    batch_size, q_seq_len, k_seq_len = q.shape[0], q.shape[2], k.shape[2]
    att_mask = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)
    d = q.shape[-1]
    model = FlashAttention(d)

    # test accuracy
    output = model(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask)).asnumpy()

    output1 = model(Tensor(q[0:2, :, :, :]), Tensor(k[0:2, :, :, :]), Tensor(v[0:2, :, :, :]),
                    Tensor(att_mask)).asnumpy()
    output2 = model(Tensor(q[2:, :, :, :]), Tensor(k[2:, :, :, :]), Tensor(v[2:, :, :, :]),
                    Tensor(att_mask)).asnumpy()
    concated_out = np.concatenate((output1, output2), axis=0)

    print(f"\nout_max_diff: {np.max(np.abs(output - concated_out))}")
    assert np.allclose(output, concated_out, rtol=1e-8)


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((4, 8, 256, 80), (4, 8, 256, 80)),
        pytest.param((4, 8, 1024, 80), (4, 8, 1024, 80), marks=pytest.mark.time_consuming),
        pytest.param((4, 8, 4096, 128), (4, 8, 4096, 128), marks=pytest.mark.time_consuming),
    ],
)
def test_fa_grad_shard(q_shape, kv_shape):
    q = np.random.random(q_shape).astype("float16")
    k = np.random.random(kv_shape).astype("float16")
    v = np.random.random(kv_shape).astype("float16")

    # sequence_mask
    batch_size, q_seq_len, k_seq_len = q.shape[0], q.shape[2], k.shape[2]
    att_mask = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)

    d = q.shape[-1]
    model = FlashAttention(d)

    sens = np.random.random(q_shape).astype("float16")
    grad = ops.GradOperation(sens_param=True, get_all=True)

    dQ, dK, dV, dMask = grad(model)(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask), Tensor(sens))

    dQ1, dK1, dV1, dMask1 = grad(model)(Tensor(q[0:2, :, :, :]), Tensor(k[0:2, :, :, :]), Tensor(v[0:2, :, :, :]),
                                        Tensor(att_mask), Tensor(sens[0:2, :, :, :]))
    dQ2, dK2, dV2, dMask2 = grad(model)(Tensor(q[2:, :, :, :]), Tensor(k[2:, :, :, :]), Tensor(v[2:, :, :, :]),
                                        Tensor(att_mask), Tensor(sens[2:, :, :, :]))

    concated_dQ = np.concatenate((dQ1.asnumpy(), dQ2.asnumpy()), axis=0)
    concated_dK = np.concatenate((dK1.asnumpy(), dK2.asnumpy()), axis=0)
    concated_dV = np.concatenate((dV1.asnumpy(), dV2.asnumpy()), axis=0)

    print(f"\ndQ_max_diff: {np.max(np.abs(dQ.asnumpy() - concated_dQ))}")
    print(f"dK_max_diff: {np.max(np.abs(dK.asnumpy() - concated_dK))}")
    print(f"dV_max_diff: {np.max(np.abs(dV.asnumpy() - concated_dV))}")

    rtol = 1e-8
    assert np.allclose(dQ.asnumpy(), concated_dQ, rtol=rtol)
    assert np.allclose(dK.asnumpy(), concated_dK, rtol=rtol)
    assert np.allclose(dV.asnumpy(), concated_dV, rtol=rtol)


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((4, 8, 1024, 80), (4, 8, 1024, 80)),
        ((4, 8, 4096, 128), (4, 8, 4096, 128)),
    ],
)
def test_fa_grad_stability(q_shape, kv_shape):
    q = np.random.random(q_shape).astype("float16")
    k = np.random.random(kv_shape).astype("float16")
    v = np.random.random(kv_shape).astype("float16")

    # sequence_mask
    batch_size, q_seq_len, k_seq_len = q.shape[0], q.shape[2], k.shape[2]
    att_mask = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)

    d = q.shape[-1]
    model = FlashAttention(d)

    sens = np.random.random(q_shape).astype("float16")
    grad = ops.GradOperation(sens_param=True, get_all=True)

    dQ1, dK1, dV1, dMask1 = grad(model)(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask), Tensor(sens))
    dQ2, dK2, dV2, dMask2 = grad(model)(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask), Tensor(sens))

    print(f"\ndQ_max_diff: {np.max(np.abs(dQ1.asnumpy() - dQ2.asnumpy()))}")
    print(f"dK_max_diff: {np.max(np.abs(dK1.asnumpy() - dK2.asnumpy()))}")
    print(f"dV_max_diff: {np.max(np.abs(dV1.asnumpy() - dV2.asnumpy()))}")

    rtol = 1e-8
    assert np.allclose(dQ1.asnumpy(), dQ2.asnumpy(), rtol=rtol)
    assert np.allclose(dK1.asnumpy(), dK2.asnumpy(), rtol=rtol)
    assert np.allclose(dV1.asnumpy(), dV2.asnumpy(), rtol=rtol)