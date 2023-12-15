# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import pytest
from mindspore import Tensor
from common import data_compare, set_env
set_env()


def np_attention_grad(Q, K, V, dO):
    assert len(Q.shape) == 2
    # forward
    d = Q.shape[-1]
    Sim = np.matmul(Q, K.T)
    row_max = np.max(Sim, axis=-1)
    row_max = row_max.reshape(-1, 1)
    Sim -= row_max
    Sim = Sim.astype("float32")
    Sim = np.exp(Sim)
    row_sum = np.sum(Sim, axis=-1)
    row_sum = row_sum.reshape((-1, 1))
    row_sum_rec = np.reciprocal(row_sum)
    P = Sim * row_sum_rec
    P = P.astype("float16")
    O = np.matmul(P, V)

    # backward
    dV = np.matmul(P.T, dO)
    dP = np.matmul(dO, V.T)
    ret_dP = dP.copy()

    P_dP = np.multiply(P, dP)
    P_dP = P_dP.astype("float32")
    row_sum = np.sum(P_dP, axis=-1)
    row_sum = row_sum.reshape((-1, 1))
    row_sum = row_sum.astype("float16")
    dP -= row_sum
    dS = np.multiply(P, dP)

    dQ = np.matmul(dS, K)
    dK = np.matmul(dS.T, Q)

    return dQ, dK, dV, dS, ret_dP


class FirstMatMul(nn.Cell):
    def __init__(self):
        super(FirstMatMul, self).__init__()
        self.transpose = ops.Transpose()

    def construct(self, Q, K):
        out = ops.matmul(Q, self.transpose(K, (1, 0)))
        return out


class SoftMax(nn.Cell):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.softmax = ops.Softmax()

    def construct(self, Sim):
        out = self.softmax(Sim)
        return out


class SecondMatMul(nn.Cell):
    def __init__(self):
        super(SecondMatMul, self).__init__()

    def construct(self, P, V):
        out = ops.matmul(P, V)
        return out


@pytest.mark.experiment
@pytest.mark.parametrize(
    "seq_len, dim",
    [pytest.param(1024, 80, marks=pytest.mark.time_consuming),
     pytest.param(4096, 128, marks=pytest.mark.time_consuming)],
)
def test_grad_attention(seq_len, dim):
    Q = 20.0 * np.random.random((seq_len, dim)).astype("float16")
    K = 20.0 * np.random.random((seq_len, dim)).astype("float16")
    V = 20.0 * np.random.random((seq_len, dim)).astype("float16")
    model1 = FirstMatMul()
    model2 = SoftMax()
    model3 = SecondMatMul()

    # forward
    Sim = model1(Tensor(Q), Tensor(K))
    P = model2(Sim)
    O = model3(P, Tensor(V))

    # backward
    sens = 10.0 * np.random.random((seq_len, dim)).astype("float16")
    grad = ops.GradOperation(sens_param=True, get_all=True)

    dP, dV = grad(model3)(P, Tensor(V), Tensor(sens))
    (dS,) = grad(model2)(Sim, dP)
    dQ, dK = grad(model1)(Tensor(Q), Tensor(K), dS)

    np_dQ, np_dK, np_dV, np_dS, np_dP = np_attention_grad(Q, K, V, sens)

    print(f"\n test case shape: {Q.shape}, {K.shape}")
    result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = data_compare(
        dQ.asnumpy(), np_dQ
    )
    print(
        f"Compare result: {result}, dQ error_less_than_thousandth_proportion: {error_less_than_thousandth_proportion}, "
        f"dQ error_greater_than_tenth_count: {error_greater_than_tenth_count}"
    )
    result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = data_compare(
        dK.asnumpy(), np_dK
    )
    print(
        f"Compare result: {result}, dK error_less_than_thousandth_proportion: {error_less_than_thousandth_proportion}, "
        f"dK error_greater_than_tenth_count: {error_greater_than_tenth_count}"
    )
    result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = data_compare(
        dV.asnumpy(), np_dV
    )
    print(
        f"Compare result: {result}, dV error_less_than_thousandth_proportion: {error_less_than_thousandth_proportion}, "
        f"dV error_greater_than_tenth_count: {error_greater_than_tenth_count}"
    )


@pytest.mark.experiment
@pytest.mark.parametrize(
    "seq_len, dim", [(1024, 1024), (4096, 4096), ],
)
def test_grad_softmax(seq_len, dim):
    Sim = 20.0 * np.random.random((seq_len, dim)).astype("float16")
    model = SoftMax()
    sens = 10.0 * np.random.random((seq_len, dim)).astype("float16")

    grad = ops.GradOperation(sens_param=True, get_all=True)
    (ms_dSim,) = grad(model)(Tensor(Sim), Tensor(sens))

    dP = sens
    P = model(Tensor(Sim))
    P = P.asnumpy()

    P_dP = np.multiply(P, dP)
    P_dP = P_dP.astype("float32")
    row_sum = np.sum(P_dP, axis=-1)
    row_sum = row_sum.reshape((-1, 1))
    row_sum = row_sum.astype("float16")
    dP -= row_sum
    np_dS = np.multiply(P, dP)

    dS_diff_max = np.max(np.abs((ms_dSim.asnumpy() - np_dS)))
    print(f"\n test case shape: {Sim.shape}")
    print(f"dS max diff: {dS_diff_max}")
    result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = data_compare(
        ms_dSim.asnumpy(), np_dS
    )
    print(
        f"Compare result: {result}, dS error_less_than_thousandth_proportion: {error_less_than_thousandth_proportion}, "
        f"dS error_greater_than_tenth_count: {error_greater_than_tenth_count}"
    )


# test precision: P * dP = P * (dO @ V.T) vs dO * O = dO * (P @ V)
@pytest.mark.experiment
@pytest.mark.parametrize(
    "seq_len, dim", [(64, 80), (128, 80), (256, 80), ],
)
def test_update_Di(seq_len, dim):
    P = np.random.random((seq_len, seq_len)).astype("float32")
    row_sum = np.sum(P, axis=-1, keepdims=True)
    P = P / row_sum
    P = P.astype("float16")
    V = np.random.random((seq_len, dim)).astype("float16")
    dO = np.random.random((seq_len, dim)).astype("float16")

    sa = P * (np.dot(dO, V.T))
    sa_sum = np.sum(sa.astype("float32"), -1)
    fa = dO * (np.dot(P, V))
    fa_sum = np.sum(fa.astype("float32"), -1)

    assert np.allclose(sa_sum.astype("float16"), fa_sum.astype("float16"), 1e-3)
