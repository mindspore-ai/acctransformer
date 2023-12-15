import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import pytest
from mindspore import Tensor

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=5, save_graphs=False)


class MindsporeAttention(nn.Cell):
    def __init__(self):
        super().__init__()
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()

    def construct(self, q, k, v):
        sim = ops.matmul(q, self.transpose(k, (0, 1, 3, 2)))
        print(f"ms Sim : {sim.dtype}")
        attn = self.softmax(sim)
        out = ops.matmul(attn, v)
        return out, sim, attn


class FirstMatMul(nn.Cell):
    def __init__(self):
        super(FirstMatMul, self).__init__()
        self.transpose = ops.Transpose()

    def construct(self, Q, K):
        K = self.transpose(K, (0, 1, 3, 2))
        out = ops.matmul(Q, K)
        return out


class SoftMax(nn.Cell):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.softmax = ops.Softmax(axis=-1)

    def construct(self, Sim):
        print(f"Sim : {Sim.dtype}")
        out = self.softmax(Sim)
        return out


class SecondMatMul(nn.Cell):
    def __init__(self):
        super(SecondMatMul, self).__init__()

    def construct(self, P, V):
        out = ops.matmul(P, V)
        return out


def ms_impl_attention_forward(q, k, v):
    softmax = ops.Softmax(axis=-1)
    transpose = ops.Transpose()
    k = transpose(k, (0, 1, 3, 2))
    sim = ops.matmul(q, k)
    attn = softmax(sim)
    out = ops.matmul(attn, v)
    return out


def data_compare(ground_truth, predict, diff_thd=0.001, pct_thd=0.001, max_diff_thd=0.1):
    total_count = np.prod(ground_truth.shape)
    greater_than_diff_thd_count = np.sum(
        np.abs(predict.astype("float32") - ground_truth.astype("float32")) > diff_thd *
        (np.abs(ground_truth.astype("float32")) + 1e-9)
    )
    greater_than_max_diff_thd_count = np.sum(
        np.abs(predict.astype("float32") - ground_truth.astype("float32")) > max_diff_thd *
        (np.abs(ground_truth.astype("float32")) + 1e-9)
    )

    error_less_than_diff_thd_proportion = 1.0 - greater_than_diff_thd_count / total_count
    result = "Pass"
    if error_less_than_diff_thd_proportion < 1 - pct_thd or greater_than_max_diff_thd_count > 0:
        result = "Failed"
    return result, error_less_than_diff_thd_proportion, greater_than_max_diff_thd_count


@pytest.mark.parametrize(
    "q_shape, kv_shape, dtype",
    [
        ((1, 1, 128, 80), (1, 1, 128, 80), "float16"),
        # ((4, 8, 1024, 80), (4, 8, 1024, 80), "float16"),
        # ((4, 8, 4096, 128), (4, 8, 4096, 128), "float16"),
    ],
)
def test_attention_forward(q_shape, kv_shape, dtype):
    Q = np.random.random(q_shape).astype(dtype)
    K = np.random.random(kv_shape).astype(dtype)
    V = np.random.random(kv_shape).astype(dtype)

    model1 = FirstMatMul()
    model2 = SoftMax()
    model3 = SecondMatMul()
    # forward
    Sim = model1(Tensor(Q), Tensor(K))
    P = model2(Sim)
    O = model3(P, Tensor(V))
    # backward
    # sens = 10.0 * np.random.random(q_shape).astype("float16")
    # grad = ops.GradOperation(sens_param=True, get_all=True)

    # dP, dV = grad(model3)(P, Tensor(V), Tensor(sens))
    # dS, = grad(model2)(Sim, dP)
    # dQ, dK = grad(model1)(Tensor(Q), Tensor(K), dS)

    model = MindsporeAttention()
    ms_O, ms_Sim, ms_P = model(Tensor(Q), Tensor(K), Tensor(V))
    # ms_dQ, ms_dK, ms_dV = grad(model)(Tensor(Q), Tensor(K), Tensor(V), Tensor(sens))

    print("\n------------ Sim -----------------")
    result, error_less_than_thousandth_proportion, error_greater_than_tenth_count \
        = data_compare(ms_Sim.asnumpy(), Sim.asnumpy())
    print(
        f"Compare result: {result}, error_less_than_thousandth_proportion: {error_less_than_thousandth_proportion}, "
        f"error_greater_than_tenth_count: {error_greater_than_tenth_count}")

    print("-------------- P attn-----------------")
    result, error_less_than_thousandth_proportion, error_greater_than_tenth_count \
        = data_compare(ms_P.asnumpy(), P.asnumpy())
    print(
        f"Compare result: {result}, error_less_than_thousandth_proportion: {error_less_than_thousandth_proportion}, "
        f"error_greater_than_tenth_count: {error_greater_than_tenth_count}")
    # assert np.allclose(P.asnumpy(), ms_P.asnumpy(), 1e-3)
    print(f"\n P: {P.dtype, P.asnumpy()} \n")
    print(f"\n ms_P: {ms_P.dtype, ms_P.asnumpy} \n")

    print("-------------- Out -----------------")
    result, error_less_than_thousandth_proportion, error_greater_than_tenth_count \
        = data_compare(ms_O.asnumpy(), O.asnumpy())
    print(
        f"Compare result: {result}, error_less_than_thousandth_proportion: {error_less_than_thousandth_proportion}, "
        f"error_greater_than_tenth_count: {error_greater_than_tenth_count}")

    # print("-------------- dQ -----------------")
    # result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = data_compare(
    #                                                                                     ms_dQ.asnumpy(), dQ.asnumpy())
    # print(
    #     f"Compare result: {result}, error_less_than_thousandth_proportion: {error_less_than_thousandth_proportion}, "
    #     f"error_greater_than_tenth_count: {error_greater_than_tenth_count}")
    #
    # print("-------------- dK -----------------")
    # result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = data_compare(
    #     ms_dK.asnumpy(), dK.asnumpy())
    # print(
    #     f"Compare result: {result}, error_less_than_thousandth_proportion: {error_less_than_thousandth_proportion}, "
    #     f"error_greater_than_tenth_count: {error_greater_than_tenth_count}")
    #
    # print("-------------- dV -----------------")
    # result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = data_compare(
    #     ms_dV.asnumpy(), dV.asnumpy())
    # print(
    #     f"Compare result: {result}, error_less_than_thousandth_proportion: {error_less_than_thousandth_proportion}, "
    #     f"error_greater_than_tenth_count: {error_greater_than_tenth_count}")