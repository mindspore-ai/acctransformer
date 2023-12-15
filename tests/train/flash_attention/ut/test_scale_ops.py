import mindspore
import mindspore.context as context
import numpy as np
import pytest
from tbe import tik

from flash_attention_test_agent import FlashAttentionTestAgent

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)


class ScaleComputer(FlashAttentionTestAgent):
    def define_ios(self, *args, **kwargs):
        dim, shape, dtype = (
            kwargs.get("dim"),
            kwargs.get("shape"),
            kwargs.get("dtype"),
        )
        self.dim = dim
        self.M, self.N = shape
        self.dtype = dtype
        self.data = self.tik_instance.Tensor(dtype, shape, name="data", scope=tik.scope_gm)
        self.scale_data = self.tik_instance.Tensor(
            dtype, shape, name="scale_data", scope=tik.scope_gm
        )

    def scale_compute_vector(self, Sij_ub, dim):
        scale_value = dim ** -0.5
        scale = self.tik_instance.Scalar(dtype="float16")
        scale.set_as(scale_value)
        self.tik_instance.h_mul(Sij_ub, Sij_ub, scale)
        return Sij_ub

    def compute(self, *args, **kwargs):
        data_ub = self.tik_instance.Tensor(
            self.dtype, (self.M, self.N), name="data_ub", scope=tik.scope_ubuf
        )
        self.fa.tik_instance.data_move(data_ub, self.data, 0, 1, self.M * self.N // 16, 0, 0)
        # scale
        scaled_data_ub = self.scale_compute_vector(data_ub, self.dim)
        self.fa.tik_instance.data_move(
            self.scale_data, scaled_data_ub, 0, 1, self.M * self.N // 16, 0, 0
        )
        self.tik_instance.BuildCCE(
            kernel_name="mock_scale",
            inputs=[self.data],
            outputs=[self.scale_data],
        )
        return self.tik_instance


@pytest.mark.skip(reason="The scale computing was moved outside flash-attention")
@pytest.mark.parametrize(
    "dim, shape, dtype",
    [
        (64, (128, 128), "float16"),
        (80, (128, 128), "float16"),
        (128, (128, 128), "float16"),
    ],
)
def test_scale_compute(dim, shape, dtype):
    dim = dim
    data = np.random.random(shape).astype(dtype)
    # np impl scale
    np_scale = dim ** -0.5
    np_scale_data = data * np_scale
    print(f"\nnp type: {type(np_scale)}, {np_scale_data.dtype}")
    # ms impl scale
    ms_scale = mindspore.Tensor(dim) ** -0.5
    ms_scale_data = mindspore.ops.multiply(mindspore.Tensor(data), ms_scale)
    print(f"\nms type: {ms_scale.dtype}, {ms_scale_data.dtype}")
    # tik impl scale
    mfa = ScaleComputer(**dict(dim=dim, shape=shape, dtype=dtype))
    tik_instance = mfa.compute()
    feed_dict = {"data": data}
    tik_scale_data, = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)

    print(f"np_scale: {np_scale}")
    print(f"ms_scale: {ms_scale.asnumpy()}")
    # print(f"tik_scale: {tik_scale}")

    assert np.allclose(np_scale, ms_scale.asnumpy(), 1e-8)
    np_ms_eq = np.allclose(np_scale_data, ms_scale_data.asnumpy(), 1e-3)
    tik_ms_eq = np.allclose(tik_scale_data, ms_scale_data.asnumpy(), 1e-3)
    np_tik_eq = np.allclose(np_scale_data, tik_scale_data, 1e-8)
    # pytest.assume(np_ms_eq)
    # pytest.assume(tik_ms_eq)
    pytest.assume(np_tik_eq)
