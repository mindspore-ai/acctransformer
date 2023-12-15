# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
import numpy as np
import pytest
from tbe import tik

from flash_attention_test_agent import FlashAttentionTestAgent


class BroadCast(FlashAttentionTestAgent):
    def define_ios(self, *args, **kwargs):
        M, N, dtype = (
            kwargs.get("M"),
            kwargs.get("N"),
            kwargs.get("dtype"),
        )
        self.M = M
        self.N = N
        self.dtype = dtype
        self.src_gm = self.tik_instance.Tensor(dtype, (M,), name="src_gm", scope=tik.scope_gm)
        self.dst_gm = self.tik_instance.Tensor(dtype, (M, N), name="dst_gm", scope=tik.scope_gm)

    def compute(self, *args, **kwargs):
        src_ub = self.tik_instance.Tensor(self.dtype, (self.M,), name="src_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(src_ub, self.src_gm, 0, 1, self.M // 16, 0, 0)
        result_ub = self.fa.tik_ops_utils.broadcast(src_ub, (self.M, self.N))
        self.tik_instance.data_move(self.dst_gm, result_ub, 0, 1, self.M * self.N // 16, 0, 0)
        self.tik_instance.BuildCCE(
            kernel_name="mock_flash_attention", inputs=[self.src_gm], outputs=[self.dst_gm],
        )
        return self.tik_instance


class BroadCastRow(FlashAttentionTestAgent):
    def define_ios(self, *args, **kwargs):
        M, N, dtype = (
            kwargs.get("M"),
            kwargs.get("N"),
            kwargs.get("dtype"),
        )
        self.M = M
        self.N = N
        self.dtype = dtype
        self.src_gm = self.tik_instance.Tensor(dtype, (N,), name="src_gm", scope=tik.scope_gm)
        self.dst_gm = self.tik_instance.Tensor(dtype, (M, N), name="dst_gm", scope=tik.scope_gm)

    def compute(self, *args, **kwargs):
        src_ub = self.tik_instance.Tensor(self.dtype, (self.N,), name="src_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(src_ub, self.src_gm, 0, 1, self.N // 16, 0, 0)
        result_ub = self.fa.tik_ops_utils.broadcast_row(src_ub, (self.M, self.N))
        self.tik_instance.data_move(self.dst_gm, result_ub, 0, 1, self.M * self.N // 16, 0, 0)
        self.tik_instance.BuildCCE(
            kernel_name="mock_flash_attention", inputs=[self.src_gm], outputs=[self.dst_gm],
        )
        return self.tik_instance


class TestBroadCast:

    @staticmethod
    def np_broadcast(vec, M, N):
        matrix = np.broadcast_to(vec, (N, M))
        matrix = matrix.T
        return matrix

    @staticmethod
    def np_broadcast_row(vec, M, N):
        matrix = np.broadcast_to(vec, (M, N))
        return matrix

    @pytest.mark.parametrize(
        "M, N", [(64, 64), (128, 128), (128, 192), (256, 256), (512, 128), (128, 512)],
    )
    def test_broadcast(self, M, N):
        dtype = "float16"
        mfa = BroadCast(**dict(M=M, N=N, dtype=dtype))
        tik_instance = mfa.compute()
        vec = np.random.random([M]).astype(dtype)
        (tik_matrix,) = tik_instance.tikdb.start_debug(
            feed_dict={"src_gm": vec}, interactive=False
        )
        np_matrix = self.np_broadcast(vec, M, N)
        rtol = 1e-10
        print(vec)
        print(tik_matrix)
        assert np.allclose(tik_matrix, np_matrix, rtol)

    @pytest.mark.parametrize(
        "M, N", [(24, 64), (64, 64), (127, 128), (128, 192), (256, 256), (512, 128), (128, 512)],
    )
    def test_broadcast_row(self, M, N):
        dtype = "float16"
        mfa = BroadCastRow(**dict(M=M, N=N, dtype=dtype))
        tik_instance = mfa.compute()
        vec = np.random.random([N]).astype(dtype)
        (tik_matrix,) = tik_instance.tikdb.start_debug(
            feed_dict={"src_gm": vec}, interactive=False
        )
        np_matrix = self.np_broadcast_row(vec, M, N)
        rtol = 1e-10
        print(vec)
        print(tik_matrix)
        assert np.allclose(tik_matrix, np_matrix, rtol)



