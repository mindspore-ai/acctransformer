# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
import numpy as np
import pytest
from mindspore import Tensor
from mindspore import ops
from tbe import tik

from flash_attention_test_agent import FlashAttentionTestAgent


class RowSumCubeImplementedComputer(FlashAttentionTestAgent):
    def define_ios(self, *args, **kwargs):
        M, N, dtype = (
            kwargs.get("M"),
            kwargs.get("N"),
            kwargs.get("dtype"),
        )
        self.M = M
        self.N = N
        self.dtype = dtype
        self.matrix_gm = self.tik_instance.Tensor(
            dtype, (M, N), name="matrix_gm", scope=tik.scope_gm
        )
        self.row_sum_gm = self.tik_instance.Tensor(
            dtype, (M,), name="row_sum_gm", scope=tik.scope_gm
        )

    def compute(self, *args, **kwargs):
        matrix_ub = self.tik_instance.Tensor(
            self.dtype, (self.M, self.N), name="matrix_ub", scope=tik.scope_ubuf
        )
        self.tik_instance.data_move(matrix_ub, self.matrix_gm, 0, 1, self.M * self.N // 16, 0, 0)
        matrix_l1_K1MK0_ed = self.tik_instance.Tensor(
            self.dtype, (self.N // 16, self.M, 16), name="matrix_l1", scope=tik.scope_cbuf
        )
        matrix_l1_K1MK0_ed = self.fa.tik_ops_utils.MK_TO_K1MK0(matrix_ub, matrix_l1_K1MK0_ed)
        row_sum_ub = self.tik_instance.Tensor(
            self.dtype, (self.M,), name="row_sum_ub", scope=tik.scope_ubuf
        )
        ones_l1 = self.tik_instance.Tensor(self.dtype, (self.N, 16), name="ones_l1", scope=tik.scope_cbuf)
        ones_ub = self.tik_instance.Tensor(self.dtype, (self.N, 16), name="ones_ub", scope=tik.scope_ubuf)
        self.tik_instance.h_duplicate(ones_ub, 1.0)
        self.tik_instance.data_move(ones_l1, ones_ub, 0, 1, self.N, 0, 0)

        row_sum_ub = self.fa.tik_ops_utils.row_sum_cube_impl(matrix_l1_K1MK0_ed, ones_l1, row_sum_ub, self.M, self.N,
                                                             precision_type=self.dtype)
        self.tik_instance.data_move(self.row_sum_gm, row_sum_ub, 0, 1, self.M // 16, 0, 0)
        self.tik_instance.BuildCCE(
            kernel_name="mock_flash_attention", inputs=[self.matrix_gm], outputs=[self.row_sum_gm],
        )
        return self.tik_instance


class TestCubeRowSum:

    @staticmethod
    def ms_impl_matrix_row_sum(matrix):
        matrix = Tensor(matrix)
        row_sum = ops.reduce_sum(matrix, axis=-1)
        return row_sum

    @pytest.mark.parametrize(
        "M, N", [(64, 64), (128, 128), (256, 256), (512, 128), (128, 512)],
    )
    def test_cube_row_sum(self, M, N):
        dtype = "float16"
        mfa = RowSumCubeImplementedComputer(**dict(M=M, N=N, dtype=dtype))
        tik_instance = mfa.compute()
        matrix = np.random.random([M, N]).astype(dtype)
        ops_row_sum = self.ms_impl_matrix_row_sum(matrix)
        (tik_row_sum,) = tik_instance.tikdb.start_debug(
            feed_dict={"matrix_gm": matrix}, interactive=False
        )
        print("\n############## tik result #################")
        print(tik_row_sum)
        print("\n############## ms result #################")
        print(ops_row_sum)
        rtol = 1e-10
        assert np.allclose(tik_row_sum, ops_row_sum.asnumpy(), rtol)


