# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
import numpy as np
import pytest
from mindspore import Tensor
from mindspore import ops
from tbe import tik

from flash_attention_test_agent import FlashAttentionTestAgent


class MatmulComputer(FlashAttentionTestAgent):
    def define_ios(self, *args, **kwargs):
        M, K, N, dtype = kwargs.get("M"), kwargs.get("K"), kwargs.get("N"), kwargs.get("dtype")
        self.dtype = dtype
        self.M = M
        self.K = K
        self.k_real = kwargs.get("k_real")
        self.N = N
        self.A_gm = self.tik_instance.Tensor(dtype, (M, K), name="A_gm", scope=tik.scope_gm)
        self.B_gm = self.tik_instance.Tensor(dtype, (K, N), name="B_gm", scope=tik.scope_gm)
        self.C_gm = self.tik_instance.Tensor(dtype, (M, N), name="C_gm", scope=tik.scope_gm)

    def compute(self, *args, **kwargs):
        M_alig = (self.M + 16 - 1) // 16 * 16
        K_alig = (self.K + 16 - 1) // 16 * 16
        N_alig = (self.N + 16 - 1) // 16 * 16
        A_l1 = self.tik_instance.Tensor(
            self.dtype, (M_alig, K_alig), name="A_l1", scope=tik.scope_cbuf
        )
        B_l1 = self.tik_instance.Tensor(
            self.dtype, (K_alig, N_alig), name="B_l1", scope=tik.scope_cbuf
        )
        self.fa.tik_instance.data_move(A_l1, self.A_gm[0], 0, 1, self.M * self.K // 16, 0, 0)
        self.fa.tik_instance.data_move(B_l1, self.B_gm[0], 0, 1, self.K * self.N // 16, 0, 0)
        A_l1 = self.fa.tik_ops_utils.MK_TO_K1MK0(A_l1)
        B_l1 = self.fa.tik_ops_utils.KN_TO_K1NK0(B_l1)
        C_ub = self.fa.tik_ops_utils.matmul_compute(A_l1, B_l1, self.M, self.K, self.N)
        self.fa.tik_instance.data_move(self.C_gm[0], C_ub, 0, 1, self.M * self.N // 16, 0, 0)
        self.tik_instance.BuildCCE(
            kernel_name="mock_flash_attention", inputs=[self.A_gm, self.B_gm], outputs=[self.C_gm],
            config={"dump_cce_code": False, "save_temp_cce_file": True, "enable_const_fold": True},
        )
        return self.tik_instance

    def compute_mmad(self, *args, **kwargs):
        M_alig = (self.M + 16 - 1) // 16 * 16
        K_alig = (self.K + 16 - 1) // 16 * 16
        N_alig = (self.N + 16 - 1) // 16 * 16
        A_l1 = self.tik_instance.Tensor(
            self.dtype, (M_alig, K_alig), name="A_l1", scope=tik.scope_cbuf
        )
        B_l1 = self.tik_instance.Tensor(
            self.dtype, (K_alig, N_alig), name="B_l1", scope=tik.scope_cbuf
        )

        self.fa.tik_instance.data_move(A_l1, self.A_gm[0], 0, 1, self.M * self.K // 16, 0, 0)
        self.fa.tik_instance.data_move(B_l1, self.B_gm[0], 0, 1, self.K * self.N // 16, 0, 0)
        A_l1 = self.fa.tik_ops_utils.MK_TO_K1MK0(A_l1)
        B_l1 = self.fa.tik_ops_utils.MK_TO_K1MK0(B_l1)
        C_ub = self.fa.tik_ops_utils.mmad_compute(A_l1, B_l1, self.M, self.k_real, self.N, N1MN0_to_MN=False)
        self.fa.tik_instance.data_move(self.C_gm[0], C_ub, 0, 1, self.M * self.N // 16, 0, 0)
        self.tik_instance.BuildCCE(
            kernel_name="mock_flash_attention", inputs=[self.A_gm, self.B_gm], outputs=[self.C_gm],
            config={"dump_cce_code": False, "save_temp_cce_file": True, "enable_const_fold": True},
        )
        return self.tik_instance

    def compute_mmad_transpose(self, a_transpose=False, b_transpose=False):
        M_alig = (self.M + 16 - 1) // 16 * 16
        K_alig = (self.K + 16 - 1) // 16 * 16
        N_alig = (self.N + 16 - 1) // 16 * 16
        a_transpose = a_transpose
        b_transpose = b_transpose
        A_l1 = self.tik_instance.Tensor(
            self.dtype, (M_alig, K_alig), name="A_l1", scope=tik.scope_cbuf
        )
        B_l1 = self.tik_instance.Tensor(
            self.dtype, (K_alig, N_alig), name="B_l1", scope=tik.scope_cbuf
        )
        self.fa.tik_instance.data_move(A_l1, self.A_gm[0], 0, 1, self.M * self.K // 16, 0, 0)
        self.fa.tik_instance.data_move(B_l1, self.B_gm[0], 0, 1, self.K * self.N // 16, 0, 0)
        C_ub = self.fa.tik_ops_utils.mmad_compute_transpose(A_l1, B_l1, self.M, self.K, self.N, N1MN0_to_MN=False,
                                                  a_transpose=a_transpose, b_transpose=b_transpose)

        self.fa.tik_instance.data_move(self.C_gm[0], C_ub, 0, 1, self.M * self.N // 16, 0, 0)
        self.tik_instance.BuildCCE(
            kernel_name="mock_flash_attention", inputs=[self.A_gm, self.B_gm], outputs=[self.C_gm],
        )
        return self.tik_instance


class TestMatmulCompute:
    @pytest.mark.parametrize(
        "M, K, N, dtype",
        [
            # (64, 40, 64, "float16"),  # 需要在没对齐的维度pad
            # (128, 40, 64, "float16"),
            # (64, 40, 128, "float16"),
            (64, 64, 64, "float16"),
            (128, 64, 64, "float16"),
            (64, 64, 128, "float16"),
            (1, 128, 128, "float16"),  # matmul 能处理m=1
        ],
    )
    def test_matmul_compute(self, M, K, N, dtype):
        mfa = MatmulComputer(**dict(dtype=dtype, M=M, K=K, N=N))
        tik_instance = mfa.compute()

        A = np.random.random([M, K]).astype(dtype)
        B = np.random.random([K, N]).astype(dtype)
        feed_dict = {"A_gm": A, "B_gm": B}
        (tik_out,) = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)
        ms_out = ops.matmul(Tensor(A), Tensor(B))
        print("\n############## custom op result #################")
        print(tik_out)
        print("\n############## ms op result #################")
        print(ms_out)
        rtol = 1e-10
        assert np.allclose(tik_out, ms_out.asnumpy(), rtol)

    @pytest.mark.parametrize(
        "M, K, N, dtype",
        [
            (64, 64, 64, "float16"),
            (128, 64, 64, "float16"),
            (64, 288, 128, "float16"),
            (64, 15, 128, "float16"),  # K=15 非对齐
            (64, 32, 128, "float16"),
            (64, 40, 128, "float16"),  # K // 16 % 2 > 0
            (64, 48, 128, "float16"),
            (64, 60, 128, "float16"),
        ],
    )
    def test_matmul_compute_mmad(self, M, K, N, dtype):
        k_real = K
        if K % 16 != 0:
            K = (K + 15) // 16 * 16
        mfa = MatmulComputer(**dict(dtype=dtype, M=M, K=K, N=N, k_real=k_real))
        tik_instance = mfa.compute_mmad()
        A = np.random.random([M, K]).astype(dtype)
        B = np.random.random([K, N]).astype(dtype)
        ms_out = ops.matmul(Tensor(A[:, :k_real]), Tensor(B[:k_real, :]))
        print("\n############## ms op result #################")
        print(ms_out)

        # mmad Nz input
        # A = A.reshape(M, K // 16, 16)
        # A = A.transpose(1, 0, 2).reshape(M, K)
        # B = B.reshape(K, N // 16, 16)
        # B = B.transpose(1, 0, 2).reshape(K, N)
        feed_dict = {"A_gm": A, "B_gm": B}
        (tik_out,) = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)
        tik_out = tik_out.reshape(N // 16, M, 16)  # (M, N) -> (N1, M, N0)
        tik_out = tik_out.transpose(1, 0, 2).reshape(M, N)
        print("\n############## custom op result #################")
        print(tik_out)
        rtol = 1e-10
        assert np.allclose(tik_out, ms_out.asnumpy(), rtol)

    @pytest.mark.parametrize(
        "K, M, N, dtype",
        [
            (64, 64, 64, "float16"),
            (64, 128, 64, "float16"),
            (64, 64, 128, "float16"),
            (32, 64, 128, "float16"),
        ],
    )
    def test_matmul_compute_mmad_transposeA(self, K, M, N, dtype):
        mfa = MatmulComputer(**dict(dtype=dtype, M=M, K=K, N=N, a_transpose=True))
        tik_instance = mfa.compute_mmad_transpose(a_transpose=True, b_transpose=False)

        A = np.random.random([K, M]).astype(dtype)
        B = np.random.random([K, N]).astype(dtype)
        ms_out = ops.matmul(Tensor(A.T), Tensor(B))
        print("\n############## ms op result #################")
        print(ms_out)

        # mmad Nz input
        A = A.reshape(K, M // 16, 16)
        A = A.transpose(1, 0, 2).reshape(M, K)
        B = B.reshape(K, N // 16, 16)
        B = B.transpose(1, 0, 2).reshape(K, N)
        feed_dict = {"A_gm": A, "B_gm": B}
        (tik_out,) = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)
        tik_out = tik_out.reshape(N // 16, M, 16)  # (M, N) -> (N1, M, N0)
        tik_out = tik_out.transpose(1, 0, 2).reshape(M, N)
        print("\n############## custom op result #################")
        print(tik_out)

        rtol = 1e-10
        assert np.allclose(tik_out, ms_out.asnumpy(), rtol)

    @pytest.mark.parametrize(
        "M, K, N, dtype",
        [
            (64, 64, 64, "float16"),
            (128, 64, 64, "float16"),
            (64, 64, 128, "float16"),
            (64, 32, 128, "float16"),
        ],
    )
    def test_matmul_compute_mmad_transposeB(self, M, K, N, dtype):
        mfa = MatmulComputer(**dict(dtype=dtype, M=M, K=K, N=N))
        tik_instance = mfa.compute_mmad_transpose(a_transpose=False, b_transpose=True)

        A = np.random.random([M, K]).astype(dtype)
        B = np.random.random([N, K]).astype(dtype)
        ms_out = ops.matmul(Tensor(A), Tensor(B.T))
        print("\n############## ms op result #################")
        print(ms_out)

        # mmad Nz input
        A = A.reshape(M, K // 16, 16)
        A = A.transpose(1, 0, 2).reshape(M, K)
        B = B.reshape(N, K // 16, 16)
        B = B.transpose(1, 0, 2).reshape(K, N)
        feed_dict = {"A_gm": A, "B_gm": B}
        (tik_out,) = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)
        tik_out = tik_out.reshape(N // 16, M, 16)  # (M, N) -> (N1, M, N0)
        tik_out = tik_out.transpose(1, 0, 2).reshape(M, N)
        print("\n############## custom op result #################")
        print(tik_out)

        rtol = 1e-10
        assert np.allclose(tik_out, ms_out.asnumpy(), rtol)