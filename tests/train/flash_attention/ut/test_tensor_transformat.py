# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
import numpy as np
import pytest
from tbe import tik

from flash_attention_test_agent import FlashAttentionTestAgent


class MK_TO_K1MK0_TransFormatter(FlashAttentionTestAgent):
    def define_ios(self, *args, **kwargs):
        dtype = kwargs.get("dtype")
        self.dtype = dtype
        M, K = kwargs.get("M"), kwargs.get("K")
        self.input_MK_gm = self.tik_instance.Tensor(
            dtype=self.dtype, shape=(M, K), name="input_MK_gm", scope=tik.scope_gm
        )
        K1 = K // 16
        K0 = 16
        self.output_K1MK0_gm = self.tik_instance.Tensor(
            dtype=dtype, shape=(K1, M, K0), name="output_K1MK0_gm", scope=tik.scope_gm
        )
        self.K1MK0_workspace = self.tik_instance.Tensor(
            dtype=dtype, shape=(K1, M, K0), name="K1MK0_workspace", scope=tik.scope_ubuf
        )

    def compute(self, *args, **kwargs):
        inplace = kwargs.get("inplace")
        if inplace:
            self.fa.tik_ops_utils.MK_TO_K1MK0(self.input_MK_gm)
        else:
            self.fa.tik_ops_utils.MK_TO_K1MK0(self.input_MK_gm, workspace_tensor=self.K1MK0_workspace)

        K1, M, K0 = self.output_K1MK0_gm.shape
        ub = self.tik_instance.Tensor(
            dtype=self.dtype, shape=(K1, M, K0), name="ub", scope=tik.scope_ubuf
        )
        if inplace:
            self.tik_instance.data_move(ub, self.input_MK_gm, 0, 1, K1 * M, 0, 0)
        else:
            self.tik_instance.data_move(ub, self.K1MK0_workspace, 0, 1, K1 * M, 0, 0)

        self.tik_instance.data_move(self.output_K1MK0_gm, ub, 0, 1, K1 * M, 0, 0)

        self.tik_instance.BuildCCE(
            kernel_name="mock_flash_attention",
            inputs=[self.input_MK_gm],
            outputs=[self.output_K1MK0_gm],
        )

        return self.tik_instance


class KN_TO_K1NK0_TransFormatter(FlashAttentionTestAgent):
    def define_ios(self, *args, **kwargs):
        dtype = kwargs.get("dtype")
        self.dtype = dtype
        K, N = kwargs.get("K"), kwargs.get("N")
        self.input_KN_gm = self.tik_instance.Tensor(
            dtype=dtype, shape=(K, N), name="input_KN_gm", scope=tik.scope_gm
        )
        K1 = K // 16
        K0 = 16
        self.output_K1NK0_gm = self.tik_instance.Tensor(
            dtype=dtype, shape=(K1, N, K0), name="output_K1NK0_gm", scope=tik.scope_gm
        )
        self.K1NK0_workspace = self.tik_instance.Tensor(
            dtype=dtype,
            shape=(K1, N, K0),
            name="K1NK0_workspace",
            scope=tik.scope_gm,
            is_workspace=True,
        )

    def compute(self, *args, **kwargs):
        inplace = kwargs.get("inplace")
        if inplace:
            self.fa.tik_ops_utils.KN_TO_K1NK0(self.input_KN_gm)
        else:
            self.fa.tik_ops_utils.KN_TO_K1NK0(self.input_KN_gm, workspace_tensor=self.K1NK0_workspace)
        K1, N, K0 = self.output_K1NK0_gm.shape

        ub = self.tik_instance.Tensor(
            dtype=self.dtype, shape=(K1, N, K0), name="ub", scope=tik.scope_ubuf
        )
        if inplace:
            self.tik_instance.data_move(ub, self.input_KN_gm, 0, 1, K1 * N, 0, 0)
        else:
            self.tik_instance.data_move(ub, self.K1NK0_workspace, 0, 1, K1 * N, 0, 0)

        self.tik_instance.data_move(self.output_K1NK0_gm, ub, 0, 1, K1 * N, 0, 0)

        self.tik_instance.BuildCCE(
            kernel_name="mock_flash_attention",
            inputs=[self.input_KN_gm],
            outputs=[self.output_K1NK0_gm],
        )

        return self.tik_instance


class N1MN0_TO_MN_TransFormatter(FlashAttentionTestAgent):
    def define_ios(self, *args, **kwargs):
        dtype = kwargs.get("dtype")
        K, N, M = kwargs.get("K"), kwargs.get("N"), kwargs.get("M")
        self.dtype = dtype
        self.input_KN_gm = self.tik_instance.Tensor(
            dtype=dtype, shape=(K, N), name="input_KN_gm", scope=tik.scope_gm
        )
        K0 = 16
        N1 = N // K0
        N0 = 16
        self.input_N1MN0 = self.tik_instance.Tensor(
            dtype=dtype, shape=(N1, M, N0), name="input_N1MN0", scope=tik.scope_gm
        )
        self.output_MN_gm = self.tik_instance.Tensor(
            dtype=dtype, shape=(M, N), name="output_MN_gm", scope=tik.scope_gm
        )

    def compute(self, *args, **kwargs):
        self.fa.tik_ops_utils.N1MN0_TO_MN(self.input_N1MN0)
        M, N = self.output_MN_gm.shape
        N1, M, N0 = self.input_N1MN0.shape

        ub = self.tik_instance.Tensor(
            dtype=self.dtype, shape=(M, N), name="ub", scope=tik.scope_ubuf
        )
        self.tik_instance.data_move(ub, self.input_N1MN0, 0, 1, N1 * M, 0, 0)
        self.tik_instance.data_move(self.output_MN_gm, ub, 0, 1, N1 * M, 0, 0)

        self.tik_instance.BuildCCE(
            kernel_name="mock_flash_attention",
            inputs=[self.input_N1MN0],
            outputs=[self.output_MN_gm],
        )

        return self.tik_instance


class TestTensorTransFormat:
    # @pytest.mark.skip(reason="Not support inplace operation")
    @pytest.mark.parametrize(
        "M, K, dtype",
        [
            (16, 16, "int16"),
            (16, 32, "int16"),
            (16, 64, "int16"),
            (16, 128, "int16"),
            (32, 16, "int16"),
            (32, 32, "int16"),
            (32, 64, "int16"),
            (32, 128, "int16"),
            (16, 16, "float16"),
            (16, 32, "float16"),
            (16, 64, "float16"),
            (16, 128, "float16"),
            (32, 16, "float16"),
            (32, 32, "float16"),
            (32, 64, "float16"),
            (32, 128, "float16"),
        ],
    )
    def test_MK_TO_K1MK0_inplace(self, M, K, dtype):
        K0 = 16
        K1 = K // 16
        mfa = MK_TO_K1MK0_TransFormatter(**dict(dtype=dtype, M=M, K=K))
        mfa.input_MK_gm = mfa.tik_instance.Tensor(
            dtype=dtype, shape=(M, K), name="input_MK_gm", scope=tik.scope_gm
        )
        k_1 = K // 16
        k_ = 16
        mfa.output_K1MK0_gm = mfa.tik_instance.Tensor(
            dtype=dtype, shape=(k_1, M, k_), name="output_K1MK0_gm", scope=tik.scope_gm
        )
        mfa.K1MK0_workspace = mfa.tik_instance.Tensor(
            dtype=dtype, shape=(k_1, M, k_), name="K1MK0_workspace", scope=tik.scope_ubuf
        )
        tik_instance = mfa.compute(**dict(inplace=True))
        mk_input = np.arange(np.prod((M, K))).reshape((M, K)).astype(dtype)
        np_res = np.stack(np.hsplit(mk_input, K1), axis=0)

        print(f"original mk_input=\n{mk_input}")

        feed_dict = {"input_MK_gm": mk_input}
        (tik_output,) = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)
        modified_mk_input = mk_input.reshape(K1, M, K0)

        print(f"after tik modified_mk_input=\n{modified_mk_input}")
        print(f"mk_input.shape={mk_input.shape}")
        print(f"np_res=\n{np_res}")
        print(f"np_res.shape={np_res.shape}")

        assert np.allclose(np_res, tik_output)
        assert np.allclose(np_res, modified_mk_input)

    @pytest.mark.parametrize(
        "M, K, dtype",
        [
            (16, 16, "int16"),
            (16, 32, "int16"),
            (16, 64, "int16"),
            (16, 128, "int16"),
            (32, 16, "int16"),
            (32, 32, "int16"),
            (32, 64, "int16"),
            (32, 128, "int16"),
            (16, 16, "float16"),
            (16, 32, "float16"),
            (16, 64, "float16"),
            (16, 128, "float16"),
            (32, 16, "float16"),
            (32, 32, "float16"),
            (32, 64, "float16"),
            (32, 128, "float16"),
        ],
    )
    def test_MK_TO_K1MK0_in_workspace(self, M, K, dtype):
        K1 = K // 16
        mfa = MK_TO_K1MK0_TransFormatter(**dict(dtype=dtype, M=M, K=K))
        tik_instance = mfa.compute(**dict(inplace=False))
        mk_input = np.arange(np.prod((M, K))).reshape((M, K)).astype(dtype)
        kn_input_backup = mk_input.copy()
        np_res = np.stack(np.hsplit(kn_input_backup, K1), axis=0)

        print(f"original mk_input=\n{mk_input}")

        feed_dict = {"input_MK_gm": mk_input}
        (tik_output,) = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)

        print(f"mk_input.shape={mk_input.shape}")
        print(f"np_res=\n{np_res}")
        print(f"tik_output=\n{tik_output}")
        print(f"np_res.shape={np_res.shape}")

        assert np.allclose(np_res, tik_output)
        assert np.allclose(kn_input_backup, mk_input)

    # @pytest.mark.skip(reason="Not support inplace operation")
    @pytest.mark.parametrize(
        "K, N, dtype",
        [
            (16, 16, "int16"),
            (16, 32, "int16"),
            (16, 64, "int16"),
            (16, 128, "int16"),
            (32, 16, "int16"),
            (32, 32, "int16"),
            (32, 64, "int16"),
            (32, 128, "int16"),
            (16, 16, "float16"),
            (16, 32, "float16"),
            (16, 64, "float16"),
            (16, 128, "float16"),
            (32, 16, "float16"),
            (32, 32, "float16"),
            (32, 64, "float16"),
            (32, 128, "float16"),
        ],
    )
    def test_KN_TO_K1NK0_inplace(self, K, N, dtype):
        K0 = 16
        K1 = K // 16
        mfa = KN_TO_K1NK0_TransFormatter(**dict(dtype=dtype, K=K, N=N))
        tik_instance = mfa.compute(**dict(inplace=True))
        kn_input = np.arange(np.prod((K, N))).reshape((K, N)).astype(dtype)
        np_res = kn_input.copy().reshape(K1, K0, N).swapaxes(1, 2)

        print(f"original kn_input=\n{kn_input}")

        feed_dict = {"input_KN_gm": kn_input}
        (tik_output,) = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)
        modified_kn_input = kn_input.reshape(K1, N, K0)

        print(f"after tik modified_kn_input=\n{modified_kn_input}")
        print(f"kn_input.shape={kn_input.shape}")
        print(f"np_res=\n{np_res}")
        print(f"np_res.shape={np_res.shape}")

        assert np.allclose(np_res, tik_output)
        assert np.allclose(np_res, modified_kn_input)

    @pytest.mark.parametrize(
        "K, N, dtype",
        [
            (16, 16, "int16"),
            (16, 32, "int16"),
            (16, 64, "int16"),
            (16, 128, "int16"),
            (32, 16, "int16"),
            (32, 32, "int16"),
            (32, 64, "int16"),
            (32, 128, "int16"),
            (16, 16, "float16"),
            (16, 32, "float16"),
            (16, 64, "float16"),
            (16, 128, "float16"),
            (32, 16, "float16"),
            (32, 32, "float16"),
            (32, 64, "float16"),
            (32, 128, "float16"),
        ],
    )
    def test_KN_TO_K1NK0_in_workspace(self, K, N, dtype):
        K0 = 16
        K1 = K // 16
        mfa = KN_TO_K1NK0_TransFormatter(**dict(dtype=dtype, K=K, N=N))
        tik_instance = mfa.compute(**dict(inplace=False))
        kn_input = np.arange(np.prod((K, N))).reshape((K, N)).astype(dtype)
        kn_input_backup = kn_input.copy()
        np_res = kn_input.copy().reshape(K1, K0, N).swapaxes(1, 2)

        print(f"original kn_input=\n{kn_input}")

        feed_dict = {"input_KN_gm": kn_input}
        (tik_output,) = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)

        print(f"kn_input.shape={kn_input.shape}")
        print(f"np_res=\n{np_res}")
        print(f"np_res.shape={np_res.shape}")

        assert np.allclose(np_res, tik_output)
        assert np.allclose(kn_input_backup, kn_input)

    @pytest.mark.parametrize(
        "N1, M, dtype",
        [
            (1, 16, "int16"),
            (1, 32, "int16"),
            (1, 64, "int16"),
            (1, 128, "int16"),
            (2, 16, "int16"),
            (2, 32, "int16"),
            (2, 64, "int16"),
            (2, 128, "int16"),
            (1, 16, "float16"),
            (1, 32, "float16"),
            (1, 64, "float16"),
            (1, 128, "float16"),
            (2, 16, "float16"),
            (2, 32, "float16"),
            (2, 64, "float16"),
            (2, 128, "float16"),
        ],
    )
    def test_N1MN0_TO_MN_inplace(self, N1, M, dtype):
        mfa = N1MN0_TO_MN_TransFormatter(**dict(M=M, K=16, N=16 * N1, dtype=dtype))
        tik_instance = mfa.compute()
        N1MN0_input = np.arange(np.prod((N1, M, 16))).reshape((N1, M, 16)).astype(dtype)
        np_res = np.concatenate(list(map(np.squeeze, np.split(N1MN0_input, N1))), axis=1)
        print(f"np_res=\n{np_res}")
        print(f"np_res.shape={np_res.shape}")

        print(f"original N1MN0_input=\n{N1MN0_input}")

        feed_dict = {"input_N1MN0": N1MN0_input}
        (tik_output,) = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)
        modified_N1MN0_input = N1MN0_input.reshape(M, 16 * N1)

        print(f"after tik modified_N1MN0_input=\n{modified_N1MN0_input}")
        print(f"N1MN0_input.shape={N1MN0_input.shape}")

        assert np.allclose(np_res, tik_output)
        assert np.allclose(np_res, modified_N1MN0_input)
