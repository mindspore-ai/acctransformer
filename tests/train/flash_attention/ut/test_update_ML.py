# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
import numpy as np
import pytest
from tbe import tik

from flash_attention_test_agent import FlashAttentionTestAgent


class MLUpdater(FlashAttentionTestAgent):
    def define_ios(self, *args, **kwargs):
        block_h, dtype = kwargs.get("M"), kwargs.get("dtype")
        # mi_old_ub, mij_ub, li_old_ub, lij_ub, vec_len
        self.dtype = dtype
        self.block_h = block_h
        self.input_mi = self.tik_instance.Tensor(
            dtype, (block_h,), name="input_mi", scope=tik.scope_gm
        )
        self.input_mij = self.tik_instance.Tensor(
            dtype, (block_h,), name="input_mij", scope=tik.scope_gm
        )
        self.input_li = self.tik_instance.Tensor(
            dtype, (block_h,), name="input_li", scope=tik.scope_gm
        )
        self.input_lij = self.tik_instance.Tensor(
            dtype, (block_h,), name="input_lij", scope=tik.scope_gm
        )

        self.output_mi_new = self.tik_instance.Tensor(
            dtype, (block_h,), name="output_mi_new", scope=tik.scope_gm
        )
        self.output_li_new = self.tik_instance.Tensor(
            dtype, (block_h,), name="output_li_new", scope=tik.scope_gm
        )

    def compute(self, *args, **kwargs):
        mi_old_ub = self.tik_instance.Tensor(
            self.dtype, (self.block_h,), name="mi_old_ub", scope=tik.scope_ubuf
        )
        li_old_ub = self.tik_instance.Tensor(
            self.dtype, (self.block_h,), name="li_old_ub", scope=tik.scope_ubuf
        )
        mij_ub = self.tik_instance.Tensor(
            self.dtype, (self.block_h,), name="mi_cur_ub", scope=tik.scope_ubuf
        )
        lij_ub = self.tik_instance.Tensor(
            self.dtype, (self.block_h,), name="li_cur_ub", scope=tik.scope_ubuf
        )
        self.tik_instance.h_data_move(mi_old_ub, self.input_mi)
        self.tik_instance.h_data_move(li_old_ub, self.input_li)
        self.tik_instance.h_data_move(mij_ub, self.input_mij)
        self.tik_instance.h_data_move(lij_ub, self.input_lij)
        #  mi_old_ub, mij_ub, li_old_ub, lij_ub, vec_len
        mi_new_ub, li_new_ub = self.fa.update_m_l(
            mi_old_ub, mij_ub, li_old_ub, lij_ub, self.block_h
        )
        self.tik_instance.h_data_move(self.output_mi_new, mi_new_ub)
        self.tik_instance.h_data_move(self.output_li_new, li_new_ub)
        self.tik_instance.BuildCCE(
            kernel_name="mock_flash_attention",
            inputs=[self.input_mi, self.input_mij, self.input_li, self.input_lij],
            outputs=[self.output_li_new, self.output_mi_new],
        )
        return self.tik_instance


class TestUpdateML:
    @staticmethod
    def py_update_l_m(mi, mij, li, lij):
        mi_new = np.maximum(mij, mi)
        li_new = np.exp(mi - mi_new) * li + np.exp(mij - mi_new) * lij
        return mi_new, li_new

    @pytest.mark.parametrize(
        "M, dtype", [(32, "float16"), (128, "float16"), (256, "float16")],
    )
    def test_update_l_m(self, M, dtype):
        mfa = MLUpdater(**dict(dtype=dtype, M=M))
        tik_instance = mfa.compute()
        input_mi = np.random.random([M]).astype(dtype)
        input_mij = np.random.random([M]).astype(dtype)
        input_li = np.random.random([M]).astype(dtype)
        input_lij = np.random.random([M]).astype(dtype)
        np_mi_new, np_li_new = self.py_update_l_m(input_mi, input_mij, input_li, input_lij)

        feed_dict = {
            "input_mi": input_mi,
            "input_mij": input_mij,
            "input_li": input_li,
            "input_lij": input_lij,
        }
        tik_l_new, tik_m_new = tik_instance.tikdb.start_debug(
            feed_dict=feed_dict, interactive=False
        )
        assert np.allclose(np_mi_new, tik_m_new, 1e-8)
        assert np.allclose(np_li_new, tik_l_new, 1e-8)


