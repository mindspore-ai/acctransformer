# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
import abc
import os
import shutil
from abc import abstractmethod

import numpy as np
from tbe.common.platform import set_current_compile_soc_info

from acctransformer.flash_attention.ops.flash_attention.tiling_strategy.sparse_tiling import SparseTiling

if os.path.exists("./rank_0"):
    shutil.rmtree("./rank_0", ignore_errors=True)
    shutil.rmtree("./kernel_meta", ignore_errors=True)
set_current_compile_soc_info("Ascend910")


class FlashAttentionTestAgent(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        q_ori_shape = {"ori_shape": (1, 1, 1024, 80)}
        q_shape = {"shape": (1, 1, 80 // 16, 1024 // 16, 16, 16)}

        k_ori_shape = {"ori_shape": (1, 1, 1024, 80)}
        k_shape = {"shape": (1, 1, 80 // 16, 1024 // 16, 16, 16)}

        v_ori_shape = {"ori_shape": (1, 1, 1024, 80)}
        v_shape = {"shape": (1, 1, 80 // 16, 1024 // 16, 16, 16)}

        att_mask_ori_shape = {"ori_shape": (1, 1024, 1024)}
        att_mask_shape = {"shape": (1, 1024 // 16, 1024 // 16, 16, 16)}

        drop_mask_shape = {"shape": (1, 1024, 1024)}

        fa = FlashAttentionFwd(query={**q_ori_shape, **q_shape}, key={**k_ori_shape, **k_shape},
                               value={**v_ori_shape, **v_shape}, attn_mask={**att_mask_shape, **att_mask_ori_shape},
                               dropout_mask=drop_mask_shape, alibi_mask=None, kernel_name='flash_attention_fwd',
                               tiling_stgy=SparseTiling, disable_debug=False)
        self.fa = fa
        self.tik_instance = fa.tik_instance

        self.define_ios(*args, **kwargs)

    @staticmethod
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

    @abstractmethod
    def define_ios(self, *args, **kwargs):
        raise NotImplemented

    @abstractmethod
    def compute(self, *args, **kwargs):
        raise NotImplemented
