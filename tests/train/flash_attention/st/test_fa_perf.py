# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import numpy as np

from mindspore import Tensor
from mindspore import Profiler

from common import set_env
from common import FlashAttention
from common import FlashAttentionGrad
from common import DropGenMask
from common import np_impl_sa_fwd

set_env()


def test_fa_fwd_perf():
    q_shape = (4, 8, 4096, 128)
    kv_shape = q_shape
    q = np.random.random(q_shape).astype("float16")
    k = np.random.random(kv_shape).astype("float16")
    v = np.random.random(kv_shape).astype("float16")
    batch_size, head_num, q_seq_len, k_seq_len = q_shape[0], q_shape[1], q_shape[2], kv_shape[2]

    att_mask = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)
    gen_drop_mask = DropGenMask(shape=(batch_size, head_num, q_seq_len, k_seq_len), dropout_rate=0.1)
    drop_mask = gen_drop_mask()

    model = FlashAttention()
    profiler = Profiler(output_path="./prof_data/fa_fwd", profile_communication=True)
    model(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask), drop_mask)
    profiler.analyse()


def test_fa_grad_perf():
    q_shape = (4, 8, 4096, 128)
    kv_shape = q_shape
    q = np.random.random(q_shape).astype("float16")
    k = np.random.random(kv_shape).astype("float16")
    v = np.random.random(kv_shape).astype("float16")
    batch_size, head_num, q_seq_len, k_seq_len = q_shape[0], q_shape[1], q_shape[2], kv_shape[2]

    att_mask = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)
    gen_drop_mask = DropGenMask(shape=(batch_size, head_num, q_seq_len, k_seq_len), dropout_rate=0.1)
    drop_mask = gen_drop_mask()
    # cpu fp16 大shape计算太慢，性能测试直接构造random数据作为输入
    # O, mid_results = np_impl_sa_fwd(q, k, v, att_mask, drop_mask.asnumpy())
    # _, P, row_sum, row_max = mid_results
    # l = row_max.astype("float32") + np.log(row_sum)
    # l = np.squeeze(l, axis=-1)
    O = np.random.randn(*q_shape).astype("float16")
    l = np.random.randn(*q_shape[:-1]).astype("float32")
    dO = np.random.randn(*q_shape).astype("float16")
    douts = (Tensor(dO), None)

    model = FlashAttentionGrad()
    profiler = Profiler(output_path="./prof_data/fa_grad", profile_communication=True)
    model(Tensor(q), Tensor(k), Tensor(v), Tensor(att_mask), Tensor(l), Tensor(O), douts, drop_mask)
    profiler.analyse()
