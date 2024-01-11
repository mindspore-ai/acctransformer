# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""test triangle attention accuracy"""
import numpy as np
from mindspore import nn
import mindspore.ops as P
from mindspore import Tensor
from mindspore import dtype as mstype

from acctransformer.triangle_attention.triangle_attention import TriangleAttention


class SelfAttention(nn.Cell):
    """
    A Self Attention
    """

    def __init__(self):
        super(SelfAttention, self).__init__()
        self.qk_bmm = P.BatchMatMul(transpose_b=True)
        self.mul = P.Mul()
        self.expand_dims = P.ExpandDims()
        self.add = P.Add()
        self.softmax = P.Softmax()
        self.pv_bmm = P.BatchMatMul()
        self.multiply_data = Tensor([-10000.0], dtype=mstype.float16)


    def construct(self, q, k, v, attn_mask):
        """Forward process"""
        # q * k.T
        sim = self.qk_bmm(q, k)
        # softmax
        adder = self.mul(attn_mask, self.multiply_data)
        adder = self.expand_dims(adder, 1)
        sim = self.add(adder, sim)
        probs = self.softmax(sim)
        # probs * v
        out = self.pv_bmm(probs, v)
        return out


def data_compare(ground_truth, predict, diff_thd=0.001, pct_thd=0.001, max_diff_thd=0.1):
    """compare ground_truth and predict"""
    total_count = np.prod(ground_truth.shape)
    greater_than_diff_thd_count = np.sum(
        np.abs(predict.astype("float32") -
               ground_truth.astype("float32")) > diff_thd *
        (np.abs(ground_truth.astype("float32")) + 1e-9))
    greater_than_max_diff_thd_count = np.sum(
        np.abs(predict.astype("float32") -
               ground_truth.astype("float32")) > max_diff_thd *
        (np.abs(ground_truth.astype("float32")) + 1e-9))

    diff_gt_thd_proportion = greater_than_diff_thd_count / total_count
    diff_gt_max_thd_proportion = greater_than_max_diff_thd_count / total_count
    if diff_gt_thd_proportion > pct_thd or diff_gt_max_thd_proportion > 0:
        return False
    return True

def test_tri_attn_should_be_equal_with_self_attn_given_seq_len_divided_by_block_size():
    """
    Feature: Test Triangle Attention
    Description: Test Triangle Attention Accuracy
    Expectation: result == True
    """
    input_shape = (4, 32, 4096, 128)
    block_size = 1024
    q = Tensor(np.random.random(input_shape).astype("float16"))
    k = Tensor(np.random.random(input_shape).astype("float16"))
    v = Tensor(np.random.random(input_shape).astype("float16"))
    batch_size, seq_len = q.shape[0], q.shape[2]
    att_mask = Tensor(np.triu(
        np.ones(shape=(batch_size, seq_len, seq_len)), k=1), dtype=mstype.float16)

    model1 = TriangleAttention(block_size)
    model2 = SelfAttention()

    out1 = model1(q, k, v, att_mask).asnumpy()
    out2 = model2(q, k, v, att_mask).asnumpy()

    result = data_compare(out1, out2)

    assert result

def test_tri_attn_should_be_equal_with_self_attn_given_seq_len_undivided_by_block_size():
    """
    Feature: Test Triangle Attention
    Description: Test Triangle Attention Accuracy
    Expectation: result == True
    """
    input_shape = (4, 32, 4608, 128)
    block_size = 1024
    q = Tensor(np.random.random(input_shape).astype("float16"))
    k = Tensor(np.random.random(input_shape).astype("float16"))
    v = Tensor(np.random.random(input_shape).astype("float16"))
    batch_size, seq_len = q.shape[0], q.shape[2]
    att_mask = Tensor(np.triu(
        np.ones(shape=(batch_size, seq_len, seq_len)), k=1), dtype=mstype.float16)

    model1 = TriangleAttention(block_size)
    model2 = SelfAttention()

    out1 = model1(q, k, v, att_mask).asnumpy()
    out2 = model2(q, k, v, att_mask).asnumpy()

    result = data_compare(out1, out2)

    assert result
