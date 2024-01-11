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
"""
A Triangle Attention Layer.
"""
from __future__ import absolute_import

import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor
from mindspore import dtype as mstype

__all__ = ["TriangleAttention"]


class TriangleAttention(nn.Cell):
    """Triangle Attention Layer.

    This function contains the triangle attention primitives.
    The triangle attention divides the q, k and v into blocks to eliminate invalid calculations and invalid data in the
    mask part of attention score.

    Specifically, it includes the following:

    1. An interface for calling triangle operation.
    2. A configuration parameter for adjusting block size.

    Args:
        block_size(int): An integer determining the block size.
            Default 512.
        dropout_rate(float): The dropout rate of the attention score.
            Default 0.0.
        dp(int): data parallel.
            Default 1.
        mp(int): model parallel.
            Default 1.


    Inputs:
      - **query** (Tensor) - Tensor query (:class:`mstype.fp16` [batch_size, head_num, seq_length, head_dim])
      - **key** (Tensor) - Tensor key (:class:`mstype.fp16` [batch_size, head_num, seq_length, head_dim])
      - **value** (Tensor) - Tensor value (:class:`mstype.fp16` [batch_size, head_num, seq_length, head_dim])
      - **attention_mask** (Tensor) - Float Tensor the mask of (:class:`mstype.fp16` [batch_size, seq_length,
          seq_length]): A matrix to pass masked information.

    Outputs:
        A Tensor. The output of the attention with shape [batch_size, head_num, seq_length, head_dim]

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> from acctransformer.triangle_attention.triangle_attention import TriangleAttention
        >>> model = TriangleAttention(block_size=1024)
        >>> query = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> key = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> value = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> attention_mask = Tensor(np.triu(np.ones((2, 4096, 4096)), k=1), mstype.float16)
        >>> output = model(query, key, value, attention_mask)
        >>> print(output.shape)
        (2, 16, 4096, 128)
    """

    def __init__(self, block_size=512, dropout_rate=0., dp=1, mp=1):
        super(TriangleAttention, self).__init__()
        if block_size <= 0:
            raise ValueError(f"block size must be positive integer, but got {block_size}")
        if dropout_rate < 0. or dropout_rate >= 1.:
            raise ValueError(f"dropout rate must be between 0 and 1, but got {dropout_rate}")
        if dp <= 0:
            raise ValueError(f"data parallel must be positive integer, but got {dp}")
        if mp <= 0:
            raise ValueError(f"model parallel must be positive integer, but got {mp}")
        self.block_size = block_size
        self.qkv_slice = P.StridedSlice().shard(((dp, mp, 1, 1),))
        self.attn_mask_slice = P.StridedSlice().shard(((dp, 1, 1),))
        self.qk_bmm = P.BatchMatMul(transpose_b=True).shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.attn_mask_mul = P.Mul().shard(((dp, 1, 1), (1,)))
        self.attn_mask_expend_dims = P.ExpandDims().shard(((dp, 1, 1),))
        self.attn_mask_add = P.Add().shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
        self.softmax = P.Softmax().shard(((dp, mp, 1, 1),))
        self.enable_dropout = dropout_rate > 0.
        if self.enable_dropout:
            self.attn_dropout = nn.Dropout(keep_prob=1 - dropout_rate)
            self.attn_dropout.dropout.shard(((dp, mp, 1, 1),))
        self.pv_bmm = P.BatchMatMul().shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.o_concat = P.Concat(axis=2).shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.multiply_data = Tensor([-10000.,], dtype=mstype.float16)

    def _compute_attn(self, q, k, v, attention_mask, output):
        """compute attention of each q, k and v block"""
        # q * k.T
        sim = self.qk_bmm(q, k)
        # softmax
        adder = self.attn_mask_mul(attention_mask, self.multiply_data)
        adder = self.attn_mask_expend_dims(adder, 1)
        sim = self.attn_mask_add(adder, sim)
        probs = self.softmax(sim)
        # dropout
        if self.enable_dropout:
            probs = self.attn_dropout(probs)
        # p * v
        o = self.pv_bmm(probs, v)
        # concat result
        if output is None:
            output = o
        else:
            output = self.o_concat((output, o))
        return output

    def _slice_tensor(self, q, k, v, attention_mask, slice_indexes):
        """slice q, k, v and attention_mask by slice_indexes"""
        bsz, head_num, _, head_dim = q.shape
        q_begin, q_end, kv_begin, kv_end = slice_indexes
        cur_q = self.qkv_slice(q, (0, 0, q_begin, 0), (bsz, head_num, q_end, head_dim), (1, 1, 1, 1))
        cur_k = self.qkv_slice(k, (0, 0, kv_begin, 0), (bsz, head_num, kv_end, head_dim), (1, 1, 1, 1))
        cur_v = self.qkv_slice(v, (0, 0, kv_begin, 0), (bsz, head_num, kv_end, head_dim), (1, 1, 1, 1))
        cur_attn_mask = self.attn_mask_slice(attention_mask, (0, q_begin, kv_begin), (bsz, q_end, kv_end),
                                             (1, 1, 1))
        return cur_q, cur_k, cur_v, cur_attn_mask

    def construct(self, query, key, value, attention_mask):
        """Forward process"""
        seq_length = query.shape[2]
        groups, tail = divmod(seq_length, self.block_size)

        kv_begin = 0
        output = None
        for i in range(groups):
            q_begin = i * self.block_size
            q_end = (i + 1) * self.block_size
            kv_end = q_end
            # slice
            block_tensors = self._slice_tensor(query, key, value, attention_mask, (q_begin, q_end, kv_begin, kv_end))
            cur_q, cur_k, cur_v, cur_attn_mask = block_tensors[0], block_tensors[1], block_tensors[2], block_tensors[3]
            # compute attn
            output = self._compute_attn(cur_q, cur_k, cur_v, cur_attn_mask, output)

        if tail > 0:
            q_begin = groups * self.block_size
            q_end = seq_length
            kv_end = q_end
            # slice
            block_tensors = self._slice_tensor(query, key, value, attention_mask,
                                               (q_begin, q_end, kv_begin, kv_end))
            cur_q, cur_k, cur_v, cur_attn_mask = block_tensors[0], block_tensors[1], block_tensors[2], block_tensors[3]
            # compute attn
            output = self._compute_attn(cur_q, cur_k, cur_v, cur_attn_mask, output)

        return output
