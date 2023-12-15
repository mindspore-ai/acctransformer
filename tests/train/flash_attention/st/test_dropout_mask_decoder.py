import functools
import os
import shutil

import mindspore as ms
import mindspore.nn as nn
import numpy as np
import pytest
from mindspore import Profiler
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops

from common import set_env
from flash_attention.ops.dropout_mask_decoder.dropout_mask_decoder_impl import get_dropout_mask_decoder

set_env()


class DropoutMaskDecoder(nn.Cell):
    def __init__(self, bsz):
        super(DropoutMaskDecoder, self).__init__()
        self.do_mask = get_dropout_mask_decoder(bsz)

    def construct(self, input_mask):
        return self.do_mask(input_mask)


class MsDropoutMaskNet(nn.Cell):
    def __init__(self, o_mask_shape, dropout_rate=0.0):
        """
        :param o_mask_shape:   # bsz, head_num, seq_len, seq_len
        :param dropout_rate:
        """
        super(MsDropoutMaskNet, self).__init__()
        self.fill_v2 = ops.FillV2()
        self.tensor_one = Tensor(1.0, mstype.float16)
        self.do_domask = ops.DropoutDoMask()
        self.tensor_one = Tensor(1.0, mstype.float16)
        self.o_mask_num = functools.reduce(lambda x, y: x * y, o_mask_shape)
        self.keep_prob = Tensor(1 - dropout_rate, dtype=mstype.float16)

    def construct(self, input_mask):
        """
        :param input_mask: input drop mask of dtype uint8
        :return:o  Tensor with shape same as input byte_mask, with dtype of u8
        """
        ones = self.fill_v2((self.o_mask_num,), self.tensor_one)
        return self.do_domask(ones, input_mask, self.keep_prob)


# fmt: off
@pytest.mark.parametrize("input_mask_shape, batch_size",
    [
        ((1,), 1),
        ((128,), 1),
        ((256,), 1),
        ((1024,), 1),
        ((1025,), 1),
        ((1, 31, 1023), 1),
        ((1, 32, 1024, 1024), 1),
        ((1, 32, 2048, 2048), 1),
        ((1, 8, 4096, 4096), 1),
        # (1, 32, 8192, 8192), # Two large for ms to get tensor res.
    ],
    ids=[
        'decode_1B',
        'decode_128B',
        'decode_256B',
        'decode_1K',
        'decode_1K_plus_1B',
        'decode_with_tail',
        'decode_32k',
        'decode_128k',
        'decode_128MB',
         # '8k'
        ]
)
# fmt: on
def test_decoder_precision(input_mask_shape, batch_size):
    i_mask_sz = up_align_128B_input_mask_sz(input_mask_shape)
    print(f'i_mask_sz = {i_mask_sz}')

    i_mask_np = np.random.randint(0, 256, size=(i_mask_sz,))
    i_mask = Tensor(i_mask_np, dtype=ms.uint8)

    dsl_decoder = DropoutMaskDecoder(batch_size)
    dsl_res = dsl_decoder(i_mask).asnumpy()
    print(f'dsl_res.input_mask_shape={dsl_res.shape}')

    ms_decoder = MsDropoutMaskNet((i_mask_sz * 8,))
    ms_res = ms_decoder(i_mask).asnumpy()

    assert np.allclose(dsl_res, ms_res)


def up_align_128B_input_mask_sz(input_mask_shape):
    return (functools.reduce(lambda x, y: x * y, input_mask_shape) + 127) // 128 * 128


# fmt: off
@pytest.mark.parametrize('input_mask_shape, batch_size',
                         [
                             ((1,), 1),
                             ((128,), 1),
                             ((256,), 1),
                             ((1024,), 1),
                             ((1024, 1024), 1),
                             ((1, 32//8, 4096, 4096), 1),
                             # ((1, 32//8, 8192, 8192), 1), # too large for ms
                             # ((1, 40//8, 8192, 8192), 1), # too large for ms
                         ],
                         ids=[
                             '1B',
                             '128B',
                             '256B',
                             '1K',
                             '1M',
                             '64MB',
                             # 'xunfei_2.6B_8k_256MB',
                             # 'xunfei_13B_8k_320MB',
                         ])
# fmt: on
def test_decoder_perf(input_mask_shape, batch_size, request):
    i_mask_sz = up_align_128B_input_mask_sz(input_mask_shape)

    i_mask_np = np.random.randint(0, 2, size=(i_mask_sz,))
    i_mask = Tensor(i_mask_np, dtype=ms.uint8)

    test_id = request.node.callspec.id
    prof_path = f'./dropout_mask_decoder_profiling_cmp/{test_id}'
    if os.path.exists(prof_path):
        shutil.rmtree(prof_path, ignore_errors=True)
        
    profiler = Profiler(output_path=prof_path,
                        profile_communication=True)

    dsl_decoder = DropoutMaskDecoder(bsz=batch_size)
    dsl_decoder(i_mask)

    ms_decoder = MsDropoutMaskNet((i_mask_sz * 8,))
    ms_decoder(i_mask)

    profiler.analyse()
