# Copyright 2023-2023 Huawei Technologies Co., Ltd
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

import torch
import numpy as np

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value):
        ret = torch.zeros_like(query)
        return ret

    @staticmethod
    def symbolic(g, query, key, value):
        return g.op('FlashAttentionTik', query, key, value)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        ret = FlashAttention.apply(query, key, value)
        return ret


def export():
    models = Model()
    query = torch.from_numpy(np.load("./query.npy"))
    key = torch.from_numpy(np.load("./key.npy"))
    value = torch.from_numpy(np.load("./value.npy"))
    models.eval()
    torch.onnx.export(
        models,
        (query, key, value),
        "fa_tik.onnx",
        opset_version=11,
        export_params=False,
        verbose=False,
        input_names=["q", "k", "v"],
        output_names=["y"],
    )


if __name__ == "__main__":
    export()