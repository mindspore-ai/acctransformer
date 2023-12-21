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

import sys
import torch
import numpy as np
from torch import nn

def save_data(x, name):
    print(f"{name.split('.')[0]}.shape is {x.shape}")
    np.save(name, x)


def main(batch, heads, q_tokens, kv_tokens, embed):
    np.random.seed(42)
    torch.manual_seed(42)

    query = torch.randn((batch, heads, q_tokens, embed), dtype=torch.half)
    key = torch.randn((batch, heads, kv_tokens, embed), dtype=torch.half)
    value = torch.randn_like(key)
    score = torch.matmul(query.float(), key.permute(0, 1, 3, 2).float())
    score_softmax = torch.nn.functional.softmax(score, dim=-1)
    golden = torch.matmul(score_softmax, value.float()).half()

    save_data(query, "query.npy")
    save_data(key, "key.npy")
    save_data(value, "value.npy")
    save_data(golden, "golden.npy")


if __name__ == "__main__":
    param_num = 5
    if len(sys.argv) == (param_num + 1):
        batch = int(sys.argv[1])
        heads = int(sys.argv[2])
        q_tokens = int(sys.argv[3])
        kv_tokens = int(sys.argv[4])
        embed = int(sys.argv[5])
        main(batch, heads, q_tokens, kv_tokens, embed)