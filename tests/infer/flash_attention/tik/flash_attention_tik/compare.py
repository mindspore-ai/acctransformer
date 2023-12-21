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

import numpy as np

golden = np.load("./golden.npy")
output = np.load("./result/fa_tik/query_0.npy")

print(golden.shape)
print(output.shape)

golden = golden.flatten()
output = output.flatten()
count = golden.shape[0]

threhold = 0.001
max_diff = 0
errcnt_0 = 0
errcnt_1 = 0

for i in range(count):
    gap = abs(golden[i] - output[i])
    if gap > threhold:
        errcnt_0 = errcnt_0 + 1
    if gap > abs(threhold * golden[i]):
        errcnt_1 = errcnt_1 + 1
    if gap > max_diff:
        max_diff = gap

if count > 0:
    err_ratio_0 = errcnt_0 / count * 100
    err_ratio_1 = errcnt_1 / count * 100
    print("count: " + str(count))
    print("error count 0: " + str(errcnt_0))
    print("error ratio 0: " + str(err_ratio_0) + "%")
    print("error count 1: " + str(errcnt_1))
    print("error ratio 1: " + str(err_ratio_1) + "%")
    print("max gap value: " + str(max_diff))