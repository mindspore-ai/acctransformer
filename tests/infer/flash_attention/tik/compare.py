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


def data_compare(ground_truth, predict, rtol=0.001, atol=0.001, max_rtol=0.1):
    total_count = np.prod(ground_truth.shape)
    greater_than_diff_thd_count = np.sum(
        np.abs(predict - ground_truth) > rtol * np.abs(ground_truth) + atol
    )
    greater_than_max_diff_thd_count = np.sum(
        np.abs(predict - ground_truth) > max_rtol * np.abs(ground_truth) + atol
    )

    diff_gt_thd_proportion = greater_than_diff_thd_count / total_count
    diff_gt_max_thd_proportion = greater_than_max_diff_thd_count / total_count
    return diff_gt_thd_proportion, diff_gt_max_thd_proportion


def display(ground_truth, predict, proportion1, proportion2, thd1=0.001, thd2=0.1):
    print(
        f"relative diff greater than {thd1} proportion: {proportion1}, "
        f"relative diff greater than {thd2} proportion: {proportion2}"
    )

    diff = np.abs(predict - ground_truth)
    diff_max = np.max(diff)
    diff_avg = np.average(diff)
    index = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
    index = tuple(map(int, index))
    print(f"avg diff: {diff_avg}\nmax diff {diff_max}, ground_truth: {ground_truth[index]}, predict: {predict[index]}")


def check():
    golden = np.load("./golden.npy")
    output = np.load("./result/fa_tik/query_0.npy")

    print(golden.shape)
    print(output.shape)

    diff_gt_rtol_ratio, diff_gt_max_rtol_ratio = data_compare(golden, output, rtol=0.05)
    display(golden, output, diff_gt_rtol_ratio, diff_gt_max_rtol_ratio, thd1=0.05)


if __name__ == '__main__':
    check()
