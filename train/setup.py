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
Setup package.
"""
from setuptools import find_packages
from setuptools import setup

print('WARING: begin to install acctransformer.')
setup(
    name='acctransformer',
    version='1.0.0',
    author="Huawei",
    description="acctransformer training tik.",
    url="https://gitee.com/mindspore/acctransformer",
    license="Apache License, Version 2.0",
    packages=find_packages(where='.'),
    zip_safe=False
)
