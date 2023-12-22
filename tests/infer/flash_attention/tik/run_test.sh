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

if [ ${SOC_VERSION} == "Ascend310P" ]; then
    soc_version="Ascend310P3"
else
    soc_version=${SOC_VERSION}
fi

> test.log

batch=(1)
heads=(1 2 3 4 5 6 7 8)
q_tokens=(512 1024 1536 2048 2560 3072 3584 4096)
kv_tokens=(512 1024 1536 2048 2560 3072 3584 4096)
embed=(40 64)

for b in ${batch[@]}
do
    for h in ${heads[@]}
    do
        for q in ${q_tokens[@]}
        do
            for kv in ${kv_tokens[@]}
            do
                for d in ${embed[@]}
                do
                    echo "Case: batch =" $b ", heads =" $h ", q_seqlen =" $q ", kv_seqlen =" $kv ", embed =" $d | tee -a test.log
                    python3 data_gen.py $b $h $q $kv $d
                    python3 model.py
                    atc --model=fa_tik.onnx --framework=5 --output=fa_tik --soc_version=${soc_version}
                    python3 -m ais_bench --model fa_tik.om --input query.npy,key.npy,value.npy --output result --output_dirname "fa_tik" --outfmt NPY --warmup_count 0
                    python3 compare.py | tee -a test.log
                    echo -e "\n"
                    sleep 1
                done
            done
        done
    done
done