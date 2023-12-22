#!/bin/bash
# -*- coding: utf-8 -*-

src=${current_script_dir}/../flash_attention/tik
dst=${current_script_dir}/custom_project

function create_empty_custom_project(){
    cd ${current_script_dir}
    rm -rf ${dst}
    if [ ${SOC_VERSION} == "Ascend310P"]; then
        ${msopgen} gen -i ir_demo.json -f onnx -c ai_core-ascend310p -out ${dst}
        rm ${dst}/tbe/op_info_cfg/ai_core/ascend310p/*.ini
    elif [ ${SOC_VERSION} == "Ascend910" ]; then
        ${msopgen} gen -i ir_demo.json -f onnx -c ai_core-ascend910 -out ${dst}
        rm ${dst}/tbe/op_info_cfg/ai_core/ascend910/*.ini
    else
        echo "${SOC_VERSION} not support"
        exit 1
    fi
    rm ${dst}/framework/onnx_plugin/*.cc
    rm ${dst}/op_proto/*.h
    rm ${dst}/op_proto/*.cc
    rm ${dst}/tbe/impl/*.py
}

function release_framework_onnx(){
    cd ${src}/framework/onnx_plugin
    local files=(
        flash_attention_tik_plugin.cpp
    )
    cp ${files[@]} ${dst}/framework/onnx_plugin
}

function release_op_proto(){
    cd ${src}/op_proto
    local files=(
        flash_attention_tik.cpp
        flash_attention_tik.h
    )
    cp ${files[@]} ${dst}/op_proto
}

function release_op_impl(){
    cd ${src}/tbe/impl
    local files=(
        constants.py
        flash_attention_fwd.py
        flash_attention_tik.py
        tik_ops_utils.py
    )
    cp ${files[@]} ${dst}/tbe/impl
}

function release_cfg(){
    local files=(
        flash_attention_tik.ini
    )
    if [ ${SOC_VERSION} == "Ascend310P"]; then
        cd ${src}/tbe/op_info_cfg/ai_core/ascend310p
        cp ${files[@]} ${dst}/tbe/op_info_cfg/ai_core/ascend310p
    elif [ ${SOC_VERSION} == "Ascend910" ]; then
        cd ${src}/tbe/op_info_cfg/ai_core/ascend910
        cp ${files[@]} ${dst}/tbe/op_info_cfg/ai_core/ascend910
    else
        echo "${SOC_VERSION} not support"
        exit 1
    fi
}

function revise_settings(){
    cd ${dst}
    sed -i "43i export ASCEND_TENSOR_COMPILER_INCLUDE=${local_toolkit}/include" build.sh
    sed -i "6s# <foss@huawei.com>##g" cmake/util/makeself/makeself-header.sh
}

function build_and_install(){
    cd ${dst}
    bash build.sh
    rm -rf ${current_script_dir}/vendors
    mkdir ${current_script_dir}/vendors
    chmod +w ${current_script_dir}/vendors
    bash ${dst}/build_out/*.run --install-path=${current_script_dir}/vendors
    chmod -w ${current_script_dir}/vendors
}

function build_tik_ops(){
    ori_path=${PWD}
    create_empty_custom_project
    release_framework_onnx
    release_op_proto
    release_op_impl
    release_cfg
    revise_settings
    build_and_install
    cd ${ori_path}
}

build_tik_ops