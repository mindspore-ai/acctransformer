#!/bin/bash
# -*- coding: utf-8 -*-

# 构建环境使用CANN主线包，容易引入兼容性问题。同时为了更好地控制对外发布内容，我们
# 在构建环境用msopgen工具生成工程，然后将要发布的算子交付件拷贝到新生成的工程构建
set -e

VERSION="1.0.0"
pkg_name = "flash_attention_infer_${VERSION}.run"
current_script_dir=$(dirname $(readlink -f $0))

if [ "x${ASCEND_TOOLKIT_HOME}" != "x" ]; then
    local_toolkit=${ASCEND_TOOLKIT_HOME}
else
    echo "Can not find toolkit path, please set ASCEND_TOOLKIT_HOME"
    echo "eg: export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest"
    exit 1
fi

msopgen=${local_toolkit}/python/site-packages/bin/msopgen
if [ ! -f ${msopgen} ]; then
    echo "${msopgen} not exists"
    exit 1
fi

function make_package(){
    cd ${current_script_dir}
    rm -rf pkg
    mkdir pkg
    chmod +w vendors
    mv vendors pkg
    chmod -w pkg/vendors
    chmod -w pkg
    ./custom_project/cmake/util/makeself/makeself.sh \
        --header ./custom_project/cmake/util/makeself/makeself-header.sh \
        --gzip --notemp --complevel 4 --nomd5 --sha256 --chown \
        ./pkg ${pkg_name} 'flash attention infer'
}

function build_ops(){
    ori_path=${PWD}
    cd ${current_script_dir}
    rm -rf vendors
    if [ ! -d ${current_script_dir}/nlohmannJson ]; then
        if [ -f /opt/ubuntu20/json-3.10.1.tar.gz ]; then
            echo "Found json-3.10.1.tar.gz"
            tar xf /opt/ubuntu20/json-3.10.1.tar.gz -C ${current_script_dir}
            mv ${current_script_dir}/json-3.10.1 ${current_script_dir}/nlohmannJson
        else
            echo "Did not find json-3.10.1.tar.gz, downloading……"
            git clone -b v3.10.1 https://github.com/nlohmann/json.git ${current_script_dir}/nlohmannJson
        fi
    fi

    source ${current_script_dir}/build_tik_ops.sh
    rm -rf ${current_script_dir}/vendors/customize/bin
    make_package
    cd ${ori_path}
}

build_ops