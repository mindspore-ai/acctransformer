#Env setting
export GLOG_v=3
export DEVICE_ID=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

export LD_LIBRARY_PATH=/usr/include/hdf5/lib:/usr/local/gcc-7.3.0/lib64:${LD_LIBRARY_PATH}

export PYTHONPATH=/usr/local/python3.7.5/lib/python3.7/site-packages:$PYTHONPATH

export PATH=/usr/local/gcc-7.3.0/bin:${PATH}

export OPENSSL_ROOT_DIR=/usr/local/openssl

export CC=/usr/local/gcc-7.3.0/bin/gcc
export CXX=/usr/local/gcc-7.3.0/bin/g++

export ASCEND_TENSOR_COMPILER_INCLUDE=/usr/local/Ascend/ascend-toolkit/latest/include
#export TOOLCHAIN_DIR=/usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc

export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux
export NPU_HOST_LIB=$DDK_PATH/acllib/lib64/stub

source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# For v100 bug, use v200 specially.
unset TOOLCHAIN_HOME
# lib libraries that the run package depends on
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/toolkit/tools/simulator/Ascend310P1/lib/:${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}
#export PYTHONPATH=$PYTHONPATH:/home/leisu/memory_bridge/memory-bridge/ai-ascend/memory-offload/ops/mindspore

export ASCEND_CUSTOM_PATH=/usr/local/Ascend
