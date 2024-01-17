RANK_SIZE=8
EXEC_PATH=$(dirname $(readlink -f $0))

export RANK_TABLE_FILE=${EXEC_PATH}/ranktables/ranktable_8p.json
export RANK_SIZE=$RANK_SIZE

check_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

# get log path
DATE_TIME=`date +'%Y_%m_%d_%H_%M_%S'`
LOG_DIR=$(check_real_path "${EXEC_PATH}/log_${DATE_TIME}")
echo "log dir is ${LOG_DIR}"

for((i=0;i<${RANK_SIZE};i++))
do
    mkdir -p ${LOG_DIR}/device$i
    cd ${LOG_DIR}/device$i || exit
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start test for device $i"
    pytest -sv ${EXEC_PATH}/test_fa_grad_precision_distribute.py > test_fa_grad_precision_distribute.log 2>&1 &
    cd ${EXEC_PATH} || exit
done