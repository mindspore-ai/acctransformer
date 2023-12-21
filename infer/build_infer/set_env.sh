current_script_dir=$(dirname $(readlink -f $0))
export ASCEND_CUSTOM_OPP_PATH=${current_script_dir}/pkg/vendors/customize:${current_script_dir}/pkg/vendors/aie_ascendc:$ASCEND_CUSTOM_OPP_PATH