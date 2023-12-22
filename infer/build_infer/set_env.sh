#!/bin/bash
# -*- coding: utf-8 -*-

current_script_dir=$(dirname $(readlink -f $0))
export ASCEND_CUSTOM_OPP_PATH=${current_script_dir}/pkg/vendors/customize:$ASCEND_CUSTOM_OPP_PATH