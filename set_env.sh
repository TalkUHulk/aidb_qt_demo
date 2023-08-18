#!/bin/bash
#source set_env.sh

CURRENT_DIR="ai.deploy.box/libs/mac/x86_64"
lib_path="${CURRENT_DIR}"
lib_tnn_path="${CURRENT_DIR}/tnn"
lib_openvino_path="${CURRENT_DIR}/openvino"
lib_paddlelite_path="${CURRENT_DIR}/paddlelite"


LD_LIBRARY_PATH=${lib_path}:${lib_tnn_path}:${lib_openvino_path}:${lib_paddlelite_path}:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH


