#!/bin/bash
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"
echo ${ScriptPath}
ModelPath="${ScriptPath}/../model"
common_script_dir=${THIRDPART_PATH}/common
# . ${common_script_dir}/sample_common.sh

function main()
{
  echo "[INFO] Sample preparation"

  mkdir -p build && cd build
  if [ $? -ne 0 ];then
    return 1
  fi
    
  cmake .. && make && make install
  if [ $? -ne 0 ];then
    return 1
  fi
    
  echo "[INFO] Sample preparation is complete"
}
main