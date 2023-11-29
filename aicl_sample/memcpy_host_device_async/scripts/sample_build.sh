#!/bin/bash
ScriptPath=`pwd -P`
CodePath=`(cd "$(dirname "${ScriptPath}")"; pwd -P)`
cd ${CodePath}

function main()
{
  echo "[INFO] Sample preparation"

  mkdir build && cd build;
  if [ $? -ne 0 ];then
    return 1
  fi

  cmake .. && make VERBOSE=1;
  if [ $? -ne 0 ];then
    return 1
  fi
    
  echo "[INFO] Sample preparation is complete"
}
main