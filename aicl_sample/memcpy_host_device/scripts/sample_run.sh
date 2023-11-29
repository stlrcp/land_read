#!/bin/bash
ScriptPath=`pwd -P`
CodePath=`(cd "$(dirname "${ScriptPath}")"; pwd -P)`
BuildPath=${CodePath}/build
cd ${BuildPath}/src

function main()
{
    echo "[INFO] The sample starts to run"

    running_command="./main --release_cycle 2 --number_of_cycles 2\
    --device_id 0 --memory_size 10485760 --write_back_host 1 --memory_reuse 1"
    # start runing
    ${running_command}
    if [ $? -ne 0 ];then
        return 1
    fi
}
main
