#!/bin/bash
ScriptPath=`pwd -P`
CodePath=`(cd "$(dirname "${ScriptPath}")"; pwd -P)`
BuildPath=${CodePath}/build
cd ${BuildPath}/src

function main()
{
    echo "[INFO] The sample starts to run"

    running_command="./main"
    # start runing
    ${running_command}
    if [ $? -ne 0 ];then
        return 1
    fi
}
main