#!/bin/bash

script_path="$( cd "$(dirname $BASH_SOURCE)" ; pwd -P)"
project_path=${script_path}/..
input1_path=${project_path}/run/out/test_data/data/input_0.bin
input2_path=${project_path}/run/out/test_data/data/input_1.bin
output_path=${project_path}/run/out/result_files/output_0.bin

declare -i success=0
declare -i runError=1
declare -i verifyResError=2

function envCheck() {
    export HOME_ROOT=/opt/apps
    export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    export IGIE_ROOT=/home/zhenpeng.wang/lib/igie

    if [ -d ${ASCEND_INSTALL_PATH}/compiler ]; then
        echo "WARNING: op online compile can only run with compiler, please check it installed correctly!"
    fi
}

function setEnv() {
    # set environment
    export PYTHONPATH=/opt/rh/devtoolset-7/root/usr/lib64/python2.7/site-packages:/opt/rh/devtoolset-7/root/usr/lib/python2.7/site-packages:${HOME_ROOT}/local/lib64/python3/dist-packages:${PYTHONPATH}
    export LD_LIBRARY_PATH=${IGIE_ROOT}/build/:${LD_LIBRARY_PATH}
    export ASCEND_OPP_PATH=${ASCEND_INSTALL_PATH}/opp
    export PATH=${HOME_ROOT}/local/lib64/python3/dist-packages/bin:/opt/rh/devtoolset-7/root/usr/bin:/opt/sw_home/local/bin:/opt/sw_home/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
}

function compile() {
    # compile 
    # Create a directory to hold the compiled files
    mkdir -p ${project_path}/build/
    if [ $? -ne 0 ];then
        echo "ERROR: mkdir build folder failed. please check your project"
        return ${inferenceError}
    fi
    cd ${project_path}/build/

    # Set the environment variables that your code needs to compile
    if [ $? -ne 0 ];then
        echo "ERROR: set build environment failed"
        return ${inferenceError}
    fi

    # Generate Makefile
    cmake ${project_path}/src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
    if [ $? -ne 0 ];then
        echo "ERROR: cmake failed. please check your project"
        return ${inferenceError}
    fi

    make && make install
    if [ $? -ne 0 ];then
        echo "ERROR: make failed. please check your project"
        return ${inferenceError}
    fi
}

function runCase() {
    # Generate test data
    cd ${project_path}/run/out/test_data/data
    python3 generate_data.py
    if [ $? -ne 0 ];then
        echo "ERROR: generate input data failed!"
        return ${verifyResError}
    fi

    cd ${project_path}/run/out
    # Run the program
    ./compile_add_op
    if [ $? -ne 0 ];then
        echo "ERROR: run failed. please check your project"
        return ${inferenceError}
    fi
}

function main() {
    envCheck
    setEnv
    compile
    if [ $? -ne 0 ]; then
        echo "ERROR: compile failed, please check your project."
        return ${runError}
    fi

    runCase
    if [ $? -ne 0 ]; then
        echo "ERROR: run case failed, please check your project."
        return ${runError}
    fi

    # Call the python script to determine if the results of this project reasoning are correct
    python3 ${script_path}/verify_result.py ${input1_path} ${input2_path} ${output_path}
    if [ $? -ne 0 ];then
        echo "ERROR: run failed. the result of reasoning is wrong!"
        return ${inferenceError}
    fi
    echo "run success"
    return ${success}
}

main
