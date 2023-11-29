#!/bin/bash
script_path="$( cd "$(dirname $BASH_SOURCE)" ; pwd -P)"

project_path=${script_path}/..
input1_path=${project_path}/run/out/test_data/data/input_0.bin
input2_path=${project_path}/run/out/test_data/data/input_1.bin
output_path=${project_path}/run/out/result_files/output_0.bin

declare -i success=0
declare -i runError=1
declare -i verifyResError=2

echo ${script_path}
# generate singleop model
cd ${script_path}
python3 gen_opModel.py --singleop="../run/out/test_data/config/add_op.json" --output="../run/out/"

# generate test data
cd ${project_path}/run/out/test_data/data/
python3 generate_data.py 
cd ${project_path}

# generate an executable
mkdir -p ${project_path}/build/ && cd ${project_path}/build/
cmake ${project_path}/src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE && make && make install

# run the executable
cd ${project_path}/run/out/
./execute_add_op
if [ $? -ne 0 ]; then
    echo "ERROR: run case failed, please check your project."
fi

# Verify the result 
cd ${script_path}
python3 ${script_path}/verify_result.py ${input1_path} ${input2_path} ${output_path}
if [ $? -ne 0 ];then
    echo "ERROR: run failed. the result of reasoning is wrong!"
fi
echo "run success"

