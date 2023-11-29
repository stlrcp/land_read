ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"
echo ${ScriptPath}
cd ${ScriptPath}

# create test data

input_shape='[2,1024,1024,3]'  # NHWC
filter_shape='[6,3,3,3]'       # NCHW 
output_shape='[2,1024,1024,6]' # NHWC

input_file=$(python3 tools_generate_data.py x -s ${input_shape} -r [1,10] -d float16)
filter_file=$(python3 tools_generate_data.py filter -s ${filter_shape} -r [1,3] -d float16)
output_file=$(python3 tools_generate_data.py out -s ${output_shape} -r [1,100] -d float16)

x_format='NHWC'

mv ${input_file} ../data
mv ${filter_file} ../data
mv ${output_file} ../data
../out/main '../data/'$input_file '../data/'$filter_file '../data/'$output_file


if tensorflow=$(python3 -c "import tensorflow;print(tensorflow.__version__)" 2>/dev/null);then
    if [ ${tensorflow} != 1.15.0 ];then
        cd /opt/apps/tensorflow/build_pip/
        pip3 install * --user 2>/dev/null
    fi
else
    cd /opt/apps/tensorflow/build_pip/
    pip3 install * --user 2>/dev/null
fi

cd -
python3 computebytf.py '../data/'$input_file '../data/'$filter_file '../data/'$output_file
