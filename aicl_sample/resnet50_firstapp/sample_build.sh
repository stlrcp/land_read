ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"
model_name="MyFirstApp_build"

cd ${ScriptPath}/model
git clone https://github.com/htshinichi/caffe-onnx.git
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/resnet50/resnet50.prototxt
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/resnet50/resnet50.caffemodel
python3 caffe-onnx/convert2onnx.py resnet50.prototxt resnet50.caffemodel resnet50 ./ > /dev/null
export LD_LIBRARY_PATH=/home/zhenpeng.wang/lib/igie/build/:$LD_LIBRARY_PATH
igie-exec --model_path resnet50.onnx  --input input:1,3,224,224 --precision fp32 --engine_path resnet50.so

cd ${ScriptPath}/data

wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog1_1024_683.jpg

python3 ../script/transferPic.py

if [ -d ${ScriptPath}/build/intermediates/host ];then
	rm -rf ${ScriptPath}/build/intermediates/host
fi

mkdir -p ${ScriptPath}/build/
cd ${ScriptPath}/build/

cmake ../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE

make

if [ $? == 0 ];then
	echo "make for app ${model_name} Successfully"
	exit 0
else
	echo "make for app ${model_name} failed"
	exit 1
fi

