#include <iostream>
#include <fstream>
#include <cstring>
#include <map>
// #include "acl/acl.h"
#include "aicl.h"


using namespace std;
int32_t deviceId_ = 0;
uint32_t modelId;
size_t pictureDataSize = 0;
void *pictureHostData;
void *pictureDeviceData;
aiclmdlDataset *inputDataSet;
aiclDataBuffer *inputDataBuffer;
aiclmdlDataset *outputDataSet;
aiclDataBuffer *outputDataBuffer;
aiclmdlDesc *modelDesc;
size_t outputDataSize = 0;
void *outputDeviceData;
void *outputHostData;


// AscendCL初始化、运行管理资源申请（指定计算设备）
void InitResource()
{
	aiclRet ret = aiclInit(nullptr);
	ret = aiclrtSetDevice(deviceId_);
}


// 申请内存，使用C/C++标准库的函数将测试图片读入内存
void ReadPictureTotHost(const char *picturePath)
{
	string fileName = picturePath;
	ifstream binFile(fileName, ifstream::binary);
	binFile.seekg(0, binFile.end);
	pictureDataSize = binFile.tellg();
	binFile.seekg(0, binFile.beg);
	aiclRet ret = aiclrtMallocHost(&pictureHostData, pictureDataSize);
	binFile.read((char*)pictureHostData, pictureDataSize);
	binFile.close();
}


// 申请Device侧的内存，再以复制内存的方式将内存中的图片数据传输到Device
void CopyDataFromHostToDevice()
{
	aiclRet ret = aiclrtMalloc(&pictureDeviceData, pictureDataSize, AICL_MEM_MALLOC_HUGE_FIRST);
	ret = aiclrtMemcpy(pictureDeviceData, pictureDataSize, pictureHostData, pictureDataSize, AICL_MEMCPY_HOST_TO_DEVICE);
}


// 准备模型推理的输入数据结构
void CreateModelInput()
{
	// 创建aiclmdlDataset类型的数据，描述模型推理的输入
	inputDataSet = aiclmdlCreateDataset();
	inputDataBuffer = aiclCreateDataBuffer(pictureDeviceData, pictureDataSize);
	aiclRet ret = aiclmdlAddDatasetBuffer(inputDataSet, inputDataBuffer);
}


// 准备模型推理的输出数据结构
void CreateModelOutput()
{
	// 创建模型描述信息
	modelDesc =  aiclmdlCreateDesc();
	aiclRet ret = aiclmdlGetDesc(modelDesc, modelId);
	
	// 创建aiclmdlDataset类型的数据，描述模型推理的输出
	outputDataSet = aiclmdlCreateDataset();
	
	// 获取模型输出数据需占用的内存大小，单位为Byte
	outputDataSize = aiclmdlGetOutputSizeByIndex(modelDesc, 0);
	
	// 申请输出内存
	ret = aiclrtMalloc(&outputDeviceData, outputDataSize, AICL_MEM_MALLOC_HUGE_FIRST);
	outputDataBuffer = aiclCreateDataBuffer(outputDeviceData, outputDataSize);
	ret = aiclmdlAddDatasetBuffer(outputDataSet, outputDataBuffer);
}


// 将图片数据读入内存
void LoadPicture(const char* picturePath)
{
	ReadPictureTotHost(picturePath);
	CopyDataFromHostToDevice();
}


// 加载模型
void LoadModel(const char* modelPath)
{
	aiclRet ret = aiclmdlLoadFromFile(modelPath, &modelId);
}


// 执行推理
void Inference()
{
    CreateModelInput();
	CreateModelOutput();
	aiclRet ret = aiclmdlExecute(modelId, inputDataSet, outputDataSet);
}


// 在终端上屏显测试图片的top5置信度的类别编号
void PrintResult()
{
	aiclRet ret = aiclrtMallocHost(&outputHostData, outputDataSize);
	ret = aiclrtMemcpy(outputHostData, outputDataSize, outputDeviceData, outputDataSize, AICL_MEMCPY_DEVICE_TO_HOST);
	float* outFloatData = reinterpret_cast<float *>(outputHostData);
	
	map<float, unsigned int, greater<float>> resultMap;
	for (unsigned int j = 0; j < outputDataSize / sizeof(float);++j)
	{
		resultMap[*outFloatData] = j;
		outFloatData++;
	}
	
	int cnt = 0;
	for (auto it = resultMap.begin();it != resultMap.end();++it)
	{
		if(++cnt > 5)
		{
			break;
		}
		printf("top %d: index[%d] value[%lf] \n", cnt, it->second, it->first);
	}
}


// 卸载模型
void UnloadModel()
{
	aiclmdlDestroyDesc(modelDesc);
	aiclmdlUnload(modelId);
}


// 释放内存、销毁推理相关的数据类型，防止内存泄露
void UnloadPicture()
{
	aiclRet ret = aiclrtFreeHost(pictureHostData);
	pictureHostData = nullptr;
	ret = aiclrtFree(pictureDeviceData);
	pictureDeviceData = nullptr;
	aiclDestroyDataBuffer(inputDataBuffer);
	inputDataBuffer = nullptr;
	aiclmdlDestroyDataset(inputDataSet);
	inputDataSet = nullptr;
	
	ret = aiclrtFreeHost(outputHostData);
	outputHostData = nullptr;
	ret = aiclrtFree(outputDeviceData);
	outputDeviceData = nullptr;
	aiclDestroyDataBuffer(outputDataBuffer);
	outputDataBuffer = nullptr;
	aiclmdlDestroyDataset(outputDataSet);
	outputDataSet = nullptr;
}


// AscendCL去初始化、运行管理资源释放（指定计算设备）
void DestroyResource()
{
	aiclRet ret = aiclrtResetDevice(deviceId_);
	aiclFinalize();
}

int main()
{
	// 1.定义一个资源初始化的函数，用于AscendCL初始化、运行管理资源申请（指定计算设备）
	InitResource();
	
	// 2.定义一个模型加载的函数，加载图片分类的模型，用于后续推理使用
	const char *mdoelPath = "../model/resnet50.so";
	LoadModel(mdoelPath);
	
	// 3.定义一个读图片数据的函数，将测试图片数据读入内存，并传输到Device侧，用于后续推理使用
	const char *picturePath = "../data/dog1_1024_683.bin";
	LoadPicture(picturePath);
	
	// 4.定义一个推理的函数，用于执行推理
	Inference();
	
	// 5.定义一个推理结果数据处理的函数，用于在终端上屏显测试图片的top5置信度的类别编号
	PrintResult();
	
	// 6.定义一个模型卸载的函数，卸载图片分类的模型
	UnloadModel();
	
	// 7.定义一个函数，用于释放内存、销毁推理相关的数据类型，防止内存泄露
	UnloadPicture();
	
	// 8.定义一个资源去初始化的函数，用于AscendCL去初始化、运行管理资源释放（指定计算设备）
	DestroyResource();
}