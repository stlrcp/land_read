// int main()
// #include "acl/acl.h"
#include "utils.h"
// #include "acl/ops/acl_cblas.h"
// #include "acl/acl_op_compiler.h"
#include "aicl.h"

using namespace std;

void PrintResult(void * out_buffers,uint32_t out_tensor_size, std::string out_file){
    void* hostBuffer = nullptr;
    void* outData = nullptr;
    aiclRet ret = aiclrtMallocHost(&hostBuffer, out_tensor_size);
    if (ret != AICL_RET_SUCCESS) {
        ERROR_LOG("fail to print result, malloc host failed");
	    
    }
    ret = aiclrtMemcpy(hostBuffer, out_tensor_size, out_buffers,out_tensor_size, AICL_MEMCPY_DEVICE_TO_HOST);
    if (ret != AICL_RET_SUCCESS) {
        ERROR_LOG("fail to print result, memcpy device to host failed, errorCode is %d", static_cast<int32_t>(ret));
        aiclrtFreeHost(hostBuffer);
	    
    }
    outData = reinterpret_cast<aiclFloat16*>(hostBuffer);
    ofstream outstr(out_file, ios::out | ios::binary);
    outstr.write((char*)outData, out_tensor_size);
    outstr.close();

}

int main(int argc, char* argv[])
{ 

    for (int i = 0; i < argc; i++) {
        cout << "No." << i << " parameter is:" << argv[i] << endl;
    }

    std::string input_x_file = argv[1];
    std::string input_filter_file = argv[2];
    std::string out_file = argv[3];

    std::vector<int64_t> inputShapeCast{2, 1024, 1024, 3};
    std::vector<int64_t> inputFilterShapeCast{6, 3, 3, 3};
    std::vector<int64_t> outputShapeCast{2, 1024, 1024, 6};

    // single op call
    const char* opType_ = "Conv2D";
    int numInput = 3;
    int numOutput = 1;

    aiclDataType inputDataTypeCast = AICL_FLOAT16;
    aiclDataType outputDataTypeCast = AICL_FLOAT16;

    // AICL init
    const char *aiclConfigPath = "../src/aicl.json";
    aiclRet ret = aiclInit(aiclConfigPath);
    if (ret != AICL_RET_SUCCESS) {
        ERROR_LOG("aicl init failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }
    INFO_LOG("aicl init success");

    int32_t deviceId_ = 0;
    aiclrtContext context_;
    aiclrtStream stream_;

    // set device
    ret = aiclrtSetDevice(deviceId_);
    ret = aiclrtCreateContext(&context_, deviceId_);
    ret = aiclrtCreateStream(&stream_);   
        
    aiclTensorDesc *inputDescCast[numInput];
    aiclTensorDesc *OutputDescCast[numOutput];

    // Create aiclTensorDesc, to describe the shape/format/datatype, etc.
    inputDescCast[0] = aiclCreateTensorDesc(inputDataTypeCast, 
                                           inputShapeCast.size(), 
                                           inputShapeCast.data(), 
                                           AICL_FORMAT_NHWC);
    inputDescCast[1] = aiclCreateTensorDesc(inputDataTypeCast, 
                                           inputFilterShapeCast.size(), 
                                           inputFilterShapeCast.data(), 
                                           AICL_FORMAT_NCHW);
    inputDescCast[2] = aiclCreateTensorDesc(AICL_DT_UNDEFINED, 0, nullptr, AICL_FORMAT_UNDEFINED);
    OutputDescCast[0] = aiclCreateTensorDesc(outputDataTypeCast, 
                                            outputShapeCast.size(), 
                                            outputShapeCast.data(), 
                                            AICL_FORMAT_NHWC);
 
    // set Conv2D attr
    aiclopAttr *opAttr = aiclopCreateAttr();
    if (opAttr == nullptr) {
        ERROR_LOG("singleOp create attr failed");
        return FAILED;
    }
    int64_t intList[4]{1, 1, 1, 1};

    ret = aiclopSetAttrListInt(opAttr, "strides", 4, intList);
    if (ret != AICL_RET_SUCCESS) {
        ERROR_LOG("singleOp set strides attr failed");
        aiclopDestroyAttr(opAttr);
        return FAILED;
    }
    
    ret = aiclopSetAttrListInt(opAttr, "pads", 4, intList);
    if (ret != AICL_RET_SUCCESS) {
        ERROR_LOG("singleOp set pads attr failed");
        aiclopDestroyAttr(opAttr);
        return FAILED;
    }

    ret = aiclopSetAttrListInt(opAttr, "dilations", 4, intList);
    if (ret != AICL_RET_SUCCESS) {
        ERROR_LOG("singleOp set dilations attr failed");
        aiclopDestroyAttr(opAttr);
        return FAILED;
    }   

    void* x_tensor_ptr = nullptr;
    void* filter_tensor_ptr = nullptr;
    void* out_tensor_ptr = nullptr;
    uint32_t x_tensor_size;
    uint32_t filter_tensor_size;
    uint32_t out_tensor_size = 25165824;
    std::vector<aiclDataBuffer*> in_buffers;
    std::vector<aiclDataBuffer*> out_buffers;

    x_tensor_ptr = Utils::GetDeviceBufferOfFile(input_x_file, x_tensor_size);
    filter_tensor_ptr = Utils::GetDeviceBufferOfFile(input_filter_file, filter_tensor_size);

    ret = aiclrtMalloc(&out_tensor_ptr, out_tensor_size, AICL_MEM_MALLOC_HUGE_FIRST);

    aiclDataBuffer* x_tensor_data = aiclCreateDataBuffer(x_tensor_ptr, x_tensor_size);
  
    if (x_tensor_data == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        return FAILED;
    }

    aiclDataBuffer* filter_tensor_data = aiclCreateDataBuffer(filter_tensor_ptr, filter_tensor_size);
    if (filter_tensor_data == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        return FAILED;
    }

    aiclDataBuffer* bias_tensor_data = aiclCreateDataBuffer(nullptr, 0);

    aiclDataBuffer* out_tensor_data = aiclCreateDataBuffer(out_tensor_ptr, out_tensor_size);
    if (out_tensor_data == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        return FAILED;
    }

    in_buffers.push_back(x_tensor_data);
    in_buffers.push_back(filter_tensor_data);
    in_buffers.push_back(bias_tensor_data);
    out_buffers.push_back(out_tensor_data);
    for(int i = 0; i < 1; i++)
    {
        ret = aiclopCompileAndExecute(opType_, numInput, inputDescCast, 
        in_buffers.data(), numOutput, OutputDescCast, out_buffers.data(),
        opAttr, AICL_ENGINE_SYS, AICL_COMPILE_SYS, NULL, stream_);
        ret = aiclrtSynchronizeStream(stream_);
        if (ret != AICL_RET_SUCCESS) {
            ERROR_LOG("execute singleOp conv2d failed, errorCode is %d", static_cast<int32_t>(ret));
            aiclDestroyTensorDesc(inputDescCast[0]);
            aiclDestroyTensorDesc(OutputDescCast[0]);
            return FAILED;
        }

        INFO_LOG("execute conv2d %d", i);
        PrintResult(out_tensor_ptr, out_tensor_size, out_file);

    }    

    INFO_LOG("execute op success");

    if (stream_ != nullptr) {
        ret = aiclrtDestroyStream(stream_);
        if (ret != AICL_RET_SUCCESS) {
            ERROR_LOG("destroy stream failed, errorCode = %d", static_cast<int32_t>(ret));
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    if (context_ != nullptr) {
        ret = aiclrtDestroyContext(context_);
        if (ret != AICL_RET_SUCCESS) {
            ERROR_LOG("destroy context failed, errorCode = %d", static_cast<int32_t>(ret));
        }
        context_ = nullptr;
    }
    INFO_LOG("end to destroy context");

    ret = aiclrtResetDevice(deviceId_);
    if (ret != AICL_RET_SUCCESS) {
        ERROR_LOG("reset device %d failed, errorCode = %d", deviceId_, static_cast<int32_t>(ret));
    }
    INFO_LOG("end to reset device ");

    ret = aiclFinalize();
    if (ret != AICL_RET_SUCCESS) {
        ERROR_LOG("finalize aicl failed, errorCode = %d", static_cast<int32_t>(ret));
    }
    INFO_LOG("end to finalize aicl");

    return SUCCESS;
}
