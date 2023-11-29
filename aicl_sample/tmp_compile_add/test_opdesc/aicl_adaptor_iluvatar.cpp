/*
 * Copyright © 2023 Iluvatar CoreX. All rights reserved.
 * Copyright Declaration: This software, including all of its code and documentation, except for the third-party
 * software it contains, is a copyrighted work of Shanghai Iluvatar CoreX Semiconductor Co., Ltd. and its affiliates
 * (“Iluvatar CoreX”) in accordance with the PRC Copyright Law and relevant international treaties, and all rights
 * contained therein are enjoyed by Iluvatar CoreX. No user of this software shall have any right, ownership or interest
 * in this software and any use of this software shall be in compliance with the terms and conditions of the End User
 * License Agreement.
 */

#include "aicl.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <dmlc/logging.h>
#include <tvm/te/operation.h>
#include </usr/local/include/python3.7m/Python.h>

using namespace std;



cudaError_t cudaGetRunMode(aiclrtRunMode * runMode){
    *runMode = AICL_HOST;
    return cudaSuccess;
}

CUresult cuFinalize(void){
    return CUDA_SUCCESS;
}

//  -- trans cudaError_t type
aiclRet errcod_trans(cudaError_t ret) {
  switch (ret)
  {
      case cudaSuccess:        
          return AICL_RET_SUCCESS; 
      case cudaErrorNoDevice:     
          return AICL_RET_INVALID_DEVICE;
      case cudaErrorInvalidDevice:    
          return AICL_RET_INVALID_DEVICE;
      case cudaErrorInvalidValue:   
          return AICL_RET_INVALID_DEVICE_ID;
      case cudaErrorFileNotFound:     
          return AICL_RET_INVALID_FILE;
      case cudaErrorAlreadyAcquired:       
          return AICL_RET_RESOURCE_NOT_RELEASED;
      case cudaErrorMemoryAllocation:    
          return AICL_RET_MEM_ALLOC_FAILURE;
      case cudaErrorInvalidAddressSpace:    
          return AICL_RET_INVALID_MEM_TYPE;
      case cudaErrorMisalignedAddress:     
          return AICL_RET_MEMORY_ADDR_UNALIGNED;
      case cudaErrorNotMapped:      
          return AICL_RET_RESOURCE_NOT_MATCH;
      case cudaErrorInvalidResourceHandle:      
          return AICL_RET_INVALID_RESOURCE_HANDLE;
      case cudaErrorInitializationError:       
          return AICL_RET_NOT_INITIALIZED;
      case cudaErrorInvalidDeviceFunction:    
          return AICL_RET_INVALID_API;
      case cudaErrorLaunchTimeout:        
          return AICL_RET_API_TIMEOUT;
      default:
          return AICL_RET_ERROR;
  }
}


//  -- trans CUresult type
aiclRet errcod_trans(CUresult ret) {
  switch (ret)
  {
      case CUDA_SUCCESS:          
          return AICL_RET_SUCCESS;
      case CUDA_ERROR_NO_DEVICE:           
          return AICL_RET_INVALID_DEVICE;
      case CUDA_ERROR_INVALID_DEVICE:  
          return AICL_RET_INVALID_DEVICE_ID;
      case CUDA_ERROR_FILE_NOT_FOUND:     
          return AICL_RET_INVALID_FILE;
      case CUDA_ERROR_OUT_OF_MEMORY:       
          return AICL_RET_MEM_ALLOC_FAILURE;
      case CUDA_ERROR_INVALID_ADDRESS_SPACE:      
          return AICL_RET_INVALID_MEM_TYPE;
      case CUDA_ERROR_MISALIGNED_ADDRESS:      
          return AICL_RET_MEMORY_ADDR_UNALIGNED;
      case CUDA_ERROR_NOT_MAPPED:      
          return AICL_RET_RESOURCE_NOT_MATCH;
      case CUDA_ERROR_NOT_INITIALIZED:     
          return AICL_RET_NOT_INITIALIZED;
      case CUDA_ERROR_INVALID_CONTEXT:     
          return AICL_RET_INVALID_CONTEXT; 
      case CUDA_ERROR_INVALID_IMAGE:       
          return AICL_RET_INVALID_MODEL;
      case CUDA_ERROR_INVALID_HANDLE:      
          return AICL_RET_INVALID_HANDLE;
      case CUDA_ERROR_LAUNCH_TIMEOUT:      
          return AICL_RET_API_TIMEOUT;
      case CUDA_ERROR_INVALID_VALUE:    
          return AICL_RET_INVALID_PARAM;
      default:
          return AICL_RET_ERROR;
  }
}


aiclRet aiclrtGetRunMode(aiclrtRunMode *runMode) {
    cudaError_t ret = cudaGetRunMode(runMode);     
    return errcod_trans(ret);
}


aiclRet aiclInit(const char *configPath) {
    CUresult ret = cuInit(0);     
    return errcod_trans(ret);
}


aiclRet aiclrtSetDevice(int32_t deviceId) {
    cudaError_t ret = cudaSetDevice(deviceId);
    return errcod_trans(ret);
}


aiclRet aiclrtCreateContext(aiclrtContext *context, int32_t deviceId) {
    CUresult ret = cuCtxCreate((CUcontext*)context, 0, deviceId);      
    return errcod_trans(ret);
}


aiclRet aiclrtResetDevice(int32_t deviceId) {
    cudaError_t ret = cudaDeviceReset();
    return errcod_trans(ret);
}


aiclRet aiclFinalize() {
    CUresult ret = cuFinalize();       
    return errcod_trans(ret);
}


aiclRet aiclrtDestroyContext(aiclrtContext context) {
    CUresult ret = cuCtxDestroy((CUcontext)context);        //   typedef CUctx_st *  CUcontext
    return errcod_trans(ret);
}


aiclRet aiclrtGetCurrentContext(aiclrtContext *context) {
    CUresult ret = cuCtxGetCurrent((CUcontext*)context);
    return errcod_trans(ret);
}


aiclRet aiclrtMallocHost(void **hostPtr, size_t size) {
    cudaError_t ret = cudaMallocHost(hostPtr, size);
    return errcod_trans(ret);
}


aiclRet aiclrtMemcpy(void *dst, size_t destMax, const void *src,
                     size_t count, aiclrtMemcpyKind kind) {
    cudaError_t ret = cudaMemcpy(dst, src, count, (cudaMemcpyKind)kind);
    return errcod_trans(ret);
}


aiclRet aiclrtFreeHost(void *hostPtr) {
    cudaError_t ret = cudaFreeHost(hostPtr);
    return errcod_trans(ret);
}


aiclRet aiclrtGetMemInfo(aiclrtMemAttr attr, size_t *free, size_t *total) {
    cudaError_t ret = cudaMemGetInfo(free, total);
    return errcod_trans(ret);
}


aiclRet aiclrtMalloc(void **devPtr, size_t size,
                    aiclrtMemMallocPolicy policy) {     
    cudaError_t ret = cudaMalloc(devPtr, size);
    return errcod_trans(ret);
}


aiclRet aiclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count) {
    cudaError_t ret = cudaMemset(devPtr, value, count);
    return errcod_trans(ret);
}


aiclRet aiclrtFree(void *devPtr) {
    cudaError_t ret = cudaFree(devPtr);
    return errcod_trans(ret);
}


aiclRet aiclrtSetCurrentContext(aiclrtContext context) {
    CUresult ret = cuCtxSetCurrent((CUcontext)context);
    return errcod_trans(ret);
}


aiclRet aiclrtCreateStream(aiclrtStream *stream) {
    cudaError_t ret = cudaStreamCreate((cudaStream_t*)stream);
    return errcod_trans(ret);
}


aiclRet aiclrtDestroyStream(aiclrtStream stream) {
    cudaError_t ret = cudaStreamDestroy((cudaStream_t)stream);
    return errcod_trans(ret);
}


aiclRet aiclrtMemcpyAsync(void *dst, size_t destMax, const void *src,
                         size_t count, aiclrtMemcpyKind kind, aiclrtStream stream) {
    cudaError_t ret = cudaMemcpyAsync(dst, src, count, (cudaMemcpyKind)kind, (cudaStream_t)stream);
    return errcod_trans(ret);
}


aiclRet aiclrtSynchronizeStream(aiclrtStream stream) {
    cudaError_t ret = cudaStreamSynchronize(( cudaStream_t)stream);
    return errcod_trans(ret);
}


// -- 0
aiclDataBuffer *aiclCreateDataBuffer(void *data, size_t size) {
    cudaMallocHost(&data, size);
    return (aiclDataBuffer *)data;
}


// -- 0
aiclRet aiclDestroyDataBuffer(const aiclDataBuffer *dataBuffer) {
    void *data = const_cast<void *>((void *)dataBuffer);
    cudaError_t ret = cudaFreeHost(data);
    return errcod_trans(ret);
}

// -- 0
aiclTensorDesc *aiclCreateTensorDesc(aiclDataType dataType, int numDims,
                                     const int64_t *dims,  aiclFormat format) {
    // std::unique_ptr<DLTensor> from(new DLTensor);
    DLTensor *from = new DLTensor;
    DLDataType dtype;
    switch (dataType) {
    case 0:   // AICL_FLOAT
        dtype = {kDLFloat, 32, 1};
        break;
    case 1:    // AICL_FLOAT16
        dtype = {kDLFloat, 16, 1};
        break;
    case 2:     // AICL_INT8
        dtype = {kDLInt, 8, 1};
        break;
    case 3:     // AICL_INT32
        dtype = {kDLInt, 32, 1};
        break;
    case 4:     // AICL_UINT8
        dtype = {kDLUInt, 8, 1};
        break;
    case 6:     // AICL_INT16
        dtype = {kDLInt, 16, 1};
        break;
    case 7:     // AICL_UINT16
        dtype = {kDLUInt, 16, 1};
        break;
    case 8:    // AICL_UINT32
        dtype = {kDLUInt, 32, 1};
        break;
    default:
        break;
    }
    from->dtype = dtype;
    from->ndim = static_cast<int>(numDims);
    from->shape = const_cast<int64_t *>(dims);
    // return (aiclTensorDesc *)from.get();
    return (aiclTensorDesc *)from;
}

// -- 0
size_t aiclGetTensorDescSize(const aiclTensorDesc *desc) {
    DLTensor *TDesc = const_cast<DLTensor*>((DLTensor*)desc);
    return tvm::runtime::GetDataSize(*TDesc);
}

// -- 0
size_t aiclGetTensorDescNumDims(const aiclTensorDesc *desc) {
    DLTensor *TDesc = const_cast<DLTensor*>((DLTensor*)desc);
    return TDesc->ndim;
}

// -- 0
aiclRet aiclGetTensorDescDim(const aiclTensorDesc *desc, size_t index, int64_t *dimSize) {
    DLTensor *TDesc = const_cast<DLTensor*>((DLTensor*)desc);
    return TDesc->shape[index];
}

// -- 0
size_t aiclGetTensorDescElementCount(const aiclTensorDesc *desc) {
    DLTensor *TDesc = const_cast<DLTensor*>((DLTensor*)desc);
    size_t count = 1;
    for (int n = 0; n < TDesc->ndim; n++)
    {
        count *= TDesc->shape[n];
    }
    return count;
}


// -- 0
aiclDataType aiclGetTensorDescType(const aiclTensorDesc *desc) {
    DLTensor *TDesc = const_cast<DLTensor*>((DLTensor*)desc);
    switch ((int)TDesc->dtype.code)
    {
    case 0:
        if((int)TDesc->dtype.bits == 8){
            return AICL_INT8;
        } else if ((int)TDesc->dtype.bits == 16){
            return AICL_INT16;
        } else {
            return AICL_INT32;
        }
        break;
    case 1:
        if((int)TDesc->dtype.bits == 8){
            return AICL_UINT8;
        } else if ((int)TDesc->dtype.bits == 16){
            return AICL_UINT16;
        } else {
            return AICL_UINT32;
        }
        break;
    case 2:
        if((int)TDesc->dtype.bits == 16){
            return AICL_FLOAT16;
        } else {
            return AICL_FLOAT;
        }
        break;
    case 4:
        return AICL_BF16;
        break;
    default:
        return AICL_DT_UNDEFINED;
        break;
    }
}

// -- 0
aiclFormat aiclGetTensorDescFormat(const aiclTensorDesc *desc) {
    return AICL_FORMAT_ND;
}

// -- 0
void aiclDestroyTensorDesc(const aiclTensorDesc *desc) {
    // aclDestroyTensorDesc((aclTensorDesc *)desc);
    DLTensor *from = const_cast<DLTensor *>((DLTensor *)desc);
    // delete from;
    return;
}

// -- 0
aiclRet aiclopCompile(const char *opType, int numInputs,
                      const aiclTensorDesc *const inputDesc[],
                      int numOutputs,
                      const aiclTensorDesc *const outputDesc[],
                      const aiclopAttr *attr,
                      aiclopEngineType engineType,
                      aiclopCompileType compileFlag, const char *opPath) {
    
    std::string in_str = "inputs = [ {";
    std::vector<std::string> shape_str;
    std::vector<std::string> input_name;

    for (int i = 0; i < numInputs; i++)
    {
        std::string in_k = "'in_" + std::to_string(i) + "' : ";
        std::string tvm_arr = "tvm.nd.array(np.random.uniform(-1, 1, (";
        std::string tmp = in_k + tvm_arr;
        int numD = aiclGetTensorDescNumDims(inputDesc[i]);
        std::string shape_k = "shape=(";
        for (int n = 0; n < numD; n++)
        {
            int64_t dimSize;
            int ind = aiclGetTensorDescDim(inputDesc[i], n, &dimSize);
            if (n < (numD - 1 )){
                tmp += std::to_string(ind) + ",";
                shape_k += std::to_string(ind) + ",";
            }
            else
            {
                tmp += std::to_string(ind) + "))), ";
                shape_k += std::to_string(ind) + ")";
            }
        }
        shape_str.emplace_back(shape_k);
        std::string t_name = "in_" + std::to_string(i);
        input_name.emplace_back(t_name);
        in_str += tmp;
        if (i == (numInputs-1)){
            in_str += "}]";
        }
    }

    Py_Initialize(); //  初始化 python 接口
    PyRun_SimpleString("import numpy as np");
    PyRun_SimpleString("import tvm");
    PyRun_SimpleString("from tvm import relay");
    PyRun_SimpleString("from tvm.contrib import graph_executor");

    PyRun_SimpleString(in_str.c_str());
    PyRun_SimpleString("t_iter = iter(inputs[0])");

    std::string optype(opType);
    optype[0] = std::tolower(optype[0]);
    std::string func_str = "func = relay." + optype + "(";
    for (int i = 0; i < input_name.size(); i++){
        std::string input_str = input_name[i] + " = relay.var(next(t_iter), ";
        input_str = input_str + shape_str[i] + ", dtype='int32')";
        std::cout << input_str << std::endl;
        PyRun_SimpleString(input_str.c_str());

        if (i == (input_name.size() - 1)){
            func_str += input_name[i] + ')';
        } else {
            func_str += input_name[i] + ',';
        }  
    }

    PyRun_SimpleString(func_str.c_str());
    PyRun_SimpleString("mod = tvm.IRModule.from_expr(func)");
    PyRun_SimpleString("engine_cpu = relay.build(mod, target='llvm', params=None)");
    PyRun_SimpleString("engine_gpu = relay.build(mod, target='iluvatar', params=None)");
    PyRun_SimpleString("engine_cpu.export_library('./cpu_add.so')");
    int result = PyRun_SimpleString("engine_gpu.export_library('./gpu_add.so')");
    if (result != 0){
        return AICL_RET_ERROR;
    }
    else
    {
        return AICL_RET_SUCCESS;
    }
    Py_Finalize();

}




// -- 0
aiclRet aiclopExecute(const char *opType,int numInputs,
                        aiclTensorDesc *inputDesc[],
                        aiclDataBuffer *inputs[],
                        int numOutputs,
                        aiclTensorDesc *outputDesc[],
                        aiclDataBuffer *outputs[],
                        aiclopAttr *attr,
                        aiclrtStream stream) {
    
    DLDevice dev{kDLILUVATAR, 0};
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("./gpu_add.so");
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
    tvm::runtime::PackedFunc get_input = gmod.GetFunction("get_input");
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");
    std::vector<tvm::runtime::NDArray> input_vec;

    for (int i = 0; i < numInputs; i++){
        tvm::runtime::NDArray input_ = get_input(i);
        tvm::runtime::ShapeTuple input_shape = input_.Shape();
        tvm::runtime::DataType input_type = input_.DataType();
        auto input = tvm::runtime::NDArray::Empty(input_shape, input_type, dev);
        input_vec.emplace_back(input);
    }

    for (int n = 0; n < input_vec.size(); n++){
        int input_size = tvm::runtime::GetDataSize(*(input_vec[n].operator->()));
        std::cout << input_size << std::endl;
        input_vec[n].CopyFromBytes((int *)inputs[n], input_size);
    }

    std::cout << ((int *)inputs[0])[1] << std::endl;

    for (int i = 0; i < numInputs; i++){
        std::string in_str = "in_" + std::to_string(i);
        set_input(in_str.c_str(), input_vec[i]);
    }

    run();

    tvm::runtime::NDArray Y = get_output(0);
    // LOG(INFO) << Y.DataType();
    Y.CopyToBytes((void*)outputs[0], 8 * 16 * sizeof(float));

    // LOG(INFO) << 11111;

    return AICL_RET_SUCCESS;

}

// -- 0
aiclopAttr *aiclopCreateAttr() {

    tvm::Attrs *attr = new tvm::Attrs;
    *attr = tvm::Attrs();
    return (aiclopAttr *)attr;
}

// -- 0
void aiclopDestroyAttr(const aiclopAttr *attr) {
    // delete (tvm::Attrs *)attr;
    return;
}