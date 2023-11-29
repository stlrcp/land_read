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
#include <cuda_fp16.h>
#include </usr/local/include/python3.7m/Python.h>

using namespace std;

cudaError_t cudaGetRunMode(aiclrtRunMode * runMode){
    *runMode = AICL_HOST;
    return cudaSuccess;
}

CUresult cuFinalize(){
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
    // cudaError_t ret = cudaDeviceReset();
    // return errcod_trans(ret);
    return AICL_RET_SUCCESS;
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

tvm::runtime::Module op_model;
aiclRet aiclopSetModelDir(const char *modelDir) {
    op_model = tvm::runtime::Module::LoadFromFile(modelDir);
    if (!(op_model.defined())){
        return AICL_RET_OP_LOAD_FAILED;
    }
    return AICL_RET_SUCCESS;
}

aiclDataBuffer *aiclCreateDataBuffer(void *data, size_t size) {
    return (aiclDataBuffer *)data;
}

aiclRet aiclDestroyDataBuffer(const aiclDataBuffer *dataBuffer) {
    void *data = const_cast<void *>((void *)dataBuffer);
    cudaError_t ret = cudaFree(data);
    return errcod_trans(ret);
}

struct aiclTensorDesc {
    DLTensor *dtensor;
    aiclFormat format;
};

aiclTensorDesc *aiclCreateTensorDesc(aiclDataType dataType, int numDims,
                                     const int64_t *dims,  aiclFormat format) {
    aiclTensorDesc *tmpTensor = new aiclTensorDesc;
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
    tmpTensor->dtensor = from;
    tmpTensor->format = format;
    return tmpTensor;
}


size_t aiclGetTensorDescSize(const aiclTensorDesc *desc) {
    DLTensor *TDesc = desc->dtensor;
    return tvm::runtime::GetDataSize(*TDesc);
}


size_t aiclGetTensorDescNumDims(const aiclTensorDesc *desc) {
    DLTensor *TDesc = desc->dtensor;
    return TDesc->ndim;
}


aiclRet aiclGetTensorDescDim(const aiclTensorDesc *desc, size_t index, int64_t *dimSize) {
    DLTensor *TDesc = desc->dtensor;
    return TDesc->shape[index];
}


size_t aiclGetTensorDescElementCount(const aiclTensorDesc *desc) {
    DLTensor *TDesc = desc->dtensor;
    size_t count = 1;
    for (int n = 0; n < TDesc->ndim; n++)
    {
        count *= TDesc->shape[n];
    }
    return count;
}



aiclDataType aiclGetTensorDescType(const aiclTensorDesc *desc) {
    DLTensor *TDesc = desc->dtensor;
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


aiclFormat aiclGetTensorDescFormat(const aiclTensorDesc *desc) {
    return desc->format;
}


void aiclDestroyTensorDesc(const aiclTensorDesc *desc) {
    DLTensor *from = desc->dtensor;
    if (from)
        delete from;
    if (desc)
        delete desc;
}


aiclRet aiclopCompile(const char *opType, int numInputs,
                      const aiclTensorDesc *const inputDesc[],
                      int numOutputs,
                      const aiclTensorDesc *const outputDesc[],
                      const aiclopAttr *attr,
                      aiclopEngineType engineType,
                      aiclopCompileType compileFlag, const char *opPath) {

    std::vector<std::string> input_str;
    std::vector<std::string> input_name;
    for (int i = 0; i < numInputs; i++){
        std::string inp_k = "input" + std::to_string(i);
        std::string dtype_k = "dtype='";
        int numD = aiclGetTensorDescNumDims(inputDesc[i]);
        aiclDataType type_index = aiclGetTensorDescType(inputDesc[i]);
        switch ((int)type_index){
            case 0:
                dtype_k += "float32'";
                break;
            case 1:
                dtype_k += "float16'";
                break;
            case 2:
                dtype_k += "int8'";
                break;
            case 3:
                dtype_k += "int32'";
                break;
            case 4:
                dtype_k += "uint8'";
                break;
            default:
                dtype_k += "'";
                break;
        }

        std::string tmp = inp_k + " = relay.var('"+ inp_k + "',";

        std::string shape_k = "shape=(";
        for (int n = 0; n < numD; n++)
        {
            int64_t dimSize;
            int ind = aiclGetTensorDescDim(inputDesc[i], n, &dimSize);
            if (n < (numD - 1 )){
                tmp += " shape=(" + std::to_string(ind) + ",";
            }
            else
            {
                tmp += std::to_string(ind) + "), " + dtype_k + ")";
            }
        }
        input_str.emplace_back(tmp);
        input_name.emplace_back(inp_k);
    }

    Py_Initialize(); 
    PyRun_SimpleString("import numpy as np");
    PyRun_SimpleString("import tvm");
    PyRun_SimpleString("from tvm import relay");
    PyRun_SimpleString("from tvm.contrib import graph_executor");
    for (auto tmp_str : input_str){
        PyRun_SimpleString(tmp_str.c_str());
    }
    std::string optype(opType);
    optype[0] = std::tolower(optype[0]);
    std::string func_str = "func = relay." + optype + "(";

    for (int i = 0; i < input_name.size(); i++){
        if (i == (input_name.size() - 1)){
            func_str += input_name[i] + ')';
        } else {
            func_str += input_name[i] + ',';
        }  
    }

    PyRun_SimpleString(func_str.c_str());
    PyRun_SimpleString("mod = tvm.IRModule.from_expr(func)");
    PyRun_SimpleString("runmod = relay.build(mod, target='iluvatar', params=None)");
    std::string export_engine = "runmod.export_library('" + optype + ".so')";
    int result = PyRun_SimpleString(export_engine.c_str());
    Py_Finalize();

    std::string run_mod_path = "./" + optype + ".so";
    op_model = tvm::runtime::Module::LoadFromFile(run_mod_path.c_str());

    if (result != 0){
        return AICL_RET_ERROR;
    }
    else
    {
        return AICL_RET_SUCCESS;
    }
}

aiclRet aiclopExecute(const char *opType,int numInputs,
                        aiclTensorDesc *inputDesc[],
                        aiclDataBuffer *inputs[],
                        int numOutputs,
                        aiclTensorDesc *outputDesc[],
                        aiclDataBuffer *outputs[],
                        aiclopAttr *attr,
                        aiclrtStream stream) {
    
    DLDevice dev{kDLILUVATAR, 0};

    if (!(op_model.defined())){
        return AICL_RET_OP_LOAD_FAILED;
    }

    // tvm::runtime::Module op_model = tvm::runtime::Module::LoadFromFile(run_mod_path.c_str());
    tvm::runtime::Module gmod = op_model.GetFunction("default")(dev);
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
        std::string in_str = "input" + std::to_string(i);
        set_input(in_str.c_str(), input_vec[i]);
    }

    run();

    tvm::runtime::NDArray output0 = get_output(0);
    size_t output_size = tvm::runtime::GetDataSize(*(output0.operator->()));

    output0.CopyToBytes((void*)outputs[0], output_size);

    return AICL_RET_SUCCESS;
}


aiclopAttr *aiclopCreateAttr() {
    std::map<std::string, std::vector<int>> *map_ptr = new std::map<std::string, std::vector<int>>;
    return (aiclopAttr *)map_ptr;
}


void aiclopDestroyAttr(const aiclopAttr *attr) {
    std::map<std::string, std::vector<int>> * map_ptr = (std::map<std::string, std::vector<int>> *)(attr);
    if (map_ptr)
        delete map_ptr;
}

aiclRet aiclopSetAttrListInt(aiclopAttr *attr, const char *attrName, int numValues,
                             const int64_t *values) {
    std::map<std::string, std::vector<int>> * map_ptr = (std::map<std::string, std::vector<int>> *)(attr);

    if (numValues <= 0){
        return AICL_RET_ERROR;
    } else {
        std::vector<int> intList;
        for (int n = 0; n < numValues; n++){
            intList.emplace_back(values[n]);
        }
        (*map_ptr)[attrName] = intList;
    }
    return AICL_RET_SUCCESS;
}

aiclRet aiclopCompileAndExecute(const char *opType,
    int numInputs, aiclTensorDesc *inputDesc[], aiclDataBuffer *inputs[],
    int numOutputs, aiclTensorDesc *outputDesc[], aiclDataBuffer *outputs[],
    aiclopAttr *attr, aiclopEngineType engineType, aiclopCompileType compileFlag,
    const char *opPath, aiclrtStream stream) {
    
    std::string optype(opType);
    for (auto &i : optype){
        i = tolower(i);
    }
    std::vector<std::string> input_str;
    std::vector<std::string> input_name;
    std::vector<std::string> shape_vec;
    std::vector<std::string> fmat_vec;
    std::vector<std::string> dtype_vec;
    if (optype == "conv2d"){
        for (int i = 0; i < numInputs; i++){
            int numD = aiclGetTensorDescNumDims(inputDesc[i]);
            int fmat = aiclGetTensorDescFormat(inputDesc[i]);
            std::string dt_str;
            switch (fmat)
            {
            case 0:   
                dt_str = "NCHW";
                break;
            case 1:    
                dt_str = "NHWC";
                break;
            default:
                dt_str = "default";
                break;
            }
            fmat_vec.emplace_back(dt_str);
            if (numD)
            {
                std::string shape_k = "shape=(";
                for (int n = 0; n < numD; n++)
                {
                    int64_t dimSize;
                    int ind = aiclGetTensorDescDim(inputDesc[i], n, &dimSize);
                    shape_k += std::to_string(ind) + ",";
                }
                shape_k.pop_back();
                shape_k += ")";
                shape_vec.emplace_back(shape_k);
            }
            std::string dtype_k = "dtype='";
            aiclDataType type_index = aiclGetTensorDescType(inputDesc[i]);
            switch ((int)type_index){
                case 0:
                    dtype_k += "float32'";
                    break;
                case 1:
                    dtype_k += "float16'";
                    break;
                case 2:
                    dtype_k += "int8'";
                    break;
                case 3:
                    dtype_k += "int32'";
                    break;
                case 4:
                    dtype_k += "uint8'";
                    break;
                default:
                    dtype_k += "'";
                    break;
            }
            dtype_vec.emplace_back(dtype_k);
        }
        std::string inp_str = "input=relay.var('input', " + shape_vec[0] + ", " + dtype_vec[0] + ")";
        std::string wei_str = "weight=relay.var('weight', " + shape_vec[1] + ", " + dtype_vec[1] + ")";
        input_str.emplace_back(inp_str);
        input_str.emplace_back(wei_str);
        input_name.emplace_back("input");
        input_name.emplace_back("weight");
    }

    Py_Initialize();   
    PyRun_SimpleString("import tvm");
    PyRun_SimpleString("from tvm import relay");
    PyRun_SimpleString("import numpy as np");
    PyRun_SimpleString("from tvm.contrib import graph_executor");

    for (auto tmp_str : input_str){
        PyRun_SimpleString(tmp_str.c_str());
    }

    std::string func_str = "func = relay.nn.conv2d(" + input_name[0] + ", " + input_name[1];
    std::string weight_fmat = fmat_vec[1];
    int weight_numD = aiclGetTensorDescNumDims(inputDesc[1]);
    std::string kn_size = "kernel_size=(";
    for (int i = 0; i < weight_numD; i++){
        int64_t dimSize;
        int nDim = aiclGetTensorDescDim(inputDesc[1], i, &dimSize);
        if (weight_fmat[i] == 'N')
            func_str += ", channels=" + std::to_string(nDim) + ", ";
        if (weight_fmat[i] == 'H')
            kn_size += std::to_string(nDim) + ",";
        if (weight_fmat[i] == 'W')
            kn_size += std::to_string(nDim) + "), ";
    }

    std::map<std::string, std::vector<int>> * map_ptr = (std::map<std::string, std::vector<int>> *)(attr);
    std::string pad_str = "padding=(" + std::to_string((*map_ptr)["pads"][0]) + ", " + std::to_string((*map_ptr)["pads"][2]) + "), ";
    std::string stride_str = "strides=(" + std::to_string((*map_ptr)["strides"][0]) + ", " + std::to_string((*map_ptr)["strides"][2]) + "), ";
    std::string dilation_str = "dilation=(" + std::to_string((*map_ptr)["dilations"][0]) + ", " + std::to_string((*map_ptr)["dilations"][2]) + "),";
    func_str += kn_size + pad_str + stride_str + dilation_str;
    std::replace(fmat_vec[1].begin(), fmat_vec[1].end(), 'N', 'O');
    std::replace(fmat_vec[1].begin(), fmat_vec[1].end(), 'C', 'I');
    func_str += "data_layout='" + fmat_vec[0] + "', ";
    func_str += "kernel_layout='" + fmat_vec[1] + "',)";

    PyRun_SimpleString(func_str.c_str());
    PyRun_SimpleString("mod = tvm.IRModule.from_expr(func)");
    PyRun_SimpleString("desired_layout = {'nn.conv2d': ['NHWC', 'HWIO']}");
    PyRun_SimpleString("with tvm.transform.PassContext(opt_level=3):\n\t\
                        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layout)])\n\t\
                        mod = seq(mod)");
    PyRun_SimpleString("target = tvm.target.iluvatar(model='MR', options='-libs=cudnn')");
    PyRun_SimpleString("run_mod = relay.build(mod, target=target, params=None)");
    std::string export_engine = "run_mod.export_library('" + optype + ".so')";
    PyRun_SimpleString(export_engine.c_str());
    Py_Finalize();

    std::string run_mod_path = "./" + optype + ".so";
    std::cout << "run_mod_path = " << run_mod_path << std::endl;

    DLDevice dev{kDLILUVATAR, 0};
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(run_mod_path.c_str());
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
    tvm::runtime::PackedFunc get_input = gmod.GetFunction("get_input");
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");

    tvm::runtime::NDArray get_input0 = get_input(0);
    tvm::runtime::ShapeTuple input0_shape = get_input0.Shape();
    tvm::runtime::DataType input0_type = get_input0.DataType();
    auto input0 = tvm::runtime::NDArray::Empty(input0_shape, input0_type, dev);
    tvm::runtime::NDArray get_input1 = get_input(1);
    tvm::runtime::ShapeTuple input1_shape = get_input1.Shape();
    tvm::runtime::DataType input1_type = get_input1.DataType();
    auto input1 = tvm::runtime::NDArray::Empty(input1_shape, input1_type, dev);

    auto input0_data = static_cast<void *>(input0->data);
    auto input1_data = static_cast<void *>(input1->data);

    std::cout << tvm::runtime::GetDataSize(*(input0.operator->())) << std::endl;
    std::cout << tvm::runtime::GetDataSize(*(input1.operator->())) << std::endl;

    cudaMemcpy(input0_data, inputs[0], tvm::runtime::GetDataSize(*(input0.operator->())), cudaMemcpyDeviceToDevice);
    cudaMemcpy(input1_data, inputs[1], tvm::runtime::GetDataSize(*(input1.operator->())), cudaMemcpyDeviceToDevice);

    set_input(input_name[0].c_str(), input0);
    set_input(input_name[1].c_str(), input1);
    run();

    tvm::runtime::NDArray output0 = get_output(0);
    LOG(INFO) << output0.DataType();

    auto output0_data = static_cast<void *>(output0->data);
    cudaMemcpy(outputs[0], output0_data, tvm::runtime::GetDataSize(*(output0.operator->())), cudaMemcpyDeviceToDevice);

    return AICL_RET_SUCCESS;
}

aiclmdlDataset *aiclmdlCreateDataset() {
    std::vector<aiclDataBuffer *> *dataDesc = new std::vector<aiclDataBuffer *>;
    return (aiclmdlDataset *)dataDesc;
}

aiclRet aiclmdlAddDatasetBuffer(aiclmdlDataset *dataset, aiclDataBuffer *dataBuffer) {
    std::vector<aiclDataBuffer *> *dataDesc;
    dataDesc = (std::vector<aiclDataBuffer *> *)dataset;
    dataDesc->emplace_back(dataBuffer);
}

tvm::runtime::Module mod_factory;
aiclRet aiclmdlLoadFromFile(const char *modelPath, uint32_t *modelId) {
    mod_factory = tvm::runtime::Module::LoadFromFile(modelPath);
    return AICL_RET_SUCCESS;
}

aiclmdlDesc *aiclmdlCreateDesc() {
    DLDevice dev{kDLILUVATAR, 0};
    tvm::runtime::Module *gmod = new tvm::runtime::Module;
    *gmod = mod_factory.GetFunction("default")(dev);
    return (aiclmdlDesc *)gmod;
}

aiclRet aiclmdlGetDesc(aiclmdlDesc *modelDesc, uint32_t modelId) {
    return AICL_RET_SUCCESS;
}

size_t aiclmdlGetOutputSizeByIndex(aiclmdlDesc *modelDesc, size_t index) {
    tvm::runtime::Module *gmod = (tvm::runtime::Module *)modelDesc;
    tvm::runtime::PackedFunc get_output = gmod->GetFunction("get_output");
    tvm::runtime::NDArray output_ = get_output(0);

    return tvm::runtime::GetDataSize(*(output_.operator->()));
}

aiclRet aiclmdlExecute(uint32_t modelId, const aiclmdlDataset *input, aiclmdlDataset *output) {
    DLDevice dev{kDLILUVATAR, 0};
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input_zero_copy");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc get_input = gmod.GetFunction("get_input");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");

    // get num of input
    tvm::runtime::PackedFunc get_num_inputs = gmod.GetFunction("get_num_inputs");
    int input_num = get_num_inputs();

    // set_input all inputs
    for (int i = 0; i < input_num; i++){
        tvm::runtime::NDArray get_input_ = get_input(i);
        tvm::runtime::ShapeTuple input_shape = get_input_.Shape();
        tvm::runtime::DataType input_type = get_input_.DataType();
        auto input_ = tvm::runtime::NDArray::Empty(input_shape, input_type, dev);
        auto input_data = static_cast<void*>(input_->data);
        auto input_i = (void *)(((std::vector<aiclDataBuffer *> *)input)->at(i));
        size_t mem_size = tvm::runtime::GetDataSize(*(input_.operator->()));
        cudaMemcpy(input_data, input_i, mem_size, cudaMemcpyDeviceToDevice);
        set_input(i, input_);
    }

    // run model
    run();

    // get num of output
    tvm::runtime::PackedFunc get_num_outputs = gmod.GetFunction("get_num_outputs");
    int output_num = get_num_outputs();

    // get the output
    for (int n = 0; n < output_num; n++){
        tvm::runtime::NDArray get_output_ = get_output(n);
        auto output_data = static_cast<void*>(get_output_->data);
        size_t out_memsize = tvm::runtime::GetDataSize(*(get_output_.operator->()));
        auto output_n = (void *)(((std::vector<aiclDataBuffer *> *)output)->at(n));
        cudaMemcpy(output_n, output_data, out_memsize, cudaMemcpyDeviceToDevice);
    }

    return AICL_RET_SUCCESS;
}

aiclRet aiclmdlDestroyDesc(aiclmdlDesc *modelDesc) {
    tvm::runtime::Module *gmod = (tvm::runtime::Module *)modelDesc;
    if (gmod)
        delete gmod;
    return AICL_RET_SUCCESS;
}

aiclRet aiclmdlUnload(uint32_t modelId) {
    return AICL_RET_SUCCESS;
}

aiclRet aiclmdlDestroyDataset(const aiclmdlDataset *dataset) {
    std::vector<aiclDataBuffer *> *dataDesc = (std::vector<aiclDataBuffer *> *)dataset;
    if (dataDesc)
        delete dataDesc;
    return AICL_RET_SUCCESS;
}