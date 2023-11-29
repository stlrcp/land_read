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

using namespace std;

typedef enum cudaRunMode {
    CUDA,
    CPU
} cudaRunMode;

// 自我实现的 cudaGetRunMode
cudaError_t cudaGetRunMode(cudaRunMode * runMode){
    if (*runMode == CUDA || *runMode == CPU){
        return cudaSuccess;
    } else {
        return cudaErrorUnknown;      //   返回值待定
    }
}

// 自我实现的 cudaGetRunMode
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
    cudaError_t ret = cudaGetRunMode((cudaRunMode *)runMode);     
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