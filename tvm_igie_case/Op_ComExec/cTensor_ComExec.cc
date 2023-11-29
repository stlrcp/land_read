#include </usr/local/include/python3.7m/Python.h>
#include "aicl.h"
#include <vector>
#include <iostream>
#include <dmlc/logging.h>
#include <tvm/te/operation.h>
#include <cuda_runtime.h>
#include <cuda.h>


aiclRet aiclrtCreateStream(aiclrtStream *stream) {
    cudaError_t ret = cudaStreamCreate((cudaStream_t*)stream);
    return 0;
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

size_t aiclGetTensorDescNumDims(const aiclTensorDesc *desc) {
    DLTensor *TDesc = desc->dtensor;
    return TDesc->ndim;
}

aiclRet aiclGetTensorDescDim(const aiclTensorDesc *desc, size_t index, int64_t *dimSize) {
    DLTensor *TDesc = desc->dtensor;
    return TDesc->shape[index];
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
    // aclDestroyTensorDesc((aclTensorDesc *)desc);
    DLTensor *from = desc->dtensor;
    delete from;
    delete desc;
    return;
}


int main(){

    std::vector<int64_t> shape{8, 16};
    std::string opType = "Add";
    aiclopAttr *opAttr;
    aiclDataType dataType = AICL_INT32;
    std::cout << "dataType size = " << sizeof(dataType) << std::endl;
    std::cout << "AICL_INT32 size = " << sizeof(AICL_INT32) << std::endl;
    aiclFormat format = AICL_FORMAT_ND;

    aiclTensorDesc *desc1 = aiclCreateTensorDesc(dataType, shape.size(), shape.data(), format);

    std::cout << desc1 << std::endl;
    size_t numD =  aiclGetTensorDescNumDims(desc1);
    std::cout << "numD = " << numD << std::endl;

    aiclFormat fmat = aiclGetTensorDescFormat(desc1);
    std::cout << "fmat = " << fmat << std::endl;

    aiclDestroyTensorDesc(desc1);

    return 0;
}