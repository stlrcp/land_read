/**
* @file op_runner.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "op_runner.h"

#include <limits>
// #include "acl/acl_op_compiler.h"
#include "common.h"

using namespace std;

extern bool g_isDevice;

OpRunner::OpRunner(OperatorDesc *opDesc) : opDesc_(opDesc)
{
    numInputs_ = opDesc->inputDesc.size();
    numOutputs_ = opDesc->outputDesc.size();
}

OpRunner::~OpRunner()
{
    for (size_t i = 0; i < numInputs_; ++i) {
        (void)aiclDestroyDataBuffer(inputBuffers_[i]);
        (void)aiclrtFree(devInputs_[i]);
        if (g_isDevice) {
            (void)aiclrtFree(hostInputs_[i]);
        } else {
            (void)aiclrtFreeHost(hostInputs_[i]);
        }
    }

    for (size_t i = 0; i < numOutputs_; ++i) {
        (void)aiclDestroyDataBuffer(outputBuffers_[i]);
        (void)aiclrtFree(devOutputs_[i]);
        if (g_isDevice) {
            (void)aiclrtFree(hostOutputs_[i]);
        } else {
            (void)aiclrtFreeHost(hostOutputs_[i]);
        }
    }
}

bool OpRunner::Init()
{
    for (size_t i = 0; i < numInputs_; ++i) {
        auto size = GetInputSize(i);
        void *devMem = nullptr;
        if (aiclrtMalloc(&devMem, size, AICL_MEM_MALLOC_NORMAL_ONLY) != AICL_RET_SUCCESS) {
            ERROR_LOG("Malloc device memory for input[%zu] failed", i);
            return false;
        }
        devInputs_.emplace_back(devMem);
        inputBuffers_.emplace_back(aiclCreateDataBuffer(devMem, size));

        void *hostMem = nullptr;
        if (g_isDevice) {
            if (aiclrtMalloc(&hostMem, size, AICL_MEM_MALLOC_NORMAL_ONLY) != AICL_RET_SUCCESS) {
                ERROR_LOG("Malloc device memory for input[%zu] failed", i);
                return false;
            }
        } else {
            if (aiclrtMallocHost(&hostMem, size) != AICL_RET_SUCCESS) {
                ERROR_LOG("Malloc device memory for input[%zu] failed", i);
                return false;
            }
        }
        if (hostMem == nullptr) {
            ERROR_LOG("Malloc memory for input[%zu] failed", i);
            return false;
        }
        hostInputs_.emplace_back(hostMem);
    }

    for (size_t i = 0; i < numOutputs_; ++i) {
        auto size = GetOutputSize(i);
        void *devMem = nullptr;
        if (aiclrtMalloc(&devMem, size, AICL_MEM_MALLOC_NORMAL_ONLY) != AICL_RET_SUCCESS) {
            ERROR_LOG("Malloc device memory for output[%zu] failed", i);
            return false;
        }
        devOutputs_.emplace_back(devMem);
        outputBuffers_.emplace_back(aiclCreateDataBuffer(devMem, size));

        void *hostOutput = nullptr;
        if (g_isDevice) {
            if (aiclrtMalloc(&hostOutput, size, AICL_MEM_MALLOC_NORMAL_ONLY) != AICL_RET_SUCCESS) {
                ERROR_LOG("Malloc device memory for output[%zu] failed", i);
                return false;
            }
        } else {
            if (aiclrtMallocHost(&hostOutput, size) != AICL_RET_SUCCESS) {
                ERROR_LOG("Malloc device memory for output[%zu] failed", i);
                return false;
            }
        }
        if (hostOutput == nullptr) {
            ERROR_LOG("Malloc host memory for output[%zu] failed", i);
            return false;
        }
        hostOutputs_.emplace_back(hostOutput);
    }

    return true;
}

const size_t OpRunner::NumInputs()
{
    return numInputs_;
}

const size_t OpRunner::NumOutputs()
{
    return numOutputs_;
}

const size_t OpRunner::GetInputSize(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }

    return aiclGetTensorDescSize(opDesc_->inputDesc[index]);
}

std::vector<int64_t> OpRunner::GetInputShape(size_t index) const
{
    std::vector<int64_t> ret;
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return ret;
    }

    auto desc = opDesc_->inputDesc[index];
    for (size_t i = 0; i < aiclGetTensorDescNumDims(desc); ++i) {
        int64_t dimSize;
        if (aiclGetTensorDescDim(desc, i, &dimSize) != AICL_RET_SUCCESS) {
            ERROR_LOG("get dims from tensor desc failed. dims index = %zu", i);
            ret.clear();
            return ret;
        }
        ret.emplace_back(dimSize);
    }

    return ret;
}

std::vector<int64_t> OpRunner::GetOutputShape(size_t index) const
{
    std::vector<int64_t> ret;
    if (index >= opDesc_->outputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return ret;
    }

    auto desc = opDesc_->outputDesc[index];
    for (size_t i = 0; i < aiclGetTensorDescNumDims(desc); ++i) {
        int64_t dimSize;
        if (aiclGetTensorDescDim(desc, i, &dimSize) != AICL_RET_SUCCESS) {
            ERROR_LOG("get dims from tensor desc failed. dims index = %zu", i);
            ret.clear();
            return ret;
        }
        ret.emplace_back(dimSize);
    }
    return ret;
}

size_t OpRunner::GetInputElementCount(size_t index) const
{
    if (index >= opDesc_->inputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }

    return aiclGetTensorDescElementCount(opDesc_->inputDesc[index]);
}

size_t OpRunner::GetOutputElementCount(size_t index) const
{
    if (index >= opDesc_->outputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }

    return aiclGetTensorDescElementCount(opDesc_->outputDesc[index]);
}

size_t OpRunner::GetOutputSize(size_t index) const
{
    if (index >= opDesc_->outputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }

    return aiclGetTensorDescSize(opDesc_->outputDesc[index]);
}

bool OpRunner::CompileStaticOp()
{
    auto ret = aiclopCompile(opDesc_->opType.c_str(),
                            numInputs_,
                            opDesc_->inputDesc.data(),
                            numOutputs_,
                            opDesc_->outputDesc.data(),
                            opDesc_->opAttr,
                            AICL_ENGINE_SYS,
                            AICL_COMPILE_SYS,
                            nullptr);
    if (ret != AICL_RET_SUCCESS) {
        ERROR_LOG("compile static op %s failed. errorCode is %d", opDesc_->opType.c_str(), static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("compile static op %s success", opDesc_->opType.c_str());
    return true;
}

bool OpRunner::CompileDynamicOp()
{
    std::vector<int64_t> shape = {-2};
    std::vector<aiclTensorDesc *> inputDesc;
    std::vector<aiclTensorDesc *> outputDesc;
    for (size_t i = 0; i < opDesc_->inputDesc.size(); ++i) {
        aiclDataType dataType = aiclGetTensorDescType(opDesc_->inputDesc[i]);
        aiclFormat format = aiclGetTensorDescFormat(opDesc_->inputDesc[i]);
        aiclTensorDesc *desc = aiclCreateTensorDesc(dataType, shape.size(), shape.data(), format);
        if (desc == nullptr) {
            return false;
        }
        inputDesc.emplace_back(desc);
    }
    for (size_t i = 0; i < opDesc_->outputDesc.size(); ++i) {
        aiclDataType dataType = aiclGetTensorDescType(opDesc_->outputDesc[i]);
        aiclFormat format = aiclGetTensorDescFormat(opDesc_->outputDesc[i]);
        aiclTensorDesc *desc = aiclCreateTensorDesc(dataType, shape.size(), shape.data(), format);
        if (desc == nullptr) {
            return false;
        }
        outputDesc.emplace_back(desc);
    }
    auto ret = aiclopCompile(opDesc_->opType.c_str(),
                            numInputs_,
                            inputDesc.data(),
                            numOutputs_,
                            outputDesc.data(),
                            opDesc_->opAttr,
                            AICL_ENGINE_SYS,
                            AICL_COMPILE_SYS,
                            nullptr);
    if (ret != AICL_RET_SUCCESS) {
        ERROR_LOG("compile dynamic op %s failed. errorCode is %d", opDesc_->opType.c_str(), static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("compile dynamic op %s success", opDesc_->opType.c_str());
    return true;
}

bool OpRunner::RunOp()
{
    for (size_t i = 0; i < numInputs_; ++i) {
        auto size = GetInputSize(i);
        aiclrtMemcpyKind kind = AICL_MEMCPY_HOST_TO_DEVICE;
        if (g_isDevice) {
            kind = AICL_MEMCPY_DEVICE_TO_DEVICE;
        }
        if (aiclrtMemcpy(devInputs_[i], size, hostInputs_[i], size, kind) != AICL_RET_SUCCESS) {
            ERROR_LOG("Copy input[%zu] failed", i);
            return false;
        }
        INFO_LOG("Copy input[%zu] success", i);
    }

    aiclrtStream stream = nullptr;
    if (aiclrtCreateStream(&stream) != AICL_RET_SUCCESS) {
        ERROR_LOG("Create stream failed");
        return false;
    }
    INFO_LOG("Create stream success");

    auto ret = aiclopExecute(opDesc_->opType.c_str(),
                            numInputs_,
                            opDesc_->inputDesc.data(),
                            inputBuffers_.data(),
                            numOutputs_,
                            opDesc_->outputDesc.data(),
                            outputBuffers_.data(),
                            opDesc_->opAttr,
                            stream);
    if (ret == AICL_RET_INVALID_OP_TYPE || ret == AICL_RET_INVALID_OP_INPUT ||
        ret == AICL_RET_INVALID_OP_OUTPUT || ret == AICL_RET_INVALID_OP_ATTR) {
        ERROR_LOG("[%s] op with the given description is not compiled. Please run atc first, errorCode is %d",
            opDesc_->opType.c_str(), static_cast<int32_t>(ret));
        (void)aiclrtDestroyStream(stream);
        return false;
    } else if (ret != AICL_RET_SUCCESS) {
        (void)aiclrtDestroyStream(stream);
        ERROR_LOG("Execute %s failed. errorCode is %d", opDesc_->opType.c_str(), static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("Execute %s success", opDesc_->opType.c_str());

    if (aiclrtSynchronizeStream(stream) != AICL_RET_SUCCESS) {
        ERROR_LOG("Synchronize stream failed");
        (void)aiclrtDestroyStream(stream);
        return false;
    }
    INFO_LOG("Synchronize stream success");

    for (size_t i = 0; i < numOutputs_; ++i) {
        auto size = GetOutputSize(i);
        aiclrtMemcpyKind kind = AICL_MEMCPY_DEVICE_TO_HOST;
        if (g_isDevice) {
            kind = AICL_MEMCPY_DEVICE_TO_DEVICE;
        }
        if (aiclrtMemcpy(hostOutputs_[i], size, devOutputs_[i], size, kind) != AICL_RET_SUCCESS) {
            INFO_LOG("Copy output[%zu] success", i);
            (void)aiclrtDestroyStream(stream);
            return false;
        }
        INFO_LOG("Copy output[%zu] success", i);
    }

    (void)aiclrtDestroyStream(stream);
    return true;
}


template<typename T>
void DoPrintData(const T *data, size_t count, size_t elementsPerRow)
{
    for (size_t i = 0; i < count; ++i) {
        std::cout << std::setw(10) << data[i];
        if (i % elementsPerRow == elementsPerRow - 1) {
            std::cout << std::endl;
        }
    }
}

void DoPrintFp16Data(const aiclFloat16 *data, size_t count, size_t elementsPerRow)
{
    for (size_t i = 0; i < count; ++i) {
        // std::cout << std::setw(10) << std::setprecision(4) << aiclFloat16ToFloat(data[i]);
        if (i % elementsPerRow == elementsPerRow - 1) {
            std::cout << std::endl;
        }
    }
}

void PrintData(const void *data, size_t count, aiclDataType dataType, size_t elementsPerRow)
{
    if (data == nullptr) {
        ERROR_LOG("Print data failed. data is nullptr");
        return;
    }

    switch (dataType) {
        case AICL_BOOL:
            DoPrintData(reinterpret_cast<const bool *>(data), count, elementsPerRow);
            break;
        case AICL_INT8:
            DoPrintData(reinterpret_cast<const int8_t *>(data), count, elementsPerRow);
            break;
        case AICL_UINT8:
            DoPrintData(reinterpret_cast<const uint8_t *>(data), count, elementsPerRow);
            break;
        case AICL_INT16:
            DoPrintData(reinterpret_cast<const int16_t *>(data), count, elementsPerRow);
            break;
        case AICL_UINT16:
            DoPrintData(reinterpret_cast<const uint16_t *>(data), count, elementsPerRow);
            break;
        case AICL_INT32:
            DoPrintData(reinterpret_cast<const int32_t *>(data), count, elementsPerRow);
            break;
        case AICL_UINT32:
            DoPrintData(reinterpret_cast<const uint32_t *>(data), count, elementsPerRow);
            break;
        case AICL_INT64:
            DoPrintData(reinterpret_cast<const int64_t *>(data), count, elementsPerRow);
            break;
        case AICL_UINT64:
            DoPrintData(reinterpret_cast<const uint64_t *>(data), count, elementsPerRow);
            break;
        case AICL_FLOAT16:
            DoPrintFp16Data(reinterpret_cast<const aiclFloat16 *>(data), count, elementsPerRow);
            break;
        case AICL_FLOAT:
            DoPrintData(reinterpret_cast<const float *>(data), count, elementsPerRow);
            break;
        case AICL_DOUBLE:
            DoPrintData(reinterpret_cast<const double *>(data), count, elementsPerRow);
            break;
        default:
            ERROR_LOG("Unsupported type: %d", dataType);
    }
}

void OpRunner::PrintInput(size_t index, size_t numElementsPerRow)
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numInputs_);
        return;
    }

    auto desc = opDesc_->inputDesc[index];
    PrintData(hostInputs_[index], GetInputElementCount(index), aiclGetTensorDescType(desc), numElementsPerRow);
}

void OpRunner::PrintOutput(size_t index, size_t numElementsPerRow)
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return;
    }

    auto desc = opDesc_->outputDesc[index];
    PrintData(hostOutputs_[index], GetOutputElementCount(index), aiclGetTensorDescType(desc), numElementsPerRow);
}
