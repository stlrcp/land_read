/**
* @file operator_desc.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
// #include "common.h"
#include "operator_desc.h"
#include <dmlc/logging.h>
#include <tvm/te/operation.h>

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

using namespace std;

OperatorDesc::OperatorDesc(std::string opType) : opType(std::move(opType))
{
    opAttr = aiclopCreateAttr();
}

OperatorDesc::~OperatorDesc()
{
    for (auto *desc : inputDesc) {
        aiclDestroyTensorDesc(desc);
    }

    for (auto *desc : outputDesc) {
        aiclDestroyTensorDesc(desc);
    }

    aiclopDestroyAttr(opAttr);
}


OperatorDesc &OperatorDesc::AddInputTensorDesc(aiclDataType dataType,
                                               int numDims,
                                               const int64_t *dims,
                                               aiclFormat format)
{
    aiclTensorDesc *desc = aiclCreateTensorDesc(dataType, numDims, dims, format);
    
    DLTensor *fromA = (DLTensor *)desc;
    LOG(INFO) << "size = " << tvm::runtime::GetDataSize(*fromA);
    
    if (desc == nullptr) {
        ERROR_LOG("create tensor failed");
        return *this;
    }
    inputDesc.emplace_back(desc);
    return *this;
}

OperatorDesc &OperatorDesc::AddOutputTensorDesc(aiclDataType dataType,
                                                int numDims,
                                                const int64_t *dims,
                                                aiclFormat format)
{
    aiclTensorDesc *desc = aiclCreateTensorDesc(dataType, numDims, dims, format);
    if (desc == nullptr) {
        ERROR_LOG("create tensor failed");
        return *this;
    }

    outputDesc.emplace_back(desc);
    return *this;
}
