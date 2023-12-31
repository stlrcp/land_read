/**
* @file operator_desc.h
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef OPERATOR_DESC_H
#define OPERATOR_DESC_H

#include <string>
#include <vector>

// #include "acl/acl.h"
#include "aicl.h"

struct OperatorDesc {
    /**
     * Constructor
     * @param [in] opType: op type
     */
    explicit OperatorDesc(std::string opType);

    /**
     * Destructor
     */
    virtual ~OperatorDesc();

    /**
     * Add an input tensor description
     * @param [in] dataType: data type
     * @param [in] numDims: number of dims
     * @param [in] dims: dims
     * @param [in] format: format
     * @return OperatorDesc
     */
    OperatorDesc &AddInputTensorDesc(aiclDataType dataType, int numDims, const int64_t *dims, aiclFormat format);

    /**
     * Add an output tensor description
     * @param [in] dataType: data type
     * @param [in] numDims: number of dims
     * @param [in] dims: dims
     * @param [in] format: format
     * @return OperatorDesc
     */
    OperatorDesc &AddOutputTensorDesc(aiclDataType dataType, int numDims, const int64_t *dims, aiclFormat format);

    std::string opType;
    std::vector<aiclTensorDesc *> inputDesc;
    std::vector<aiclTensorDesc *> outputDesc;
    aiclopAttr *opAttr;
};

#endif // OPERATOR_DESC_H
