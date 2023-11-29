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

    Py_Initialize(); //  初始化 python 接口
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

    std::cout << "func_str = " << func_str << std::endl;

    PyRun_SimpleString(func_str.c_str());
    PyRun_SimpleString("mod = tvm.IRModule.from_expr(func)");
    PyRun_SimpleString("run_mod = relay.build(mod, target='iluvatar', params=None)");
    std::string export_engine = "run_mod.export_library('" + optype + ".so')";
    int result = PyRun_SimpleString(export_engine.c_str());

    if (result != 0){
        return AICL_RET_ERROR;
    }
    else
    {
        return AICL_RET_SUCCESS;
    }
    Py_Finalize();

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
    std::string optype(opType);
    for (auto &i : optype){
        i = tolower(i);
    }
    std::string run_mod_path = "./" + optype + ".so";

    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(run_mod_path.c_str());
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

    int *tmp_a;
    cudaMallocHost(&tmp_a, 8 * 16 * sizeof(int));
    cudaMemcpy(tmp_a, inputs[1], 8 * 16 * sizeof(int), cudaMemcpyDeviceToHost);

    // std::cout << inputs[0] << std::endl;
    // std::cout << ((int *)inputs[0])[5] << std::endl;
    // std::cout << ((int *)inputs[1])[5] << std::endl;
    std::cout << tmp_a[5] << std::endl;
    


    for (int n = 0; n < input_vec.size(); n++){
        int input_size = tvm::runtime::GetDataSize(*(input_vec[n].operator->()));
        std::cout << input_size << std::endl;
        input_vec[n].CopyFromBytes((int *)inputs[n], input_size);
    }

    for (int i = 0; i < numInputs; i++){
        std::string in_str = "input" + std::to_string(i);
        set_input(in_str.c_str(), input_vec[i]);
    }

    run();

    tvm::runtime::NDArray Y = get_output(0);
    LOG(INFO) << Y.DataType();

    int* outy = new int[8 * 16];
    Y.CopyToBytes((void*)outy, 8 * 16 * sizeof(int));

    LOG(INFO) << outy[0];
    LOG(INFO) << 11111;
    for (int i = 0; i < 128; ++i) {
        // LOG(INFO) << outy[i];
        ICHECK_LT(fabs(outy[i] - (i + (i + 1))), 1e-4);
    }
}




int main(){
    // char* optype = "add";
    int numIn = 2;
    int numOu = 1;
    std::vector<int64_t> shape{8, 16};
    std::string opType = "Add";
    aiclopAttr *opAttr;
    aiclDataType dataType = AICL_INT32;
    std::cout << "dataType size = " << sizeof(dataType) << std::endl;
    std::cout << "AICL_INT32 size = " << sizeof(AICL_INT32) << std::endl;
    aiclFormat format = AICL_FORMAT_ND;

    std::vector<aiclTensorDesc *> inputDesc;
    std::vector<aiclTensorDesc *> outputDesc;
    aiclTensorDesc *desc1 = aiclCreateTensorDesc(dataType, shape.size(), shape.data(), format);
    aiclTensorDesc *desc2 = aiclCreateTensorDesc(dataType, shape.size(), shape.data(), format);
    inputDesc.emplace_back(desc1);
    inputDesc.emplace_back(desc2);

    aiclTensorDesc *out1 = aiclCreateTensorDesc(dataType, shape.size(), shape.data(), format);
    outputDesc.emplace_back(out1);
    
    aiclopCompile(opType.c_str(),
                        numIn,
                        inputDesc.data(),
                        numOu,
                        outputDesc.data(),
                        opAttr,
                        AICL_ENGINE_SYS,
                        AICL_COMPILE_SYS,
                        nullptr);

    std::cout << "end of acilopCompile !!!" << std::endl;



    int *h_a;
    int *h_b;
    int *h_out;
    cudaMallocHost(&h_a, 8 * 16 * sizeof(int));
    cudaMallocHost(&h_b, 8 * 16 * sizeof(int));
    cudaMallocHost(&h_out, 8 * 16 * sizeof(int));
    for (int i = 0; i < 128; i++) {
        h_a[i] = i;
        h_b[i] = i+1;
    }

    int *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, 8 * 16 * sizeof(int));
    cudaMalloc(&d_b, 8 * 16 * sizeof(int));
    cudaMalloc(&d_out, 8 * 16 * sizeof(int));
    cudaMemcpy(d_a, h_a, 8 * 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 8 * 16 * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<aiclDataBuffer *> input_bu;
    std::vector<aiclDataBuffer *> output_bu;
    input_bu.emplace_back((aiclDataBuffer *)d_a);
    input_bu.emplace_back((aiclDataBuffer *)d_b);
    output_bu.emplace_back((aiclDataBuffer *)d_out);

    aiclrtStream stream = nullptr;
    aiclrtCreateStream(&stream);

    aiclopExecute(opType.c_str(), 
                  numIn,
                  inputDesc.data(),
                  input_bu.data(),
                  numOu,
                  outputDesc.data(),
                  output_bu.data(),
                  opAttr,
                  stream);

    return 0;
}