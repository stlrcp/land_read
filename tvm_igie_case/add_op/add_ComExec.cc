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


aiclTensorDesc *aiclCreateTensorDesc(aiclDataType dataType, int numDims,
                                     const int64_t *dims,  aiclFormat format) {
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
    return (aiclTensorDesc *)from;
}

size_t aiclGetTensorDescNumDims(const aiclTensorDesc *desc) {
    DLTensor *TDesc = const_cast<DLTensor*>((DLTensor*)desc);
    return TDesc->ndim;
}

aiclRet aiclGetTensorDescDim(const aiclTensorDesc *desc, size_t index, int64_t *dimSize) {
    DLTensor *TDesc = const_cast<DLTensor*>((DLTensor*)desc);
    return TDesc->shape[index];
}



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
        std::cout << in_str << std::endl;
    }
    std::cout << shape_str[0] << std::endl;
    std::cout << input_name[1] << std::endl;

    Py_Initialize(); //  初始化 python 接口
    PyRun_SimpleString("import numpy as np");
    PyRun_SimpleString("import tvm");
    PyRun_SimpleString("from tvm import relay");
    PyRun_SimpleString("from tvm.contrib import graph_executor");
    PyRun_SimpleString(in_str.c_str());
    PyRun_SimpleString("t_iter = iter(inputs[0])");

    std::string optype(opType);
    optype[0] = std::tolower(optype[0]);
    std::cout << "optype = " << optype << std::endl;

    std::string func_str = "func = relay." + optype + "(";

    for (int i = 0; i < input_name.size(); i++){
        std::string input_str = input_name[i] + " = relay.var(next(t_iter), ";
        input_str = input_str + shape_str[i] + ", dtype='float32')";
        std::cout << input_str << std::endl;
        PyRun_SimpleString(input_str.c_str());

        if (i == (input_name.size() - 1)){
            func_str += input_name[i] + ')';
        } else {
            func_str += input_name[i] + ',';
        }  
    }

    std::cout << "func_str = " << func_str << std::endl;

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

    std::cout << inputs[0] << std::endl;
    std::cout << ((int *)inputs[0])[5] << std::endl;
    std::cout << ((int *)inputs[1])[5] << std::endl;
    


    for (int n = 0; n < input_vec.size(); n++){
        int input_size = tvm::runtime::GetDataSize(*(input_vec[n].operator->()));
        std::cout << input_size << std::endl;
        input_vec[n].CopyFromBytes((float *)inputs[n], input_size);
    }

    for (int i = 0; i < numInputs; i++){
        std::string in_str = "in_" + std::to_string(i);
        set_input(in_str.c_str(), input_vec[i]);
    }

    run();

    tvm::runtime::NDArray Y = get_output(0);
    LOG(INFO) << Y.DataType();

    float* outy = new float[8 * 16];
    Y.CopyToBytes((void*)outy, 8 * 16 * sizeof(float));

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



    float *h_a;
    float *h_b;
    float *h_out;
    cudaMallocHost(&h_a, 8 * 16 * sizeof(float));
    cudaMallocHost(&h_b, 8 * 16 * sizeof(float));
    cudaMallocHost(&h_out, 8 * 16 * sizeof(float));
    for (int i = 0; i < 128; i++) {
        h_a[i] = i;
        h_b[i] = i+1;
    }


    std::vector<aiclDataBuffer *> input_bu;
    std::vector<aiclDataBuffer *> output_bu;
    input_bu.emplace_back((aiclDataBuffer *)h_a);
    input_bu.emplace_back((aiclDataBuffer *)h_b);
    output_bu.emplace_back((aiclDataBuffer *)h_out);

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