#include </usr/local/include/python3.7m/Python.h>
#include "aicl.h"
#include <vector>
#include <iostream>
#include <dmlc/logging.h>
#include <tvm/te/operation.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>


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
    return;
}

aiclDataBuffer *aiclCreateDataBuffer(void *data, size_t size) {
    return (aiclDataBuffer *)data;
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

    Py_Initialize();   //  初始化 python 接口
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
    // std::cout << "run_mod_path = " << run_mod_path << std::endl;

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

    // std::cout << aiclGetTensorDescSize((const aiclTensorDesc*)(inputs[0])) << std::endl;
    // std::cout << aiclGetTensorDescSize((const aiclTensorDesc*)(inputs[1])) << std::endl;

    cudaMemcpy(input0_data, inputs[0], tvm::runtime::GetDataSize(*(input0.operator->())), cudaMemcpyDeviceToDevice);
    cudaMemcpy(input1_data, inputs[1], tvm::runtime::GetDataSize(*(input1.operator->())), cudaMemcpyDeviceToDevice);

    set_input(input_name[0].c_str(), input0);
    set_input(input_name[1].c_str(), input1);
    run();

    tvm::runtime::NDArray output0 = get_output(0);
    LOG(INFO) << output0.DataType();

    auto output0_data = static_cast<void *>(output0->data);
    cudaMemcpy(outputs[0], output0_data, tvm::runtime::GetDataSize(*(output0.operator->())), cudaMemcpyDeviceToDevice);
    // half* outy = new half[2 * 1024 * 1024 * 6];
    // output0.CopyToBytes((void*)outy, 12 * 1024 * 1024 * sizeof(half));

    // std::ifstream outfile("./out", std::ifstream::binary);
    // half *out_data = new half[12582912];
    // outfile.read((char *)out_data, 25165824);
    // for (int i = 0; i < 12582912; i++){
    //     std::cout << outy[i] << "  ";
    //     ICHECK_LT(fabs(outy[i] - out_data[i]), 1e-4);
    // }    

    // LOG(INFO) << out;
}


void* ReadBinFile(std::string fileName, uint32_t& fileSize){
    std::ifstream BinFile(fileName, std::ifstream::binary);
    BinFile.seekg(0, BinFile.end);
    fileSize = BinFile.tellg();
    BinFile.seekg(0, BinFile.beg);
    void *binFileBufferData = nullptr;
    cudaMallocHost(&binFileBufferData, fileSize);
    BinFile.read(static_cast<char *>(binFileBufferData), fileSize);
    BinFile.close();
    return binFileBufferData;
}

int main(int argc, char* argv[]){

    for (int i = 0; i < argc; i++) {
        std::cout << "No." << i << " parameter is:" << argv[i] << std::endl;
    }
    std::string input_x_file = argv[1];
    std::string input_filter_file = argv[2];
    std::string out_file = argv[3];

    // char* optype = "add";
    int numIn = 3;
    int numOu = 1;
    // std::vector<int64_t> shape{8, 16};
    std::vector<int64_t> inputShapeCast{2, 1024, 1024, 3};
    std::vector<int64_t> inputFilterShapeCast{6, 3, 3, 3};
    std::vector<int64_t> outputShapeCast{2, 1024, 1024, 6};

    // std::string opType = "Add";
    const char* opType_ = "Conv2D";

    // aiclopAttr *opAttr;
    aiclopAttr *opAttr = aiclopCreateAttr();
    int64_t intList[4]{1, 1, 1, 1};
    aiclopSetAttrListInt(opAttr, "strides", 4, intList);
    aiclopSetAttrListInt(opAttr, "pads", 4, intList);
    aiclopSetAttrListInt(opAttr, "dilations", 4, intList);

    // aiclDataType dataType = AICL_INT32;
    aiclDataType inputDataTypeCast = AICL_FLOAT16;
    aiclDataType outputDataTypeCast = AICL_FLOAT16;

    aiclFormat format = AICL_FORMAT_ND;

    // std::vector<aiclTensorDesc *> inputDesc;
    // std::vector<aiclTensorDesc *> outputDesc;
    // aiclTensorDesc *desc1 = aiclCreateTensorDesc(dataType, shape.size(), shape.data(), format);
    // aiclTensorDesc *desc2 = aiclCreateTensorDesc(dataType, shape.size(), shape.data(), format);
    // inputDesc.emplace_back(desc1);
    // inputDesc.emplace_back(desc2);
    // aiclTensorDesc *out1 = aiclCreateTensorDesc(dataType, shape.size(), shape.data(), format);
    // outputDesc.emplace_back(out1);
    aiclTensorDesc *inputDescCast[numIn];
    aiclTensorDesc *OutputDescCast[numOu];
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

    void* x_tensor_host = nullptr;
    void* x_tensor_ptr = nullptr;
    void* filter_tensor_host = nullptr;
    void* filter_tensor_ptr = nullptr;
    void* out_tensor_host = nullptr;
    void* out_tensor_ptr = nullptr;
    uint32_t x_tensor_size;
    uint32_t filter_tensor_size;
    uint32_t out_tensor_size = 25165824;
    std::vector<aiclDataBuffer*> in_buffers;
    std::vector<aiclDataBuffer*> out_buffers;

    x_tensor_host = ReadBinFile(input_x_file, x_tensor_size);
    filter_tensor_host = ReadBinFile(input_filter_file, filter_tensor_size);

    cudaMalloc(&x_tensor_ptr, x_tensor_size);
    cudaMalloc(&filter_tensor_ptr, filter_tensor_size);
    cudaMalloc(&out_tensor_ptr, out_tensor_size);
    cudaMemcpy(x_tensor_ptr, x_tensor_host, x_tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(filter_tensor_ptr, filter_tensor_host, filter_tensor_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(out_tensor_ptr, out_tensor_host, out_tensor_size, cudaMemcpyHostToDevice);

    aiclDataBuffer* x_tensor_data = aiclCreateDataBuffer(x_tensor_ptr, x_tensor_size);
    aiclDataBuffer* filter_tensor_data = aiclCreateDataBuffer(filter_tensor_ptr, filter_tensor_size);
    aiclDataBuffer* bias_tensor_data = aiclCreateDataBuffer(nullptr, 0);

    aiclDataBuffer* out_tensor_data = aiclCreateDataBuffer(out_tensor_ptr, out_tensor_size);
    in_buffers.push_back(x_tensor_data);
    in_buffers.push_back(filter_tensor_data);
    in_buffers.push_back(bias_tensor_data);
    out_buffers.push_back(out_tensor_data);

    aiclrtStream stream_;

    aiclopCompileAndExecute(opType_, 
                            numIn, 
                            inputDescCast, 
                            in_buffers.data(), 
                            numOu, 
                            OutputDescCast, 
                            out_buffers.data(),
                            opAttr, 
                            AICL_ENGINE_SYS, 
                            AICL_COMPILE_SYS, 
                            NULL, 
                            stream_);

    return 0;
}