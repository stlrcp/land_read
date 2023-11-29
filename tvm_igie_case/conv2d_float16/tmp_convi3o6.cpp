// #include "aicl.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <dmlc/logging.h>
#include <tvm/te/operation.h>
#include </usr/local/include/python3.7m/Python.h>
#include <cuda_fp16.h>


int main(){
    Py_Initialize();   //  初始化 python 接口
    PyRun_SimpleString("import tvm");
    PyRun_SimpleString("from tvm import relay");
    PyRun_SimpleString("import numpy as np");
    PyRun_SimpleString("from tvm.contrib import graph_executor");
    PyRun_SimpleString("x = relay.var('x', shape=(2, 1024, 1024, 3), dtype='float16')");
    PyRun_SimpleString("weight = relay.var('weight', shape=(6, 3, 3, 3), dtype='float16')");
    PyRun_SimpleString("func = relay.nn.conv2d(x, \
                       weight,      \
                       channels=6,     \
                       kernel_size=(3, 3),      \
                       padding=(1, 1),      \
                       data_layout='NHWC',      \
                       kernel_layout='OHWI',)");

    PyRun_SimpleString("mod = tvm.IRModule.from_expr(func)");
    PyRun_SimpleString("target = tvm.target.iluvatar(model='MR', options='-libs=cudnn')");
    PyRun_SimpleString("run_mod = relay.build(mod, target=target, params=None)");
    // PyRun_SimpleString("dev = tvm.iluvatar(0)");
    // PyRun_SimpleString("gen_module = graph_executor.GraphModule(run_mod["default"](dev))");
    PyRun_SimpleString("run_mod.export_library('./conv.so')");
    Py_Finalize();

    DLDevice dev{kDLILUVATAR, 0};
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("./conv.so");
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
    tvm::runtime::PackedFunc get_input = gmod.GetFunction("get_input");
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");

    auto x_input = tvm::runtime::NDArray::Empty({2, 1024, 1024, 3}, {kDLFloat, 16, 1}, {kDLILUVATAR, 0});
    auto x_weight = tvm::runtime::NDArray::Empty({6, 3, 3, 3}, {kDLFloat, 16, 1}, {kDLILUVATAR, 0});

    half *x_data;
    half *w_data;
    cudaMallocHost(&x_data, 6 * 1024 * 1024 * sizeof(half));
    cudaMallocHost(&w_data, 9 * 6 * 3 * sizeof(half));

    std::ifstream infile("./input_data.txt");
    for (int i = 0; i < 6291456; i++){
        std::string line;
        std::getline(infile, line);
        half f = std::stof(line);
        x_data[i] = f;
    }

    std::ifstream wfile("./weight_data.txt");
    for (int n = 0; n < 162; n++){
        std::string line;
        std::getline(wfile, line);
        half f = std::stof(line);
        w_data[n] = f;
    }

    x_input.CopyFromBytes(x_data, 6 * 1024 * 1024 * sizeof(half));
    x_weight.CopyFromBytes(w_data, 9 * 6 * 3 * sizeof(half));

    set_input("x", x_input);
    set_input("weight", x_weight);
    run();

    tvm::runtime::NDArray Y = get_output(0);
    LOG(INFO) << Y.DataType();
    half* outy = new half[2 * 1024 * 1024 * 6];
    Y.CopyToBytes((void*)outy, 12 * 1024 * 1024 * sizeof(half));
    // int *outy = (int *)outputs[0];
    // Y.CopyToBytes((void *)outy, 8 * 16 * sizeof(float));
    std::ifstream outfile("./out_data.txt");
    for (int i = 0; i < 12582912; i++){
        std::string line;
        std::getline(outfile, line);
        half f = std::stof(line);
        // ICHECK_LT(fabs(outy[i] - f), 1e-4);
        ICHECK_LT(fabs(outy[i] - f), 1e-3);
        std::cout << outy[i] << " ";
    }
    LOG(INFO) << Y;


    return 0;
}
