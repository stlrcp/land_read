#include <cuda_runtime.h>
#include <cuda.h>
#include <dmlc/logging.h>
#include <tvm/te/operation.h>
#include </usr/local/include/python3.7m/Python.h>
#include <fstream>


int main(){
    Py_Initialize();   //  初始化 python 接口
    PyRun_SimpleString("import tvm");
    PyRun_SimpleString("from tvm import relay");
    PyRun_SimpleString("import numpy as np");
    PyRun_SimpleString("from tvm.contrib import graph_executor");
    PyRun_SimpleString("x = relay.var('x', shape=(1, 56, 56, 64))");
    PyRun_SimpleString("weight = relay.var('weight', shape=(3, 3, 64, 32))");
    PyRun_SimpleString("func = relay.nn.conv2d(x, \
                       weight,      \
                       channels=32,     \
                       kernel_size=(3, 3),      \
                       padding=(1, 1),      \
                       data_layout='NHWC',      \
                       kernel_layout='HWIO',)");

    PyRun_SimpleString("mod = tvm.IRModule.from_expr(func)");
    PyRun_SimpleString("run_mod = relay.build(mod, target='llvm', params=None)");
    // PyRun_SimpleString("run_mod = relay.build(mod, target='iluvatar', params=None)");

    PyRun_SimpleString("run_mod.export_library('./conv.so')");
    Py_Finalize();

    // DLDevice dev{kDLILUVATAR, 0};
    DLDevice dev{kDLCPU, 0};
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("./conv.so");
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
    tvm::runtime::PackedFunc get_input = gmod.GetFunction("get_input");
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");

    auto x_input = tvm::runtime::NDArray::Empty({1, 56, 56, 64}, {kDLFloat, 32, 1}, {kDLCPU, 0});
    auto x_weight = tvm::runtime::NDArray::Empty({3, 3, 64, 32}, {kDLFloat, 32, 1}, {kDLCPU, 0});

    auto x_data = static_cast<float*>(x_input->data);
    auto w_data = static_cast<float*>(x_weight->data);

    std::ifstream infile("./input_data.txt");
    for (int i = 0; i < 200704; i++){
        std::string line;
        std::getline(infile, line);
        float f = std::stof(line);
        x_data[i] = f;
    }

    std::ifstream wfile("./weight_data.txt");
    for (int n = 0; n < 18432; n++)
    {
        std::string line;
        std::getline(wfile, line);
        float f = std::stof(line);
        w_data[n] = f;
    }

    set_input("x", x_input);
    set_input("weight", x_weight);
    run();

    tvm::runtime::NDArray Y = get_output(0);
    LOG(INFO) << Y.DataType();
    float* outy = new float[1 * 56 * 56 * 32];
    Y.CopyToBytes((void*)outy, 56 * 56 * 32 * sizeof(float));

    std::ifstream outfile("./out_data.txt");
    for (int i = 0; i < 100352; i++){
        std::string line;
        std::getline(outfile, line);
        float f = std::stof(line);
        ICHECK_LT(fabs(outy[i] - f), 1e-4);
        std::cout << outy[i] << " ";
    }

    LOG(INFO) << Y;
    return 0;
}