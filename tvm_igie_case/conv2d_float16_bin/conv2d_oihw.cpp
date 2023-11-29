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
                       kernel_layout='OIHW',)");

    PyRun_SimpleString("mod = tvm.IRModule.from_expr(func)");

    PyRun_SimpleString("desired_layout = {'nn.conv2d': ['NHWC', 'HWIO']}");
    PyRun_SimpleString("with tvm.transform.PassContext(opt_level=3):\n\t\
                        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layout)])\n\t\
                        mod = seq(mod)");
    // PyRun_SimpleString("    seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layout)])");
    // PyRun_SimpleString("    mod = seq(mod)");

    PyRun_SimpleString("target = tvm.target.iluvatar(model='MR', options='-libs=cudnn')");
    PyRun_SimpleString("run_mod = relay.build(mod, target=target, params=None)");
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

    // auto x_data = static_cast<half *>(x_input->data);
    // auto w_data = static_cast<half *>(x_weight->data);
    // cudaMemcpy(x_data, inputs[0], (int)12582912, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(w_data, inputs[1], (int)324, cudaMemcpyDeviceToDevice);

    half *x_data;
    half *w_data;
    cudaMallocHost(&x_data, 6 * 1024 * 1024 * sizeof(half));
    cudaMallocHost(&w_data, 9 * 6 * 3 * sizeof(half));

    std::ifstream infile("./x_float16_2x1024x1024x3.bin.in", std::ifstream::binary);
    half *input_data = new half[6291456];
    infile.read((char *)input_data, 12582912);
    for (int i = 0; i < 6291456; i++){
        x_data[i] = input_data[i];
    }

    std::ifstream wfile("./filter_float16_6x3x3x3.bin.in", std::ifstream::binary);
    half *weight_data = new half[162];
    wfile.read((char *)weight_data, 324);
    for (int n = 0; n < 162; n++){
        w_data[n] = weight_data[n];
    }

    x_input.CopyFromBytes(x_data, 6 * 1024 * 1024 * sizeof(half));
    x_weight.CopyFromBytes(w_data, 9 * 6 * 3 * sizeof(half));



    set_input("x", x_input);
    set_input("weight", x_weight);
    run();

    tvm::runtime::NDArray out = get_output(0);
    LOG(INFO) << out.DataType();
    // auto out_data = static_cast<half *>(out->data);
    // cudaMemcpy(outputs[0], out_data, (int)25165824, cudaMemcpyDeviceToDevice);
    half* outy = new half[2 * 1024 * 1024 * 6];
    out.CopyToBytes((void*)outy, 12 * 1024 * 1024 * sizeof(half));

    std::ifstream outfile("./out", std::ifstream::binary);
    half *out_data = new half[12582912];
    outfile.read((char *)out_data, 25165824);
    for (int i = 0; i < 12582912; i++){
        // half f = out_data[i];
        // ICHECK_LT(fabs(outy[i] - f), 1e-4);
        ICHECK_LT(fabs(outy[i] - out_data[i]), 1e-4);
        //std::cout << outy[i] << " ";
    }    

    // LOG(INFO) << Y;
    return 0;
}
