#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/profiling.h>
#include <tvm/relay/runtime.h>
#include <cstdio>
#include <dmlc/logging.h>
#include <tvm/relay/runtime.h>
#include <cuda_runtime.h>
#include <cuda.h>

int main(){

    DLDevice dev{kDLILUVATAR, 0};
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("./gpu_add.so");
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");


    auto A = tvm::runtime::NDArray::Empty({8, 16}, {kDLFloat, 32, 1}, dev);
    auto B = tvm::runtime::NDArray::Empty({8, 16}, {kDLFloat, 32, 1}, dev);
    
    LOG(INFO) << A.DataType();

    float *h_a;
    float *h_b;
    cudaMallocHost(&h_a, 8 * 16 * sizeof(float));
    cudaMallocHost(&h_b, 8 * 16 * sizeof(float));

    auto pA = static_cast<float*>(A->data);
    auto pB = static_cast<float*>(B->data);




    for (int i = 0; i < 128; i++) {
        h_a[i] = i+1;
        h_b[i] = i+2;
    }

    cudaMemcpy(pA, h_a, 8*16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pB, h_b, 8*16 * sizeof(float), cudaMemcpyHostToDevice);

    set_input("in_0", A);
    set_input("in_1", B);

    run();

    tvm::runtime::NDArray Y = get_output(0);
    LOG(INFO) << Y.DataType();

    auto pY = static_cast<float*>(Y->data);
    float *outy;
    cudaMallocHost(&outy, 8 * 16 * sizeof(float));
    cudaMemcpy(outy, pY, 8*16 * sizeof(float), cudaMemcpyDeviceToHost);

    LOG(INFO) << outy[0];
    LOG(INFO) << 11111;
    for (int i = 0; i < 128; ++i) {
        LOG(INFO) << outy[i];
        ICHECK_LT(fabs(outy[i] - (i+1 + (i + 2))), 1e-4);
    }
    return 0;
}