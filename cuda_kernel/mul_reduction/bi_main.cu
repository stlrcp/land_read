#include <iostream>
#include <cstdlib>
#include "bi_kernel.cu"


__global__ void printTensor(Tensor<float, 2> T0)
{
    printf("========= T0.size[0] ===== %d \n", T0.size[0]);
    printf("========= T0.size[1] ===== %d \n", T0.size[1]);
    printf("====== t3[1] ===== %f \n", T0[10]);
}

int main() {
    int N = 8*4*16;
    int M = 4*16;
    int C = 8 * 16;
    int K = 4;
    float *h_a, *h_b, *h_c;
    int64_t *h_d;
    size_t nBytes = N * sizeof(float);
    size_t mBytes = M * sizeof(float);
    size_t cBytes = C * sizeof(float);
    size_t kBytes = K * sizeof(int64_t);
    h_a = (float *)malloc(nBytes);
    h_b = (float *)malloc(mBytes);
    h_c = (float *)malloc(cBytes);
    h_d = (int64_t *)malloc(kBytes);
    for (int i=0; i<N; i++) {
        h_a[i] = (float)1;
        if (i < M)
            h_b[i] = (float)0;
        if (i < C)
            h_c[i] = (float)0;
        if (i < K)
            h_d[i] = (int64_t)0;
    }

    float *d_a;
    float *d_b;
    float *d_c;
    int64_t *d_d;
    cudaMalloc((float **)&d_a, nBytes);
    cudaMalloc((float **)&d_b, mBytes);
    cudaMalloc((float **)&d_c, cBytes);
    cudaMalloc((int64_t **)&d_d, kBytes);
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, mBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, cBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, kBytes, cudaMemcpyHostToDevice);
    Tensor<float, 3> D0;
    Tensor<float, 2> D1;
    Tensor<float, 1> D2;
    Tensor<int64_t, 1> D3;
    D0.size[0] = 8;
    D0.size[1] = 4;
    D0.size[2] = 16;
    D0.data = d_a;
    D1.size[0] = 4;
    D1.size[1] = 16;
    D1.data = d_b;
    D2.size[0] = 128;
    D2.data = d_c;
    D3.size[0] = 4;
    D3.data = d_d;

    dim3 grid(4, 2, 1);
    dim3 block(16, 4, 1);
    const int smem = 2176;

    kernel1<<<grid, block, smem>>>(D0, D1, D2, D3);
    float *h_d2;
    h_d2 = (float *)malloc(mBytes);
    cudaMemcpy(h_d2, d_b, mBytes, cudaMemcpyDeviceToHost);

    printTensor<<<1, 1>>>(D1);
    cudaDeviceSynchronize();
    for (int i = 0; i < M; i++) {
        printf("========= h_d2[i] %d  = %f \n", i, h_d2[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    free(h_d2);
    return 0;
}