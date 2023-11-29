#include <iostream>
#include <cstdlib>
#include "bi_kernel.cu"


__global__ void printTensor(Tensor<float, 4> T0)
{
    printf("========= T0.size[0] ===== %d \n", T0.size[0]);
    printf("========= T0.size[1] ===== %d \n", T0.size[1]);
    printf("========= T0.size[2] ===== %d \n", T0.size[2]);
    printf("====== t3[1] ===== %f \n", T0[66009]);
}

void checkResult(float *hostRef, const int N) {
    float epsilon = 1.0E-6;
    bool match = 1;
    int num = 1;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - (float)(0.003906)) > epsilon) {
            match = 0;
            // printf("Arrays do not match ! \n");
            // printf("host %5.6f at current %d \n", hostRef[i], i);
            // break;
            num += 1;
        }
    }
    if (match)
        printf("Arrays match. \n\n");
    printf("============ num = %d \n", num);
}

int main() {
    int N = 256*10*10*10;
    float *h_a, *h_c;
    size_t nBytes = N * sizeof(float);
    h_a = (float *)malloc(nBytes);
    h_c = (float *)malloc(nBytes);
    for (int i=0; i<N; i++) {
        h_a[i] = (float)1;
        h_c[i] = (float)0;
    }
    float *d_a;
    float *d_b;
    float *d_c;
    cudaMalloc((float **)&d_a, nBytes);
    cudaMalloc((float **)&d_b, nBytes);
    cudaMalloc((float **)&d_c, nBytes);
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, nBytes, cudaMemcpyHostToDevice);
    Tensor<float, 4> D0;
    Tensor<float, 4> D1;
    Tensor<float, 4> D2;
    D0.size[0] = 256;
    D0.size[1] = 10;
    D0.size[2] = 10;
    D0.size[3] = 10;
    D0.data = d_a;
    D1.size[0] = 256;
    D1.size[1] = 10;
    D1.size[2] = 10;
    D1.size[3] = 10;
    D1.data = d_b;
    D2.size[0] = 256;
    D2.size[1] = 10;
    D2.size[2] = 10;
    D2.size[3] = 10;
    D2.data = d_c;

    dim3 grid(16, 1, 1);
    dim3 block(32, 64, 1);
    const int smem = 8196;
    kernel1<<<grid, block, smem>>>(D0, D1, D2);
    float *h_d2;
    h_d2 = (float *)malloc(nBytes);
    cudaMemcpy(h_d2, d_c, nBytes, cudaMemcpyDeviceToHost);

    // initTensor<<<1, 1>>>(d_a, &D0);
    printTensor<<<1, 1>>>(D0);
    cudaDeviceSynchronize();
    printTensor<<<1, 1>>>(D1);
    cudaDeviceSynchronize();
    printTensor<<<1, 1>>>(D2);
    // std::cout << h_d2[66009] << std::endl;
    checkResult(h_d2, N);
    // for (int i = 0; i < N; i++) {
    //     printf("========= h_d2[i] %d  = %f \n", i, h_d2[i]);
    // }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_d2);
    return 0;
}
