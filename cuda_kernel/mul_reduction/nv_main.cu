#include <iostream>
#include <cstdlib>
#include "nv_kernel.cu"


__global__ void printTensor(Tensor<float, 2> T0)
{
    printf("========= T0.size[0] ===== %d \n", T0.size[0]);
    printf("========= T0.size[1] ===== %d \n", T0.size[1]);
    printf("====== t3[1] ===== %f \n", T0[10]);
}

void checkResult(float *hostRef, const int N) {
    float epsilon = 1.0E-6;
    bool match = 1;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - (float)(0.014925)) > epsilon) {
            match = 0;
            printf("Arrays do not match ! \n");
            printf("host %5.6f at current %d \n", hostRef[i], i);
            break;
        }
    }
    if (match)
        printf("Arrays match. \n\n");
}


int main() {
    int N = 8*4*16;
    int M = 4*16;
    float *h_a, *h_c;
    size_t nBytes = N * sizeof(float);
    size_t mBytes = M * sizeof(float);
    h_a = (float *)malloc(nBytes);
    h_c = (float *)malloc(mBytes);
    for (int i=0; i<N; i++) {
        h_a[i] = (float)1;
    }
    for (int j = 0; j < M; j++) {
        h_c[j] = (float)1;
    }

    float *d_a;
    float *d_c;
    cudaMalloc((float **)&d_a, nBytes);
    cudaMalloc((float **)&d_c, mBytes);
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, mBytes, cudaMemcpyHostToDevice);
    Tensor<float, 3> D0;
    Tensor<float, 2> D1;
    D0.size[0] = 8;
    D0.size[1] = 4;
    D0.size[2] = 16;
    D0.data = d_a;
    D1.size[0] = 4;
    D1.size[1] = 16;
    D1.data = d_c;

    dim3 grid(4, 1, 1);
    dim3 block(16, 2, 1);
    const int smem = 2176;

    kernel1<<<grid, block, smem>>>(D0, D1);
    float *h_d2;
    h_d2 = (float *)malloc(mBytes);
    cudaMemcpy(h_d2, d_c, mBytes, cudaMemcpyDeviceToHost);

    printTensor<<<1, 1>>>(D1);
    cudaDeviceSynchronize();

    for (int i = 0; i < M; i++) {
        printf("========= h_d2[i] %d  = %f \n", i, h_d2[i]);
    }

    cudaFree(d_a);
    // cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_c);
    free(h_d2);
    return 0;
}