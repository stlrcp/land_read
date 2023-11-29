#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <iostream>
#include <string>

using namespace std;

#define imin(a, b) (a<b? a:b)
const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c)
{
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    // 每个线程负责计算的点乘，再加和
    while (tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    // 每个线程块中线程计算的加和保存到缓冲区cache，一共有blocksPerGrid个缓冲区副本
    cache[cacheIndex] = temp;
    // 对线程块中的线程进行同步
    __syncthreads();

    // 归约运算，将每个缓冲区中的值加和，存放到缓冲区第一个元素位置
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cacheIndex < i)
        {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    // 使用第一个线程取出每个缓冲区第一个元素赋值到c数组
    if (cacheIndex == 0)
    {
        c[blockIdx.x] = cache[0];
    }
}

int main()
{
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    // 分配CPU内存
    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    partial_c = (float *)malloc(blocksPerGrid * sizeof(float));

    // 分配GPU内存
    cudaMalloc(&dev_a, N * sizeof(float));
    cudaMalloc(&dev_b, N * sizeof(float));
    cudaMalloc(&dev_partial_c, blocksPerGrid * sizeof(float));

    float sum = 0;
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    // 将数组上传到GPU
    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU 完成最终求和
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        c += partial_c[i];
    }

#define sum_squares(x) (x*(x+1)*(2*x+1)/6)
    printf("does GPU value %.6g = %.6g ? \n", c, 2 * sum_squares((float)(N - 1)));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    free(a);
    free(b);
    free(partial_c);

    return 0;
}