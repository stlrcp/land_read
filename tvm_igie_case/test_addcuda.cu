#include <stdio.h>
#include <stdlib.h>
#include <iostream>



#ifdef __ILUVATAR__
#define POS_INFINITY INFINITY
#define NEG_INFINITY -INFINITY
#else
#define NAN __int_as_float(0x7fffffff)
#define POS_INFINITY __int_as_float(0x7f800000)
#define NEG_INFINITY __int_as_float(0xff800000)
#endif

template<typename T>
__device__ T maximum(T a, T b) {
  return isnan(a) ? a : (a > b ? a : b);
}

template<typename T>
__device__ T minimum(T a, T b) {
  return isnan(a) ? a : (a < b ? a : b);
}

extern "C" __global__
void fused_add(bool* tx, float* aten_add) {
{
  // printf("===================== threadIdx.x = %d, blockIdx.x = %d\n", threadIdx.x, blockIdx.x);
  if ((long long)(threadIdx.x) + 512ll * (long long)(blockIdx.x) < 2ll ? 1 : 0)
  {
    // double v = (double)(tx[(long long)(threadIdx.x) + 512ll * (long long)(blockIdx.x)]);
    double v = (double)(tx[(long long)(threadIdx.x) + 512ll * (long long)(blockIdx.x)]);
    // printf("============ double v = %f\n", v);
    printf("============ double v ============\n");
    aten_add[(long long)(threadIdx.x) + 512ll * (long long)(blockIdx.x)] = (float)(v + 1.5);
  }
}
}


int main() {

  int N = 2;
  bool *tx;
  float *aten_add;
  size_t nBytes = N * sizeof(bool);
  size_t mBytes = N * sizeof(float);
  tx = (bool *)malloc(nBytes);
  aten_add = (float *)malloc(mBytes);
  tx[0] = true;
  tx[1] = true;
  aten_add[0] = 0.0;
  aten_add[1] = 0.0;
  dim3 grid(1, 1, 1);
  dim3 block(512, 1, 1);
  bool *d_tx;
  float *d_aten_add;

  std::cout << "before aten_add[0] = " << aten_add[0] << std::endl;

  cudaMalloc((bool **)&d_tx, nBytes);
  cudaMalloc((float **)&d_aten_add, mBytes);
  cudaMemcpy(d_tx, tx, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_aten_add, aten_add, mBytes, cudaMemcpyHostToDevice);
  fused_add<<<grid, block>>>(d_tx, d_aten_add);
  cudaDeviceSynchronize();
  float *f_aten_add;
  f_aten_add = (float *)malloc(mBytes);
  f_aten_add[0] = 0;
  f_aten_add[1] = 0;
  cudaMemcpy(f_aten_add, d_aten_add, mBytes, cudaMemcpyDeviceToHost);
  std::cout << "after f_aten_add[0] = " << f_aten_add[0] << std::endl;

  cudaFree(d_tx);
  cudaFree(d_aten_add);
  free(tx);
  free(aten_add);
  free(f_aten_add);
  return 0;
}
