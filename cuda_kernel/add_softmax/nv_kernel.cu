

template <typename T, int N>
struct Tensor {
  __device__ T& operator[](int ind) {
    return data[ind];
  };

  T* data;
  int size[N];
  int stride[N];
};

// __global__ void initTensor(float *li_a, Tensor<float, 3> T3)
// {
//     T3.size[0] = 8;
//     T3.size[1] = 4;
//     T3.size[2] = 16;
//     T3.data = li_a;
//     printf("====== t3[2] ===== %f \n", T3[2]);
// }


#define NVFUSER_DEFINE_MAGIC_ZERO          \
  __shared__ int nvfuser_zero_s;           \
  if (threadIdx.x == 0)                    \
    nvfuser_zero_s = 0;                    \
  __syncthreads();                         \
  atomicMin(&nvfuser_zero_s, threadIdx.x); \
  int nvfuser_zero = nvfuser_zero_s;

#define NVFUSER_UPDATE_MAGIC_ZERO \
  do {                            \
    nvfuser_zero <<= 1;           \
  } while (0);


__device__ constexpr int ceilDiv(int a, int b) {
  return (a + b - 1) / b;
}

__device__ constexpr int64_t ceilDiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

__device__ constexpr int64_t ceilDiv(int64_t a, int b) {
  return ceilDiv(a, (int64_t)b);
}

__device__ constexpr int64_t ceilDiv(int a, int64_t b) {
  return ceilDiv((int64_t)a, b);
}

#ifndef __ILUVATAR__
__device__ constexpr double ceilDiv(double a, double b) {
  return std::ceil(a / b);
}
#else
__device__ constexpr float ceilDiv(float a, float b) {
  return std::ceil(a / b);
}
#endif

#ifndef __ILUVATAR__
__device__ constexpr double ceilDiv(double a, int64_t b) {
  return std::ceil(a / b);
}
#else
__device__ constexpr float ceilDiv(float a, int64_t b) {
  return std::ceil(a / b);
}
#endif

#ifndef __ILUVATAR__
__device__ constexpr double ceilDiv(int64_t a, double b) {
  return std::ceil(a / b);
}
#else
__device__ constexpr float ceilDiv(int64_t a, float b) {
  return std::ceil(a / b);
}
#endif

#define POS_INFINITY INFINITY
#define NEG_INFINITY -INFINITY

namespace index_utils {

// Utility functions

// Total size of provided dimension
template <typename _dim3>
__device__ __forceinline__ int size(const _dim3& d) {
  return (int)d.x * (int)d.y * (int)d.z;
}

// Linearized indexing of idx based on dim, if bool==false that dimension does
// not participate
template <bool X, bool Y, bool Z, typename _dim3, typename _dim3_2>
__device__ int maskedOffset(const _dim3& idx, const _dim3_2& dim) {
  int offset = 0;
  if (Z)
    offset += idx.z;
  if (Y)
    offset = offset * dim.y + idx.y;
  if (X)
    offset = offset * dim.x + idx.x;
  return offset;
}

// Linearized indexing of idx based on dim. All dimensions participate.
template <typename _dim3, typename _dim3_2>
__device__ int offset(const _dim3& idx, const _dim3_2& dim) {
  int offset = idx.z;
  offset = offset * dim.y + idx.y;
  offset = offset * dim.x + idx.x;
  return offset;
}

// Masks the provided dim3, those == false get truncated to 1
template <bool X, bool Y, bool Z, typename _dim3>
__device__ dim3 maskedDims(const _dim3& dim) {
  return dim3{
      X ? (unsigned)dim.x : 1U,
      Y ? (unsigned)dim.y : 1U,
      Z ? (unsigned)dim.z : 1U};
}

// Provides total size of dim with masking, those dims == false do not
// participate in the size calculation
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, typename _dim3>
__device__ int maskedSize(const _dim3& dim) {
  return size(maskedDims<X_BLOCK, Y_BLOCK, Z_BLOCK>(dim));
}

// Checks if provided idx is zero on those dims == true
template <bool X, bool Y, bool Z, typename _dim3>
__device__ bool maskedIsZero(const _dim3& idx) {
  bool isZero = true;
  if (X)
    isZero = isZero && idx.x == 0;
  if (Y)
    isZero = isZero && idx.y == 0;
  if (Z)
    isZero = isZero && idx.z == 0;
  return isZero;
}

// Checks if provided idx is zero on those dims == true
template <bool X, bool Y, bool Z, typename _dim3, typename _dim3_2>
__device__ bool maskedIsLast(const _dim3& idx, const _dim3_2& dim) {
  bool isZero = true;
  if (X)
    isZero = isZero && idx.x == dim.x - 1;
  if (Y)
    isZero = isZero && idx.y == dim.y - 1;
  if (Z)
    isZero = isZero && idx.z == dim.z - 1;
  return isZero;
}

} // namespace index_utils

// Default block synchronization. Just use __barrier_sync
namespace block_sync {

__forceinline__ __device__ void init() {}

// Thread-block synchronization
__forceinline__ __device__ void sync() {
  __syncthreads();
}

} // namespace block_sync



template <bool X_REDUCE, bool Y_REDUCE, bool Z_REDUCE, typename T, typename Func, typename _dim3, typename _dim3_2>
__device__ void blockReduce( T& out, const T& inp_val, Func reduction_op, const _dim3& thread_idx,
    const _dim3_2& block_dim, T* shared_mem, bool read_pred, bool write_pred, T init_val) {
  // If this thread will output a final result
  bool should_write = index_utils::maskedIsZero<X_REDUCE, Y_REDUCE, Z_REDUCE>(thread_idx);

  // Size of the reduction segments
  unsigned int reduction_size = index_utils::maskedSize<X_REDUCE, Y_REDUCE, Z_REDUCE>(block_dim);

  // Index into the reduction segment
  unsigned int reduction_tid = index_utils::maskedOffset<X_REDUCE, Y_REDUCE, Z_REDUCE>(thread_idx, block_dim);

  // Index of the reduction segment
  unsigned int reduction_idx = index_utils::maskedOffset<!X_REDUCE, !Y_REDUCE, !Z_REDUCE>(thread_idx, block_dim);

  // Offset into smem for the current thread
  unsigned int smem_offset = reduction_idx * reduction_size + reduction_tid;

  // Initialize shared memory
  if (read_pred) {
    shared_mem[smem_offset] = inp_val;
  } else {
    shared_mem[smem_offset] = init_val;
  }

  block_sync::sync();
  // Reduce down to nearest power of 2 for the tree reduction:
  int np2 = 1 << (31 - __clz(reduction_size));

  if (reduction_tid < np2 && reduction_tid + np2 < reduction_size) {
    reduction_op(shared_mem[smem_offset], shared_mem[smem_offset + np2]);
  }
  block_sync::sync();

  // loop peel the final iteration to save one syncthread for the end
  for (int factor = np2 / 2; factor > 1; factor >>= 1) {
    if (reduction_tid < factor) {
      reduction_op(shared_mem[smem_offset], shared_mem[smem_offset + factor]);
    }
    block_sync::sync();
  }

  if (should_write && write_pred) {
    T result = out;
    reduction_op(result, shared_mem[smem_offset]);
    if (reduction_size > 1) {
      reduction_op(result, shared_mem[smem_offset + 1]);
    }
    out = result;
  }
  block_sync::sync();
}


namespace broadcast {
// Broadcasts within partitioned groups of threads.
//
// X_THREAD: Broadcast from threadIdx.x == 0 if true
// Y_THREAD: Broadcast from threadIdx.y == 0 if true
// Z_THREAD: Broadcast from threadIdx.z == 0 if true
// inp_val: Per-thread source value. Only valid when the thread is a source.
// out: Per-thread output location
//
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, typename T>
__device__ void blockBroadcast(
    T& out,
    const T& inp_val,
    T* shared_mem,
    bool read_write_pred) {
  const bool has_valid_data = (!X_THREAD || threadIdx.x == 0) &&
      (!Y_THREAD || threadIdx.y == 0) && (!Z_THREAD || threadIdx.z == 0);

  const auto shared_offset =
      index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
          threadIdx, blockDim);

  if (has_valid_data && read_write_pred) {
    shared_mem[shared_offset] = inp_val;
  }

  block_sync::sync();

  if (read_write_pred) {
    out = shared_mem[shared_offset];
  }

  block_sync::sync();
}

} // namespace broadcast

__device__ float reciprocal(float x) {
  return 1 / x;
}




__global__ void kernel1(Tensor<float, 4> T0, Tensor<float, 4> T1, Tensor<float, 4> T10) {
  alignas(16) extern __shared__ char array[];
  void* shared_mem = array;
  NVFUSER_DEFINE_MAGIC_ZERO
  int i230;
  i230 = (((int)blockIdx.x) * ((int)blockDim.x)) + ((int)threadIdx.x);
  float T2[((1 * 2) * 1)];
  if ((((((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2)) + ((int)threadIdx.y)) < T0.size[0]) && (i230 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && (((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2)) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
    float T12[(2 * 1)];
    #pragma unroll
    for(int i189 = 0; i189 < 2; ++i189) {
      T12[i189]
         = T1[((((i189 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y)) * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i230];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    float T11[(2 * 1)];
    #pragma unroll
    for(int i181 = 0; i181 < 2; ++i181) {
      T11[i181]
         = T0[((((i181 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y)) * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i230];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(int i192 = 0; i192 < 2; ++i192) {
      T2[i192]
        = T11[i192]
        + T12[i192];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  } else {
    float T12[(2 * 1)];
    #pragma unroll
    for(int i189 = 0; i189 < 2; ++i189) {
      int i270;
      i270 = ((i189 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y);
      if ((((i270 < T0.size[0]) && (i230 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i189 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
        T12[i189]
           = T1[(i270 * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i230];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    float T11[(2 * 1)];
    #pragma unroll
    for(int i181 = 0; i181 < 2; ++i181) {
      int i287;
      i287 = ((i181 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y);
      if ((((i287 < T0.size[0]) && (i230 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i181 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
        T11[i181]
           = T0[(i287 * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i230];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(int i192 = 0; i192 < 2; ++i192) {
      if (((((((i192 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y)) < T0.size[0]) && (i230 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i192 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
        T2[i192]
          = T11[i192]
          + T12[i192];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
  float T3[1];
  T3[0] = NEG_INFINITY;
  float T14[1];
  T14[0] = NEG_INFINITY;
  if ((((((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2)) + ((int)threadIdx.y)) < T0.size[0]) && (i230 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && (((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2)) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
    #pragma unroll
    for(int i195 = 0; i195 < 2; ++i195) {
      T14[0] = fmax(
        T14[0],
        T2[i195]);
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  } else {
    #pragma unroll
    for(int i195 = 0; i195 < 2; ++i195) {
      if (((((((i195 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y)) < T0.size[0]) && (i230 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i195 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
        T14[0] = fmax(
          T14[0],
          T2[i195]);
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
  blockReduce<false, true, false>(
    T3[0],
    T14[0],
    [](float &a, float b) { a = fmax(a, b); },
    threadIdx,
    blockDim,
    static_cast<float*>(shared_mem),
    true,
    true,
    float(NEG_INFINITY));
  float T4[1];
  broadcast::blockBroadcast<false, true, false>(
    T4[0],
    T3[0],
    static_cast<float*>(shared_mem),
    true);
  // Alias Allocation - register
  auto& T6 = T2;
  if ((((((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2)) + ((int)threadIdx.y)) < T0.size[0]) && (i230 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && (((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2)) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
    #pragma unroll
    for(int i199 = 0; i199 < 2; ++i199) {
      float T5[1];
      T5[0]
        = T2[i199]
        - T4[0];
      T6[i199]
         = expf(T5[0]);
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  } else {
    #pragma unroll
    for(int i199 = 0; i199 < 2; ++i199) {
      int i877;
      i877 = ((i199 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y);
      int i866;
      i866 = ((i199 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y);
      float T5[1];
      if ((((i866 < T0.size[0]) && (i230 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && (i877 < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
        T5[0]
          = T2[i199]
          - T4[0];
      }
      if ((((i866 < T0.size[0]) && (i230 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && (i877 < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
        T6[i199]
           = expf(T5[0]);
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
  float T7[1];
  T7[0] = 0.00000000000000000e+00;
  float T15[1];
  T15[0] = 0.00000000000000000e+00;
  if ((((((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2)) + ((int)threadIdx.y)) < T0.size[0]) && (i230 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && (((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2)) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
    #pragma unroll
    for(int i202 = 0; i202 < 2; ++i202) {
      T15[0]
        = T15[0]
        + T6[i202];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  } else {
    #pragma unroll
    for(int i202 = 0; i202 < 2; ++i202) {
      if (((((((i202 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y)) < T0.size[0]) && (i230 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i202 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
        T15[0]
          = T15[0]
          + T6[i202];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
  blockReduce<false, true, false>(
    T7[0],
    T15[0],
    [](float &a, float b) { a = a + b; },
    threadIdx,
    blockDim,
    static_cast<float*>(shared_mem),
    true,
    true,
    float(0.00000000000000000e+00));
  float T8[1];
  broadcast::blockBroadcast<false, true, false>(
    T8[0],
    T7[0],
    static_cast<float*>(shared_mem),
    true);
  float T9[1];
  T9[0]
     = reciprocal(T8[0]);
  if ((((((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2)) + ((int)threadIdx.y)) < T0.size[0]) && (i230 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && (((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2)) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
    float T13[(2 * 1)];
    #pragma unroll
    for(int i204 = 0; i204 < 2; ++i204) {
      T13[i204]
        = T6[i204]
        * T9[0];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(int i207 = 0; i207 < 2; ++i207) {
      T10[((((i207 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y)) * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i230]
         = T13[i207];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  } else {
    float T13[(2 * 1)];
    #pragma unroll
    for(int i204 = 0; i204 < 2; ++i204) {
      if (((((((i204 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y)) < T0.size[0]) && (i230 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i204 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
        T13[i204]
          = T6[i204]
          * T9[0];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(int i207 = 0; i207 < 2; ++i207) {
      int i423;
      i423 = ((i207 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y);
      if ((((i423 < T0.size[0]) && (i230 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i207 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 2))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
        T10[(i423 * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i230]
           = T13[i207];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
  // printf("============ t3[1] ===== %f \n", T10[66009]);
}