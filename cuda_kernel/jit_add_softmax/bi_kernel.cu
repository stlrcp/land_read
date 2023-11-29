

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

// aligned register array for vectorized load/store
template <typename scalar_t, int size, int align_size>
struct alignas(sizeof(scalar_t) * align_size) Array {
  scalar_t array[size];

  __device__ void set(scalar_t v) {
#pragma unroll
    for (int i = 0; i < size; ++i) {
      array[i] = v;
    }
  }

  __device__ scalar_t& operator[](const unsigned int i) {
    return array[i];
  }
};

// Type trait utils
template <typename Type, bool is_volatile>
struct MaybeVolatile;

template <typename Type>
struct MaybeVolatile<Type, true> {
  using type = volatile Type;
};

template <typename Type>
struct MaybeVolatile<Type, false> {
  using type = Type;
};


// Volatile version only works with c++ fundamnetal types
template <
    typename scalar_t,
    int vec_size,
    bool is_volatile_to,
    bool is_volatile_from>
__device__ void loadGenericVolatile(
    typename MaybeVolatile<scalar_t, is_volatile_to>::type* to,
    typename MaybeVolatile<scalar_t, is_volatile_from>::type* from) {
  switch (sizeof(scalar_t) * vec_size) {
    // Reinterpret cast like this with volatile types only works for C++
    // fundamental types otherwise the = operator is not defined
    case 1:
      *reinterpret_cast<
          typename MaybeVolatile<unsigned char, is_volatile_to>::type*>(to) =
          *reinterpret_cast<
              typename MaybeVolatile<unsigned char, is_volatile_from>::type*>(
              from);
      break;
    case 2:
      *reinterpret_cast<typename MaybeVolatile<short, is_volatile_to>::type*>(
          to) =
          *reinterpret_cast<
              typename MaybeVolatile<short, is_volatile_from>::type*>(from);
      break;
    case 4:
      *reinterpret_cast<
          typename MaybeVolatile<unsigned int, is_volatile_to>::type*>(to) =
          *reinterpret_cast<
              typename MaybeVolatile<unsigned int, is_volatile_from>::type*>(
              from);
      break;
    case 8:
      *reinterpret_cast<typename MaybeVolatile<double, is_volatile_to>::type*>(
          to) =
          *reinterpret_cast<
              typename MaybeVolatile<double, is_volatile_from>::type*>(from);
      break;
  }
}



template <typename scalar_t, int vec_size, bool is_volatile>
__device__ void loadGlobalToLocal(
    scalar_t* to,
    typename MaybeVolatile<scalar_t, is_volatile>::type* from) {
  switch (sizeof(scalar_t) * vec_size) {
    case 1:
    case 2:
    case 4:
      loadGenericVolatile<scalar_t, vec_size, false, is_volatile>(to, from);
      break;
    case 8: {
      if (is_volatile) {
        uint2& _to = *reinterpret_cast<uint2*>(to);
        uint2& _from = *reinterpret_cast<uint2*>(from);
        _to = _from;
      } else {
        uint2& _to = *reinterpret_cast<uint2*>(to);
        uint2& _from = *reinterpret_cast<uint2*>(from);
        _to = _from;
      }
      break;
    }
    case 12: {
      if (is_volatile) {
        uint3& _to = *reinterpret_cast<uint3*>(to);
        uint3& _from = *reinterpret_cast<uint3*>(from);
        _to = _from;
      } else {
        uint3& _to = *reinterpret_cast<uint3*>(to);
        uint3& _from = *reinterpret_cast<uint3*>(from);
        _to = _from;
      }
      break;
    }
    case 16: {
      if (is_volatile) {
        uint4& _to = *reinterpret_cast<uint4*>(to);
        uint4& _from = *reinterpret_cast<uint4*>(from);
        _to = _from;
      } else {
        uint4& _to = *reinterpret_cast<uint4*>(to);
        uint4& _from = *reinterpret_cast<uint4*>(from);
        _to = _from;
      }
      break;
    }
  }
}


template <typename scalar_t, int vec_size, bool is_volatile>
__device__ void loadLocalToGlobal(
    typename MaybeVolatile<scalar_t, is_volatile>::type* to,
    scalar_t* from) {
  switch (sizeof(scalar_t) * vec_size) {
    case 1:
    case 2:
    case 4:
      loadGenericVolatile<scalar_t, vec_size, is_volatile, false>(to, from);
      break;
    case 8: {
      if (is_volatile) {
        uint2 const& _from = *reinterpret_cast<uint2*>(from);
        uint2 & _to = *reinterpret_cast<uint2*>(to);
        _to = _from;
      } else {
        uint2 const& _from = *reinterpret_cast<uint2*>(from);
        uint2 & _to = *reinterpret_cast<uint2*>(to);
        _to = _from;
      }
      break;
    }
    case 12: {
      if (is_volatile) {
        uint3 const& _from = *reinterpret_cast<uint3*>(from);
        uint3 & _to = *reinterpret_cast<uint3*>(to);
        _to = _from;
      } else {
        uint3 const& _from = *reinterpret_cast<uint3*>(from);
        uint3 & _to = *reinterpret_cast<uint3*>(to);
        _to = _from;
      }
      break;
    }
    case 16: {
      if (is_volatile) {
        uint4 const& _from = *reinterpret_cast<uint4*>(from);
        uint4 & _to = *reinterpret_cast<uint4*>(to);
        _to = _from;
      } else {
        uint4 const& _from = *reinterpret_cast<uint4*>(from);
        uint4 & _to = *reinterpret_cast<uint4*>(to);
        _to = _from;
      }
      break;
    }
  }
}




__global__ void kernel1(Tensor<float, 4> T0, Tensor<float, 4> T1, Tensor<float, 4> T10) {
  alignas(16) extern __shared__ char array[];
  void* shared_mem = array;
  NVFUSER_DEFINE_MAGIC_ZERO
  int i292;
  i292 = ((((int)blockIdx.x) * ((int)blockDim.x)) + ((int)threadIdx.x)) * 2;
  if (((((((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4)) * 3) + ((int)threadIdx.y)) < T0.size[0]) && (((((((int)blockIdx.x) * ((int)blockDim.x)) + ((int)threadIdx.x)) * 2) + 1) < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4)) * 3) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
    Array<float, (((2 * 1) * 4) * 1), 2> T12;
    #pragma unroll
    for(int i235 = 0; i235 < 4; ++i235) {
      loadGlobalToLocal<float, 2, false>(&T12[(i235 * 2)],  &T1[((((i235 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y)) * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i292]);
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    Array<float, (((2 * 1) * 4) * 1), 2> T11;
    #pragma unroll
    for(int i223 = 0; i223 < 4; ++i223) {
      loadGlobalToLocal<float, 2, false>(&T11[(i223 * 2)],  &T0[((((i223 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y)) * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i292]);
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    // Alias Allocation - register
    auto& T13 = T11;
    #pragma unroll
    for(int i250 = 0; i250 < 2; ++i250) {
      float T2[((1 * 4) * 1)];
      #pragma unroll
      for(int i238 = 0; i238 < 4; ++i238) {
        T2[i238]
          = T11[(i238 * 2) + i250]
          + T12[(i238 * 2) + i250];
      }
      float T3[1];
      T3[0] = NEG_INFINITY;
      float T14[1];
      T14[0] = NEG_INFINITY;
      #pragma unroll
      for(int i241 = 0; i241 < 4; ++i241) {
        T14[0] = fmax(
          T14[0],
          T2[i241]);
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
      #pragma unroll
      for(int i245 = 0; i245 < 4; ++i245) {
        float T5[1];
        T5[0]
          = T2[i245]
          - T4[0];
        T6[i245]
           = expf(T5[0]);
      }
      float T7[1];
      T7[0] = 0.00000000000000000e+00;
      float T15[1];
      T15[0] = 0.00000000000000000e+00;
      #pragma unroll
      for(int i248 = 0; i248 < 4; ++i248) {
        T15[0]
          = T15[0]
          + T6[i248];
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
      #pragma unroll
      for(int i252 = 0; i252 < 4; ++i252) {
        T13[(i252 * 2) + i250]
          = T6[i252]
          * T9[0];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(int i257 = 0; i257 < 4; ++i257) {
      loadLocalToGlobal<float, 2, false>( &T10[((((i257 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y)) * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i292], &T13[(i257 * 2)]);
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  } else {
    Array<float, (((2 * 1) * 4) * 1), 2> T12;
    #pragma unroll
    for(int i235 = 0; i235 < 4; ++i235) {
      int i1208;
      i1208 = ((i235 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y);
      int i430;
      i430 = ((i235 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y);
      if ((((i430 < T0.size[0]) && (((((((int)blockIdx.x) * ((int)blockDim.x)) + ((int)threadIdx.x)) * 2) + 1) < (T0.size[1] * (T0.size[2] * T0.size[3])))) && (i1208 < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
        loadGlobalToLocal<float, 2, false>(&T12[(i235 * 2)],  &T1[(i430 * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i292]);
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    Array<float, (((2 * 1) * 4) * 1), 2> T11;
    #pragma unroll
    for(int i223 = 0; i223 < 4; ++i223) {
      int i1293;
      i1293 = ((i223 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y);
      int i455;
      i455 = ((i223 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y);
      if ((((i455 < T0.size[0]) && (((((((int)blockIdx.x) * ((int)blockDim.x)) + ((int)threadIdx.x)) * 2) + 1) < (T0.size[1] * (T0.size[2] * T0.size[3])))) && (i1293 < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
        loadGlobalToLocal<float, 2, false>(&T11[(i223 * 2)],  &T0[(i455 * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i292]);
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    // Alias Allocation - register
    auto& T13 = T11;
    #pragma unroll
    for(int i250 = 0; i250 < 2; ++i250) {
      int i1335;
      i1335 = (((((int)blockIdx.x) * ((int)blockDim.x)) + ((int)threadIdx.x)) * 2) + (i250 + nvfuser_zero);
      float T2[((1 * 4) * 1)];
      #pragma unroll
      for(int i238 = 0; i238 < 4; ++i238) {
        if (((((((i238 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y)) < T0.size[0]) && (i1335 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i238 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
          T2[i238]
            = T11[(i238 * 2) + i250]
            + T12[(i238 * 2) + i250];
        }
      }
      float T3[1];
      T3[0] = NEG_INFINITY;
      float T14[1];
      T14[0] = NEG_INFINITY;
      #pragma unroll
      for(int i241 = 0; i241 < 4; ++i241) {
        if (((((((i241 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y)) < T0.size[0]) && (i1335 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i241 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
          T14[0] = fmax(
            T14[0],
            T2[i241]);
        }
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
      #pragma unroll
      for(int i245 = 0; i245 < 4; ++i245) {
        int i1445;
        i1445 = ((i245 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y);
        int i1429;
        i1429 = ((i245 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y);
        float T5[1];
        if ((((i1429 < T0.size[0]) && (i1335 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && (i1445 < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
          T5[0]
            = T2[i245]
            - T4[0];
        }
        if ((((i1429 < T0.size[0]) && (i1335 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && (i1445 < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
          T6[i245]
             = expf(T5[0]);
        }
      }
      float T7[1];
      T7[0] = 0.00000000000000000e+00;
      float T15[1];
      T15[0] = 0.00000000000000000e+00;
      #pragma unroll
      for(int i248 = 0; i248 < 4; ++i248) {
        if (((((((i248 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y)) < T0.size[0]) && (i1335 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i248 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
          T15[0]
            = T15[0]
            + T6[i248];
        }
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
      #pragma unroll
      for(int i252 = 0; i252 < 4; ++i252) {
        if (((((((i252 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y)) < T0.size[0]) && (i1335 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i252 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
          T13[(i252 * 2) + i250]
            = T6[i252]
            * T9[0];
        }
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(int i257 = 0; i257 < 4; ++i257) {
      int i1684;
      i1684 = ((i257 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y);
      int i549;
      i549 = ((i257 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 1)), 1)), 4))) + ((int)threadIdx.y);
      if ((((i549 < T0.size[0]) && (((((((int)blockIdx.x) * ((int)blockDim.x)) + ((int)threadIdx.x)) * 2) + 1) < (T0.size[1] * (T0.size[2] * T0.size[3])))) && (i1684 < (ceilDiv((ceilDiv(T0.size[0], 1)), 1))))) {
        loadLocalToGlobal<float, 2, false>( &T10[(i549 * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i292], &T13[(i257 * 2)]);
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
}