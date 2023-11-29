

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
  int i228;
  i228 = (((int)blockIdx.x) * ((int)blockDim.x)) + ((int)threadIdx.x);
  float T2[((2 * 1) * 4)];
  #pragma unroll
  for(int i189 = 0; i189 < 2; ++i189) {
    if (((((((i189 + nvfuser_zero) * (ceilDiv(T0.size[0], 2))) + (((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4)) * 3) + ((int)threadIdx.y))) < T0.size[0]) && (i228 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4)) * 3) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 2)), 1))))) {
      float T12[4];
      #pragma unroll
      for(int i188 = 0; i188 < 4; ++i188) {
        T12[i188]
           = T1[(((i189 * (ceilDiv(T0.size[0], 2))) + (((i188 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y))) * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i228];
      }
      float T11[4];
      #pragma unroll
      for(int i181 = 0; i181 < 4; ++i181) {
        T11[i181]
           = T0[(((i189 * (ceilDiv(T0.size[0], 2))) + (((i181 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y))) * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i228];
      }
      #pragma unroll
      for(int i191 = 0; i191 < 4; ++i191) {
        T2[((i189 * 4) + i191)]
          = T11[i191]
          + T12[i191];
      }
    } else {
      float T12[4];
      #pragma unroll
      for(int i188 = 0; i188 < 4; ++i188) {
        int i288;
        i288 = (i189 * (ceilDiv(T0.size[0], 2))) + (((i188 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y));
        if ((((i288 < T0.size[0]) && (i228 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i188 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 2)), 1))))) {
          T12[i188]
             = T1[(i288 * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i228];
        }
      }
      float T11[4];
      #pragma unroll
      for(int i181 = 0; i181 < 4; ++i181) {
        int i309;
        i309 = (i189 * (ceilDiv(T0.size[0], 2))) + (((i181 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y));
        if ((((i309 < T0.size[0]) && (i228 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i181 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 2)), 1))))) {
          T11[i181]
             = T0[(i309 * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i228];
        }
      }
      #pragma unroll
      for(int i191 = 0; i191 < 4; ++i191) {
        if ((((((i189 * (ceilDiv(T0.size[0], 2))) + (((i191 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y))) < T0.size[0]) && (i228 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i191 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 2)), 1))))) {
          T2[((i189 * 4) + i191)]
            = T11[i191]
            + T12[i191];
        }
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  float T3[1];
  T3[0] = NEG_INFINITY;
  float T14[1];
  T14[0] = NEG_INFINITY;
  #pragma unroll
  for(int i192 = 0; i192 < 2; ++i192) {
    if (((((((i192 + nvfuser_zero) * (ceilDiv(T0.size[0], 2))) + (((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4)) * 3) + ((int)threadIdx.y))) < T0.size[0]) && (i228 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4)) * 3) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 2)), 1))))) {
      #pragma unroll
      for(int i194 = 0; i194 < 4; ++i194) {
        T14[0] = fmax(
          T14[0],
          T2[((i192 * 4) + i194)]);
      }
    } else {
      #pragma unroll
      for(int i194 = 0; i194 < 4; ++i194) {
        if ((((((i192 * (ceilDiv(T0.size[0], 2))) + (((i194 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y))) < T0.size[0]) && (i228 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i194 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 2)), 1))))) {
          T14[0] = fmax(
            T14[0],
            T2[((i192 * 4) + i194)]);
        }
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
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
  for(int i196 = 0; i196 < 2; ++i196) {
    if (((((((i196 + nvfuser_zero) * (ceilDiv(T0.size[0], 2))) + (((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4)) * 3) + ((int)threadIdx.y))) < T0.size[0]) && (i228 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4)) * 3) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 2)), 1))))) {
      #pragma unroll
      for(int i198 = 0; i198 < 4; ++i198) {
        int i398;
        i398 = (i196 * 4) + i198;
        float T5[1];
        T5[0]
          = T2[i398]
          - T4[0];
        T6[i398]
           = expf(T5[0]);
      }
    } else {
      #pragma unroll
      for(int i198 = 0; i198 < 4; ++i198) {
        int i1247;
        i1247 = (i196 * (ceilDiv(T0.size[0], 2))) + (((i198 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y));
        int i433;
        i433 = (i196 * 4) + i198;
        int i1258;
        i1258 = ((i198 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y);
        float T5[1];
        if ((((i1247 < T0.size[0]) && (i228 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && (i1258 < (ceilDiv((ceilDiv(T0.size[0], 2)), 1))))) {
          T5[0]
            = T2[i433]
            - T4[0];
        }
        if ((((i1247 < T0.size[0]) && (i228 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && (i1258 < (ceilDiv((ceilDiv(T0.size[0], 2)), 1))))) {
          T6[i433]
             = expf(T5[0]);
        }
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  float T7[1];
  T7[0] = 0.00000000000000000e+00;
  float T15[1];
  T15[0] = 0.00000000000000000e+00;
  #pragma unroll
  for(int i199 = 0; i199 < 2; ++i199) {
    if (((((((i199 + nvfuser_zero) * (ceilDiv(T0.size[0], 2))) + (((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4)) * 3) + ((int)threadIdx.y))) < T0.size[0]) && (i228 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4)) * 3) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 2)), 1))))) {
      #pragma unroll
      for(int i201 = 0; i201 < 4; ++i201) {
        T15[0]
          = T15[0]
          + T6[((i199 * 4) + i201)];
      }
    } else {
      #pragma unroll
      for(int i201 = 0; i201 < 4; ++i201) {
        if ((((((i199 * (ceilDiv(T0.size[0], 2))) + (((i201 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y))) < T0.size[0]) && (i228 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i201 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 2)), 1))))) {
          T15[0]
            = T15[0]
            + T6[((i199 * 4) + i201)];
        }
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
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
  for(int i203 = 0; i203 < 2; ++i203) {
    if (((((((i203 + nvfuser_zero) * (ceilDiv(T0.size[0], 2))) + (((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4)) * 3) + ((int)threadIdx.y))) < T0.size[0]) && (i228 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4)) * 3) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 2)), 1))))) {
      float T13[4];
      #pragma unroll
      for(int i202 = 0; i202 < 4; ++i202) {
        T13[i202]
          = T6[((i203 * 4) + i202)]
          * T9[0];
      }
      #pragma unroll
      for(int i205 = 0; i205 < 4; ++i205) {
        T10[(((i203 * (ceilDiv(T0.size[0], 2))) + (((i205 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y))) * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i228]
           = T13[i205];
      }
    } else {
      float T13[4];
      #pragma unroll
      for(int i202 = 0; i202 < 4; ++i202) {
        if ((((((i203 * (ceilDiv(T0.size[0], 2))) + (((i202 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y))) < T0.size[0]) && (i228 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i202 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 2)), 1))))) {
          T13[i202]
            = T6[((i203 * 4) + i202)]
            * T9[0];
        }
      }
      #pragma unroll
      for(int i205 = 0; i205 < 4; ++i205) {
        int i617;
        i617 = (i203 * (ceilDiv(T0.size[0], 2))) + (((i205 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y));
        if ((((i617 < T0.size[0]) && (i228 < (T0.size[1] * (T0.size[2] * T0.size[3])))) && ((((i205 + nvfuser_zero) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], 2)), 1)), 4))) + ((int)threadIdx.y)) < (ceilDiv((ceilDiv(T0.size[0], 2)), 1))))) {
          T10[(i617 * ((T0.size[3] * T0.size[2]) * T0.size[1])) + i228]
             = T13[i205];
        }
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
}