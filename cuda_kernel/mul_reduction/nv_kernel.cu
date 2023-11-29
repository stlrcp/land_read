

#ifdef __ILUVATAR__
#define POS_INFINITY INFINITY
#define NEG_INFINITY -INFINITY
#else
#define POS_INFINITY __int_as_float(0x7f800000)
#define INFINITY POS_INFINITY
#define NEG_INFINITY __int_as_float(0xff800000)
#define NAN __int_as_float(0x7fffffff)
#endif

#define NVFUSER_UPDATE_MAGIC_ZERO \
  do {                            \
    nvfuser_zero <<= 1;           \
  } while (0);

#define NVFUSER_DEFINE_MAGIC_ZERO          \
  __shared__ int nvfuser_zero_s;           \
  if (threadIdx.x == 0)                    \
    nvfuser_zero_s = 0;                    \
  __syncthreads();                         \
  atomicMin(&nvfuser_zero_s, threadIdx.x); \
  int nvfuser_zero = nvfuser_zero_s;


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


template <typename T, int N>
struct Tensor {
  __device__ T& operator[](int ind) {
    return data[ind];
  };

  T* data;
  int size[N];
  int stride[N];
};

// Specialization for 0-dim case as it does not need size and stride arrays.
// They will be an error as well since zero-length arrays are not allowed.
template <typename T>
struct Tensor<T, 0> {
  __device__ T& operator[](int) {
    return *data;
  };

  T* data;
};


namespace block_sync {

__forceinline__ __device__ void init() {}

// Thread-block synchronization
__forceinline__ __device__ void sync() {
  __syncthreads();
}

} // namespace block_sync

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

template <bool X_REDUCE, bool Y_REDUCE, bool Z_REDUCE, typename T, typename Func, typename _dim3, typename _dim3_2>
__device__ void blockReduce(T &out, const T &inp_val, Func reduction_op, const _dim3 &thread_idx,
    const _dim3_2 &block_dim, T *shared_mem, bool read_pred, bool write_pred, T init_val)
{
    // If this thread will output a final result
    bool should_write =
        index_utils::maskedIsZero<X_REDUCE, Y_REDUCE, Z_REDUCE>(thread_idx);

    // Size of the reduction segments
    unsigned int reduction_size =
        index_utils::maskedSize<X_REDUCE, Y_REDUCE, Z_REDUCE>(block_dim);

    // Index into the reduction segment
    unsigned int reduction_tid =
        index_utils::maskedOffset<X_REDUCE, Y_REDUCE, Z_REDUCE>(
            thread_idx, block_dim);

    // Index of the reduction segment
    unsigned int reduction_idx =
        index_utils::maskedOffset<!X_REDUCE, !Y_REDUCE, !Z_REDUCE>(
            thread_idx, block_dim);

    // Offset into smem for the current thread
    unsigned int smem_offset = reduction_idx * reduction_size + reduction_tid;

    // Initialize shared memory
    if (read_pred)
    {
        shared_mem[smem_offset] = inp_val;
    }
    else
    {
        shared_mem[smem_offset] = init_val;
    }

    block_sync::sync();
    // Reduce down to nearest power of 2 for the tree reduction:
    int np2 = 1 << (31 - __clz(reduction_size));

    if (reduction_tid < np2 && reduction_tid + np2 < reduction_size)
    {
        reduction_op(shared_mem[smem_offset], shared_mem[smem_offset + np2]);
    }
    block_sync::sync();

    // loop peel the final iteration to save one syncthread for the end
    for (int factor = np2 / 2; factor > 1; factor >>= 1)
    {
        if (reduction_tid < factor)
        {
            reduction_op(shared_mem[smem_offset], shared_mem[smem_offset + factor]);
        }
        block_sync::sync();
    }

    if (should_write && write_pred)
    {
        T result = out;
        reduction_op(result, shared_mem[smem_offset]);
        if (reduction_size > 1)
        {
            reduction_op(result, shared_mem[smem_offset + 1]);
        }
        out = result;
    }
    block_sync::sync();
}

// Use the same pred for both reads and writes
template <bool X_REDUCE, bool Y_REDUCE, bool Z_REDUCE, typename T, typename Func, typename _dim3, typename _dim3_2>
__device__ void blockReduce(T &out, const T &inp_val, Func reduction_op, const _dim3 &thread_idx, 
    const _dim3_2 &block_dim, T *shared_mem, bool read_write_pred, T init_val)
{
    blockReduce<X_REDUCE, Y_REDUCE, Z_REDUCE, T, Func, _dim3, _dim3_2>(
        out,
        inp_val,
        reduction_op,
        thread_idx,
        block_dim,
        shared_mem,
        read_write_pred,
        read_write_pred,
        init_val);
}

__global__ void kernel1(Tensor<float, 3> T0, Tensor<float, 2> T2)
{
    alignas(16) extern __shared__ char array[];
    void *shared_mem = array;
    NVFUSER_DEFINE_MAGIC_ZERO
    int i75;
    i75 = (((int)blockIdx.x) * ((int)blockDim.x)) + ((int)threadIdx.x);
    float T5[1];
    T5[0] = NEG_INFINITY;
    #pragma unroll 1
    for (int i52 = 0; i52 < (ceilDiv((ceilDiv((ceilDiv(T0.size[0], ((int)blockDim.y))), 4)), 1)); ++i52)
    {
        if (((((((i52 * 4) + 3) * ((int)blockDim.y)) + ((int)threadIdx.y)) < T0.size[0]) && (i75 < (T0.size[1] * T0.size[2]))))
        {
            float T3[4];
            #pragma unroll
            for (int i51 = 0; i51 < 4; ++i51)
            {
                T3[i51] = 0;
            }
            NVFUSER_UPDATE_MAGIC_ZERO
            #pragma unroll
            for (int i51 = 0; i51 < 4; ++i51)
            {
                T3[i51] = T0[(((((i52 * 4) + (i51 + nvfuser_zero)) * ((int)blockDim.y)) + ((int)threadIdx.y)) * (T0.size[2] * T0.size[1])) + i75];
            }
            NVFUSER_UPDATE_MAGIC_ZERO
            #pragma unroll
            for (int i54 = 0; i54 < 4; ++i54)
            {
                float T1[1];
                T1[0] = T3[i54] * (float)2.00000000000000000e+00;
                T5[0] = fmax(
                    T5[0],
                    T1[0]);
            }
            NVFUSER_UPDATE_MAGIC_ZERO
        }
        else
        {
            float T3[4];
            #pragma unroll
            for (int i51 = 0; i51 < 4; ++i51)
            {
                T3[i51] = 0;
            }
            NVFUSER_UPDATE_MAGIC_ZERO
            #pragma unroll
            for (int i51 = 0; i51 < 4; ++i51)
            {
                int i108;
                i108 = (((i52 * 4) + (i51 + nvfuser_zero)) * ((int)blockDim.y)) + ((int)threadIdx.y);
                if (((i108 < T0.size[0]) && (i75 < (T0.size[1] * T0.size[2]))))
                {
                    T3[i51] = T0[(i108 * (T0.size[2] * T0.size[1])) + i75];
                }
            }
            NVFUSER_UPDATE_MAGIC_ZERO
            #pragma unroll
            for (int i54 = 0; i54 < 4; ++i54)
            {
                float T1[1];
                T1[0] = T3[i54] * (float)2.00000000000000000e+00;
                if (((((((i52 * 4) + (i54 + nvfuser_zero)) * ((int)blockDim.y)) + ((int)threadIdx.y)) < T0.size[0]) && (i75 < (T0.size[1] * T0.size[2]))))
                {
                    T5[0] = fmax(
                        T5[0],
                        T1[0]);
                }
            }
            NVFUSER_UPDATE_MAGIC_ZERO
        }
    }
    float T4[1];
    T4[0] = NEG_INFINITY;
    blockReduce<false, true, false>(
        T4[0],
        T5[0],
        [](float &a, float b)
        { a = fmax(a, b); },
        threadIdx,
        blockDim,
        static_cast<float *>(shared_mem),
        true,
        true,
        float(NEG_INFINITY));
    if (((i75 < (T0.size[1] * T0.size[2])) && (((int)threadIdx.y) == 0)))
    {
        T2[i75] = T4[0];
    }
}