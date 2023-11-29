

#ifdef __ILUVATAR__
#define POS_INFINITY INFINITY
#define NEG_INFINITY -INFINITY
#else
#define POS_INFINITY __int_as_float(0x7f800000)
#define INFINITY POS_INFINITY
#define NEG_INFINITY __int_as_float(0xff800000)
#define NAN __int_as_float(0x7fffffff)
#endif


#ifndef __ILUVATAR__
// typedef long long int int64_t;
typedef unsigned long long int uint64_t;
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


namespace grid_sync {

// Get the first bit in a 64 bit integer
#define FIRST_UINT64_BIT ((uint64_t)1 << (sizeof(uint64_t) * 8 - 1))
#define LOW_UINT64_BIT ((uint32_t)((FIRST_UINT64_BIT) & 0xFFFFFFFF))
#define HIGH_UINT64_BIT ((uint32_t)(((FIRST_UINT64_BIT) >> 32) & 0xFFFFFFFF))

template <typename T>
__device__ T globalAsVolatile(volatile T& global_val) {
  return global_val;
}

// A grid synchronization that can be called multiple times in a kernel assuming
// all the blocks fit on device at once. The semaphore is an integer semaphore
// assumed to be initialized to 0 before launching the kernel. The persistent
// option should be envoked if this sync will be called multiple times in one
// kernel (i.e. having a grid reduce within a loop). Having multiple grid syncs
// called once in the same kernel does not require persistent mode. Segment size
// is the number of blocks participating in the sync in the dimensions marked by
// [X,Y,Z]_BLOCK. The granularity of this sync are those dimensions. I.E.
// Marking X and Y but not Z means there should be Z semaphores of size X*Y.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, bool PERSISTENT>
__device__ void sync(
    int64_t& semaphore,
    const uint64_t& segment_size,
    const bool last_block) {
  // Finish all global memory transactions before synchronizing
  __threadfence();

  // Synchronize all threads in a block before synchronizing blocks
  block_sync::sync();

  // Only allow linear_tid == 0 to participate in the synchronization
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    // Get increment value, only want a single block to have the large
    // increment, doesn't really matter which one, the goal is to flip/flop the
    // first bit of a uint64_t value, since our semaphores are actualy int64_t
    // we will just reinterpret_cast it to act as a uint64_t
    uint64_t semaphore_increment = 1;

    // Makes the assumption that blocks are in increasing order, this is not
    // guaranteed by CUDA but this is the current behavior, and unlikely to
    // change.
    if (last_block) {
      semaphore_increment = FIRST_UINT64_BIT - (segment_size - 1);
    }

  #ifdef __ILUVATAR__
    // operate last_block after other blocks complete atomicAdd() (will fail if segment_size > 0x7FFFFFFF)
    uint2 *address_uint2 = reinterpret_cast<uint2 *>(&semaphore);
    const uint2 &val_uint2 = reinterpret_cast<uint2 &>(semaphore_increment);
    uint2 old_val_uint2;
    if (!last_block) {
        old_val_uint2.x = atomicAdd(&((*address_uint2).x), val_uint2.x);
        if (old_val_uint2.x + val_uint2.x < val_uint2.x) {
            old_val_uint2.y = atomicAdd(&((*address_uint2).y), (1 + val_uint2.y));
        } else {
            old_val_uint2.y = atomicAdd(&((*address_uint2).y), val_uint2.y);
        }
    } else {
        while (globalAsVolatile(semaphore) != (segment_size - 1)) {
        }
        old_val_uint2.x = atomicExch(&((*address_uint2).x), LOW_UINT64_BIT);
        old_val_uint2.y = atomicExch(&((*address_uint2).y), HIGH_UINT64_BIT);
    }
    uint64_t oldArrive = reinterpret_cast<uint64_t &>(old_val_uint2);
  #else
    uint64_t oldArrive =
        atomicAdd(reinterpret_cast<uint64_t*>(&semaphore), semaphore_increment);
  #endif

    // If for persistent kernels, lock all blocks until the semaphore has been
    // reached. Make sure we access semaphore as a volatile address so we get
    // the global memory updates.
    unsigned int ns = 8;
    while ((PERSISTENT || last_block) &&
           ((oldArrive ^ globalAsVolatile(semaphore)) & FIRST_UINT64_BIT) ==
               0) {
      // Put a sleep here so we have some breaks in probing the global
      // semaphore, giving a better chance for other warps/blocks to catch up.
#if __CUDA_ARCH__ >= 700
      // __nanosleep only available on compute capability 7.0 or higher
      __nanosleep(ns); // avoids busy waiting
      if (ns < 256) {
        ns *= 2;
      }
#endif
    }
  }

  // Sync block to make sure all other threads are waiting on the sync
  block_sync::sync();
}

template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, bool PERSISTENT>
__device__ void sync(int64_t& semaphore, const uint64_t& segment_size) {
  sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT>(
      semaphore,
      segment_size,
      index_utils::maskedIsLast<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim));
}

// Grid sync that can be called multiple times in the same kernel without all
// blocks being resident on device. This allows grid sync to be called multiple
// times as long as it's not broadcasted on the parallel axis it was reduced on.
//
// n_entrances is how many times every block is expected to enter into this
// function. All blocks must enter n_entrances times. The last block is only
// allowed to proceed once all other blocks have entered n_entrance
// times.
//
// Note that this is not currently used by grid and welford reduction
// as they use a separate sync flag for each each grid sync call.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK>
__device__ void sync(
    int64_t& semaphore,
    const uint64_t& segment_size,
    const int n_entrances) {
  // Finish all global memory transactions before synchronizing
  __threadfence();

  // Synchronize all threads in a block before synchronizing blocks
  block_sync::sync();

  // Only allow linear_tid == 0 to participate in the synchronization
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    // Makes the assumption that blocks are in increasing order, this is not
    // guaranteed by CUDA but this is the current behavior, and unlikely to
    // change.
    bool last_block =
        index_utils::maskedIsLast<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
    if (last_block) {
      int64_t finished_val =
          ((int64_t)(
              index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim) -
              1)) *
          ((int64_t)n_entrances);

      unsigned int ns = 8;
      // Last block needs to wait for all other blocks to finish
      while (globalAsVolatile(semaphore) < finished_val) {
#if __CUDA_ARCH__ >= 700
        // __nanosleep only available on compute capability 7.0 or higher
        __nanosleep(ns); // avoids busy waiting
        if (ns < 256) {
          ns *= 2;
        }
#endif
      }
    } else {
#ifdef __ILUVATAR__
      auto old = atomicAdd(reinterpret_cast<uint32_t*>(&semaphore), 1);
#else
      auto old = atomicAdd(reinterpret_cast<uint64_t*>(&semaphore), 1);
#endif
    }
  }

  // Sync block to make sure all other threads are waiting on the sync
  block_sync::sync();
}

} // namespace grid_sync




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


namespace reduction {

// Reduces all the reduction blocks in each reduction segment. This is the
// "cleanup" stage of a grid reduction.
//
// This is only called by one thread block per reduction segment. The input
// reduction blocks of the segment are stored in an intermediate buffer pointed
// by parameter in. Template parameters X/Y/Z_THREAD denote how the reduction
// block is formed.
//
// The size of a reduction block is by definition smaller or equal to the size
// of a thread block. We use the remaining threads to parallelize reductions
// across reduction blocks. For example, when X/Y/Z_THREAD = {true, false,
// false}, we use blockDim.y*blockDim.z threads for each output value. This is
// done first by loading the input values in parallel and then by reducing
// across threads of dimensions whose XYZ_THREAD are false.
//
// Note that what is done here after the loading from global memory is similar
// to what the existing blockReduce function does.
template <
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    typename T,
    typename Func>
__device__ void gridReduceLastBlock(
    T& out,
    const volatile T* in,
    const int
        grid_reduction_segment_size, // Number of reductions across
                                     // grid reduce dimensions
    const int
        block_reduction_segment_size, // Number of reductions across the block
    Func reduction_op,
    T* shared_buf,
    bool write_pred,
    T init_val) {
  // We have to do num_reductions across reduction_size. The reductions are
  // contiguous, but offset by reduction_size. There is an entry in "in" for
  // every block, and every thread marked as true. Threads in dimensions marked
  // as false can be used to parallelize the reduction.

  // Find the reduction id of the participating threads
  const auto block_reduction_segment_idx =
      index_utils::maskedOffset<X_THREAD, Y_THREAD, Z_THREAD>(
          threadIdx, blockDim);

  // Find an id associated within a reduction segment for all
  // "non-participating" threads, which will parallelize the reductions for the
  // "participating" threads
  const auto id_in_block_segment =
      index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
          threadIdx, blockDim);

  // Stride by the "non-participating" threads
  const auto input_stride_for_thread_in_segment =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(blockDim);

  T inp = init_val;

  // Block stride across the reduction until we only have one value per thread
  for (int reduction_i = id_in_block_segment;
       reduction_i < grid_reduction_segment_size;
       reduction_i += input_stride_for_thread_in_segment) {
    auto work_buf_offset = reduction_i * block_reduction_segment_size +
        block_reduction_segment_idx;
    reduction_op(inp, in[work_buf_offset]);
  }

  // Block reduce the per thread values into per "participating" thread values
  T inp_tmp = init_val;
  blockReduce<!X_THREAD, !Y_THREAD, !Z_THREAD>(
      inp_tmp,
      inp,
      reduction_op,
      threadIdx,
      blockDim,
      shared_buf,
      true,
      init_val);
  const bool should_write = (X_THREAD || threadIdx.x == 0) &&
      (Y_THREAD || threadIdx.y == 0) && (Z_THREAD || threadIdx.z == 0);
  if (should_write && write_pred) {
    reduction_op(out, inp_tmp);
  }
}

// Reduces per-thread values across threads and thread blocks.
//
// Function parameters:
// - out: Per-thread output location
// - inp_val: Per-thread input value
// - reduction_op: Scalar reduction function
// - work_buf: Temporary buffer for cross-block reductions
// - sync_flags: A vector of integers for synchronizations
// - shared_buf: Shared memory buffer for intra-block reduction
//
// Thread has valid results based on if it's the last block in the grid
// reduction dimension
//
// Template parameters:
// - X/Y/Z_BLOCK/THREAD: When true, reduces across thread blocks along the X/Y/Z
//   dimensions
// - PERSISTENT_REDUCTION: Indicates grid reduction will be called in a loop, or
//   the result of the grid reduction will be broadcasted and used across the
//   grid. These requires cross grid communication and the grid synchronizations
//   here to actually synchronize across the entire grid. When false the grid is
//   not synchronized, the last block just waits for everyone else to finish and
//   the other blocks can exit early.
// - T: Scalar data type of input/output data
// - Func: Type of scalara reduction function
//
// Template parameters X/Y/Z_BLOCK define a group of thread blocks that are
// reduced together. We call it a reduction segment. Some examples are:
//
// Case 1: X/Y/Z_BLOCK == true/true/true -> There is only one segment, which
// includes all thread blocks. It is effecively the same as the grid.
//
// Case 2: X/Y/Z_BLOCK == false/false/false -> Each thread block comprises an
// individual segment by itself.
//
// Case 3: X/Y/Z_BLOCK == true/false/false -> Each segment contains thread
// blocks that have the same blockDim.x. There will be blockDim.y*blockDim.z
// such segments.
//
// X/Y/Z_THREAD also works similarly as X/Y/Z_BLOCK and defines a
// group of threads that are reduced togather.
//
// After the function completes, only one thread block per reduction segment
// gets valid reduction results. There is no guarantee which particular block
// gets the final results.
//
// entrance_ind and n_entrances are allowed when PERSISTENT_REDUCTION = false.
// If a grid reduction call is only called once per thread, entrance_ind == 0
// and n_entrances == 1. However, grid reduction can be called in a loop in a
// thread, in that case entrance_ind is the count of times the function has been
// called, and n_entrances is the total number of times it will be called.
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool PERSISTENT_REDUCTION,
    typename T,
    typename Func>
__device__ void gridReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    volatile T* work_buf,
    int64_t* sync_flags,
    T* shared_buf,
    bool read_pred,
    bool write_pred,
    T init_val,
    const int entrance_ind,
    const int n_entrances) {
  T block_reduction_val = init_val;

  // Do block reduction when required
  if (X_THREAD || Y_THREAD || Z_THREAD) {
    blockReduce<X_THREAD, Y_THREAD, Z_THREAD>(
        block_reduction_val,
        inp_val,
        reduction_op,
        threadIdx,
        blockDim,
        shared_buf,
        read_pred,
        true,
        init_val);
  } else if (read_pred) {
    block_reduction_val = inp_val;
  }

  // Number of values to reduce in the reduction segment
  const auto grid_reduction_segment_size =
      index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);

  // Index of the reduction we're performing out of the
  // grid_reduction_segment_size
  const auto idx_in_grid_segment =
      index_utils::maskedOffset<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(
          blockIdx, gridDim);

  // Number of threads we can use in final reduction, Seems to assume all
  // threads in the block participate
  const auto block_reduction_segment_size =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(blockDim);

  // Number of reductions in the grid
  const int grid_segment_size = PERSISTENT_REDUCTION
      ? 1
      : index_utils::maskedSize<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(gridDim);

  // advance to the offset for this segment
  // index of reduction * size of the reduction * size of threads
  work_buf += (entrance_ind * grid_segment_size + idx_in_grid_segment) *
      grid_reduction_segment_size * block_reduction_segment_size;

  if ((!X_THREAD || threadIdx.x == 0) && (!Y_THREAD || threadIdx.y == 0) &&
      (!Z_THREAD || threadIdx.z == 0)) {
    auto block_offset =
        index_utils::maskedOffset<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
    auto thread_offset =
        index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
            threadIdx, blockDim);
    auto work_buf_offset =
        block_offset * block_reduction_segment_size + thread_offset;
    work_buf[work_buf_offset] = block_reduction_val;
  }
  if (PERSISTENT_REDUCTION) {
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION>(
        sync_flags[idx_in_grid_segment], grid_reduction_segment_size);

  } else {
    // Use a different sync flag for each call
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION>(
        sync_flags[entrance_ind * grid_segment_size + idx_in_grid_segment],
        grid_reduction_segment_size);
  }

  bool last_block =
      index_utils::maskedIsLast<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);

  if (last_block) {
    // Cleanup with block reduction
    gridReduceLastBlock<!X_THREAD, !Y_THREAD, !Z_THREAD>(
        out,
        (T*)work_buf,
        grid_reduction_segment_size,
        block_reduction_segment_size,
        reduction_op,
        shared_buf,
        write_pred,
        init_val);
  }

  if (PERSISTENT_REDUCTION) {
    // Make sure we're done with global memory before we allow the kernel to
    // continue
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION>(
        sync_flags[idx_in_grid_segment], grid_reduction_segment_size);
  }
}

// This is just a wrapper of the above grid reduction routine to
// measure the elapsed cycles. The measurement must be done just by
// one thread, and in this case it should be done by one of the
// threads in the last thread block.
#ifdef PYTORCH_NVFUSER_PROFILE_KERNEL
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool PERSISTENT_REDUCTION,
    typename T,
    typename Func>
__device__ void gridReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    volatile T* work_buf,
    int64_t* sync_flags,
    T* shared_buf,
    bool read_pred,
    bool write_pred,
    T init_val,
    const int entrance_ind,
    const int n_entrances,
    int64_t& cycles,
    int64_t& count) {
  int64_t start_counter = 0;

  if (index_utils::maskedIsLast<true, true, true>(blockIdx, gridDim) &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    start_counter = readCycleCounter();
  }

  gridReduce<
      X_BLOCK,
      Y_BLOCK,
      Z_BLOCK,
      X_THREAD,
      Y_THREAD,
      Z_THREAD,
      PERSISTENT_REDUCTION,
      T,
      Func>(
      out,
      inp_val,
      reduction_op,
      work_buf,
      sync_flags,
      shared_buf,
      read_pred,
      write_pred,
      init_val,
      entrance_ind,
      n_entrances);

  if (index_utils::maskedIsLast<true, true, true>(blockIdx, gridDim) &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    cycles += readCycleCounter() - start_counter;
    ++count;
  }
}
#endif // PYTORCH_NVFUSER_PROFILE_KERNEL

template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    typename T,
    typename Func>
__device__ void gridReduce2PartialReduction(
    const T& inp_val,
    T init_val,
    Func reduction_op,
    volatile T* work_buf,
    T* shared_buf,
    bool read_pred,
    int grid_reduction_segment_size,
    int idx_in_grid_segment,
    int block_reduction_segment_size) {
  T block_reduction_val = init_val;

  // Do block reduction when required
  if (X_THREAD || Y_THREAD || Z_THREAD) {
    blockReduce<X_THREAD, Y_THREAD, Z_THREAD>(
        block_reduction_val,
        inp_val,
        reduction_op,
        threadIdx,
        blockDim,
        shared_buf,
        read_pred,
        true,
        init_val);
  } else if (read_pred) {
    block_reduction_val = inp_val;
  }

  if ((!X_THREAD || threadIdx.x == 0) && (!Y_THREAD || threadIdx.y == 0) &&
      (!Z_THREAD || threadIdx.z == 0)) {
    auto block_offset =
        index_utils::maskedOffset<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
    auto thread_offset =
        index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
            threadIdx, blockDim);
    auto work_buf_offset =
        block_offset * block_reduction_segment_size + thread_offset;
    work_buf[work_buf_offset] = block_reduction_val;
  }
}

// 2-way horizontally fused grid reduction
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool PERSISTENT_REDUCTION,
    typename T1,
    typename Func1,
    typename T2,
    typename Func2>
__device__ void gridReduceGroup(
    T1& out1,
    const T1& inp_val1,
    T1 init_val1,
    Func1 reduction_op1,
    volatile T1* work_buf1,
    T2& out2,
    const T2& inp_val2,
    T2 init_val2,
    Func2 reduction_op2,
    volatile T2* work_buf2,
    int64_t* sync_flags,
    void* shared_buf,
    bool read_pred,

    bool write_pred,
    const int entrance_ind,
    const int n_entrances) {
  // Number of values to reduce in the reduction segment
  const auto grid_reduction_segment_size =
      index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);

  // Index of the reduction we're performing out of the
  // grid_reduction_segment_size
  const auto idx_in_grid_segment =
      index_utils::maskedOffset<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(
          blockIdx, gridDim);

  // Number of threads we can use in final reduction, Seems to assume all
  // threads in the block participate
  const auto block_reduction_segment_size =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(blockDim);

  // Number of reductions in the grid
  const int grid_segment_size = PERSISTENT_REDUCTION
      ? 1
      : index_utils::maskedSize<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(gridDim);

  // advance to the offset for this segment
  // index of reduction * size of the reduction * size of threads
  work_buf1 += (entrance_ind * grid_segment_size + idx_in_grid_segment) *
      grid_reduction_segment_size * block_reduction_segment_size;

  work_buf2 += (entrance_ind * grid_segment_size + idx_in_grid_segment) *
      grid_reduction_segment_size * block_reduction_segment_size;

  gridReduce2PartialReduction<
      X_BLOCK,
      Y_BLOCK,
      Z_BLOCK,
      X_THREAD,
      Y_THREAD,
      Z_THREAD>(
      inp_val1,
      init_val1,
      reduction_op1,
      work_buf1,
      (T1*)shared_buf,
      read_pred,
      grid_reduction_segment_size,
      idx_in_grid_segment,
      block_reduction_segment_size);

  gridReduce2PartialReduction<
      X_BLOCK,
      Y_BLOCK,
      Z_BLOCK,
      X_THREAD,
      Y_THREAD,
      Z_THREAD>(
      inp_val2,
      init_val2,
      reduction_op2,
      work_buf2,
      (T2*)shared_buf,
      read_pred,
      grid_reduction_segment_size,
      idx_in_grid_segment,
      block_reduction_segment_size);

  if (PERSISTENT_REDUCTION) {
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION>(
        sync_flags[idx_in_grid_segment], grid_reduction_segment_size);
  } else {
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION>(
        sync_flags[entrance_ind * grid_segment_size + idx_in_grid_segment],
        grid_reduction_segment_size);
  }

  bool last_block =
      index_utils::maskedIsLast<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);

  if (last_block) {
    // Cleanup with block reduction
    gridReduceLastBlock<!X_THREAD, !Y_THREAD, !Z_THREAD>(
        out1,
        work_buf1,
        grid_reduction_segment_size,
        block_reduction_segment_size,
        reduction_op1,
        (T1*)shared_buf,
        write_pred,
        init_val1);
    gridReduceLastBlock<!X_THREAD, !Y_THREAD, !Z_THREAD>(
        out2,
        work_buf2,
        grid_reduction_segment_size,
        block_reduction_segment_size,
        reduction_op2,
        (T2*)shared_buf,
        write_pred,
        init_val2);
  }

  if (PERSISTENT_REDUCTION) {
    // Make sure we're done with global memory before we allow the kernel to
    // continue
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION>(
        sync_flags[idx_in_grid_segment], grid_reduction_segment_size);
  }
}

#ifdef PYTORCH_NVFUSER_PROFILE_KERNEL
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool PERSISTENT_REDUCTION,
    typename T1,
    typename Func1,
    typename T2,
    typename Func2>
__device__ void gridReduceGroup(
    T1& out1,
    const T1& inp_val1,
    T1 init_val1,
    Func1 reduction_op1,
    volatile T1* work_buf1,
    T2& out2,
    const T2& inp_val2,
    T2 init_val2,
    Func2 reduction_op2,
    volatile T2* work_buf2,
    int64_t* sync_flags,
    void* shared_buf,
    bool read_pred,
    bool write_pred,
    const int entrance_ind,
    const int n_entrances,
    int64_t& cycles,
    int64_t& count) {
  int64_t start_counter = 0;

  if (index_utils::maskedIsLast<true, true, true>(blockIdx, gridDim) &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    start_counter = readCycleCounter();
  }

  gridReduceGroup<
      X_BLOCK,
      Y_BLOCK,
      Z_BLOCK,
      X_THREAD,
      Y_THREAD,
      Z_THREAD,
      PERSISTENT_REDUCTION,
      T1,
      Func1,
      T2,
      Func2>(
      out1,
      inp_val1,
      init_val1,
      reduction_op1,
      work_buf1,
      out2,
      inp_val2,
      init_val2,
      reduction_op2,
      work_buf2,
      sync_flags,
      shared_buf,
      read_pred,
      write_pred,
      entrance_ind,
      n_entrances);

  if (index_utils::maskedIsLast<true, true, true>(blockIdx, gridDim) &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    cycles += readCycleCounter() - start_counter;
    ++count;
  }
}
#endif // PYTORCH_NVFUSER_PROFILE_KERNEL

} // namespace reduction








__global__ void kernel1(Tensor<float, 3> T0, Tensor<float, 2> T2, Tensor<float, 1> T4, Tensor<int64_t, 1> T5) {
  alignas(16) extern __shared__ char array[];
  void* shared_mem = array;
  // Allocate global tensor T4
  // Allocate global tensor T5
  int i61;
  i61 = (((int)blockIdx.x) * ((int)blockDim.x)) + ((int)threadIdx.x);
  float T3[1];
  T3[0] = NEG_INFINITY;
  #pragma unroll 1
  for(int i43 = 0; i43 < (ceilDiv((ceilDiv((ceilDiv(T0.size[0], ((int)blockDim.y))), 1)), ((int)gridDim.y))); ++i43) {
    int i59;
    i59 = (((((int)blockIdx.y) * (ceilDiv((ceilDiv((ceilDiv(T0.size[0], ((int)blockDim.y))), 1)), ((int)gridDim.y)))) + i43) * ((int)blockDim.y)) + ((int)threadIdx.y);
    if (((i59 < T0.size[0]) && (i61 < (T0.size[1] * T0.size[2])))) {
      float T1[1];
      T1[0] = NEG_INFINITY;
      T1[0]
        = T0[(i59 * (T0.size[2] * T0.size[1])) + i61]
        * (float) 2.00000000000000000e+00;
      T3[0] = fmax(
        T3[0],
        T1[0]);
    }
  }
  if ((i61 < (T0.size[1] * T0.size[2]))) {
    T2[i61] = NEG_INFINITY;
  }
  reduction::gridReduce<false, true, false, false, true, false, false>(
    T2[i61],
    T3[0],
    [](float &a, float b) { a = fmax(a, b); },
    &T4[0],
    &T5[0],
    static_cast<float*>(shared_mem),
    (i61 < (T0.size[1] * T0.size[2])),
    (i61 < (T0.size[1] * T0.size[2])),
    float(NEG_INFINITY),
    0,
    1);
}