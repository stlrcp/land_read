#define THREADS 1024
__global__ void CustomAddKernel(float *input1, float *input2, float *output, size_t size) {
    auto idx = blockIdx.x * THREADS + threadIdx.x;
    if (idx < size) {
        output[idx] = input1[idx] + input2[idx];
    }
}

extern "C" int CustomAdd(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra) {
    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    if (nparam != 3)
        return 1;
    void *input1 = params[0];
    void *input2 = params[1];
    void *output = params[2];
    size_t size = 1;
    for (int i = 0; i < ndims[2]; i++) {
        size *= shapes[2][i];
    }
    int n = size / THREADS;
    for (int i = 0; i < nparam; i++) {
        if (strcmp(dtypes[i], "float32") != 0) {
            return 2;
        }
    }
    CustomAddKernel<<<n + 1, THREADS, 0, custream>>>(static_cast<float *>(input1), static_cast<float *>(input2), static_cast<float *>(output), size);
    return 0;
}