
// Build (CUDA 13.1+):
//   nvcc -std=c++17 -O2 two_contexts_illegal_access.cu -o two_contexts_illegal_access -lcuda
// If CUDA isn't on PATH, add includes/libs explicitly:
//   nvcc -I/usr/local/cuda-13.1/include -L/usr/local/cuda-13.1/lib64 \
//        -std=c++17 -O2 two_contexts_illegal_access.cu -o two_contexts_illegal_access -lcuda
//
// Run:
//   ./two_contexts_illegal_access
//
// Expected:
//   - ctxA kernel: OK, reads initialized value
//   - ctxB kernel with ctxA pointer: cudaDeviceSynchronize -> "an illegal memory access was encountered"

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CU(call) do { \
    CUresult e = (call); \
    if (e != CUDA_SUCCESS) { \
        const char *n = nullptr, *s = nullptr; \
        cuGetErrorName(e, &n); cuGetErrorString(e, &s); \
        fprintf(stderr, "[Driver] %s:%d: %s - %s\n", \
                __FILE__, __LINE__, n ? n : "?", s ? s : "?"); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "[Runtime] %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Simple kernel: read *remote, write to *out.
// Will fault if 'remote' is not valid in the current context.
__global__ void read_kernel(const int* remote, int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int v = remote[0];
        out[0] = v;
    }
}

int main() {
    // Initialize Driver API and pick device 0
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));

    CUcontext current_ctx;
    cuCtxGetCurrent(&current_ctx);
    printf("device primary context %p\n", (void *)current_ctx);

    // ---- Create two independent contexts using the CUDA 13.x signature ----
    // CUresult cuCtxCreate(CUcontext*, CUctxCreateParams*, unsigned int flags, CUdevice);
    // Zero-initialized CUctxCreateParams is valid (same effect as legacy create).  [3](https://github.com/eyalroz/cuda-api-wrappers/issues/746)
    CUctxCreateParams params{};
    CUcontext ctxA = nullptr, ctxB = nullptr, ctxC = nullptr;

#if CUDART_VERSION >= 13000
    CHECK_CU(cuCtxCreate(&ctxA, &params, /*flags*/0, dev));
    CHECK_CU(cuCtxCreate(&ctxB, &params, /*flags*/0, dev));
    CHECK_CU(cuCtxCreate(&ctxC, &params, /*flags*/0, dev));
#else
    CHECK_CU(cuCtxCreate(&ctxA, /*flags*/0, dev));
    CHECK_CU(cuCtxCreate(&ctxB, /*flags*/0, dev));
    CHECK_CU(cuCtxCreate(&ctxC, /*flags*/0, dev));
#endif

    // =========================== Context A ===========================
    CHECK_CU(cuCtxSetCurrent(ctxA));
    printf("context: ctxA %p\n", (void *)ctxA);
    cuCtxGetCurrent(&current_ctx);
    printf("device current context after set ctxA %p\n", (void *)current_ctx);

    int *dA = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(int)));
    int hval = 12345;
    CHECK_CUDA(cudaMemcpy(dA, &hval, sizeof(int), cudaMemcpyHostToDevice));

    int *dOutA = nullptr;
    CHECK_CUDA(cudaMalloc(&dOutA, sizeof(int)));
    CHECK_CUDA(cudaMemset(dOutA, 0, sizeof(int)));

    read_kernel<<<1,1>>>(dA, dOutA);
    CHECK_CUDA(cudaDeviceSynchronize());

    int resA = 0;
    CHECK_CUDA(cudaMemcpy(&resA, dOutA, sizeof(int), cudaMemcpyDeviceToHost));
    printf("ctxA: kernel OK, read value = %d\n\n", resA);

    // =========================== Context B ===========================
    CHECK_CU(cuCtxSetCurrent(ctxB));
    printf("context: ctxB %p\n", (void *)ctxB);
    cuCtxGetCurrent(&current_ctx);
    printf("device current context after set ctxB %p\n", (void *)current_ctx);

    int *dOutB = nullptr;
    CHECK_CUDA(cudaMalloc(&dOutB, sizeof(int)));
    CHECK_CUDA(cudaMemset(dOutB, 0, sizeof(int)));

    printf("ctxB: launching kernel on pointer from ctxA...\n");
    // NOTE: dA belongs to ctxA and is not mapped in ctxB's VA space
    read_kernel<<<1,1>>>(dA, dOutB);

    cudaError_t syncErr = cudaDeviceSynchronize();
    printf("ctxB: cudaDeviceSynchronize = %s\n", cudaGetErrorString(syncErr));

    int resB = -1;
    cudaError_t cpErr = cudaMemcpy(&resB, dOutB, sizeof(int), cudaMemcpyDeviceToHost);
    printf("ctxB: cudaMemcpy = %s, out = %d\n", cudaGetErrorString(cpErr), resB);

    // =========================== Context C ===========================
    CHECK_CU(cuCtxSetCurrent(ctxC));
    printf("context: ctxC %p\n", (void *)ctxC);
    cuCtxGetCurrent(&current_ctx);
    printf("device current context after set ctxC %p\n", (void *)current_ctx);

    int *dOutC = nullptr;
    CHECK_CUDA(cudaMalloc(&dOutC, sizeof(int)));
    CHECK_CUDA(cudaMemset(dOutC, 0, sizeof(int)));

    printf("ctxC: launching kernel on pointer from ctxA...\n");
    // NOTE: dA belongs to ctxA and is not mapped in ctxB's VA space
    read_kernel<<<1,1>>>(dA, dOutC);

    syncErr = cudaDeviceSynchronize();
    printf("ctxC: cudaDeviceSynchronize = %s\n", cudaGetErrorString(syncErr));

    int resC = -1;
    cpErr = cudaMemcpy(&resC, dOutC, sizeof(int), cudaMemcpyDeviceToHost);
    printf("ctxC: cudaMemcpy = %s, out = %d\n", cudaGetErrorString(cpErr), resC);

    // Cleanup in the right contexts
    CHECK_CU(cuCtxSetCurrent(ctxA));  cudaFree(dOutA); cudaFree(dA);
    CHECK_CU(cuCtxSetCurrent(ctxB));  cudaFree(dOutB);
    CHECK_CU(cuCtxSetCurrent(ctxC));  cudaFree(dOutC);

    cuCtxDestroy(ctxB);
    cuCtxDestroy(ctxC);
    cuCtxDestroy(ctxA);
    return 0;
}

