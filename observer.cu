#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <mpi.h>

__global__ void small_work_kernel(float* out, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1024) return;
    float x = i * 0.001f;
    for (int k = 0; k < iters; ++k) {
        x = x * 1.0001f + 0.0003f;
    }
    out[i] = x;
}

static void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

int main(int argc, char** argv) {
    int device = 0;
    int loops = 2000;   // number of rapid launches
    int kernel_iters = 5000; // per-kernel small compute
    if (argc > 1) loops = atoi(argv[1]);
    if (argc > 2) kernel_iters = atoi(argv[2]);

    MPI_Init(&argc, &argv);
    check(cudaSetDevice(device), "cudaSetDevice");

    // High-priority stream to prefer scheduling
    int leastPriority, greatestPriority;
    check(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority), "cudaDeviceGetStreamPriorityRange");
    cudaStream_t stream;
    check(cudaStreamCreateWithPriority(&stream, cudaStreamDefault, greatestPriority), "cudaStreamCreateWithPriority");

    float* d_out = nullptr;
    check(cudaMalloc(&d_out, 1024 * sizeof(float)), "cudaMalloc d_out");

    // CSV header
    printf("[Observer] iter,ms\n");

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");

    dim3 block(256);
    dim3 grid( (1024 + block.x - 1) / block.x );

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < loops; ++i) {
	nvtx3::scoped_range r{"observer:iter"};
        check(cudaEventRecord(start, stream), "event record start");
        if (i % 100 == 0) printf("\033[31m [Observer] Iter %d start \033[0m\n", i);
        small_work_kernel<<<grid, block, 0, stream>>>(d_out, kernel_iters);
        check(cudaGetLastError(), "kernel launch");
        check(cudaEventRecord(stop, stream), "event record stop");
        check(cudaEventSynchronize(stop), "event sync stop");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
        if (i % 100 == 0) printf("\033[31m [Observer] Iter %d done,%.6f ms \033[0m\n", i, ms);
    }

    cudaFree(d_out);
    cudaStreamDestroy(stream);
    printf("[Observer] exit\n");
    MPI_Finalize();
    return 0;
}
