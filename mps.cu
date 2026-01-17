
// mpi_cuda_mps_demo.cu
#include <mpi.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

#define CHECK_MPI(call) do { \
    int _e = (call); \
    if (_e != MPI_SUCCESS) { \
        fprintf(stderr, "MPI error %d at %s:%d\n", _e, __FILE__, __LINE__); \
        MPI_Abort(MPI_COMM_WORLD, _e); \
    } \
} while (0)

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        MPI_Abort(MPI_COMM_WORLD, -1); \
    } \
} while (0)

__global__ void long_kernel(float* out, int iters) {
    // 128 threads per block by default; 1 or more blocks.
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Do heavy transcendental math to keep SFUs/ALUs busy for a long time
    float x = 0.0001f * (tid + 1);
    float y = 1.000001f;
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        // Mix fast math and dependencies to reduce compiler elimination.
        x = sinf(x) + cosf(x) * y;
        y = y * 1.0000001f + 1e-8f;
        if ((i & 1023) == 0) x += 1e-7f;
    }
    // Prevent dead-code elimination
    out[tid & 1023] = x + y;
}

static void usage_and_exit(int rank) {
    if (rank == 0) {
        std::cerr <<
        "Usage: ./mpi_cuda_mps_demo [--seconds S] [--blocks B] [--threads T]\n"
        "Defaults: S=2.0, B=1, T=128 (must be multiple of 32).\n"
        "Tip: Use B=1, T=128 to underutilize a big GPU; run 2 ranks to demonstrate MPS.\n";
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
}

int main(int argc, char** argv) {
    CHECK_MPI(MPI_Init(&argc, &argv));
    int rank = 0, size = 1;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

    // Parse args
    double target_seconds = 2.0;
    int blocks = 1;
    int threads = 128;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--seconds" && i + 1 < argc) target_seconds = atof(argv[++i]);
        else if (a == "--blocks" && i + 1 < argc) blocks = atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = atoi(argv[++i]);
        else if (a == "--help" || a == "-h") usage_and_exit(rank);
        else { if (rank==0) std::cerr << "Unknown arg: " << a << "\n"; usage_and_exit(rank); }
    }
    if (threads <= 0 || (threads % 32) != 0 || blocks <= 0 || target_seconds <= 0.0) usage_and_exit(rank);

    // Pin all ranks to the same GPU (GPU 0). Prefer setting CUDA_VISIBLE_DEVICES=0 externally.
    CHECK_CUDA(cudaSetDevice(0));

    // Show device basics once
    if (rank == 0) {
        cudaDeviceProp prop{};
        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Device: " << prop.name
                  << " | SMs: " << prop.multiProcessorCount
                  << " | Warp size: " << prop.warpSize
                  << " | Max blocks/SM: " << prop.maxBlocksPerMultiProcessor
                  << "\n";
    }

    // Allocate small output buffer
    const size_t out_elems = 1024;
    float *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out, out_elems * sizeof(float)));

    // Calibration: pick iterations to approximate target_seconds
    const int base_iters = 50'000'000; // 50M
    dim3 grid(blocks), block(threads);

    cudaEvent_t c_start, c_stop;
    CHECK_CUDA(cudaEventCreate(&c_start));
    CHECK_CUDA(cudaEventCreate(&c_stop));

    // Warmup (optional)
    long_kernel<<<grid, block>>>(d_out, 10'000);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD)); // align calibration
    CHECK_CUDA(cudaEventRecord(c_start));
    long_kernel<<<grid, block>>>(d_out, base_iters);
    CHECK_CUDA(cudaEventRecord(c_stop));
    CHECK_CUDA(cudaEventSynchronize(c_stop));

    float ms_calib = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_calib, c_start, c_stop));
    double s_calib = ms_calib / 1000.0;
    // Calculate iterations for target_seconds, clamp to at least 1
    long long timed_iters = std::max(1LL, (long long)std::llround((target_seconds / std::max(1e-6, s_calib)) * base_iters));

    if (rank == 0) {
        std::cout << "Calibration: " << s_calib << " s for " << base_iters
                  << " iters -> using " << timed_iters << " iters for target ~"
                  << target_seconds << " s\n";
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD)); // synchronize start of measured pass

    double t0 = MPI_Wtime();
    CHECK_CUDA(cudaEventRecord(c_start));
    long_kernel<<<grid, block>>>(d_out, (int)timed_iters);
    CHECK_CUDA(cudaEventRecord(c_stop));
    CHECK_CUDA(cudaEventSynchronize(c_stop));
    double t1 = MPI_Wtime();

    float ms_kernel = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_kernel, c_start, c_stop));

    // Reduce stats to rank 0
    double host_time = t1 - t0;
    double max_host_time = 0.0, min_host_time = 0.0, sum_host_time = 0.0;
    CHECK_MPI(MPI_Reduce(&host_time, &max_host_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Reduce(&host_time, &min_host_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Reduce(&host_time, &sum_host_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

    float max_dev_ms = 0.0f, min_dev_ms = 0.0f, sum_dev_ms = 0.0f;
    CHECK_MPI(MPI_Reduce(&ms_kernel, &max_dev_ms, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Reduce(&ms_kernel, &min_dev_ms, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Reduce(&ms_kernel, &sum_dev_ms, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));

    // Per-rank print
    std::ostringstream oss;
    oss.setf(std::ios::fixed); oss.precision(3);
    oss << "[rank " << rank << "] blocks=" << blocks << " threads=" << threads
        << " kernel=" << (ms_kernel/1000.0) << " s, host wall=" << host_time << " s";
    std::cout << oss.str() << std::endl;

    if (rank == 0) {
        std::cout.setf(std::ios::fixed); std::cout.precision(3);
        std::cout << "Summary (ranks=" << size << "):\n"
                  << "  Device time per-rank: min=" << (min_dev_ms/1000.0)
                  << " s, max=" << (max_dev_ms/1000.0)
                  << " s, avg=" << (sum_dev_ms/size/1000.0) << " s\n"
                  << "  Host wall per-rank:   min=" << min_host_time
                  << " s, max=" << max_host_time
                  << " s, avg=" << (sum_host_time/size) << " s\n";
    }

    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaEventDestroy(c_start));
    CHECK_CUDA(cudaEventDestroy(c_stop));
    CHECK_MPI(MPI_Finalize());
    return 0;
}

