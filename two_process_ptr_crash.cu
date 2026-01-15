#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <mpi.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[%s:%d] CUDA error: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(_e), (int)_e); \
        fflush(stderr); \
        MPI_Abort(MPI_COMM_WORLD, (int)_e); \
    } \
} while(0)

#define CHECK_MPI(call) do { \
    int _e = (call); \
    if (_e != MPI_SUCCESS) { \
        char errstr[MPI_MAX_ERROR_STRING]; int len = 0; \
        MPI_Error_string(_e, errstr, &len); \
        fprintf(stderr, "[%s:%d] MPI error: %.*s\n", __FILE__, __LINE__, len, errstr); \
        fflush(stderr); \
        MPI_Abort(MPI_COMM_WORLD, _e); \
    } \
} while(0)

__global__ void read_remote_kernel(const int* remote, int* out) {
    // Intentionally read from "remote" pointer (actually from another process)
    // This should cause illegal memory access in this process.
    int val = remote[0];  // <- expected to fault
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = val; // If we somehow get here, write to out
    }
}

int main(int argc, char** argv) {
    CHECK_MPI(MPI_Init(&argc, &argv));
    int rank = -1, size = -1;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "This demo requires exactly 2 MPI processes.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // For reproducibility: force both ranks to use device 0.
    // (Even on the same GPU, different processes have different CUDA contexts,
    // so the pointer will still be invalid in rank 1.)
    CHECK_CUDA(cudaSetDevice(0));

    if (rank == 0) {
        // Rank 0: allocate device memory and send raw pointer value
        const size_t N = 1024;
        int* dA = nullptr;
        CHECK_CUDA(cudaMalloc(&dA, N * sizeof(int)));

        // Initialize to a recognizable pattern (each byte 0x2A -> ints ~0x2A2A2A2A)
        CHECK_CUDA(cudaMemset(dA, 0x2A, N * sizeof(int)));

        uint64_t ptr_val = reinterpret_cast<uint64_t>(dA);
        fprintf(stdout, "Rank 0: cudaMalloc dA = 0x%016llx, sending pointer value to rank 1\n",
                (unsigned long long)ptr_val);
        fflush(stdout);

        CHECK_MPI(MPI_Send(&ptr_val, 1, MPI_UINT64_T, 1, 0, MPI_COMM_WORLD));

        // Wait for rank 1 to finish its attempt before freeing
        int ack = 0;
        CHECK_MPI(MPI_Recv(&ack, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

        CHECK_CUDA(cudaFree(dA));
        fprintf(stdout, "Rank 0: Freed dA and exiting.\n");
        fflush(stdout);
    } else {
        // Rank 1: receive pointer value and try to use it
        uint64_t ptr_val = 0;
        CHECK_MPI(MPI_Recv(&ptr_val, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        int* remote = reinterpret_cast<int*>(ptr_val);

        fprintf(stdout, "Rank 1: Received device pointer value 0x%016llx; launching kernel to read it...\n",
                (unsigned long long)ptr_val);
        fflush(stdout);

        // Allocate a small output buffer on rank 1's device to capture any result (if it doesn't crash)
        int* dOut = nullptr;
        CHECK_CUDA(cudaMalloc(&dOut, sizeof(int)));
        CHECK_CUDA(cudaMemset(dOut, 0, sizeof(int)));

        // Launch kernel that attempts to read the "remote" pointer
        read_remote_kernel<<<1, 1>>>(remote, dOut);

        // Check for immediate launch failure (unlikely here)
        cudaError_t kerr = cudaGetLastError();
        fprintf(stdout, "Rank 1: cudaGetLastError after launch: %s (%d)\n",
                cudaGetErrorString(kerr), (int)kerr);
        fflush(stdout);

        // Synchronize to force any illegal access to surface
        cudaError_t sync_err = cudaDeviceSynchronize();
        fprintf(stdout, "Rank 1: cudaDeviceSynchronize returned: %s (%d)\n",
                cudaGetErrorString(sync_err), (int)sync_err);
        fflush(stdout);

        // Try to read back dOut anyway to show further errors cascade
        int hOut = -1;
        cudaError_t cperr = cudaMemcpy(&hOut, dOut, sizeof(int), cudaMemcpyDeviceToHost);
        fprintf(stdout, "Rank 1: cudaMemcpy(dOut) returned: %s (%d), hOut=%d\n",
                cudaGetErrorString(cperr), (int)cperr, hOut);
        fflush(stdout);

        // Clean up
        cudaError_t free_err = cudaFree(dOut);
        if (free_err != cudaSuccess) {
            fprintf(stdout, "Rank 1: cudaFree(dOut) returned: %s (%d)\n",
                    cudaGetErrorString(free_err), (int)free_err);
            fflush(stdout);
        }

        // Signal rank 0 we are done
        int ack = 1;
        CHECK_MPI(MPI_Send(&ack, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    CHECK_MPI(MPI_Finalize());
    return 0;
}

