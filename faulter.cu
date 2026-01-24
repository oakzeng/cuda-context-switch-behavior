#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <mpi.h>


// Fault-inducing kernel: touch managed memory at page-sized strides to
// trigger GPU page faults and migrations from CPU -> GPU.
__global__ void fault_touch_kernel(unsigned char* data, size_t nbytes, size_t stride) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = gridDim.x * (size_t)blockDim.x;

    // Walk the buffer with a stride (approx. page size) per thread
    for (size_t i = tid * stride; i < nbytes; i += total_threads * stride) {
        // simple read-modify-write to ensure a fault + dirty page
        unsigned char val = data[i];
        data[i] = val + 1;

        // Dummy work to avoid all touches happening back-to-back and being optimized out.
        // Use a volatile accumulator to prevent compiler removing the loop.
        volatile int acc = 0;
        for (int k = 0; k < 256; ++k) {
            acc += k;
        }
        // Optionally reference acc so it isn't optimized away entirely
        if (acc == -1) data[i] = val; // never true, but prevents elimination
    }
}


static void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

int main(int argc, char** argv) {
    int device = 0;
    size_t mib = 2048;          // default 2 GiB
    size_t stride = 65536;      // 64 KiB stride to roughly match UVM granularity
    int iters = 1;              // number of kernel iterations
    int use_managed_memory = 1;

    printf("faulter PID %d\n", getpid());

    if (argc > 1) mib = strtoull(argv[1], nullptr, 0);
    if (argc > 2) stride = strtoull(argv[2], nullptr, 0);
    if (argc > 3) iters = atoi(argv[3]);
    if (argc > 4) use_managed_memory = atoi(argv[4]);
    MPI_Init(&argc, &argv);

    check(cudaSetDevice(device), "cudaSetDevice");

    size_t nbytes = mib * (size_t)1024 * 1024;
    printf("[faulter] Allocating %zu MiB managed memory...\n", mib);

    unsigned char* data = nullptr;
    if (!use_managed_memory)
	    check(cudaMalloc(&data, nbytes), "cudaMalloc");
    else {
	    cudaMemLocation hostLoc;
	    hostLoc.type = cudaMemLocationTypeHost;
	    check(cudaMallocManaged(&data, nbytes, cudaMemAttachGlobal), "cudaMallocManaged");
	    // Initialize on CPU to ensure valid contents
	    for (size_t i = 0; i < nbytes; i += 4096) {
		    data[i] = (unsigned char)(i & 0xFF);
	    }

	    // Advise preferred location: CPU, and prefetch to CPU to guarantee GPU page faults on first touch
	    //check(cudaMemAdvise(data, nbytes, cudaMemAdviseSetPreferredLocation, hostLoc), "cudaMemAdvise SetPreferredLocation CPU");
	    //check(cudaMemAdvise(data, nbytes, cudaMemAdviseSetAccessedBy, device), "cudaMemAdvise SetAccessedBy device");
	    check(cudaMemPrefetchAsync(data, nbytes, hostLoc, 0, 0), "cudaMemPrefetchAsync CPU");
	    check(cudaDeviceSynchronize(), "prefetch sync");
    }

    // Create a low-priority stream to bias scheduling (optional)
    int leastPriority, greatestPriority;
    check(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority), "cudaDeviceGetStreamPriorityRange");
    cudaStream_t stream;
    check(cudaStreamCreateWithPriority(&stream, cudaStreamDefault, leastPriority), "cudaStreamCreateWithPriority");

    dim3 block(256);
    dim3 grid( (int) ( (nbytes / stride + block.x - 1) / block.x ) );
    if (grid.x > 65535) grid.x = 65535; // cap grid size

    printf("[faulter] Launching kernel to touch %zu MiB with stride %zu bytes, grid=%u block=%u, iters=%d...\n",
           mib, stride, grid.x, block.x, iters);

    // Use CUDA events to timestamp
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(1000000);
    for (int t = 0; t < iters; ++t) {
	nvtx3::scoped_range r{"faulter:fault_touch_kernel"};
        check(cudaEventRecord(start, stream), "event record start");
        if (t % 1 == 0) printf("\033[35m [faulter] Iter %d start \033[0m\n", t);
	//when hipMalloc is used, the kernel complete too quick for observation; use a smaller stride
        fault_touch_kernel<<<grid, block, 0, stream>>>(data, nbytes, stride);
        check(cudaGetLastError(), "kernel launch");
        check(cudaEventRecord(stop, stream), "event record stop");
        check(cudaEventSynchronize(stop), "event sync stop");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
        if (t % 1 == 0) printf("\033[35m [faulter] Iter %d done, elapsed %.3f ms \033[0m\n", t, ms);
    }

    check(cudaDeviceSynchronize(), "device sync end");

    // Prevent compiler from optimizing away data
    //volatile unsigned char sink = data[12345];
    //printf("[faulter] Done. sink=%u\n", (unsigned)sink);

    cudaStreamDestroy(stream);
    cudaFree(data);
    MPI_Finalize();
    return 0;
}
