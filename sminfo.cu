
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaGetDevice(&device);

    int smCount, warpSize;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device);

    printf("SM count: %d\n", smCount);
    printf("Warp size: %d\n", warpSize);

    return 0;
}

