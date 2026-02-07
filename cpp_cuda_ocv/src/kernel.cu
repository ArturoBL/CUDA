#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("Hola desde kernel CUDA 🚀\n");
}

extern "C" void launch_kernel() {
    hello_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
}
