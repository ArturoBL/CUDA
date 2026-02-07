#include <cstdio>
#include <cuda_runtime.h>

/*
Compilamos así:
mkdir -p build
cd build
cmake ..
make -j$(nproc)
*/

__global__ void hello_cuda() {
    printf("Hola desde el kernel CUDA 👋\n");
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No se encontraron GPUs CUDA\n");
        return 1;
    }

    printf("GPUs CUDA detectadas: %d\n", deviceCount);

    hello_cuda<<<1, 1>>>();
    cudaDeviceSynchronize();

    printf("Ejecución completada ✅\n");
    return 0;
}
