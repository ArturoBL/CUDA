#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>

#define CUDA_CHECK(x) do { \
  cudaError_t err = x; \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) \
              << " at line " << __LINE__ << std::endl; \
    return -1; \
  } \
} while(0)


/*Compilar con:
 g++ cuda_gl_interop_test.cpp -o cuda_gl_interop_test   -lglfw -lGLEW -lGL   -lcuda -lcudart -I/usr/local/cuda/include -L/usr/local/cuda/lib64

Ejecutar así:
  __NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
./cuda_gl_interop_test
*/

int main()
{
    if (!glfwInit())
        return -1;

    GLFWwindow* win = glfwCreateWindow(640, 480, "Interop Test", nullptr, nullptr);
    glfwMakeContextCurrent(win);

    glewInit();

    // Fuerza contexto CUDA DESPUÉS de OpenGL
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    // Crear PBO
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 640 * 480, nullptr, GL_DYNAMIC_DRAW);

    cudaGraphicsResource* res = nullptr;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &res, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // Map / unmap de prueba
    CUDA_CHECK(cudaGraphicsMapResources(1, &res));

    void* ptr = nullptr;
    size_t size = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&ptr, &size, res));

    std::cout << "PBO mapeado correctamente, size = " << size << std::endl;

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &res));

    std::cout << "Interop CUDA–OpenGL FUNCIONA\n";

    cudaGraphicsUnregisterResource(res);
    glfwTerminate();
    return 0;
}

