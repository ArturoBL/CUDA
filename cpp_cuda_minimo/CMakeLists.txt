cmake_minimum_required(VERSION 3.18)
project(cuda_minimo LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_minimo src/main.cu)

set_target_properties(cuda_minimo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
