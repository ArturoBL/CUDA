cmake_minimum_required(VERSION 3.18)
project(cuda_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_cpp
    src/main.cpp
    src/kernel.cu
)

set_target_properties(cuda_cpp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
