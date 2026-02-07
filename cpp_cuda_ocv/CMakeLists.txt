cmake_minimum_required(VERSION 3.18)
project(cuda_cpp_opencv LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

find_package(OpenCV REQUIRED)

add_executable(cuda_cpp_opencv
    src/main.cpp
    src/kernel.cu
)

target_include_directories(cuda_cpp_opencv PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cuda_cpp_opencv
    ${OpenCV_LIBS}
)

set_target_properties(cuda_cpp_opencv PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
