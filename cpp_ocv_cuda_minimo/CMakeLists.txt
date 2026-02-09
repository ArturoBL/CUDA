cmake_minimum_required(VERSION 3.18)
project(opencv_cuda_minimo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(opencv_cuda_minimo src/main.cpp)

target_include_directories(opencv_cuda_minimo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(opencv_cuda_minimo
    ${OpenCV_LIBS}
)
