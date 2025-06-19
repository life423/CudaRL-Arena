cmake_minimum_required(VERSION 3.18)
project(SimpleCudaTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDAToolkit REQUIRED)

add_executable(simple_cuda_test main.cpp ../../src/gpu/kernels.cu)

target_include_directories(simple_cuda_test PRIVATE ../../src/gpu)

target_link_libraries(simple_cuda_test PRIVATE CUDA::cudart)
