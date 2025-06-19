cmake_minimum_required(VERSION 3.18)
project(SimpleCudaTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDAToolkit REQUIRED)

# Define executable
add_executable(simple_cuda_test main.cpp ../../src/gpu/kernels.cu)

set_target_properties(simple_cuda_test PROPERTIES CUDA_ARCHITECTURES "75;80;86")

# Explicitly include the src/gpu directory
target_include_directories(simple_cuda_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gpu
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(simple_cuda_test PRIVATE CUDA::cudart)
