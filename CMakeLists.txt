cmake_minimum_required(VERSION 3.18)
project(CudaRL LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Core library
add_library(cudarl_core STATIC
    src/core/environment.cpp
    src/core/vectorized_environment.cpp
)
target_include_directories(cudarl_core PUBLIC src/core)

# GPU kernels library
add_library(cudarl_gpu STATIC
    src/gpu/kernels.cu
    src/gpu/step_kernel.cu
)
target_include_directories(cudarl_gpu PUBLIC src/gpu)
target_link_libraries(cudarl_gpu PRIVATE cuda cudart)

# Build options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

# Python bindings (optional - requires pybind11)
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(cudarl_core_python src/python_bindings.cpp)
    target_link_libraries(cudarl_core_python PRIVATE cudarl_core cudarl_gpu)
endif()

# ------ Tests -----------------------------------------------
# Main batch test
add_executable(test_batch_step tests/test_batch_step.cu)
target_link_libraries(test_batch_step PRIVATE cudarl_core cudarl_gpu cuda cudart)

# Simple CUDA test
add_executable(simple_cuda_test tests/simple_cuda_test/main.cpp)
target_link_libraries(simple_cuda_test PRIVATE cudarl_gpu cuda cudart)

# ------ Test configuration ----------------------------------
enable_testing()
add_test(NAME batch_step_test COMMAND test_batch_step)
add_test(NAME simple_cuda_test COMMAND simple_cuda_test)