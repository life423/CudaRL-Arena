cmake_minimum_required(VERSION 3.18)
project(CudaArenaTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures for RTX 5070
set(CMAKE_CUDA_ARCHITECTURES "120")

# Test executable
add_executable(test_cuda_arena
    test_cuda_arena.cpp
    cuda_arena.cu
)

target_link_libraries(test_cuda_arena PRIVATE
    CUDA::cudart
)

target_include_directories(test_cuda_arena PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set runtime library for Windows
if(WIN32)
    set_target_properties(test_cuda_arena PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )
endif()