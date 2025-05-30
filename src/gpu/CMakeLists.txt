find_package(CUDA REQUIRED)
cuda_add_library(cudarl_gpu
    kernels.cu
)
target_include_directories(cudarl_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
