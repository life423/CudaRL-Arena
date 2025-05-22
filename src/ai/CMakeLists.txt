# CUDA-Accelerated AI Library

add_library(ai_cuda STATIC
    # Add your CUDA/C++ source files here, e.g.:
    # qlearning_cuda.cu
)

target_include_directories(ai_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Add additional CUDA-specific properties or libraries as needed
