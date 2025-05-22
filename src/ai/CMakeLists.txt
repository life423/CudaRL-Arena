# CUDA-Accelerated AI Library

add_library(ai_cuda STATIC
    qlearning_cuda.cu
    dqn_cuda.cu
)

target_include_directories(ai_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Add additional CUDA-specific properties or libraries as needed
set_target_properties(ai_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)