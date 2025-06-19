cmake_minimum_required(VERSION 3.22)
project(CudaRL_Arena_KernelTest LANGUAGES CXX CUDA)

# ───── Build flags ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)          # NVIDIA 5070 (Ada, SM 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -O3 --use_fast_math")

# ───── Sources ────────────────────────────────────────────────────────────────
set(KERNEL_SRC
    ${CMAKE_SOURCE_DIR}/cuda/kernels/step_kernel.cu
    ${CMAKE_SOURCE_DIR}/cuda/kernels/reset_kernel.cu
)

add_library(kernels OBJECT ${KERNEL_SRC})
target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/cuda/kernels)

add_executable(test_batch_step tests/test_batch_step.cu
               $<TARGET_OBJECTS:kernels>)
target_link_libraries(test_batch_step PRIVATE cuda cudart)

enable_testing()
add_test(NAME kernel_batch_step COMMAND test_batch_step)
