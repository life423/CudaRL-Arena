cmake_minimum_required(VERSION 3.24)
project(CudaRL-Arena VERSION 0.1.0 LANGUAGES CXX CUDA)

# ──────────────── Global Settings ────────────────
set(CMAKE_CXX_STANDARD        17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD       17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ──────────────── Dependencies ────────────────
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python3 ${Python3_VERSION} at ${Python3_EXECUTABLE}")

# ──────────────── CUDA Settings ────────────────
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
set(CMAKE_CUDA_RUNTIME_LIBRARY MultiThreadedDebugDLL)

# CUDA compilation flags
set(CUDA_COMPILE_FLAGS "--use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_COMPILE_FLAGS}")

# ──────────────── Core Library ────────────────
# Create core library with CUDA functionality
add_library(cudarl_core STATIC
  src/core/cuda_utils.h
  src/core/environment.h
  src/gpu/environment.cu
  src/gpu/kernels.cuh
)

target_include_directories(cudarl_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cudarl_core PUBLIC
  CUDA::cudart
)

target_compile_options(cudarl_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

# ──────────────── Main Application ────────────────
add_executable(cudarl_app src/core/main.cu)
target_link_libraries(cudarl_app PRIVATE cudarl_core)

# ──────────────── CUDA Test ────────────────
add_executable(cuda_test src/gpu/cuda_test.cu)
target_link_libraries(cuda_test PRIVATE cudarl_core)

# ──────────────── Custom Targets ────────────────
add_custom_target(run
  COMMAND $<TARGET_FILE:cudarl_app>
  DEPENDS cudarl_app
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Building and running cudarl_app"
)

add_custom_target(run_cuda_test
  COMMAND $<TARGET_FILE:cuda_test>
  DEPENDS cuda_test
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Building and running CUDA functionality test"
)