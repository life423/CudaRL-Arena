cmake_minimum_required(VERSION 3.20)
project(cuda_project LANGUAGES CXX CUDA)

# C++17 + CUDA17
set(CMAKE_CXX_STANDARD      17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD     17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Pull in the toolkit via CMake’s package registry
find_package(CUDAToolkit REQUIRED)

# Only emit code for SM ≥7.5 (pre-7.5 targets are deprecated)
# adjust this list to match the cards you want to support
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Force CUDA to use the same DLL runtime that MSVC uses (/MDd in Debug)
set(CMAKE_CUDA_RUNTIME_LIBRARY MultiThreadedDebugDLL)

add_executable(poc_app src/main.cu)

# suppress that “offline compilation for <75 will be removed” warning
target_compile_options(poc_app PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

# link against the cudart from the toolkit
target_link_libraries(poc_app PRIVATE CUDA::cudart)
