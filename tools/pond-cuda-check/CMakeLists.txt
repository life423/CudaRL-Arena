cmake_minimum_required(VERSION 3.24)
project(PondCudaCheck LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
find_package(CUDAToolkit REQUIRED)
add_executable(pond_cuda_check check.cu ../../src/core/checkpoint.cpp ../../src/nn/motor/saved_motor_contract.cpp)
target_include_directories(pond_cuda_check PRIVATE ../../src ../../src/nn)
target_compile_options(pond_cuda_check PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>
    $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CXX>:/WX>)
target_link_libraries(pond_cuda_check PRIVATE CUDA::cudart_static)
