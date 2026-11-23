cmake_minimum_required(VERSION 3.24)
project(FrozenInterfaceComparison LANGUAGES CXX CUDA)
get_filename_component(comparison_source "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
find_package(CUDAToolkit REQUIRED)
string(REPLACE "/W1" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(FROZEN_PARENT_BUILD "${comparison_source}/build/continuous-pond-native-2026-11-23/Release" CACHE PATH "Unchanged reviewed native static libraries")
add_executable(frozen_interface_comparison "${comparison_source}/tools/frozen_interface_comparison.cu")
target_include_directories(frozen_interface_comparison PRIVATE "${comparison_source}/src" "${comparison_source}/src/nn")
target_compile_definitions(frozen_interface_comparison PRIVATE CUDARL_WITH_CUDA=1 CUDARL_APPEND_HOLD41=1)
target_compile_options(frozen_interface_comparison PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>
    $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
target_link_libraries(frozen_interface_comparison PRIVATE "${FROZEN_PARENT_BUILD}/creature_gpu.lib"
    "${FROZEN_PARENT_BUILD}/creature_physics.lib" CUDA::cudart_static)
