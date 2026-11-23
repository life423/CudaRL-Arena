cmake_minimum_required(VERSION 3.24)
project(FrozenRequestedTurnProbe LANGUAGES CXX CUDA)
get_filename_component(probe_source "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
find_package(CUDAToolkit REQUIRED)
string(REPLACE "/W1" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(FROZEN_PARENT_BUILD "${probe_source}/build/continuous-pond-native-2026-11-23/Release" CACHE PATH "Existing reviewed static native libraries")
add_executable(frozen_requested_turn_probe "${probe_source}/tools/frozen_requested_turn_probe.cu")
target_include_directories(frozen_requested_turn_probe PRIVATE "${probe_source}/src" "${probe_source}/src/nn")
target_compile_definitions(frozen_requested_turn_probe PRIVATE CUDARL_WITH_CUDA=1 CUDARL_APPEND_HOLD41=1)
target_compile_options(frozen_requested_turn_probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>
    $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
target_link_libraries(frozen_requested_turn_probe PRIVATE "${FROZEN_PARENT_BUILD}/creature_gpu.lib"
    "${FROZEN_PARENT_BUILD}/creature_physics.lib" CUDA::cudart_static)
