cmake_minimum_required(VERSION 3.24)
project(FrozenInterfaceComparisonV2 LANGUAGES CXX CUDA)
get_filename_component(comparison_source "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
find_package(CUDAToolkit REQUIRED)
string(REPLACE "/W1" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(FROZEN_PARENT_BUILD "${comparison_source}/build/continuous-pond-native-2026-11-23/Release" CACHE PATH "Unchanged reviewed native static libraries")

function(comparison_target target source unfused)
  add_executable(${target} "${comparison_source}/${source}")
  target_include_directories(${target} PRIVATE "${comparison_source}/src" "${comparison_source}/src/nn")
  target_compile_definitions(${target} PRIVATE CUDARL_WITH_CUDA=1 CUDARL_APPEND_HOLD41=1 CUDARL_INTERFACE_UNFUSED=${unfused})
  # This profile changes implicit CUDA contraction only. Host arithmetic,
  # model equations, guards, limits, saved NN parameters and old builds remain.
  if(unfused)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  else()
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
  endif()
  target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211> $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
  target_link_libraries(${target} PRIVATE "${FROZEN_PARENT_BUILD}/creature_gpu.lib"
      "${FROZEN_PARENT_BUILD}/creature_physics.lib" CUDA::cudart_static)
endfunction()

comparison_target(frozen_interface_comparison_v2 tools/frozen_interface_comparison_v2.cu 1)
comparison_target(frozen_interface_arithmetic_v2 tools/frozen_interface_arithmetic_v2.cu 1)
comparison_target(frozen_interface_arithmetic_fused_reference_v2 tools/frozen_interface_arithmetic_v2.cu 0)
comparison_target(frozen_interface_drag_arithmetic_v2 tools/frozen_interface_drag_arithmetic_v2.cu 1)
comparison_target(frozen_interface_arithmetic_corrected_v2 tools/frozen_interface_arithmetic_corrected_v2.cu 1)
