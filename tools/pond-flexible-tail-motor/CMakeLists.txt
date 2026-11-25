cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()
project(PondFlexibleTailMotor LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(source_root "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
add_executable(pond_flexible_tail_motor_fit fit.cu
    "${source_root}/nn/mlp_gpu.cu"
    "${source_root}/nn/gemm.cu"
    "${source_root}/nn/loss.cu"
    "${source_root}/nn/adam.cu"
    "${source_root}/core/checkpoint.cpp")
set_target_properties(pond_flexible_tail_motor_fit PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES CUDA_RUNTIME_LIBRARY Static)
target_include_directories(pond_flexible_tail_motor_fit PRIVATE "${source_root}" "${source_root}/nn")
target_compile_features(pond_flexible_tail_motor_fit PRIVATE cxx_std_17)

set(habitat_path "${CMAKE_CURRENT_SOURCE_DIR}/../../flexible-tail-preview/habitat.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${habitat_path}")
file(READ "${habitat_path}" habitat)
string(JSON habitat_schema GET "${habitat}" schema)
if(NOT habitat_schema EQUAL 1)
    message(FATAL_ERROR "Unsupported flexible-tail habitat descriptor")
endif()
foreach(dimension IN ITEMS water_radius_x_m water_radius_z_m flat_radius_x_m
                           flat_radius_z_m floor_y_m reference_scale_m)
    string(JSON ${dimension} GET "${habitat}" ${dimension})
endforeach()
configure_file("../pond-flexible-tail-check/habitat_config.h.in" habitat_config.h @ONLY)
add_executable(pond_flexible_tail_motor_fixture fixture.cpp "${source_root}/core/checkpoint.cpp")
target_include_directories(pond_flexible_tail_motor_fixture PRIVATE
    "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(pond_flexible_tail_motor_fixture PRIVATE cxx_std_17)

add_executable(pond_flexible_tail_motor_turn_fixture turn-fixture.cpp
    "${source_root}/core/checkpoint.cpp")
target_include_directories(pond_flexible_tail_motor_turn_fixture PRIVATE
    "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(pond_flexible_tail_motor_turn_fixture PRIVATE cxx_std_17)

add_executable(pond_flexible_tail_motor_turn_ppo turn-ppo.cu turn-kernels.cu
    "${source_root}/nn/mlp_gpu.cu"
    "${source_root}/nn/gemm.cu"
    "${source_root}/nn/adam.cu"
    "${source_root}/rl/gaussian_kernels.cu"
    "${source_root}/rl/ppo_kernels.cu"
    "${source_root}/core/checkpoint.cpp")
set_target_properties(pond_flexible_tail_motor_turn_ppo PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES CUDA_RUNTIME_LIBRARY Static)
target_include_directories(pond_flexible_tail_motor_turn_ppo PRIVATE
    "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(pond_flexible_tail_motor_turn_ppo PRIVATE cxx_std_17)
set_property(SOURCE turn-ppo.cu turn-kernels.cu
    APPEND PROPERTY COMPILE_OPTIONS --fmad=false)

add_executable(pond_flexible_tail_motor_navigation_fixture navigation-fixture.cpp
    "${source_root}/core/checkpoint.cpp")
target_include_directories(pond_flexible_tail_motor_navigation_fixture PRIVATE
    "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(pond_flexible_tail_motor_navigation_fixture PRIVATE cxx_std_17)

add_executable(pond_flexible_tail_motor_launch_fixture launch-fixture.cpp
    "${source_root}/core/checkpoint.cpp")
target_include_directories(pond_flexible_tail_motor_launch_fixture PRIVATE
    "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(pond_flexible_tail_motor_launch_fixture PRIVATE cxx_std_17)

add_executable(pond_flexible_tail_motor_yaw_offset_fixture yaw-offset-fixture.cpp
    "${source_root}/core/checkpoint.cpp")
target_include_directories(pond_flexible_tail_motor_yaw_offset_fixture PRIVATE
    "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(pond_flexible_tail_motor_yaw_offset_fixture PRIVATE cxx_std_17)

add_executable(pond_flexible_tail_motor_yaw_trace yaw-trace.cpp
    "${source_root}/core/checkpoint.cpp")
target_include_directories(pond_flexible_tail_motor_yaw_trace PRIVATE
    "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(pond_flexible_tail_motor_yaw_trace PRIVATE cxx_std_17)

add_executable(pond_flexible_tail_motor_gait_seed gait-seed.cpp
    "${source_root}/core/checkpoint.cpp")
target_include_directories(pond_flexible_tail_motor_gait_seed PRIVATE
    "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(pond_flexible_tail_motor_gait_seed PRIVATE cxx_std_17)

add_executable(pond_flexible_tail_motor_navigation_ppo navigation-ppo.cu navigation-kernels.cu
    "${source_root}/nn/mlp_gpu.cu"
    "${source_root}/nn/gemm.cu"
    "${source_root}/nn/adam.cu"
    "${source_root}/rl/gaussian_kernels.cu"
    "${source_root}/rl/ppo_kernels.cu"
    "${source_root}/core/checkpoint.cpp")
set_target_properties(pond_flexible_tail_motor_navigation_ppo PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES CUDA_RUNTIME_LIBRARY Static)
target_include_directories(pond_flexible_tail_motor_navigation_ppo PRIVATE
    "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(pond_flexible_tail_motor_navigation_ppo PRIVATE cxx_std_17)
set_property(SOURCE navigation-ppo.cu navigation-kernels.cu
    APPEND PROPERTY COMPILE_OPTIONS --fmad=false)
