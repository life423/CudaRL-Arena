cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()
project(PondFlexibleTailSteering LANGUAGES CXX CUDA)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(source_root "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
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

# An optimizer session cannot be resumed against changed task, dynamics, math,
# model layout, or generated habitat geometry.
set(task_identity "flexible-tail-steering-session-v1;sm_${CMAKE_CUDA_ARCHITECTURES};fmad=false")
foreach(path IN ITEMS train.cu session.cuh
    ../../src/rl/flexible_tail_fast_steering.h ../../src/rl/flexible_tail_yaw_offset.h
    ../../src/rl/flexible_tail_gait.h ../../src/rl/flexible_tail_navigation.h
    ../../src/rl/flexible_tail_transition.h ../../src/rl/flexible_tail_motor.h
    ../../src/physics/flexible_tail.h ../../src/environment/flexible_tail_clearance.h
    ../../src/environment/pond_geometry_v1.h ../../src/nn/mlp_gpu.cu
    ../../src/nn/gemm.cu ../../src/nn/adam.cu ../../src/rl/gaussian_kernels.cu
    ../../src/rl/gaussian_math.h ../../src/rl/ppo_kernels.cu ../../src/rl/ppo_math.h
    ../../flexible-tail-preview/habitat.json)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${path}")
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${path}" component_hash)
    string(APPEND task_identity ";${component_hash}")
endforeach()
string(SHA256 task_identity_hash "${task_identity}")

add_executable(pond_flexible_tail_steering_ppo train.cu
    "${source_root}/nn/mlp_gpu.cu" "${source_root}/nn/gemm.cu" "${source_root}/nn/adam.cu"
    "${source_root}/rl/gaussian_kernels.cu" "${source_root}/rl/ppo_kernels.cu"
    "${source_root}/core/checkpoint.cpp")
set_target_properties(pond_flexible_tail_steering_ppo PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES CUDA_RUNTIME_LIBRARY Static)
target_compile_features(pond_flexible_tail_steering_ppo PRIVATE cxx_std_17)
target_compile_definitions(pond_flexible_tail_steering_ppo PRIVATE
    CUDARL_STEERING_TASK_ID="${task_identity_hash}")
target_include_directories(pond_flexible_tail_steering_ppo PRIVATE
    "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_options(pond_flexible_tail_steering_ppo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

add_executable(pond_flexible_tail_steering_evaluate evaluate.cpp "${source_root}/core/checkpoint.cpp")
target_compile_features(pond_flexible_tail_steering_evaluate PRIVATE cxx_std_17)
target_include_directories(pond_flexible_tail_steering_evaluate PRIVATE
    "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(pond_flexible_tail_steering_evaluate PRIVATE Threads::Threads)
