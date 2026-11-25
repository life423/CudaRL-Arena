cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()
project(PondFlexibleTailChooser LANGUAGES CXX CUDA)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
find_package(CUDAToolkit REQUIRED)

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

add_executable(pond_flexible_tail_chooser_ppo train.cu
    "${source_root}/nn/mlp_gpu.cu"
    "${source_root}/nn/gemm.cu"
    "${source_root}/nn/adam.cu"
    "${source_root}/rl/ppo_kernels.cu"
    "${source_root}/core/checkpoint.cpp")
set_target_properties(pond_flexible_tail_chooser_ppo PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES CUDA_RUNTIME_LIBRARY Static)
target_compile_features(pond_flexible_tail_chooser_ppo PRIVATE cxx_std_17)
target_include_directories(pond_flexible_tail_chooser_ppo PRIVATE
    "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")
set_property(SOURCE train.cu APPEND PROPERTY COMPILE_OPTIONS --fmad=false)

add_executable(pond_flexible_tail_chooser_eval evaluate.cpp "${source_root}/core/checkpoint.cpp")
target_compile_features(pond_flexible_tail_chooser_eval PRIVATE cxx_std_17)
target_include_directories(pond_flexible_tail_chooser_eval PRIVATE
    "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(pond_flexible_tail_steering_check steering-check.cpp "${source_root}/core/checkpoint.cpp")
target_compile_features(pond_flexible_tail_steering_check PRIVATE cxx_std_17)
target_include_directories(pond_flexible_tail_steering_check PRIVATE
    "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")

set(session_identity "chooser-session-v2;sm_${CMAKE_CUDA_ARCHITECTURES};fmad=false")
foreach(path IN ITEMS train.cu sustained_session.cuh bank_training_starts.h
    ../../src/rl/flexible_tail_beat_course.h
    ../../src/rl/flexible_tail_chooser.h ../../src/rl/flexible_tail_gait.h
    ../../src/rl/flexible_tail_yaw_offset.h ../../src/rl/flexible_tail_transition.h
    ../../src/physics/flexible_tail.h ../../src/environment/flexible_tail_clearance.h
    ../../src/environment/pond_geometry_v1.h ../../src/nn/adam.cu)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${path}")
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${path}" component_hash)
    string(APPEND session_identity ";${component_hash}")
endforeach()
string(SHA256 session_identity_hash "${session_identity}")
add_executable(pond_flexible_tail_chooser_session train.cu
    "${source_root}/nn/mlp_gpu.cu" "${source_root}/nn/gemm.cu" "${source_root}/nn/adam.cu"
    "${source_root}/rl/ppo_kernels.cu" "${source_root}/core/checkpoint.cpp")
set_target_properties(pond_flexible_tail_chooser_session PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES CUDA_RUNTIME_LIBRARY Static)
target_compile_features(pond_flexible_tail_chooser_session PRIVATE cxx_std_17)
target_compile_definitions(pond_flexible_tail_chooser_session PRIVATE
    CUDARL_SUSTAINED_CHOOSER=1 CUDARL_SESSION_TASK_ID="${session_identity_hash}")
target_include_directories(pond_flexible_tail_chooser_session PRIVATE
    "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")
