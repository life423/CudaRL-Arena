cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()
project(PondFlexibleTailChooser LANGUAGES CXX CUDA)
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

add_executable(pond_generate_coverage_map generate-coverage-map.cpp "${source_root}/core/checkpoint.cpp")
target_compile_features(pond_generate_coverage_map PRIVATE cxx_std_17)
target_include_directories(pond_generate_coverage_map PRIVATE "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")

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

# Same batch/PPO implementation, explicitly versioned rate/speed request task.
option(CUDARL_RELIABLE_CHOOSER "Fifteen-action chooser with demonstrated-competence bank curriculum" OFF)
option(CUDARL_COVERAGE_CHOOSER "Novelty-weighted safe transport reward for coverage continuation" OFF)
option(CUDARL_DISTANT_COVERAGE "Full-circle distant visitation sensory field" OFF)
option(CUDARL_APPENDED_COVERAGE "Append distant sensing after all 32 legacy inputs" OFF)
if(CUDARL_APPENDED_COVERAGE AND NOT CUDARL_DISTANT_COVERAGE)
    message(FATAL_ERROR "Appended sensing requires the distant sensory cache")
endif()
if(CUDARL_DISTANT_COVERAGE AND NOT CUDARL_COVERAGE_CHOOSER)
    message(FATAL_ERROR "Distant feedback uses the established coverage reward")
endif()
if(CUDARL_COVERAGE_CHOOSER AND NOT CUDARL_RELIABLE_CHOOSER)
    message(FATAL_ERROR "Coverage continuation requires the fifteen-action chooser")
endif()
set(rate_identity "rate-chooser-v1;reliable_${CUDARL_RELIABLE_CHOOSER};coverage_${CUDARL_COVERAGE_CHOOSER};distant_${CUDARL_DISTANT_COVERAGE};appended_${CUDARL_APPENDED_COVERAGE};sm_${CMAKE_CUDA_ARCHITECTURES};fmad=false")
foreach(path IN ITEMS train.cu rate_session.cuh bank_training_starts.h
    ../../src/rl/flexible_tail_distant_coverage.h ../../src/rl/flexible_tail_distant_map.h
    ../../src/rl/flexible_tail_rate_chooser.h ../../src/rl/flexible_tail_bank_observation.h
    ../../src/rl/flexible_tail_chooser.h ../../src/rl/flexible_tail_fast_steering.h
    ../../src/rl/flexible_tail_steering_policy.h ../../src/rl/flexible_tail_stroke_metrics.h
    ../../src/rl/flexible_tail_transition.h ../../src/physics/flexible_tail.h
    ../../src/environment/flexible_tail_clearance.h ../../src/environment/pond_geometry_v1.h
    ../../src/nn/mlp_gpu.cu ../../src/nn/gemm.cu ../../src/nn/adam.cu
    ../../src/rl/ppo_kernels.cu ../../src/rl/ppo_math.h ../../flexible-tail-preview/habitat.json)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${path}")
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${path}" part_hash)
    string(APPEND rate_identity ";${part_hash}")
endforeach()
string(SHA256 rate_identity_hash "${rate_identity}")
add_executable(pond_flexible_tail_rate_chooser train.cu
    "${source_root}/nn/mlp_gpu.cu" "${source_root}/nn/gemm.cu" "${source_root}/nn/adam.cu"
    "${source_root}/rl/ppo_kernels.cu" "${source_root}/core/checkpoint.cpp")
set_target_properties(pond_flexible_tail_rate_chooser PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES CUDA_RUNTIME_LIBRARY Static)
target_compile_features(pond_flexible_tail_rate_chooser PRIVATE cxx_std_17)
target_compile_definitions(pond_flexible_tail_rate_chooser PRIVATE CUDARL_SUSTAINED_CHOOSER=1
    CUDARL_RATE_CHOOSER=1 CUDARL_RATE_TASK_ID="${rate_identity_hash}")
target_include_directories(pond_flexible_tail_rate_chooser PRIVATE "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")
add_executable(pond_flexible_tail_rate_evaluate rate-evaluate.cpp "${source_root}/core/checkpoint.cpp")
target_compile_features(pond_flexible_tail_rate_evaluate PRIVATE cxx_std_17)
target_include_directories(pond_flexible_tail_rate_evaluate PRIVATE "${source_root}" "${source_root}/nn" "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(pond_flexible_tail_rate_evaluate PRIVATE Threads::Threads)
if(CUDARL_RELIABLE_CHOOSER)
    target_compile_definitions(pond_flexible_tail_rate_chooser PRIVATE CUDARL_RELIABLE_CHOOSER=1)
    target_compile_definitions(pond_flexible_tail_rate_evaluate PRIVATE CUDARL_RELIABLE_CHOOSER=1)
endif()
if(CUDARL_APPENDED_COVERAGE)
    target_compile_definitions(pond_flexible_tail_rate_chooser PRIVATE CUDARL_APPENDED_COVERAGE=1)
    target_compile_definitions(pond_flexible_tail_rate_evaluate PRIVATE CUDARL_APPENDED_COVERAGE=1)
endif()
if(CUDARL_COVERAGE_CHOOSER)
    target_compile_definitions(pond_flexible_tail_rate_chooser PRIVATE CUDARL_COVERAGE_CHOOSER=1)
endif()
if(CUDARL_DISTANT_COVERAGE)
    target_compile_definitions(pond_flexible_tail_rate_chooser PRIVATE CUDARL_DISTANT_COVERAGE=1)
endif()
