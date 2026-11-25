cmake_minimum_required(VERSION 3.24)
project(PondCandidateCheck LANGUAGES CXX)
option(CUDARL_RESIDUAL_EVALUATOR "Build the frozen-base residual policy assessment" OFF)
option(CUDARL_RESIDUAL_FOUR_EVALUATOR "Assess the separate four-axis residual policy" OFF)
get_filename_component(worktree "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(identity "CPU-evaluator-v3;CXX=${CMAKE_CXX_COMPILER_VERSION};runtime4;query-arithmetic2;conditional-scheduler-stride;observation5-or6;full-baseline-retention;all-soft-cases;41-or42x32x32x4;external-distilled-actor-v1;")
set(manifest "")
string(APPEND identity "residual-evaluator=${CUDARL_RESIDUAL_EVALUATOR};")
string(APPEND identity "four-residual-evaluator=${CUDARL_RESIDUAL_FOUR_EVALUATOR};")
set(residual_contract "tools/pond-residual-ppo/contract.h")
if(CUDARL_RESIDUAL_FOUR_EVALUATOR)
    set(residual_contract "tools/pond-residual-four-ppo/contract.h")
endif()
foreach(relative IN ITEMS
        tools/pond-candidate-check/check.cpp tools/pond-candidate-check/CMakeLists.txt
        tools/pond-ppo/turn_course_observation.h
        tools/pond-residual-ppo/evaluator.h ${residual_contract}
        src/rl/pond_ppo_contract.h src/rl/pond_ppo_task.h src/rl/pond_ppo_schedule.h src/simulation/pond_runtime.h
        src/environment/pond_geometry_v1.h src/environment/pond_body_geometry.h src/environment/pond_swept_clearance.h
        src/nn/motor/pond_motor_observation.h src/nn/motor/movement_model.h src/nn/motor/joint_motor.h
        src/physics/dynamics.h src/physics/contact.h src/simulation/world.h
        src/nn/mlp_cpu.h src/nn/mlp_spec.h src/nn/motor/saved_motor_contract.h
        src/nn/motor/saved_motor_contract.cpp src/rl/learned_amplitude_math.h
        src/core/checkpoint.h src/core/checkpoint.cpp
        assets/ppo-four-axis-checkpoints/update-336-episode64/long-windows.csv)
    file(SHA256 "${worktree}/${relative}" digest)
    string(APPEND identity "${relative}:${digest};")
    string(APPEND manifest "${relative} ${digest}\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${worktree}/${relative}")
endforeach()
string(SHA256 digest "${identity}")
file(WRITE "${CMAKE_BINARY_DIR}/pond_candidate_identity.h" "#pragma once\ninline constexpr char pond_candidate_source_sha256[]=\"${digest}\";\n")
file(WRITE "${CMAKE_BINARY_DIR}/source-manifest.txt" "${manifest}")
add_executable(pond_candidate_check check.cpp ../../src/core/checkpoint.cpp ../../src/nn/motor/saved_motor_contract.cpp)
target_include_directories(pond_candidate_check PRIVATE ../../src ../../src/nn "${CMAKE_BINARY_DIR}")
set_source_files_properties(check.cpp PROPERTIES COMPILE_DEFINITIONS "NOMINMAX;WIN32_LEAN_AND_MEAN")
target_compile_options(pond_candidate_check PRIVATE /W4 /WX /permissive-)
if(CUDARL_RESIDUAL_EVALUATOR)
    target_compile_definitions(pond_candidate_check PRIVATE CUDARL_RESIDUAL_EVALUATOR)
endif()
if(CUDARL_RESIDUAL_FOUR_EVALUATOR)
    target_compile_definitions(pond_candidate_check PRIVATE CUDARL_RESIDUAL_EVALUATOR CUDARL_RESIDUAL_FOUR_EVALUATOR)
    add_executable(pond_course_rate_pure ../pond-residual-four-ppo/course_rate_reward_test.cpp)
    target_include_directories(pond_course_rate_pure PRIVATE ../../src ../../src/nn)
endif()
target_link_libraries(pond_candidate_check PRIVATE bcrypt)
