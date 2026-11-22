cmake_minimum_required(VERSION 3.24)
project(CudaRLFoundation LANGUAGES CXX)

option(CUDARL_ENABLE_CUDA "Build shared CUDA physics checks and learning primitives" ON)
option(CUDARL_BUILD_GODOT "Build the single Godot inspection extension" ON)
option(CUDARL_WARNINGS_AS_ERRORS "Treat project warnings as errors" ON)
option(CUDARL_BUILD_PPO "Build the native PPO trainer (never run by CTest)" ON)
option(CUDARL_BUILD_LEARNED_AMPLITUDE "Enable the opt-in whole-episode PPO compilation target; never executes training" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

function(project_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CXX>:/permissive->
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>)
        if(CUDARL_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/WX>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>
                $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
        endif()
    else()
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra> $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
        if(CUDARL_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
        endif()
    endif()
endfunction()

# The physical world has no Godot, learner, food, reward or CUDA dependency.
add_library(creature_physics STATIC src/core/checkpoint.cpp src/simulation/world.cpp)
target_include_directories(creature_physics PUBLIC src)
set_target_properties(creature_physics PROPERTIES POSITION_INDEPENDENT_CODE ON)
project_warnings(creature_physics)

if(CUDARL_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 120)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
    string(REPLACE "/W1" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
    add_library(creature_gpu STATIC
        src/simulation/world_cuda.cu src/cuda/device.cu
        src/nn/gemm.cu src/nn/mlp_gpu.cu src/nn/adam.cu src/nn/loss.cu
        src/rl/ppo_kernels.cu src/rl/gaussian_kernels.cu)
    target_include_directories(creature_gpu PUBLIC src)
    target_link_libraries(creature_gpu PUBLIC creature_physics CUDA::cudart_static)
    target_compile_definitions(creature_gpu PUBLIC CUDARL_WITH_CUDA=1)
    project_warnings(creature_gpu)
endif()

set(FROZEN_CONTRACT_ENTRIES "")
set(FROZEN_CONTRACT_COUNT 0)
file(STRINGS "${CMAKE_SOURCE_DIR}/src/nn/motor/controller_contract.txt" controller_sources)
foreach(path IN LISTS controller_sources)
    file(READ "${CMAKE_SOURCE_DIR}/${path}" contract_text)
    string(REPLACE "\r\n" "\n" contract_text "${contract_text}")
    string(SHA256 contract_hash "${contract_text}")
    string(APPEND FROZEN_CONTRACT_ENTRIES "\"${contract_hash}\",")
    math(EXPR FROZEN_CONTRACT_COUNT "${FROZEN_CONTRACT_COUNT}+1")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/${path}")
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/nn/motor/controller_contract.txt")
configure_file(src/nn/motor/frozen_contract.in.h generated/frozen_contract.h @ONLY)
add_library(frozen_controller STATIC src/nn/motor/frozen_controller.cpp)
target_link_libraries(frozen_controller PUBLIC creature_physics)
target_include_directories(frozen_controller PUBLIC "${CMAKE_SOURCE_DIR}/src/nn" PRIVATE "${CMAKE_BINARY_DIR}/generated")
project_warnings(frozen_controller)

add_executable(frozen_brake_probe tools/frozen_brake_probe.cpp)
target_link_libraries(frozen_brake_probe PRIVATE creature_physics)
target_include_directories(frozen_brake_probe PRIVATE "${CMAKE_SOURCE_DIR}/src/nn")
project_warnings(frozen_brake_probe)

add_executable(motor_compare tools/motor_compare.cpp)
target_link_libraries(motor_compare PRIVATE frozen_controller)
project_warnings(motor_compare)

if(CUDARL_ENABLE_CUDA AND CUDARL_BUILD_PPO)
    if(CUDARL_BUILD_LEARNED_AMPLITUDE)
        # Explicit compilation only. Closed research budgets are not reopened by building.
        add_executable(ppo_learned_amplitude EXCLUDE_FROM_ALL tools/hold_learned_amplitude_study.cu)
        target_compile_definitions(ppo_learned_amplitude PRIVATE CUDARL_APPEND_HOLD41=1)
        target_link_libraries(ppo_learned_amplitude PRIVATE creature_gpu frozen_controller)
        project_warnings(ppo_learned_amplitude)
    endif()
    add_executable(ppo_rest_hold tools/rest_hold_study.cu)
    target_compile_definitions(ppo_rest_hold PRIVATE CUDARL_APPEND_HOLD41=1)
    target_link_libraries(ppo_rest_hold PRIVATE creature_gpu frozen_controller)
    project_warnings(ppo_rest_hold)
    add_executable(ppo_hold_scale tools/hold_scale_study.cu)
    target_compile_definitions(ppo_hold_scale PRIVATE CUDARL_APPEND_HOLD41=1)
    target_link_libraries(ppo_hold_scale PRIVATE creature_gpu frozen_controller)
    project_warnings(ppo_hold_scale)
    add_executable(ppo_append_hold tools/append_hold8.cu)
    target_compile_definitions(ppo_append_hold PRIVATE CUDARL_APPEND_HOLD41=1)
    target_link_libraries(ppo_append_hold PRIVATE creature_gpu frozen_controller)
    project_warnings(ppo_append_hold)
    add_executable(ppo_motor tools/ppo_motor.cu)
    target_link_libraries(ppo_motor PRIVATE creature_gpu frozen_controller)
    project_warnings(ppo_motor)
endif()

if(CUDARL_BUILD_GODOT)
    set(default_godot_cpp_root "${CMAKE_SOURCE_DIR}/external/godot-cpp")
    set(CUDARL_GODOT_CPP_ROOT "${default_godot_cpp_root}" CACHE PATH "Matching Godot 4.4 C++ bindings")
    set(godot_cpp_root "${CUDARL_GODOT_CPP_ROOT}")
    # A linked worktree can share the original checkout's built dependency.
    if(godot_cpp_root STREQUAL default_godot_cpp_root AND
       NOT EXISTS "${godot_cpp_root}/include/godot_cpp/godot.hpp" AND
       EXISTS "${CMAKE_SOURCE_DIR}/../../external/godot-cpp/include/godot_cpp/godot.hpp")
        get_filename_component(godot_cpp_root "${CMAKE_SOURCE_DIR}/../../external/godot-cpp" ABSOLUTE)
    endif()
    set(CUDARL_GODOT_CPP_LIBRARY "" CACHE FILEPATH "Override the prebuilt Windows x64 godot-cpp release library")
    set(godot_cpp_library "${CUDARL_GODOT_CPP_LIBRARY}")
    if(NOT godot_cpp_library)
        set(godot_cpp_library "${godot_cpp_root}/bin/libgodot-cpp.windows.template_release.x86_64.lib")
    endif()
    foreach(binding IN ITEMS
            "${godot_cpp_root}/include/godot_cpp/godot.hpp"
            "${godot_cpp_root}/gen/include/godot_cpp/classes/project_settings.hpp"
            "${godot_cpp_root}/gdextension/gdextension_interface.h"
            "${godot_cpp_library}")
        if(NOT EXISTS "${binding}")
            message(FATAL_ERROR "Missing Godot C++ binding: ${binding}\n"
                "Use matching, built godot-cpp bindings via CUDARL_GODOT_CPP_ROOT and, if needed, "
                "CUDARL_GODOT_CPP_LIBRARY; or use CUDARL_BUILD_GODOT=OFF.")
        endif()
    endforeach()
    message(STATUS "Godot C++ bindings: ${godot_cpp_root}")
    set(CUDARL_GODOT_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/live/addons/frozen_swimmer/bin" CACHE PATH
        "Frozen viewer Release DLL directory; override to verify a build without replacing the live DLL")
    add_library(godot_cpp STATIC IMPORTED)
    set_target_properties(godot_cpp PROPERTIES IMPORTED_LOCATION "${godot_cpp_library}"
        INTERFACE_INCLUDE_DIRECTORIES "${godot_cpp_root}/include;${godot_cpp_root}/gen/include;${godot_cpp_root}/gdextension"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${godot_cpp_root}/include;${godot_cpp_root}/gen/include;${godot_cpp_root}/gdextension")
    add_library(frozen_swimmer SHARED src/godot/register_live_types.cpp src/godot/frozen_swimmer.cpp src/godot/hold_lab_preview.cpp)
    target_link_libraries(frozen_swimmer PRIVATE frozen_controller godot_cpp)
    target_compile_definitions(frozen_swimmer PRIVATE TYPED_METHOD_BIND $<$<PLATFORM_ID:Windows>:WINDOWS_ENABLED>)
    set_target_properties(frozen_swimmer PROPERTIES PREFIX "" RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CUDARL_GODOT_OUTPUT_DIR}")
    project_warnings(frozen_swimmer)
endif()

include(CTest)
if(BUILD_TESTING)
add_executable(foundation_tests tests/test_main.cpp tests/test_check.cpp tests/test_joint_motor.cpp tests/test_straight_task.cpp tests/test_speed_request.cpp tests/test_physics.cpp tests/test_mlp_cpu.cpp tests/test_checkpoint.cpp tests/test_ppo_checkpoint.cpp)
target_include_directories(foundation_tests PRIVATE tests)
target_link_libraries(foundation_tests PRIVATE creature_physics frozen_controller)
if(CUDARL_ENABLE_CUDA)
    target_sources(foundation_tests PRIVATE tests/test_physics_cuda.cu tests/test_gemm.cpp tests/test_mlp_gpu.cpp tests/test_adam.cpp tests/test_ppo.cpp tests/test_gaussian_ppo.cpp tests/test_ar_gaussian.cpp)
    target_link_libraries(foundation_tests PRIVATE creature_gpu)
else()
    target_compile_definitions(foundation_tests PRIVATE CUDARL_WITH_CUDA=0)
endif()
project_warnings(foundation_tests)
add_test(NAME foundation COMMAND foundation_tests)
set_tests_properties(foundation PROPERTIES TIMEOUT 30)
endif()
