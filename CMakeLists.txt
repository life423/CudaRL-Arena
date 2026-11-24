cmake_minimum_required(VERSION 3.24)
project(CudaRLFoundation LANGUAGES CXX)

option(CUDARL_ENABLE_CUDA "Build shared CUDA physics checks and learning primitives" ON)
option(CUDARL_BUILD_GODOT "Build the single Godot inspection extension" ON)
option(CUDARL_WARNINGS_AS_ERRORS "Treat project warnings as errors" ON)
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

# The saved motor's original provenance is immutable. A changed pond runtime
# gets separate qualification; source hashes are never regenerated to bypass it.
add_library(saved_motor_contract STATIC src/nn/motor/saved_motor_contract.cpp)
target_include_directories(saved_motor_contract PUBLIC "${CMAKE_SOURCE_DIR}/src/nn")
project_warnings(saved_motor_contract)

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
    set(CUDARL_GODOT_OUTPUT_DIR "${CMAKE_BINARY_DIR}/godot-staged" CACHE PATH
        "Frozen viewer Release DLL directory; override to verify a build without replacing the live DLL")
    add_library(godot_cpp STATIC IMPORTED)
    set_target_properties(godot_cpp PROPERTIES IMPORTED_LOCATION "${godot_cpp_library}"
        INTERFACE_INCLUDE_DIRECTORIES "${godot_cpp_root}/include;${godot_cpp_root}/gen/include;${godot_cpp_root}/gdextension"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${godot_cpp_root}/include;${godot_cpp_root}/gen/include;${godot_cpp_root}/gdextension")
    add_library(frozen_swimmer SHARED src/godot/register_live_types.cpp src/godot/hold_lab_preview.cpp)
    target_link_libraries(frozen_swimmer PRIVATE creature_physics saved_motor_contract godot_cpp)
    target_compile_definitions(frozen_swimmer PRIVATE TYPED_METHOD_BIND $<$<PLATFORM_ID:Windows>:WINDOWS_ENABLED>)
    # A cached developer build must never overwrite the shipped, hash-bound DLL.
    cmake_path(ABSOLUTE_PATH CUDARL_GODOT_OUTPUT_DIR BASE_DIRECTORY "${CMAKE_BINARY_DIR}" NORMALIZE OUTPUT_VARIABLE godot_output_absolute)
    string(TOLOWER "${godot_output_absolute}/" godot_output_lower)
    string(TOLOWER "${CMAKE_SOURCE_DIR}/live/" live_output_lower)
    string(FIND "${godot_output_lower}" "${live_output_lower}" live_output_index)
    if(live_output_index EQUAL 0)
        set(CUDARL_GODOT_OUTPUT_DIR "${CMAKE_BINARY_DIR}/godot-staged" CACHE PATH "Staged native developer library" FORCE)
        message(STATUS "Redirected native output out of the verified live project: ${CUDARL_GODOT_OUTPUT_DIR}")
    endif()
    set_target_properties(frozen_swimmer PROPERTIES PREFIX "" OUTPUT_NAME "frozen_swimmer-steering-v1" RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CUDARL_GODOT_OUTPUT_DIR}")
    project_warnings(frozen_swimmer)
endif()

include(CTest)
if(BUILD_TESTING)
add_executable(foundation_tests tests/test_main.cpp tests/test_check.cpp tests/test_joint_motor.cpp tests/test_straight_task.cpp tests/test_speed_request.cpp tests/test_saved_motor_contract.cpp tests/test_physics.cpp tests/test_mlp_cpu.cpp tests/test_checkpoint.cpp tests/test_ppo_checkpoint.cpp)
target_sources(foundation_tests PRIVATE tests/test_pond_request_feedback.cpp tests/test_pond_turn_schedule.cpp)
target_include_directories(foundation_tests PRIVATE tests)
target_link_libraries(foundation_tests PRIVATE creature_physics saved_motor_contract)
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
