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

if(CUDARL_BUILD_GODOT)
    set(CUDARL_GODOT_CPP_ROOT "${CMAKE_SOURCE_DIR}/external/godot-cpp" CACHE PATH "Matching Godot 4.4 C++ bindings")
    # The baseline and minimal worktree share this repository's existing dependency.
    if(NOT EXISTS "${CUDARL_GODOT_CPP_ROOT}/include/godot_cpp/godot.hpp" AND EXISTS "${CMAKE_SOURCE_DIR}/../../external/godot-cpp/include/godot_cpp/godot.hpp")
        get_filename_component(CUDARL_GODOT_CPP_ROOT "${CMAKE_SOURCE_DIR}/../../external/godot-cpp" ABSOLUTE)
    endif()
    set(CUDARL_GODOT_CPP_LIBRARY "${CUDARL_GODOT_CPP_ROOT}/bin/libgodot-cpp.windows.template_release.x86_64.lib" CACHE FILEPATH "Prebuilt platform godot-cpp library")
    if(NOT EXISTS "${CUDARL_GODOT_CPP_LIBRARY}")
        message(FATAL_ERROR "Set CUDARL_GODOT_CPP_ROOT/LIBRARY to matching bindings, or CUDARL_BUILD_GODOT=OFF")
    endif()
    add_library(godot_cpp STATIC IMPORTED)
    set_target_properties(godot_cpp PROPERTIES IMPORTED_LOCATION "${CUDARL_GODOT_CPP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDARL_GODOT_CPP_ROOT}/include;${CUDARL_GODOT_CPP_ROOT}/gen/include;${CUDARL_GODOT_CPP_ROOT}/gdextension"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${CUDARL_GODOT_CPP_ROOT}/include;${CUDARL_GODOT_CPP_ROOT}/gen/include;${CUDARL_GODOT_CPP_ROOT}/gdextension")
    add_library(creature_foundation SHARED src/godot/register_types.cpp src/godot/creature_lab.cpp)
    target_link_libraries(creature_foundation PRIVATE creature_physics godot_cpp)
    target_compile_definitions(creature_foundation PRIVATE TYPED_METHOD_BIND $<$<PLATFORM_ID:Windows>:WINDOWS_ENABLED>)
    set_target_properties(creature_foundation PROPERTIES PREFIX ""
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/godot-project/addons/creature_foundation/bin"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/godot-project/addons/creature_foundation/bin")
    project_warnings(creature_foundation)
endif()

include(CTest)
if(BUILD_TESTING)
add_executable(foundation_tests tests/test_main.cpp tests/test_physics.cpp tests/test_mlp_cpu.cpp tests/test_checkpoint.cpp)
target_include_directories(foundation_tests PRIVATE tests)
target_link_libraries(foundation_tests PRIVATE creature_physics)
if(CUDARL_ENABLE_CUDA)
    target_sources(foundation_tests PRIVATE tests/test_physics_cuda.cu tests/test_gemm.cpp tests/test_mlp_gpu.cpp tests/test_adam.cpp tests/test_ppo.cpp tests/test_gaussian_ppo.cpp)
    target_link_libraries(foundation_tests PRIVATE creature_gpu)
else()
    target_compile_definitions(foundation_tests PRIVATE CUDARL_WITH_CUDA=0)
endif()
project_warnings(foundation_tests)
add_test(NAME foundation COMMAND foundation_tests)
set_tests_properties(foundation PROPERTIES TIMEOUT 30)
endif()
