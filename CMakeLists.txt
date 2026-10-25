cmake_minimum_required(VERSION 3.24)

# Default to the RTX 50-series (Blackwell, sm_120). Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

project(CudaRL-Arena LANGUAGES CXX CUDA)

# Godot is the real-time frontend and is built by default. CI turns it off to
# build and test the core without the prebuilt godot-cpp library.
option(CUDARL_BUILD_GODOT "Build the Godot extension (needs a prebuilt godot-cpp)" ON)
option(CUDARL_BUILD_BENCH "Build nn_bench (needs the toolkit's cuBLAS component)" ON)
option(CUDARL_WARNINGS_AS_ERRORS "Treat warnings in this project's code as errors" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Static CRT and static CUDA runtime, to match the prebuilt godot-cpp library and
# so the extension DLL ships without any CUDA DLLs next to it.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
# Our warning level is set per target below; drop the default host level for CUDA files.
string(REPLACE "/W1" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

find_package(CUDAToolkit REQUIRED)

# Strict warnings for this project's own code. Third-party headers (CUDA, godot-cpp)
# come in as SYSTEM includes, so they're exempt.
function(cudarl_strict_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CXX>:/permissive->
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
            # C4211 fires in the host stub nvcc generates for every .cu file, not in our code.
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>)
        if(CUDARL_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/WX>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>
                $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
        endif()
    endif()
endfunction()

# --- Core: CUDA code and environments, no Godot dependency --------------------
add_library(cudarl_core STATIC
    src/cuda/device.cu
    src/envs/maze/maze_env.cpp
    src/envs/cartpole/cartpole.cu
    src/envs/racing/racing.cu
    src/rl/ppo_kernels.cu
    src/rl/ppo.cpp
    src/rl/gaussian_kernels.cu
    src/rl/racing_ppo.cpp
    src/nn/gemm.cu
    src/nn/gemm_bf16.cu
    src/nn/attention.cu
    src/nn/mlp_gpu.cu
    src/nn/adam.cu
    src/nn/loss.cu
    src/demos/function_fit.cpp
    src/demos/cartpole_showcase.cpp
    src/demos/racing_showcase.cpp
)
target_include_directories(cudarl_core PUBLIC src)
target_link_libraries(cudarl_core PUBLIC CUDA::cudart_static)
cudarl_strict_warnings(cudarl_core)

# --- Godot extension ------------------------------------------------------------
if(CUDARL_BUILD_GODOT)
    set(GODOT_CPP_DIR "${CMAKE_SOURCE_DIR}/external/godot-cpp")
    set(GODOT_CPP_LIB "${GODOT_CPP_DIR}/bin/libgodot-cpp.windows.template_release.x86_64.lib")
    if(NOT EXISTS "${GODOT_CPP_LIB}")
        message(FATAL_ERROR "godot-cpp library not found: ${GODOT_CPP_LIB}\n"
                            "Build it (see README) or configure with -DCUDARL_BUILD_GODOT=OFF.")
    endif()
    set(GODOT_CPP_INCLUDES "${GODOT_CPP_DIR}/include;${GODOT_CPP_DIR}/gen/include;${GODOT_CPP_DIR}/gdextension")
    add_library(godot-cpp STATIC IMPORTED)
    set_target_properties(godot-cpp PROPERTIES
        IMPORTED_LOCATION "${GODOT_CPP_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${GODOT_CPP_INCLUDES}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${GODOT_CPP_INCLUDES}"
    )

    add_library(cudarl SHARED
        src/godot/register_types.cpp
        src/godot/cuda_device.cpp
        src/godot/maze_arena.cpp
        src/godot/network_trainer.cpp
        src/godot/ppo_lab.cpp
        src/godot/racing_lab.cpp
    )
    target_link_libraries(cudarl PRIVATE cudarl_core godot-cpp)
    target_compile_definitions(cudarl PRIVATE TYPED_METHOD_BIND WINDOWS_ENABLED)
    cudarl_strict_warnings(cudarl)

    set(CUDARL_ADDON_BIN "${CMAKE_SOURCE_DIR}/godot-project/addons/cudarl/bin")
    set_target_properties(cudarl PROPERTIES
        PREFIX ""
        RUNTIME_OUTPUT_DIRECTORY         "${CUDARL_ADDON_BIN}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CUDARL_ADDON_BIN}"
    )
endif()

# --- Tests (run with: ctest --preset release) ---------------------------------
# GPU tests report SKIP when no CUDA device is present (e.g. on CI runners);
# set CUDARL_REQUIRE_GPU=1 to make a missing GPU a failure instead.
enable_testing()
add_executable(cudarl_tests
    tests/test_main.cpp
    tests/test_mlp_cpu.cpp
    tests/test_gemm.cpp
    tests/test_gemm_bf16.cpp
    tests/test_attention.cpp
    tests/test_mlp_gpu.cpp
    tests/test_adam.cpp
    tests/test_function_fit.cpp
    tests/test_cartpole.cpp
    tests/test_ppo.cpp
    tests/test_ppo_demo.cpp
    tests/test_racing.cpp
    tests/test_racing_ppo.cpp
)
target_include_directories(cudarl_tests PRIVATE tests)
target_link_libraries(cudarl_tests PRIVATE cudarl_core)
cudarl_strict_warnings(cudarl_tests)
add_test(NAME cudarl_tests COMMAND cudarl_tests)

# --- Benchmark (the only target that links cuBLAS, as a reference) ---------------
if(CUDARL_BUILD_BENCH)
    add_executable(nn_bench tools/nn_bench.cpp tools/attention_baseline.cu)
    target_link_libraries(nn_bench PRIVATE cudarl_core CUDA::cublas)
    cudarl_strict_warnings(nn_bench)
endif()
