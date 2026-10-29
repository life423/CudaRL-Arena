cmake_minimum_required(VERSION 3.24)

# Default to the RTX 50-series (Blackwell, sm_120). Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

project(CudaRL-Arena LANGUAGES CXX)
option(CUDARL_ENABLE_CUDA "Build NVIDIA training and optional CUDA playback" ON)
if(CUDARL_ENABLE_CUDA)
    enable_language(CUDA)
endif()

# Godot is the real-time frontend and is built by default. CI turns it off to
# build and test the core without the prebuilt godot-cpp library.
option(CUDARL_BUILD_GODOT "Build the Godot extension (needs a prebuilt godot-cpp)" ON)
option(CUDARL_BUILD_BENCH "Build benchmarks (nn_bench needs the toolkit's cuBLAS component)" ON)
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

if(CUDARL_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

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

# Portable runtime: the exact saved networks and shared physical equations,
# without a CUDA compiler, headers, driver or runtime dependency.
add_library(cudarl_runtime STATIC src/core/checkpoint.cpp src/demos/pond_playback.cpp src/demos/feeding_cpu_session.cpp)
set_target_properties(cudarl_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(cudarl_runtime PUBLIC src)
target_compile_definitions(cudarl_runtime PRIVATE CUDARL_WITH_CUDA=0)
target_link_libraries(cudarl_runtime PRIVATE ${CMAKE_DL_LIBS})
cudarl_strict_warnings(cudarl_runtime)
if(NOT CUDARL_ENABLE_CUDA)
    include(cmake/PortablePond.cmake)
    return()
endif()

# --- Core: CUDA code and environments, no Godot dependency --------------------
add_library(cudarl_core STATIC
    src/cuda/device.cu
    src/envs/maze/maze_env.cpp
    src/envs/cartpole/cartpole.cu
    src/envs/racing/racing.cu
    src/envs/swimmer/swimmer.cu
    src/envs/swimmer/foraging.cu
    src/envs/swimmer/deep.cu
    src/envs/swimmer/deep_pond.cu
    src/envs/swimmer/pond_selector.cu
    src/envs/swimmer/progress_selector.cu
    src/envs/swimmer/feeding.cu
    src/envs/swimmer/arch_batch.cu
    src/demos/feeding_session.cu
    src/demos/arch_session.cu
    src/rl/ppo_kernels.cu
    src/rl/ppo.cpp
    src/rl/gaussian_kernels.cu
    src/rl/racing_ppo.cpp
    src/nn/gemm.cu
    src/nn/gemm_bf16.cu
    src/nn/attention.cu
    src/nn/attention_bf16.cu
    src/nn/mlp_gpu.cu
    src/nn/adam.cu
    src/nn/loss.cu
    src/demos/function_fit.cpp
    src/demos/cartpole_showcase.cpp
    src/demos/racing_showcase.cpp
)
target_include_directories(cudarl_core PUBLIC src)
target_link_libraries(cudarl_core PUBLIC cudarl_runtime CUDA::cudart_static)
target_compile_definitions(cudarl_core PUBLIC CUDARL_WITH_CUDA=1)
cudarl_strict_warnings(cudarl_core)

add_library(cudarl_pond_cuda SHARED src/demos/pond_playback_cuda.cu)
target_link_libraries(cudarl_pond_cuda PRIVATE cudarl_core)
cudarl_strict_warnings(cudarl_pond_cuda)
set(CUDARL_ADDON_BIN "${CMAKE_SOURCE_DIR}/godot-project/addons/cudarl/bin" CACHE PATH "Godot extension output directory")
set_target_properties(cudarl_pond_cuda PROPERTIES PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CUDARL_ADDON_BIN}")

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
        src/godot/tadpole_lab.cpp
        src/godot/deep_water_lab.cpp
        src/godot/deep_pond_lab.cpp
    )
    add_dependencies(cudarl cudarl_pond_cuda)
    target_link_libraries(cudarl PRIVATE cudarl_core godot-cpp)
    target_compile_definitions(cudarl PRIVATE TYPED_METHOD_BIND WINDOWS_ENABLED)
    cudarl_strict_warnings(cudarl)

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
    tests/test_attention_bf16.cpp
    tests/test_mlp_gpu.cpp
    tests/test_adam.cpp
    tests/test_function_fit.cpp
    tests/test_cartpole.cpp
    tests/test_ppo.cpp
    tests/test_ppo_demo.cpp
    tests/test_racing.cpp
    tests/test_swimmer.cpp
    tests/test_swimmer_inertial.cpp
    tests/test_foraging.cpp
    tests/test_deep_water.cpp
    tests/test_deep_solver.cu
    tests/test_deep_pond.cpp
    tests/test_pond_current.cpp
    tests/test_pond_playback_reliability.cpp
    tests/test_arch_geometry.cpp
    tests/test_arch_navigation.cpp
    tests/test_arch_current.cpp
    tests/test_cove_geometry.cpp
    tests/test_fixed_target.cpp
    tests/test_progress_selector.cpp
    tests/test_pond_selector.cpp
    tests/test_training_resume.cpp
    tests/test_race.cpp
    tests/test_racing_ppo.cpp
)
target_include_directories(cudarl_tests PRIVATE tests)
target_link_libraries(cudarl_tests PRIVATE cudarl_core)
add_dependencies(cudarl_tests cudarl_pond_cuda)
target_compile_definitions(cudarl_tests PRIVATE CUDARL_POND_MODULE="$<TARGET_FILE:cudarl_pond_cuda>")
cudarl_strict_warnings(cudarl_tests)
add_test(NAME cudarl_tests COMMAND cudarl_tests)

# --- Benchmarks (only nn_bench links cuBLAS, as a reference) -------------------
if(CUDARL_BUILD_BENCH)
    add_executable(arch_resume_verify tools/arch_resume_verify.cpp)
    target_link_libraries(arch_resume_verify PRIVATE cudarl_core)
    cudarl_strict_warnings(arch_resume_verify)
    add_executable(feeding_heldout_eval tools/feeding_heldout_eval.cu)
    target_link_libraries(feeding_heldout_eval PRIVATE cudarl_core)
    cudarl_strict_warnings(feeding_heldout_eval)
    add_executable(pond_reliability tools/pond_reliability.cpp)
    target_link_libraries(pond_reliability PRIVATE cudarl_runtime)
    target_compile_definitions(pond_reliability PRIVATE CUDARL_WITH_CUDA=0)
    cudarl_strict_warnings(pond_reliability)
    add_executable(pond_bank_study tools/pond_bank_study.cu)
    target_link_libraries(pond_bank_study PRIVATE cudarl_core)
    cudarl_strict_warnings(pond_bank_study)
    add_executable(pond_fixed_target_eval tools/pond_fixed_target_eval.cu)
    target_link_libraries(pond_fixed_target_eval PRIVATE cudarl_core)
    cudarl_strict_warnings(pond_fixed_target_eval)
    add_executable(pond_progress_profile tools/pond_progress_profile.cpp)
    target_link_libraries(pond_progress_profile PRIVATE cudarl_core)
    set(CUDARL_NVTX_INCLUDE_DIR "" CACHE PATH "Optional NVTX headers for profiling-only phase labels")
    if(EXISTS "${CUDARL_NVTX_INCLUDE_DIR}/nvtx3/nvToolsExt.h")
        target_include_directories(pond_progress_profile SYSTEM PRIVATE "${CUDARL_NVTX_INCLUDE_DIR}")
        target_compile_definitions(pond_progress_profile PRIVATE CUDARL_PROFILE_NVTX=1)
    endif()
    cudarl_strict_warnings(pond_progress_profile)
    add_executable(pond_progress_train tools/pond_progress_train.cpp)
    target_link_libraries(pond_progress_train PRIVATE cudarl_core)
    cudarl_strict_warnings(pond_progress_train)
    add_executable(pond_current_eval tools/pond_current_eval.cu)
    target_link_libraries(pond_current_eval PRIVATE cudarl_core)
    cudarl_strict_warnings(pond_current_eval)
    add_executable(pond_recovery_probe tools/pond_recovery_probe.cpp)
    target_link_libraries(pond_recovery_probe PRIVATE cudarl_core)
    cudarl_strict_warnings(pond_recovery_probe)
    add_executable(pond_selector_train tools/pond_selector_train.cpp)
    target_link_libraries(pond_selector_train PRIVATE cudarl_core)
    cudarl_strict_warnings(pond_selector_train)
    add_executable(pond_choice_eval tools/pond_choice_eval.cpp)
    target_link_libraries(pond_choice_eval PRIVATE cudarl_core)
    cudarl_strict_warnings(pond_choice_eval)
    add_executable(deep_swimmer_train tools/deep_swimmer_train.cpp)
    target_link_libraries(deep_swimmer_train PRIVATE cudarl_core)
    cudarl_strict_warnings(deep_swimmer_train)
    add_executable(pond_probe tools/pond_probe.cpp)
    target_link_libraries(pond_probe PRIVATE cudarl_core)
    cudarl_strict_warnings(pond_probe)
    add_executable(swimmer_experiment tools/swimmer_experiment.cpp)
    target_link_libraries(swimmer_experiment PRIVATE cudarl_core)
    cudarl_strict_warnings(swimmer_experiment)
    add_executable(foraging_experiment tools/foraging_experiment.cpp)
    target_link_libraries(foraging_experiment PRIVATE cudarl_core)
    cudarl_strict_warnings(foraging_experiment)
    add_executable(food_order_bias tools/food_order_bias.cpp)
    target_link_libraries(food_order_bias PRIVATE cudarl_core)
    cudarl_strict_warnings(food_order_bias)
    add_executable(shuffle_experiment tools/shuffle_experiment.cpp)
    target_link_libraries(shuffle_experiment PRIVATE cudarl_core)
    cudarl_strict_warnings(shuffle_experiment)
    add_executable(paired_pond_eval tools/paired_pond_eval.cpp)
    target_link_libraries(paired_pond_eval PRIVATE cudarl_core)
    cudarl_strict_warnings(paired_pond_eval)
    add_executable(export_demo_policies tools/export_demo_policies.cpp)
    target_link_libraries(export_demo_policies PRIVATE cudarl_core)
    cudarl_strict_warnings(export_demo_policies)
    # End-to-end racing benchmark uses only our core and the CUDA runtime.
    add_executable(racing_bench tools/racing_bench.cpp)
    target_link_libraries(racing_bench PRIVATE cudarl_core)
    cudarl_strict_warnings(racing_bench)
    add_executable(nn_bench tools/nn_bench.cpp tools/attention_baseline.cu)
    target_link_libraries(nn_bench PRIVATE cudarl_core CUDA::cublas)
    cudarl_strict_warnings(nn_bench)
endif()
