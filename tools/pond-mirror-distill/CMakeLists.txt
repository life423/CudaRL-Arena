cmake_minimum_required(VERSION 3.24)
project(PondMirrorDistill LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
get_filename_component(root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
# Build the same native GPU GEMM, MLP, loss and Adam implementation as the PPO trainer.
add_subdirectory(../pond-ppo "${CMAKE_BINARY_DIR}/ppo" EXCLUDE_FROM_ALL)
set(identity "mirror-distill-v1;CXX=${CMAKE_CXX_COMPILER_VERSION};GPU=${CMAKE_CUDA_COMPILER_VERSION};sm120;")
foreach(path IN ITEMS tools/pond-mirror-distill/distill.cpp tools/pond-mirror-distill/CMakeLists.txt
        tools/pond-mirror-check/mirror.h tools/pond-mirror-check/check.cpp
        tools/pond-ppo/admission.cpp tools/pond-ppo/admission.h tools/pond-ppo/CMakeLists.txt
        src/rl/pond_ppo_contract.h src/nn/mlp_gpu.cu src/nn/mlp_gpu.h src/nn/mlp_spec.h src/nn/gemm.cu src/nn/gemm.h
        src/nn/adam.cu src/nn/adam.h src/nn/adam_config.h src/nn/loss.cu src/nn/loss.h src/cuda/cuda_memory.h
        src/core/checkpoint.cpp src/core/checkpoint.h)
    file(SHA256 "${root}/${path}" digest)
    string(APPEND identity "${path}=${digest};")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${root}/${path}")
endforeach()
string(SHA256 identity "${identity}")
file(WRITE "${CMAKE_BINARY_DIR}/distill_identity.h" "#pragma once\ninline constexpr char source_hash[]=\"${identity}\";\n")
add_executable(pond_mirror_distill distill.cpp ../pond-ppo/admission.cpp ../../src/core/checkpoint.cpp)
target_include_directories(pond_mirror_distill PRIVATE "${root}/src" "${root}/src/nn" "${CMAKE_BINARY_DIR}" "${CMAKE_BINARY_DIR}/ppo")
target_compile_options(pond_mirror_distill PRIVATE /W4 /WX /permissive-)
target_link_libraries(pond_mirror_distill PRIVATE pond_ppo_gpu_core bcrypt)
