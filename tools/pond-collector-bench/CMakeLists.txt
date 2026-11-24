cmake_minimum_required(VERSION 3.24)
project(PondCollectorReplay LANGUAGES CXX)
get_filename_component(worktree "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
find_package(CUDAToolkit REQUIRED)
set(native_core "${worktree}/build/pt26/native/Release/pond_ppo_gpu_core.lib" CACHE FILEPATH "Already compiled ordinary native MLP library")
set(trainer_identity_dir "${worktree}/build/pt26/native" CACHE PATH "Frozen trainer identity header")
set(identity "CXX=${CMAKE_CXX_COMPILER_VERSION};host-only-replay;")
file(SHA256 "${native_core}" native_core_sha256)
string(APPEND identity "native_core:${native_core_sha256};")
file(SHA256 "${trainer_identity_dir}/pond_ppo_identity.h" trainer_header_sha256)
string(APPEND identity "trainer_identity_header:${trainer_header_sha256};")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${native_core}" "${trainer_identity_dir}/pond_ppo_identity.h")
foreach(relative IN ITEMS tools/pond-collector-bench/bench.cpp tools/pond-collector-bench/CMakeLists.txt
        tools/pond-ppo/admission.h tools/pond-ppo/admission.cpp src/rl/pond_ppo_task.h src/rl/pond_ppo_contract.h
        src/cuda/pond_math_profile.h src/simulation/pond_runtime.h src/environment/pond_geometry_v1.h
        src/environment/pond_body_geometry.h src/nn/motor/pond_motor_observation.h src/nn/motor/movement_model.h
        src/nn/motor/joint_motor.h src/nn/motor/state.h src/physics/dynamics.h src/physics/contact.h
        src/nn/mlp_gpu.h src/nn/mlp_spec.h src/core/checkpoint.h src/core/checkpoint.cpp)
    file(SHA256 "${worktree}/${relative}" digest)
    string(APPEND identity "${relative}:${digest};")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${worktree}/${relative}")
endforeach()
string(SHA256 bench_identity "${identity}")
file(WRITE "${CMAKE_BINARY_DIR}/collector_bench_identity.h" "#pragma once\ninline constexpr char collector_bench_sha256[]=\"${bench_identity}\";\ninline constexpr char collector_bench_native_core_sha256[]=\"${native_core_sha256}\";\n")
add_executable(pond_collector_bench bench.cpp "${worktree}/tools/pond-ppo/admission.cpp"
    "${worktree}/src/core/checkpoint.cpp" "${worktree}/src/nn/motor/saved_motor_contract.cpp")
target_include_directories(pond_collector_bench PRIVATE "${worktree}/src" "${worktree}/src/nn"
    "${worktree}/tools/pond-ppo" "${trainer_identity_dir}" "${CMAKE_BINARY_DIR}")
target_compile_options(pond_collector_bench PRIVATE /W4 /WX /permissive-)
set_source_files_properties(bench.cpp PROPERTIES COMPILE_DEFINITIONS "NOMINMAX;WIN32_LEAN_AND_MEAN")
target_link_libraries(pond_collector_bench PRIVATE "${native_core}" CUDA::cuda_driver CUDA::cudart_static bcrypt)
