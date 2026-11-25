cmake_minimum_required(VERSION 3.24)
project(PondPpoPreparation LANGUAGES CXX CUDA)
get_filename_component(worktree "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
find_package(CUDAToolkit REQUIRED)

function(pond_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CXX>:/WX>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>
        $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
endfunction()

# Optimized learning kernels are separate from the isolated physics profile.
add_library(pond_ppo_gpu_core STATIC
    "${worktree}/src/nn/gemm.cu" "${worktree}/src/nn/mlp_gpu.cu"
    "${worktree}/src/nn/adam.cu" "${worktree}/src/nn/loss.cu"
    "${worktree}/src/rl/ppo_kernels.cu" "${worktree}/src/rl/gaussian_kernels.cu")
target_include_directories(pond_ppo_gpu_core PUBLIC "${worktree}/src" "${worktree}/src/nn")
target_link_libraries(pond_ppo_gpu_core PUBLIC CUDA::cudart_static)
pond_warnings(pond_ppo_gpu_core)

add_library(pond_ppo_device_geometry STATIC
    "${worktree}/src/cuda/pond_geometry_inspect.cu"
    "${worktree}/src/cuda/pond_geometry_inspect_link.cu")
target_include_directories(pond_ppo_device_geometry PRIVATE "${worktree}/src" "${worktree}/src/nn")
target_compile_definitions(pond_ppo_device_geometry PRIVATE CUDARL_POND_GEOMETRY_RDC=1)
target_compile_options(pond_ppo_device_geometry PRIVATE --fmad=false)
set_target_properties(pond_ppo_device_geometry PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
pond_warnings(pond_ppo_device_geometry)

function(pond_link_shared_inspection target)
    target_compile_definitions(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:CUDARL_POND_GEOMETRY_RDC=1>)
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${target} PRIVATE pond_ppo_device_geometry)
endfunction()

set(identity "CUDA=${CMAKE_CUDA_COMPILER_VERSION};CXX=${CMAKE_CXX_COMPILER_VERSION};sm120;physics-fmad-false;physics-optimized;learning-default;CUDA_FLAGS=${CMAKE_CUDA_FLAGS};CUDA_RELEASE=${CMAKE_CUDA_FLAGS_RELEASE};CXX_FLAGS=${CMAKE_CXX_FLAGS};CXX_RELEASE=${CMAKE_CXX_FLAGS_RELEASE};")
set(physics_identity "CUDA=${CMAKE_CUDA_COMPILER_VERSION};CXX=${CMAKE_CXX_COMPILER_VERSION};sm120;physics-fmad-false;physics-optimized;shared-device-inspection;")
set(source_manifest "")
foreach(relative IN ITEMS
        tools/pond-ppo/train.cu tools/pond-ppo/correlated_exploration.h tools/pond-ppo/turn_course_observation.h
        tools/pond-ppo/transition_trace.h tools/pond-ppo/CMakeLists.txt tools/pond-ppo/identity.h.in
        tools/pond-ppo/admission.h tools/pond-ppo/admission.cpp
        src/cuda/pond_math_profile.h src/cuda/pond_geometry_inspect.cu src/cuda/pond_geometry_inspect_link.cu src/environment/pond_geometry_v1.h
        src/environment/pond_body_geometry.h src/environment/pond_swept_clearance.h src/simulation/pond_runtime.h
        src/nn/motor/pond_motor_observation.h src/nn/motor/saved_motor_contract.cpp
        src/nn/motor/saved_motor_contract.h src/nn/motor/movement_model.h
        src/nn/motor/joint_motor.h src/nn/motor/state.h src/physics/dynamics.h src/physics/contact.h
        src/simulation/world.h src/rl/pond_ppo_contract.h src/rl/pond_ppo_task.h src/rl/pond_ppo_schedule.h
        src/rl/gaussian_math.h src/rl/ar_gaussian_math.h src/rl/gaussian_kernels.cu
        src/rl/gaussian_kernels.h src/rl/ppo_math.h src/rl/ppo_kernels.h src/rl/learned_amplitude_math.h
        src/rl/ppo_kernels.cu src/nn/mlp_gpu.cu src/nn/mlp_gpu.h src/nn/mlp_spec.h
        src/nn/gemm.cu src/nn/gemm.h src/nn/adam.cu src/nn/adam.h src/nn/loss.cu src/nn/loss.h
        src/nn/adam_config.h src/core/checkpoint.h src/core/checkpoint.cpp src/core/hd.h src/core/rng.h
        src/cuda/cuda_memory.h src/cuda/block_reduce.h)
    file(SHA256 "${worktree}/${relative}" digest)
    string(APPEND identity "${relative}:${digest};")
    string(APPEND source_manifest "${relative} ${digest}\n")
    if(relative MATCHES "^src/(physics/|environment/|simulation/|cuda/pond_(math_profile|geometry_inspect)|nn/motor/(movement_model|joint_motor|state)|core/hd)")
        string(APPEND physics_identity "${relative}:${digest};")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${worktree}/${relative}")
endforeach()
string(SHA256 source_identity "${identity}")
string(SHA256 physics_source_identity "${physics_identity}")
file(WRITE "${CMAKE_BINARY_DIR}/pond-ppo-source-manifest.txt" "${source_manifest}")
configure_file(identity.h.in "${CMAKE_BINARY_DIR}/pond_ppo_identity.h" @ONLY)

add_executable(pond_ppo train.cu admission.cpp "${worktree}/src/core/checkpoint.cpp"
    "${worktree}/src/nn/motor/saved_motor_contract.cpp")
option(POND_BUILD_COURSE_CONTEXT "Keep the earlier trial executable alongside the course-context build" OFF)
if(POND_BUILD_COURSE_CONTEXT)
    set_target_properties(pond_ppo PROPERTIES OUTPUT_NAME pond_ppo_course_context)
endif()
target_include_directories(pond_ppo PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(pond_ppo PRIVATE pond_ppo_gpu_core bcrypt)
set_source_files_properties(train.cu PROPERTIES COMPILE_DEFINITIONS "NOMINMAX;WIN32_LEAN_AND_MEAN")
target_compile_options(pond_ppo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
pond_link_shared_inspection(pond_ppo)
pond_warnings(pond_ppo)

add_executable(pond_ppo_pure pure_check.cpp admission.cpp "${worktree}/src/core/checkpoint.cpp")
target_include_directories(pond_ppo_pure PRIVATE "${worktree}/src" "${CMAKE_BINARY_DIR}")
target_link_libraries(pond_ppo_pure PRIVATE bcrypt)
pond_warnings(pond_ppo_pure)

add_executable(pond_ppo_sampler_pure sampler_check.cpp)
target_include_directories(pond_ppo_sampler_pure PRIVATE "${worktree}/src" "${worktree}/src/nn")
pond_warnings(pond_ppo_sampler_pure)
