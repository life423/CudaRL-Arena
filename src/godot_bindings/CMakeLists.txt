cmake_minimum_required(VERSION 3.18)
project(CudaRLGodotBindings LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (adjust for your GPU)
# RTX 5070 has compute capability 12.0
set(CMAKE_CUDA_ARCHITECTURES "120")

# Path to godot-cpp
set(GODOT_CPP_PATH "${CMAKE_SOURCE_DIR}/../../external/godot-cpp")

# Include directories
include_directories(
    ${GODOT_CPP_PATH}/include
    ${GODOT_CPP_PATH}/gen/include
    ${GODOT_CPP_PATH}/gdextension
    ${CMAKE_SOURCE_DIR}/../gpu
)

# Create CUDA library first
add_library(cudarl_gpu STATIC
    ../gpu/cuda_arena.cu
)
target_link_libraries(cudarl_gpu PRIVATE CUDA::cudart)
target_include_directories(cudarl_gpu PUBLIC ../gpu)
set_target_properties(cudarl_gpu PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)

# Add the main extension library
add_library(cudarl_godot SHARED
    arena_gdextension.cpp
    register_types.cpp
)

# Link against CUDA library and godot-cpp
target_link_libraries(cudarl_godot PRIVATE 
    cudarl_gpu
    CUDA::cudart_static
    ${GODOT_CPP_PATH}/bin/libgodot-cpp.windows.template_release.x86_64.lib
)

target_compile_definitions(cudarl_godot PRIVATE
    TYPED_METHOD_BIND
    WINDOWS_ENABLED
)

# Match godot-cpp runtime library
set_target_properties(cudarl_godot PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)

# Set output properties
set_target_properties(cudarl_godot PROPERTIES
    PREFIX ""
    SUFFIX ".dll"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../godot-project/addons/cudarl_plugin
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../godot-project/addons/cudarl_plugin
)

# For Windows, we need to specify the runtime library
if(WIN32)
    set_target_properties(cudarl_godot PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    )
endif()
