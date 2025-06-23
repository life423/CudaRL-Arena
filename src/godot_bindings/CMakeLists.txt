cmake_minimum_required(VERSION 3.18)
project(CudaRLGodotBindings LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Path to godot-cpp
set(GODOT_CPP_PATH "${CMAKE_SOURCE_DIR}/../../external/godot-cpp")

# Include directories
include_directories(
    ${GODOT_CPP_PATH}/include
    ${GODOT_CPP_PATH}/gen/include
    ${GODOT_CPP_PATH}/gdextension
    ${CMAKE_SOURCE_DIR}/../core
    ${CMAKE_SOURCE_DIR}/../gpu
)

# Add the library
add_library(cudarl_godot SHARED
    arena_gdextension.cpp
    register_types.cpp
)

# Link against CUDA and godot-cpp
target_link_libraries(cudarl_godot PRIVATE 
    CUDA::cudart
    ${GODOT_CPP_PATH}/bin/libgodot-cpp.windows.release.x86_64.lib
)

# Set output properties
set_target_properties(cudarl_godot PROPERTIES
    PREFIX ""
    SUFFIX ".dll"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../godot-project/bin
)

# For Windows, we need to specify the runtime library
if(WIN32)
    set_target_properties(cudarl_godot PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    )
endif()
