cmake_minimum_required(VERSION 3.21)

# Default to the RTX 50-series (Blackwell, sm_120). Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

project(CudaRL-Arena LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Static CRT and static CUDA runtime, to match the prebuilt godot-cpp library and
# so the extension DLL ships without any CUDA DLLs next to it.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

find_package(CUDAToolkit REQUIRED)

# --- godot-cpp (prebuilt; see README) ---------------------------------------
set(GODOT_CPP_DIR "${CMAKE_SOURCE_DIR}/external/godot-cpp")
set(GODOT_CPP_LIB "${GODOT_CPP_DIR}/bin/libgodot-cpp.windows.template_release.x86_64.lib")
if(NOT EXISTS "${GODOT_CPP_LIB}")
    message(FATAL_ERROR "godot-cpp library not found: ${GODOT_CPP_LIB}")
endif()
add_library(godot-cpp STATIC IMPORTED)
set_target_properties(godot-cpp PROPERTIES
    IMPORTED_LOCATION "${GODOT_CPP_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${GODOT_CPP_DIR}/include;${GODOT_CPP_DIR}/gen/include;${GODOT_CPP_DIR}/gdextension"
)

# --- Core: CUDA code and environments, no Godot dependency --------------------
add_library(cudarl_core STATIC
    src/cuda/device.cu
    src/envs/maze/maze_env.cpp
)
target_include_directories(cudarl_core PUBLIC src)
target_link_libraries(cudarl_core PUBLIC CUDA::cudart_static)

# --- Godot extension ------------------------------------------------------------
add_library(cudarl SHARED
    src/godot/register_types.cpp
    src/godot/cuda_device.cpp
    src/godot/maze_arena.cpp
)
target_link_libraries(cudarl PRIVATE cudarl_core godot-cpp)
target_compile_definitions(cudarl PRIVATE TYPED_METHOD_BIND WINDOWS_ENABLED)

set(CUDARL_ADDON_BIN "${CMAKE_SOURCE_DIR}/godot-project/addons/cudarl/bin")
set_target_properties(cudarl PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY         "${CUDARL_ADDON_BIN}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CUDARL_ADDON_BIN}"
)
