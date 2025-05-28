cmake_minimum_required(VERSION 3.24)
project(CudaRL-Arena VERSION 0.1.0 LANGUAGES CXX CUDA)

# ──────────────── Global Settings ────────────────
set(CMAKE_CXX_STANDARD        17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD       17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ──────────────── Windows SDK CUDA 12.9 Compatibility ────────────────
# Detect and apply fixes for Windows SDK + CUDA 12.9 conflicts
if(WIN32 AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.9")
    message(STATUS "Applying Windows SDK CUDA 12.9 compatibility fixes")
    
    # Add Windows SDK conflict prevention definitions
    add_definitions(-DWIN32_LEAN_AND_MEAN -DNOMINMAX)
    
    # Auto-include compatibility header for all CUDA files
    set(CUDA_WINDOWS_COMPAT_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/cuda_windows_fix.h")
    if(EXISTS ${CUDA_WINDOWS_COMPAT_HEADER})
        message(STATUS "Using Windows CUDA compatibility header: ${CUDA_WINDOWS_COMPAT_HEADER}")
    endif()
endif()

# ──────────────── Dependencies ────────────────
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python3 ${Python3_VERSION} at ${Python3_EXECUTABLE}")

# Find pybind11 - simplified approach
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE PYBIND11_RESULT
)

if(NOT PYBIND11_RESULT EQUAL 0)
    message(STATUS "pybind11 not found via Python, installing...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pip install pybind11
        RESULT_VARIABLE PIP_RESULT
    )
    if(NOT PIP_RESULT EQUAL 0)
        message(WARNING "Failed to install pybind11 via pip. Using FetchContent instead.")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    else()
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    endif()
else()
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
endif()

find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    message(STATUS "Using mock environment instead of building Python bindings")
    add_custom_target(mock_env
        COMMAND ${CMAKE_COMMAND} -E echo "Using mock environment"
    )
else()
    message(STATUS "Found pybind11 v${pybind11_VERSION}")
endif()

# ──────────────── CUDA Settings ────────────────
# Enhanced architecture support including RTX 5070 (Ada Lovelace SM 8.9)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Runtime library configuration
if(WIN32)
    # Use static runtime for Windows to avoid DLL conflicts
    set(CMAKE_CUDA_RUNTIME_LIBRARY MultiThreaded)
    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
else()
    set(CMAKE_CUDA_RUNTIME_LIBRARY MultiThreadedDebugDLL)
endif()

# Enhanced CUDA compilation flags for performance and compatibility
set(CUDA_COMPILE_FLAGS "--use_fast_math --expt-relaxed-constexpr")

# Windows-specific CUDA flags
if(WIN32)
    set(CUDA_COMPILE_FLAGS "${CUDA_COMPILE_FLAGS} -Xcompiler /MT")
    set(CUDA_COMPILE_FLAGS "${CUDA_COMPILE_FLAGS} -Wno-deprecated-gpu-targets")
endif()

# Set base CUDA flags (without maxrregcount to avoid conflicts)
set(CMAKE_CUDA_FLAGS "${CUDA_COMPILE_FLAGS}")
message(STATUS "Base CUDA flags: ${CMAKE_CUDA_FLAGS}")

# ──────────────── Core Library ────────────────
# Create core library with CUDA functionality - now modular structure
add_library(cudarl_core STATIC
  # Core logic (environment-agnostic)
  src/core/environment.h
  src/core/environment_cpu.cpp
  src/core/cuda_utils.h
  
  # GPU kernels and CUDA implementation
  src/gpu/environment.cu
  src/gpu/kernels.cu
  src/gpu/kernels.cuh
)

target_include_directories(cudarl_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu
)

target_link_libraries(cudarl_core PUBLIC
  CUDA::cudart
  CUDA::curand
)

# Apply Windows SDK compatibility fixes
if(WIN32 AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.9")
    if(EXISTS ${CUDA_WINDOWS_COMPAT_HEADER})
        target_compile_options(cudarl_core PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-include ${CUDA_WINDOWS_COMPAT_HEADER}>
        )
        message(STATUS "Auto-including Windows compatibility header for CUDA files")
    endif()
endif()

# RTX 5070 specific optimizations - set at target level to override CMake defaults
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
    target_compile_options(cudarl_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=64>
    )
    message(STATUS "RTX 5070 optimizations enabled (maxrregcount=64)")
endif()

# Standard CUDA compile options
target_compile_options(cudarl_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

# Windows-specific compile definitions
if(WIN32)
    target_compile_definitions(cudarl_core PRIVATE 
        WIN32_LEAN_AND_MEAN 
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
    )
endif()

# ──────────────── Python Bindings ────────────────
if(pybind11_FOUND)
    pybind11_add_module(cudarl_core_python src/python_bindings.cpp)
    target_link_libraries(cudarl_core_python PRIVATE cudarl_core)
    
    # Force static linking on Windows
    if(WIN32)
        set_property(TARGET cudarl_core_python PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
        target_compile_definitions(cudarl_core_python PRIVATE 
            WIN32_LEAN_AND_MEAN 
            NOMINMAX
        )
        # Link CUDA libraries statically
        target_link_libraries(cudarl_core_python PRIVATE 
            CUDA::cudart_static
        )
    endif()
endif()

# ──────────────── Main Application ────────────────
add_executable(cudarl_app src/core/main.cu)
target_link_libraries(cudarl_app PRIVATE cudarl_core)

# ──────────────── CUDA Test ────────────────
add_executable(cuda_test src/gpu/cuda_test.cu)
target_link_libraries(cuda_test PRIVATE cudarl_core)

# ──────────────── Enhanced Test Targets ────────────────
add_custom_target(run
  COMMAND $<TARGET_FILE:cudarl_app>
  DEPENDS cudarl_app
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Building and running cudarl_app"
)

add_custom_target(run_cuda_test
  COMMAND $<TARGET_FILE:cuda_test>
  DEPENDS cuda_test
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Building and running CUDA functionality test"
)

# New verification target for Windows SDK compatibility
add_custom_target(verify_cuda_compat
  COMMAND ${CMAKE_COMMAND} -E echo "=== CUDA Compatibility Verification ==="
  COMMAND ${CMAKE_COMMAND} -E echo "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}"
  COMMAND ${CMAKE_COMMAND} -E echo "Architectures: ${CMAKE_CUDA_ARCHITECTURES}"
  COMMAND ${CMAKE_COMMAND} -E echo "Platform: ${CMAKE_SYSTEM_NAME}"
  COMMAND ${CMAKE_COMMAND} -E echo "Windows SDK Fix: $<IF:$<BOOL:${WIN32}>,Enabled,Not Needed>"
  COMMAND $<TARGET_FILE:cuda_test>
  DEPENDS cuda_test
  COMMENT "Verifying CUDA compatibility and Windows SDK fixes"
)

# ──────────────── Godot GDExtension ────────────────
option(BUILD_GODOT_EXTENSION "Build Godot GDExtension" ON)

if(BUILD_GODOT_EXTENSION)
    message(STATUS "Building Godot GDExtension...")
    
    # Fetch godot-cpp if not already available
    include(FetchContent)
    FetchContent_Declare(
        godot-cpp
        GIT_REPOSITORY https://github.com/godotengine/godot-cpp.git
        GIT_TAG 4.2
    )
    FetchContent_MakeAvailable(godot-cpp)
      # Create GDExtension library
    add_library(cudarl_godot SHARED
        src/bindings/godot/godot_bindings.cpp
    )
    
    target_link_libraries(cudarl_godot PRIVATE
        cudarl_core
        godot-cpp
    )
    
    # Windows-specific configuration
    if(WIN32)
        set_target_properties(cudarl_godot PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/godot/bin/windows"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/godot/bin/windows"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/godot/bin/windows"
            OUTPUT_NAME_DEBUG "cudarl.debug"
            OUTPUT_NAME_RELEASE "cudarl"
        )
        
        # Static linking for Windows
        set_property(TARGET cudarl_godot PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
        target_compile_definitions(cudarl_godot PRIVATE 
            WIN32_LEAN_AND_MEAN 
            NOMINMAX
        )
    else()
        set_target_properties(cudarl_godot PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/godot/bin/linux"
            OUTPUT_NAME "cudarl"
        )
    endif()
    
    message(STATUS "Godot GDExtension will be built")
endif()

# ──────────────── Build Summary ────────────────
message(STATUS "")
message(STATUS "═══════════ CudaRL-Arena Build Configuration ═══════════")
message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Python Version: ${Python3_VERSION}")
message(STATUS "pybind11 Found: ${pybind11_FOUND}")
if(WIN32 AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.9")
    message(STATUS "Windows SDK CUDA 12.9 Fix: ENABLED")
else()
    message(STATUS "Windows SDK CUDA 12.9 Fix: Not needed")
endif()
message(STATUS "═══════════════════════════════════════════════════════")
message(STATUS "")
