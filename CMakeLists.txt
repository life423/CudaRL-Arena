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
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
set(CMAKE_CUDA_RUNTIME_LIBRARY MultiThreadedDebugDLL)

# CUDA compilation flags
set(CUDA_COMPILE_FLAGS "--use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_COMPILE_FLAGS}")

# ──────────────── Core Library ────────────────
# Create core library with CUDA functionality
add_library(cudarl_core STATIC
  src/environment.h
  src/environment.cu
  src/environment_bridge.h
  src/environment_bridge.cu
  src/kernels.cuh
  src/kernels.cu
  src/q_learning.cuh
  src/q_learning.cu
)

target_include_directories(cudarl_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cudarl_core PUBLIC
  CUDA::cudart
)

target_compile_options(cudarl_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

# ──────────────── Python Bindings ────────────────
if(pybind11_FOUND)
    pybind11_add_module(cudarl_core_python src/python_bindings.cpp)
    set_target_properties(cudarl_core_python PROPERTIES OUTPUT_NAME cudarl_core)
    target_link_libraries(cudarl_core_python PRIVATE cudarl_core)
endif()

# ──────────────── Main Application ────────────────
add_executable(cudarl_app src/main.cu)
target_link_libraries(cudarl_app PRIVATE cudarl_core)

# ──────────────── CUDA Test ────────────────
add_executable(cuda_test src/cuda_test.cu)
target_link_libraries(cuda_test PRIVATE cudarl_core)

# ──────────────── Custom Targets ────────────────
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