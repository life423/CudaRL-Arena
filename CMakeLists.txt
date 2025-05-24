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

# ──────────────── CUDA Settings ────────────────
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
set(CMAKE_CUDA_RUNTIME_LIBRARY MultiThreadedDebugDLL)

# CUDA compilation flags
set(CUDA_COMPILE_FLAGS "--use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_COMPILE_FLAGS}")

# ──────────────── pybind11 ────────────────
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# ──────────────── Core Library ────────────────
# Create core library with CUDA functionality
add_library(cudarl_core STATIC
  src/core/cuda_utils.h
  src/core/environment.h
  src/gpu/environment.cu
  src/gpu/kernels.cuh
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

# ──────────────── Main Application ────────────────
add_executable(cudarl_app src/core/main.cu)
target_link_libraries(cudarl_app PRIVATE cudarl_core)

# ──────────────── CUDA Test ────────────────
add_executable(cuda_test src/gpu/cuda_test.cu)
target_link_libraries(cuda_test PRIVATE cudarl_core)

# ──────────────── Python Bindings ────────────────
pybind11_add_module(cudarl_python_bindings src/bindings/python_bindings.cpp)
target_link_libraries(cudarl_python_bindings PRIVATE cudarl_core)
set_target_properties(cudarl_python_bindings PROPERTIES
  OUTPUT_NAME "cudarl_core"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/cudarl"
)

# ──────────────── Godot GDExtension plugin ────────────────
# Set the Godot executable path - adjust as needed
set(GODOT_EXECUTABLE "C:/Users/aiand/Godot/Godot.exe")

# 1. Fetch the Godot Engine source for API headers
FetchContent_Declare(
  godot_engine
  GIT_REPOSITORY https://github.com/godotengine/godot.git
  GIT_TAG        4.4.1-stable
)
# Use the old method to avoid the warning for now
set(CMAKE_POLICY_DEFAULT_CMP0169 OLD)
FetchContent_GetProperties(godot_engine)
if(NOT godot_engine_POPULATED)
  FetchContent_Populate(godot_engine)
  set(GODOT_ENGINE_SOURCE_DIR ${godot_engine_SOURCE_DIR})
endif()

# 2. Fetch and build godot-cpp
FetchContent_Declare(
  godot_cpp
  GIT_REPOSITORY https://github.com/godotengine/godot-cpp.git
  GIT_TAG        godot-4.4-stable
)
FetchContent_MakeAvailable(godot_cpp)
set(GODOT_CPP_SOURCE_DIR  ${godot_cpp_SOURCE_DIR})
set(GODOT_CPP_BINARY_DIR  ${godot_cpp_BINARY_DIR})

# 3. Copy godot_engine headers to expected location for godot-cpp
set(GODOT_CPP_EXPECTED_ENGINE_DIR ${GODOT_CPP_SOURCE_DIR}/../godot)
add_custom_command(
  OUTPUT ${GODOT_CPP_EXPECTED_ENGINE_DIR}/modules/gdextension/include/gdextension_interface.h
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GODOT_CPP_EXPECTED_ENGINE_DIR}/modules/gdextension/include
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${GODOT_ENGINE_SOURCE_DIR}/core/extension ${GODOT_CPP_EXPECTED_ENGINE_DIR}/modules/gdextension/include
  COMMENT "Copying Godot engine GDExtension headers for godot-cpp SCons build"
)
add_custom_target(copy_godot_headers ALL
  DEPENDS ${GODOT_CPP_EXPECTED_ENGINE_DIR}/modules/gdextension/include/gdextension_interface.h
)

# 4. Auto-generate extension_api.json
set(EXTENSION_API_FILE ${CMAKE_BINARY_DIR}/extension_api.json)
add_custom_command(
  OUTPUT ${EXTENSION_API_FILE}
  COMMAND ${GODOT_EXECUTABLE} --dump-extension-api --headless
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Generating GDExtension API from Godot"
)
add_custom_target(generate_extension_api ALL DEPENDS ${EXTENSION_API_FILE})

# 5. Build the GDExtension plugin
add_library(gd_cudarl MODULE src/bindings/godot_bindings.cpp)
add_dependencies(gd_cudarl generate_extension_api copy_godot_headers)

target_include_directories(gd_cudarl PRIVATE
  ${GODOT_CPP_SOURCE_DIR}/include
  ${GODOT_CPP_BINARY_DIR}/gen/include
  ${GODOT_CPP_EXPECTED_ENGINE_DIR}/modules/gdextension/include
  ${GODOT_ENGINE_SOURCE_DIR}/core
  ${GODOT_ENGINE_SOURCE_DIR}/core/math
  ${GODOT_ENGINE_SOURCE_DIR}/core/extension
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(gd_cudarl PRIVATE
  cudarl_core
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(GODOT_CPP_LIB_NAME "libgodot-cpp.windows.template_debug.x86_64.lib")
else()
  set(GODOT_CPP_LIB_NAME "libgodot-cpp.windows.template_release.x86_64.lib")
endif()

target_link_libraries(gd_cudarl PRIVATE
  "${GODOT_CPP_BINARY_DIR}/bin/${GODOT_CPP_LIB_NAME}"
)

set_target_properties(gd_cudarl PROPERTIES
  PREFIX ""
  SUFFIX ".gdextension"
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD            17
  CXX_STANDARD_REQUIRED   ON
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/godot/bin"
)

# Generate the .gdextension config file
configure_file(
  ${CMAKE_SOURCE_DIR}/godot/cudarl.gdextension.in
  ${CMAKE_SOURCE_DIR}/godot/cudarl.gdextension
  @ONLY
)

# ──────────────── Tests ────────────────
add_subdirectory(tests)

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

# ──────────────── Installation ────────────────
install(TARGETS cudarl_app cuda_test
  RUNTIME DESTINATION bin
)

install(TARGETS cudarl_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

install(FILES 
  src/core/environment.h
  src/core/cuda_utils.h
  DESTINATION include/cudarl
)

# Install Python module
install(TARGETS cudarl_python_bindings
  LIBRARY DESTINATION python/cudarl
)

# Install Godot extension
install(TARGETS gd_cudarl
  LIBRARY DESTINATION godot/bin
)

install(FILES
  ${CMAKE_SOURCE_DIR}/godot/cudarl.gdextension
  DESTINATION godot
)