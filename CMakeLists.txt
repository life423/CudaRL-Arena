cmake_minimum_required(VERSION 3.20)
project(cuda_project LANGUAGES CXX CUDA)

# ──────────────── C++ & CUDA ────────────────
set(CMAKE_CXX_STANDARD        17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD       17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# only emit for SM ≥7.5, adjust as needed
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# use the same runtime DLL that MSVC uses in Debug
set(CMAKE_CUDA_RUNTIME_LIBRARY MultiThreadedDebugDLL)

# ──────────────── Python ────────────────
# locate the interpreter + headers + libs
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "Found Python3 ${Python3_VERSION} at ${Python3_EXECUTABLE}")

# ──────────────── your executable ────────────────
add_executable(poc_app src/main.cu)

# suppress the “offline compilation for <75 will be removed” warning
target_compile_options(poc_app PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

# link CUDA runtime + Python embedding lib
target_link_libraries(poc_app PRIVATE
  CUDA::cudart
  Python3::Python
)

# make sure Python headers are in your include path
target_include_directories(poc_app PRIVATE
  ${Python3_INCLUDE_DIRS}
)

# ──────────────── handy “run” target ────────────────
add_custom_target(run
  COMMAND $<TARGET_FILE:poc_app>
  DEPENDS poc_app
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Building and running poc_app"
)


