cmake_minimum_required(VERSION 3.20)
project(cuda_project LANGUAGES CXX CUDA)

# ——————————————
# 1) C++ & CUDA settings
# ——————————————
set(CMAKE_CXX_STANDARD        17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD       17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Pull in the CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Only emit code for SM ≥7.5
# (tweak these numbers to match your GPU architectures)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Force the CUDA runtime to use the same DLL CRT that MSVC uses in Debug
set(CMAKE_CUDA_RUNTIME_LIBRARY MultiThreadedDebugDLL)

# ——————————————
# 2) Python integration
# ——————————————
# Use CMake’s built-in Python support to find headers, libs & interpreter
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python3 ${Python3_VERSION} at ${Python3_EXECUTABLE}")

# If you intend to embed the Python interpreter in your app:
#   link against Python3::Python
# If you’re building a CPython extension, you might instead use
#   add_library(myext MODULE …)
#   target_link_libraries(myext PRIVATE Python3::Python)

# ——————————————
# 3) Your PoC executable
# ——————————————
add_executable(poc_app src/main.cu)

# suppress “offline compilation for <75 will be removed” warning
target_compile_options(poc_app
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

# link in CUDA’s runtime
target_link_libraries(poc_app
  PRIVATE
    CUDA::cudart
    Python3::Python            # <-- embed the interpreter
)

# if you need Python headers for host‐side code, add:
target_include_directories(poc_app
  PRIVATE
    ${Python3_INCLUDE_DIRS}
)

# ────────────────────
# 4) Handy “run” target
# ────────────────────
add_custom_target(run
  COMMAND $<TARGET_FILE:poc_app>
  DEPENDS poc_app
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Building and running poc_app"
)

# now you can do:
#   cmake --build build --target run --config Debug
