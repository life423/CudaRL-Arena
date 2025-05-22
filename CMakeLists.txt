cmake_minimum_required(VERSION 3.18)
project(cuda_gridworld
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optionally enable CUDA
enable_language(CUDA)

# Find SDL2 for rendering
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# Find pybind11 for Python bindings
find_package(pybind11 REQUIRED)

# Add subdirectories for modular build
add_subdirectory(src/game)
add_subdirectory(src/ai)
add_subdirectory(src/bindings)

# Optionally, add tests and Python scripts later
# add_subdirectory(tests/cpp)
# add_subdirectory(python)

# Notes:
# - src/game: C++ game engine (SDL2)
# - src/ai: CUDA-accelerated AI routines
# - src/bindings: pybind11 bindings exposing C++/CUDA to Python
# - python/: Python RL agent, training scripts, visualization

# To build:
#   mkdir build && cd build
#   cmake ..
#   make

# To run Python bindings, ensure you build the bindings target and set PYTHONPATH accordingly.
