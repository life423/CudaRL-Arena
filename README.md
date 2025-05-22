# CUDA Gridworld RL Demo

This project is a real-time, CUDA-accelerated 2D gridworld game that demonstrates reinforcement learning (tabular Q-learning) with both human and AI agents. It is designed as an educational and benchmarking platform to showcase the power of GPU acceleration in AI, and to make machine learning concepts tangible and interactive.

## Features

- Minimal 2D gridworld game (C++/SDL2)
- Tabular Q-learning agent (Python)
- CUDA-accelerated AI routines (C++/CUDA)
- pybind11 bindings for Python/C++/CUDA integration
- Real-time visualization and human-AI interaction
- Modular, extensible architecture

## Build System

- CMake-based build with targets for:
  - C++ game engine (SDL2)
  - CUDA-accelerated AI library
  - Python bindings (pybind11)

## Directory Structure

- `src/game/` — C++ game engine and SDL2 rendering
- `src/ai/` — CUDA-accelerated AI routines
- `src/bindings/` — pybind11 bindings
- `python/` — Python RL agent, training scripts, visualization
- `tests/` — Unit and integration tests
- `docs/` — Documentation and educational materials

## Getting Started

1. Install SDL2, CUDA Toolkit, and pybind11.
2. Clone this repository.
3. Build with CMake:
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```
4. Run the game or Python demos as described in the docs.

## License

MIT License
