# CUDA-Accelerated Reinforcement Learning GridWorld

This project demonstrates real-time reinforcement learning using CUDA acceleration. It combines C++/CUDA for performance-critical components with Python for high-level logic and visualization.

## Overview

The project creates a 2D grid-based game environment where AI agents learn optimal strategies through reinforcement learning. The key components are:

- **C++ Game Engine**: Implements the GridWorld environment with SDL2 for rendering
- **CUDA AI Library**: Accelerates reinforcement learning algorithms using GPU
- **Python Bindings**: Connects C++/CUDA components to Python using pybind11
- **Python Agent**: Implements Q-learning algorithm using the CUDA-accelerated backend
- **Visualization**: Real-time visualization of the learning process

## Requirements

- CUDA Toolkit (11.0+)
- CMake (3.18+)
- C++ Compiler with C++17 support
- Python 3.6+
- SDL2 library
- pybind11
- NumPy
- PyGame (for visualization)

## Building the Project

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
cmake --build . --config Release
```

## Running the Demo

After building, you can train an agent:

```bash
# Navigate to the Python scripts directory
cd python/scripts

# Train the agent
python train_agent.py --grid-width 10 --grid-height 10 --episodes 1000 --render

# Run the trained agent
python run_demo.py --model q_table.npy --episodes 10
```

## Command-line Options

### Training

- `--grid-width`: Width of the grid (default: 10)
- `--grid-height`: Height of the grid (default: 10)
- `--episodes`: Number of training episodes (default: 1000)
- `--max-steps`: Maximum steps per episode (default: 100)
- `--learning-rate`: Learning rate (alpha) (default: 0.1)
- `--discount-factor`: Discount factor (gamma) (default: 0.99)
- `--exploration-rate`: Initial exploration rate (epsilon) (default: 1.0)
- `--min-exploration-rate`: Minimum exploration rate (default: 0.01)
- `--exploration-decay`: Exploration rate decay (default: 0.995)
- `--render`: Enable visualization during training
- `--render-fps`: FPS for rendering (default: 10)
- `--save-model`: Path to save the trained model (default: q_table.npy)

### Demo

- `--grid-width`: Width of the grid (default: 10)
- `--grid-height`: Height of the grid (default: 10)
- `--model`: Path to the trained model (default: q_table.npy)
- `--episodes`: Number of episodes to run (default: 10)
- `--max-steps`: Maximum steps per episode (default: 100)
- `--fps`: FPS for rendering (default: 5)
- `--human`: Enable human player mode (WASD keys)

## Project Structure

```
cuda_project/
├── CMakeLists.txt          # Main CMake configuration
├── src/
│   ├── ai/                 # CUDA-accelerated AI components
│   │   ├── qlearning_cuda.cu
│   │   └── CMakeLists.txt
│   ├── game/               # C++ game engine
│   │   ├── gridworld.cpp
│   │   ├── gridworld.h
│   │   └── CMakeLists.txt
│   ├── bindings/           # Python bindings
│   │   ├── bindings.cpp
│   │   └── CMakeLists.txt
│   └── utils/              # Utility functions
├── python/
│   ├── agent/              # RL agent implementation
│   │   └── qlearning_agent.py
│   ├── interface/          # Generated Python bindings
│   ├── scripts/            # Training and demo scripts
│   │   ├── train_agent.py
│   │   └── run_demo.py
│   └── visualization/      # Visualization tools
│       └── gridworld_visualizer.py
├── tests/                  # Unit tests
│   ├── cpp/
│   └── python/
└── docs/                   # Documentation
```

## How It Works

1. The C++ GridWorld environment simulates a 2D grid with walls, goals, and traps
2. The CUDA-accelerated Q-learning algorithm processes batches of experiences in parallel
3. Python bindings expose the C++/CUDA functionality to Python
4. The Python agent implements the reinforcement learning logic
5. Visualization shows the learning process in real-time

## Future Extensions

- 3D environments with more complex physics
- Multi-agent systems with cooperation and competition
- Procedurally generated environments
- Additional reinforcement learning algorithms (DQN, PPO, etc.)
- Performance benchmarks comparing CPU vs. GPU implementations