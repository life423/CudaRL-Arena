# CudaRL-Arena

A CUDA-accelerated reinforcement learning environment for training and benchmarking RL agents.

## Requirements
- CUDA Toolkit 11.x or higher
- C++17 compatible compiler
- Python 3.8+
- CMake 3.18+

## Building
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Running
```bash
# Run C++ proof of concept
./bin/Debug/cudarl_app.exe

# Run Python training script
python python/scripts/train.py --agent qtable --episodes 100
```

## Architecture
- C++/CUDA: Implements environment dynamics and Q-learning kernel
- Python: Orchestrates training, visualization, and experiment management