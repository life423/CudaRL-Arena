# CudaRL-Arena

A CUDA-accelerated reinforcement learning framework for training agents in grid-based environments.

## Overview

CudaRL-Arena leverages GPU acceleration via CUDA to speed up reinforcement learning training. The project uses a hybrid architecture:
- C++/CUDA for environment simulation and performance-critical operations
- Python for agent implementation and training orchestration

## Features

- CUDA-accelerated environment simulation
- Grid-based world with customizable dimensions
- Q-learning implementation with GPU acceleration
- Python interface for easy agent development
- Visualization capabilities

## Quick Start

### Running with Mock Environment

If you want to test the training script without building the C++ code:

```bash
# From the project root directory
python run_training.py --episodes 100 --plot
```

This uses a mock environment that simulates the behavior of the real CUDA environment.

### Running the GUI

To run the interactive visualization:

```bash
# From the project root directory
python run_gui.py
```

This will open a GUI that shows the environment, agent position, Q-values, and training metrics.

### Building the Full Project

To build the complete project with CUDA acceleration:

```bash
# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
cmake --build . --config Debug

# Run the C++ proof-of-concept
./bin/Debug/cudarl_app.exe

# Run the Python training script
cd ..
python src/train.py --episodes 1000
```

## Project Structure

- `src/` - Core C++ and CUDA source files
  - `environment.h/cu` - Environment implementation
  - `environment_bridge.h/cu` - Bridge between C++ and Python
  - `kernels.cuh/cu` - CUDA kernels
  - `q_learning.cuh/cu` - Q-learning implementation
  - `main.cu` - C++ entry point
  - `train.py` - Python training script
  - `gui.py` - Visualization tools
- `python/` - Python package and scripts
  - `cudarl/` - Python module
  - `scripts/` - Training and utility scripts
- `docs/` - Documentation

## Command-line Options

The training script supports several command-line options:

```
--width WIDTH           Environment width (default: 10)
--height HEIGHT         Environment height (default: 10)
--learning-rate RATE    Learning rate (default: 0.1)
--discount-factor GAMMA Discount factor (default: 0.99)
--exploration-rate EPS  Initial exploration rate (default: 1.0)
--exploration-decay DEC Exploration decay rate (default: 0.995)
--episodes NUM          Number of training episodes (default: 1000)
--max-steps STEPS       Maximum steps per episode (default: 500)
--eval-episodes NUM     Number of evaluation episodes (default: 10)
--output-dir DIR        Directory to save results (default: results)
--save-agent            Save the trained agent
--plot                  Plot training results
```

## License

MIT License