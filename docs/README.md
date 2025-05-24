# CudaRL-Arena

A high-performance reinforcement learning environment using CUDA acceleration.

## Overview

CudaRL-Arena provides a CUDA-accelerated environment for reinforcement learning research and experimentation. It features:

- CUDA-accelerated environment simulation
- Python bindings for easy integration with ML frameworks
- Godot integration for visualization
- Modular architecture for extensibility

## Project Structure

```
CudaRL-Arena/
├── src/                    # C++ and CUDA source code
│   ├── core/               # Core C++ headers and implementation
│   ├── gpu/                # CUDA kernels and GPU-specific code
│   └── bindings/           # Language bindings (Python, Godot)
├── python/                 # Python package
│   ├── cudarl/             # Python module
│   └── scripts/            # Training and utility scripts
├── godot/                  # Godot integration
│   ├── bin/                # Compiled GDExtension binaries
│   ├── gdextension/        # GDExtension source
│   └── Scenes/             # Godot scenes for visualization
├── tests/                  # Unit and integration tests
└── docs/                   # Documentation
```

## Building

### Prerequisites

- CUDA Toolkit 11.0+
- CMake 3.24+
- Python 3.7+
- Godot 4.4+ (optional, for visualization)

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/CudaRL-Arena.git
cd CudaRL-Arena

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
cmake --build .

# Install (optional)
cmake --install .
```

## Python Usage

```python
from cudarl import Environment, Trainer
from cudarl.agent import QTableAgent

# Create environment and agent
env = Environment(width=10, height=10)
agent = QTableAgent(action_space_size=4, observation_shape=(10, 10))

# Create trainer and train
trainer = Trainer(env, agent)
metrics = trainer.train(num_episodes=1000)

# Evaluate
eval_metrics = trainer.evaluate(num_episodes=10, render=True)
```

## Godot Integration

1. Copy the compiled GDExtension from `godot/bin/` to your Godot project
2. Add the `cudarl.gdextension` file to your Godot project
3. Use the `CudaRLEnvironment` node in your scenes

## License

This project is licensed under the MIT License - see the LICENSE file for details.