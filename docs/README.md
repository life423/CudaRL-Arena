# CudaRL-Arena Documentation

Welcome to the CudaRL-Arena documentation. This directory contains detailed information about the project architecture, usage, and development.

## Contents

- [Architecture](ARCHITECTURE.md) - Detailed description of the project architecture
- [API Reference](#api-reference) - API documentation for the Python and C++ interfaces
- [Examples](#examples) - Example usage of the framework
- [Development Guide](#development-guide) - Guide for developers contributing to the project

## API Reference

### Python API

#### Environment

```python
# Create an environment
env = cudarl_core.Environment(width=10, height=10)

# Reset the environment
observation = env.reset()

# Take a step in the environment
observation, reward, done, info = env.step(action)

# Get the agent's position
agent_x, agent_y = env.get_agent_position()
```

#### Q-Learning Agent

```python
# Create a Q-learning agent
agent = QTableAgent(
    width=10,
    height=10,
    learning_rate=0.1,
    discount_factor=0.99,
    exploration_rate=1.0,
    exploration_decay=0.995
)

# Select an action
action = agent.select_action(observation, agent_position)

# Update the agent
metrics = agent.update(
    observation, action, reward, next_observation, done,
    agent_position, next_agent_position
)
```

#### Training

```python
# Train an agent
metrics = train(
    env=env,
    agent=agent,
    num_episodes=1000,
    max_steps=500
)

# Evaluate an agent
eval_metrics = evaluate(env, agent, num_episodes=10)

# Plot results
plot_results(metrics)
```

### C++ API

#### Environment

```cpp
// Create an environment
Environment env(0, 10, 10);

// Reset the environment
env.reset();

// Take a step in the environment
env.step(action);

// Get the agent's position
int x = env.getAgentX();
int y = env.getAgentY();
```

#### Q-Learning

```cpp
// Create a Q-learning instance
CudaQLearning q_learning(10, 10, 4, 0.1f, 0.99f);

// Update Q-values in batch
q_learning.update_batch(states, actions, rewards, next_states, dones);

// Get the best action for a state
int action = q_learning.get_best_action(x, y);
```

## Examples

### Basic Training

```python
import cudarl_core
from train import QTableAgent, train, evaluate, plot_results

# Create environment and agent
env = cudarl_core.Environment(10, 10)
agent = QTableAgent(10, 10)

# Train the agent
metrics = train(env, agent, num_episodes=1000)

# Evaluate the agent
eval_metrics = evaluate(env, agent)

# Plot results
plot_results(metrics)
```

### Custom Environment Setup

```python
import cudarl_core

# Create environment
env = cudarl_core.Environment(10, 10)

# Add obstacles
env.add_obstacle(2, 3)
env.add_obstacle(3, 3)
env.add_obstacle(4, 3)

# Set goal
env.set_goal(9, 0)

# Add traps
env.add_trap(5, 5, -0.5)
env.add_trap(7, 3, -0.8)
```

## Development Guide

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/CudaRL-Arena.git
cd CudaRL-Arena

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
cmake --build . --config Debug
```

### Running Tests

```bash
# Run C++ tests
./bin/Debug/cuda_test.exe

# Run Python tests
python -m unittest discover tests
```

### Adding New Features

1. **New Environment Features**
   - Extend the `Environment` class in `environment.h/cu`
   - Add new CUDA kernels in `kernels.cuh/cu` if needed
   - Update the `EnvironmentBridge` class to expose the new features to Python

2. **New RL Algorithms**
   - Add new algorithm implementations in Python or C++/CUDA
   - For CUDA-accelerated algorithms, follow the pattern in `q_learning.cuh/cu`
   - Expose the new algorithms to Python through bindings if implemented in C++