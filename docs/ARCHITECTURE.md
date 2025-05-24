# CudaRL-Arena Architecture

This document describes the architecture and design decisions of the CudaRL-Arena project.

## Core Components

### Environment

The `Environment` class is the central component of the system. It manages:

- Grid-based state representation
- Agent position and movement
- Reward calculation
- CUDA-accelerated state updates

The environment uses a hybrid CPU/GPU approach:
- State is maintained on both host and device
- State transitions are computed on the GPU
- Results are synchronized back to the host for API access

### CUDA Memory Management

Memory management follows RAII principles using the `CudaMemory` template class:

- Automatic allocation and deallocation of GPU memory
- Move semantics for ownership transfer
- Error checking for all CUDA operations
- Prevention of memory leaks

### Python Integration

Python bindings are provided through pybind11, exposing:

- Environment creation and manipulation
- State observation and action execution
- Grid data access as NumPy arrays

The Python layer adds:
- Gym-like interface for compatibility with RL frameworks
- Agent implementations (Random, Q-Table)
- Training and evaluation utilities
- Visualization tools

### Godot Integration

Godot integration is provided through GDExtension, allowing:

- Real-time visualization of the environment
- Interactive agent control
- Signal-based communication for events
- Custom rendering of the grid state

## Data Flow

1. **Initialization**:
   - Environment is created with specified dimensions
   - Grid is initialized with random values
   - Agent is placed at the center
   - State is copied to GPU memory

2. **Step Execution**:
   - Action is received from agent
   - CUDA kernel processes the action and updates state
   - Updated state is synchronized back to host
   - Observation, reward, and done flag are returned

3. **Training Loop**:
   - Agent selects action based on observation
   - Environment executes action
   - Agent updates policy based on reward
   - Process repeats until episode completion

## Design Decisions

### CUDA Acceleration

- Single-threaded kernels for simple grid environments
- Potential for parallel processing in more complex environments
- Memory transfers minimized to reduce overhead
- Error checking for all CUDA operations

### C++ Modernization

- RAII for resource management
- Smart pointers for ownership semantics
- Move semantics for efficient transfers
- Strong type safety with `enum class`
- Exception-based error handling

### Python API Design

- Gym-compatible interface for easy integration
- NumPy arrays for efficient data transfer
- Type hints for better IDE support
- Logging for debugging and monitoring
- Command-line interface for training scripts

### Godot Integration

- GDExtension for high-performance integration
- Signal-based communication for event handling
- Automatic build process with CMake
- Clean separation between simulation and visualization

## Future Extensions

- Multi-agent support
- Complex environment dynamics
- Neural network agents with PyTorch/TensorFlow
- Distributed training
- Benchmark suite for performance testing