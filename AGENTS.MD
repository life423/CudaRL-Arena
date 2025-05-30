# AGENTS.md - CudaRL-Arena Project Configuration for Codex

## Project Overview
CudaRL-Arena is a high-speed reinforcement learning simulator that leverages CUDA for massively parallelized training environments. The goal is to train complex agents (game AI, autonomous navigation) at ultra-high speeds by simulating multiple agents simultaneously in the same virtual space.

## Architecture & Key Components

### Core Technologies
- **C++/CUDA**: Environment simulation and performance-critical operations
- **Python**: Agent implementation and training orchestration  
- **Godot**: Interactive visualization frontend
- **CMake**: Build system

### Project Structure
```
CudaRL-Arena/
├── src/                    # Core C++ and CUDA source files
│   ├── environment.h/cu    # Environment implementation
│   ├── environment_bridge.h/cu  # C++/Python bridge
│   ├── kernels.cuh/cu      # CUDA kernels
│   ├── q_learning.cuh/cu   # Q-learning implementation
│   └── main.cu             # C++ entry point
├── python/                 # Python package and scripts
│   ├── cudarl/            # Python modules
│   └── scripts/           # Training scripts
├── godot/                 # Godot visualization
├── CMakeLists.txt         # Build configuration
└── run_training.py        # Main training script
```

## Current Technical Debt & Refactoring Priorities

### 1. CUDA Macro Updates (HIGH PRIORITY)
- **Issue**: `WITH_CUDA` macro needs to be replaced with `USE_CUDA` 
- **Location**: CMakeLists.txt and throughout C++ source files
- **Context**: Following CMake 3.x best practices for CUDA detection

### 2. API/Endpoint Standardization
- **Issue**: Mixed naming conventions for API endpoints
- **Needed**: Consistent RESTful naming scheme
- **Affected Files**: `environment_bridge.h/cu`, Python API wrappers

### 3. Config Management Refactor
- **Issue**: Configuration scattered across multiple files
- **Goal**: Centralized config management system
- **Approach**: Single source of truth for all hyperparameters and environment settings

### 4. State Synchronization Architecture
- **Challenge**: Optimizing state sync across thousands of simultaneous training instances
- **Current Issues**: Race conditions, memory bottlenecks
- **Goal**: Lock-free synchronization mechanisms

### 5. Parallel Training Infrastructure
- **Requirements**: 
  - Multiple agents in same GPU space
  - Consistent learning outcomes
  - Efficient memory management
  - Scalable to 1000+ agents

## Coding Standards & Conventions

### C++/CUDA Code
- Use modern C++17 features where appropriate
- CUDA kernels should follow naming pattern: `kernel_<functionality>`
- All GPU memory allocations must have corresponding error checks
- Prefer `cudaMallocManaged` for unified memory when possible
- Use `__restrict__` for pointer parameters in kernels

### Python Code
- Follow PEP 8 style guide
- Type hints for all function signatures
- Docstrings in NumPy style
- Use `dataclasses` for configuration objects

### Error Handling
- CUDA: Always check return codes with `CUDA_CHECK` macro
- Python: Use custom exceptions inheriting from base `CudaRLException`
- Provide meaningful error messages with context

### Performance Considerations
- Minimize CPU-GPU transfers
- Use streams for concurrent kernel execution
- Profile with nvprof before optimizing
- Target 90%+ GPU utilization for training loops

## Testing Requirements
- Unit tests for all CUDA kernels using Google Test
- Python tests using pytest
- Integration tests for C++/Python bridge
- Performance benchmarks for each major component

## Build Configuration Notes
- Minimum CUDA version: 11.0
- C++ standard: C++17
- Python version: 3.8+
- CMake version: 3.18+

## Common Pitfalls to Avoid
1. Don't assume memory is contiguous in multi-agent scenarios
2. Always profile before optimizing - bottlenecks aren't always where expected
3. Be careful with warp divergence in agent logic
4. Remember that Python GIL can bottleneck multi-threaded training

## Debugging Tips
- Use `cuda-memcheck` for memory issues
- Enable `CUDA_LAUNCH_BLOCKING=1` for better error localization
- Use Nsight Systems for performance profiling
- Python debugging: Use `faulthandler` for segfaults in C++ extensions

## Future Enhancements Under Consideration
- Multi-GPU support for even larger scale training
- Integration with popular RL frameworks (Stable Baselines3, RLlib)
- Support for continuous action spaces
- Advanced visualization features in Godot frontend

## Code Review Checklist
When reviewing or generating code, ensure:
- [ ] CUDA error checking is comprehensive
- [ ] Memory leaks are prevented (use RAII where possible)
- [ ] Thread safety is maintained for shared resources
- [ ] Documentation is clear and up-to-date
- [ ] Performance impact is considered and measured
- [ ] Python bindings properly handle GIL
- [ ] Tests are included for new functionality