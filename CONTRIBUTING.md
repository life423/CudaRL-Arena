# Contributing to CudaRL-Arena

Thank you for your interest in contributing to CudaRL-Arena! This document provides guidelines and instructions for contributing to the project.

## Development Setup

### Prerequisites

- CUDA Toolkit 11.0+
- C++17 compatible compiler
- Python 3.8+
- CMake 3.24+

### Building the Project

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

### Running with Mock Environment

If you don't have CUDA installed or want to test the Python code without building the C++ components:

```bash
# From the project root directory
python run_training.py --episodes 100 --plot
```

## Project Structure

- `src/` - Core C++ and CUDA source files
- `python/` - Python package and scripts
- `docs/` - Documentation

## Coding Standards

- C++ code should follow the Google C++ Style Guide
- Python code should follow PEP 8
- Use descriptive variable and function names
- Add comments for complex logic
- Write unit tests for new functionality

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure they pass
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

By contributing to CudaRL-Arena, you agree that your contributions will be licensed under the project's MIT License.