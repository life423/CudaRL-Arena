#!/bin/bash
set -e

echo "========================================"
echo "    CudaRL-Arena Build Script"
echo "========================================"

# Check prerequisites
if ! command -v cmake &> /dev/null; then
    echo "ERROR: CMake not found in PATH"
    exit 1
fi

if ! command -v godot &> /dev/null; then
    echo "ERROR: Godot not found in PATH"
    exit 1
fi

# Build the GDExtension
echo "Building GDExtension..."
cd src/godot_bindings
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Test the extension
echo "Testing extension..."
cd ../../../godot-project
godot --headless --path . --script res://test_basic.gd

echo "========================================"
echo "    Build completed successfully!"
echo "========================================"