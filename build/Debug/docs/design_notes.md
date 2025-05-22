# CudaRL-Arena Design Notes

This directory contains documentation and design notes for the project. The
initial layout separates the CUDA environment implementation under `game/` from
the reinforcement learning code in `python/`.

The `game/` directory is intended for low-level C++/CUDA code that exposes a
clean C API. The `python/` directory will then use this API for training
algorithms.
