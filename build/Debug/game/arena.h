#pragma once

/// Step the 2D environment using CUDA.
/// \param env Pointer to environment data stored in row-major order.
/// \param width Width of the environment grid.
/// \param height Height of the environment grid.
void step(float* env, int width, int height);

