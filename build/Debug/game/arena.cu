#include "arena.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void step_kernel(float* env, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int idx = y * width + x;
        env[idx] += 1.0f; // placeholder computation
    }
}

void step(float* env, int width, int height)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    step_kernel<<<grid, block>>>(env, width, height);
    cudaDeviceSynchronize();
}

