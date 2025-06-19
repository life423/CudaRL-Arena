#pragma once
#include <cuda_runtime.h>

// Step kernel declaration
__global__ void step_kernel(
    const float* obs,
    float* next_obs, 
    float* rewards,
    unsigned char* done,
    int num_envs,
    const int* actions
);