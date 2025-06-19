#include "step_kernel.cuh"

// Step kernel implementation
__global__ void step_kernel(
    const float* obs,
    float* next_obs, 
    float* rewards,
    unsigned char* done,
    int num_envs,
    const int* actions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_envs) {
        // Simple stub implementation
        // Just copy observation to next observation for now
        next_obs[idx] = obs[idx];
        rewards[idx] = 0.1f;  // Small reward
        done[idx] = 0;        // Not done
    }
}