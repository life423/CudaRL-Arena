#pragma once

#include <cuda_runtime.h>
#include "../core/environment.h"

namespace cudarl {

// Enhanced CUDA kernels for batch processing multiple environments
__global__ void updateEnvironmentKernel(
    EnvironmentState* states,    // Array of environment states [batch_size]
    float* grids,               // Flattened grids for all environments [batch_size * width * height]
    int batch_size,             // Number of environments
    int width,                  // Environment width
    int height                  // Environment height
);

__global__ void resetEnvironmentKernel(
    EnvironmentState* states,    // Array of environment states [batch_size]
    float* grids,               // Flattened grids for all environments [batch_size * width * height]
    int batch_size,             // Number of environments
    int width,                  // Environment width
    int height,                 // Environment height
    unsigned long long seed     // Random seed for initialization
);

__global__ void stepEnvironmentKernel(
    EnvironmentState* states,    // Array of environment states [batch_size]
    float* grids,               // Flattened grids for all environments [batch_size * width * height]
    const int* actions,         // Actions for each environment [batch_size]
    int batch_size,             // Number of environments
    int width,                  // Environment width
    int height                  // Environment height
);

// Advanced obstacle and reward computation kernels
__global__ void placeObstaclesKernel(
    float* grids,               // Flattened grids for all environments [batch_size * width * height]
    const EnvironmentState* states, // Environment states for bounds checking
    int batch_size,
    int width,
    int height,
    float obstacle_density,     // Probability of obstacle placement (0.0-1.0)
    unsigned long long seed
);

__global__ void computeAdvancedRewardsKernel(
    EnvironmentState* states,   // Array of environment states [batch_size]
    const float* grids,         // Flattened grids for all environments [batch_size * width * height]
    int batch_size,
    int width,
    int height,
    float step_penalty,         // Penalty per step
    float goal_reward,          // Reward for reaching goal
    float obstacle_penalty,     // Penalty for hitting obstacles
    float trap_penalty          // Penalty for falling into traps
);

// Utility kernels for environment management
__global__ void checkCollisionsKernel(
    EnvironmentState* states,   // Array of environment states [batch_size]
    const float* grids,         // Flattened grids for all environments
    int batch_size,
    int width,
    int height
);

__global__ void updateAgentPositionsKernel(
    EnvironmentState* states,   // Array of environment states [batch_size]
    float* grids,               // Flattened grids for all environments
    const int* actions,         // Actions for each environment [batch_size]
    int batch_size,
    int width,
    int height
);

// Host wrapper functions for easier kernel launches
void launchBatchReset(
    EnvironmentState* d_states,
    float* d_grids,
    int batch_size,
    int width,
    int height,
    unsigned long long seed = 0
);

void launchBatchStep(
    EnvironmentState* d_states,
    float* d_grids,
    const int* d_actions,
    int batch_size,
    int width,
    int height
);

void launchBatchUpdate(
    EnvironmentState* d_states,
    float* d_grids,
    int batch_size,
    int width,
    int height
);

void launchPlaceObstacles(
    float* d_grids,
    const EnvironmentState* d_states,
    int batch_size,
    int width,
    int height,
    float obstacle_density,
    unsigned long long seed = 0
);

} // namespace cudarl
