#include "batch_kernels.cuh"
#include <curand_kernel.h>
#include <cstdio>

namespace cudarl {

// ══════════════════════════════════════════════════════════════════════════════
// ENHANCED BATCH PROCESSING KERNELS
// ══════════════════════════════════════════════════════════════════════════════

__global__ void updateEnvironmentKernel(
    EnvironmentState* states,
    float* grids,
    int batch_size,
    int width,
    int height
) {
    int env_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    if (env_idx >= batch_size) return;
    
    EnvironmentState* env = &states[env_idx];
    float* grid = &grids[env_idx * width * height];
    
    // Use threads to process different parts of the environment
    if (thread_idx == 0) {
        // Main thread updates agent marker in grid
        int agent_pos = env->agent_y * width + env->agent_x;
        
        // Clear previous agent position and set new one
        for (int i = 0; i < width * height; i++) {
            if (grid[i] == 0.5f) grid[i] = 0.0f; // Clear old agent position
        }
        grid[agent_pos] = 0.5f; // Set new agent position
    }
    
    __syncthreads();
    
    // Additional threads can handle other environment updates
    if (thread_idx == 1) {
        // Update environment metadata or perform additional checks
        env->grid = grid; // Ensure state points to correct grid
    }
}

__global__ void resetEnvironmentKernel(
    EnvironmentState* states,
    float* grids,
    int batch_size,
    int width,
    int height,
    unsigned long long seed
) {
    int env_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    if (env_idx >= batch_size) return;
    
    EnvironmentState* env = &states[env_idx];
    float* grid = &grids[env_idx * width * height];
    
    // Initialize cuRAND state
    curandState rand_state;
    curand_init(seed + env_idx * 1000 + thread_idx, 0, 0, &rand_state);
    
    if (thread_idx == 0) {
        // Reset environment state
        env->agent_x = width / 2;
        env->agent_y = height / 2;
        env->reward = 0.0f;
        env->done = false;
        env->width = width;
        env->height = height;
        env->grid = grid;
    }
    
    __syncthreads();
    
    // Parallel grid initialization
    int grid_size = width * height;
    int threads_per_block = blockDim.x;
    
    for (int i = thread_idx; i < grid_size; i += threads_per_block) {
        // Initialize with small random values (0.0 to 0.1)
        grid[i] = curand_uniform(&rand_state) * 0.1f;
    }
    
    __syncthreads();
    
    if (thread_idx == 0) {
        // Set goal at top-right corner
        grid[(height - 1) * width + (width - 1)] = 1.0f;
        
        // Set agent position
        grid[env->agent_y * width + env->agent_x] = 0.5f;
    }
}

__global__ void stepEnvironmentKernel(
    EnvironmentState* states,
    float* grids,
    const int* actions,
    int batch_size,
    int width,
    int height
) {
    int env_idx = blockIdx.x;
    
    if (env_idx >= batch_size) return;
    
    EnvironmentState* env = &states[env_idx];
    float* grid = &grids[env_idx * width * height];
    int action = actions[env_idx];
    
    if (threadIdx.x == 0 && !env->done) {
        // Clear current agent position in grid
        grid[env->agent_y * width + env->agent_x] = 0.0f;
        
        // Calculate movement
        int dx = 0, dy = 0;
        switch (action) {
            case 0: dy = -1; break; // up
            case 1: dx = 1;  break; // right
            case 2: dy = 1;  break; // down
            case 3: dx = -1; break; // left
            default: break;         // invalid action, no movement
        }
        
        // Update agent position with bounds checking
        int new_x = env->agent_x + dx;
        int new_y = env->agent_y + dy;
        
        // Check bounds
        if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
            int new_pos = new_y * width + new_x;
            float cell_value = grid[new_pos];
            
            // Check for obstacles (values > 0.8f and < 1.0f are obstacles)
            if (cell_value > 0.8f && cell_value < 1.0f) {
                // Hit obstacle - don't move, apply penalty
                env->reward = -0.5f;
            } else {
                // Valid move
                env->agent_x = new_x;
                env->agent_y = new_y;
                env->reward = -0.01f; // Small step penalty
                
                // Check for goal (value = 1.0f)
                if (cell_value == 1.0f) {
                    env->reward = 1.0f;
                    env->done = true;
                }
                // Check for traps (values between 0.6f and 0.8f)
                else if (cell_value >= 0.6f && cell_value < 0.8f) {
                    env->reward = -0.2f; // Trap penalty
                }
            }
        } else {
            // Out of bounds - apply penalty
            env->reward = -0.3f;
        }
        
        // Update agent position in grid
        grid[env->agent_y * width + env->agent_x] = 0.5f;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ADVANCED ENVIRONMENT FEATURES
// ══════════════════════════════════════════════════════════════════════════════

__global__ void placeObstaclesKernel(
    float* grids,
    const EnvironmentState* states,
    int batch_size,
    int width,
    int height,
    float obstacle_density,
    unsigned long long seed
) {
    int env_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    if (env_idx >= batch_size) return;
    
    float* grid = &grids[env_idx * width * height];
    const EnvironmentState* env = &states[env_idx];
    
    // Initialize cuRAND state
    curandState rand_state;
    curand_init(seed + env_idx * 1000 + thread_idx, 0, 0, &rand_state);
    
    int grid_size = width * height;
    int threads_per_block = blockDim.x;
    
    for (int i = thread_idx; i < grid_size; i += threads_per_block) {
        int x = i % width;
        int y = i / width;
        
        // Don't place obstacles on agent, goal, or adjacent to agent start
        bool is_agent_start = (x == width/2 && y == height/2);
        bool is_goal = (x == width-1 && y == height-1);
        bool near_agent_start = (abs(x - width/2) <= 1 && abs(y - height/2) <= 1);
        
        if (!is_agent_start && !is_goal && !near_agent_start) {
            float rand_val = curand_uniform(&rand_state);
            
            if (rand_val < obstacle_density) {
                // Place obstacle (value 0.9f)
                grid[i] = 0.9f;
            } else if (rand_val < obstacle_density + 0.1f) {
                // Place trap (value 0.7f)
                grid[i] = 0.7f;
            }
        }
    }
}

__global__ void computeAdvancedRewardsKernel(
    EnvironmentState* states,
    const float* grids,
    int batch_size,
    int width,
    int height,
    float step_penalty,
    float goal_reward,
    float obstacle_penalty,
    float trap_penalty
) {
    int env_idx = blockIdx.x;
    
    if (env_idx >= batch_size || threadIdx.x != 0) return;
    
    EnvironmentState* env = &states[env_idx];
    const float* grid = &grids[env_idx * width * height];
    
    if (env->done) return;
    
    int agent_pos = env->agent_y * width + env->agent_x;
    float cell_value = grid[agent_pos];
    
    // Base step penalty
    env->reward = step_penalty;
    
    // Check cell type and apply appropriate reward/penalty
    if (cell_value == 1.0f) {
        // Goal reached
        env->reward = goal_reward;
        env->done = true;
    } else if (cell_value > 0.8f && cell_value < 1.0f) {
        // Obstacle hit
        env->reward = obstacle_penalty;
    } else if (cell_value >= 0.6f && cell_value < 0.8f) {
        // Trap stepped on
        env->reward = trap_penalty;
    }
    
    // Distance-based reward bonus (encourage moving toward goal)
    int goal_x = width - 1;
    int goal_y = height - 1;
    float distance = sqrtf((env->agent_x - goal_x) * (env->agent_x - goal_x) + 
                          (env->agent_y - goal_y) * (env->agent_y - goal_y));
    float max_distance = sqrtf(width * width + height * height);
    float distance_bonus = 0.01f * (1.0f - distance / max_distance);
    env->reward += distance_bonus;
}

__global__ void checkCollisionsKernel(
    EnvironmentState* states,
    const float* grids,
    int batch_size,
    int width,
    int height
) {
    int env_idx = blockIdx.x;
    
    if (env_idx >= batch_size || threadIdx.x != 0) return;
    
    EnvironmentState* env = &states[env_idx];
    const float* grid = &grids[env_idx * width * height];
    
    int agent_pos = env->agent_y * width + env->agent_x;
    float cell_value = grid[agent_pos];
    
    // Check for collision with obstacles
    if (cell_value > 0.8f && cell_value < 1.0f) {
        env->done = true; // End episode on obstacle collision
    }
}

__global__ void updateAgentPositionsKernel(
    EnvironmentState* states,
    float* grids,
    const int* actions,
    int batch_size,
    int width,
    int height
) {
    int env_idx = blockIdx.x;
    
    if (env_idx >= batch_size || threadIdx.x != 0) return;
    
    EnvironmentState* env = &states[env_idx];
    float* grid = &grids[env_idx * width * height];
    int action = actions[env_idx];
    
    if (env->done) return;
    
    // Clear current agent position
    grid[env->agent_y * width + env->agent_x] = 0.0f;
    
    // Calculate new position
    int dx = 0, dy = 0;
    switch (action) {
        case 0: dy = -1; break; // up
        case 1: dx = 1;  break; // right
        case 2: dy = 1;  break; // down
        case 3: dx = -1; break; // left
    }
    
    int new_x = env->agent_x + dx;
    int new_y = env->agent_y + dy;
    
    // Bounds checking
    if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
        int new_pos = new_y * width + new_x;
        float cell_value = grid[new_pos];
        
        // Only move if not hitting obstacle
        if (!(cell_value > 0.8f && cell_value < 1.0f)) {
            env->agent_x = new_x;
            env->agent_y = new_y;
        }
    }
    
    // Update agent position in grid
    grid[env->agent_y * width + env->agent_x] = 0.5f;
}

// ══════════════════════════════════════════════════════════════════════════════
// HOST WRAPPER FUNCTIONS
// ══════════════════════════════════════════════════════════════════════════════

void launchBatchReset(
    EnvironmentState* d_states,
    float* d_grids,
    int batch_size,
    int width,
    int height,
    unsigned long long seed
) {
    dim3 block_size(256);
    dim3 grid_size(batch_size);
    
    if (seed == 0) {
        seed = static_cast<unsigned long long>(clock());
    }
    
    resetEnvironmentKernel<<<grid_size, block_size>>>(
        d_states, d_grids, batch_size, width, height, seed
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch error in launchBatchReset: %s\n", cudaGetErrorString(error));
    }
}

void launchBatchStep(
    EnvironmentState* d_states,
    float* d_grids,
    const int* d_actions,
    int batch_size,
    int width,
    int height
) {
    dim3 block_size(256);
    dim3 grid_size(batch_size);
    
    stepEnvironmentKernel<<<grid_size, block_size>>>(
        d_states, d_grids, d_actions, batch_size, width, height
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch error in launchBatchStep: %s\n", cudaGetErrorString(error));
    }
}

void launchBatchUpdate(
    EnvironmentState* d_states,
    float* d_grids,
    int batch_size,
    int width,
    int height
) {
    dim3 block_size(256);
    dim3 grid_size(batch_size);
    
    updateEnvironmentKernel<<<grid_size, block_size>>>(
        d_states, d_grids, batch_size, width, height
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch error in launchBatchUpdate: %s\n", cudaGetErrorString(error));
    }
}

void launchPlaceObstacles(
    float* d_grids,
    const EnvironmentState* d_states,
    int batch_size,
    int width,
    int height,
    float obstacle_density,
    unsigned long long seed
) {
    dim3 block_size(256);
    dim3 grid_size(batch_size);
    
    if (seed == 0) {
        seed = static_cast<unsigned long long>(clock());
    }
    
    placeObstaclesKernel<<<grid_size, block_size>>>(
        d_grids, d_states, batch_size, width, height, obstacle_density, seed
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch error in launchPlaceObstacles: %s\n", cudaGetErrorString(error));
    }
}

} // namespace cudarl
