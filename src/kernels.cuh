#pragma once
#include <cuda_runtime.h>
#include "environment.h"

/**
 * CUDA kernel to reset the environment
 * @param state Pointer to environment state in device memory
 */
__global__ void reset_environment_kernel(EnvironmentState* state);

/**
 * CUDA kernel to step the environment based on action
 * @param state Pointer to environment state in device memory
 * @param action Action to take (0=up, 1=right, 2=down, 3=left)
 */
__global__ void step_environment_kernel(EnvironmentState* state, int action);

/**
 * CUDA kernel to update Q-values in batch
 * @param q_table Q-table in device memory
 * @param states Array of states (x, y coordinates)
 * @param actions Array of actions taken
 * @param rewards Array of rewards received
 * @param next_states Array of next states
 * @param dones Array of done flags
 * @param num_samples Number of samples to process
 * @param learning_rate Learning rate (alpha)
 * @param discount_factor Discount factor (gamma)
 */
__global__ void update_q_values_kernel(
    float* q_table,
    int* states_x,
    int* states_y,
    int* actions,
    float* rewards,
    int* next_states_x,
    int* next_states_y,
    bool* dones,
    int num_samples,
    int width,
    int height,
    int action_space_size,
    float learning_rate,
    float discount_factor
);