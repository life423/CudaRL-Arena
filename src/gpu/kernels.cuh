#pragma once

#include <cuda_runtime.h>

namespace cudarl {

// CUDA kernel for updating Q-values in batch
__global__ void updateQValuesKernel(
    float* q_table,           // Q-table [state_size * action_size]
    const int* states,        // Current states [batch_size]
    const int* actions,       // Actions taken [batch_size]
    const float* rewards,     // Rewards received [batch_size]
    const int* next_states,   // Next states [batch_size]
    const bool* dones,        // Done flags [batch_size]
    const int state_size,     // Total number of states
    const int action_size,    // Total number of actions
    const float learning_rate,// Alpha
    const float discount,     // Gamma
    const int batch_size      // Number of samples to process
);

// Host function to launch the Q-value update kernel
void updateQValuesBatch(
    float* d_q_table,         // Device Q-table
    const int* d_states,      // Device states
    const int* d_actions,     // Device actions
    const float* d_rewards,   // Device rewards
    const int* d_next_states, // Device next states
    const bool* d_dones,      // Device done flags
    const int state_size,     // Total number of states
    const int action_size,    // Total number of actions
    const float learning_rate,// Alpha
    const float discount,     // Gamma
    const int batch_size      // Number of samples to process
);

} // namespace cudarl