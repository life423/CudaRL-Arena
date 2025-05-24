#include "kernels.cuh"
#include <stdio.h>

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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        int state = states[idx];
        int action = actions[idx];
        float reward = rewards[idx];
        int next_state = next_states[idx];
        bool done = dones[idx];
        
        // Calculate Q-table index for current state-action pair
        int q_idx = state * action_size + action;
        
        // Get current Q-value
        float q_value = q_table[q_idx];
        
        // Calculate max Q-value for next state
        float max_next_q = 0.0f;
        
        if (!done) {
            // Find max Q-value for next state
            for (int a = 0; a < action_size; a++) {
                int next_q_idx = next_state * action_size + a;
                float next_q = q_table[next_q_idx];
                max_next_q = (a == 0) ? next_q : fmaxf(max_next_q, next_q);
            }
        }
        
        // Q-learning update rule: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        float target = reward + discount * max_next_q;
        float new_q_value = q_value + learning_rate * (target - q_value);
        
        // Update Q-table
        q_table[q_idx] = new_q_value;
    }
}

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
) {
    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    updateQValuesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_q_table, d_states, d_actions, d_rewards, d_next_states, d_dones,
        state_size, action_size, learning_rate, discount, batch_size
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
}

} // namespace cudarl