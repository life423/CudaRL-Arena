#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// CUDA kernel for Q-learning update
__global__ void update_q_table_kernel(
    float* q_table,
    int* states,
    int* actions,
    float* rewards,
    int* next_states,
    float learning_rate,
    float discount_factor,
    int batch_size,
    int state_space_size,
    int action_space_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        int state = states[idx];
        int action = actions[idx];
        float reward = rewards[idx];
        int next_state = next_states[idx];
        
        // Find max Q-value for next state
        float max_q_next = 0.0f;
        for (int a = 0; a < action_space_size; a++) {
            float q_next = q_table[next_state * action_space_size + a];
            max_q_next = (a == 0) ? q_next : fmaxf(max_q_next, q_next);
        }
        
        // Q-learning update rule: Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        int q_idx = state * action_space_size + action;
        float current_q = q_table[q_idx];
        float target_q = reward + discount_factor * max_q_next;
        q_table[q_idx] = current_q + learning_rate * (target_q - current_q);
    }
}

// C++ wrapper function for the CUDA kernel
extern "C" void update_q_table_cuda(
    float* q_table,
    int* states,
    int* actions,
    float* rewards,
    int* next_states,
    float learning_rate,
    float discount_factor,
    int batch_size,
    int state_space_size,
    int action_space_size
) {
    // Allocate device memory
    float* d_q_table;
    int* d_states;
    int* d_actions;
    float* d_rewards;
    int* d_next_states;
    
    cudaMalloc((void**)&d_q_table, state_space_size * action_space_size * sizeof(float));
    cudaMalloc((void**)&d_states, batch_size * sizeof(int));
    cudaMalloc((void**)&d_actions, batch_size * sizeof(int));
    cudaMalloc((void**)&d_rewards, batch_size * sizeof(float));
    cudaMalloc((void**)&d_next_states, batch_size * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_q_table, q_table, state_space_size * action_space_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, states, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_actions, actions, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rewards, rewards, batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_next_states, next_states, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    update_q_table_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_q_table, d_states, d_actions, d_rewards, d_next_states,
        learning_rate, discount_factor, batch_size, state_space_size, action_space_size
    );
    
    // Copy results back to host
    cudaMemcpy(q_table, d_q_table, state_space_size * action_space_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_q_table);
    cudaFree(d_states);
    cudaFree(d_actions);
    cudaFree(d_rewards);
    cudaFree(d_next_states);
}

// Helper function to get best action for a state
extern "C" int get_best_action(float* q_table, int state_id, int action_space_size) {
    float max_q_value = q_table[state_id * action_space_size];
    int best_action = 0;
    
    for (int a = 1; a < action_space_size; a++) {
        float q_value = q_table[state_id * action_space_size + a];
        if (q_value > max_q_value) {
            max_q_value = q_value;
            best_action = a;
        }
    }
    
    return best_action;
}