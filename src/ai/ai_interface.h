#pragma once

namespace ai {
    // CUDA-accelerated Q-learning update
    void update_q_table_cuda(float* q_table, int* states, int* actions, float* rewards, int* next_states, 
                            float learning_rate, float discount_factor, int batch_size, int state_space_size, int action_space_size);
    
    // Get best action from Q-table for a given state
    int get_best_action(float* q_table, int state_id, int action_space_size);
    
    // CUDA-accelerated DQN forward pass
    void dqn_forward_cuda(float* states, float* hidden_weights, float* hidden_biases, float* hidden_output,
                         float* output_weights, float* output_biases, float* q_values,
                         int batch_size, int state_size, int hidden_size, int action_size);
}