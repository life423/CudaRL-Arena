#include "q_learning.cuh"
#include "kernels.cuh"
#include <stdexcept>
#include <iostream>

CudaQLearning::CudaQLearning(int width, int height, int action_space_size,
                           float learning_rate, float discount_factor)
    : width(width), height(height), action_space_size(action_space_size),
      learning_rate(learning_rate), discount_factor(discount_factor),
      max_batch_size(1024) {
    
    // Initialize host Q-table with zeros
    h_q_table.resize(width * height * action_space_size, 0.0f);
    
    // Allocate device memory for Q-table
    cudaError_t error = cudaMalloc(&d_q_table, width * height * action_space_size * sizeof(float));
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for Q-table");
    }
    
    // Copy Q-table to device
    error = cudaMemcpy(d_q_table, h_q_table.data(), h_q_table.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(d_q_table);
        throw std::runtime_error("Failed to copy Q-table to device");
    }
    
    // Allocate device memory for batch updates
    error = cudaMalloc(&d_states_x, max_batch_size * sizeof(int));
    if (error != cudaSuccess) {
        cudaFree(d_q_table);
        throw std::runtime_error("Failed to allocate device memory for states_x");
    }
    
    error = cudaMalloc(&d_states_y, max_batch_size * sizeof(int));
    if (error != cudaSuccess) {
        cudaFree(d_q_table);
        cudaFree(d_states_x);
        throw std::runtime_error("Failed to allocate device memory for states_y");
    }
    
    error = cudaMalloc(&d_actions, max_batch_size * sizeof(int));
    if (error != cudaSuccess) {
        cudaFree(d_q_table);
        cudaFree(d_states_x);
        cudaFree(d_states_y);
        throw std::runtime_error("Failed to allocate device memory for actions");
    }
    
    error = cudaMalloc(&d_rewards, max_batch_size * sizeof(float));
    if (error != cudaSuccess) {
        cudaFree(d_q_table);
        cudaFree(d_states_x);
        cudaFree(d_states_y);
        cudaFree(d_actions);
        throw std::runtime_error("Failed to allocate device memory for rewards");
    }
    
    error = cudaMalloc(&d_next_states_x, max_batch_size * sizeof(int));
    if (error != cudaSuccess) {
        cudaFree(d_q_table);
        cudaFree(d_states_x);
        cudaFree(d_states_y);
        cudaFree(d_actions);
        cudaFree(d_rewards);
        throw std::runtime_error("Failed to allocate device memory for next_states_x");
    }
    
    error = cudaMalloc(&d_next_states_y, max_batch_size * sizeof(int));
    if (error != cudaSuccess) {
        cudaFree(d_q_table);
        cudaFree(d_states_x);
        cudaFree(d_states_y);
        cudaFree(d_actions);
        cudaFree(d_rewards);
        cudaFree(d_next_states_x);
        throw std::runtime_error("Failed to allocate device memory for next_states_y");
    }
    
    error = cudaMalloc(&d_dones, max_batch_size * sizeof(bool));
    if (error != cudaSuccess) {
        cudaFree(d_q_table);
        cudaFree(d_states_x);
        cudaFree(d_states_y);
        cudaFree(d_actions);
        cudaFree(d_rewards);
        cudaFree(d_next_states_x);
        cudaFree(d_next_states_y);
        throw std::runtime_error("Failed to allocate device memory for dones");
    }
    
    std::cout << "CudaQLearning initialized with " << width << "x" << height << " states and " 
              << action_space_size << " actions" << std::endl;
}

CudaQLearning::~CudaQLearning() {
    // Free device memory
    cudaFree(d_q_table);
    cudaFree(d_states_x);
    cudaFree(d_states_y);
    cudaFree(d_actions);
    cudaFree(d_rewards);
    cudaFree(d_next_states_x);
    cudaFree(d_next_states_y);
    cudaFree(d_dones);
}

void CudaQLearning::update_batch(
    const std::vector<std::pair<int, int>>& states,
    const std::vector<int>& actions,
    const std::vector<float>& rewards,
    const std::vector<std::pair<int, int>>& next_states,
    const std::vector<bool>& dones
) {
    int batch_size = states.size();
    
    if (batch_size > max_batch_size) {
        throw std::runtime_error("Batch size exceeds maximum batch size");
    }
    
    if (batch_size != actions.size() || batch_size != rewards.size() || 
        batch_size != next_states.size() || batch_size != dones.size()) {
        throw std::runtime_error("Inconsistent batch sizes");
    }
    
    // Prepare host data
    std::vector<int> h_states_x(batch_size);
    std::vector<int> h_states_y(batch_size);
    std::vector<int> h_next_states_x(batch_size);
    std::vector<int> h_next_states_y(batch_size);
    
    for (int i = 0; i < batch_size; i++) {
        h_states_x[i] = states[i].first;
        h_states_y[i] = states[i].second;
        h_next_states_x[i] = next_states[i].first;
        h_next_states_y[i] = next_states[i].second;
    }
    
    // Copy data to device
    cudaMemcpy(d_states_x, h_states_x.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_states_y, h_states_y.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_actions, actions.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rewards, rewards.data(), batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_next_states_x, h_next_states_x.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_next_states_y, h_next_states_y.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dones, dones.data(), batch_size * sizeof(bool), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    
    update_q_values_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_q_table,
        d_states_x,
        d_states_y,
        d_actions,
        d_rewards,
        d_next_states_x,
        d_next_states_y,
        d_dones,
        batch_size,
        width,
        height,
        action_space_size,
        learning_rate,
        discount_factor
    );
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // Copy updated Q-table back to host
    cudaMemcpy(h_q_table.data(), d_q_table, h_q_table.size() * sizeof(float), cudaMemcpyDeviceToHost);
}

std::vector<float> CudaQLearning::get_q_table() const {
    return h_q_table;
}

void CudaQLearning::set_q_table(const std::vector<float>& q_table) {
    if (q_table.size() != width * height * action_space_size) {
        throw std::runtime_error("Invalid Q-table size");
    }
    
    h_q_table = q_table;
    
    // Copy Q-table to device
    cudaMemcpy(d_q_table, h_q_table.data(), h_q_table.size() * sizeof(float), cudaMemcpyHostToDevice);
}

int CudaQLearning::get_best_action(int x, int y) const {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        throw std::runtime_error("Invalid state");
    }
    
    int base_idx = (y * width + x) * action_space_size;
    int best_action = 0;
    float best_value = h_q_table[base_idx];
    
    for (int a = 1; a < action_space_size; a++) {
        float value = h_q_table[base_idx + a];
        if (value > best_value) {
            best_value = value;
            best_action = a;
        }
    }
    
    return best_action;
}

float CudaQLearning::get_q_value(int x, int y, int action) const {
    if (x < 0 || x >= width || y < 0 || y >= height || action < 0 || action >= action_space_size) {
        throw std::runtime_error("Invalid state or action");
    }
    
    return h_q_table[(y * width + x) * action_space_size + action];
}