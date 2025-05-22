#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// CUDA kernel for DQN forward pass
__global__ void dqn_forward_kernel(
    float* input_layer,
    float* hidden_weights,
    float* hidden_biases,
    float* hidden_output,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * hidden_size) {
        int batch_idx = idx / hidden_size;
        int hidden_idx = idx % hidden_size;
        
        float sum = hidden_biases[hidden_idx];
        for (int i = 0; i < input_size; i++) {
            sum += input_layer[batch_idx * input_size + i] * hidden_weights[i * hidden_size + hidden_idx];
        }
        
        // ReLU activation
        hidden_output[idx] = fmaxf(0.0f, sum);
    }
}

// CUDA kernel for DQN output layer
__global__ void dqn_output_kernel(
    float* hidden_output,
    float* output_weights,
    float* output_biases,
    float* q_values,
    int batch_size,
    int hidden_size,
    int output_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * output_size) {
        int batch_idx = idx / output_size;
        int output_idx = idx % output_size;
        
        float sum = output_biases[output_idx];
        for (int i = 0; i < hidden_size; i++) {
            sum += hidden_output[batch_idx * hidden_size + i] * output_weights[i * output_size + output_idx];
        }
        
        q_values[idx] = sum;
    }
}

// C++ wrapper function for DQN forward pass
extern "C" void dqn_forward_cuda(
    float* states,
    float* hidden_weights,
    float* hidden_biases,
    float* hidden_output,
    float* output_weights,
    float* output_biases,
    float* q_values,
    int batch_size,
    int state_size,
    int hidden_size,
    int action_size
) {
    // Allocate device memory
    float *d_states, *d_hidden_weights, *d_hidden_biases, *d_hidden_output;
    float *d_output_weights, *d_output_biases, *d_q_values;
    
    cudaMalloc((void**)&d_states, batch_size * state_size * sizeof(float));
    cudaMalloc((void**)&d_hidden_weights, state_size * hidden_size * sizeof(float));
    cudaMalloc((void**)&d_hidden_biases, hidden_size * sizeof(float));
    cudaMalloc((void**)&d_hidden_output, batch_size * hidden_size * sizeof(float));
    cudaMalloc((void**)&d_output_weights, hidden_size * action_size * sizeof(float));
    cudaMalloc((void**)&d_output_biases, action_size * sizeof(float));
    cudaMalloc((void**)&d_q_values, batch_size * action_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_states, states, batch_size * state_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden_weights, hidden_weights, state_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden_biases, hidden_biases, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_weights, output_weights, hidden_size * action_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_biases, output_biases, action_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernels
    int threadsPerBlock = 256;
    int hidden_blocks = (batch_size * hidden_size + threadsPerBlock - 1) / threadsPerBlock;
    int output_blocks = (batch_size * action_size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Forward pass through hidden layer
    dqn_forward_kernel<<<hidden_blocks, threadsPerBlock>>>(
        d_states, d_hidden_weights, d_hidden_biases, d_hidden_output,
        batch_size, state_size, hidden_size
    );
    
    // Forward pass through output layer
    dqn_output_kernel<<<output_blocks, threadsPerBlock>>>(
        d_hidden_output, d_output_weights, d_output_biases, d_q_values,
        batch_size, hidden_size, action_size
    );
    
    // Copy results back to host
    cudaMemcpy(q_values, d_q_values, batch_size * action_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_states);
    cudaFree(d_hidden_weights);
    cudaFree(d_hidden_biases);
    cudaFree(d_hidden_output);
    cudaFree(d_output_weights);
    cudaFree(d_output_biases);
    cudaFree(d_q_values);
}