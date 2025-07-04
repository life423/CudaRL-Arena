#include "cuda_arena.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

__global__ void hello_kernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from GPU (thread %d)!\n", tid);
}

__global__ void step_environments_kernel(
    float* observations,
    float* rewards,
    int* dones,
    int* actions,
    int num_envs
) {
    int env_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (env_id >= num_envs) return;
    
    // Grid world: obs = [agent_x, agent_y, goal_x, goal_y]
    float agent_x = observations[env_id * 4 + 0];
    float agent_y = observations[env_id * 4 + 1];
    float goal_x = observations[env_id * 4 + 2];
    float goal_y = observations[env_id * 4 + 3];
    
    int action = actions[env_id];
    
    // Move agent (0=up, 1=right, 2=down, 3=left)
    if (action == 0) agent_y += 1.0f;
    else if (action == 1) agent_x += 1.0f;
    else if (action == 2) agent_y -= 1.0f;
    else if (action == 3) agent_x -= 1.0f;
    
    // Clamp to grid bounds [0, 9]
    agent_x = fmaxf(0.0f, fminf(9.0f, agent_x));
    agent_y = fmaxf(0.0f, fminf(9.0f, agent_y));
    
    // Calculate reward
    float dist_to_goal = sqrtf((agent_x - goal_x) * (agent_x - goal_x) + 
                               (agent_y - goal_y) * (agent_y - goal_y));
    rewards[env_id] = (dist_to_goal < 1.0f) ? 10.0f : -0.1f;
    
    // Check if done
    dones[env_id] = (dist_to_goal < 1.0f) ? 1 : 0;
    
    // Update observations
    observations[env_id * 4 + 0] = agent_x;
    observations[env_id * 4 + 1] = agent_y;
    observations[env_id * 4 + 2] = goal_x;
    observations[env_id * 4 + 3] = goal_y;
}

CudaArena::CudaArena(int num_envs) : m_num_envs(num_envs) {
    allocate_memory();
    reset_environments();
}

CudaArena::~CudaArena() {
    free_memory();
}

void CudaArena::allocate_memory() {
    size_t obs_size = m_num_envs * 4 * sizeof(float);
    size_t reward_size = m_num_envs * sizeof(float);
    size_t done_size = m_num_envs * sizeof(int);
    size_t action_size = m_num_envs * sizeof(int);
    
    cudaMalloc(&d_observations, obs_size);
    cudaMalloc(&d_rewards, reward_size);
    cudaMalloc(&d_dones, done_size);
    cudaMalloc(&d_actions, action_size);
    
    h_observations.resize(m_num_envs * 4);
    h_rewards.resize(m_num_envs);
    h_dones.resize(m_num_envs);
    h_actions.resize(m_num_envs);
}

void CudaArena::free_memory() {
    cudaFree(d_observations);
    cudaFree(d_rewards);
    cudaFree(d_dones);
    cudaFree(d_actions);
}

void CudaArena::reset_environments() {
    srand(time(nullptr));
    
    for (int i = 0; i < m_num_envs; i++) {
        h_observations[i * 4 + 0] = 5.0f; // agent_x (start at middle left)
        h_observations[i * 4 + 1] = 0.0f; // agent_y
        h_observations[i * 4 + 2] = 5.0f; // goal_x (middle right)
        h_observations[i * 4 + 3] = 9.0f; // goal_y
        h_rewards[i] = 0.0f;
        h_dones[i] = 0;
        h_actions[i] = 0;
    }
    
    cudaMemcpy(d_observations, h_observations.data(), h_observations.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rewards, h_rewards.data(), h_rewards.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dones, h_dones.data(), h_dones.size() * sizeof(int), cudaMemcpyHostToDevice);
}

void CudaArena::step_environments(const std::vector<int>& actions) {
    h_actions = actions;
    cudaMemcpy(d_actions, h_actions.data(), h_actions.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int grid_size = (m_num_envs + block_size - 1) / block_size;
    
    step_environments_kernel<<<grid_size, block_size>>>(
        d_observations, d_rewards, d_dones, d_actions, m_num_envs
    );
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_observations.data(), d_observations, h_observations.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rewards.data(), d_rewards, h_rewards.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dones.data(), d_dones, h_dones.size() * sizeof(int), cudaMemcpyDeviceToHost);
}

std::vector<float> CudaArena::get_observations() const {
    return h_observations;
}

std::vector<float> CudaArena::get_rewards() const {
    return h_rewards;
}

std::vector<int> CudaArena::get_dones() const {
    return h_dones;
}

int CudaArena::get_device_count() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

std::string CudaArena::get_device_name(int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return std::string(prop.name);
}

void CudaArena::hello_cuda() {
    hello_kernel<<<1, 4>>>();
    cudaDeviceSynchronize();
}