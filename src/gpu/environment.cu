#include "../core/environment.h"
#include <cstdlib>
#include <ctime>
#include <random>
#include <iostream>

namespace cudarl {

// CUDA kernel to reset the environment
__global__ void reset_environment(EnvironmentState* state) {
    // Initialize agent position to center
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        state->agent_x = state->width / 2;
        state->agent_y = state->height / 2;
        state->reward = 0.0f;
        state->done = false;
    }
}

// CUDA kernel to step the environment based on action
__global__ void step_environment(EnvironmentState* state, int action) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Action: 0=up, 1=right, 2=down, 3=left
        int dx = 0, dy = 0;
        
        switch (action) {
            case 0: dy = -1; break; // up
            case 1: dx = 1;  break; // right
            case 2: dy = 1;  break; // down
            case 3: dx = -1; break; // left
            default: break;         // invalid action, no movement
        }
        
        // Update agent position with bounds checking
        int new_x = state->agent_x + dx;
        int new_y = state->agent_y + dy;
        
        if (new_x >= 0 && new_x < state->width && 
            new_y >= 0 && new_y < state->height) {
            state->agent_x = new_x;
            state->agent_y = new_y;
        }
        
        // Simple reward: -0.01 per step, +1 for reaching goal (top-right corner)
        state->reward = -0.01f;
        
        // Check if agent reached goal (top-right corner)
        if (state->agent_x == state->width - 1 && state->agent_y == 0) {
            state->reward = 1.0f;
            state->done = true;
        }
    }
}

Environment::Environment(int id, int width, int height) : m_envId(id) {
    // Initialize host state
    m_state.width = width;
    m_state.height = height;
    m_state.agent_x = width / 2;
    m_state.agent_y = height / 2;
    m_state.reward = 0.0f;
    m_state.done = false;
    
    // Allocate host grid
    m_state.grid = new float[width * height];
    initializeGrid();
    
    // Allocate device state
    m_deviceState.allocate(1);
    
    // Allocate device grid
    m_deviceGrid.allocate(width * height);
    m_deviceGrid.copyFromHost(m_state.grid, width * height);
    
    // Copy state to device (need to handle the grid pointer separately)
    EnvironmentState temp_state = m_state;
    temp_state.grid = m_deviceGrid.get();
    CUDA_CHECK(cudaMemcpy(m_deviceState.get(), &temp_state, sizeof(EnvironmentState), cudaMemcpyHostToDevice));
    
    std::cout << "Environment " << m_envId << " constructed (" << width << "x" << height << ")." << std::endl;
}

Environment::~Environment() {
    // Free host grid
    if (m_state.grid) {
        delete[] m_state.grid;
        m_state.grid = nullptr;
    }
    
    std::cout << "Environment " << m_envId << " destructed." << std::endl;
}

// Move constructor
Environment::Environment(Environment&& other) noexcept
    : m_envId(other.m_envId),
      m_state(other.m_state),
      m_deviceState(std::move(other.m_deviceState)),
      m_deviceGrid(std::move(other.m_deviceGrid)) {
    
    // Transfer ownership of host grid
    other.m_state.grid = nullptr;
}

// Move assignment
Environment& Environment::operator=(Environment&& other) noexcept {
    if (this != &other) {
        // Free existing resources
        if (m_state.grid) {
            delete[] m_state.grid;
        }
        
        // Move data
        m_envId = other.m_envId;
        m_state = other.m_state;
        m_deviceState = std::move(other.m_deviceState);
        m_deviceGrid = std::move(other.m_deviceGrid);
        
        // Transfer ownership of host grid
        other.m_state.grid = nullptr;
    }
    return *this;
}

void Environment::reset() {
    // Reset on device
    reset_environment<<<1, 1>>>(m_deviceState.get());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Update host state
    updateHostState();
    
    std::cout << "Environment " << m_envId << " reset." << std::endl;
}

void Environment::step(int action) {
    // Step on device
    step_environment<<<1, 1>>>(m_deviceState.get(), action);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Update host state
    updateHostState();
    
    std::cout << "Environment " << m_envId << " performed action " << action 
              << ", reward: " << m_state.reward 
              << ", position: (" << m_state.agent_x << "," << m_state.agent_y << ")"
              << ", done: " << (m_state.done ? "true" : "false") << std::endl;
}

float Environment::getCellValue(int x, int y) const {
    if (x >= 0 && x < m_state.width && y >= 0 && y < m_state.height) {
        return m_state.grid[y * m_state.width + x];
    }
    return 0.0f;
}

std::vector<float> Environment::getGrid() const {
    return std::vector<float>(m_state.grid, m_state.grid + (m_state.width * m_state.height));
}

void Environment::initializeGrid() {
    // Use modern C++ random number generation
    std::mt19937 rng(std::time(nullptr) + m_envId);
    std::uniform_real_distribution<float> dist(0.0f, 0.5f);
    
    // Initialize with random values
    for (int i = 0; i < m_state.width * m_state.height; i++) {
        m_state.grid[i] = dist(rng);
    }
    
    // Set goal (top-right corner) to a distinct value
    m_state.grid[m_state.width - 1] = 1.0f;
}

void Environment::updateHostState() {
    // Get device state (need to handle the grid pointer separately)
    EnvironmentState temp_state;
    CUDA_CHECK(cudaMemcpy(&temp_state, m_deviceState.get(), sizeof(EnvironmentState), cudaMemcpyDeviceToHost));
    
    // Update host state fields (except grid pointer)
    m_state.agent_x = temp_state.agent_x;
    m_state.agent_y = temp_state.agent_y;
    m_state.reward = temp_state.reward;
    m_state.done = temp_state.done;
    
    // Copy grid data from device to host
    m_deviceGrid.copyToHost(m_state.grid, m_state.width * m_state.height);
}

void Environment::syncToDevice() {
    // Copy grid to device
    m_deviceGrid.copyFromHost(m_state.grid, m_state.width * m_state.height);
    
    // Copy state to device
    EnvironmentState temp_state = m_state;
    temp_state.grid = m_deviceGrid.get();
    CUDA_CHECK(cudaMemcpy(m_deviceState.get(), &temp_state, sizeof(EnvironmentState), cudaMemcpyHostToDevice));
}

} // namespace cudarl