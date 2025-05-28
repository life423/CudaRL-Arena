#include "environment.h"
#include <curand_kernel.h>
#include <chrono>
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

Environment::Environment(int id, int width, int height) 
    : m_envId(id)
    , m_deviceState(1)
    , m_deviceGrid(width * height) {
    
    // Initialize host state
    m_state.width = width;
    m_state.height = height;
    m_state.agent_x = width / 2;
    m_state.agent_y = height / 2;
    m_state.reward = 0.0f;
    m_state.done = false;
    m_state.grid = nullptr; // We'll use device memory only
      // Initialize grid on device
    initializeGridOnDevice();
    
    // Copy state to device (grid pointer will be set on device)
    syncStateToDevice();
    
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
    // Reset on device - no unnecessary synchronization
    reset_environment<<<1, 32>>>(m_deviceState.get());
    
    // Update host state (sync happens here when needed)
    updateHostState();
    
    std::cout << "Environment " << m_envId << " reset." << std::endl;
}

void Environment::step(int action) {
    // Step on device - no unnecessary synchronization
    step_environment<<<1, 32>>>(m_deviceState.get(), action);
    
    // Update host state (sync happens here when needed)
    updateHostState();
    
    std::cout << "Environment " << m_envId << " performed action " << action 
              << ", reward: " << m_state.reward 
              << ", position: (" << m_state.agent_x << "," << m_state.agent_y << ")"
              << ", done: " << (m_state.done ? "true" : "false") << std::endl;
}

float Environment::getCellValue(int x, int y) const {
    if (x < 0 || x >= m_state.width || y < 0 || y >= m_state.height) {
        return 0.0f;
    }
    
    // Get value from device grid
    std::vector<float> hostGrid = m_deviceGrid.copyToHostAll();
    return hostGrid[y * m_state.width + x];
}

std::vector<float> Environment::getGrid() const {
    return m_deviceGrid.copyToHostAll();
}

// GPU kernel to initialize grid with random values using cuRAND
__global__ void initializeGridKernel(float* grid, int width, int height, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = width * height;
    
    if (idx < totalSize) {
        // Initialize cuRAND state for this thread
        curandState state;
        curand_init(seed + idx, idx, 0, &state);
        
        // Generate random value between 0.0f and 0.5f
        grid[idx] = curand_uniform(&state) * 0.5f;
    }
    
    // Set goal (top-right corner) to a distinct value
    if (idx == width - 1) {
        grid[idx] = 1.0f;
    }
}

void Environment::initializeGridOnDevice() {
    // Calculate grid size and launch configuration
    int totalSize = m_state.width * m_state.height;
    int blockSize = 256;
    int numBlocks = (totalSize + blockSize - 1) / blockSize;
    
    // Generate seed based on environment ID and current time
    unsigned long long seed = static_cast<unsigned long long>(std::chrono::steady_clock::now().time_since_epoch().count()) + m_envId;
    
    // Launch GPU kernel to initialize grid
    initializeGridKernel<<<numBlocks, blockSize>>>(m_deviceGrid.get(), m_state.width, m_state.height, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Environment::syncStateToDevice() {
    // Copy state to device (grid pointer will be set correctly on device)
    EnvironmentState temp_state = m_state;
    temp_state.grid = m_deviceGrid.get();
    CUDA_CHECK(cudaMemcpy(m_deviceState.get(), &temp_state, sizeof(EnvironmentState), cudaMemcpyHostToDevice));
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
    
    // Note: We don't copy grid data to host here since we use device memory directly
}

void Environment::syncToDevice() {
    // Copy state to device
    EnvironmentState temp_state = m_state;
    temp_state.grid = m_deviceGrid.get();
    CUDA_CHECK(cudaMemcpy(m_deviceState.get(), &temp_state, sizeof(EnvironmentState), cudaMemcpyHostToDevice));
}

} // namespace cudarl
