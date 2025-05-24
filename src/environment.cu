#include "environment.h"
#include <cstdlib>
#include <ctime>

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

Environment::Environment(int id, int width, int height) : env_id(id), d_state(nullptr) {
    // Initialize host state
    state.width = width;
    state.height = height;
    state.agent_x = width / 2;
    state.agent_y = height / 2;
    state.reward = 0.0f;
    state.done = false;
    
    // Allocate host grid
    state.grid = new float[width * height];
    initializeGrid();
    
    // Allocate device state
    cudaMalloc(&d_state, sizeof(EnvironmentState));
    
    // Allocate device grid
    float* d_grid;
    cudaMalloc(&d_grid, width * height * sizeof(float));
    cudaMemcpy(d_grid, state.grid, width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy state to device (need to handle the grid pointer separately)
    EnvironmentState temp_state = state;
    temp_state.grid = d_grid;
    cudaMemcpy(d_state, &temp_state, sizeof(EnvironmentState), cudaMemcpyHostToDevice);
    
    std::cout << "Environment " << env_id << " constructed (" << width << "x" << height << ")." << std::endl;
}

Environment::~Environment() {
    if (d_state) {
        // Get device grid pointer before freeing state
        EnvironmentState temp_state;
        cudaMemcpy(&temp_state, d_state, sizeof(EnvironmentState), cudaMemcpyDeviceToHost);
        
        // Free device grid
        if (temp_state.grid) {
            cudaFree(temp_state.grid);
        }
        
        // Free device state
        cudaFree(d_state);
    }
    
    // Free host grid
    if (state.grid) {
        delete[] state.grid;
    }
    
    std::cout << "Environment " << env_id << " destructed." << std::endl;
}

void Environment::reset() {
    // Reset on device
    reset_environment<<<1, 1>>>(d_state);
    cudaDeviceSynchronize();
    
    // Update host state
    updateHostState();
    
    std::cout << "Environment " << env_id << " reset." << std::endl;
}

void Environment::step(int action) {
    // Step on device
    step_environment<<<1, 1>>>(d_state, action);
    cudaDeviceSynchronize();
    
    // Update host state
    updateHostState();
    
    std::cout << "Environment " << env_id << " performed action " << action 
              << ", reward: " << state.reward 
              << ", position: (" << state.agent_x << "," << state.agent_y << ")"
              << ", done: " << (state.done ? "true" : "false") << std::endl;
}

float Environment::getCellValue(int x, int y) const {
    if (x >= 0 && x < state.width && y >= 0 && y < state.height) {
        return state.grid[y * state.width + x];
    }
    return 0.0f;
}

std::vector<float> Environment::getGrid() const {
    return std::vector<float>(state.grid, state.grid + (state.width * state.height));
}

void Environment::initializeGrid() {
    // Initialize with random values between 0 and 1
    std::srand(std::time(nullptr) + env_id);
    for (int i = 0; i < state.width * state.height; i++) {
        state.grid[i] = static_cast<float>(std::rand()) / RAND_MAX * 0.5f;
    }
    
    // Set goal (top-right corner) to a distinct value
    state.grid[state.width - 1] = 1.0f;
}

void Environment::updateHostState() {
    // Get device state (need to handle the grid pointer separately)
    EnvironmentState temp_state;
    cudaMemcpy(&temp_state, d_state, sizeof(EnvironmentState), cudaMemcpyDeviceToHost);
    
    // Update host state fields (except grid pointer)
    state.agent_x = temp_state.agent_x;
    state.agent_y = temp_state.agent_y;
    state.reward = temp_state.reward;
    state.done = temp_state.done;
    
    // Copy grid data from device to host
    cudaMemcpy(state.grid, temp_state.grid, state.width * state.height * sizeof(float), cudaMemcpyDeviceToHost);
}