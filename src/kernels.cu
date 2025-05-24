#include "kernels.cuh"

__global__ void reset_environment_kernel(EnvironmentState* state) {
    // Initialize agent position to center
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        state->agent_x = state->width / 2;
        state->agent_y = state->height / 2;
        state->reward = 0.0f;
        state->done = false;
    }
}

__global__ void step_environment_kernel(EnvironmentState* state, int action) {
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
        
        // Check for obstacles (value < 0 in grid)
        int grid_idx = new_y * state->width + new_x;
        bool is_obstacle = false;
        
        if (new_x >= 0 && new_x < state->width && 
            new_y >= 0 && new_y < state->height) {
            is_obstacle = state->grid[grid_idx] < 0.0f;
        }
        
        // Only move if within bounds and not an obstacle
        if (new_x >= 0 && new_x < state->width && 
            new_y >= 0 && new_y < state->height && 
            !is_obstacle) {
            state->agent_x = new_x;
            state->agent_y = new_y;
        }
        
        // Default reward is small negative (step penalty)
        state->reward = -0.01f;
        
        // Check for special cells
        if (new_x >= 0 && new_x < state->width && 
            new_y >= 0 && new_y < state->height) {
            
            float cell_value = state->grid[grid_idx];
            
            // Goal (value = 1.0)
            if (cell_value >= 0.99f) {
                state->reward = 1.0f;
                state->done = true;
            }
            // Trap (value between -0.9 and -0.1)
            else if (cell_value < 0.0f && cell_value > -1.0f) {
                state->reward = cell_value; // Negative reward
            }
        }
    }
}

__global__ void update_q_values_kernel(
    float* q_table,
    int* states_x,
    int* states_y,
    int* actions,
    float* rewards,
    int* next_states_x,
    int* next_states_y,
    bool* dones,
    int num_samples,
    int width,
    int height,
    int action_space_size,
    float learning_rate,
    float discount_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_samples) {
        int x = states_x[idx];
        int y = states_y[idx];
        int action = actions[idx];
        float reward = rewards[idx];
        int next_x = next_states_x[idx];
        int next_y = next_states_y[idx];
        bool done = dones[idx];
        
        // Calculate Q-table indices
        int q_idx = (y * width + x) * action_space_size + action;
        
        // Current Q-value
        float current_q = q_table[q_idx];
        
        // Find max Q-value for next state
        float max_next_q = 0.0f;
        
        if (!done) {
            for (int a = 0; a < action_space_size; a++) {
                int next_q_idx = (next_y * width + next_x) * action_space_size + a;
                float next_q = q_table[next_q_idx];
                max_next_q = max(max_next_q, next_q);
            }
        }
        
        // Q-learning update
        float target = reward + discount_factor * max_next_q;
        q_table[q_idx] += learning_rate * (target - current_q);
    }
}