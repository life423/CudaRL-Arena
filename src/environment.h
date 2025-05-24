#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Simple struct for environment state
struct EnvironmentState {
    int width;
    int height;
    float* grid;  // Flattened grid data
    int agent_x;
    int agent_y;
    float reward;
    bool done;
};

class Environment {
public:
    Environment(int id = 0, int width = 10, int height = 10);
    ~Environment();

    void reset();
    void step(int action);
    
    // Getters for Godot integration
    int getWidth() const { return state.width; }
    int getHeight() const { return state.height; }
    int getAgentX() const { return state.agent_x; }
    int getAgentY() const { return state.agent_y; }
    float getReward() const { return state.reward; }
    bool isDone() const { return state.done; }
    
    // Get grid cell value at (x,y)
    float getCellValue(int x, int y) const;
    
    // Get the entire grid as a vector
    std::vector<float> getGrid() const;

private:
    int env_id;
    EnvironmentState state;
    EnvironmentState* d_state; // Device pointer for state
    
    // Helper methods
    void initializeGrid();
    void updateHostState();
};