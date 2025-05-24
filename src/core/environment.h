#pragma once

#include "cuda_utils.h"
#include <vector>
#include <memory>

namespace cudarl {

// Environment state structure
struct EnvironmentState {
    int width;
    int height;
    float* grid;  // Flattened grid data
    int agent_x;
    int agent_y;
    float reward;
    bool done;
};

// Environment class with RAII and modern C++ practices
class Environment {
public:
    // Constructor with default values
    explicit Environment(int id = 0, int width = 10, int height = 10);
    
    // Destructor with proper cleanup
    ~Environment();
    
    // Disable copy
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;
    
    // Allow move
    Environment(Environment&&) noexcept;
    Environment& operator=(Environment&&) noexcept;

    // Core methods
    void reset();
    void step(int action);
    
    // Getters
    int getWidth() const { return m_state.width; }
    int getHeight() const { return m_state.height; }
    int getAgentX() const { return m_state.agent_x; }
    int getAgentY() const { return m_state.agent_y; }
    float getReward() const { return m_state.reward; }
    bool isDone() const { return m_state.done; }
    
    // Get grid cell value at (x,y)
    float getCellValue(int x, int y) const;
    
    // Get the entire grid as a vector
    std::vector<float> getGrid() const;

private:
    int m_envId;
    EnvironmentState m_state;
    
    // Device memory
    CudaMemory<EnvironmentState> m_deviceState;
    CudaMemory<float> m_deviceGrid;
    
    // Helper methods
    void initializeGrid();
    void updateHostState();
    void syncToDevice();
};

} // namespace cudarl