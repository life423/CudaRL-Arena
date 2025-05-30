#pragma once

#include "cuda_utils.h"
#include <vector>
#include <memory>
#include <utility>
#include <string>

namespace cudarl {

// Cell types for enhanced environment logic
enum class CellType : int {
    EMPTY = 0,        // Empty cell (value 0.0f)
    AGENT = 1,        // Agent position (value 0.5f)
    GOAL = 2,         // Goal position (value 1.0f)
    OBSTACLE = 3,     // Obstacle (value 0.9f)
    TRAP = 4,         // Trap/penalty zone (value 0.7f)
    REWARD_ZONE = 5   // Bonus reward zone (value 0.3f)
};

// Environment configuration structure
struct EnvironmentConfig {
    float obstacle_density = 0.1f;
    float trap_density = 0.05f;
    float reward_zone_density = 0.03f;
    float step_penalty = -0.01f;
    float goal_reward = 1.0f;
    float obstacle_penalty = -0.5f;
    float trap_penalty = -0.2f;
    float reward_zone_bonus = 0.1f;
    bool use_advanced_rewards = true;
    bool enable_distance_bonus = true;
    int max_episode_steps = 200;
};

// Enhanced environment state structure
struct EnvironmentState {
    int width;
    int height;
    float* grid;  // Flattened grid data
    int agent_x;
    int agent_y;
    float reward;
    bool done;
    
    // Enhanced state tracking
    int episode_steps;
    int total_rewards_collected;
    float cumulative_reward;
    bool goal_reached;
    int obstacles_hit;
    int traps_triggered;
    
    // Configuration
    EnvironmentConfig config;
};

// Environment class with RAII and modern C++ practices
class Environment {
public:
    // Constructor with enhanced configuration
    explicit Environment(
        int id = 0, 
        int width = 10, 
        int height = 10,
        const EnvironmentConfig& config = EnvironmentConfig{}
    );
    
    // Destructor with proper cleanup
    ~Environment();
    
    // Disable copy
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;
    
    // Allow move
    Environment(Environment&&) noexcept;
    Environment& operator=(Environment&&) noexcept;

    // ══════════════════════════════════════════════════════════════════════════════
    // CORE METHODS (Enhanced)
    // ══════════════════════════════════════════════════════════════════════════════
    
    void reset();
    void step(int action);
    
    // New enhanced methods
    void resetWithObstacles(float obstacle_density = -1.0f);
    bool isValidMove(int new_x, int new_y) const;
    CellType getCellType(int x, int y) const;
    void regenerateEnvironment();
    
    // ══════════════════════════════════════════════════════════════════════════════
    // CONFIGURATION METHODS
    // ══════════════════════════════════════════════════════════════════════════════
    
    void setEnvironmentConfig(const EnvironmentConfig& config);
    EnvironmentConfig getEnvironmentConfig() const { return m_state.config; }
    void setObstacleDensity(float density) { m_state.config.obstacle_density = density; }
    void setRewardParameters(float step_penalty, float goal_reward, float obstacle_penalty = -0.5f);
    void enableAdvancedRewards(bool enable) { m_state.config.use_advanced_rewards = enable; }
    
    // ══════════════════════════════════════════════════════════════════════════════
    // STATE GETTERS (Enhanced)
    // ══════════════════════════════════════════════════════════════════════════════
    
    int getWidth() const { return m_state.width; }
    int getHeight() const { return m_state.height; }
    int getAgentX() const { return m_state.agent_x; }
    int getAgentY() const { return m_state.agent_y; }
    float getReward() const { return m_state.reward; }
    bool isDone() const { return m_state.done; }
    
    // Enhanced state information
    int getEpisodeSteps() const { return m_state.episode_steps; }
    float getCumulativeReward() const { return m_state.cumulative_reward; }
    bool isGoalReached() const { return m_state.goal_reached; }
    int getObstaclesHit() const { return m_state.obstacles_hit; }
    int getTrapsTriggered() const { return m_state.traps_triggered; }
      // ══════════════════════════════════════════════════════════════════════════════
    // GRID ACCESS METHODS (Enhanced)
    // ══════════════════════════════════════════════════════════════════════════════
    
    float getCellValue(int x, int y) const;
    // Get the entire grid as a vector
    std::vector<float> getGrid() const;
    
    // Copy grid data directly to a host buffer (for Python bindings)
    void copyGridToBuffer(float* host_buffer, size_t num_elements) const;
    
    // Enhanced grid methods
    std::vector<std::pair<int, int>> getObstaclePositions() const;
    std::vector<std::pair<int, int>> getTrapPositions() const;
    std::vector<std::pair<int, int>> getRewardZonePositions() const;
    std::pair<int, int> getGoalPosition() const;
    std::pair<int, int> getAgentPosition() const { return {m_state.agent_x, m_state.agent_y}; }
    
    // ══════════════════════════════════════════════════════════════════════════════
    // UTILITY AND DEBUG METHODS
    // ══════════════════════════════════════════════════════════════════════════════
    
    void printGrid() const;
    void printStats() const;
    std::string getStateString() const;
    
    // Performance and analysis
    double calculateOptimalPathLength() const;
    bool isReachable(int target_x, int target_y) const;
    std::vector<std::pair<int, int>> getAdjacentCells(int x, int y) const;

private:
    int m_envId;
    EnvironmentState m_state;
    // Device memory
    DeviceBuffer<EnvironmentState> m_deviceState;
    DeviceBuffer<float> m_deviceGrid;
    
    // Helper methods (enhanced)
    void initializeGrid();
    void initializeGridOnDevice();
    void updateHostState();
    void syncToDevice();
    void syncStateToDevice();
    
    // New enhanced helper methods
    void placeObstacles();
    void placeTraps();
    void placeRewardZones();
    void validateConfiguration();
    void resetStatistics();
    CellType getDeviceCellType(int x, int y) const;
    float calculateDistanceBonus() const;
};

} // namespace cudarl