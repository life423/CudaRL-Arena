#pragma once

#include "environment.h"
#include "../gpu/batch_kernels.cuh"
#include <vector>
#include <memory>

namespace cudarl {

/**
 * VectorizedEnvironment manages multiple environments in parallel on GPU
 * This enables efficient RL training with batch processing
 */
class VectorizedEnvironment {
public:
    // Constructor
    explicit VectorizedEnvironment(
        int num_envs = 8,
        int width = 10,
        int height = 10,
        float obstacle_density = 0.1f
    );
    
    // Destructor
    ~VectorizedEnvironment();
    
    // Disable copy
    VectorizedEnvironment(const VectorizedEnvironment&) = delete;
    VectorizedEnvironment& operator=(const VectorizedEnvironment&) = delete;
    
    // Allow move
    VectorizedEnvironment(VectorizedEnvironment&&) noexcept;
    VectorizedEnvironment& operator=(VectorizedEnvironment&&) noexcept;

    // ══════════════════════════════════════════════════════════════════════════════
    // CORE VECTORIZED OPERATIONS
    // ══════════════════════════════════════════════════════════════════════════════
    
    /**
     * Reset all environments to initial state
     * @return Vector of initial observations for all environments
     */
    std::vector<std::vector<float>> reset();
    
    /**
     * Step all environments with given actions
     * @param actions Vector of actions for each environment
     * @return Tuple of (observations, rewards, dones, infos)
     */
    std::tuple<
        std::vector<std::vector<float>>,  // observations
        std::vector<float>,               // rewards
        std::vector<bool>,                // dones
        std::vector<std::string>          // infos
    > step(const std::vector<int>& actions);
    
    /**
     * Step a single environment (for debugging or single-env use)
     * @param env_idx Environment index
     * @param action Action to take
     * @return Tuple of (observation, reward, done, info)
     */
    std::tuple<std::vector<float>, float, bool, std::string> 
    stepSingle(int env_idx, int action);
    
    /**
     * Reset a single environment
     * @param env_idx Environment index
     * @return Initial observation
     */
    std::vector<float> resetSingle(int env_idx);

    // ══════════════════════════════════════════════════════════════════════════════
    // CONFIGURATION AND UTILITIES
    // ══════════════════════════════════════════════════════════════════════════════
    
    /**
     * Regenerate obstacles for all environments
     * @param density Obstacle density (0.0 - 1.0)
     */
    void regenerateObstacles(float density = -1.0f);
    
    /**
     * Set custom reward parameters
     */
    void setRewardParameters(
        float step_penalty = -0.01f,
        float goal_reward = 1.0f,
        float obstacle_penalty = -0.5f,
        float trap_penalty = -0.2f
    );
    
    /**
     * Enable or disable advanced reward computation
     */
    void setAdvancedRewards(bool enabled) { m_useAdvancedRewards = enabled; }
    
    /**
     * Get environment configuration
     */
    int getNumEnvironments() const { return m_numEnvs; }
    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }
    float getObstacleDensity() const { return m_obstacleDensity; }

    // ══════════════════════════════════════════════════════════════════════════════
    // STATE ACCESS AND MONITORING
    // ══════════════════════════════════════════════════════════════════════════════
    
    /**
     * Get current states of all environments
     */
    std::vector<EnvironmentState> getStates() const;
    
    /**
     * Get state of a specific environment
     */
    EnvironmentState getState(int env_idx) const;
    
    /**
     * Get grid data for all environments
     */
    std::vector<std::vector<float>> getGrids() const;
    
    /**
     * Get grid data for a specific environment
     */
    std::vector<float> getGrid(int env_idx) const;
    
    /**
     * Get agent positions for all environments
     */
    std::vector<std::pair<int, int>> getAgentPositions() const;
    
    /**
     * Check if any environments are done
     */
    std::vector<bool> getDoneFlags() const;
    
    /**
     * Get rewards for all environments
     */
    std::vector<float> getRewards() const;

    // ══════════════════════════════════════════════════════════════════════════════
    // PERFORMANCE AND MONITORING
    // ══════════════════════════════════════════════════════════════════════════════
    
    /**
     * Get performance statistics
     */
    struct PerformanceStats {
        double avg_step_time_ms;
        double avg_reset_time_ms;
        size_t total_steps;
        size_t total_resets;
        double gpu_memory_usage_mb;
    };
    
    PerformanceStats getPerformanceStats() const;
    
    /**
     * Reset performance counters
     */
    void resetPerformanceCounters();

private:
    // Environment configuration
    int m_numEnvs;
    int m_width;
    int m_height;
    float m_obstacleDensity;
    
    // Reward parameters
    float m_stepPenalty;
    float m_goalReward;
    float m_obstaclePenalty;
    float m_trapPenalty;
    bool m_useAdvancedRewards;
    
    // Device memory management
    DeviceBuffer<EnvironmentState> m_deviceStates;
    DeviceBuffer<float> m_deviceGrids;
    DeviceBuffer<int> m_deviceActions;
    
    // Host memory for data transfer
    std::vector<EnvironmentState> m_hostStates;
    std::vector<float> m_hostGrids;
    std::vector<int> m_hostActions;
    
    // Performance tracking
    mutable size_t m_totalSteps;
    mutable size_t m_totalResets;
    mutable double m_totalStepTime;
    mutable double m_totalResetTime;
    
    // Helper methods
    void initializeDeviceMemory();
    void syncStatesFromDevice() const;
    void syncStatesToDevice();
    void syncGridsFromDevice() const;
    void updatePerformanceStats(double step_time, bool is_reset = false) const;
    
    // Grid indexing helper
    size_t getGridOffset(int env_idx) const {
        return static_cast<size_t>(env_idx) * m_width * m_height;
    }
};

} // namespace cudarl
