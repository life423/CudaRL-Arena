#include "vectorized_environment.h"
#include "cuda_utils.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <random>
#include <tuple>

namespace cudarl {

// Constructor
VectorizedEnvironment::VectorizedEnvironment(
    int num_envs, 
    int width, 
    int height, 
    float obstacle_density)
    : m_numEnvs(num_envs)
    , m_width(width)
    , m_height(height)
    , m_obstacleDensity(obstacle_density)
    , m_stepPenalty(-0.01f)
    , m_goalReward(1.0f)
    , m_obstaclePenalty(-0.5f)
    , m_trapPenalty(-0.2f)
    , m_useAdvancedRewards(true)
    , m_deviceStates(num_envs)
    , m_deviceGrids(static_cast<size_t>(num_envs) * width * height)
    , m_deviceActions(num_envs)
    , m_hostStates(num_envs)
    , m_hostGrids(static_cast<size_t>(num_envs) * width * height)
    , m_hostActions(num_envs)
    , m_totalSteps(0)
    , m_totalResets(0)
    , m_totalStepTime(0.0)
    , m_totalResetTime(0.0) {
    
    // Initialize device memory
    initializeDeviceMemory();
    
    // Initialize all environments
    reset();
}

// Destructor
VectorizedEnvironment::~VectorizedEnvironment() = default;

// Move constructor
VectorizedEnvironment::VectorizedEnvironment(VectorizedEnvironment&& other) noexcept
    : m_numEnvs(other.m_numEnvs)
    , m_width(other.m_width)
    , m_height(other.m_height)
    , m_obstacleDensity(other.m_obstacleDensity)
    , m_stepPenalty(other.m_stepPenalty)
    , m_goalReward(other.m_goalReward)
    , m_obstaclePenalty(other.m_obstaclePenalty)
    , m_trapPenalty(other.m_trapPenalty)
    , m_useAdvancedRewards(other.m_useAdvancedRewards)
    , m_deviceStates(std::move(other.m_deviceStates))
    , m_deviceGrids(std::move(other.m_deviceGrids))
    , m_deviceActions(std::move(other.m_deviceActions))
    , m_hostStates(std::move(other.m_hostStates))
    , m_hostGrids(std::move(other.m_hostGrids))
    , m_hostActions(std::move(other.m_hostActions))
    , m_totalSteps(other.m_totalSteps)
    , m_totalResets(other.m_totalResets)
    , m_totalStepTime(other.m_totalStepTime)
    , m_totalResetTime(other.m_totalResetTime) {
}

// Move assignment
VectorizedEnvironment& VectorizedEnvironment::operator=(VectorizedEnvironment&& other) noexcept {
    if (this != &other) {
        m_numEnvs = other.m_numEnvs;
        m_width = other.m_width;
        m_height = other.m_height;
        m_obstacleDensity = other.m_obstacleDensity;
        m_stepPenalty = other.m_stepPenalty;
        m_goalReward = other.m_goalReward;
        m_obstaclePenalty = other.m_obstaclePenalty;
        m_trapPenalty = other.m_trapPenalty;
        m_useAdvancedRewards = other.m_useAdvancedRewards;
        m_deviceStates = std::move(other.m_deviceStates);
        m_deviceGrids = std::move(other.m_deviceGrids);
        m_deviceActions = std::move(other.m_deviceActions);
        m_hostStates = std::move(other.m_hostStates);
        m_hostGrids = std::move(other.m_hostGrids);
        m_hostActions = std::move(other.m_hostActions);
        m_totalSteps = other.m_totalSteps;
        m_totalResets = other.m_totalResets;
        m_totalStepTime = other.m_totalStepTime;
        m_totalResetTime = other.m_totalResetTime;
    }
    return *this;
}

// Reset all environments
std::vector<std::vector<float>> VectorizedEnvironment::reset() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize all environment states
    for (int i = 0; i < m_numEnvs; i++) {
        EnvironmentState& state = m_hostStates[i];
        state.width = m_width;
        state.height = m_height;
        state.agent_x = i % m_width; // Simple initial positioning
        state.agent_y = i / m_width % m_height;
        state.reward = 0.0f;
        state.done = false;
        state.episode_steps = 0;
        state.total_rewards_collected = 0;
        state.cumulative_reward = 0.0f;
        state.goal_reached = false;
        state.obstacles_hit = 0;
        state.traps_triggered = 0;
        
        // Set default config
        state.config.obstacle_density = m_obstacleDensity;
        state.config.step_penalty = m_stepPenalty;
        state.config.goal_reward = m_goalReward;
        state.config.obstacle_penalty = m_obstaclePenalty;
        state.config.trap_penalty = m_trapPenalty;
        state.config.use_advanced_rewards = m_useAdvancedRewards;
        state.config.max_episode_steps = 200;
        
        // Initialize grid for this environment
        size_t grid_offset = getGridOffset(i);
        float* env_grid = &m_hostGrids[grid_offset];
        
        // Fill with empty cells
        std::fill(env_grid, env_grid + m_width * m_height, 0.0f);
        
        // Place goal at bottom-right
        env_grid[(m_height - 1) * m_width + (m_width - 1)] = 1.0f;
        
        // Place some obstacles randomly
        std::random_device rd;
        std::mt19937 gen(rd() + i); // Different seed per environment
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for (int y = 0; y < m_height; y++) {
            for (int x = 0; x < m_width; x++) {
                if (env_grid[y * m_width + x] == 0.0f && dis(gen) < m_obstacleDensity) {
                    env_grid[y * m_width + x] = 0.9f; // Obstacle
                }
            }
        }
        
        state.grid = env_grid; // Point to the grid section
    }
    
    // Sync to device
    syncStatesToDevice();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    updatePerformanceStats(duration.count() / 1000.0, true);
    
    // Return observations
    return getGrids();
}

// Step all environments
std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<bool>, std::vector<std::string>>
VectorizedEnvironment::step(const std::vector<int>& actions) {
    assert(actions.size() == static_cast<size_t>(m_numEnvs));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Copy actions to device buffer
    std::copy(actions.begin(), actions.end(), m_hostActions.begin());
    
    // Process each environment
    for (int i = 0; i < m_numEnvs; i++) {
        if (m_hostStates[i].done) continue;
        
        EnvironmentState& state = m_hostStates[i];
        int action = actions[i];
        
        int new_x = state.agent_x;
        int new_y = state.agent_y;
        
        // Convert action to movement
        switch (action) {
            case 0: new_y--; break; // UP
            case 1: new_y++; break; // DOWN
            case 2: new_x--; break; // LEFT
            case 3: new_x++; break; // RIGHT
        }
        
        state.reward = state.config.step_penalty;
        
        // Check bounds and move
        if (new_x >= 0 && new_x < m_width && new_y >= 0 && new_y < m_height) {
            state.agent_x = new_x;
            state.agent_y = new_y;
            
            // Check cell type at new position
            size_t grid_offset = getGridOffset(i);
            float cell_value = m_hostGrids[grid_offset + new_y * m_width + new_x];
            
            if (std::abs(cell_value - 1.0f) < 0.01f) { // Goal
                state.reward = state.config.goal_reward;
                state.done = true;
                state.goal_reached = true;
            } else if (std::abs(cell_value - 0.9f) < 0.01f) { // Obstacle
                state.reward = state.config.obstacle_penalty;
                state.obstacles_hit++;
            } else if (std::abs(cell_value - 0.7f) < 0.01f) { // Trap
                state.reward = state.config.trap_penalty;
                state.traps_triggered++;
            }
        } else {
            // Out of bounds - treat as obstacle
            state.reward = state.config.obstacle_penalty;
        }
        
        state.episode_steps++;
        state.cumulative_reward += state.reward;
        
        if (state.episode_steps >= state.config.max_episode_steps) {
            state.done = true;
        }
    }
    
    // Sync to device
    syncStatesToDevice();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    updatePerformanceStats(duration.count() / 1000.0, false);
    
    // Prepare return values
    auto observations = getGrids();
    std::vector<float> rewards;
    std::vector<bool> dones;
    std::vector<std::string> infos;
    
    for (int i = 0; i < m_numEnvs; i++) {
        rewards.push_back(m_hostStates[i].reward);
        dones.push_back(m_hostStates[i].done);
        infos.push_back("env_" + std::to_string(i) + "_steps_" + std::to_string(m_hostStates[i].episode_steps));
    }
    
    return std::make_tuple(observations, rewards, dones, infos);
}

// Step single environment
std::tuple<std::vector<float>, float, bool, std::string> 
VectorizedEnvironment::stepSingle(int env_idx, int action) {
    assert(env_idx >= 0 && env_idx < m_numEnvs);
    
    std::vector<int> actions(m_numEnvs, 0);
    actions[env_idx] = action;
    
    auto [observations, rewards, dones, infos] = step(actions);
    
    return std::make_tuple(observations[env_idx], rewards[env_idx], dones[env_idx], infos[env_idx]);
}

// Reset single environment
std::vector<float> VectorizedEnvironment::resetSingle(int env_idx) {
    assert(env_idx >= 0 && env_idx < m_numEnvs);
    
    // Reset just this environment
    EnvironmentState& state = m_hostStates[env_idx];
    state.agent_x = env_idx % m_width;
    state.agent_y = env_idx / m_width % m_height;
    state.reward = 0.0f;
    state.done = false;
    state.episode_steps = 0;
    state.cumulative_reward = 0.0f;
    state.goal_reached = false;
    state.obstacles_hit = 0;
    state.traps_triggered = 0;
    
    return getGrid(env_idx);
}

// Regenerate obstacles
void VectorizedEnvironment::regenerateObstacles(float density) {
    if (density >= 0.0f) {
        m_obstacleDensity = density;
    }
    
    // Regenerate for all environments
    for (int i = 0; i < m_numEnvs; i++) {
        size_t grid_offset = getGridOffset(i);
        float* env_grid = &m_hostGrids[grid_offset];
        
        // Clear existing obstacles but keep goal
        for (int y = 0; y < m_height; y++) {
            for (int x = 0; x < m_width; x++) {
                if (x == m_width - 1 && y == m_height - 1) {
                    env_grid[y * m_width + x] = 1.0f; // Keep goal
                } else {
                    env_grid[y * m_width + x] = 0.0f; // Clear
                }
            }
        }
        
        // Place new obstacles
        std::random_device rd;
        std::mt19937 gen(rd() + i);
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for (int y = 0; y < m_height; y++) {
            for (int x = 0; x < m_width; x++) {
                if (env_grid[y * m_width + x] == 0.0f && dis(gen) < m_obstacleDensity) {
                    env_grid[y * m_width + x] = 0.9f; // Obstacle
                }
            }
        }
    }
    
    syncStatesToDevice();
}

// Set reward parameters
void VectorizedEnvironment::setRewardParameters(
    float step_penalty, 
    float goal_reward, 
    float obstacle_penalty, 
    float trap_penalty) {
    
    m_stepPenalty = step_penalty;
    m_goalReward = goal_reward;
    m_obstaclePenalty = obstacle_penalty;
    m_trapPenalty = trap_penalty;
    
    // Update all environment configs
    for (int i = 0; i < m_numEnvs; i++) {
        m_hostStates[i].config.step_penalty = step_penalty;
        m_hostStates[i].config.goal_reward = goal_reward;
        m_hostStates[i].config.obstacle_penalty = obstacle_penalty;
        m_hostStates[i].config.trap_penalty = trap_penalty;
    }
}

// Get states
std::vector<EnvironmentState> VectorizedEnvironment::getStates() const {
    return m_hostStates;
}

EnvironmentState VectorizedEnvironment::getState(int env_idx) const {
    assert(env_idx >= 0 && env_idx < m_numEnvs);
    return m_hostStates[env_idx];
}

// Get grids
std::vector<std::vector<float>> VectorizedEnvironment::getGrids() const {
    std::vector<std::vector<float>> grids;
    grids.reserve(m_numEnvs);
    
    for (int i = 0; i < m_numEnvs; i++) {
        grids.push_back(getGrid(i));
    }
    
    return grids;
}

std::vector<float> VectorizedEnvironment::getGrid(int env_idx) const {
    assert(env_idx >= 0 && env_idx < m_numEnvs);
    
    size_t grid_offset = getGridOffset(env_idx);
    const float* env_grid = &m_hostGrids[grid_offset];
    
    return std::vector<float>(env_grid, env_grid + m_width * m_height);
}

// Get agent positions
std::vector<std::pair<int, int>> VectorizedEnvironment::getAgentPositions() const {
    std::vector<std::pair<int, int>> positions;
    positions.reserve(m_numEnvs);
    
    for (int i = 0; i < m_numEnvs; i++) {
        positions.emplace_back(m_hostStates[i].agent_x, m_hostStates[i].agent_y);
    }
    
    return positions;
}

// Get done flags
std::vector<bool> VectorizedEnvironment::getDoneFlags() const {
    std::vector<bool> dones;
    dones.reserve(m_numEnvs);
    
    for (int i = 0; i < m_numEnvs; i++) {
        dones.push_back(m_hostStates[i].done);
    }
    
    return dones;
}

// Get rewards
std::vector<float> VectorizedEnvironment::getRewards() const {
    std::vector<float> rewards;
    rewards.reserve(m_numEnvs);
    
    for (int i = 0; i < m_numEnvs; i++) {
        rewards.push_back(m_hostStates[i].reward);
    }
    
    return rewards;
}

// Get performance stats
VectorizedEnvironment::PerformanceStats VectorizedEnvironment::getPerformanceStats() const {
    PerformanceStats stats;
    
    if (m_totalSteps > 0) {
        stats.avg_step_time_ms = m_totalStepTime / m_totalSteps;
    } else {
        stats.avg_step_time_ms = 0.0;
    }
    
    if (m_totalResets > 0) {
        stats.avg_reset_time_ms = m_totalResetTime / m_totalResets;
    } else {
        stats.avg_reset_time_ms = 0.0;
    }
    
    stats.total_steps = m_totalSteps;
    stats.total_resets = m_totalResets;
    
    // Estimate GPU memory usage (simplified)
    size_t state_memory = m_numEnvs * sizeof(EnvironmentState);
    size_t grid_memory = static_cast<size_t>(m_numEnvs) * m_width * m_height * sizeof(float);
    size_t action_memory = m_numEnvs * sizeof(int);
    stats.gpu_memory_usage_mb = (state_memory + grid_memory + action_memory) / (1024.0 * 1024.0);
    
    return stats;
}

// Reset performance counters
void VectorizedEnvironment::resetPerformanceCounters() {
    m_totalSteps = 0;
    m_totalResets = 0;
    m_totalStepTime = 0.0;
    m_totalResetTime = 0.0;
}

// Private helper methods
void VectorizedEnvironment::initializeDeviceMemory() {
    // Resize device buffers
    m_deviceStates.resize(m_numEnvs);
    m_deviceGrids.resize(static_cast<size_t>(m_numEnvs) * m_width * m_height);
    m_deviceActions.resize(m_numEnvs);
    
    // Resize host buffers
    m_hostStates.resize(m_numEnvs);
    m_hostGrids.resize(static_cast<size_t>(m_numEnvs) * m_width * m_height);
    m_hostActions.resize(m_numEnvs);
}

void VectorizedEnvironment::syncStatesFromDevice() const {
    // In a real CUDA implementation, this would copy from device to host
    // For now, it's a no-op since we're using host memory as stub
}

void VectorizedEnvironment::syncStatesToDevice() {
    // In a real CUDA implementation, this would copy from host to device
    // For now, it's a no-op since we're using host memory as stub
}

void VectorizedEnvironment::syncGridsFromDevice() const {
    // In a real CUDA implementation, this would copy grids from device to host
    // For now, it's a no-op since we're using host memory as stub
}

void VectorizedEnvironment::updatePerformanceStats(double time_ms, bool is_reset) const {
    if (is_reset) {
        m_totalResets++;
        m_totalResetTime += time_ms;
    } else {
        m_totalSteps++;
        m_totalStepTime += time_ms;
    }
}

} // namespace cudarl
