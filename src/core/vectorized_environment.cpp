#include "vectorized_environment.h"
#include "../gpu/batch_kernels.cuh"
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iostream>

namespace cudarl {

// ══════════════════════════════════════════════════════════════════════════════
// CONSTRUCTOR AND DESTRUCTOR
// ══════════════════════════════════════════════════════════════════════════════

VectorizedEnvironment::VectorizedEnvironment(
    int num_envs,
    int width,
    int height,
    float obstacle_density
)
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
    , m_deviceGrids(num_envs * width * height)
    , m_deviceActions(num_envs)
    , m_totalSteps(0)
    , m_totalResets(0)
    , m_totalStepTime(0.0)
    , m_totalResetTime(0.0)
{
    // Initialize host memory
    m_hostStates.resize(m_numEnvs);
    m_hostGrids.resize(m_numEnvs * m_width * m_height);
    m_hostActions.resize(m_numEnvs);
    
    // Initialize device memory and environments
    initializeDeviceMemory();
    
    std::cout << "VectorizedEnvironment created with " << m_numEnvs 
              << " environments (" << m_width << "x" << m_height << ")" << std::endl;
}

VectorizedEnvironment::~VectorizedEnvironment() {
    std::cout << "VectorizedEnvironment destroyed." << std::endl;
}

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
    , m_totalResetTime(other.m_totalResetTime)
{
    // Reset moved-from object
    other.m_numEnvs = 0;
}

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
        
        // Reset moved-from object
        other.m_numEnvs = 0;
    }
    return *this;
}

// ══════════════════════════════════════════════════════════════════════════════
// CORE VECTORIZED OPERATIONS
// ══════════════════════════════════════════════════════════════════════════════

std::vector<std::vector<float>> VectorizedEnvironment::reset() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Launch batch reset kernel
    launchBatchReset(
        m_deviceStates.get(),
        m_deviceGrids.get(),
        m_numEnvs,
        m_width,
        m_height
    );
    
    // Add obstacles if configured
    if (m_obstacleDensity > 0.0f) {
        regenerateObstacles(m_obstacleDensity);
    }
    
    // Synchronize and get observations
    CUDA_CHECK(cudaDeviceSynchronize());
    syncStatesFromDevice();
    syncGridsFromDevice();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double reset_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    updatePerformanceStats(reset_time, true);
    
    // Return observations for all environments
    std::vector<std::vector<float>> observations;
    observations.reserve(m_numEnvs);
    
    for (int i = 0; i < m_numEnvs; i++) {
        size_t grid_offset = getGridOffset(i);
        observations.emplace_back(
            m_hostGrids.begin() + grid_offset,
            m_hostGrids.begin() + grid_offset + m_width * m_height
        );
    }
    
    m_totalResets++;
    return observations;
}

std::tuple<
    std::vector<std::vector<float>>,
    std::vector<float>,
    std::vector<bool>,
    std::vector<std::string>
> VectorizedEnvironment::step(const std::vector<int>& actions) {
    if (actions.size() != static_cast<size_t>(m_numEnvs)) {
        throw std::invalid_argument("Number of actions must match number of environments");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Copy actions to device
    m_deviceActions.copyFromHost(actions.data(), actions.size());
    
    // Launch batch step kernel
    launchBatchStep(
        m_deviceStates.get(),
        m_deviceGrids.get(),
        m_deviceActions.get(),
        m_numEnvs,
        m_width,
        m_height
    );
    
    // Apply advanced rewards if enabled
    if (m_useAdvancedRewards) {
        computeAdvancedRewardsKernel<<<m_numEnvs, 1>>>(
            m_deviceStates.get(),
            m_deviceGrids.get(),
            m_numEnvs,
            m_width,
            m_height,
            m_stepPenalty,
            m_goalReward,
            m_obstaclePenalty,
            m_trapPenalty
        );
    }
    
    // Synchronize and get results
    CUDA_CHECK(cudaDeviceSynchronize());
    syncStatesFromDevice();
    syncGridsFromDevice();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double step_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    updatePerformanceStats(step_time, false);
    
    // Prepare return values
    std::vector<std::vector<float>> observations;
    std::vector<float> rewards;
    std::vector<bool> dones;
    std::vector<std::string> infos;
    
    observations.reserve(m_numEnvs);
    rewards.reserve(m_numEnvs);
    dones.reserve(m_numEnvs);
    infos.reserve(m_numEnvs);
    
    for (int i = 0; i < m_numEnvs; i++) {
        // Observation
        size_t grid_offset = getGridOffset(i);
        observations.emplace_back(
            m_hostGrids.begin() + grid_offset,
            m_hostGrids.begin() + grid_offset + m_width * m_height
        );
        
        // Reward, done, info
        const auto& state = m_hostStates[i];
        rewards.push_back(state.reward);
        dones.push_back(state.done);
        
        std::ostringstream info;
        info << "env_" << i << ":pos(" << state.agent_x << "," << state.agent_y 
             << "),reward=" << state.reward << ",done=" << (state.done ? "true" : "false");
        infos.push_back(info.str());
    }
    
    m_totalSteps++;
    return std::make_tuple(observations, rewards, dones, infos);
}

std::tuple<std::vector<float>, float, bool, std::string> 
VectorizedEnvironment::stepSingle(int env_idx, int action) {
    if (env_idx < 0 || env_idx >= m_numEnvs) {
        throw std::out_of_range("Environment index out of range");
    }
    
    // Create action vector with only one environment's action
    std::vector<int> actions(m_numEnvs, 0);
    actions[env_idx] = action;
    
    auto [observations, rewards, dones, infos] = step(actions);
    
    return std::make_tuple(
        observations[env_idx],
        rewards[env_idx],
        dones[env_idx],
        infos[env_idx]
    );
}

std::vector<float> VectorizedEnvironment::resetSingle(int env_idx) {
    if (env_idx < 0 || env_idx >= m_numEnvs) {
        throw std::out_of_range("Environment index out of range");
    }
    
    // For now, reset all environments (can be optimized later)
    auto observations = reset();
    return observations[env_idx];
}

// ══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION AND UTILITIES
// ══════════════════════════════════════════════════════════════════════════════

void VectorizedEnvironment::regenerateObstacles(float density) {
    if (density < 0.0f) {
        density = m_obstacleDensity;
    }
    
    // Sync current states to device (needed for obstacle placement)
    syncStatesToDevice();
    
    // Launch obstacle placement kernel
    launchPlaceObstacles(
        m_deviceGrids.get(),
        m_deviceStates.get(),
        m_numEnvs,
        m_width,
        m_height,
        density
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    m_obstacleDensity = density;
}

void VectorizedEnvironment::setRewardParameters(
    float step_penalty,
    float goal_reward,
    float obstacle_penalty,
    float trap_penalty
) {
    m_stepPenalty = step_penalty;
    m_goalReward = goal_reward;
    m_obstaclePenalty = obstacle_penalty;
    m_trapPenalty = trap_penalty;
}

// ══════════════════════════════════════════════════════════════════════════════
// STATE ACCESS AND MONITORING
// ══════════════════════════════════════════════════════════════════════════════

std::vector<EnvironmentState> VectorizedEnvironment::getStates() const {
    syncStatesFromDevice();
    return m_hostStates;
}

EnvironmentState VectorizedEnvironment::getState(int env_idx) const {
    if (env_idx < 0 || env_idx >= m_numEnvs) {
        throw std::out_of_range("Environment index out of range");
    }
    
    syncStatesFromDevice();
    return m_hostStates[env_idx];
}

std::vector<std::vector<float>> VectorizedEnvironment::getGrids() const {
    syncGridsFromDevice();
    
    std::vector<std::vector<float>> grids;
    grids.reserve(m_numEnvs);
    
    for (int i = 0; i < m_numEnvs; i++) {
        size_t grid_offset = getGridOffset(i);
        grids.emplace_back(
            m_hostGrids.begin() + grid_offset,
            m_hostGrids.begin() + grid_offset + m_width * m_height
        );
    }
    
    return grids;
}

std::vector<float> VectorizedEnvironment::getGrid(int env_idx) const {
    if (env_idx < 0 || env_idx >= m_numEnvs) {
        throw std::out_of_range("Environment index out of range");
    }
    
    syncGridsFromDevice();
    size_t grid_offset = getGridOffset(env_idx);
    
    return std::vector<float>(
        m_hostGrids.begin() + grid_offset,
        m_hostGrids.begin() + grid_offset + m_width * m_height
    );
}

std::vector<std::pair<int, int>> VectorizedEnvironment::getAgentPositions() const {
    syncStatesFromDevice();
    
    std::vector<std::pair<int, int>> positions;
    positions.reserve(m_numEnvs);
    
    for (const auto& state : m_hostStates) {
        positions.emplace_back(state.agent_x, state.agent_y);
    }
    
    return positions;
}

std::vector<bool> VectorizedEnvironment::getDoneFlags() const {
    syncStatesFromDevice();
    
    std::vector<bool> dones;
    dones.reserve(m_numEnvs);
    
    for (const auto& state : m_hostStates) {
        dones.push_back(state.done);
    }
    
    return dones;
}

std::vector<float> VectorizedEnvironment::getRewards() const {
    syncStatesFromDevice();
    
    std::vector<float> rewards;
    rewards.reserve(m_numEnvs);
    
    for (const auto& state : m_hostStates) {
        rewards.push_back(state.reward);
    }
    
    return rewards;
}

// ══════════════════════════════════════════════════════════════════════════════
// PERFORMANCE AND MONITORING
// ══════════════════════════════════════════════════════════════════════════════

VectorizedEnvironment::PerformanceStats VectorizedEnvironment::getPerformanceStats() const {
    PerformanceStats stats;
    
    stats.total_steps = m_totalSteps;
    stats.total_resets = m_totalResets;
    
    stats.avg_step_time_ms = (m_totalSteps > 0) ? 
        (m_totalStepTime / m_totalSteps) : 0.0;
    stats.avg_reset_time_ms = (m_totalResets > 0) ? 
        (m_totalResetTime / m_totalResets) : 0.0;
    
    // Calculate GPU memory usage
    size_t state_memory = m_numEnvs * sizeof(EnvironmentState);
    size_t grid_memory = m_numEnvs * m_width * m_height * sizeof(float);
    size_t action_memory = m_numEnvs * sizeof(int);
    size_t total_memory = state_memory + grid_memory + action_memory;
    
    stats.gpu_memory_usage_mb = static_cast<double>(total_memory) / (1024.0 * 1024.0);
    
    return stats;
}

void VectorizedEnvironment::resetPerformanceCounters() {
    m_totalSteps = 0;
    m_totalResets = 0;
    m_totalStepTime = 0.0;
    m_totalResetTime = 0.0;
}

// ══════════════════════════════════════════════════════════════════════════════
// PRIVATE HELPER METHODS
// ══════════════════════════════════════════════════════════════════════════════

void VectorizedEnvironment::initializeDeviceMemory() {
    // Initialize host states
    for (int i = 0; i < m_numEnvs; i++) {
        EnvironmentState& state = m_hostStates[i];
        state.width = m_width;
        state.height = m_height;
        state.agent_x = m_width / 2;
        state.agent_y = m_height / 2;
        state.reward = 0.0f;
        state.done = false;
        state.grid = nullptr; // Will be set on device
    }
    
    // Initialize host grids with zeros
    std::fill(m_hostGrids.begin(), m_hostGrids.end(), 0.0f);
    
    // Copy initial data to device
    syncStatesToDevice();
    m_deviceGrids.copyFromHost(m_hostGrids.data(), m_hostGrids.size());
}

void VectorizedEnvironment::syncStatesFromDevice() const {
    m_deviceStates.copyToHost(m_hostStates.data(), m_hostStates.size());
}

void VectorizedEnvironment::syncStatesToDevice() {
    m_deviceStates.copyFromHost(m_hostStates.data(), m_hostStates.size());
}

void VectorizedEnvironment::syncGridsFromDevice() const {
    m_deviceGrids.copyToHost(m_hostGrids.data(), m_hostGrids.size());
}

void VectorizedEnvironment::updatePerformanceStats(double time_ms, bool is_reset) const {
    if (is_reset) {
        m_totalResetTime += time_ms;
    } else {
        m_totalStepTime += time_ms;
    }
}

} // namespace cudarl
