#include "environment.h"
#include "../gpu/batch_kernels.cuh"
#include <algorithm>
#include <sstream>
#include <iostream>
#include <queue>
#include <cmath>

namespace cudarl {

// ══════════════════════════════════════════════════════════════════════════════
// ENHANCED ENVIRONMENT IMPLEMENTATION
// ══════════════════════════════════════════════════════════════════════════════

Environment::Environment(int id, int width, int height, const EnvironmentConfig& config)
    : m_envId(id)
    , m_deviceState(1)
    , m_deviceGrid(width * height)
{
    // Initialize enhanced state
    m_state.width = width;
    m_state.height = height;
    m_state.grid = nullptr; // Will be managed by device memory
    m_state.config = config;
    
    // Validate configuration
    validateConfiguration();
    
    // Reset state and initialize
    resetStatistics();
    initializeGrid();
    syncToDevice();
    
    std::cout << "Enhanced Environment " << m_envId << " created (" 
              << width << "x" << height << ") with obstacles: " 
              << (config.obstacle_density * 100) << "%" << std::endl;
}

Environment::~Environment() {
    // RAII cleanup handled by DeviceBuffer destructors
}

Environment::Environment(Environment&& other) noexcept
    : m_envId(other.m_envId)
    , m_state(other.m_state)
    , m_deviceState(std::move(other.m_deviceState))
    , m_deviceGrid(std::move(other.m_deviceGrid))
{
    // Reset moved-from object
    other.m_state = {};
    other.m_envId = -1;
}

Environment& Environment::operator=(Environment&& other) noexcept {
    if (this != &other) {
        m_envId = other.m_envId;
        m_state = other.m_state;
        m_deviceState = std::move(other.m_deviceState);
        m_deviceGrid = std::move(other.m_deviceGrid);
        
        // Reset moved-from object
        other.m_state = {};
        other.m_envId = -1;
    }
    return *this;
}

// ══════════════════════════════════════════════════════════════════════════════
// CORE METHODS (Enhanced)
// ══════════════════════════════════════════════════════════════════════════════

void Environment::reset() {
    resetStatistics();
    
    // Reset agent position
    m_state.agent_x = m_state.width / 2;
    m_state.agent_y = m_state.height / 2;
    m_state.reward = 0.0f;
    m_state.done = false;
    m_state.episode_steps = 0;
    
    // Reinitialize grid with obstacles
    initializeGrid();
    
    // Sync to device and launch reset kernel
    syncToDevice();
    
    // Use enhanced kernels for resetting
    launchBatchReset(
        m_deviceState.get(),
        m_deviceGrid.get(),
        1, // Single environment
        m_state.width,
        m_state.height
    );
    
    // Update host state
    updateHostState();
    
    std::cout << "Enhanced Environment " << m_envId << " reset with " 
              << getObstaclePositions().size() << " obstacles." << std::endl;
}

void Environment::step(int action) {
    if (m_state.done) {
        return; // Episode already finished
    }
    
    m_state.episode_steps++;
    
    // Check episode length limit
    if (m_state.episode_steps >= m_state.config.max_episode_steps) {
        m_state.done = true;
        m_state.reward = m_state.config.step_penalty * 2; // Extra penalty for timeout
        return;
    }
    
    // Store previous position for validation
    int prev_x = m_state.agent_x;
    int prev_y = m_state.agent_y;
    
    // Calculate new position
    int new_x = m_state.agent_x;
    int new_y = m_state.agent_y;
    
    switch (action) {
        case 0: new_y--; break; // up
        case 1: new_x++; break; // right
        case 2: new_y++; break; // down
        case 3: new_x--; break; // left
        default: break;         // invalid action, no movement
    }
    
    // Validate move
    if (isValidMove(new_x, new_y)) {
        m_state.agent_x = new_x;
        m_state.agent_y = new_y;
        
        // Calculate reward based on cell type
        CellType cell_type = getCellType(new_x, new_y);
        calculateRewardForCellType(cell_type);
        
        // Add distance bonus if enabled
        if (m_state.config.enable_distance_bonus) {
            m_state.reward += calculateDistanceBonus();
        }
    } else {
        // Invalid move - apply penalty
        m_state.reward = m_state.config.obstacle_penalty;
        
        // Check if it was hitting an obstacle specifically
        if (new_x >= 0 && new_x < m_state.width && new_y >= 0 && new_y < m_state.height) {
            CellType cell_type = getCellType(new_x, new_y);
            if (cell_type == CellType::OBSTACLE) {
                m_state.obstacles_hit++;
            }
        }
    }
    
    // Update cumulative reward
    m_state.cumulative_reward += m_state.reward;
    
    // Sync changes to device
    syncToDevice();
    
    // Launch enhanced step kernel for consistency
    std::vector<int> actions = {action};
    launchBatchStep(
        m_deviceState.get(),
        m_deviceGrid.get(),
        actions.data(),
        1, // Single environment
        m_state.width,
        m_state.height
    );
    
    // Update host state
    updateHostState();
}

void Environment::resetWithObstacles(float obstacle_density) {
    if (obstacle_density >= 0.0f) {
        m_state.config.obstacle_density = obstacle_density;
    }
    reset();
}

bool Environment::isValidMove(int new_x, int new_y) const {
    // Check bounds
    if (new_x < 0 || new_x >= m_state.width || new_y < 0 || new_y >= m_state.height) {
        return false;
    }
    
    // Check for obstacles
    CellType cell_type = getCellType(new_x, new_y);
    return cell_type != CellType::OBSTACLE;
}

CellType Environment::getCellType(int x, int y) const {
    if (x < 0 || x >= m_state.width || y < 0 || y >= m_state.height) {
        return CellType::EMPTY;
    }
    
    float cell_value = getCellValue(x, y);
    
    if (std::abs(cell_value - 1.0f) < 0.01f) return CellType::GOAL;
    if (std::abs(cell_value - 0.9f) < 0.01f) return CellType::OBSTACLE;
    if (std::abs(cell_value - 0.7f) < 0.01f) return CellType::TRAP;
    if (std::abs(cell_value - 0.5f) < 0.01f) return CellType::AGENT;
    if (std::abs(cell_value - 0.3f) < 0.01f) return CellType::REWARD_ZONE;
    
    return CellType::EMPTY;
}

void Environment::regenerateEnvironment() {
    initializeGrid();
    syncToDevice();
    updateHostState();
}

// ══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION METHODS
// ══════════════════════════════════════════════════════════════════════════════

void Environment::setEnvironmentConfig(const EnvironmentConfig& config) {
    m_state.config = config;
    validateConfiguration();
}

void Environment::setRewardParameters(float step_penalty, float goal_reward, float obstacle_penalty) {
    m_state.config.step_penalty = step_penalty;
    m_state.config.goal_reward = goal_reward;
    m_state.config.obstacle_penalty = obstacle_penalty;
}

// ══════════════════════════════════════════════════════════════════════════════
// ENHANCED GRID METHODS
// ══════════════════════════════════════════════════════════════════════════════

std::vector<std::pair<int, int>> Environment::getObstaclePositions() const {
    std::vector<std::pair<int, int>> obstacles;
    std::vector<float> grid = getGrid();
    
    for (int y = 0; y < m_state.height; y++) {
        for (int x = 0; x < m_state.width; x++) {
            if (std::abs(grid[y * m_state.width + x] - 0.9f) < 0.01f) {
                obstacles.emplace_back(x, y);
            }
        }
    }
    
    return obstacles;
}

std::vector<std::pair<int, int>> Environment::getTrapPositions() const {
    std::vector<std::pair<int, int>> traps;
    std::vector<float> grid = getGrid();
    
    for (int y = 0; y < m_state.height; y++) {
        for (int x = 0; x < m_state.width; x++) {
            if (std::abs(grid[y * m_state.width + x] - 0.7f) < 0.01f) {
                traps.emplace_back(x, y);
            }
        }
    }
    
    return traps;
}

std::vector<std::pair<int, int>> Environment::getRewardZonePositions() const {
    std::vector<std::pair<int, int>> reward_zones;
    std::vector<float> grid = getGrid();
    
    for (int y = 0; y < m_state.height; y++) {
        for (int x = 0; x < m_state.width; x++) {
            if (std::abs(grid[y * m_state.width + x] - 0.3f) < 0.01f) {
                reward_zones.emplace_back(x, y);
            }
        }
    }
    
    return reward_zones;
}

std::pair<int, int> Environment::getGoalPosition() const {
    return std::make_pair(m_state.width - 1, m_state.height - 1);
}

// ══════════════════════════════════════════════════════════════════════════════
// UTILITY AND DEBUG METHODS
// ══════════════════════════════════════════════════════════════════════════════

void Environment::printGrid() const {
    std::vector<float> grid = getGrid();
    std::cout << "\n=== Environment " << m_envId << " Grid ===" << std::endl;
    
    for (int y = 0; y < m_state.height; y++) {
        for (int x = 0; x < m_state.width; x++) {
            float value = grid[y * m_state.width + x];
            
            if (x == m_state.agent_x && y == m_state.agent_y) {
                std::cout << "A ";
            } else if (std::abs(value - 1.0f) < 0.01f) {
                std::cout << "G ";
            } else if (std::abs(value - 0.9f) < 0.01f) {
                std::cout << "# ";
            } else if (std::abs(value - 0.7f) < 0.01f) {
                std::cout << "T ";
            } else if (std::abs(value - 0.3f) < 0.01f) {
                std::cout << "+ ";
            } else {
                std::cout << ". ";
            }
        }
        std::cout << std::endl;
    }
    
    std::cout << "Legend: A=Agent, G=Goal, #=Obstacle, T=Trap, +=Reward Zone, .=Empty" << std::endl;
}

void Environment::printStats() const {
    std::cout << "\n=== Environment " << m_envId << " Statistics ===" << std::endl;
    std::cout << "Agent Position: (" << m_state.agent_x << ", " << m_state.agent_y << ")" << std::endl;
    std::cout << "Episode Steps: " << m_state.episode_steps << "/" << m_state.config.max_episode_steps << std::endl;
    std::cout << "Current Reward: " << m_state.reward << std::endl;
    std::cout << "Cumulative Reward: " << m_state.cumulative_reward << std::endl;
    std::cout << "Goal Reached: " << (m_state.goal_reached ? "Yes" : "No") << std::endl;
    std::cout << "Obstacles Hit: " << m_state.obstacles_hit << std::endl;
    std::cout << "Traps Triggered: " << m_state.traps_triggered << std::endl;
    std::cout << "Done: " << (m_state.done ? "Yes" : "No") << std::endl;
    std::cout << "Optimal Path Length: " << calculateOptimalPathLength() << std::endl;
}

std::string Environment::getStateString() const {
    std::ostringstream oss;
    oss << "Env" << m_envId << ":(" << m_state.agent_x << "," << m_state.agent_y 
        << "),steps:" << m_state.episode_steps 
        << ",reward:" << m_state.reward 
        << ",cumulative:" << m_state.cumulative_reward
        << ",done:" << (m_state.done ? "true" : "false");
    return oss.str();
}

double Environment::calculateOptimalPathLength() const {
    // Simple Manhattan distance to goal
    int goal_x = m_state.width - 1;
    int goal_y = m_state.height - 1;
    return std::abs(m_state.agent_x - goal_x) + std::abs(m_state.agent_y - goal_y);
}

bool Environment::isReachable(int target_x, int target_y) const {
    if (target_x < 0 || target_x >= m_state.width || target_y < 0 || target_y >= m_state.height) {
        return false;
    }
    
    // Simple BFS to check reachability
    std::vector<std::vector<bool>> visited(m_state.height, std::vector<bool>(m_state.width, false));
    std::queue<std::pair<int, int>> queue;
    
    queue.push({m_state.agent_x, m_state.agent_y});
    visited[m_state.agent_y][m_state.agent_x] = true;
    
    while (!queue.empty()) {
        auto [x, y] = queue.front();
        queue.pop();
        
        if (x == target_x && y == target_y) {
            return true;
        }
        
        // Check all 4 directions
        int dx[] = {0, 1, 0, -1};
        int dy[] = {-1, 0, 1, 0};
        
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            
            if (nx >= 0 && nx < m_state.width && ny >= 0 && ny < m_state.height &&
                !visited[ny][nx] && getCellType(nx, ny) != CellType::OBSTACLE) {
                visited[ny][nx] = true;
                queue.push({nx, ny});
            }
        }
    }
    
    return false;
}

std::vector<std::pair<int, int>> Environment::getAdjacentCells(int x, int y) const {
    std::vector<std::pair<int, int>> adjacent;
    int dx[] = {0, 1, 0, -1};
    int dy[] = {-1, 0, 1, 0};
    
    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        
        if (nx >= 0 && nx < m_state.width && ny >= 0 && ny < m_state.height) {
            adjacent.emplace_back(nx, ny);
        }
    }
    
    return adjacent;
}

// ══════════════════════════════════════════════════════════════════════════════
// PRIVATE HELPER METHODS (Enhanced)
// ══════════════════════════════════════════════════════════════════════════════

void Environment::initializeGrid() {
    std::vector<float> hostGrid(m_state.width * m_state.height, 0.0f);
    
    // Set goal at top-right corner
    hostGrid[(m_state.height - 1) * m_state.width + (m_state.width - 1)] = 1.0f;
    
    // Place obstacles, traps, and reward zones
    placeObstacles();
    placeTraps();
    placeRewardZones();
    
    // Set agent position
    hostGrid[m_state.agent_y * m_state.width + m_state.agent_x] = 0.5f;
    
    // Copy to device
    m_deviceGrid.copyFromHost(hostGrid.data(), hostGrid.size());
}

void Environment::placeObstacles() {
    if (m_state.config.obstacle_density <= 0.0f) return;
    
    std::vector<float> hostGrid = m_deviceGrid.copyToHostAll();
    
    // Simple random obstacle placement (can be enhanced with better algorithms)
    for (int y = 0; y < m_state.height; y++) {
        for (int x = 0; x < m_state.width; x++) {
            // Don't place obstacles on agent start, goal, or adjacent to agent
            bool is_agent_start = (x == m_state.width/2 && y == m_state.height/2);
            bool is_goal = (x == m_state.width-1 && y == m_state.height-1);
            bool near_agent = (std::abs(x - m_state.width/2) <= 1 && std::abs(y - m_state.height/2) <= 1);
            
            if (!is_agent_start && !is_goal && !near_agent) {
                float rand_val = static_cast<float>(rand()) / RAND_MAX;
                if (rand_val < m_state.config.obstacle_density) {
                    hostGrid[y * m_state.width + x] = 0.9f; // Obstacle value
                }
            }
        }
    }
    
    m_deviceGrid.copyFromHost(hostGrid.data(), hostGrid.size());
}

void Environment::placeTraps() {
    if (m_state.config.trap_density <= 0.0f) return;
    
    std::vector<float> hostGrid = m_deviceGrid.copyToHostAll();
    
    for (int y = 0; y < m_state.height; y++) {
        for (int x = 0; x < m_state.width; x++) {
            // Only place traps on empty cells
            if (std::abs(hostGrid[y * m_state.width + x]) < 0.01f) {
                float rand_val = static_cast<float>(rand()) / RAND_MAX;
                if (rand_val < m_state.config.trap_density) {
                    hostGrid[y * m_state.width + x] = 0.7f; // Trap value
                }
            }
        }
    }
    
    m_deviceGrid.copyFromHost(hostGrid.data(), hostGrid.size());
}

void Environment::placeRewardZones() {
    if (m_state.config.reward_zone_density <= 0.0f) return;
    
    std::vector<float> hostGrid = m_deviceGrid.copyToHostAll();
    
    for (int y = 0; y < m_state.height; y++) {
        for (int x = 0; x < m_state.width; x++) {
            // Only place reward zones on empty cells
            if (std::abs(hostGrid[y * m_state.width + x]) < 0.01f) {
                float rand_val = static_cast<float>(rand()) / RAND_MAX;
                if (rand_val < m_state.config.reward_zone_density) {
                    hostGrid[y * m_state.width + x] = 0.3f; // Reward zone value
                }
            }
        }
    }
    
    m_deviceGrid.copyFromHost(hostGrid.data(), hostGrid.size());
}

void Environment::validateConfiguration() {
    // Clamp values to reasonable ranges
    m_state.config.obstacle_density = std::max(0.0f, std::min(0.8f, m_state.config.obstacle_density));
    m_state.config.trap_density = std::max(0.0f, std::min(0.5f, m_state.config.trap_density));
    m_state.config.reward_zone_density = std::max(0.0f, std::min(0.3f, m_state.config.reward_zone_density));
    m_state.config.max_episode_steps = std::max(10, m_state.config.max_episode_steps);
}

void Environment::resetStatistics() {
    m_state.agent_x = m_state.width / 2;
    m_state.agent_y = m_state.height / 2;
    m_state.reward = 0.0f;
    m_state.done = false;
    m_state.episode_steps = 0;
    m_state.total_rewards_collected = 0;
    m_state.cumulative_reward = 0.0f;
    m_state.goal_reached = false;
    m_state.obstacles_hit = 0;
    m_state.traps_triggered = 0;
}

float Environment::calculateDistanceBonus() const {
    int goal_x = m_state.width - 1;
    int goal_y = m_state.height - 1;
    float distance = std::sqrt((m_state.agent_x - goal_x) * (m_state.agent_x - goal_x) + 
                              (m_state.agent_y - goal_y) * (m_state.agent_y - goal_y));
    float max_distance = std::sqrt(m_state.width * m_state.width + m_state.height * m_state.height);
    return 0.01f * (1.0f - distance / max_distance);
}

void Environment::calculateRewardForCellType(CellType cell_type) {
    switch (cell_type) {
        case CellType::GOAL:
            m_state.reward = m_state.config.goal_reward;
            m_state.done = true;
            m_state.goal_reached = true;
            break;
        case CellType::TRAP:
            m_state.reward = m_state.config.trap_penalty;
            m_state.traps_triggered++;
            break;
        case CellType::REWARD_ZONE:
            m_state.reward = m_state.config.reward_zone_bonus;
            m_state.total_rewards_collected++;
            break;
        case CellType::EMPTY:
        case CellType::AGENT:
        default:
            m_state.reward = m_state.config.step_penalty;
            break;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// EXISTING METHODS (Keep compatibility)
// ══════════════════════════════════════════════════════════════════════════════

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

void Environment::copyGridToBuffer(float* host_buffer, size_t num_elements) const {
    size_t grid_size = static_cast<size_t>(m_state.width) * m_state.height;
    if (num_elements != grid_size) {
        throw std::invalid_argument("Buffer size mismatch");
    }
    
    m_deviceGrid.copyToHost(host_buffer, num_elements);
}

void Environment::initializeGridOnDevice() {
    // This method is kept for compatibility but enhanced version is used
    initializeGrid();
}

void Environment::updateHostState() {
    // Copy state from device to host
    EnvironmentState deviceState;
    CUDA_CHECK(cudaMemcpy(&deviceState, m_deviceState.get(), sizeof(EnvironmentState), cudaMemcpyDeviceToHost));
    
    // Update host state (but keep host grid pointer as nullptr since we use device memory)
    m_state.agent_x = deviceState.agent_x;
    m_state.agent_y = deviceState.agent_y;
    m_state.reward = deviceState.reward;
    m_state.done = deviceState.done;
}

void Environment::syncToDevice() {
    // Copy state to device
    EnvironmentState temp_state = m_state;
    temp_state.grid = m_deviceGrid.get();
    CUDA_CHECK(cudaMemcpy(m_deviceState.get(), &temp_state, sizeof(EnvironmentState), cudaMemcpyHostToDevice));
}

void Environment::syncStateToDevice() {
    syncToDevice();
}

} // namespace cudarl
