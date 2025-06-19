#include "environment.h"
#include "cuda_utils.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <queue>
#include <cmath>
#include <utility>
#include <cstring>

namespace cudarl {

// Constructor
Environment::Environment(int id, int width, int height, const EnvironmentConfig& config)
    : m_envId(id), m_deviceState(1), m_deviceGrid(width * height) {
    
    m_state.width = width;
    m_state.height = height;
    m_state.config = config;
    m_state.grid = new float[width * height];
    
    validateConfiguration();
    initializeGrid();
    initializeGridOnDevice();
    reset();
}

// Destructor
Environment::~Environment() {
    delete[] m_state.grid;
}

// Move constructor
Environment::Environment(Environment&& other) noexcept
    : m_envId(other.m_envId)
    , m_state(std::move(other.m_state))
    , m_deviceState(std::move(other.m_deviceState))
    , m_deviceGrid(std::move(other.m_deviceGrid)) {
    other.m_state.grid = nullptr;
}

// Move assignment
Environment& Environment::operator=(Environment&& other) noexcept {
    if (this != &other) {
        delete[] m_state.grid;
        
        m_envId = other.m_envId;
        m_state = std::move(other.m_state);
        m_deviceState = std::move(other.m_deviceState);
        m_deviceGrid = std::move(other.m_deviceGrid);
        
        other.m_state.grid = nullptr;
    }
    return *this;
}

// Reset environment
void Environment::reset() {
    resetStatistics();
    
    // Reset agent to random position
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_x(0, m_state.width - 1);
    std::uniform_int_distribution<> dis_y(0, m_state.height - 1);
    
    do {
        m_state.agent_x = dis_x(gen);
        m_state.agent_y = dis_y(gen);
    } while (getCellType(m_state.agent_x, m_state.agent_y) != CellType::EMPTY);
    
    m_state.reward = 0.0f;
    m_state.done = false;
    m_state.episode_steps = 0;
    
    syncToDevice();
}

// Step environment
void Environment::step(int action) {
    if (m_state.done) return;
    
    int new_x = m_state.agent_x;
    int new_y = m_state.agent_y;
    
    // Convert action to movement
    switch (action) {
        case 0: new_y--; break; // UP
        case 1: new_y++; break; // DOWN
        case 2: new_x--; break; // LEFT
        case 3: new_x++; break; // RIGHT
    }
    
    m_state.reward = m_state.config.step_penalty;
    
    if (isValidMove(new_x, new_y)) {
        m_state.agent_x = new_x;
        m_state.agent_y = new_y;
        
        CellType cell = getCellType(new_x, new_y);
        switch (cell) {
            case CellType::GOAL:
                m_state.reward = m_state.config.goal_reward;
                m_state.done = true;
                m_state.goal_reached = true;
                break;
            case CellType::OBSTACLE:
                m_state.reward = m_state.config.obstacle_penalty;
                m_state.obstacles_hit++;
                break;
            case CellType::TRAP:
                m_state.reward = m_state.config.trap_penalty;
                m_state.traps_triggered++;
                break;
            case CellType::REWARD_ZONE:
                m_state.reward += m_state.config.reward_zone_bonus;
                break;
        }
        
        if (m_state.config.enable_distance_bonus) {
            m_state.reward += calculateDistanceBonus();
        }
    } else {
        m_state.reward = m_state.config.obstacle_penalty;
    }
    
    m_state.episode_steps++;
    m_state.cumulative_reward += m_state.reward;
    
    if (m_state.episode_steps >= m_state.config.max_episode_steps) {
        m_state.done = true;
    }
    
    syncToDevice();
}

// Reset with obstacles
void Environment::resetWithObstacles(float obstacle_density) {
    if (obstacle_density >= 0.0f) {
        m_state.config.obstacle_density = obstacle_density;
    }
    regenerateEnvironment();
    reset();
}

// Check if move is valid
bool Environment::isValidMove(int new_x, int new_y) const {
    return new_x >= 0 && new_x < m_state.width && 
           new_y >= 0 && new_y < m_state.height;
}

// Get cell type
CellType Environment::getCellType(int x, int y) const {
    if (x < 0 || x >= m_state.width || y < 0 || y >= m_state.height) {
        return CellType::OBSTACLE;
    }
    
    float value = getCellValue(x, y);
    
    if (std::abs(value - 1.0f) < 0.01f) return CellType::GOAL;
    if (std::abs(value - 0.9f) < 0.01f) return CellType::OBSTACLE;
    if (std::abs(value - 0.7f) < 0.01f) return CellType::TRAP;
    if (std::abs(value - 0.3f) < 0.01f) return CellType::REWARD_ZONE;
    if (std::abs(value - 0.5f) < 0.01f) return CellType::AGENT;
    
    return CellType::EMPTY;
}

// Regenerate environment
void Environment::regenerateEnvironment() {
    initializeGrid();
    placeObstacles();
    placeTraps();
    placeRewardZones();
    initializeGridOnDevice();
}

// Configuration methods
void Environment::setEnvironmentConfig(const EnvironmentConfig& config) {
    m_state.config = config;
    validateConfiguration();
}

void Environment::setRewardParameters(float step_penalty, float goal_reward, float obstacle_penalty) {
    m_state.config.step_penalty = step_penalty;
    m_state.config.goal_reward = goal_reward;
    m_state.config.obstacle_penalty = obstacle_penalty;
}

// Grid access methods
float Environment::getCellValue(int x, int y) const {
    if (x < 0 || x >= m_state.width || y < 0 || y >= m_state.height) {
        return 0.9f; // Obstacle value for out of bounds
    }
    return m_state.grid[y * m_state.width + x];
}

std::vector<float> Environment::getGrid() const {
    std::vector<float> grid(m_state.width * m_state.height);
    std::copy(m_state.grid, m_state.grid + m_state.width * m_state.height, grid.begin());
    return grid;
}

void Environment::copyGridToBuffer(float* host_buffer, size_t num_elements) const {
    size_t grid_size = static_cast<size_t>(m_state.width * m_state.height);
    size_t copy_size = std::min(num_elements, grid_size);
    std::copy(m_state.grid, m_state.grid + copy_size, host_buffer);
}

// Utility methods
void Environment::printGrid() const {
    for (int y = 0; y < m_state.height; y++) {
        for (int x = 0; x < m_state.width; x++) {
            if (x == m_state.agent_x && y == m_state.agent_y) {
                std::cout << "A ";
            } else {
                float val = getCellValue(x, y);
                if (val == 1.0f) std::cout << "G ";
                else if (val == 0.9f) std::cout << "# ";
                else if (val == 0.7f) std::cout << "T ";
                else if (val == 0.3f) std::cout << "R ";
                else std::cout << ". ";
            }
        }
        std::cout << std::endl;
    }
}

void Environment::printStats() const {
    std::cout << "Environment Stats:" << std::endl;
    std::cout << "  Steps: " << m_state.episode_steps << std::endl;
    std::cout << "  Cumulative Reward: " << m_state.cumulative_reward << std::endl;
    std::cout << "  Obstacles Hit: " << m_state.obstacles_hit << std::endl;
    std::cout << "  Traps Triggered: " << m_state.traps_triggered << std::endl;
    std::cout << "  Goal Reached: " << (m_state.goal_reached ? "Yes" : "No") << std::endl;
}

// Private helper methods
void Environment::initializeGrid() {
    // Initialize empty grid
    std::fill(m_state.grid, m_state.grid + m_state.width * m_state.height, 0.0f);
    
    // Place goal in bottom-right corner
    m_state.grid[(m_state.height - 1) * m_state.width + (m_state.width - 1)] = 1.0f;
}

void Environment::initializeGridOnDevice() {
    updateHostState();
    syncToDevice();
}

void Environment::updateHostState() {
    // Copy current state to device buffers (this is a stub - would use CUDA in real implementation)
}

void Environment::syncToDevice() {
    // Sync state to device (this is a stub - would use CUDA in real implementation)
}

void Environment::syncStateToDevice() {
    // Sync just the state to device (stub)
}

void Environment::placeObstacles() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int y = 0; y < m_state.height; y++) {
        for (int x = 0; x < m_state.width; x++) {
            if (getCellValue(x, y) == 0.0f && dis(gen) < m_state.config.obstacle_density) {
                m_state.grid[y * m_state.width + x] = 0.9f; // Obstacle
            }
        }
    }
}

void Environment::placeTraps() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int y = 0; y < m_state.height; y++) {
        for (int x = 0; x < m_state.width; x++) {
            if (getCellValue(x, y) == 0.0f && dis(gen) < m_state.config.trap_density) {
                m_state.grid[y * m_state.width + x] = 0.7f; // Trap
            }
        }
    }
}

void Environment::placeRewardZones() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int y = 0; y < m_state.height; y++) {
        for (int x = 0; x < m_state.width; x++) {
            if (getCellValue(x, y) == 0.0f && dis(gen) < m_state.config.reward_zone_density) {
                m_state.grid[y * m_state.width + x] = 0.3f; // Reward zone
            }
        }
    }
}

void Environment::validateConfiguration() {
    // Ensure valid ranges
    m_state.config.obstacle_density = std::max(0.0f, std::min(1.0f, m_state.config.obstacle_density));
    m_state.config.trap_density = std::max(0.0f, std::min(1.0f, m_state.config.trap_density));
    m_state.config.reward_zone_density = std::max(0.0f, std::min(1.0f, m_state.config.reward_zone_density));
    m_state.config.max_episode_steps = std::max(1, m_state.config.max_episode_steps);
}

void Environment::resetStatistics() {
    m_state.episode_steps = 0;
    m_state.total_rewards_collected = 0;
    m_state.cumulative_reward = 0.0f;
    m_state.goal_reached = false;
    m_state.obstacles_hit = 0;
    m_state.traps_triggered = 0;
}

float Environment::calculateDistanceBonus() const {
    // Calculate distance bonus based on proximity to goal
    int goal_x = m_state.width - 1;
    int goal_y = m_state.height - 1;
    
    float distance = std::sqrt(std::pow(goal_x - m_state.agent_x, 2) + 
                              std::pow(goal_y - m_state.agent_y, 2));
    float max_distance = std::sqrt(std::pow(goal_x, 2) + std::pow(goal_y, 2));
    
    return (max_distance - distance) / max_distance * 0.01f; // Small bonus
}

// Additional utility methods (stubs for now)
std::vector<std::pair<int, int>> Environment::getObstaclePositions() const {
    std::vector<std::pair<int, int>> positions;
    for (int y = 0; y < m_state.height; y++) {
        for (int x = 0; x < m_state.width; x++) {
            if (getCellType(x, y) == CellType::OBSTACLE) {
                positions.emplace_back(x, y);
            }
        }
    }
    return positions;
}

std::vector<std::pair<int, int>> Environment::getTrapPositions() const {
    std::vector<std::pair<int, int>> positions;
    for (int y = 0; y < m_state.height; y++) {
        for (int x = 0; x < m_state.width; x++) {
            if (getCellType(x, y) == CellType::TRAP) {
                positions.emplace_back(x, y);
            }
        }
    }
    return positions;
}

std::vector<std::pair<int, int>> Environment::getRewardZonePositions() const {
    std::vector<std::pair<int, int>> positions;
    for (int y = 0; y < m_state.height; y++) {
        for (int x = 0; x < m_state.width; x++) {
            if (getCellType(x, y) == CellType::REWARD_ZONE) {
                positions.emplace_back(x, y);
            }
        }
    }
    return positions;
}

std::pair<int, int> Environment::getGoalPosition() const {
    for (int y = 0; y < m_state.height; y++) {
        for (int x = 0; x < m_state.width; x++) {
            if (getCellType(x, y) == CellType::GOAL) {
                return {x, y};
            }
        }
    }
    return {m_state.width - 1, m_state.height - 1}; // Default goal position
}

std::string Environment::getStateString() const {
    return "Environment State: Agent(" + std::to_string(m_state.agent_x) + "," + 
           std::to_string(m_state.agent_y) + ") Steps:" + std::to_string(m_state.episode_steps) +
           " Reward:" + std::to_string(m_state.cumulative_reward) + 
           " Done:" + (m_state.done ? "true" : "false");
}

double Environment::calculateOptimalPathLength() const {
    // Simple Manhattan distance to goal (stub implementation)
    auto goal = getGoalPosition();
    return std::abs(goal.first - m_state.agent_x) + std::abs(goal.second - m_state.agent_y);
}

bool Environment::isReachable(int target_x, int target_y) const {
    // BFS to check reachability (stub implementation)
    return true; // Simplified for now
}

std::vector<std::pair<int, int>> Environment::getAdjacentCells(int x, int y) const {
    std::vector<std::pair<int, int>> adjacent;
    const int dx[] = {0, 0, -1, 1};
    const int dy[] = {-1, 1, 0, 0};
    
    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (isValidMove(nx, ny)) {
            adjacent.emplace_back(nx, ny);
        }
    }
    
    return adjacent;
}

} // namespace cudarl
