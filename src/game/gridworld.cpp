#include "gridworld.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>

namespace game {

GridWorld::GridWorld(int width, int height) 
    : width_(width), height_(height), agent_x_(0), agent_y_(0) {
    
    // Seed random number generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    rng_.seed(seed);
    
    // Initialize grid with walls, goals, and traps
    initialize_grid();
}

GridWorld::~GridWorld() {
    // Nothing to clean up
}

int GridWorld::reset() {
    // Place agent at random empty cell
    std::vector<std::pair<int, int>> empty_cells;
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            if (grid_[y][x] == CellType::EMPTY) {
                empty_cells.push_back({x, y});
            }
        }
    }
    
    std::uniform_int_distribution<int> dist(0, empty_cells.size() - 1);
    int idx = dist(rng_);
    agent_x_ = empty_cells[idx].first;
    agent_y_ = empty_cells[idx].second;
    
    return position_to_state(agent_x_, agent_y_);
}

std::tuple<int, float, bool> GridWorld::step(int action) {
    // Calculate new position based on action
    int new_x = agent_x_;
    int new_y = agent_y_;
    
    switch (static_cast<Action>(action)) {
        case Action::UP:
            new_y = std::max(0, agent_y_ - 1);
            break;
        case Action::RIGHT:
            new_x = std::min(width_ - 1, agent_x_ + 1);
            break;
        case Action::DOWN:
            new_y = std::min(height_ - 1, agent_y_ + 1);
            break;
        case Action::LEFT:
            new_x = std::max(0, agent_x_ - 1);
            break;
    }
    
    // Check if new position is valid
    if (!is_valid_position(new_x, new_y) || grid_[new_y][new_x] == CellType::WALL) {
        // Hit a wall, stay in place and get negative reward
        return {position_to_state(agent_x_, agent_y_), -0.1f, false};
    }
    
    // Update agent position
    agent_x_ = new_x;
    agent_y_ = new_y;
    
    // Check for goal or trap
    bool done = false;
    float reward = -0.01f;  // Small negative reward for each step
    
    if (grid_[agent_y_][agent_x_] == CellType::GOAL) {
        reward = 1.0f;
        done = true;
    } else if (grid_[agent_y_][agent_x_] == CellType::TRAP) {
        reward = -1.0f;
        done = true;
    }
    
    return {position_to_state(agent_x_, agent_y_), reward, done};
}

std::vector<std::vector<int>> GridWorld::get_state_representation() {
    std::vector<std::vector<int>> state(height_, std::vector<int>(width_, 0));
    
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            state[y][x] = static_cast<int>(grid_[y][x]);
        }
    }
    
    // Mark agent position
    state[agent_y_][agent_x_] = static_cast<int>(CellType::AGENT);
    
    return state;
}

int GridWorld::get_state_space_size() const {
    return width_ * height_;
}

int GridWorld::get_action_space_size() const {
    return 4;  // UP, RIGHT, DOWN, LEFT
}

void GridWorld::render() {
    std::vector<std::vector<int>> state = get_state_representation();
    
    std::cout << "GridWorld State:" << std::endl;
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            char symbol;
            switch (state[y][x]) {
                case static_cast<int>(CellType::EMPTY):
                    symbol = '.';
                    break;
                case static_cast<int>(CellType::WALL):
                    symbol = '#';
                    break;
                case static_cast<int>(CellType::GOAL):
                    symbol = 'G';
                    break;
                case static_cast<int>(CellType::TRAP):
                    symbol = 'T';
                    break;
                case static_cast<int>(CellType::AGENT):
                    symbol = 'A';
                    break;
                default:
                    symbol = '?';
            }
            std::cout << symbol << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int GridWorld::position_to_state(int x, int y) const {
    return y * width_ + x;
}

bool GridWorld::is_valid_position(int x, int y) const {
    return x >= 0 && x < width_ && y >= 0 && y < height_;
}

void GridWorld::initialize_grid() {
    // Initialize grid with empty cells
    grid_.resize(height_, std::vector<CellType>(width_, CellType::EMPTY));
    
    // Add walls around the edges
    for (int x = 0; x < width_; x++) {
        grid_[0][x] = CellType::WALL;
        grid_[height_ - 1][x] = CellType::WALL;
    }
    for (int y = 0; y < height_; y++) {
        grid_[y][0] = CellType::WALL;
        grid_[y][width_ - 1] = CellType::WALL;
    }
    
    // Add some random walls
    std::uniform_int_distribution<int> x_dist(1, width_ - 2);
    std::uniform_int_distribution<int> y_dist(1, height_ - 2);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    for (int i = 0; i < (width_ * height_) / 10; i++) {
        int x = x_dist(rng_);
        int y = y_dist(rng_);
        grid_[y][x] = CellType::WALL;
    }
    
    // Add goals and traps
    int num_goals = std::max(1, (width_ * height_) / 50);
    int num_traps = std::max(1, (width_ * height_) / 30);
    
    for (int i = 0; i < num_goals; i++) {
        int x, y;
        do {
            x = x_dist(rng_);
            y = y_dist(rng_);
        } while (grid_[y][x] != CellType::EMPTY);
        grid_[y][x] = CellType::GOAL;
    }
    
    for (int i = 0; i < num_traps; i++) {
        int x, y;
        do {
            x = x_dist(rng_);
            y = y_dist(rng_);
        } while (grid_[y][x] != CellType::EMPTY);
        grid_[y][x] = CellType::TRAP;
    }
}

}