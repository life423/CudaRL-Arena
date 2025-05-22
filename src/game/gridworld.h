#pragma once
#include <tuple>
#include <vector>
#include <random>

namespace game {
    class GridWorld {
    public:
        enum class CellType {
            EMPTY = 0,
            WALL = 1,
            GOAL = 2,
            TRAP = 3,
            AGENT = 4
        };
        
        enum class Action {
            UP = 0,
            RIGHT = 1,
            DOWN = 2,
            LEFT = 3
        };
        
        GridWorld(int width, int height);
        ~GridWorld();
        
        // Reset environment and return initial state
        int reset();
        
        // Take action and return (next_state, reward, done)
        std::tuple<int, float, bool> step(int action);
        
        // Get 2D grid representation of the current state
        std::vector<std::vector<int>> get_state_representation();
        
        // Get state space size (number of possible states)
        int get_state_space_size() const;
        
        // Get action space size (number of possible actions)
        int get_action_space_size() const;
        
        // Render the grid (for visualization)
        void render();
        
    private:
        int width_;
        int height_;
        int agent_x_;
        int agent_y_;
        std::vector<std::vector<CellType>> grid_;
        std::mt19937 rng_;
        
        // Convert 2D position to state ID
        int position_to_state(int x, int y) const;
        
        // Check if position is valid
        bool is_valid_position(int x, int y) const;
        
        // Initialize grid with walls, goals, and traps
        void initialize_grid();
    };
}