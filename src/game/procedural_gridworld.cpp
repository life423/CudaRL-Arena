#include "procedural_gridworld.h"
#include <algorithm>
#include <queue>
#include <stack>
#include <chrono>

namespace game {

ProceduralGridWorld::ProceduralGridWorld(int width, int height, GenerationType gen_type)
    : GridWorld(width, height), gen_type_(gen_type) {
    // Generate initial environment
    generate_environment();
}

ProceduralGridWorld::~ProceduralGridWorld() {
    // Nothing to clean up
}

int ProceduralGridWorld::reset() {
    // Generate a new environment
    generate_environment();
    
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

void ProceduralGridWorld::set_generation_type(GenerationType gen_type) {
    gen_type_ = gen_type;
}

ProceduralGridWorld::GenerationType ProceduralGridWorld::get_generation_type() const {
    return gen_type_;
}

void ProceduralGridWorld::generate_environment() {
    // Clear the grid
    grid_.clear();
    grid_.resize(height_, std::vector<CellType>(width_, CellType::EMPTY));
    
    // Generate based on the selected type
    switch (gen_type_) {
        case GenerationType::MAZE:
            generate_maze();
            break;
        case GenerationType::ROOMS:
            generate_rooms();
            break;
        case GenerationType::CAVE:
            generate_cave();
            break;
        case GenerationType::OPEN:
            generate_open();
            break;
    }
    
    // Add goals and traps
    int num_goals = std::max(1, (width_ * height_) / 50);
    int num_traps = std::max(1, (width_ * height_) / 30);
    
    std::uniform_int_distribution<int> x_dist(1, width_ - 2);
    std::uniform_int_distribution<int> y_dist(1, height_ - 2);
    
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
        } while (grid_[y][x] != CellType::EMPTY && grid_[y][x] != CellType::GOAL);
        grid_[y][x] = CellType::TRAP;
    }
}

void ProceduralGridWorld::generate_maze() {
    // Fill the grid with walls
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            grid_[y][x] = CellType::WALL;
        }
    }
    
    // Start from a random cell with odd coordinates
    std::uniform_int_distribution<int> x_dist(0, (width_ - 3) / 2);
    std::uniform_int_distribution<int> y_dist(0, (height_ - 3) / 2);
    int start_x = x_dist(rng_) * 2 + 1;
    int start_y = y_dist(rng_) * 2 + 1;
    
    // Carve passages using recursive backtracking
    grid_[start_y][start_x] = CellType::EMPTY;
    carve_maze_passage(start_x, start_y);
}

void ProceduralGridWorld::carve_maze_passage(int x, int y) {
    // Directions: North, East, South, West
    const int dx[] = {0, 1, 0, -1};
    const int dy[] = {-1, 0, 1, 0};
    
    // Randomize directions
    std::vector<int> directions = {0, 1, 2, 3};
    std::shuffle(directions.begin(), directions.end(), rng_);
    
    // Try each direction
    for (int dir : directions) {
        int nx = x + dx[dir] * 2;
        int ny = y + dy[dir] * 2;
        
        // Check if the new position is valid
        if (nx >= 1 && nx < width_ - 1 && ny >= 1 && ny < height_ - 1 && grid_[ny][nx] == CellType::WALL) {
            // Carve passage
            grid_[y + dy[dir]][x + dx[dir]] = CellType::EMPTY;
            grid_[ny][nx] = CellType::EMPTY;
            
            // Continue from the new position
            carve_maze_passage(nx, ny);
        }
    }
}

void ProceduralGridWorld::generate_rooms() {
    // Fill the grid with walls
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            grid_[y][x] = CellType::WALL;
        }
    }
    
    // Number of rooms to generate
    int num_rooms = std::max(3, (width_ * height_) / 100);
    
    // Room size range
    int min_room_size = 3;
    int max_room_size = std::min(7, std::min(width_ / 3, height_ / 3));
    
    // Generate rooms
    std::vector<std::tuple<int, int, int, int>> rooms; // x, y, width, height
    
    for (int i = 0; i < num_rooms * 2; i++) { // Try more rooms than needed
        std::uniform_int_distribution<int> room_width_dist(min_room_size, max_room_size);
        std::uniform_int_distribution<int> room_height_dist(min_room_size, max_room_size);
        
        int room_width = room_width_dist(rng_);
        int room_height = room_height_dist(rng_);
        
        std::uniform_int_distribution<int> x_dist(1, width_ - room_width - 1);
        std::uniform_int_distribution<int> y_dist(1, height_ - room_height - 1);
        
        int room_x = x_dist(rng_);
        int room_y = y_dist(rng_);
        
        // Check for overlap with existing rooms
        bool overlap = false;
        for (const auto& room : rooms) {
            int rx = std::get<0>(room);
            int ry = std::get<1>(room);
            int rw = std::get<2>(room);
            int rh = std::get<3>(room);
            
            if (room_x <= rx + rw + 1 && room_x + room_width + 1 >= rx &&
                room_y <= ry + rh + 1 && room_y + room_height + 1 >= ry) {
                overlap = true;
                break;
            }
        }
        
        if (!overlap) {
            rooms.push_back(std::make_tuple(room_x, room_y, room_width, room_height));
            
            // Carve out the room
            for (int y = room_y; y < room_y + room_height; y++) {
                for (int x = room_x; x < room_x + room_width; x++) {
                    grid_[y][x] = CellType::EMPTY;
                }
            }
            
            if (rooms.size() >= num_rooms) {
                break;
            }
        }
    }
    
    // Connect rooms with corridors
    for (size_t i = 0; i < rooms.size() - 1; i++) {
        int x1 = std::get<0>(rooms[i]) + std::get<2>(rooms[i]) / 2;
        int y1 = std::get<1>(rooms[i]) + std::get<3>(rooms[i]) / 2;
        int x2 = std::get<0>(rooms[i + 1]) + std::get<2>(rooms[i + 1]) / 2;
        int y2 = std::get<1>(rooms[i + 1]) + std::get<3>(rooms[i + 1]) / 2;
        
        // Horizontal corridor
        int x_start = std::min(x1, x2);
        int x_end = std::max(x1, x2);
        for (int x = x_start; x <= x_end; x++) {
            grid_[y1][x] = CellType::EMPTY;
        }
        
        // Vertical corridor
        int y_start = std::min(y1, y2);
        int y_end = std::max(y1, y2);
        for (int y = y_start; y <= y_end; y++) {
            grid_[y][x2] = CellType::EMPTY;
        }
    }
}

void ProceduralGridWorld::generate_cave() {
    // Initialize with random walls
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float wall_probability = 0.45f;
    
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            if (x == 0 || y == 0 || x == width_ - 1 || y == height_ - 1) {
                grid_[y][x] = CellType::WALL; // Border walls
            } else {
                grid_[y][x] = (dist(rng_) < wall_probability) ? CellType::WALL : CellType::EMPTY;
            }
        }
    }
    
    // Apply cellular automaton rules to create cave-like structures
    cellular_automaton(5, 0.5f, 0.5f);
}

void ProceduralGridWorld::cellular_automaton(int iterations, float birth_threshold, float death_threshold) {
    for (int iter = 0; iter < iterations; iter++) {
        std::vector<std::vector<CellType>> new_grid = grid_;
        
        for (int y = 1; y < height_ - 1; y++) {
            for (int x = 1; x < width_ - 1; x++) {
                // Count walls in 3x3 neighborhood
                int wall_count = 0;
                for (int ny = y - 1; ny <= y + 1; ny++) {
                    for (int nx = x - 1; nx <= x + 1; nx++) {
                        if (grid_[ny][nx] == CellType::WALL) {
                            wall_count++;
                        }
                    }
                }
                
                // Apply cellular automaton rules
                float wall_ratio = static_cast<float>(wall_count) / 9.0f;
                
                if (grid_[y][x] == CellType::WALL) {
                    // Death rule: if too few walls around, become empty
                    if (wall_ratio < death_threshold) {
                        new_grid[y][x] = CellType::EMPTY;
                    }
                } else {
                    // Birth rule: if enough walls around, become a wall
                    if (wall_ratio > birth_threshold) {
                        new_grid[y][x] = CellType::WALL;
                    }
                }
            }
        }
        
        grid_ = new_grid;
    }
    
    // Ensure border walls
    for (int y = 0; y < height_; y++) {
        grid_[y][0] = CellType::WALL;
        grid_[y][width_ - 1] = CellType::WALL;
    }
    for (int x = 0; x < width_; x++) {
        grid_[0][x] = CellType::WALL;
        grid_[height_ - 1][x] = CellType::WALL;
    }
}

void ProceduralGridWorld::generate_open() {
    // Create an open environment with just a few obstacles
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            if (x == 0 || y == 0 || x == width_ - 1 || y == height_ - 1) {
                grid_[y][x] = CellType::WALL; // Border walls
            } else {
                grid_[y][x] = CellType::EMPTY;
            }
        }
    }
    
    // Add a few random walls
    std::uniform_int_distribution<int> x_dist(1, width_ - 2);
    std::uniform_int_distribution<int> y_dist(1, height_ - 2);
    std::uniform_int_distribution<int> len_dist(2, 5);
    
    int num_obstacles = (width_ * height_) / 50;
    
    for (int i = 0; i < num_obstacles; i++) {
        int x = x_dist(rng_);
        int y = y_dist(rng_);
        int length = len_dist(rng_);
        bool horizontal = (rng_() % 2 == 0);
        
        if (horizontal) {
            for (int j = 0; j < length && x + j < width_ - 1; j++) {
                grid_[y][x + j] = CellType::WALL;
            }
        } else {
            for (int j = 0; j < length && y + j < height_ - 1; j++) {
                grid_[y + j][x] = CellType::WALL;
            }
        }
    }
}

}