#pragma once
#include "gridworld.h"
#include <random>
#include <vector>
#include <functional>

namespace game {
    class ProceduralGridWorld : public GridWorld {
    public:
        enum class GenerationType {
            MAZE,
            ROOMS,
            CAVE,
            OPEN
        };
        
        ProceduralGridWorld(int width, int height, GenerationType gen_type = GenerationType::MAZE);
        ~ProceduralGridWorld();
        
        // Override reset to generate a new environment each time
        int reset() override;
        
        // Set the generation type
        void set_generation_type(GenerationType gen_type);
        
        // Get the current generation type
        GenerationType get_generation_type() const;
        
    private:
        GenerationType gen_type_;
        
        // Generate a new environment based on the generation type
        void generate_environment();
        
        // Different generation algorithms
        void generate_maze();
        void generate_rooms();
        void generate_cave();
        void generate_open();
        
        // Helper methods for maze generation
        void carve_maze_passage(int x, int y);
        
        // Helper methods for cave generation
        void cellular_automaton(int iterations, float birth_threshold, float death_threshold);
    };
}