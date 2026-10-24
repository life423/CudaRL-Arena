#ifndef ARENA_GDEXTENSION_H
#define ARENA_GDEXTENSION_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector2i.hpp>
#include <memory>
#include "../gpu/cuda_arena.h"

namespace godot {

class ArenaGDExtension : public RefCounted {
    GDCLASS(ArenaGDExtension, RefCounted);

protected:
    static void _bind_methods();

private:
    // Null when no usable CUDA device/driver is present; the maze game still works.
    std::unique_ptr<CudaArena> cuda_arena;

    // Spawns sit in opposite corners; the goal is equidistant (7 moves) from both.
    static constexpr int kHumanStartX = 1, kHumanStartY = 1;
    static constexpr int kAIStartX    = 8, kAIStartY    = 8;
    static constexpr int kGoalX       = 1, kGoalY       = 8;

    // Current positions on a 10×10 board
    Vector2i human_state = Vector2i(kHumanStartX, kHumanStartY);
    Vector2i ai_state    = Vector2i(kAIStartX, kAIStartY);
    
    // Maze layout (1 = wall, 0 = empty)
    int maze[10][10];
    
    bool is_wall(int x, int y) const {
        if (x < 0 || x >= 10 || y < 0 || y >= 10) return true;
        return maze[y][x] == 1;
    }

public:
    ArenaGDExtension();
    ~ArenaGDExtension();
    
    String hello_cuda();
    bool has_cuda() const { return cuda_arena != nullptr; }
    void run_training(int episodes);
    Vector2i get_goal() const { return Vector2i(kGoalX, kGoalY); }
    
    /// Steps both human and AI; input=[human_action, ai_action]
    Array step_environment(Array actions);
    // Resets the env to its initial state
    void reset_environment();
    void reset();
    // Set maze layout from Godot TileMap
    void set_maze(PackedVector2Array cells);
};

}

#endif // ARENA_GDEXTENSION_H
