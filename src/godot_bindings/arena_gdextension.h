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
#include "../gpu/cuda_arena.h"

namespace godot {

class ArenaGDExtension : public RefCounted {
    GDCLASS(ArenaGDExtension, RefCounted);

protected:
    static void _bind_methods();

private:
    CudaArena* cuda_arena;
    
    // Current positions on a 10×10 board
    Vector2i human_state = Vector2i(1, 1);
    Vector2i ai_state    = Vector2i(8, 8);
    
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
    void run_training(int episodes);
    
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
