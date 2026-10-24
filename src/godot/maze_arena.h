#ifndef CUDARL_GODOT_MAZE_ARENA_H
#define CUDARL_GODOT_MAZE_ARENA_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include "envs/maze/maze_env.h"

namespace godot {

// GDScript wrapper around cudarl::maze::MazeEnv.
class MazeArena : public RefCounted {
    GDCLASS(MazeArena, RefCounted);

protected:
    static void _bind_methods();

public:
    void     set_walls(const PackedVector2Array& cells);
    Vector2i get_goal() const;
    void     reset();
    // actions = [human_action, ai_action]; returns [human, ai], each
    // { "state": Vector2, "reward": float, "done": bool }.
    Array    step(const Array& actions);

private:
    cudarl::maze::MazeEnv env_;
};

} // namespace godot

#endif // CUDARL_GODOT_MAZE_ARENA_H
