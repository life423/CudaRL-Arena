#ifndef ARENA_GDEXTENSION_H
#define ARENA_GDEXTENSION_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include "../gpu/cuda_arena.h"

namespace godot {

class ArenaGDExtension : public RefCounted {
    GDCLASS(ArenaGDExtension, RefCounted);

protected:
    static void _bind_methods();

private:
    CudaArena* cuda_arena;

public:
    ArenaGDExtension();
    ~ArenaGDExtension();
    
    String hello_cuda();
    void run_training(int episodes);
    
    // Advances the env by one action; returns { state, reward, done }
    Dictionary step_environment(int action);
    // Resets the env to its initial state
    void reset_environment();
};

}

#endif // ARENA_GDEXTENSION_H
