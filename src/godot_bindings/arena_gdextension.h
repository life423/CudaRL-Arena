#ifndef ARENA_GDEXTENSION_H
#define ARENA_GDEXTENSION_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

class ArenaGDExtension : public RefCounted {
    GDCLASS(ArenaGDExtension, RefCounted)

protected:
    static void _bind_methods();

public:
    ArenaGDExtension();
    ~ArenaGDExtension();

    // Smoke test method
    String hello_cuda();
};

}

#endif // ARENA_GDEXTENSION_H
