#ifndef ARENA_GDEXTENSION_H
#define ARENA_GDEXTENSION_H
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {

class ArenaGDExtension : public Node {
    GDCLASS(ArenaGDExtension, Node)

protected:
    static void _bind_methods();

public:
    ArenaGDExtension();
    ~ArenaGDExtension();

    String hello_cuda();
    void   run_training(int episodes);
};

} // namespace godot

#endif