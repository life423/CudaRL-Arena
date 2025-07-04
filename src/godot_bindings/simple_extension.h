#ifndef SIMPLE_EXTENSION_H
#define SIMPLE_EXTENSION_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>

namespace godot {

class SimpleExtension : public RefCounted {
    GDCLASS(SimpleExtension, RefCounted)

protected:
    static void _bind_methods();

public:
    SimpleExtension();
    ~SimpleExtension();
    
    void hello_world();
};

}

#endif