#include "arena_gdextension.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {

void ArenaGDExtension::_bind_methods() {
    ClassDB::bind_method(D_METHOD("hello_cuda"), &ArenaGDExtension::hello_cuda);
}

ArenaGDExtension::ArenaGDExtension() {
    UtilityFunctions::print("🚀 CUDA Arena node created");
}

ArenaGDExtension::~ArenaGDExtension() {
}

String ArenaGDExtension::hello_cuda() {
    UtilityFunctions::print("✅ CudaRL Plugin loaded and ready!");
    return String("✅ CudaRL Plugin loaded and ready!");
}

} // namespace godot
