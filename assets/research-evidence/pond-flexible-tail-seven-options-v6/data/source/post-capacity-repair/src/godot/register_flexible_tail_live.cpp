#include "flexible_tail_live.h"

#include <godot_cpp/godot.hpp>

namespace {
void initialize_flexible_tail(godot::ModuleInitializationLevel level) {
    if (level == godot::MODULE_INITIALIZATION_LEVEL_SCENE) {
        godot::ClassDB::register_class<godot::FlexibleTailLive>();
    }
}

void uninitialize_flexible_tail(godot::ModuleInitializationLevel) {}
} // namespace

extern "C" GDExtensionBool GDE_EXPORT flexible_tail_live_init(
    GDExtensionInterfaceGetProcAddress address, GDExtensionClassLibraryPtr library,
    GDExtensionInitialization* initialization) {
    godot::GDExtensionBinding::InitObject init(address, library, initialization);
    init.register_initializer(initialize_flexible_tail);
    init.register_terminator(uninitialize_flexible_tail);
    init.set_minimum_library_initialization_level(godot::MODULE_INITIALIZATION_LEVEL_SCENE);
    return init.init();
}

