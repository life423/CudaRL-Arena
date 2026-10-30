#include "register_types.h"
#include "creature_lab.h"
#include <gdextension_interface.h>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

void initialize_creature_module(godot::ModuleInitializationLevel level) {
    if (level == godot::MODULE_INITIALIZATION_LEVEL_SCENE)
        godot::ClassDB::register_class<godot::CreatureLab>();
}

void uninitialize_creature_module(godot::ModuleInitializationLevel) {}

extern "C" GDExtensionBool GDE_EXPORT creature_library_init(
        GDExtensionInterfaceGetProcAddress get_proc_address,
        GDExtensionClassLibraryPtr library,
        GDExtensionInitialization* initialization) {
    godot::GDExtensionBinding::InitObject init(get_proc_address, library, initialization);
    init.register_initializer(initialize_creature_module);
    init.register_terminator(uninitialize_creature_module);
    init.set_minimum_library_initialization_level(godot::MODULE_INITIALIZATION_LEVEL_SCENE);
    return init.init();
}
