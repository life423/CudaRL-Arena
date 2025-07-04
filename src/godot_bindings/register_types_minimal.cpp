#include <godot_cpp/godot.hpp>
#include <godot_cpp/core/class_db.hpp>
#include "arena_gdextension_minimal.h"

using namespace godot;

void initialize(ModuleInitializationLevel level) {
    if (level != MODULE_INITIALIZATION_LEVEL_SCENE) return;
    ClassDB::register_class<ArenaGDExtension>();
}

extern "C" GDExtensionBool GDE_EXPORT
cudarl_gdextension_library_init(GDExtensionInterfaceGetProcAddress p_get_proc_address,
                                GDExtensionClassLibraryPtr p_library,
                                GDExtensionInitialization *r_initialization) {
    godot::GDExtensionBinding::InitObject init_obj(
        p_get_proc_address, p_library, r_initialization);
    init_obj.register_initializer(initialize);
    init_obj.set_minimum_library_initialization_level(
        MODULE_INITIALIZATION_LEVEL_SCENE);
    return init_obj.init();
}