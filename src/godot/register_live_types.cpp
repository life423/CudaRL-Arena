#include "frozen_swimmer.h"
#include <gdextension_interface.h>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>
void initialize_frozen_swimmer(godot::ModuleInitializationLevel level){if(level==godot::MODULE_INITIALIZATION_LEVEL_SCENE)godot::ClassDB::register_class<godot::FrozenSwimmer>();}
void uninitialize_frozen_swimmer(godot::ModuleInitializationLevel){}
extern "C" GDExtensionBool GDE_EXPORT frozen_swimmer_init(GDExtensionInterfaceGetProcAddress address,GDExtensionClassLibraryPtr library,GDExtensionInitialization* initialization){godot::GDExtensionBinding::InitObject init(address,library,initialization);init.register_initializer(initialize_frozen_swimmer);init.register_terminator(uninitialize_frozen_swimmer);init.set_minimum_library_initialization_level(godot::MODULE_INITIALIZATION_LEVEL_SCENE);return init.init();}
