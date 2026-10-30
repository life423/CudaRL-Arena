#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include "motor_control_lab.h"
namespace {
void initialize_motor_bench(godot::ModuleInitializationLevel level) {
    if(level==godot::MODULE_INITIALIZATION_LEVEL_SCENE) godot::ClassDB::register_class<godot::MotorControlLab>();
}
void uninitialize_motor_bench(godot::ModuleInitializationLevel) {}
}
extern "C" GDExtensionBool GDE_EXPORT motor_bench_library_init(
    GDExtensionInterfaceGetProcAddress address,const GDExtensionClassLibraryPtr library,GDExtensionInitialization* initialization) {
    godot::GDExtensionBinding::InitObject init(address,library,initialization);
    init.register_initializer(initialize_motor_bench);init.register_terminator(uninitialize_motor_bench);
    init.set_minimum_library_initialization_level(godot::MODULE_INITIALIZATION_LEVEL_SCENE);return init.init();
}
