#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

#include "cuda_device.h"
#include "maze_arena.h"
#include "network_trainer.h"
#include "ppo_lab.h"
#include "racing_lab.h"
#include "tadpole_lab.h"

using namespace godot;

namespace {

void initialize_cudarl_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) return;
    ClassDB::register_class<CudaDevice>();
    ClassDB::register_class<MazeArena>();
    ClassDB::register_class<NetworkTrainer>();
    ClassDB::register_class<PpoLab>();
    ClassDB::register_class<RacingLab>();
    ClassDB::register_class<TadpoleLab>();
}

void uninitialize_cudarl_module(ModuleInitializationLevel) {}

} // namespace

extern "C" GDExtensionBool GDE_EXPORT cudarl_gdextension_library_init(
        GDExtensionInterfaceGetProcAddress p_get_proc_address,
        const GDExtensionClassLibraryPtr p_library,
        GDExtensionInitialization* r_initialization) {
    GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);
    init_obj.register_initializer(initialize_cudarl_module);
    init_obj.register_terminator(uninitialize_cudarl_module);
    init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);
    return init_obj.init();
}
