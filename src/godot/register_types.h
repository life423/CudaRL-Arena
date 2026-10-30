#ifndef CUDARL_GODOT_REGISTER_TYPES_H
#define CUDARL_GODOT_REGISTER_TYPES_H
#include <godot_cpp/core/class_db.hpp>

void initialize_creature_module(godot::ModuleInitializationLevel level);
void uninitialize_creature_module(godot::ModuleInitializationLevel level);

#endif
