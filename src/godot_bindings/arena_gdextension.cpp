#include "arena_gdextension.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {

void ArenaGDExtension::_bind_methods() {
    ClassDB::bind_method(D_METHOD("hello_cuda"), &ArenaGDExtension::hello_cuda);
    ClassDB::bind_method(D_METHOD("run_training", "episodes"), &ArenaGDExtension::run_training);
    ClassDB::bind_method(D_METHOD("step_environment", "action"), &ArenaGDExtension::step_environment);
    ClassDB::bind_method(D_METHOD("reset_environment"), &ArenaGDExtension::reset_environment);
}

ArenaGDExtension::ArenaGDExtension() {
    cuda_arena = new CudaArena(1000);
    UtilityFunctions::print("🚀 CUDA Arena Extension created");
}

ArenaGDExtension::~ArenaGDExtension() {
    delete cuda_arena;
}

String ArenaGDExtension::hello_cuda() {
    // Run hello kernel first
    cuda_arena->hello_cuda();
    
    int device_count = CudaArena::get_device_count();
    if (device_count > 0) {
        String device_name = String(CudaArena::get_device_name(0).c_str());
        return String("✅ CUDA Ready: ") + String::num(device_count) + " devices, using " + device_name;
    }
    return "❌ No CUDA devices found";
}

void ArenaGDExtension::run_training(int episodes) {
    UtilityFunctions::print("Starting training for ", episodes, " episodes...");
    
    cuda_arena->reset_environments();
    
    for (int i = 0; i < episodes; i++) {
        std::vector<int> actions(1000, i % 4); // Cycle through actions 0-3
        cuda_arena->step_environments(actions);
        
        if (i % 100 == 0) {
            UtilityFunctions::print("Episode ", i, " completed");
        }
    }
    
    UtilityFunctions::print("✅ Training complete: ", episodes, " episodes");
}

Dictionary ArenaGDExtension::step_environment(int action) {
    Dictionary d;
    // TODO: call your CUDA kernel here
    d["state"] = Vector2(0, 0);   // placeholder
    d["reward"] = 0.0;
    d["done"] = false;
    return d;
}

void ArenaGDExtension::reset_environment() {
    // TODO: cudaMemset Q-table, reset state in GPU
}

} // namespace godot
