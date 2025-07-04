#include "arena_gdextension.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {

void ArenaGDExtension::_bind_methods() {
    ClassDB::bind_method(D_METHOD("hello_cuda"), &ArenaGDExtension::hello_cuda);
    ClassDB::bind_method(D_METHOD("run_training", "episodes"), &ArenaGDExtension::run_training);
    ClassDB::bind_method(D_METHOD("step_environment", "actions"), &ArenaGDExtension::step_environment);
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

Array ArenaGDExtension::step_environment(Array actions) {
    // Expect exactly two elements: [human_action, ai_action]
    int human_action = int(actions[0]);
    int ai_action    = int(actions[1]);

    // TODO: invoke your CUDA kernels here, passing both actions

    // Build two result dictionaries (human & AI)
    Dictionary human_res;
    human_res["state"]  = Vector2(0, 0);  // placeholder
    human_res["reward"] = 0.0;
    human_res["done"]   = false;

    Dictionary ai_res;
    ai_res["state"]     = Vector2(9, 9);  // placeholder
    ai_res["reward"]    = 0.0;
    ai_res["done"]      = false;

    // Return them in an Array
    Array result;
    result.append(human_res);
    result.append(ai_res);
    return result;
}

void ArenaGDExtension::reset_environment() {
    // TODO: cudaMemset Q-table, reset state in GPU
}

} // namespace godot
