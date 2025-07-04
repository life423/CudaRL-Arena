#include "arena_gdextension.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <cstdlib>

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
    // --- 1. pull actions ----------------------------------------------------
    ERR_FAIL_COND_V(actions.size() != 2, Array());        // safety
    int human_action = int(actions[0]);
    int ai_action    = int(actions[1]);

    // --- 2. small helpers ---------------------------------------------------
    auto apply = [](Vector2i &pos, int act) {
        switch (act) {
            case 0: pos.x += 1; break;      // → right
            case 1: pos.y += 1; break;      // ↓ down
            case 2: pos.x -= 1; break;      // ← left
            case 3: pos.y -= 1; break;      // ↑ up
        }
        pos.x = CLAMP(pos.x, 0, 9);
        pos.y = CLAMP(pos.y, 0, 9);
    };

    // --- 3. update states ---------------------------------------------------
    apply(human_state, human_action);

    // crude proto‑AI: ignore ai_action & choose random move for now
    int random_move = rand() % 4;
    apply(ai_state, random_move);

    // --- 4. pack results ----------------------------------------------------
    Array out;
    Dictionary h, a;
    h["state"]  = Vector2(human_state);   h["reward"] = 0.0; h["done"] = false;
    a["state"]  = Vector2(ai_state);      a["reward"] = 0.0; a["done"] = false;
    out.push_back(h);
    out.push_back(a);
    return out;
}

void ArenaGDExtension::reset_environment() {
    // TODO: cudaMemset Q-table, reset state in GPU
}

} // namespace godot
