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
    ClassDB::bind_method(D_METHOD("reset"), &ArenaGDExtension::reset);
    ClassDB::bind_method(D_METHOD("set_maze", "cells"), &ArenaGDExtension::set_maze);
}

ArenaGDExtension::ArenaGDExtension() {
    cuda_arena = new CudaArena(1000);
    
    // Initialize empty maze
    for (int y = 0; y < 10; y++) {
        for (int x = 0; x < 10; x++) {
            maze[y][x] = 0;
        }
    }
    
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

    // --- 2. maze-aware movement helper -------------------------------------
    auto apply = [this](Vector2i &pos, int act) -> float {
        Vector2i new_pos = pos;
        switch (act) {
            case 0: new_pos.x += 1; break;      // → right
            case 1: new_pos.y += 1; break;      // ↓ down
            case 2: new_pos.x -= 1; break;      // ← left
            case 3: new_pos.y -= 1; break;      // ↑ up
        }
        
        // Check wall collision
        if (is_wall(new_pos.x, new_pos.y)) {
            return -0.1f;  // penalty for hitting wall
        } else {
            pos = new_pos;
            return 0.0f;   // neutral reward for valid move
        }
    };

    // --- 3. update states ---------------------------------------------------
    float human_reward = apply(human_state, human_action);
    
    // crude proto‑AI: ignore ai_action & choose random move for now
    int random_move = rand() % 4;
    float ai_reward = apply(ai_state, random_move);
    
    // --- 4. check for goal reached (8,8) -----------------------------------
    bool human_done = false, ai_done = false;
    if (human_state.x == 8 && human_state.y == 8) {
        human_reward += 1.0f;
        ai_reward += -1.0f;
        human_done = ai_done = true;
    } else if (ai_state.x == 8 && ai_state.y == 8) {
        human_reward += -1.0f;
        ai_reward += 1.0f;
        human_done = ai_done = true;
    }

    // --- 5. pack results ----------------------------------------------------
    Array out;
    Dictionary h, a;
    h["state"]  = Vector2(human_state);   h["reward"] = human_reward; h["done"] = human_done;
    a["state"]  = Vector2(ai_state);      a["reward"] = ai_reward;    a["done"] = ai_done;
    out.push_back(h);
    out.push_back(a);
    return out;
}

void ArenaGDExtension::reset_environment() {
    // TODO: cudaMemset Q-table, reset state in GPU
}

void ArenaGDExtension::reset() {
    human_state = Vector2i(1, 1);
    ai_state = Vector2i(8, 8);
}

void ArenaGDExtension::set_maze(PackedVector2Array cells) {
    // Clear maze
    for (int y = 0; y < 10; y++) {
        for (int x = 0; x < 10; x++) {
            maze[y][x] = 0;
        }
    }
    
    // Set walls from TileMap
    for (int i = 0; i < cells.size(); i++) {
        Vector2 cell = cells[i];
        int x = (int)cell.x;
        int y = (int)cell.y;
        if (x >= 0 && x < 10 && y >= 0 && y < 10) {
            maze[y][x] = 1;
        }
    }
}

} // namespace godot
