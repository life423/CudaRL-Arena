#include "arena_gdextension.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <cstdlib>
#include <exception>
#include "../gpu/actions.h"

namespace godot {

void ArenaGDExtension::_bind_methods() {
    ClassDB::bind_method(D_METHOD("hello_cuda"), &ArenaGDExtension::hello_cuda);
    ClassDB::bind_method(D_METHOD("has_cuda"), &ArenaGDExtension::has_cuda);
    ClassDB::bind_method(D_METHOD("get_goal"), &ArenaGDExtension::get_goal);
    ClassDB::bind_method(D_METHOD("run_training", "episodes"), &ArenaGDExtension::run_training);
    ClassDB::bind_method(D_METHOD("step_environment", "actions"), &ArenaGDExtension::step_environment);
    ClassDB::bind_method(D_METHOD("reset_environment"), &ArenaGDExtension::reset_environment);
    ClassDB::bind_method(D_METHOD("reset"), &ArenaGDExtension::reset);
    ClassDB::bind_method(D_METHOD("set_maze", "cells"), &ArenaGDExtension::set_maze);
}

ArenaGDExtension::ArenaGDExtension() {
    // Exceptions must not escape into Godot, or they take down the editor.
    // Without a GPU we keep running with CUDA features disabled.
    if (CudaArena::get_device_count() > 0) {
        try {
            cuda_arena = std::make_unique<CudaArena>(1000);
        } catch (const std::exception& e) {
            UtilityFunctions::push_warning("CUDA init failed, GPU features disabled: ", e.what());
        }
    } else {
        UtilityFunctions::push_warning("No CUDA device found, GPU features disabled");
    }

    // Initialize empty maze
    for (int y = 0; y < 10; y++) {
        for (int x = 0; x < 10; x++) {
            maze[y][x] = 0;
        }
    }
    
    UtilityFunctions::print("🚀 CUDA Arena Extension created");
}

ArenaGDExtension::~ArenaGDExtension() = default;

String ArenaGDExtension::hello_cuda() {
    if (!cuda_arena) {
        return "❌ No CUDA devices found";
    }
    try {
        cuda_arena->hello_cuda();
    } catch (const std::exception& e) {
        return String("❌ CUDA kernel launch failed: ") + e.what();
    }

    int device_count = CudaArena::get_device_count();
    String device_name = String(CudaArena::get_device_name(0).c_str());
    return String("✅ CUDA Ready: ") + String::num_int64(device_count) + (device_count == 1 ? " device" : " devices") + ", using " + device_name;
}

void ArenaGDExtension::run_training(int episodes) {
    ERR_FAIL_COND_MSG(!cuda_arena, "run_training requires a CUDA device");
    UtilityFunctions::print("Starting training for ", episodes, " episodes...");

    try {
        cuda_arena->reset_environments();

        const int num_envs = cuda_arena->get_num_environments();
        for (int i = 0; i < episodes; i++) {
            std::vector<int> actions(num_envs, i % cudarl::kNumActions); // Cycle through actions
            cuda_arena->step_environments(actions);

            if (i % 100 == 0) {
                UtilityFunctions::print("Episode ", i, " completed");
            }
        }
    } catch (const std::exception& e) {
        ERR_FAIL_MSG(String("Training aborted: ") + e.what());
    }

    UtilityFunctions::print("✅ Training complete: ", episodes, " episodes");
}

Array ArenaGDExtension::step_environment(Array actions) {
    // --- 1. pull actions ----------------------------------------------------
    ERR_FAIL_COND_V(actions.size() != 2, Array());        // safety
    int human_action = int(actions[0]);
    int ai_action    = int(actions[1]);

    // --- 2. maze-aware movement helper -------------------------------------
    // Same encoding as the CUDA kernel (see gpu/actions.h); invalid actions don't move.
    auto apply = [this](Vector2i &pos, int act) -> float {
        Vector2i new_pos = pos + Vector2i(cudarl::action_dx(act), cudarl::action_dy(act));

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
    int random_move = rand() % cudarl::kNumActions;
    float ai_reward = apply(ai_state, random_move);

    // --- 4. check for goal reached ------------------------------------------
    const Vector2i goal = get_goal();
    const bool human_won = human_state == goal;
    const bool ai_won    = ai_state == goal;
    if (human_won && !ai_won) {
        human_reward += 1.0f;
        ai_reward    -= 1.0f;
    } else if (ai_won && !human_won) {
        human_reward -= 1.0f;
        ai_reward    += 1.0f;
    }                                    // simultaneous arrival is a draw
    const bool human_done = human_won || ai_won;
    const bool ai_done    = human_done;

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
    human_state = Vector2i(kHumanStartX, kHumanStartY);
    ai_state    = Vector2i(kAIStartX, kAIStartY);
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
