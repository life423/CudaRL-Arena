#include "maze_arena.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/vector2.hpp>

namespace godot {

namespace {

Dictionary to_dict(const cudarl::maze::AgentStep& s) {
    Dictionary d;
    d["state"]  = Vector2(static_cast<real_t>(s.pos.x), static_cast<real_t>(s.pos.y));
    d["reward"] = s.reward;
    d["done"]   = s.done;
    return d;
}

} // namespace

void MazeArena::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_walls", "cells"), &MazeArena::set_walls);
    ClassDB::bind_method(D_METHOD("get_goal"), &MazeArena::get_goal);
    ClassDB::bind_method(D_METHOD("reset"), &MazeArena::reset);
    ClassDB::bind_method(D_METHOD("step", "actions"), &MazeArena::step);
}

void MazeArena::set_walls(const PackedVector2Array& cells) {
    env_.clear_walls();
    for (int i = 0; i < cells.size(); i++) {
        env_.set_wall(static_cast<int>(cells[i].x), static_cast<int>(cells[i].y));
    }
}

Vector2i MazeArena::get_goal() const {
    return Vector2i(cudarl::maze::MazeEnv::kGoal.x, cudarl::maze::MazeEnv::kGoal.y);
}

void MazeArena::reset() {
    env_.reset();
}

Array MazeArena::step(const Array& actions) {
    ERR_FAIL_COND_V_MSG(actions.size() != 2, Array(), "step expects [human_action, ai_action]");
    const cudarl::maze::StepResult r = env_.step(int(actions[0]), int(actions[1]));
    Array out;
    out.push_back(to_dict(r.human));
    out.push_back(to_dict(r.ai));
    return out;
}

} // namespace godot
