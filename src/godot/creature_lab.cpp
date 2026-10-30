#include "creature_lab.h"
#include "simulation/world.h"
#include <array>
#include <cmath>
#include <filesystem>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/vector3.hpp>

namespace godot {
namespace {
namespace sim = cudarl::simulation;
namespace physical = cudarl::physics;
constexpr const char* diagnostic_save = "user://foundation/diagnostic-state.bin";
Vector3 vector(physical::V value) { return {value.x, value.y, value.z}; }
std::string save_path() {
    return ProjectSettings::get_singleton()->globalize_path(diagnostic_save).utf8().get_data();
}
} // namespace

struct CreatureLab::Impl {
    sim::PhysicalWorld world;
    std::array<float, 4> commands{};
};

CreatureLab::CreatureLab() : impl_(std::make_unique<Impl>()) {}
CreatureLab::~CreatureLab() = default;

void CreatureLab::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_state"), &CreatureLab::get_state);
    ClassDB::bind_method(D_METHOD("set_commands", "commands"), &CreatureLab::set_commands);
    ClassDB::bind_method(D_METHOD("set_mode", "mode"), &CreatureLab::set_mode);
    ClassDB::bind_method(D_METHOD("set_current", "speed_m_s"), &CreatureLab::set_current);
    ClassDB::bind_method(D_METHOD("set_paused", "paused"), &CreatureLab::set_paused);
    ClassDB::bind_method(D_METHOD("step"), &CreatureLab::step);
    ClassDB::bind_method(D_METHOD("single_step"), &CreatureLab::single_step);
    ClassDB::bind_method(D_METHOD("reset_world"), &CreatureLab::reset_world);
    ClassDB::bind_method(D_METHOD("save_state"), &CreatureLab::save_state);
    ClassDB::bind_method(D_METHOD("load_state"), &CreatureLab::load_state);
    ClassDB::bind_method(D_METHOD("has_saved_state"), &CreatureLab::has_saved_state);
    ClassDB::bind_method(D_METHOD("get_error"), &CreatureLab::get_error);
}

bool CreatureLab::set_commands(PackedFloat32Array commands) {
    error_ = "";
    if (commands.size() != 4) { error_ = "Exactly four actuator commands are required."; return false; }
    std::array<float, 4> checked{};
    for (int axis = 0; axis < 4; ++axis) {
        if (!std::isfinite(commands[axis]) || std::fabs(commands[axis]) > 1.f) {
            error_ = "Actuator commands must be finite and between -1 and 1."; return false;
        }
        checked[axis] = commands[axis];
    }
    // Editing controls changes pending inputs only. It never advances dynamics.
    impl_->commands = checked;
    return true;
}

bool CreatureLab::set_mode(int mode) {
    error_ = "";
    if (mode != 0 && mode != 1) { error_ = "Unknown actuation mode."; return false; }
    impl_->world.set_paused(true);
    impl_->world.set_actuation(static_cast<sim::Actuation>(mode));
    impl_->commands = {};
    return true;
}

bool CreatureLab::set_current(double speed) {
    error_ = "";
    if (!std::isfinite(speed) || speed < 0 || speed > .0075) {
        error_ = "This diagnostic supports a steady +X current from 0 to 0.75 cm/s."; return false;
    }
    try { impl_->world.set_current({static_cast<float>(speed), 0, 0}); return true; }
    catch (const std::exception& e) { error_ = String(e.what()); return false; }
}

void CreatureLab::set_paused(bool paused) {
    error_ = "";
    const auto& state = impl_->world.state();
    if (!paused && (state.unsafe || state.invalid)) {
        error_ = "The diagnostic stopped at an unsafe or invalid state. Reset before continuing."; return;
    }
    impl_->world.set_paused(paused);
}

Dictionary CreatureLab::step() {
    error_ = "";
    const auto& before = impl_->world.state();
    if (before.paused) return get_state();
    impl_->world.step(impl_->commands);
    const auto& state = impl_->world.state();
    if (state.unsafe || state.invalid) {
        impl_->world.set_paused(true);
        error_ = "Diagnostic stopped at the physical safety boundary. No rescue or projection was applied.";
    }
    return get_state();
}

Dictionary CreatureLab::single_step() {
    set_paused(false);
    if (!error_.is_empty()) return get_state();
    step();
    impl_->world.set_paused(true);
    return get_state();
}

void CreatureLab::reset_world() {
    error_ = "";
    impl_->world.reset();
    impl_->commands = {};
}

bool CreatureLab::save_state() {
    error_ = "";
    try {
        const auto path = save_path();
        std::filesystem::create_directories(std::filesystem::u8path(path).parent_path());
        impl_->world.save(path);
        return true;
    } catch (const std::exception& e) { error_ = String("Could not save diagnostic state: ") + e.what(); return false; }
}

bool CreatureLab::load_state() {
    error_ = "";
    try {
        impl_->world.load(save_path());
        // Loading never silently resumes a previously running diagnostic.
        impl_->world.set_paused(true);
        for (int axis = 0; axis < 4; ++axis) impl_->commands[axis] = impl_->world.state().raw_action[axis];
        return true;
    } catch (const std::exception& e) { error_ = String("Could not load diagnostic state: ") + e.what(); return false; }
}

bool CreatureLab::has_saved_state() const {
    std::error_code error;
    return std::filesystem::is_regular_file(std::filesystem::u8path(save_path()), error);
}

Dictionary CreatureLab::get_state() const {
    const auto& state = impl_->world.state();
    Dictionary result;
    const bool finite = sim::finite_state(state) && !state.invalid;
    result["finite"] = finite;
    result["unsafe"] = state.unsafe;
    result["paused"] = state.paused;
    result["mode"] = static_cast<int>(state.actuation);
    result["elapsed"] = state.time_seconds;
    result["steps"] = static_cast<int64_t>(state.control_steps);
    result["substeps"] = static_cast<int64_t>(state.substeps);
    result["current_x"] = state.current.x;
    result["physics_version"] = cudarl::contact::kPhysicsVersion;
    result["contact_gate_passed"] = false;
    result["learned"] = false;
    result["position_cm"] = vector(state.body.position * (physical::scale * 100.f));
    const auto direction = state.body.orientation.rotate({1, 0, 0});
    result["heading_degrees"] = std::atan2(direction.z, direction.x) * 57.295779513f;
    result["head_speed_cm_s"] = physical::norm({state.body.velocity[0], state.body.velocity[1], state.body.velocity[2]}) * 100.f;
    result["floor_force_mN"] = state.telemetry.floor_force_max * 1000.f;
    result["contact_samples"] = state.telemetry.contact_samples;
    result["first_unsafe_substep"] = state.telemetry.first_unsafe_substep;
    PackedFloat32Array pending, applied, joints, rates;
    for (int axis = 0; axis < 4; ++axis) {
        pending.append(impl_->commands[axis]);
        applied.append(state.last_actual_torques[axis] * physical::torque * 1e6f);
        joints.append(state.body.angle[axis]);
        rates.append(state.body.velocity[6 + axis]);
    }
    result["commands"] = pending;
    result["actual_torques_uNm"] = applied;
    result["joint_angles_rad"] = joints;
    result["joint_rates_rad_s"] = rates;
    Array links;
    physical::V com_velocity{};
    float mass = 0;
    if (finite) {
        const auto geometry = physical::shape(state.body);
        for (int index = 0; index < 3; ++index) {
            const auto& link = geometry.link[index];
            com_velocity += physical::multiply(link.linear, state.body.velocity) * link.mass;
            mass += link.mass;
            Dictionary visual;
            visual["position"] = vector(link.center / physical::scale);
            visual["orientation"] = Quaternion(link.q.x, link.q.y, link.q.z, link.q.w);
            visual["length_model"] = link.length / physical::scale;
            visual["half_height_model"] = (index == 0 ? .0064f : (index == 2 ? .006f : .0015f)) / physical::scale;
            visual["half_width_model"] = (index == 0 ? .008f : .0015f) / physical::scale;
            links.append(visual);
        }
        result["floor_gap_mm"] = cudarl::contact::inspect(state.body).floor_gap * 1000.f;
    } else result["floor_gap_mm"] = 0.f;
    result["com_speed_cm_s"] = mass > 0 ? physical::norm(com_velocity / mass) * 100.f : 0.f;
    result["links"] = links;
    return result;
}

} // namespace godot
