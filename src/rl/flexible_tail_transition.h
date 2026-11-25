#pragma once

#include "environment/flexible_tail_clearance.h"
#include "rl/flexible_tail_motor.h"

namespace cudarl::flexible_tail {

constexpr int turn_history_controls = 100;
constexpr int steering_history_controls = 25;
constexpr int turn_cue_control = 20;
constexpr int turn_substeps_per_control = 16;
constexpr int turn_training_controls = 400;

struct TurnWorld {
    physics::BodyState body{};
    MovementRequest request{};
    // COM positions immediately before the most recent controls. The next
    // index points at the position exactly two seconds before the body pose.
    physics::V history[turn_history_controls]{};
    int history_next = 0;
    int controls = 0;
    float last_targets[motor_outputs]{};
    float turn_radians = 0;
    int turn_controls = turn_training_controls;
    bool turn_issued = false;
    physics::V entry_course{1, 0, 0};
};

struct TurnStep {
    int fault = 0;
    int attempted_substeps = 0;
    int accepted_substeps = 0;
    float reward = 0;
    float min_gap_m = 1;
    float max_tilt_deg = 0;
    float max_depth_error_m = 0;
};

CUDARL_HD inline void record_turn_position(TurnWorld& world, physics::V com_m) {
    world.history[world.history_next] = com_m;
    world.history_next = (world.history_next + 1) % turn_history_controls;
}

CUDARL_HD inline physics::V turn_course(const TurnWorld& world, physics::V com_m, int samples) {
    const int index = (world.history_next + turn_history_controls - samples) % turn_history_controls;
    auto displacement = com_m - world.history[index];
    displacement.y = 0;
    return displacement;
}

CUDARL_HD inline void prepare_turn_request(TurnWorld& world) {
    if (world.controls == turn_cue_control && !world.turn_issued) {
        const auto pose = snapshot(world.body);
        const auto course = turn_course(world, pose.com_m, turn_history_controls);
        const float length = physics::norm(course);
        world.entry_course = length >= .001f ? course / length : world.request.heading;
        world.request.heading = physics::axis_rotation({0, 1, 0}, world.turn_radians)
            .rotate(world.entry_course);
        world.request.terminal_velocity_m_s = world.request.heading * requested_speed_m_s;
        world.request.origin_m.x = pose.com_m.x;
        world.request.origin_m.z = pose.com_m.z;
        world.request.choice = world.turn_radians > 0 ? Choice::TurnLeft : Choice::TurnRight;
        world.turn_issued = true;
    } else if (world.turn_issued && world.controls == turn_cue_control + world.turn_controls) {
        // The final swim retains the attained target heading and target depth.
        world.request.choice = Choice::Swim;
    }
}

CUDARL_HD inline void observe_turn(const TurnWorld& world, float (&out)[motor_inputs]) {
    const auto pose = snapshot(world.body);
    auto course = turn_course(world, pose.com_m, steering_history_controls);
    if (physics::norm(course) < .0015f) course = world.request.heading;
    observe_motor(world.body, pose, world.request, course, {}, world.last_targets, out);
}

CUDARL_HD inline float angle_error(physics::V course, physics::V heading) {
    return fabsf(atan2f(physics::cross(course, heading).y, physics::dot(course, heading)));
}

struct IgnoreTurnSubstep {
    CUDARL_HD void operator()(const Snapshot&) const {}
};

template <class ObserveSubstep = IgnoreTurnSubstep>
CUDARL_HD CUDARL_DEVICE_NOINLINE TurnStep step_turn_control(
    TurnWorld& world, const float (&targets)[motor_outputs], const pond_geometry_v1::Parameters& pond,
    ObserveSubstep observe_substep = {}, Material material = Material::Baseline) {
    TurnStep result;
    auto pose = snapshot(world.body);
    observe_substep(pose);
    const auto start = pose.com_m;
    const auto before = turn_course(world, start, steering_history_controls);
    const auto heading = world.request.heading;
    record_turn_position(world, start);
    for (int i = 0; i < turn_substeps_per_control; ++i) {
        ++result.attempted_substeps;
        StepInfo info;
        if (!step(world.body, targets, motor_substep_s, info, {}, {}, material)) {
            result.fault = 1;
            return result;
        }
        pose = snapshot(world.body);
        const float gap = static_cast<float>(submerged_gap_m(pond, world.body, pose));
        result.min_gap_m = fminf(result.min_gap_m, gap);
        if (!(gap > 0)) {
            result.fault = 2;
            return result;
        }
        result.max_tilt_deg = fmaxf(result.max_tilt_deg, pose.tilt_deg);
        result.max_depth_error_m = fmaxf(result.max_depth_error_m,
            fabsf(world.request.origin_m.y - pose.com_m.y));
        observe_substep(pose);
        ++result.accepted_substeps;
    }
    const auto after = turn_course(world, pose.com_m, steering_history_controls);
    const float speed = physics::dot(pose.com_m - start, heading) / motor_control_s;
    const float speed_ratio = speed / requested_speed_m_s;
    const float transport = physics::clamp(speed_ratio, -1.f, 1.f)
        - .35f * physics::clamp(speed_ratio - 1.f, 0.f, 2.f);
    const float motion = physics::clamp((physics::norm(after) / .5f) / .006f, 0.f, 1.f);
    const float after_error = angle_error(after, heading);
    constexpr float heading_width = .26179939f; // 15 degrees
    const float scaled_error = after_error / heading_width;
    const float alignment = expf(-scaled_error * scaled_error);
    float correction = 0;
    if (world.request.choice != Choice::Swim && physics::norm(before) >= .0015f &&
        physics::norm(after) >= .0015f) {
        correction = physics::clamp((angle_error(before, heading) - after_error) /
            (motor_control_s * heading_width), -1.f, 1.f);
    }
    const float tilt = fminf(3.f, result.max_tilt_deg / 10.f);
    const float depth = fminf(3.f, result.max_depth_error_m / .001f);
    result.reward = motor_control_s *
        (.65f * transport + motion * (.35f * alignment + .35f * correction)
            - tilt * tilt - .75f * depth * depth);
    if (!physics::finite_number(result.reward)) {
        result.fault = 3;
        return result;
    }
    world.last_targets[0] = targets[0];
    world.last_targets[1] = targets[1];
    ++world.controls;
    return result;
}

inline void validate_turn_worlds(const std::vector<TurnWorld>& worlds) {
    if (worlds.empty() || worlds.size() > 256) throw std::runtime_error("Invalid turn-world count");
    for (const auto& world : worlds) {
        if (!physics::finite_body(world.body) || world.controls != 0 || world.turn_issued ||
            world.history_next < 0 || world.history_next >= turn_history_controls ||
            !world.request.initialized || world.request.choice != Choice::Swim ||
            !physics::finite_vector(world.request.origin_m) || !physics::finite_vector(world.request.heading) ||
            !physics::finite_vector(world.request.terminal_velocity_m_s) ||
            fabsf(physics::norm(world.request.heading) - 1) > .001f ||
            fabsf(world.request.heading.y) > .001f ||
            !physics::finite_number(world.turn_radians) || fabsf(world.turn_radians) > requested_turn_rad + 1e-5f ||
            fabsf(world.turn_radians) < .01f || world.turn_controls < 1 || world.turn_controls > 600)
            throw std::runtime_error("Invalid initial turn world");
        for (const auto& position : world.history) {
            if (!physics::finite_vector(position)) throw std::runtime_error("Invalid turn course history");
        }
        for (const auto target : world.last_targets) {
            if (!physics::finite_number(target) || fabsf(target) > physics::limit)
                throw std::runtime_error("Invalid previous actuator target");
        }
    }
}

inline void write_turn_worlds(const std::string& path, const std::vector<TurnWorld>& worlds) {
    validate_turn_worlds(worlds);
    checkpoint::Writer out;
    out.value<uint32_t>(0x46545731); // FTW1
    out.value<uint32_t>(1);
    out.value<uint32_t>(sizeof(TurnWorld));
    out.array(worlds);
    checkpoint::write_atomic(path, out.data);
}

inline std::vector<TurnWorld> read_turn_worlds(const std::string& path) {
    const auto bytes = checkpoint::read(path);
    checkpoint::Reader in{bytes};
    if (in.value<uint32_t>() != 0x46545731 || in.value<uint32_t>() != 1 ||
        in.value<uint32_t>() != sizeof(TurnWorld)) throw std::runtime_error("Unsupported turn-world checkpoint");
    auto worlds = in.array<TurnWorld>(256);
    in.finish();
    validate_turn_worlds(worlds);
    return worlds;
}

} // namespace cudarl::flexible_tail
