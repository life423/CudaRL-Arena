#pragma once

#include "rl/flexible_tail_transition.h"

#include <algorithm>

namespace cudarl::flexible_tail {

enum class NavigationMode : uint32_t { Heading, Destination, Launch, Gait, YawOffset };
constexpr int navigation_inputs = motor_inputs + 3;
constexpr int navigation_max_legs = 4;
constexpr float navigation_position_scale_m = .10f;

struct NavigationLeg {
    physics::V goal_m{};
    physics::V heading{1, 0, 0};
    int end_control = 0;
    Choice choice = Choice::Swim;
};

struct NavigationWorld {
    TurnWorld motion{};
    NavigationLeg legs[navigation_max_legs]{};
    int leg_count = 0;
    int active_leg = 0;
};

struct HoldStepMetrics {
    float peak_speed_m_s = 0.f;
    float peak_anchor_error_m = 0.f;
    float peak_lateral_error_m = 0.f;
    float min_axial_error_m = 1.f;
    float max_axial_error_m = -1.f;
};

struct NavigationStep : TurnStep {
    HoldStepMetrics hold;
};

struct MeasureHoldSubstep {
    HoldStepMetrics& metrics;
    const MovementRequest& request;

    CUDARL_HD void operator()(const Snapshot& pose) const {
        const auto offset = pose.com_m - request.origin_m;
        const float axial = physics::dot(offset, request.heading);
        metrics.peak_speed_m_s = fmaxf(metrics.peak_speed_m_s, physics::norm(pose.com_velocity_m_s));
        metrics.peak_anchor_error_m = fmaxf(metrics.peak_anchor_error_m, physics::norm(offset));
        metrics.peak_lateral_error_m = fmaxf(metrics.peak_lateral_error_m,
            fabsf(physics::cross(offset, request.heading).y));
        metrics.min_axial_error_m = fminf(metrics.min_axial_error_m, axial);
        metrics.max_axial_error_m = fmaxf(metrics.max_axial_error_m, axial);
    }
};

CUDARL_HD inline void prepare_navigation(NavigationWorld& world) {
    if (world.active_leg + 1 < world.leg_count &&
        world.motion.controls == world.legs[world.active_leg].end_control) {
        ++world.active_leg;
    }
    const auto& leg = world.legs[world.active_leg];
    world.motion.request = {leg.choice, leg.goal_m, leg.heading,
        leg.choice == Choice::Hold ? physics::V{} : leg.heading * requested_speed_m_s, true};
}

CUDARL_HD inline void observe_navigation(const NavigationWorld& world, NavigationMode mode,
                                         float (&out)[navigation_inputs]) {
    const auto& motion = world.motion;
    const auto pose = snapshot(motion.body);
    const auto course = turn_course(motion, pose.com_m, steering_history_controls);
    float base[motor_inputs];
    observe_motor(motion.body, pose, motion.request, course, {}, motion.last_targets, base);
    // With less than 1.5 mm of horizontal travel in 0.5 s, there is no
    // reliable COM course. Zero both directional components in both arms.
    if (physics::norm(course) < .0015f) {
        base[19] = 0.f;
        base[20] = 0.f;
    }
    for (int i = 0; i < motor_inputs; ++i) out[i] = base[i];
    physics::V local_goal{};
    if (mode == NavigationMode::Destination || motion.request.choice == Choice::Hold) {
        const auto offset = motion.request.origin_m - pose.com_m;
        local_goal = motion.body.orientation.inverse(offset) / navigation_position_scale_m;
    }
    out[motor_inputs] = local_goal.x;
    out[motor_inputs + 1] = local_goal.y;
    out[motor_inputs + 2] = local_goal.z;
}

CUDARL_HD inline float navigation_progress_m(physics::V before, physics::V after, physics::V goal) {
    const auto start_offset = goal - before;
    const auto end_offset = goal - after;
    const float distance_sum = physics::norm(start_offset) + physics::norm(end_offset);
    // Factor the difference of squared distances to avoid subtracting nearly
    // equal distances when a slow swimmer makes a small but real displacement.
    return distance_sum > 0.f
        ? physics::dot(start_offset + end_offset, after - before) / distance_sum : 0.f;
}

CUDARL_HD inline float navigation_transport(float speed_m_s) {
    const float speed_ratio = speed_m_s / requested_speed_m_s;
    return physics::clamp(speed_ratio, -1.f, 1.f) -
        .35f * physics::clamp(speed_ratio - 1.f, 0.f, 2.f);
}

template <bool measure_hold = true>
CUDARL_HD CUDARL_DEVICE_NOINLINE NavigationStep step_navigation(
    NavigationWorld& world, NavigationMode mode, const float (&targets)[motor_outputs],
    const pond_geometry_v1::Parameters& pond) {
    auto& motion = world.motion;
    const auto held = motion.request;
    const auto start = snapshot(motion.body).com_m;
    NavigationStep result{};
    if constexpr (measure_hold) {
        if (held.choice == Choice::Hold) {
            static_cast<TurnStep&>(result) = step_turn_control(motion, targets, pond,
                MeasureHoldSubstep{result.hold, held});
        } else {
            static_cast<TurnStep&>(result) = step_turn_control(motion, targets, pond);
        }
    } else {
        static_cast<TurnStep&>(result) = step_turn_control(motion, targets, pond);
    }
    if (result.fault) return result;

    const auto end = snapshot(motion.body).com_m;
    if (held.choice == Choice::Hold) {
        const float drift = fminf(3.f, physics::norm(end - held.origin_m) / .002f);
        const float speed = fminf(3.f, physics::norm(end - start) / (motor_control_s * .002f));
        const float tilt = fminf(3.f, result.max_tilt_deg / 10.f);
        const float depth = fminf(3.f, result.max_depth_error_m / .001f);
        result.reward = motor_control_s *
            (1.f - drift * drift - speed * speed - tilt * tilt - .75f * depth * depth);
    } else if (mode == NavigationMode::Destination) {
        const float projected_speed = physics::dot(end - start, held.heading) / motor_control_s;
        const float approach_speed = navigation_progress_m(start, end, held.origin_m) / motor_control_s;
        result.reward += motor_control_s * .65f *
            (navigation_transport(approach_speed) - navigation_transport(projected_speed));
    }
    if (!physics::finite_number(result.reward)) result.fault = 3;
    return result;
}

inline nn::MlpSpec navigation_motor_spec() {
    auto spec = motor_spec();
    spec.sizes.front() = navigation_inputs;
    return spec;
}

inline std::vector<float> expand_navigation_weights(const std::vector<float>& original) {
    validate_motor_weights(original);
    const auto source = motor_spec();
    const auto expanded = navigation_motor_spec();
    std::vector<float> weights(expanded.num_params(), 0.f);
    for (int row = 0; row < source.out(0); ++row) {
        std::copy_n(original.data() + row * motor_inputs, motor_inputs,
            weights.data() + row * navigation_inputs);
    }
    std::copy(original.begin() + source.bias_offset(0), original.end(),
        weights.begin() + expanded.bias_offset(0));
    return weights;
}

inline void validate_navigation_weights(const std::vector<float>& weights) {
    if (weights.size() != static_cast<size_t>(navigation_motor_spec().num_params())) {
        throw std::runtime_error("Invalid navigation motor dimensions");
    }
    for (const float weight : weights) {
        if (!physics::finite_number(weight)) throw std::runtime_error("Nonfinite navigation motor weight");
    }
}

inline void write_navigation_weights(const std::string& path, NavigationMode mode,
                                     const std::vector<float>& weights) {
    validate_navigation_weights(weights);
    checkpoint::Writer out;
    out.value<uint32_t>(0x46544e4d); // FTNM
    out.value<uint32_t>(1);
    out.value(mode);
    out.array(weights);
    checkpoint::write_atomic(path, out.data);
}

inline std::vector<float> read_navigation_weights(const std::string& path, NavigationMode mode) {
    const auto bytes = checkpoint::read(path);
    checkpoint::Reader in{bytes};
    if (in.value<uint32_t>() != 0x46544e4d || in.value<uint32_t>() != 1 ||
        in.value<NavigationMode>() != mode) throw std::runtime_error("Incompatible navigation motor");
    auto weights = in.array<float>(navigation_motor_spec().num_params());
    in.finish();
    validate_navigation_weights(weights);
    return weights;
}

inline void validate_navigation_worlds(const std::vector<NavigationWorld>& worlds) {
    if (worlds.empty() || worlds.size() > 256) throw std::runtime_error("Invalid navigation world count");
    for (const auto& world : worlds) {
        const auto& motion = world.motion;
        if (!physics::finite_body(motion.body) || motion.controls != 0 ||
            motion.history_next < 0 || motion.history_next >= turn_history_controls ||
            world.active_leg != 0 || world.leg_count < 1 || world.leg_count > navigation_max_legs) {
            throw std::runtime_error("Invalid initial navigation world");
        }
        for (const auto& position : motion.history) {
            if (!physics::finite_vector(position)) throw std::runtime_error("Invalid navigation course history");
        }
        for (float target : motion.last_targets) {
            if (!physics::finite_number(target) || fabsf(target) > physics::limit) {
                throw std::runtime_error("Invalid previous navigation actuator target");
            }
        }
        int previous_end = 0;
        for (int i = 0; i < world.leg_count; ++i) {
            const auto& leg = world.legs[i];
            if (!physics::finite_vector(leg.goal_m) || !physics::finite_vector(leg.heading) ||
                fabsf(physics::norm(leg.heading) - 1.f) > .001f || fabsf(leg.heading.y) > .001f ||
                leg.choice < Choice::Swim || leg.choice > Choice::Hold ||
                leg.end_control <= previous_end || leg.end_control > 4096) {
                throw std::runtime_error("Invalid navigation request leg");
            }
            previous_end = leg.end_control;
        }
    }
}

inline void write_navigation_worlds(const std::string& path, const std::vector<NavigationWorld>& worlds) {
    validate_navigation_worlds(worlds);
    checkpoint::Writer out;
    out.value<uint32_t>(0x46544e57); // FTNW
    out.value<uint32_t>(1);
    out.value<uint32_t>(sizeof(NavigationWorld));
    out.array(worlds);
    checkpoint::write_atomic(path, out.data);
}

inline std::vector<NavigationWorld> read_navigation_worlds(const std::string& path) {
    const auto bytes = checkpoint::read(path);
    checkpoint::Reader in{bytes};
    if (in.value<uint32_t>() != 0x46544e57 || in.value<uint32_t>() != 1 ||
        in.value<uint32_t>() != sizeof(NavigationWorld)) throw std::runtime_error("Incompatible navigation worlds");
    auto worlds = in.array<NavigationWorld>(256);
    in.finish();
    validate_navigation_worlds(worlds);
    return worlds;
}

} // namespace cudarl::flexible_tail
