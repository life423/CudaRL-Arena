#pragma once

#include "rl/flexible_tail_navigation.h"

namespace cudarl::flexible_tail {

constexpr int launch_action_controls = 4;
constexpr float launch_course_window_s = steering_history_controls * motor_control_s;

CUDARL_HD inline void observe_launch(const NavigationWorld& world, float (&out)[navigation_inputs]) {
    observe_navigation(world, NavigationMode::Heading, out);
    const auto& motion = world.motion;
    const auto com = snapshot(motion.body).com_m;
    const auto heading = motion.request.heading;
    const auto course = turn_course(motion, com, steering_history_controls) / launch_course_window_s;
    out[motor_inputs] = physics::dot(course, heading) / requested_speed_m_s;
    out[motor_inputs + 1] = physics::cross(heading, course).y / requested_speed_m_s;
    out[motor_inputs + 2] = physics::cross(heading, com - motion.request.origin_m).y / .02f;
}

CUDARL_HD CUDARL_DEVICE_NOINLINE TurnStep step_launch(
    NavigationWorld& world, const float (&targets)[motor_outputs],
    const pond_geometry_v1::Parameters& pond) {
    auto& motion = world.motion;
    const auto start = snapshot(motion.body).com_m;
    auto result = step_turn_control(motion, targets, pond);
    if (result.fault) return result;

    const auto end = snapshot(motion.body).com_m;
    const auto heading = motion.request.heading;
    const auto course = turn_course(motion, end, steering_history_controls) / launch_course_window_s;
    const float forward = physics::dot(end - start, heading) / motor_control_s;
    const float steady_forward = physics::dot(course, heading);
    const float sideways = fminf(2.f, fabsf(physics::cross(heading, course).y) / .004f);
    const float lane_error = fminf(2.f, fabsf(physics::cross(heading, end - motion.request.origin_m).y) / .02f);
    const float tilt = fminf(3.f, result.max_tilt_deg / 10.f);
    const float depth = fminf(3.f, result.max_depth_error_m / .001f);
    const float excess_tilt = fmaxf(0.f, tilt - 1.f);
    const float excess_depth = fmaxf(0.f, depth - 1.f);
    result.reward = motor_control_s *
        (.70f * navigation_transport(forward) + 1.30f * navigation_transport(steady_forward)
            - .25f * sideways * sideways - .30f * lane_error * lane_error
            - .35f * tilt * tilt - .55f * excess_tilt * excess_tilt
            - .30f * depth * depth - .60f * excess_depth * excess_depth);
    if (!physics::finite_number(result.reward)) result.fault = 3;
    return result;
}

inline std::vector<float> initialize_launch_weights(const std::vector<float>& original) {
    auto weights = expand_navigation_weights(original);
    const auto spec = navigation_motor_spec();
    const int output_layer = spec.num_layers() - 1;
    const int pitch = 1;
    const auto pitch_weights = weights.begin() + spec.weight_offset(output_layer) + pitch * spec.in(output_layer);
    // Start with the fitted yaw. Pitch begins neutral and remains trainable.
    std::fill_n(pitch_weights, spec.in(output_layer), 0.f);
    weights[spec.bias_offset(output_layer) + pitch] = 0.f;
    return weights;
}

} // namespace cudarl::flexible_tail
