#pragma once

#include "rl/flexible_tail_gait.h"

namespace cudarl::flexible_tail {

inline constexpr float yaw_offset_limit = .25f;
inline constexpr float yaw_offset_slew = .02f;
inline constexpr float yaw_offset_error_scale = 5.f * pi / 180.f;

CUDARL_HD inline bool yaw_offset_turn(Choice choice) {
    return choice == Choice::TurnLeft || choice == Choice::TurnRight;
}

CUDARL_HD inline bool yaw_offset_enabled(const NavigationWorld& world) {
    return yaw_offset_turn(world.motion.request.choice) ||
        (world.active_leg > 0 && world.motion.request.choice == Choice::Swim &&
            yaw_offset_turn(world.legs[world.active_leg - 1].choice));
}

CUDARL_HD inline float yaw_offset_error(physics::V from, physics::V toward) {
    return atan2f(physics::cross(from, toward).y, physics::dot(from, toward));
}

CUDARL_HD inline void observe_yaw_offset(const NavigationWorld& world, float previous_offset,
                                         float (&out)[navigation_inputs]) {
    for (float& value : out) value = 0.f;
    if (!yaw_offset_enabled(world)) return;
    const auto& motion = world.motion;
    const auto& heading = motion.request.heading;
    const auto pose = snapshot(motion.body);
    const auto course = turn_course(motion, pose.com_m, steering_history_controls);
    const auto facing = motion.body.orientation.rotate({1.f, 0.f, 0.f});
    const auto measured = physics::norm(course) >= .0015f ? course : facing;
    out[0] = physics::clamp(yaw_offset_error(measured, heading) / yaw_offset_error_scale, -4.f, 4.f);
    out[1] = physics::clamp(yaw_offset_error(facing, heading) / yaw_offset_error_scale, -4.f, 4.f);
    out[2] = previous_offset / yaw_offset_limit;
    out[3] = physics::clamp(motion.body.velocity[4] / .5f, -3.f, 3.f);
}

CUDARL_HD inline void decode_yaw_offset(const NavigationWorld& world, const float* frozen_latent,
                                        float offset_latent, float& applied_offset,
                                        float (&targets)[motor_outputs]) {
    if (yaw_offset_enabled(world)) {
        const float requested = yaw_offset_limit * tanhf(offset_latent);
        applied_offset += physics::clamp(requested - applied_offset, -yaw_offset_slew, yaw_offset_slew);
    } else {
        applied_offset = 0.f;
    }
    const float yaw_latent = applied_offset == 0.f ? frozen_latent[0] : frozen_latent[0] + applied_offset;
    targets[0] = nn::activate(nn::Activation::Tanh, yaw_latent);
    targets[1] = nn::activate(nn::Activation::Tanh, frozen_latent[1]);
}

CUDARL_HD CUDARL_DEVICE_NOINLINE TurnStep step_yaw_offset(
    NavigationWorld& world, const float (&targets)[motor_outputs],
    const pond_geometry_v1::Parameters& pond) {
    return step_turn_control(world.motion, targets, pond);
}

inline nn::MlpSpec yaw_offset_spec() {
    return {{navigation_inputs, 1}, {nn::Activation::Identity}};
}

inline void validate_yaw_offset_weights(const std::vector<float>& weights) {
    if (weights.size() != static_cast<size_t>(yaw_offset_spec().num_params()) ||
        weights.back() != 0.f) throw std::runtime_error("Invalid yaw-offset readout dimensions or intercept");
    for (const float weight : weights) {
        if (!physics::finite_number(weight)) throw std::runtime_error("Nonfinite yaw-offset weight");
    }
}

inline void write_yaw_offset_weights(const std::string& path, const std::vector<float>& weights) {
    validate_yaw_offset_weights(weights);
    checkpoint::Writer out;
    out.value<uint32_t>(0x4f594654); // FTYO
    out.value<uint32_t>(1);
    out.array(weights);
    checkpoint::write_atomic(path, out.data);
}

inline std::vector<float> read_yaw_offset_weights(const std::string& path) {
    const auto bytes = checkpoint::read(path);
    checkpoint::Reader in{bytes};
    if (in.value<uint32_t>() != 0x4f594654 || in.value<uint32_t>() != 1) {
        throw std::runtime_error("Incompatible yaw-offset checkpoint");
    }
    auto weights = in.array<float>(yaw_offset_spec().num_params());
    in.finish();
    validate_yaw_offset_weights(weights);
    return weights;
}

} // namespace cudarl::flexible_tail
