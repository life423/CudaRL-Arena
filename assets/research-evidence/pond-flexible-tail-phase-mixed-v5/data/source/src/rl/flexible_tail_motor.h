#pragma once

#include "core/checkpoint.h"
#include "mlp_spec.h"
#include "physics/flexible_tail.h"
#include "rl/flexible_tail_choices.h"

namespace cudarl::flexible_tail {

constexpr int motor_inputs = 28;
constexpr int motor_outputs = 2;
constexpr float motor_control_s = .02f;
constexpr float motor_substep_s = .00125f;
constexpr uint32_t motor_schema = 1;

// Flat features shared by the collector and inference. Joint state and the
// previous commands provide gait feedback without a wall-clock phase input.
CUDARL_HD inline void observe_motor(const physics::BodyState& body, const Snapshot& pose,
                                   const MovementRequest& request, physics::V measured_course,
                                   physics::V current_m_s, const float (&last_targets)[motor_outputs],
                                   float (&out)[motor_inputs]) {
    const auto local_v = body.orientation.inverse(pose.com_velocity_m_s) / .02f;
    const auto up = body.orientation.inverse({0, 1, 0});
    const auto rate = body.orientation.inverse({body.velocity[3], body.velocity[4], body.velocity[5]}) / 4.f;
    const auto heading = body.orientation.inverse(request.heading);
    const auto current = body.orientation.inverse(current_m_s) / .008f;
    measured_course.y = 0;
    const float course_length = physics::norm(measured_course);
    const auto course = course_length > .0001f ? measured_course / course_length : request.heading;
    const auto error = request.origin_m - pose.com_m;
    const bool holding = request.choice == Choice::Hold;
    const float features[motor_inputs]{
        local_v.x, local_v.y, local_v.z, up.x, up.y, up.z,
        rate.x, rate.y, rate.z,
        body.angle[0], body.angle[1], body.angle[2], body.angle[3],
        body.velocity[6] / 10.f, body.velocity[7] / 10.f,
        body.velocity[8] / 15.f, body.velocity[9] / 15.f,
        heading.x, heading.z, physics::cross(course, request.heading).y,
        physics::dot(course, request.heading),
        physics::dot(request.terminal_velocity_m_s, request.heading) / requested_speed_m_s,
        error.y / .01f, holding ? physics::dot(error, request.heading) / .02f : 0.f,
        current.x, current.z, last_targets[0], last_targets[1]
    };
    for (int i = 0; i < motor_inputs; ++i) out[i] = features[i];
}

inline nn::MlpSpec motor_spec() {
    return {{motor_inputs, 32, 32, motor_outputs},
        {nn::Activation::Tanh, nn::Activation::Tanh, nn::Activation::Tanh}};
}

inline void validate_motor_weights(const std::vector<float>& weights) {
    if (weights.size() != static_cast<size_t>(motor_spec().num_params())) {
        throw std::runtime_error("Invalid flexible-tail motor dimensions");
    }
    for (const auto weight : weights) {
        if (!physics::finite_number(weight)) throw std::runtime_error("Nonfinite flexible-tail motor weight");
    }
}

inline void write_motor_weights(const std::string& path, const std::vector<float>& weights) {
    validate_motor_weights(weights);
    checkpoint::Writer out;
    out.value<uint32_t>(0x46544d31); // FTM1
    out.value(motor_schema);
    out.array(weights);
    checkpoint::write_atomic(path, out.data);
}

inline std::vector<float> read_motor_weights(const std::string& path) {
    const auto bytes = checkpoint::read(path);
    checkpoint::Reader in{bytes};
    if (in.value<uint32_t>() != 0x46544d31 || in.value<uint32_t>() != motor_schema) {
        throw std::runtime_error("Unsupported flexible-tail motor checkpoint");
    }
    auto weights = in.array<float>(static_cast<size_t>(motor_spec().num_params()));
    in.finish();
    validate_motor_weights(weights);
    return weights;
}

} // namespace cudarl::flexible_tail
