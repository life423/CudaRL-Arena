#pragma once

#include "rl/flexible_tail_yaw_offset.h"
#include "rl/flexible_tail_beat_course.h"

#include <array>

namespace cudarl::flexible_tail::steering {

// One steering readout, with explicit observation/checkpoint versions. The
// phase-aware policy extends the existing unused feature slots; it does not
// introduce another network, actuator, force or gait implementation.
enum class Kind : uint32_t { Legacy4 = 1, PhaseAware13 = 2 };
inline constexpr uint32_t phase_feature_schema = 1;
inline constexpr int base_parameters = 4;
inline constexpr int phase_feature_end = 13;
inline constexpr float signed_course_scale_rad = pi / 2.f;

enum Feature : int {
    CourseError90 = 4, RootPhasePosition = 5, RootPhaseVelocity = 6,
    BodyLateralVelocity = 7, ErrorTimesSpeed = 8, ErrorTimesCos2Phase = 9,
    ErrorTimesSin2Phase = 10, ErrorSquaredTimesPhasePosition = 11,
    ErrorSquaredTimesPhaseVelocity = 12
};

CUDARL_HD inline uint32_t feature_schema(Kind kind) {
    return kind == Kind::PhaseAware13 ? phase_feature_schema : 0u;
}

CUDARL_HD inline int trainable_count(Kind kind) {
    return kind == Kind::PhaseAware13 ? phase_feature_end - base_parameters
        : kind == Kind::Legacy4 ? base_parameters : 0;
}

inline const char* feature_layout_name(Kind kind) {
    return kind == Kind::PhaseAware13 ? "phase-aware-v1:base4+course90+root-phase+body-speed+signed-interactions"
        : "legacy-four-feature-offset";
}

inline nn::MlpSpec spec() { return yaw_offset_spec(); }

CUDARL_HD inline bool trainable_parameter(int index, Kind kind) {
    return kind == Kind::PhaseAware13
        ? index >= base_parameters && index < phase_feature_end
        : kind == Kind::Legacy4 && index >= 0 && index < base_parameters;
}

// The supplied base is bound to the protected import in a training session.
// Frozen prefix and unused/bias parameters have zero gradients and moments.
CUDARL_HD inline bool valid_parameter(int index, float value, Kind kind,
                                      const float* frozen_base = nullptr) {
    if (index < 0 || index > navigation_inputs || !physics::finite_number(value)) return false;
    if (kind == Kind::Legacy4) return index < base_parameters || value == 0.f;
    if (kind != Kind::PhaseAware13) return false;
    if (index < base_parameters) return frozen_base && value == frozen_base[index];
    return index < phase_feature_end || value == 0.f;
}

inline void validate_weights(const std::vector<float>& weights, Kind kind,
                              const float* frozen_base = nullptr) {
    if (weights.size() != static_cast<size_t>(spec().num_params()))
        throw std::runtime_error("Invalid steering policy dimensions");
    for (int index = 0; index < static_cast<int>(weights.size()); ++index) {
        if (!valid_parameter(index, weights[index], kind, frozen_base))
            throw std::runtime_error("Steering policy violates its frozen prefix or feature layout");
    }
}

struct Model {
    Kind kind = Kind::Legacy4;
    std::array<float, base_parameters> frozen_base{};
    std::vector<float> weights;
};

inline void validate_model(const Model& model) {
    validate_weights(model.weights, model.kind,
        model.kind == Kind::PhaseAware13 ? model.frozen_base.data() : nullptr);
}

inline Model initial_phase_model(const std::vector<float>& source) {
    validate_weights(source, Kind::Legacy4);
    Model model;
    model.kind = Kind::PhaseAware13;
    model.weights = source;
    for (int index = 0; index < base_parameters; ++index) model.frozen_base[index] = source[index];
    // Legacy validation already established exact zeros in all added slots.
    // Keeping those zeros preserves the initial function at every turn angle.
    validate_model(model);
    return model;
}

inline std::vector<float> initial_phase_weights(const std::vector<float>& source) {
    return initial_phase_model(source).weights;
}

inline void write_model(const std::string& path, const std::vector<float>& weights,
                         Kind kind, const float* frozen_base = nullptr) {
    validate_weights(weights, kind, frozen_base);
    if (kind == Kind::Legacy4) {
        write_yaw_offset_weights(path, weights);
        return;
    }
    checkpoint::Writer out;
    out.value<uint32_t>(0x4f594654); // FTYO, explicitly incompatible version2.
    out.value<uint32_t>(2);
    out.value(kind);
    out.value(phase_feature_schema);
    out.value<uint32_t>(navigation_inputs);
    out.value<uint32_t>(1);
    for (int index = 0; index < base_parameters; ++index) out.value(frozen_base[index]);
    out.array(weights);
    checkpoint::write_atomic(path, out.data);
}

inline void write_model(const std::string& path, const Model& model) {
    write_model(path, model.weights, model.kind,
        model.kind == Kind::PhaseAware13 ? model.frozen_base.data() : nullptr);
}

inline Model read_model(const std::string& path) {
    const auto bytes = checkpoint::read(path);
    checkpoint::Reader in{bytes};
    if (in.value<uint32_t>() != 0x4f594654)
        throw std::runtime_error("Unsupported steering policy checkpoint magic");
    const auto version = in.value<uint32_t>();
    Model model;
    if (version == 1) {
        model.kind = Kind::Legacy4;
        model.weights = in.array<float>(spec().num_params());
        if (model.weights.size() >= base_parameters)
            for (int index = 0; index < base_parameters; ++index) model.frozen_base[index] = model.weights[index];
    } else if (version == 2) {
        model.kind = in.value<Kind>();
        if (model.kind != Kind::PhaseAware13 || in.value<uint32_t>() != phase_feature_schema ||
            in.value<uint32_t>() != navigation_inputs || in.value<uint32_t>() != 1)
            throw std::runtime_error("Unsupported phase-aware steering feature layout");
        for (float& value : model.frozen_base) value = in.value<float>();
        model.weights = in.array<float>(spec().num_params());
    } else {
        throw std::runtime_error("Unsupported steering policy checkpoint version");
    }
    in.finish();
    validate_model(model);
    return model;
}

CUDARL_HD inline void observe_phase(const NavigationWorld& world, float applied_offset,
                                     const BeatCourseSample& sample,
                                     float (&out)[navigation_inputs]) {
    observe_yaw_offset(world, applied_offset, out);
    if (!yaw_offset_enabled(world)) return;
    // Preserve the old whole-beat observation in the frozen first four paths.
    const float error = sample.reliable
        ? yaw_offset_error(sample.velocity_m_s, world.motion.request.heading) : 0.f;
    out[0] = sample.reliable
        ? physics::clamp(error / yaw_offset_error_scale, -4.f, 4.f) : 0.f;
    const float e90 = error / signed_course_scale_rad;
    const auto& body = world.motion.body;
    const float yaw = body.angle[0];
    const float rate = body.velocity[6] / gait_nominal_rate;
    const float radius = sqrtf(yaw * yaw + rate * rate);
    const float scale = fmaxf(radius, gait_phase_floor_rad);
    const float phase_position = yaw / scale;
    const float phase_velocity = rate / scale;
    const auto velocity = sample.fresh ? sample.velocity_m_s : physics::V{};
    const auto local_velocity = body.orientation.inverse(velocity);
    const float speed_ratio = physics::norm(velocity) / requested_speed_m_s;

    out[CourseError90] = e90;
    out[RootPhasePosition] = phase_position;
    out[RootPhaseVelocity] = phase_velocity;
    out[BodyLateralVelocity] = local_velocity.z / requested_speed_m_s;
    out[ErrorTimesSpeed] = e90 * speed_ratio;
    // Even phase harmonics times signed error, and squared error times odd
    // phase, keep every new term odd under a full left/right reflection.
    out[ErrorTimesCos2Phase] = e90 * (phase_position * phase_position - phase_velocity * phase_velocity);
    out[ErrorTimesSin2Phase] = e90 * (2.f * phase_position * phase_velocity);
    out[ErrorSquaredTimesPhasePosition] = e90 * e90 * phase_position;
    out[ErrorSquaredTimesPhaseVelocity] = e90 * e90 * phase_velocity;
}

} // namespace cudarl::flexible_tail::steering
