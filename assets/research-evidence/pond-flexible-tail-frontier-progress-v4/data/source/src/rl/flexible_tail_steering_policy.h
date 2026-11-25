#pragma once

#include "rl/flexible_tail_yaw_offset.h"
#include "rl/flexible_tail_beat_course.h"
#include "nn/mlp_cpu.h"

#include <array>

namespace cudarl::flexible_tail::steering {

// One steering readout, with explicit observation/checkpoint versions. The
// phase-aware policy extends the existing unused feature slots; it does not
// introduce another network, actuator, force or gait implementation.
enum class Kind : uint32_t {
    Legacy4 = 1, PhaseAware13 = 2, ProgramScalar = 3, ProgramLinear = 4, ProgramNonlinear = 5
};
enum class Study : uint32_t { None = 0, Rhythm = 1, Joint = 2, Nonlinear = 3, Speed = 4, Rate = 5, RateResponse = 6 };
struct Config {
    Study study = Study::None;
    float target_speed_m_s = .008f;
    float desired_course_rate_rad_s = 0.f;
    Material material = Material::Baseline;
};
inline Study read_study(const std::string& name) {
    if (name == "none") return Study::None;
    if (name == "rhythm") return Study::Rhythm;
    if (name == "joint") return Study::Joint;
    if (name == "nonlinear") return Study::Nonlinear;
    if (name == "speed") return Study::Speed;
    if (name == "rate") return Study::Rate;
    if (name == "rate-response") return Study::RateResponse;
    throw std::runtime_error("Unknown motor study");
}
inline const char* study_name(Study study) {
    switch (study) {
    case Study::Rhythm: return "rhythm"; case Study::Joint: return "joint";
    case Study::Nonlinear: return "nonlinear"; case Study::Speed: return "speed";
    case Study::Rate: return "rate"; case Study::RateResponse: return "rate-response"; default: return "none";
    }
}
inline Material read_material(const std::string& name) {
    if (name == "baseline" || name == "0") return Material::Baseline;
    if (name == "half-yaw" || name == "yaw-half-stiffness" || name == "1") return Material::YawHalfStiffness;
    throw std::runtime_error("Material must be baseline/0 or half-yaw/1");
}
inline Config read_program_config(const std::string& name, const std::string& material = "baseline") {
    Config c; c.study = read_study(name); c.material = read_material(material); return c;
}
CUDARL_HD inline bool valid_config(const Config& c) {
    return c.study >= Study::None && c.study <= Study::RateResponse && valid_material(c.material) &&
        physics::finite_number(c.target_speed_m_s) && c.target_speed_m_s >= .006f && c.target_speed_m_s <= .012f &&
        physics::finite_number(c.desired_course_rate_rad_s) && fabsf(c.desired_course_rate_rad_s) <= 8.f * pi / 180.f;
}
CUDARL_HD inline Kind kind_for(Study study) {
    return study == Study::Rhythm ? Kind::ProgramScalar
        : study == Study::Joint ? Kind::ProgramLinear : Kind::ProgramNonlinear;
}
CUDARL_HD inline int actions(Kind kind) {
    return kind == Kind::ProgramLinear || kind == Kind::ProgramNonlinear ? 3 : 1;
}
inline constexpr uint32_t phase_feature_schema = 1;
inline constexpr int base_parameters = 4;
inline constexpr int phase_feature_end = 13;
inline constexpr int program_feature_end = 17, program_hidden = 16, maximum_actions = 3;
inline constexpr float signed_course_scale_rad = pi / 2.f;

enum Feature : int {
    CourseError90 = 4, RootPhasePosition = 5, RootPhaseVelocity = 6,
    BodyLateralVelocity = 7, ErrorTimesSpeed = 8, ErrorTimesCos2Phase = 9,
    ErrorTimesSin2Phase = 10, ErrorSquaredTimesPhasePosition = 11,
    ErrorSquaredTimesPhaseVelocity = 12, RequestedSpeed = 13, RequestedCourseRate = 14,
    PassiveYawPosition = 15, PassiveYawVelocity = 16
};

CUDARL_HD inline uint32_t feature_schema(Kind kind) {
    return kind >= Kind::ProgramScalar ? 2u : kind == Kind::PhaseAware13 ? phase_feature_schema : 0u;
}

CUDARL_HD inline int trainable_count(Kind kind) {
    return kind == Kind::ProgramNonlinear ? 563 : kind == Kind::ProgramLinear ? 49
        : kind == Kind::ProgramScalar ? 13 : kind == Kind::PhaseAware13 ? phase_feature_end - base_parameters
        : kind == Kind::Legacy4 ? base_parameters : 0;
}

inline const char* feature_layout_name(Kind kind) {
    return kind >= Kind::ProgramScalar ? "motor-program-v1:phase13+requested-speed+course-rate+passive-q-qdot"
        : kind == Kind::PhaseAware13 ? "phase-aware-v1:base4+course90+root-phase+body-speed+signed-interactions"
        : "legacy-four-feature-offset";
}

inline nn::MlpSpec spec(Kind kind = Kind::Legacy4) {
    if (kind == Kind::ProgramNonlinear)
        return {{navigation_inputs, program_hidden, 3}, {nn::Activation::Tanh, nn::Activation::Identity}};
    if (kind == Kind::ProgramLinear) return {{navigation_inputs, 3}, {nn::Activation::Identity}};
    return yaw_offset_spec();
}

CUDARL_HD inline bool trainable_parameter(int index, Kind kind) {
    if (kind == Kind::ProgramNonlinear) return index >= 0 && index < 563;
    if (kind == Kind::ProgramScalar) return index >= 4 && index < program_feature_end;
    if (kind == Kind::ProgramLinear) {
        if (index < 0 || index >= 96) return false;
        if (index < 31) return index >= 4 && index < program_feature_end;
        if (index < 93) return index % 31 < program_feature_end;
        return index > 93; // amp/frequency intercepts; yaw intercept stays frozen zero.
    }
    return kind == Kind::PhaseAware13
        ? index >= base_parameters && index < phase_feature_end
        : kind == Kind::Legacy4 && index >= 0 && index < base_parameters;
}

// The supplied base is bound to the protected import in a training session.
// Frozen prefix and unused/bias parameters have zero gradients and moments.
CUDARL_HD inline bool valid_parameter(int index, float value, Kind kind,
                                      const float* frozen_base = nullptr) {
    const int count = kind == Kind::ProgramNonlinear ? 563 : kind == Kind::ProgramLinear ? 96 : 32;
    if (index < 0 || index >= count || !physics::finite_number(value)) return false;
    if (kind == Kind::ProgramNonlinear) return true;
    if (kind == Kind::ProgramScalar || kind == Kind::ProgramLinear) {
        if (index < 4) return frozen_base && value == frozen_base[index];
        return trainable_parameter(index, kind) || value == 0.f;
    }
    if (kind == Kind::Legacy4) return index < base_parameters || value == 0.f;
    if (kind != Kind::PhaseAware13) return false;
    if (index < base_parameters) return frozen_base && value == frozen_base[index];
    return index < phase_feature_end || value == 0.f;
}

inline void validate_weights(const std::vector<float>& weights, Kind kind,
                              const float* frozen_base = nullptr) {
    if (weights.size() != static_cast<size_t>(spec(kind).num_params()))
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
    Study study = Study::None;
    Material material = Material::Baseline;
    std::vector<float> frozen_skip;
};

inline void validate_model(const Model& model) {
    if (!valid_material(model.material)) throw std::runtime_error("Invalid bound motor material");
    validate_weights(model.weights, model.kind,
        model.kind != Kind::Legacy4 ? model.frozen_base.data() : nullptr);
    if (model.kind >= Kind::ProgramScalar && (model.study == Study::None || kind_for(model.study) != model.kind))
        throw std::runtime_error("Motor program kind/study mismatch");
    if (model.kind == Kind::ProgramNonlinear) {
        validate_weights(model.frozen_skip, Kind::ProgramLinear, model.frozen_base.data());
    } else if (!model.frozen_skip.empty()) throw std::runtime_error("Unexpected motor skip parameters");
}

// One motor policy. A nonlinear correction is added to a frozen linear skip
// inside that policy, not selected by another behavioral controller.
CUDARL_HD inline void infer(Kind kind, const float* weights, const float* skip,
                            const float* observation, float* out) {
    out[0] = out[1] = out[2] = 0.f;
    const int dim = actions(kind);
    if (kind == Kind::ProgramNonlinear) {
        float hidden[program_hidden];
        for (int row = 0; row < program_hidden; ++row) {
            float value = weights[31 * program_hidden + row];
            for (int col = 0; col < 31; ++col) value += weights[row * 31 + col] * observation[col];
            hidden[row] = tanhf(value);
        }
        for (int row = 0; row < 3; ++row) {
            float value = weights[560 + row];
            for (int col = 0; col < program_hidden; ++col) value += weights[512 + row * program_hidden + col] * hidden[col];
            if (skip) {
                float base = skip[93 + row];
                for (int col = 0; col < 31; ++col) base += skip[row * 31 + col] * observation[col];
                value += base;
            }
            out[row] = value;
        }
    } else {
        for (int row = 0; row < dim; ++row) {
            float value = weights[31 * dim + row];
            for (int col = 0; col < 31; ++col) value += weights[row * 31 + col] * observation[col];
            out[row] = value;
        }
    }
}
inline void mean(const Model& model, const float* observation, float (&out)[3]) {
    infer(model.kind, model.weights.data(), model.frozen_skip.empty() ? nullptr : model.frozen_skip.data(), observation, out);
}

inline Model initial_program_model(const Model& source, Study study) {
    validate_model(source);
    Model out;
    out.kind = kind_for(study); out.study = study; out.material = source.material;
    out.frozen_base = source.frozen_base;
    out.weights.assign(spec(out.kind).num_params(), 0.f);
    if (out.kind == source.kind) {
        out.weights = source.weights; out.frozen_skip = source.frozen_skip;
    } else if (out.kind == Kind::ProgramScalar && actions(source.kind) == 1) {
        out.weights = source.weights;
    } else if (out.kind == Kind::ProgramLinear && actions(source.kind) == 1) {
        std::copy_n(source.weights.data(), navigation_inputs, out.weights.data());
        out.weights[93] = source.weights[31];
    } else if (out.kind == Kind::ProgramNonlinear && source.kind == Kind::ProgramLinear) {
        out.frozen_skip = source.weights;
        out.weights = nn::orthogonal_params(spec(out.kind), 20261124, {1.41421356f, 0.f});
    } else throw std::runtime_error("Unsupported declared motor architecture transfer");
    validate_model(out);
    return out;
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
    if (kind >= Kind::ProgramScalar) throw std::runtime_error("Program export requires the complete bound Model");
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
    if (model.kind >= Kind::ProgramScalar) {
        validate_model(model);
        checkpoint::Writer out;
        out.value<uint32_t>(0x4f594654); out.value<uint32_t>(3);
        out.value(model.kind); out.value(feature_schema(model.kind));
        out.value<uint32_t>(navigation_inputs); out.value<uint32_t>(actions(model.kind));
        out.value(model.study); out.value(model.material); out.value(model.frozen_base);
        out.array(model.frozen_skip); out.array(model.weights);
        checkpoint::write_atomic(path, out.data);
        return;
    }
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
    } else if (version == 3) {
        model.kind = in.value<Kind>();
        if (model.kind < Kind::ProgramScalar || model.kind > Kind::ProgramNonlinear ||
            in.value<uint32_t>() != feature_schema(model.kind) ||
            in.value<uint32_t>() != navigation_inputs || in.value<uint32_t>() != static_cast<uint32_t>(actions(model.kind)))
            throw std::runtime_error("Unsupported motor program layout");
        model.study = in.value<Study>(); model.material = in.value<Material>();
        model.frozen_base = in.value<decltype(model.frozen_base)>();
        model.frozen_skip = in.array<float>(96); model.weights = in.array<float>(563);
    } else {
        throw std::runtime_error("Unsupported steering policy checkpoint version");
    }
    in.finish();
    validate_model(model);
    return model;
}

CUDARL_HD inline void observe_phase(const NavigationWorld& world, float applied_offset,
                                     const BeatCourseSample& sample,
                                     float (&out)[navigation_inputs], bool force_active = false) {
    observe_yaw_offset(world, applied_offset, out);
    if (!yaw_offset_enabled(world) && !force_active) return;
    if (force_active && !yaw_offset_enabled(world)) {
        const auto facing = world.motion.body.orientation.rotate({1.f, 0.f, 0.f});
        out[1] = physics::clamp(yaw_offset_error(facing, world.motion.request.heading) / yaw_offset_error_scale, -4.f, 4.f);
        out[2] = applied_offset / yaw_offset_limit;
        out[3] = physics::clamp(world.motion.body.velocity[4] / .5f, -3.f, 3.f);
    }
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

CUDARL_HD inline bool program_enabled(const NavigationWorld& world, const Config& config) {
    return world.motion.request.choice != Choice::Hold && (yaw_offset_enabled(world) ||
        config.study >= Study::Speed ||
        fabsf(config.target_speed_m_s - requested_speed_m_s) > 1.e-7f || config.desired_course_rate_rad_s != 0.f);
}
CUDARL_HD inline void observe_program(const NavigationWorld& world, float applied_offset,
                                      const BeatCourseSample& sample, const Config& config,
                                      float (&out)[navigation_inputs]) {
    const bool active = program_enabled(world, config);
    observe_phase(world, applied_offset, sample, out, active);
    if (!active) return;
    out[RequestedSpeed] = config.target_speed_m_s / requested_speed_m_s - 1.f;
    out[RequestedCourseRate] = config.desired_course_rate_rad_s / (8.f * pi / 180.f);
    out[PassiveYawPosition] = world.motion.body.angle[2] / .2f;
    out[PassiveYawVelocity] = world.motion.body.velocity[8] / (4.f * pi * .2f);
}

} // namespace cudarl::flexible_tail::steering
