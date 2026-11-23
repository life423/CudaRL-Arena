#ifndef CUDARL_CURRENT_MOTOR_V2_H
#define CUDARL_CURRENT_MOTOR_V2_H
#include "motor/current_model_v2.h"
#include "motor/joint_motor.h"

namespace cudarl::current_motor_v2 {
namespace p = physics;
namespace legacy = joint_motor;
namespace model = current_model_v2;
using World = legacy::World;
using Action = legacy::Action;
using legacy::action_from_latent;
constexpr uint32_t kVersion = 2, kActionVersion = legacy::kActionVersion;
constexpr int kActions = legacy::kActions, kSubstepsPerControl = legacy::kSubstepsPerControl;
constexpr float kMaximumCurrent = .0025f;
CUDARL_HD inline bool in_domain(const World& w, float clearance_m = .01f) {
    const auto current = w.physical.current;
    if (!p::finite_vector(current) || current.y != 0 || p::norm(current) > kMaximumCurrent) return false;
    if (!model::moving_water(current)) return legacy::in_domain(w, clearance_m);
    // Reuse every historical physical-state guard, exempting only this explicit
    // current field. This is a validation copy; the actual world is never reset.
    auto calm_validation = w.physical;
    calm_validation.current = {};
    if (!motor::in_domain(calm_validation, clearance_m, w.arena) ||
        movement::clearance(w.physical.body, true, w.arena) < clearance_m) return false;
    for (float value : w.reference_rate)
        if (!p::finite_number(value) || fabsf(value) > 12) return false;
    return true;
}

template<int NaturalFrequencyHz = 1>
CUDARL_HD inline bool step_state(World& w, const float* command,
                                  model::Telemetry* diagnostic = nullptr,
                                  p::BodyState* before_substep = nullptr,
                                  p::BodyState* attempted = nullptr) {
    if (!model::moving_water(w.physical.current)) {
        movement::Telemetry original;
        const bool advanced = legacy::step_state<NaturalFrequencyHz>(w, command, diagnostic ? &original : nullptr,
                                                                     before_substep, attempted);
        if (diagnostic) {
            *diagnostic = {};
            static_cast<movement::Telemetry&>(*diagnostic) = original;
            diagnostic->gravity_work = original.external_work;
        }
        return advanced;
    }
    if (!command || w.physical.paused || !cudarl::current_motor_v2::in_domain(w)) return false;
    for (int j = 0; j < 4; ++j)
        if (!p::finite_number(command[j]) || fabsf(command[j]) > 1) return false;
    auto& state = w.physical;
    state.telemetry = {};
    state.mean_squared_torque = 0;
    for (int j = 0; j < 4; ++j) state.raw_action[j] = command[j];
    for (int k = 0; k < kSubstepsPerControl; ++k) {
        if (before_substep) *before_substep = state.body;
        const float h = .005f;
        movement::Reference reference;
        if (!legacy::filter_targets<NaturalFrequencyHz>(state.held_targets, w.reference_rate, command, reference)) {
            w.failure_code = 10;
            state.invalid = true;
            return false;
        }
        model::Telemetry telemetry;
        ++w.attempted_substeps;
        const bool advanced = model::substep(state.body, reference, h, state.fluid, telemetry,
                                             state.current, w.arena, attempted);
        if (diagnostic) *diagnostic = telemetry;
        if (!advanced) {
            w.failure_code = telemetry.failure_code;
            state.invalid = true;
            return false;
        }
        w.signed_work += telemetry.signed_work;
        w.positive_work += telemetry.positive_work;
        w.fluid_loss += telemetry.fluid_loss;
        w.numerical_loss += telemetry.numerical_loss;
        w.external_work += telemetry.external_work;
        w.last_energy_error = movement::kinetic(state.body) - w.signed_work + w.fluid_loss +
                              w.numerical_loss - w.external_work;
        if (fabs(w.last_energy_error) > .02*(w.positive_work+w.fluid_loss)+2e-10) {
            w.failure_code = 9;
            state.invalid = true;
            return false;
        }
        ++state.substeps;
        ++state.telemetry.executed_substeps;
        state.time_seconds = static_cast<double>(state.substeps)*h;
        state.telemetry.minimum_floor_gap = fminf(state.telemetry.minimum_floor_gap, telemetry.minimum_gap);
        state.telemetry.minimum_other_gap = fminf(state.telemetry.minimum_other_gap, telemetry.minimum_gap);
        for (int j = 0; j < 4; ++j) {
            state.held_targets[j] = reference.angle[j];
            const float normalized = telemetry.torque[j]/p::torque;
            state.last_actual_torques[j] = normalized;
            state.actual_torques[k*4+j] = normalized;
            state.requested_torques[k*4+j] = telemetry.requested[j];
            state.mean_squared_torque += normalized*normalized;
        }
    }
    state.mean_squared_torque /= 40;
    ++state.control_steps;
    return cudarl::current_motor_v2::in_domain(w);
}
} // namespace cudarl::current_motor_v2
#endif
