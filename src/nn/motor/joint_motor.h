#ifndef CUDARL_JOINT_MOTOR_H
#define CUDARL_JOINT_MOTOR_H
#include "motor/movement_model.h"
#include "motor/state.h"
namespace cudarl::joint_motor {
namespace p = physics;
namespace sim = simulation;
namespace base = motor;
namespace model = movement;
constexpr int kObservations = 40, kActions = 4;
constexpr uint32_t kActionVersion = 3;
struct Action {
    float latent[4]{}, command[4]{};
};
struct World {
    sim::WorldState physical{};
    contact::Arena arena{};
    float reference_rate[4]{};
    uint64_t attempted_substeps = 0;
    double signed_work = 0, positive_work = 0, fluid_loss = 0, numerical_loss = 0, external_work = 0,
           last_energy_error = 0;
    int failure_code = 0;
};
CUDARL_HD inline bool in_domain(const World &w, float clearance_m = .01f) {
    if (!base::in_domain(w.physical, clearance_m, w.arena) ||
        model::clearance(w.physical.body, true, w.arena) < clearance_m)
        return false;
    for (float value : w.reference_rate)
        if (!p::finite_number(value) || fabsf(value) > 12)
            return false;
    return true;
}
CUDARL_HD inline bool action_from_latent(const float *z, Action &a) {
    if (!z)
        return false;
    Action next;
    for (int j = 0; j < 4; ++j) {
        if (!p::finite_number(z[j]))
            return false;
        next.latent[j] = z[j];
        next.command[j] = tanhf(z[j]);
    }
    a = next;
    return true;
}
// Four independent bounded ANGLE targets [rad], not torques or a gait. Filter
// angle is physical.held_targets; filter rate is explicit observed state.
template <int NaturalFrequencyHz = 1>
CUDARL_HD inline bool filter_targets(float *angle, float *rate, const float *command,
                                     model::Reference &output) {
    if (!angle || !rate || !command)
        return false;
    model::Reference next;
    static_assert(NaturalFrequencyHz == 1 || NaturalFrequencyHz == 3,
                  "Only default or registered harness bandwidth");
    const float h = .005f, omega = 2 * model::pi * NaturalFrequencyHz;
    for (int j = 0; j < 4; ++j) {
        if (!p::finite_number(command[j]) || fabsf(command[j]) > 1 || !p::finite_number(angle[j]) ||
            fabsf(angle[j]) > p::limit || !p::finite_number(rate[j]) || fabsf(rate[j]) > 12)
            return false;
        next.acceleration[j] = omega * omega * (command[j] * p::limit - angle[j]) - 2 * omega * rate[j];
        next.rate[j] = rate[j] + h * next.acceleration[j];
        next.angle[j] = angle[j] + h * next.rate[j];
        if (!p::finite_number(next.acceleration[j]) || fabsf(next.acceleration[j]) > 200 ||
            fabsf(next.rate[j]) > 12 || fabsf(next.angle[j]) > p::limit)
            return false;
    }
    for (int j = 0; j < 4; ++j) {
        angle[j] = next.angle[j];
        rate[j] = next.rate[j];
    }
    output = next;
    return true;
}
template <int NaturalFrequencyHz = 1>
CUDARL_HD inline bool step_state(World &w, const float *command, model::Telemetry *diagnostic = nullptr,
                                 p::BodyState *before_substep = nullptr, p::BodyState *attempted = nullptr) {
    if (!command || w.physical.paused || !in_domain(w))
        return false;
    for (int j = 0; j < 4; ++j)
        if (!p::finite_number(command[j]) || fabsf(command[j]) > 1)
            return false;
    auto &state = w.physical;
    state.telemetry = {};
    state.mean_squared_torque = 0;
    for (int j = 0; j < 4; ++j)
        state.raw_action[j] = command[j];
    for (int k = 0; k < 10; ++k) {
        if (before_substep)
            *before_substep = state.body;
        const float h = .005f;
        model::Reference reference;
        if (!filter_targets<NaturalFrequencyHz>(state.held_targets, w.reference_rate, command, reference)) {
            w.failure_code = 10;
            state.invalid = true;
            return false;
        }
        model::Telemetry telemetry;
        ++w.attempted_substeps;
        const bool advanced =
            model::substep(state.body, reference, h, state.fluid, telemetry, true, w.arena, attempted);
        if (diagnostic)
            *diagnostic = telemetry;
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
        w.last_energy_error =
            model::kinetic(state.body) - w.signed_work + w.fluid_loss + w.numerical_loss - w.external_work;
        if (fabs(w.last_energy_error) > .02 * (w.positive_work + w.fluid_loss) + 2e-10) {
            w.failure_code = 9;
            state.invalid = true;
            return false;
        }
        ++state.substeps;
        ++state.telemetry.executed_substeps;
        state.time_seconds = static_cast<double>(state.substeps) * h;
        state.telemetry.minimum_floor_gap = fminf(state.telemetry.minimum_floor_gap, telemetry.minimum_gap);
        state.telemetry.minimum_other_gap = fminf(state.telemetry.minimum_other_gap, telemetry.minimum_gap);
        for (int j = 0; j < 4; ++j) {
            state.held_targets[j] = reference.angle[j];
            const float normalized = telemetry.torque[j] / p::torque;
            state.last_actual_torques[j] = normalized;
            state.actual_torques[k * 4 + j] = normalized;
            state.requested_torques[k * 4 + j] = telemetry.requested[j];
            state.mean_squared_torque += normalized * normalized;
        }
    }
    state.mean_squared_torque /= 40;
    ++state.control_steps;
    return in_domain(w);
}
} // namespace cudarl::joint_motor
#endif
