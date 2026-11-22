#ifndef CUDARL_RL_MOTOR_ROLLOUT_CUH
#define CUDARL_RL_MOTOR_ROLLOUT_CUH
#include "motor/straight_task.h"
#ifdef CUDARL_REST_HOLD_STUDY
#include "motor/hold_assessment.h"
#endif
#include <cuda_runtime.h>
namespace cudarl::rl::motor_rollout {
namespace m = straight_motor;
struct Counts {
    physics::BodyState last_before{}, rejected_candidate{};
    float rejected_gap_m = 0;
    unsigned long long controls = 0, substeps = 0;
    double cost_integral[5]{}, command_square[4]{};
    double reward_detail_integral[4]{}; // horizontal, recovery, roll, pitch; aggregate decomposition.
    float joint_min[4]{}, joint_max[4]{}, maximum_tilt_deg = 0;
    unsigned long long saturated_axis_substeps = 0;
    float episode_forward_mm = 0;
    double latent_square[4]{}, decoded_noise_square[4]{}, filtered_square[4]{};
};
__global__ void set_history(float *history, int n, bool valid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        history[i] = valid ? 1.f : 0.f;
}
__global__ void finite_parameters(const float *params, int count, int *fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && !physics::finite_number(params[i]))
        atomicExch(fault, 7);
}
// Persistent device worlds and counters. Only episode state resets; accounting
// and a sticky fault remain live for the entire bounded training run.
__global__ void initialize(m::World *worlds, m::ForwardTask *tasks, int n, int *fault,
                           m::TrainingEpisode episode = {}) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || *fault)
        return;

    if (!episode.valid() ||
        !m::begin_training_episode(i, episode.tilt_mixture, episode.task_config(0, i), worlds[i], tasks[i],
                                   episode.arena(), episode.initial_vertical_offset(i)))
        atomicExch(fault, 1);
    if (!*fault && episode.rest_hold && episode.hold_episode(i) &&
        !m::update_velocity_request(worlds[i], episode.task_config(0,i), tasks[i]))
        atomicExch(fault, 1);
}
#ifdef CUDARL_REST_HOLD_RESET_STUDY
constexpr int rest_hold_reset_controls = 20;
__global__ void rest_hold_reset_boundary(m::World *worlds, m::ForwardTask *tasks, float *previous_obs,
                                         float *previous_action, float *history, float *pre_action_speed,
                                         int *fault, int slot, m::TrainingEpisode episode) {
    const int i = threadIdx.x;
    if (i >= 32 || *fault)
        return;
    const bool reset = slot == 0 || (episode.hold_episode(i) && slot % rest_hold_reset_controls == 0);
    history[i] = reset ? 0.f : 1.f;
    if (reset) {
        if (slot != 0) {
            const auto config = episode.task_config(0,i);
            if (tasks[i].controls != rest_hold_reset_controls ||
                worlds[i].physical.control_steps != rest_hold_reset_controls ||
                worlds[i].physical.substeps != rest_hold_reset_controls * 10 ||
                !m::begin_training_episode(i, episode.tilt_mixture, config, worlds[i], tasks[i],
                                           episode.arena(), episode.initial_vertical_offset(i)) ||
                !m::update_velocity_request(worlds[i], config, tasks[i])) {
                atomicExch(fault, 8);
                return;
            }
        }
        for (int j = 0; j < 41; ++j)
            previous_obs[i * 41 + j] = 0;
        for (int j = 0; j < 4; ++j)
            previous_action[i * 4 + j] = 0;
    }
    pre_action_speed[i] = physics::norm(m::measure(tasks[i], worlds[i]).velocity_m_s);
}
#endif
__global__ void observe(const m::World *worlds, m::ForwardTask *tasks, float *output, int n, int *fault,
                        m::TrainingEpisode episode = {}) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const int width = episode.appended_hold ? 41 : m::kObservations;
    float *row = output + i * width;
    if (*fault) {
        for (int j = 0; j < width; ++j)
            row[j] = 0;
        return;
    }
    const auto config = episode.task_config(tasks[i].controls, i);
    if (((episode.stopping || episode.holding) && !m::update_velocity_request(worlds[i], config, tasks[i])) ||
        !(episode.appended_hold ? m::observe_appended(tasks[i], worlds[i], config, row)
                                : m::observe(tasks[i], worlds[i], config, row)))
        atomicExch(fault, 2);
}
__global__ void advance(m::World *worlds, m::ForwardTask *tasks, const float *latent, float *rewards,
                        float *dones, Counts *counts, int n, int *fault, const float *means = nullptr,
                        m::TrainingEpisode episode = {}, float *cost_output = nullptr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    rewards[i] = 0;
    dones[i] = 1;
    if (*fault)
        return;
    if (tasks[i].controls == 0) {
        for (int j = 0; j < 4; ++j) {
            counts[i].latent_square[j] = 0;
            counts[i].decoded_noise_square[j] = 0;
            counts[i].filtered_square[j] = 0;
        }
    }
    m::Action action;
    if (!m::action_from_latent(latent + i * m::kActions, action)) {
        atomicExch(fault, 3);
        return;
    }
    const auto before = worlds[i].attempted_substeps;
    ++counts[i].controls;
    movement::Telemetry telemetry;
    physics::BodyState before_state{}, candidate{};
    const bool ok = joint_motor::step_state(worlds[i], action.command, &telemetry, &before_state, &candidate);
    if (!ok) {
        counts[i].last_before = before_state;
        counts[i].rejected_candidate = candidate;
        counts[i].rejected_gap_m = telemetry.minimum_gap;
    }
    const auto attempted = worlds[i].attempted_substeps - before;
    counts[i].substeps += attempted;
    if (!ok || attempted != 10) {
        atomicExch(fault, 4);
        return;
    }
    for (float torque : worlds[i].physical.actual_torques) {
        if (!physics::finite_number(torque) || fabsf(torque) > 1) {
            atomicExch(fault, 5);
            return;
        }
        if (fabsf(torque) >= .999999f)
            ++counts[i].saturated_axis_substeps;
    }
#ifdef CUDARL_REST_HOLD_STUDY
    const auto interval = m::record_hold_control(tasks[i], worlds[i], episode.task_config(tasks[i].controls, i));
#else
    const auto interval = m::record_control(tasks[i], worlds[i], episode.task_config(tasks[i].controls, i));
#endif
    if (!interval.accepted) {
        atomicExch(fault, 6);
        return;
    }
    const float gain = episode.reward_gain(tasks[i].controls - 1, i);
    rewards[i] = interval.reward * gain;
#ifdef CUDARL_REST_HOLD_RESET_STUDY
    const bool segment_done = episode.hold_episode(i) && tasks[i].controls == rest_hold_reset_controls;
    dones[i] = (interval.done || segment_done) ? 1.f : 0.f;
#else
    dones[i] = interval.done ? 1.f : 0.f;
#endif
    const float cost[5] = {gain * interval.costs.speed, gain * interval.costs.lane, gain * interval.costs.depth,
                           gain * interval.costs.posture, gain * interval.costs.effort};
    for (int j = 0; j < 5; ++j)
        counts[i].cost_integral[j] += physics::dt * cost[j];
    const float detail[4] = {gain * interval.costs.horizontal, gain * interval.costs.recovery, gain * interval.costs.roll,
                             gain * interval.costs.pitch};
    for (int j = 0; j < 4; ++j)
        counts[i].reward_detail_integral[j] += physics::dt * detail[j];
    if (cost_output)
        for (int j = 0; j < 5; ++j)
            cost_output[i * 5 + j] = cost[j];
    for (int j = 0; j < 4; ++j) {
        counts[i].command_square[j] += action.command[j] * action.command[j];
        counts[i].joint_min[j] = fminf(counts[i].joint_min[j], worlds[i].physical.body.angle[j]);
        counts[i].joint_max[j] = fmaxf(counts[i].joint_max[j], worlds[i].physical.body.angle[j]);
    }
    counts[i].maximum_tilt_deg = fmaxf(counts[i].maximum_tilt_deg, interval.metrics.tilt_deg);
    counts[i].episode_forward_mm = interval.metrics.forward_m * 1000;
    for (int j = 0; j < 4; ++j)
        counts[i].filtered_square[j] +=
            worlds[i].physical.held_targets[j] * worlds[i].physical.held_targets[j];
    if (means)
        for (int j = 0; j < 4; ++j) {
            const float latent_noise = latent[i * 4 + j] - means[i * 4 + j];
            const float decoded = action.command[j] - tanhf(means[i * 4 + j]);
            counts[i].latent_square[j] += latent_noise * latent_noise;
            counts[i].decoded_noise_square[j] += decoded * decoded;
        }
}
} // namespace cudarl::rl::motor_rollout
#endif
