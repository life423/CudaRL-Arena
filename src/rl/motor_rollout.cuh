#ifndef CUDARL_RL_MOTOR_ROLLOUT_CUH
#define CUDARL_RL_MOTOR_ROLLOUT_CUH
#include "motor/straight_task.h"
#include <cuda_runtime.h>
namespace cudarl::rl::motor_rollout {
namespace m = straight_motor;
struct Counts {
    unsigned long long controls = 0, substeps = 0;
    double cost_integral[5]{}, command_square[4]{};
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
        !m::begin_training_episode(i, episode.tilt_mixture, episode.task_config(), worlds[i], tasks[i]))
        atomicExch(fault, 1);
}
__global__ void observe(const m::World *worlds, const m::ForwardTask *tasks, float *output, int n, int *fault,
                        m::TrainingEpisode episode = {}) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float *row = output + i * m::kObservations;
    if (*fault) {
        for (int j = 0; j < m::kObservations; ++j)
            row[j] = 0;
        return;
    }
    if (!m::observe(tasks[i], worlds[i], episode.task_config(), row))
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
    const bool ok = joint_motor::step_state(worlds[i], action.command);
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
    const auto interval = m::record_control(tasks[i], worlds[i], episode.task_config());
    if (!interval.accepted) {
        atomicExch(fault, 6);
        return;
    }
    rewards[i] = interval.reward;
    dones[i] = interval.done ? 1.f : 0.f;
    const float cost[5] = {interval.costs.speed, interval.costs.lane, interval.costs.depth,
                           interval.costs.posture, interval.costs.effort};
    for (int j = 0; j < 5; ++j)
        counts[i].cost_integral[j] += physics::dt * cost[j];
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
