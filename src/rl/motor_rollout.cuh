#ifndef CUDARL_RL_MOTOR_ROLLOUT_CUH
#define CUDARL_RL_MOTOR_ROLLOUT_CUH
#include "motor/straight_task.h"
#include <cuda_runtime.h>
namespace cudarl::rl::motor_rollout {
namespace m = straight_motor;
struct Counts {
    unsigned long long controls = 0, substeps = 0;
};
__global__ void finite_parameters(const float *params, int count, int *fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && !physics::finite_number(params[i]))
        atomicExch(fault, 7);
}
// Persistent device worlds and counters. Only episode state resets; accounting
// and a sticky fault remain live for the entire bounded training run.
__global__ void initialize(m::World *worlds, m::ForwardTask *tasks, int n, int *fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || *fault)
        return;
    worlds[i] = m::initial_world(i % 2 ? 3 : 0);
    if (!m::begin_forward(worlds[i], m::config_for_speed(.008f), tasks[i]))
        atomicExch(fault, 1);
}
__global__ void observe(const m::World *worlds, const m::ForwardTask *tasks, float *output, int n,
                        int *fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float *row = output + i * 42;
    if (*fault) {
        for (int j = 0; j < 42; ++j)
            row[j] = 0;
        return;
    }
    if (!m::observe(tasks[i], worlds[i], m::config_for_speed(.008f), row))
        atomicExch(fault, 2);
}
__global__ void advance(m::World *worlds, m::ForwardTask *tasks, const float *latent, float *rewards,
                        float *dones, Counts *counts, int n, int *fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    rewards[i] = 0;
    dones[i] = 1;
    if (*fault)
        return;
    m::Action action;
    if (!m::action_from_latent(latent + i * 2, action)) {
        atomicExch(fault, 3);
        return;
    }
    const auto before = worlds[i].attempted_substeps;
    ++counts[i].controls;
    const bool ok =
        stroke_motor::step_state(worlds[i], action.command, stroke_motor::ActionMapping::SignedCadence);
    const auto attempted = worlds[i].attempted_substeps - before;
    counts[i].substeps += attempted;
    if (!ok || attempted != 10) {
        atomicExch(fault, 4);
        return;
    }
    for (float torque : worlds[i].physical.actual_torques)
        if (!physics::finite_number(torque) || fabsf(torque) > 1) {
            atomicExch(fault, 5);
            return;
        }
    const auto interval = m::record_control(tasks[i], worlds[i], m::config_for_speed(.008f));
    if (!interval.accepted) {
        atomicExch(fault, 6);
        return;
    }
    rewards[i] = interval.reward;
    dones[i] = interval.done ? 1.f : 0.f;
}
} // namespace cudarl::rl::motor_rollout
#endif
