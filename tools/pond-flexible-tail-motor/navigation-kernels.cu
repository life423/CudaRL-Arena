#include "navigation-kernels.h"

#include "cuda/cuda_memory.h"
#include "habitat_config.h"
#include "rl/flexible_tail_yaw_offset.h"

#include <climits>
#include <cmath>

namespace cudarl::flexible_tail::navigation_cuda {
namespace {

constexpr int world_threads = 32;

// Every lane reads the same status from before the launch.
__global__ void snapshot_fault(int* first_fault) {
    first_fault[1] = first_fault[0];
}

__global__ void observe_worlds(NavigationWorld* worlds, int count, NavigationMode mode,
                               float* observations, int* first_fault, const float* offsets,
                               float* frozen_observations) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;

    float state[navigation_inputs]{};
    const bool valid = first_fault[1] == INT_MAX;
    if (valid) {
        auto& world = worlds[lane];
        prepare_navigation(world);
        if (mode == NavigationMode::YawOffset) {
            observe_yaw_offset(world, offsets[lane], state);
        } else if (mode == NavigationMode::Gait) {
            observe_gait(world, state);
        } else if (mode == NavigationMode::Launch) {
            observe_launch(world, state);
        } else {
            observe_navigation(world, mode, state);
        }
    }
    auto* row = observations + lane * navigation_inputs;
    for (int column = 0; column < navigation_inputs; ++column) {
        const float value = state[column];
        if (!isfinite(value)) atomicMin(first_fault, lane);
        row[column] = isfinite(value) ? value : 0.f;
    }
    if (frozen_observations) {
        if (valid) observe_gait(worlds[lane], state);
        auto* frozen_row = frozen_observations + lane * navigation_inputs;
        for (int column = 0; column < navigation_inputs; ++column) {
            const float value = state[column];
            if (!isfinite(value)) atomicMin(first_fault, lane);
            frozen_row[column] = isfinite(value) ? value : 0.f;
        }
    }
}

__global__ void step_worlds(NavigationWorld* worlds, int count, NavigationMode mode,
                            int action_dimensions, const float* raw_actions, float* rewards, uint32_t* attempted,
                            uint32_t* accepted, int* first_fault, int* physics_faults,
                            pond_geometry_v1::Parameters habitat, float* offsets, const float* frozen_latents) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;

    rewards[lane] = 0.f;
    if (first_fault[1] != INT_MAX) return;

    const auto* latent = raw_actions + lane * action_dimensions;
    const float pitch = action_dimensions == 1 ? 0.f : latent[1];
    if (!isfinite(latent[0]) || !isfinite(pitch)) {
        atomicMin(first_fault, lane);
        return;
    }
    auto& world = worlds[lane];
    const bool learn_offset = mode == NavigationMode::YawOffset;
    float targets[motor_outputs]{};
    if (learn_offset) {
        decode_yaw_offset(world, frozen_latents + lane * motor_outputs, latent[0], offsets[lane], targets);
    } else {
        targets[0] = tanhf(latent[0]);
        targets[1] = tanhf(pitch);
    }
    const bool straight_pilot = mode == NavigationMode::Launch || mode == NavigationMode::Gait;
    const int controls = straight_pilot || learn_offset ? launch_action_controls : 1;
    float reward = 0.f, discount = 1.f;
    for (int control = 0; control < controls; ++control) {
        if (control > 0) prepare_navigation(world);
        TurnStep step;
        if (learn_offset) {
            step = step_yaw_offset(world, targets, habitat);
        } else if (straight_pilot) {
            step = step_launch(world, targets, habitat);
        } else {
            step = step_navigation<false>(world, mode, targets, habitat);
        }
        attempted[lane] += static_cast<uint32_t>(step.attempted_substeps);
        accepted[lane] += static_cast<uint32_t>(step.accepted_substeps);
        if (step.fault || !isfinite(step.reward)) {
            physics_faults[lane] = step.fault;
            atomicMin(first_fault, lane);
            return;
        }
        reward += discount * step.reward;
        discount *= control_gamma;
    }
    rewards[lane] = reward;
}

} // namespace

void observe(NavigationWorld* worlds, int count, NavigationMode mode, float* observations,
             int* first_fault, cudaStream_t stream, const float* offsets, float* frozen_observations) {
    snapshot_fault<<<1, 1, 0, stream>>>(first_fault);
    CUDA_CHECK_THROW(cudaGetLastError());
    observe_worlds<<<(count + world_threads - 1) / world_threads, world_threads, 0, stream>>>(
        worlds, count, mode, observations, first_fault, offsets, frozen_observations);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void transition(NavigationWorld* worlds, int count, NavigationMode mode, int action_dimensions,
                const float* raw_actions, float* rewards, uint32_t* attempted,
                uint32_t* accepted, int* first_fault, int* physics_faults,
                cudaStream_t stream, float* offsets, const float* frozen_latents) {
    snapshot_fault<<<1, 1, 0, stream>>>(first_fault);
    CUDA_CHECK_THROW(cudaGetLastError());
    step_worlds<<<(count + world_threads - 1) / world_threads, world_threads, 0, stream>>>(
        worlds, count, mode, action_dimensions, raw_actions, rewards, attempted, accepted, first_fault,
        physics_faults, larger_pond, offsets, frozen_latents);
    CUDA_CHECK_THROW(cudaGetLastError());
}

} // namespace cudarl::flexible_tail::navigation_cuda
