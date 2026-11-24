#include "turn-kernels.h"

#include "cuda/cuda_memory.h"
#include "habitat_config.h"
#include "rl/flexible_tail_transition.h"

#include <climits>
#include <cmath>

namespace cudarl::flexible_tail::turn_cuda {
namespace {

constexpr int world_threads = 32;
constexpr int audit_threads = 256;

// Sample the previous launch's fault once, before any lane may update it.
// A fault in this launch must never depend on which other lane ran first.
__global__ void snapshot_fault(int* first_fault) {
    first_fault[1] = first_fault[0];
}

__global__ void observe_worlds(TurnWorld* worlds, int count, float* observations, int* first_fault) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;

    float state[motor_inputs]{};
    if (first_fault[1] == INT_MAX) {
        auto& world = worlds[lane];
        prepare_turn_request(world);
        observe_turn(world, state);
    }
    auto* row = observations + lane * motor_inputs;
    for (int column = 0; column < motor_inputs; ++column) {
        const float value = state[column];
        if (!isfinite(value)) atomicMin(first_fault, lane);
        row[column] = isfinite(value) ? value : 0.f;
    }
}

__global__ void step_worlds(TurnWorld* worlds, int count, const float* raw_actions,
                            float* rewards, uint32_t* attempted_substeps,
                            uint32_t* accepted_substeps, int* first_fault,
                            pond_geometry_v1::Parameters habitat) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;

    rewards[lane] = 0.f;
    if (first_fault[1] != INT_MAX) return;

    const auto* latent = raw_actions + lane * motor_outputs;
    if (!isfinite(latent[0]) || !isfinite(latent[1])) {
        atomicMin(first_fault, lane);
        return;
    }
    const float targets[motor_outputs]{tanhf(latent[0]), tanhf(latent[1])};
    auto& world = worlds[lane];
    prepare_turn_request(world);
    const auto step = step_turn_control(world, targets, habitat);
    attempted_substeps[lane] += static_cast<uint32_t>(step.attempted_substeps);
    accepted_substeps[lane] += static_cast<uint32_t>(step.accepted_substeps);
    if (step.fault || !isfinite(step.reward)) {
        atomicMin(first_fault, lane);
        return;
    }
    rewards[lane] = step.reward;
}

__global__ void audit_values(const float* values, int count, int* first_fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && !isfinite(values[i])) atomicMin(first_fault, i);
}

} // namespace

void observe(TurnWorld* worlds, int count, float* observations, int* first_fault,
             cudaStream_t stream) {
    snapshot_fault<<<1, 1, 0, stream>>>(first_fault);
    CUDA_CHECK_THROW(cudaGetLastError());
    observe_worlds<<<(count + world_threads - 1) / world_threads, world_threads, 0, stream>>>(
        worlds, count, observations, first_fault);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void transition(TurnWorld* worlds, int count, const float* raw_actions, float* rewards,
                uint32_t* attempted_substeps, uint32_t* accepted_substeps,
                int* first_fault, cudaStream_t stream) {
    snapshot_fault<<<1, 1, 0, stream>>>(first_fault);
    CUDA_CHECK_THROW(cudaGetLastError());
    step_worlds<<<(count + world_threads - 1) / world_threads, world_threads, 0, stream>>>(
        worlds, count, raw_actions, rewards, attempted_substeps, accepted_substeps, first_fault,
        larger_pond);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void audit(const float* values, int count, int* first_fault, cudaStream_t stream) {
    audit_values<<<(count + audit_threads - 1) / audit_threads, audit_threads, 0, stream>>>(
        values, count, first_fault);
    CUDA_CHECK_THROW(cudaGetLastError());
}

} // namespace cudarl::flexible_tail::turn_cuda
