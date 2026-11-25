#pragma once

#include "rl/flexible_tail_navigation.h"

#include <cuda_runtime.h>

namespace cudarl::flexible_tail::navigation_cuda {

inline constexpr float control_gamma = .998f;

// first_fault has two stream-ordered entries, with INT_MAX as the no-fault sentinel.
void observe(NavigationWorld* worlds, int count, NavigationMode mode, float* observations,
             int* first_fault, cudaStream_t stream, const float* offsets = nullptr,
             float* frozen_observations = nullptr);

void transition(NavigationWorld* worlds, int count, NavigationMode mode, int action_dimensions,
                const float* raw_actions, float* rewards, uint32_t* attempted_substeps,
                uint32_t* accepted_substeps, int* first_fault, int* physics_faults,
                cudaStream_t stream, float* offsets = nullptr, const float* frozen_latents = nullptr);

} // namespace cudarl::flexible_tail::navigation_cuda
