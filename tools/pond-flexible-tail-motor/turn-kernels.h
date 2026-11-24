#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace cudarl::flexible_tail {
struct TurnWorld;

namespace turn_cuda {

// Reserve two device ints. The caller sets and reads first_fault[0], with
// INT_MAX indicating success. Index 1 is private stream-ordered scratch.
void observe(TurnWorld* worlds, int count, float* observations, int* first_fault,
             cudaStream_t stream);

// Both counters contain count cumulative entries, starting at zero. Keep the
// Gaussian samples unmodified for PPO's likelihood ratio.
void transition(TurnWorld* worlds, int count, const float* raw_actions, float* rewards,
                uint32_t* attempted_substeps, uint32_t* accepted_substeps,
                int* first_fault, cudaStream_t stream);

void audit(const float* values, int count, int* first_fault, cudaStream_t stream);

} // namespace turn_cuda
} // namespace cudarl::flexible_tail
