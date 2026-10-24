#include "cartpole.h"

#include <algorithm>
#include <stdexcept>

namespace cudarl::cartpole {

namespace {

constexpr int kThreads = 256;

int blocks_for(int n) { return std::max(1, (n + kThreads - 1) / kThreads); }

__global__ void reset_kernel(State* __restrict__ states, uint32_t* __restrict__ episodes, float* __restrict__ obs,
                             int n, uint32_t seed) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= n) return;
    episodes[env] = 0;
    states[env]   = initial_state(seed, static_cast<uint32_t>(env), 0);
    write_obs(states[env], obs + env * kObsDim);
}

__global__ void step_kernel(State* __restrict__ states, uint32_t* __restrict__ episodes, float* __restrict__ obs,
                            const int* __restrict__ actions, StepOutputs out, int n, uint32_t seed) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= n) return;
    State    s       = states[env];
    uint32_t episode = episodes[env];
    const int length = s.steps + 1;   // the episode's length if this step ends it
    const StepResult r = step(s, actions[env], seed, static_cast<uint32_t>(env), episode, out.final_obs + env * kObsDim);
    const bool done = r.terminated || r.truncated;
    states[env]   = s;
    episodes[env] = episode;
    write_obs(s, obs + env * kObsDim);
    out.reward[env]          = r.reward;
    out.done[env]            = done ? 1.f : 0.f;
    out.truncated[env]       = r.truncated ? 1.f : 0.f;
    out.finished_length[env] = done ? static_cast<float>(length) : 0.f;
}

} // namespace

CartPoleBatch::CartPoleBatch(int num_envs, uint32_t seed)
    : n_(num_envs), seed_(seed), states_(static_cast<size_t>(std::max(num_envs, 1))),
      episodes_(static_cast<size_t>(std::max(num_envs, 1))),
      obs_(static_cast<size_t>(std::max(num_envs, 1)) * kObsDim) {
    if (num_envs <= 0) throw std::invalid_argument("CartPoleBatch: num_envs must be positive");
    reset();
}

void CartPoleBatch::reset(cudaStream_t stream) {
    reset_kernel<<<blocks_for(n_), kThreads, 0, stream>>>(states_.get(), episodes_.get(), obs_.get(), n_, seed_);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void CartPoleBatch::step(const int* d_actions, const StepOutputs& out, cudaStream_t stream) {
    step_kernel<<<blocks_for(n_), kThreads, 0, stream>>>(states_.get(), episodes_.get(), obs_.get(), d_actions, out,
                                                         n_, seed_);
    CUDA_CHECK_THROW(cudaGetLastError());
}

} // namespace cudarl::cartpole
