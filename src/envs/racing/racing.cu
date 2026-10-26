#include "racing.h"
#include "race.h"

#include <algorithm>
#include <stdexcept>

namespace cudarl::racing {

namespace {

constexpr int kThreads = 256;

int blocks_for(int n) { return std::max(1, (n + kThreads - 1) / kThreads); }

// Each batch owns its track: the regression circuit and traffic circuit can
// coexist on separate streams without mutating another trainer's geometry.


__global__ void reset_kernel(State* __restrict__ states, uint32_t* __restrict__ episodes, float* __restrict__ obs,
                             int n, uint32_t seed, const Track* track) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= n) return;
    episodes[env] = 0;
    states[env]   = initial_state(seed, static_cast<uint32_t>(env), 0);
    write_obs(states[env], *track, obs + env * kObsDim);
}

__global__ void reset_races_kernel(State* states, uint32_t* episodes, const Track* track, int races, uint32_t seed) {
    const int race=blockIdx.x*blockDim.x+threadIdx.x;
    if (race<races) reset_race(states+race*kRaceCars,episodes+race*kRaceCars,*track,seed,static_cast<uint32_t>(race));
}

__global__ void observe_races_kernel(const State* states, const Track* track, float* obs, int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) observe_race(states+(i/kRaceCars)*kRaceCars,i%kRaceCars,*track,obs+i*kObsDim);
}

__global__ void advance_races_kernel(State* states, State* previous, StepResult* results, const Track* track,
                                    const float* actions, int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    previous[i]=states[i];
    results[i]=advance_race(states[i],actions+i*kActionDim,*track);
}

__global__ void resolve_races_kernel(State* states, const State* previous, StepResult* results, uint32_t* episodes,
                                    const Track* track, StepOutputs out, int races, uint32_t seed) {
    const int race=blockIdx.x*blockDim.x+threadIdx.x;
    if (race>=races) return;
    const int base=race*kRaceCars;
    resolve_race(states+base,previous+base,results+base,episodes+base,*track,seed,
        static_cast<uint32_t>(race),out.final_obs+base*kObsDim);
    for (int j=0;j<kRaceCars;++j) {
        const int i=base+j;
        const StepResult r=results[i];
        const bool done=r.terminated || r.truncated;
        out.reward[i]=r.reward; out.done[i]=done ? 1.f:0.f;
        out.truncated[i]=r.truncated ? 1.f:0.f;
        out.finished_length[i]=done ? static_cast<float>(previous[i].steps+1):0.f;
        out.finished_laps[i]=done ? r.laps:0.f;
        if (out.metrics) {
            out.metrics[4*i]=r.contact ? 1.f:0.f;
            out.metrics[4*i+1]=r.clean_lap ? 1.f:0.f;
            out.metrics[4*i+2]=r.lap_seconds;
            out.metrics[4*i+3]=r.terminated && !r.contact ? 1.f:0.f;
        }
    }
}

__global__ void step_kernel(State* __restrict__ states, uint32_t* __restrict__ episodes, float* __restrict__ obs,
                            const float* __restrict__ actions, StepOutputs out, int n, uint32_t seed, const Track* track) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= n) return;
    State    s       = states[env];
    uint32_t episode = episodes[env];
    const int length = s.steps + 1;   // the episode's length if this step ends it
    const StepResult r = step(s, actions + env * kActionDim, *track, seed, static_cast<uint32_t>(env), episode,
                              out.final_obs + env * kObsDim);
    const bool done = r.terminated || r.truncated;
    states[env]   = s;
    episodes[env] = episode;
    write_obs(s, *track, obs + env * kObsDim);
    out.reward[env]          = r.reward;
    out.done[env]            = done ? 1.f : 0.f;
    out.truncated[env]       = r.truncated ? 1.f : 0.f;
    out.finished_length[env] = done ? static_cast<float>(length) : 0.f;
    out.finished_laps[env]   = done ? r.laps : 0.f;
    if (out.metrics) for (int j=0;j<4;++j) out.metrics[4*env+j]=0.f;
}

} // namespace

RacingBatch::RacingBatch(int num_envs, uint32_t seed, bool traffic)
    : n_(num_envs), seed_(seed), traffic_(traffic), track_(1),
      previous_(static_cast<size_t>(std::max(num_envs,1))), results_(static_cast<size_t>(std::max(num_envs,1))),
      states_(static_cast<size_t>(std::max(num_envs, 1))),
      episodes_(static_cast<size_t>(std::max(num_envs, 1))),
      obs_(static_cast<size_t>(std::max(num_envs, 1)) * kObsDim) {
    if (num_envs <= 0) throw std::invalid_argument("RacingBatch: num_envs must be positive");
    if (traffic && num_envs%kRaceCars!=0) throw std::invalid_argument("RacingBatch: traffic requires groups of eight");
    const Track track = make_track(traffic);
    track_.copy_from_host(&track,1);
    reset();
}

void RacingBatch::reset(cudaStream_t stream) {
    if (traffic_) {
        const int races=n_/kRaceCars;
        reset_races_kernel<<<blocks_for(races),kThreads,0,stream>>>(states_.get(),episodes_.get(),track_.get(),races,seed_);
        CUDA_CHECK_THROW(cudaGetLastError());
        observe_races_kernel<<<blocks_for(n_),kThreads,0,stream>>>(states_.get(),track_.get(),obs_.get(),n_);
        CUDA_CHECK_THROW(cudaGetLastError());
        return;
    }
    reset_kernel<<<blocks_for(n_), kThreads, 0, stream>>>(states_.get(), episodes_.get(), obs_.get(), n_, seed_, track_.get());
    CUDA_CHECK_THROW(cudaGetLastError());
}

void RacingBatch::step(const float* d_actions, const StepOutputs& out, cudaStream_t stream) {
    if (traffic_) {
        advance_races_kernel<<<blocks_for(n_),kThreads,0,stream>>>(states_.get(),previous_.get(),results_.get(),track_.get(),d_actions,n_);
        CUDA_CHECK_THROW(cudaGetLastError());
        const int races=n_/kRaceCars;
        resolve_races_kernel<<<blocks_for(races),kThreads,0,stream>>>(states_.get(),previous_.get(),results_.get(),episodes_.get(),track_.get(),out,races,seed_);
        CUDA_CHECK_THROW(cudaGetLastError());
        observe_races_kernel<<<blocks_for(n_),kThreads,0,stream>>>(states_.get(),track_.get(),obs_.get(),n_);
        CUDA_CHECK_THROW(cudaGetLastError());
        return;
    }
    step_kernel<<<blocks_for(n_), kThreads, 0, stream>>>(states_.get(), episodes_.get(), obs_.get(), d_actions, out,
                                                         n_, seed_, track_.get());
    CUDA_CHECK_THROW(cudaGetLastError());
}

} // namespace cudarl::racing
