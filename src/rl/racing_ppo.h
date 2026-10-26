#ifndef CUDARL_RL_RACING_PPO_H
#define CUDARL_RL_RACING_PPO_H

#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/cuda_memory.h"
#include "envs/racing/racing.h"
#include "nn/adam.h"
#include "nn/mlp_gpu.h"

// PPO with a Gaussian policy on the racing track, entirely on the GPU. The actor
// outputs the mean of each action (steer, throttle); a learned log standard
// deviation, shared across states, sits beside it. The critic predicts the
// value. Rollouts, GAE, the Gaussian loss and its gradient, and Adam all run as
// our own kernels. Per update, only a few statistics are copied to the host.
namespace cudarl::rl {

struct RacingPpoConfig {
    bool     graph_optimization = true; // capture/update the optimization workflow before executing once
    bool     distributed_traffic = true; // retained original launch layout is the benchmark reference
    bool     profile_phases = false; // opt-in CUDA event spans; benchmark separately from normal timing
    bool     traffic       = true;  // eight interacting cars per race; false keeps the original circuit
    int      num_envs      = 2048;
    int      rollout_steps = 64;     // per environment, per update
    int      epochs        = 4;
    int      minibatches   = 8;
    float    gamma         = 0.99f;
    float    gae_lambda    = 0.95f;
    float    clip          = 0.2f;
    float    ent_coef      = 0.f;
    float    vf_coef       = 0.5f;
    float    lr            = 1e-3f;
    bool     anneal_lr     = true;   // linear decay to 0 over total_updates
    int      total_updates = 300;
    float    max_grad_norm = 0.5f;
    int      hidden        = 64;     // two hidden tanh layers of this width, for actor and critic
    float    log_std_init  = -0.5f;  // initial exp(log_std) = 0.6
    uint32_t seed          = 1;
};

struct RacingStats {
    int       contact_cars = 0;
    int       clean_laps = 0;
    int       wall_crashes = 0;
    float     mean_lap_seconds = 0.f;
    int       update               = 0;
    long long env_steps            = 0;     // total so far
    int       episodes             = 0;     // ended during this update's rollout
    float     mean_episode_length  = 0.f;   // of those episodes (0 if none ended)
    float     mean_laps            = 0.f;   // laps covered by those episodes
    float     crash_fraction       = 0.f;   // share of those episodes that ended in a crash
    float     pg_loss              = 0.f;   // means over all minibatches of this update
    float     value_loss           = 0.f;
    float     entropy              = 0.f;
    float     approx_kl            = 0.f;
    float     clip_fraction        = 0.f;
    float     mean_std             = 0.f;   // average exp(log_std) over the action dimensions
    float     learning_rate        = 0.f;
    double    update_ms            = 0.0;   // wall clock for the whole update
    double    env_steps_per_second = 0.0;   // rollout + optimization throughput
    // GPU stream elapsed times, including submission gaps, only when profiling.
    double    rollout_ms = 0.0, bootstrap_gae_ms = 0.0, metrics_ms = 0.0;
    double    optimize_ms = 0.0, readback_ms = 0.0;
    double    advance_ms = 0.0, resolve_ms = 0.0, observe_ms = 0.0;
    double    gather_ms = 0.0, forward_loss_ms = 0.0, actor_update_ms = 0.0, critic_update_ms = 0.0;
    double    optimization_submit_ms = 0.0;
    double    optimization_graph_setup_ms = 0.0;
    int       optimization_graph_nodes = 0;
};

struct RacingPhaseTimer;
struct RacingOptimizationGraph;

class RacingPpoTrainer {
public:
    using Config = RacingPpoConfig;
    using Stats  = RacingStats;

    explicit RacingPpoTrainer(const RacingPpoConfig& config);
    ~RacingPpoTrainer();
    RacingPpoTrainer(const RacingPpoTrainer&) = delete;
    RacingPpoTrainer& operator=(const RacingPpoTrainer&) = delete;

    const RacingPpoConfig& config() const { return cfg_; }

    // One rollout of rollout_steps x num_envs steps, then `epochs` passes of minibatch updates.
    RacingStats update();

    std::vector<float> actor_params() const;    // host copy of the policy network
    std::vector<float> critic_params() const;
    std::vector<float> log_std() const;         // host copy of the action log standard deviations

private:
    void timing_mark(int index);
    void rollout();
    void optimize(float* stats);
    void submit_optimization(float* stats);

    RacingPpoConfig                              cfg_;
    cudaStream_t                                 stream_ = nullptr;
    std::unique_ptr<RacingPhaseTimer>             phase_timer_;
    std::unique_ptr<RacingOptimizationGraph>      optimization_graph_;
    int                                          batch_, minibatch_;
    int                                          update_ = 0;
    long long                                    env_steps_ = 0;
    std::unique_ptr<racing::RacingBatch>         env_;
    std::unique_ptr<nn::GpuMlp>                  actor_, critic_;
    std::unique_ptr<nn::GpuAdam>                 actor_opt_, critic_opt_, log_std_opt_;
    device_ptr<float>                            log_std_, log_std_grad_;

    // Rollout buffers, [rollout_steps x num_envs] (obs: x kObsDim, actions: x kActionDim).
    device_ptr<float> obs_, actions_, logp_, values_, rewards_, dones_, truncated_, final_obs_, final_values_;
    device_ptr<float> finished_len_, finished_laps_, advantages_, returns_, last_values_;
    // Minibatch buffers.
    device_ptr<float> mb_obs_, mb_actions_, mb_logp_, mb_adv_, mb_ret_, dmean_, dlog_std_rows_, dvalues_;
    device_ptr<float> row_stats_, adv_stats_;
    device_ptr<float> stats_;   // [4 episode totals, 5 loss sums]
    device_ptr<float> race_metrics_;
};

} // namespace cudarl::rl

#endif // CUDARL_RL_RACING_PPO_H
