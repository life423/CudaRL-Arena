#ifndef CUDARL_RL_PPO_H
#define CUDARL_RL_PPO_H

#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/cuda_memory.h"
#include "envs/cartpole/cartpole.h"
#include "nn/adam.h"
#include "nn/mlp_gpu.h"

// PPO (Schulman et al. 2017) on batched CartPole, entirely on the GPU: rollouts,
// GAE, minibatch shuffling, the clipped-surrogate loss and its gradient, and
// Adam all run as our own kernels. Per update, only a few statistics are copied
// to the host. Deterministic: the same config gives bit-identical training.
namespace cudarl::rl {

struct PpoConfig {
    int      num_envs      = 1024;
    int      rollout_steps = 32;     // per environment, per update
    int      epochs        = 4;
    int      minibatches   = 8;
    float    gamma         = 0.99f;
    float    gae_lambda    = 0.95f;
    float    clip          = 0.2f;
    float    ent_coef      = 0.01f;
    float    vf_coef       = 0.5f;
    // 10x the usual CartPole learning rate: with 32K samples per update the
    // gradient is far less noisy, and at 2.5e-4 the policy barely moves (KL ~1e-4).
    float    lr            = 2.5e-3f;
    bool     anneal_lr     = true;   // linear decay to 0 over total_updates
    int      total_updates = 200;
    float    max_grad_norm = 0.5f;
    int      hidden        = 64;     // two hidden tanh layers of this width, for actor and critic
    uint32_t seed          = 1;
};

struct PpoStats {
    int       update               = 0;
    long long env_steps            = 0;   // total so far
    int       episodes             = 0;   // finished during this update's rollout
    float     mean_episode_length  = 0.f; // of those episodes (0 if none finished)
    float     pg_loss              = 0.f; // means over all minibatches of this update
    float     value_loss           = 0.f;
    float     entropy              = 0.f;
    float     approx_kl            = 0.f;
    float     clip_fraction        = 0.f;
    float     learning_rate        = 0.f;
    double    update_ms            = 0.0; // wall clock for the whole update
    double    env_steps_per_second = 0.0; // rollout + optimization throughput
};

class PpoTrainer {
public:
    explicit PpoTrainer(const PpoConfig& config);
    ~PpoTrainer();
    PpoTrainer(const PpoTrainer&) = delete;
    PpoTrainer& operator=(const PpoTrainer&) = delete;

    const PpoConfig& config() const { return cfg_; }
    const nn::MlpSpec& actor_spec() const { return actor_->spec(); }

    // One rollout of rollout_steps x num_envs steps, then `epochs` passes of minibatch updates.
    PpoStats update();

    std::vector<float> actor_params() const;   // host copy of the policy network
    std::vector<float> critic_params() const;

private:
    void rollout();
    void optimize(float* stats);

    PpoConfig                        cfg_;
    cudaStream_t                     stream_ = nullptr;
    int                              batch_, minibatch_;
    int                              update_ = 0;
    long long                        env_steps_ = 0;
    std::unique_ptr<cartpole::CartPoleBatch> env_;
    std::unique_ptr<nn::GpuMlp>      actor_, critic_;
    std::unique_ptr<nn::GpuAdam>     actor_opt_, critic_opt_;

    // Rollout buffers, [rollout_steps x num_envs] (obs: x 4).
    device_ptr<float> obs_, logp_, values_, rewards_, dones_, truncated_, final_obs_, final_values_, finished_len_;
    device_ptr<float> advantages_, returns_, last_values_;
    device_ptr<int>   actions_;
    // Minibatch buffers.
    device_ptr<float> mb_obs_, mb_logp_, mb_adv_, mb_ret_, dlogits_, dvalues_, row_stats_, adv_stats_;
    device_ptr<int>   mb_actions_;
    device_ptr<float> stats_;   // [episode length sum, episode count, 5 loss stats]
};

} // namespace cudarl::rl

#endif // CUDARL_RL_PPO_H
