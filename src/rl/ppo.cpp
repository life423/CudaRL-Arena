#include "ppo.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "ppo_kernels.h"

namespace cudarl::rl {

namespace {

constexpr int kObs = cartpole::kObsDim, kActions = cartpole::kNumActions;
constexpr int kStatEpisodes = 0, kStatLosses = 2, kNumStats = kStatLosses + kPpoStats;

nn::MlpSpec net_spec(int hidden, int outputs) {
    return nn::MlpSpec({kObs, hidden, hidden, outputs},
                       {nn::Activation::Tanh, nn::Activation::Tanh, nn::Activation::Identity});
}

nn::AdamConfig adam_config(const PpoConfig& c) {
    nn::AdamConfig a;
    a.lr            = c.lr;
    a.eps           = 1e-5f;   // the PPO convention
    a.max_grad_norm = c.max_grad_norm;
    return a;
}

void copy_async(float* dst, const float* src, size_t count, cudaStream_t s) {
    CUDA_CHECK_THROW(cudaMemcpyAsync(dst, src, count * sizeof(float), cudaMemcpyDeviceToDevice, s));
}

} // namespace

PpoTrainer::PpoTrainer(const PpoConfig& config) : cfg_(config) {
    if (cfg_.num_envs <= 0 || cfg_.rollout_steps <= 0 || cfg_.epochs <= 0 || cfg_.minibatches <= 0 ||
        cfg_.total_updates <= 0 || cfg_.hidden <= 0) {
        throw std::invalid_argument("PpoTrainer: sizes and counts must be positive");
    }
    batch_ = cfg_.num_envs * cfg_.rollout_steps;
    if (batch_ % cfg_.minibatches != 0) throw std::invalid_argument("PpoTrainer: batch must divide into minibatches");
    minibatch_ = batch_ / cfg_.minibatches;

    // Non-blocking, so work on the legacy default stream (e.g. a renderer's
    // showcase on another thread) never serializes with training.
    CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

    env_ = std::make_unique<cartpole::CartPoleBatch>(cfg_.num_envs, cfg_.seed);
    const float root2 = std::sqrt(2.f);
    actor_  = std::make_unique<nn::GpuMlp>(net_spec(cfg_.hidden, kActions), std::max(cfg_.num_envs, minibatch_), cfg_.seed);
    critic_ = std::make_unique<nn::GpuMlp>(net_spec(cfg_.hidden, 1), std::max(batch_, minibatch_), cfg_.seed + 1);
    // PPO convention: hidden gain sqrt(2); a near-zero policy head (uniform initial policy); unit value head.
    actor_->set_params(nn::orthogonal_params(actor_->spec(), cfg_.seed, {root2, root2, 0.01f}));
    critic_->set_params(nn::orthogonal_params(critic_->spec(), cfg_.seed + 1, {root2, root2, 1.0f}));
    actor_opt_  = std::make_unique<nn::GpuAdam>(actor_->num_params(), adam_config(cfg_));
    critic_opt_ = std::make_unique<nn::GpuAdam>(critic_->num_params(), adam_config(cfg_));

    const size_t b = static_cast<size_t>(batch_), mb = static_cast<size_t>(minibatch_);
    for (auto* buf : {&logp_, &values_, &rewards_, &dones_, &truncated_, &final_values_, &finished_len_, &advantages_,
                      &returns_}) {
        buf->allocate(b);
    }
    obs_.allocate(b * kObs);
    final_obs_.allocate(b * kObs);
    CUDA_CHECK_THROW(cudaMemset(final_obs_.get(), 0, final_obs_.size_bytes()));
    actions_.allocate(b);
    last_values_.allocate(cfg_.num_envs);
    mb_obs_.allocate(mb * kObs);
    for (auto* buf : {&mb_logp_, &mb_adv_, &mb_ret_, &dvalues_}) buf->allocate(mb);
    mb_actions_.allocate(mb);
    dlogits_.allocate(mb * kActions);
    row_stats_.allocate(mb * kPpoStats);
    adv_stats_.allocate(2);
    stats_.allocate(kNumStats);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
}

PpoTrainer::~PpoTrainer() {
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
}

void PpoTrainer::rollout() {
    const int n = cfg_.num_envs;
    for (int t = 0; t < cfg_.rollout_steps; ++t) {
        const size_t row = static_cast<size_t>(t) * n;
        copy_async(obs_.get() + row * kObs, env_->obs(), static_cast<size_t>(n) * kObs, stream_);
        actor_->forward(env_->obs(), n, stream_);
        const uint32_t step_id = static_cast<uint32_t>(update_) * cfg_.rollout_steps + t;
        sample_actions(actor_->output(), n, kActions, cfg_.seed, step_id, false, actions_.get() + row,
                       logp_.get() + row, stream_);
        critic_->forward(env_->obs(), n, stream_);
        copy_async(values_.get() + row, critic_->output(), n, stream_);
        const cartpole::StepOutputs out{rewards_.get() + row, dones_.get() + row, truncated_.get() + row,
                                        final_obs_.get() + row * kObs, finished_len_.get() + row};
        env_->step(actions_.get() + row, out, stream_);
    }
    // Bootstrap values: V(next obs) after the last step, and V(final obs) for truncated episodes.
    critic_->forward(env_->obs(), n, stream_);
    copy_async(last_values_.get(), critic_->output(), n, stream_);
    critic_->forward(final_obs_.get(), batch_, stream_);
    copy_async(final_values_.get(), critic_->output(), batch_, stream_);

    compute_gae(cfg_.rollout_steps, n, rewards_.get(), values_.get(), dones_.get(), truncated_.get(),
                final_values_.get(), last_values_.get(), cfg_.gamma, cfg_.gae_lambda, advantages_.get(),
                returns_.get(), stream_);
}

void PpoTrainer::optimize(float* stats) {
    const PpoCoefficients coeffs{cfg_.clip, cfg_.ent_coef, cfg_.vf_coef};
    for (int epoch = 0; epoch < cfg_.epochs; ++epoch) {
        const uint32_t perm_seed = rng::bits(cfg_.seed, static_cast<uint32_t>(update_), static_cast<uint32_t>(epoch));
        const Permutation perm = make_permutation(static_cast<uint32_t>(batch_), perm_seed);
        for (int mb = 0; mb < cfg_.minibatches; ++mb) {
            gather_minibatch(perm, mb * minibatch_, minibatch_, kObs, obs_.get(), actions_.get(), logp_.get(),
                             advantages_.get(), returns_.get(), mb_obs_.get(), mb_actions_.get(), mb_logp_.get(),
                             mb_adv_.get(), mb_ret_.get(), stream_);
            mean_std(mb_adv_.get(), minibatch_, adv_stats_.get(), stream_);
            actor_->forward(mb_obs_.get(), minibatch_, stream_);
            critic_->forward(mb_obs_.get(), minibatch_, stream_);
            ppo_loss_grad(actor_->output(), critic_->output(), minibatch_, kActions, mb_actions_.get(), mb_logp_.get(),
                          mb_adv_.get(), mb_ret_.get(), adv_stats_.get(), coeffs, dlogits_.get(), dvalues_.get(),
                          row_stats_.get(), stats + kStatLosses, stream_);
            actor_->zero_grad(stream_);
            actor_->backward(dlogits_.get(), minibatch_, nullptr, stream_);
            actor_opt_->step(actor_->params(), actor_->grads(), stream_);
            critic_->zero_grad(stream_);
            critic_->backward(dvalues_.get(), minibatch_, nullptr, stream_);
            critic_opt_->step(critic_->params(), critic_->grads(), stream_);
        }
    }
}

PpoStats PpoTrainer::update() {
    const auto start = std::chrono::steady_clock::now();
    const float lr = cfg_.anneal_lr
                         ? cfg_.lr * std::max(0.f, 1.f - static_cast<float>(update_) / static_cast<float>(cfg_.total_updates))
                         : cfg_.lr;
    actor_opt_->set_learning_rate(lr);
    critic_opt_->set_learning_rate(lr);

    CUDA_CHECK_THROW(cudaMemsetAsync(stats_.get(), 0, stats_.size_bytes(), stream_));
    rollout();
    episode_stats(finished_len_.get(), batch_, stats_.get() + kStatEpisodes, stream_);
    optimize(stats_.get());

    float host[kNumStats];
    CUDA_CHECK_THROW(cudaMemcpyAsync(host, stats_.get(), sizeof host, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream_));

    ++update_;
    env_steps_ += batch_;
    PpoStats s;
    s.update   = update_;
    s.env_steps = env_steps_;
    s.episodes = static_cast<int>(host[kStatEpisodes + 1]);
    s.mean_episode_length = s.episodes > 0 ? host[kStatEpisodes] / host[kStatEpisodes + 1] : 0.f;
    const float rows = static_cast<float>(cfg_.epochs) * batch_;
    s.pg_loss       = host[kStatLosses + 0] / rows;
    s.value_loss    = host[kStatLosses + 1] / rows;
    s.entropy       = host[kStatLosses + 2] / rows;
    s.approx_kl     = host[kStatLosses + 3] / rows;
    s.clip_fraction = host[kStatLosses + 4] / rows;
    s.learning_rate = lr;
    s.update_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    s.env_steps_per_second = batch_ / (s.update_ms * 1e-3);
    return s;
}

std::vector<float> PpoTrainer::actor_params() const {
    std::vector<float> h(actor_->num_params());
    CUDA_CHECK_THROW(cudaMemcpyAsync(h.data(), actor_->params(), h.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream_));
    return h;
}

std::vector<float> PpoTrainer::critic_params() const {
    std::vector<float> h(critic_->num_params());
    CUDA_CHECK_THROW(cudaMemcpyAsync(h.data(), critic_->params(), h.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream_));
    return h;
}

} // namespace cudarl::rl
