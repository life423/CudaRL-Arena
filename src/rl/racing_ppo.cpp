#include "racing_ppo.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "rl/gaussian_kernels.h"
#include "rl/ppo_kernels.h"
#include "rl/ppo_math.h"

namespace cudarl::rl {

namespace {

constexpr int kObs = racing::kObsDim, kAct = racing::kActionDim;
constexpr int kStatEpisodes = 0, kStatLosses = 4, kNumStats = kStatLosses + kGaussianStats;

nn::MlpSpec net_spec(int hidden, int outputs) {
    return nn::MlpSpec({kObs, hidden, hidden, outputs},
                       {nn::Activation::Tanh, nn::Activation::Tanh, nn::Activation::Identity});
}

nn::AdamConfig adam_config(const RacingPpoConfig& c) {
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

RacingPpoTrainer::RacingPpoTrainer(const RacingPpoConfig& config) : cfg_(config) {
    if (cfg_.num_envs <= 0 || cfg_.rollout_steps <= 0 || cfg_.epochs <= 0 || cfg_.minibatches <= 0 ||
        cfg_.total_updates <= 0 || cfg_.hidden <= 0) {
        throw std::invalid_argument("RacingPpoTrainer: sizes and counts must be positive");
    }
    batch_ = cfg_.num_envs * cfg_.rollout_steps;
    if (batch_ % cfg_.minibatches != 0) {
        throw std::invalid_argument("RacingPpoTrainer: batch must divide into minibatches");
    }
    minibatch_ = batch_ / cfg_.minibatches;

    // Non-blocking, so work on the legacy default stream (the renderer's
    // showcase, say) never serializes with training.
    CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

    env_ = std::make_unique<racing::RacingBatch>(cfg_.num_envs, cfg_.seed);
    const float root2 = std::sqrt(2.f);
    actor_  = std::make_unique<nn::GpuMlp>(net_spec(cfg_.hidden, kAct), std::max(cfg_.num_envs, minibatch_), cfg_.seed);
    critic_ = std::make_unique<nn::GpuMlp>(net_spec(cfg_.hidden, 1), std::max(batch_, minibatch_), cfg_.seed + 1);
    // Hidden gain sqrt(2); a near-zero mean head (actions start near the middle); unit value head.
    actor_->set_params(nn::orthogonal_params(actor_->spec(), cfg_.seed, {root2, root2, 0.01f}));
    critic_->set_params(nn::orthogonal_params(critic_->spec(), cfg_.seed + 1, {root2, root2, 1.0f}));
    actor_opt_   = std::make_unique<nn::GpuAdam>(actor_->num_params(), adam_config(cfg_));
    critic_opt_  = std::make_unique<nn::GpuAdam>(critic_->num_params(), adam_config(cfg_));
    log_std_opt_ = std::make_unique<nn::GpuAdam>(kAct, adam_config(cfg_));
    log_std_.allocate(kAct);
    log_std_grad_.allocate(kAct);
    log_std_.copy_from_host(std::vector<float>(kAct, cfg_.log_std_init).data(), kAct);
    CUDA_CHECK_THROW(cudaMemset(log_std_grad_.get(), 0, log_std_grad_.size_bytes()));

    const size_t b = static_cast<size_t>(batch_), mb = static_cast<size_t>(minibatch_);
    obs_.allocate(b * kObs);
    final_obs_.allocate(b * kObs);
    CUDA_CHECK_THROW(cudaMemset(final_obs_.get(), 0, final_obs_.size_bytes()));
    actions_.allocate(b * kAct);
    for (auto* buf : {&logp_, &values_, &rewards_, &dones_, &truncated_, &final_values_, &finished_len_,
                      &finished_laps_, &advantages_, &returns_}) {
        buf->allocate(b);
    }
    last_values_.allocate(cfg_.num_envs);
    mb_obs_.allocate(mb * kObs);
    mb_actions_.allocate(mb * kAct);
    for (auto* buf : {&mb_logp_, &mb_adv_, &mb_ret_, &dvalues_}) buf->allocate(mb);
    dmean_.allocate(mb * kAct);
    dlog_std_rows_.allocate(mb * kAct);
    row_stats_.allocate(mb * kGaussianStats);
    adv_stats_.allocate(2);
    stats_.allocate(kNumStats);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
}

RacingPpoTrainer::~RacingPpoTrainer() {
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
}

void RacingPpoTrainer::rollout() {
    const int n = cfg_.num_envs;
    for (int t = 0; t < cfg_.rollout_steps; ++t) {
        const size_t row = static_cast<size_t>(t) * n;
        copy_async(obs_.get() + row * kObs, env_->obs(), static_cast<size_t>(n) * kObs, stream_);
        actor_->forward(env_->obs(), n, stream_);
        const uint32_t step_id = static_cast<uint32_t>(update_) * cfg_.rollout_steps + t;
        sample_gaussian(actor_->output(), log_std_.get(), n, kAct, cfg_.seed, step_id, actions_.get() + row * kAct,
                        logp_.get() + row, stream_);
        critic_->forward(env_->obs(), n, stream_);
        copy_async(values_.get() + row, critic_->output(), n, stream_);
        const racing::StepOutputs out{rewards_.get() + row,         dones_.get() + row,
                                      truncated_.get() + row,       final_obs_.get() + row * kObs,
                                      finished_len_.get() + row,    finished_laps_.get() + row};
        env_->step(actions_.get() + row * kAct, out, stream_);
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

void RacingPpoTrainer::optimize(float* stats) {
    const PpoCoefficients coeffs{cfg_.clip, cfg_.ent_coef, cfg_.vf_coef};
    for (int epoch = 0; epoch < cfg_.epochs; ++epoch) {
        const uint32_t perm_seed = rng::bits(cfg_.seed, static_cast<uint32_t>(update_), static_cast<uint32_t>(epoch));
        const Permutation perm = make_permutation(static_cast<uint32_t>(batch_), perm_seed);
        for (int mb = 0; mb < cfg_.minibatches; ++mb) {
            const int first = mb * minibatch_;
            gather_rows(perm, first, minibatch_, kObs, obs_.get(), mb_obs_.get(), stream_);
            gather_rows(perm, first, minibatch_, kAct, actions_.get(), mb_actions_.get(), stream_);
            gather_rows(perm, first, minibatch_, 1, logp_.get(), mb_logp_.get(), stream_);
            gather_rows(perm, first, minibatch_, 1, advantages_.get(), mb_adv_.get(), stream_);
            gather_rows(perm, first, minibatch_, 1, returns_.get(), mb_ret_.get(), stream_);
            mean_std(mb_adv_.get(), minibatch_, adv_stats_.get(), stream_);

            actor_->forward(mb_obs_.get(), minibatch_, stream_);
            critic_->forward(mb_obs_.get(), minibatch_, stream_);
            gaussian_ppo_loss_grad(actor_->output(), log_std_.get(), critic_->output(), minibatch_, kAct,
                                   mb_actions_.get(), mb_logp_.get(), mb_adv_.get(), mb_ret_.get(), adv_stats_.get(),
                                   coeffs, dmean_.get(), dlog_std_rows_.get(), dvalues_.get(), row_stats_.get(),
                                   stats + kStatLosses, stream_);

            actor_->zero_grad(stream_);
            actor_->backward(dmean_.get(), minibatch_, nullptr, stream_);
            actor_opt_->step(actor_->params(), actor_->grads(), stream_);

            CUDA_CHECK_THROW(cudaMemsetAsync(log_std_grad_.get(), 0, log_std_grad_.size_bytes(), stream_));
            sum_columns(dlog_std_rows_.get(), minibatch_, kAct, log_std_grad_.get(), stream_);
            log_std_opt_->step(log_std_.get(), log_std_grad_.get(), stream_);

            critic_->zero_grad(stream_);
            critic_->backward(dvalues_.get(), minibatch_, nullptr, stream_);
            critic_opt_->step(critic_->params(), critic_->grads(), stream_);
        }
    }
}

RacingStats RacingPpoTrainer::update() {
    const auto start = std::chrono::steady_clock::now();
    const float lr = cfg_.anneal_lr
                         ? cfg_.lr * std::max(0.f, 1.f - static_cast<float>(update_) / static_cast<float>(cfg_.total_updates))
                         : cfg_.lr;
    actor_opt_->set_learning_rate(lr);
    critic_opt_->set_learning_rate(lr);
    log_std_opt_->set_learning_rate(lr);

    CUDA_CHECK_THROW(cudaMemsetAsync(stats_.get(), 0, stats_.size_bytes(), stream_));
    rollout();
    episode_totals(finished_len_.get(), finished_laps_.get(), dones_.get(), truncated_.get(), batch_,
                   stats_.get() + kStatEpisodes, stream_);
    optimize(stats_.get());

    float host[kNumStats];
    CUDA_CHECK_THROW(cudaMemcpyAsync(host, stats_.get(), sizeof host, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream_));

    ++update_;
    env_steps_ += batch_;
    RacingStats s;
    s.update    = update_;
    s.env_steps = env_steps_;
    s.episodes  = static_cast<int>(host[kStatEpisodes + 2]);
    if (s.episodes > 0) {
        const float ended = host[kStatEpisodes + 2];
        s.mean_episode_length = host[kStatEpisodes] / ended;
        s.mean_laps           = host[kStatEpisodes + 1] / ended;
        s.crash_fraction      = host[kStatEpisodes + 3] / ended;
    }
    const float rows = static_cast<float>(cfg_.epochs) * batch_;
    s.pg_loss       = host[kStatLosses + 0] / rows;
    s.value_loss    = host[kStatLosses + 1] / rows;
    s.entropy       = host[kStatLosses + 2] / rows;
    s.approx_kl     = host[kStatLosses + 3] / rows;
    s.clip_fraction = host[kStatLosses + 4] / rows;
    const std::vector<float> ls = log_std();
    for (float v : ls) s.mean_std += std::exp(v) / static_cast<float>(ls.size());
    s.learning_rate = lr;
    s.update_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    s.env_steps_per_second = batch_ / (s.update_ms * 1e-3);
    return s;
}

std::vector<float> RacingPpoTrainer::actor_params() const {
    std::vector<float> h(actor_->num_params());
    CUDA_CHECK_THROW(cudaMemcpyAsync(h.data(), actor_->params(), h.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream_));
    return h;
}

std::vector<float> RacingPpoTrainer::critic_params() const {
    std::vector<float> h(critic_->num_params());
    CUDA_CHECK_THROW(cudaMemcpyAsync(h.data(), critic_->params(), h.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream_));
    return h;
}

std::vector<float> RacingPpoTrainer::log_std() const {
    std::vector<float> h(kAct);
    CUDA_CHECK_THROW(cudaMemcpyAsync(h.data(), log_std_.get(), h.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream_));
    return h;
}

} // namespace cudarl::rl
