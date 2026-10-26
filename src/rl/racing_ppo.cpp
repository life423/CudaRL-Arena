#include "racing_ppo.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "rl/gaussian_kernels.h"
#include "rl/ppo_kernels.h"
#include "rl/ppo_math.h"

namespace cudarl::rl {

struct RacingOptimizationGraph {
    cudaGraphExec_t executable = nullptr;
    int nodes = 0;
    double setup_ms = 0.0;
    ~RacingOptimizationGraph() {
        if (executable)
            cudaGraphExecDestroy(executable);
    }
};

struct RacingPhaseTimer {
    std::array<cudaEvent_t, 6> events{};
    std::vector<racing::RacingStepEvents> simulation;
    std::vector<std::array<cudaEvent_t, 5>> optimization;
    RacingPhaseTimer(int rollout_steps, int minibatches)
        : simulation(rollout_steps), optimization(minibatches) {
        try {
            for (auto& event : events)
                CUDA_CHECK_THROW(cudaEventCreate(&event));
            for (auto& step : simulation)
                for (auto* event : {&step.start, &step.advanced, &step.resolved, &step.observed})
                    CUDA_CHECK_THROW(cudaEventCreate(event));
            for (auto& minibatch : optimization)
                for (auto& event : minibatch)
                    CUDA_CHECK_THROW(cudaEventCreate(&event));
        } catch (...) {
            destroy();
            throw;
        }
    }
    void destroy() {
        for (auto event : events)
            if (event)
                cudaEventDestroy(event);
        for (auto& step : simulation)
            for (auto event : {step.start, step.advanced, step.resolved, step.observed})
                if (event)
                    cudaEventDestroy(event);
        for (auto& minibatch : optimization)
            for (auto event : minibatch)
                if (event)
                    cudaEventDestroy(event);
    }
    ~RacingPhaseTimer() { destroy(); }
    static double span(cudaEvent_t first, cudaEvent_t last) {
        float ms = 0.f;
        CUDA_CHECK_THROW(cudaEventElapsedTime(&ms, first, last));
        return ms;
    }
    double elapsed(int first, int last) const { return span(events[first], events[last]); }
};

namespace {

constexpr int kObs = racing::kObsDim, kAct = racing::kActionDim;
constexpr int kStatEpisodes = 0, kStatLosses = 4, kStatRace = kStatLosses + kGaussianStats,
              kNumStats = kStatRace + 4;

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

    env_ = std::make_unique<racing::RacingBatch>(cfg_.num_envs, cfg_.seed, cfg_.traffic, cfg_.distributed_traffic);
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
    race_metrics_.allocate(b*4);
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
    if (cfg_.profile_phases) phase_timer_ = std::make_unique<RacingPhaseTimer>(cfg_.rollout_steps,cfg_.epochs*cfg_.minibatches);
    if (cfg_.graph_optimization) optimization_graph_=std::make_unique<RacingOptimizationGraph>();
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
}

RacingPpoTrainer::~RacingPpoTrainer() {
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
}

void RacingPpoTrainer::timing_mark(int index) {
    if (phase_timer_) CUDA_CHECK_THROW(cudaEventRecord(phase_timer_->events[index], stream_));
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
                                      finished_len_.get() + row,    finished_laps_.get() + row,
                                      race_metrics_.get()+row*4, cfg_.distributed_traffic};
        env_->step(actions_.get() + row * kAct, out, stream_, phase_timer_ ? &phase_timer_->simulation[t] : nullptr);
    }
    timing_mark(1);
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
        const uint32_t perm_seed =
            rng::bits(cfg_.seed, static_cast<uint32_t>(update_), static_cast<uint32_t>(epoch));
        const Permutation perm = make_permutation(static_cast<uint32_t>(batch_), perm_seed);
        for (int mb = 0; mb < cfg_.minibatches; ++mb) {
            const auto mark = [&](int stage) {
                if (phase_timer_)
                    CUDA_CHECK_THROW(cudaEventRecordWithFlags(
                        phase_timer_->optimization[epoch * cfg_.minibatches + mb][stage], stream_,
                        cfg_.graph_optimization ? cudaEventRecordExternal : cudaEventRecordDefault));
            };
            mark(0);
            const int first = mb * minibatch_;
            gather_rows(perm, first, minibatch_, kObs, obs_.get(), mb_obs_.get(), stream_);
            gather_rows(perm, first, minibatch_, kAct, actions_.get(), mb_actions_.get(), stream_);
            gather_rows(perm, first, minibatch_, 1, logp_.get(), mb_logp_.get(), stream_);
            gather_rows(perm, first, minibatch_, 1, advantages_.get(), mb_adv_.get(), stream_);
            gather_rows(perm, first, minibatch_, 1, returns_.get(), mb_ret_.get(), stream_);
            mean_std(mb_adv_.get(), minibatch_, adv_stats_.get(), stream_);
            mark(1);

            actor_->forward(mb_obs_.get(), minibatch_, stream_);
            critic_->forward(mb_obs_.get(), minibatch_, stream_);
            gaussian_ppo_loss_grad(actor_->output(), log_std_.get(), critic_->output(), minibatch_, kAct,
                                   mb_actions_.get(), mb_logp_.get(), mb_adv_.get(), mb_ret_.get(),
                                   adv_stats_.get(), coeffs, dmean_.get(), dlog_std_rows_.get(),
                                   dvalues_.get(), row_stats_.get(), stats + kStatLosses, stream_);
            mark(2);

            actor_->zero_grad(stream_);
            actor_->backward(dmean_.get(), minibatch_, nullptr, stream_);
            actor_opt_->step(actor_->params(), actor_->grads(), stream_);

            CUDA_CHECK_THROW(cudaMemsetAsync(log_std_grad_.get(), 0, log_std_grad_.size_bytes(), stream_));
            sum_columns(dlog_std_rows_.get(), minibatch_, kAct, log_std_grad_.get(), stream_);
            log_std_opt_->step(log_std_.get(), log_std_grad_.get(), stream_);
            mark(3);

            critic_->zero_grad(stream_);
            critic_->backward(dvalues_.get(), minibatch_, nullptr, stream_);
            critic_opt_->step(critic_->params(), critic_->grads(), stream_);
            mark(4);
        }
    }
}

void RacingPpoTrainer::submit_optimization(float* stats) {
    if (!optimization_graph_) {
        optimize(stats);
        return;
    }
    const auto setup_start = std::chrono::steady_clock::now();
    // Re-capture host-derived parameters every update. In particular, Adam's
    // host counters advance once here, and its fresh double-precision bias
    // corrections are recorded. Capture submits no kernels for execution.
    CUDA_CHECK_THROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    try {
        optimize(stats);
    } catch (...) {
        cudaGraph_t abandoned = nullptr;
        cudaStreamEndCapture(stream_, &abandoned);
        if (abandoned)
            cudaGraphDestroy(abandoned);
        throw;
    }
    cudaGraph_t definition = nullptr;
    CUDA_CHECK_THROW(cudaStreamEndCapture(stream_, &definition));
    try {
        if (!optimization_graph_->executable) {
            CUDA_CHECK_THROW(cudaGraphInstantiate(&optimization_graph_->executable, definition, 0));
            size_t nodes = 0;
            CUDA_CHECK_THROW(cudaGraphGetNodes(definition, nullptr, &nodes));
            optimization_graph_->nodes = static_cast<int>(nodes);
        } else {
            cudaGraphExecUpdateResultInfo result{};
            CUDA_CHECK_THROW(cudaGraphExecUpdate(optimization_graph_->executable, definition, &result));
            if (result.result != cudaGraphExecUpdateSuccess)
                throw std::runtime_error("RacingPpoTrainer: optimization graph topology changed");
        }
        CUDA_CHECK_THROW(cudaGraphDestroy(definition));
        definition = nullptr;
        optimization_graph_->setup_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - setup_start).count();
        CUDA_CHECK_THROW(cudaGraphLaunch(optimization_graph_->executable, stream_));
    } catch (...) {
        if (definition)
            cudaGraphDestroy(definition);
        throw;
    }
}

RacingStats RacingPpoTrainer::update() {
    const auto start = std::chrono::steady_clock::now();
    const float lr = cfg_.anneal_lr
                         ? cfg_.lr * std::max(0.f, 1.f - static_cast<float>(update_) /
                                                             static_cast<float>(cfg_.total_updates))
                         : cfg_.lr;
    actor_opt_->set_learning_rate(lr);
    critic_opt_->set_learning_rate(lr);
    log_std_opt_->set_learning_rate(lr);

    CUDA_CHECK_THROW(cudaMemsetAsync(stats_.get(), 0, stats_.size_bytes(), stream_));
    timing_mark(0);
    rollout();
    timing_mark(2);
    episode_totals(finished_len_.get(), finished_laps_.get(), dones_.get(), truncated_.get(), batch_,
                   stats_.get() + kStatEpisodes, stream_);
    sum_columns(race_metrics_.get(),batch_,4,stats_.get()+kStatRace,stream_);
    timing_mark(3);
    const auto optimization_start=std::chrono::steady_clock::now();
    submit_optimization(stats_.get());
    const double optimization_submit_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-optimization_start).count();
    timing_mark(4);

    float host[kNumStats];
    CUDA_CHECK_THROW(cudaMemcpyAsync(host, stats_.get(), sizeof host, cudaMemcpyDeviceToHost, stream_));
    timing_mark(5);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream_));

    ++update_;
    env_steps_ += batch_;
    RacingStats s;
    s.contact_cars=static_cast<int>(host[kStatRace]);
    s.clean_laps=static_cast<int>(host[kStatRace+1]);
    s.mean_lap_seconds=s.clean_laps>0 ? host[kStatRace+2]/s.clean_laps:0.f;
    s.wall_crashes=static_cast<int>(host[kStatRace+3]);
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
    if (optimization_graph_) {
        s.optimization_graph_setup_ms=optimization_graph_->setup_ms;
        s.optimization_graph_nodes=optimization_graph_->nodes;
    }
    if (phase_timer_) {
        s.optimization_submit_ms=optimization_submit_ms;
        s.rollout_ms = phase_timer_->elapsed(0, 1);
        s.bootstrap_gae_ms = phase_timer_->elapsed(1, 2);
        s.metrics_ms = phase_timer_->elapsed(2, 3);
        s.optimize_ms = phase_timer_->elapsed(3, 4);
        s.readback_ms = phase_timer_->elapsed(4, 5);
        if (cfg_.traffic) for (const auto& step : phase_timer_->simulation) {
            s.advance_ms += RacingPhaseTimer::span(step.start, step.advanced);
            s.resolve_ms += RacingPhaseTimer::span(step.advanced, step.resolved);
            s.observe_ms += RacingPhaseTimer::span(step.resolved, step.observed);
        }
        for (const auto& minibatch : phase_timer_->optimization) {
            s.gather_ms+=RacingPhaseTimer::span(minibatch[0],minibatch[1]);
            s.forward_loss_ms+=RacingPhaseTimer::span(minibatch[1],minibatch[2]);
            s.actor_update_ms+=RacingPhaseTimer::span(minibatch[2],minibatch[3]);
            s.critic_update_ms+=RacingPhaseTimer::span(minibatch[3],minibatch[4]);
        }
    }
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
