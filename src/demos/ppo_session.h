#ifndef CUDARL_DEMOS_PPO_SESSION_H
#define CUDARL_DEMOS_PPO_SESSION_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "rl/evaluated_policy.h"
#include "core/checkpoint.h"

// Runs a trainer on a background thread so a renderer stays responsive, and
// publishes each update's stats and a host copy of the policy. Training can be
// throttled (updates per second) so that background learning can be observed
// without blocking presentation. Works with any trainer
// that has Config and Stats types, update(), and actor_params().
namespace cudarl::demos {

template<class T,class=void> struct HasBestPolicy:std::false_type {};
template<class T> struct HasBestPolicy<T,std::void_t<decltype(std::declval<const T&>().best_policy())>>:std::true_type {};
template<class T,class=void> struct HasTrainingResume:std::false_type {};
template<class T> struct HasTrainingResume<T,std::void_t<decltype(std::declval<const T&>().training_history()),
    decltype(std::declval<const T&>().checkpoint_status()),decltype(std::declval<T&>().save_checkpoint())>>:std::true_type {};

template <typename Trainer>
class TrainingSession {
public:
    using Config = typename Trainer::Config;
    using Stats  = typename Trainer::Stats;

    explicit TrainingSession(const Config& config, double updates_per_second = 2.0, bool initially_paused = false)
        : cfg_(config), ups_(updates_per_second), paused_(initially_paused), thread_([this] { run(); }) {}
    ~TrainingSession() { stop(); }
    void stop() {   // Idempotent; leave publications readable after joining.
        stop_.store(true);
        if (thread_.joinable()) thread_.join();
    }
    TrainingSession(const TrainingSession&) = delete;
    TrainingSession& operator=(const TrainingSession&) = delete;

    const Config& config() const { return cfg_; }
    void set_updates_per_second(double ups) { ups_.store(ups); }   // <= 0: as fast as possible
    void set_paused(bool paused) { paused_.store(paused); }
    bool paused() const { return paused_.load(); }
    bool finished() const { return finished_.load(); }   // exhausted a finite update budget
    bool ready() const {return ready_.load();}
    void request_save() {save_requested_.store(true);}
    checkpoint::Status checkpoint_status() const {std::lock_guard<std::mutex> lock(mutex_);return checkpoint_status_;}

    std::string error() const {   // non-empty if training failed
        std::lock_guard<std::mutex> lock(mutex_);
        return error_;
    }

    // Appends stats for updates [from, count) to out; returns the new count.
    size_t history_since(size_t from, std::vector<Stats>& out) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = from; i < history_.size(); ++i) out.push_back(history_[i]);
        return history_.size();
    }

    // Copies the latest policy parameters if they changed since `version`.
    bool latest_policy(uint64_t& version, std::vector<float>& params,int* update=nullptr) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (policy_version_ == version || policy_.empty()) return false;
        params  = policy_;
        version = policy_version_;
        if(update) *update=policy_update_;
        return true;
    }

    // Immutable initialization actor: update zero, or the restored actor for
    // resumable trainers, published before this session's optimizer steps.
    bool initial_policy(std::vector<float>& params) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initial_policy_.empty()) return false;
        params = initial_policy_;
        return true;
    }
    bool latest_best_policy(uint64_t& version,rl::EvaluatedPolicy& policy) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!best_policy_.valid() || version==best_policy_version_) return false;
        policy=best_policy_;version=best_policy_version_;return true;
    }

private:
    // Called under mutex, at the same publication boundary as the latest actor.
    void publish_best(const Trainer& trainer) {
        if constexpr(HasBestPolicy<Trainer>::value) {
            const auto& best=trainer.best_policy();
            if(best.valid() && (!best_policy_.valid() || best.foods>best_policy_.foods)) {
                best_policy_=best;++best_policy_version_;
            }
        }
    }
    void run() {
        using clock = std::chrono::steady_clock;
        try {
            Trainer trainer(cfg_);   // created here so all its CUDA work belongs to this thread
            {
                std::lock_guard<std::mutex> lock(mutex_);
                policy_         = trainer.actor_params();
                initial_policy_ = policy_;
                policy_version_ = 1;
                if constexpr(HasTrainingResume<Trainer>::value) {
                    history_=trainer.training_history();policy_update_=trainer.current_stats().update;
                    checkpoint_status_=trainer.checkpoint_status();
                }
                publish_best(trainer);
                ready_.store(true);
            }
            auto save_requested=[&] {
                if constexpr(HasTrainingResume<Trainer>::value) if(save_requested_.exchange(false)) {
                    trainer.save_checkpoint();std::lock_guard<std::mutex> lock(mutex_);checkpoint_status_=trainer.checkpoint_status();
                }
            };
            auto next = clock::now();
            // Zero means continuous learning; stopping joins this same trainer,
            // preserving its policy and optimizer state throughout the session.
            for (uint64_t u = 0; !stop_.load() &&
                 (cfg_.total_updates == 0 || u < static_cast<uint64_t>(cfg_.total_updates)); ++u) {
                // Wait while paused, or until the throttle allows the next update.
                while (!stop_.load() && (paused_.load() || clock::now() < next)) {
                    save_requested();
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                if (stop_.load()) break;
                save_requested();
                const double ups = ups_.load();
                next = clock::now() + (ups > 0 ? std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / ups))
                                               : clock::duration::zero());
                const Stats stats = trainer.update();
                std::vector<float> params = trainer.actor_params();
                std::lock_guard<std::mutex> lock(mutex_);
                history_.push_back(stats);
                policy_ = std::move(params);
                ++policy_version_;
                policy_update_=stats.update;
                publish_best(trainer);
                if constexpr(HasTrainingResume<Trainer>::value) checkpoint_status_=trainer.checkpoint_status();
            }
            if constexpr(HasTrainingResume<Trainer>::value) {
                trainer.save_checkpoint();std::lock_guard<std::mutex> lock(mutex_);checkpoint_status_=trainer.checkpoint_status();
            }
            finished_.store(!stop_.load());
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(mutex_);
            error_ = e.what();
        }
    }

    Config                  cfg_;
    std::atomic<double>     ups_{2.0};
    std::atomic<bool>       paused_{false}, stop_{false}, finished_{false};
    std::atomic<bool>       ready_{false},save_requested_{false};
    mutable std::mutex      mutex_;
    std::vector<Stats>      history_;
    std::vector<float>      policy_;
    std::vector<float>      initial_policy_;
    uint64_t                policy_version_ = 0;
    int policy_update_=0;
    checkpoint::Status checkpoint_status_;
    std::string             error_;
    rl::EvaluatedPolicy     best_policy_;
    uint64_t                best_policy_version_=0;
    std::thread             thread_;   // last: starts after every member above is initialized
};

} // namespace cudarl::demos

#endif // CUDARL_DEMOS_PPO_SESSION_H
