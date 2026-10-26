#ifndef CUDARL_DEMOS_PPO_SESSION_H
#define CUDARL_DEMOS_PPO_SESSION_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rl/ppo.h"

// Runs a trainer on a background thread so a renderer stays responsive, and
// publishes each update's stats and a host copy of the policy. Training can be
// throttled (updates per second) so that learning which takes a third of a
// second at full speed becomes something you can watch. Works with any trainer
// that has Config and Stats types, update(), and actor_params().
namespace cudarl::demos {

template <typename Trainer>
class TrainingSession {
public:
    using Config = typename Trainer::Config;
    using Stats  = typename Trainer::Stats;

    explicit TrainingSession(const Config& config, double updates_per_second = 2.0)
        : cfg_(config), ups_(updates_per_second), thread_([this] { run(); }) {}
    ~TrainingSession() {   // stops and joins the training thread
        stop_.store(true);
        if (thread_.joinable()) thread_.join();
    }
    TrainingSession(const TrainingSession&) = delete;
    TrainingSession& operator=(const TrainingSession&) = delete;

    const Config& config() const { return cfg_; }
    void set_updates_per_second(double ups) { ups_.store(ups); }   // <= 0: as fast as possible
    void set_paused(bool paused) { paused_.store(paused); }
    bool paused() const { return paused_.load(); }
    bool finished() const { return finished_.load(); }   // ran all config().total_updates updates

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
    bool latest_policy(uint64_t& version, std::vector<float>& params) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (policy_version_ == version || policy_.empty()) return false;
        params  = policy_;
        version = policy_version_;
        return true;
    }

    // Immutable update-zero checkpoint, published before any optimizer step.
    bool initial_policy(std::vector<float>& params) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initial_policy_.empty()) return false;
        params = initial_policy_;
        return true;
    }

private:
    void run() {
        using clock = std::chrono::steady_clock;
        try {
            Trainer trainer(cfg_);   // created here so all its CUDA work belongs to this thread
            {
                std::lock_guard<std::mutex> lock(mutex_);
                policy_         = trainer.actor_params();
                initial_policy_ = policy_;
                policy_version_ = 1;
            }
            auto next = clock::now();
            for (int u = 0; u < cfg_.total_updates && !stop_.load(); ++u) {
                // Wait while paused, or until the throttle allows the next update.
                while (!stop_.load() && (paused_.load() || clock::now() < next)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                if (stop_.load()) break;
                const double ups = ups_.load();
                next = clock::now() + (ups > 0 ? std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / ups))
                                               : clock::duration::zero());
                const Stats stats = trainer.update();
                std::vector<float> params = trainer.actor_params();
                std::lock_guard<std::mutex> lock(mutex_);
                history_.push_back(stats);
                policy_ = std::move(params);
                ++policy_version_;
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
    mutable std::mutex      mutex_;
    std::vector<Stats>      history_;
    std::vector<float>      policy_;
    std::vector<float>      initial_policy_;
    uint64_t                policy_version_ = 0;
    std::string             error_;
    std::thread             thread_;   // last: starts after every member above is initialized
};

using PpoSession = TrainingSession<rl::PpoTrainer>;

} // namespace cudarl::demos

#endif // CUDARL_DEMOS_PPO_SESSION_H
