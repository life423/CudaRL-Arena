#ifndef CUDARL_DEMOS_PPO_SESSION_H
#define CUDARL_DEMOS_PPO_SESSION_H

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rl/ppo.h"

// Runs PpoTrainer on a background thread so a renderer stays responsive, and
// publishes each update's stats and a host copy of the policy. Training can be
// throttled (updates per second) so that learning which takes a third of a
// second at full speed becomes something you can watch.
namespace cudarl::demos {

class PpoSession {
public:
    explicit PpoSession(const rl::PpoConfig& config);
    ~PpoSession();   // stops and joins the training thread
    PpoSession(const PpoSession&) = delete;
    PpoSession& operator=(const PpoSession&) = delete;

    const rl::PpoConfig& config() const { return cfg_; }
    void set_updates_per_second(double ups) { ups_.store(ups); }   // <= 0: as fast as possible
    void set_paused(bool paused) { paused_.store(paused); }
    bool paused() const { return paused_.load(); }
    bool finished() const { return finished_.load(); }   // ran all config().total_updates updates
    std::string error() const;                           // non-empty if training failed

    // Appends stats for updates [from, count) to out; returns the new count.
    size_t history_since(size_t from, std::vector<rl::PpoStats>& out) const;
    // Copies the latest policy parameters if they changed since `version`.
    bool latest_policy(uint64_t& version, std::vector<float>& params) const;

private:
    void run();

    rl::PpoConfig            cfg_;
    std::atomic<double>      ups_{2.0};
    std::atomic<bool>        paused_{false}, stop_{false}, finished_{false};
    mutable std::mutex       mutex_;
    std::vector<rl::PpoStats> history_;
    std::vector<float>       policy_;
    uint64_t                 policy_version_ = 0;
    std::string              error_;
    std::thread              thread_;
};

} // namespace cudarl::demos

#endif // CUDARL_DEMOS_PPO_SESSION_H
