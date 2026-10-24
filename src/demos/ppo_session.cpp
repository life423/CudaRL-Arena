#include "ppo_session.h"

#include <chrono>
#include <exception>

namespace cudarl::demos {

PpoSession::PpoSession(const rl::PpoConfig& config) : cfg_(config), thread_([this] { run(); }) {}

PpoSession::~PpoSession() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
}

std::string PpoSession::error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

size_t PpoSession::history_since(size_t from, std::vector<rl::PpoStats>& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = from; i < history_.size(); ++i) out.push_back(history_[i]);
    return history_.size();
}

bool PpoSession::latest_policy(uint64_t& version, std::vector<float>& params) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (policy_version_ == version || policy_.empty()) return false;
    params  = policy_;
    version = policy_version_;
    return true;
}

void PpoSession::run() {
    using clock = std::chrono::steady_clock;
    try {
        rl::PpoTrainer trainer(cfg_);   // created here so all its CUDA work belongs to this thread
        {
            std::lock_guard<std::mutex> lock(mutex_);
            policy_ = trainer.actor_params();
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
            const rl::PpoStats stats = trainer.update();
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

} // namespace cudarl::demos
