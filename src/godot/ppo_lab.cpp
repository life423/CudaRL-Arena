#include "ppo_lab.h"

#include <exception>
#include <cmath>
#include <stdexcept>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include "cuda/device.h"

namespace godot {

void PpoLab::_bind_methods() {
    ClassDB::bind_method(D_METHOD("start", "num_envs", "seed", "showcase_count", "updates_per_second"), &PpoLab::start, DEFVAL(2.0));
    ClassDB::bind_method(D_METHOD("watch_policy", "parameters"), &PpoLab::watch_policy);
    ClassDB::bind_method(D_METHOD("stop"), &PpoLab::stop);
    ClassDB::bind_method(D_METHOD("get_error"), &PpoLab::get_error);
    ClassDB::bind_method(D_METHOD("set_updates_per_second", "ups"), &PpoLab::set_updates_per_second);
    ClassDB::bind_method(D_METHOD("set_paused", "paused"), &PpoLab::set_paused);
    ClassDB::bind_method(D_METHOD("is_finished"), &PpoLab::is_finished);
    ClassDB::bind_method(D_METHOD("get_stats"), &PpoLab::get_stats);
    ClassDB::bind_method(D_METHOD("get_learning_curve"), &PpoLab::get_learning_curve);
    ClassDB::bind_method(D_METHOD("step_showcase", "greedy"), &PpoLab::step_showcase);
    ClassDB::bind_method(D_METHOD("reset_showcase"), &PpoLab::reset_showcase);
}

bool PpoLab::start(int num_envs, int seed, int showcase_count, double updates_per_second) {
    stop();
    error_ = String();
    const cudarl::DeviceInfo device = cudarl::probe_device();
    if (!device.available) {
        error_ = String("CUDA unavailable: ") + device.error.c_str();
        return false;
    }
    try {
        cudarl::rl::PpoConfig cfg;
        cfg.num_envs = num_envs;
        cfg.seed     = static_cast<uint32_t>(seed);
        showcase_ = std::make_unique<cudarl::demos::CartPoleShowcase>(showcase_count, cfg.hidden, cfg.seed + 7);
        session_  = std::make_unique<cudarl::demos::PpoSession>(cfg, updates_per_second, true);
    } catch (const std::exception& e) {
        error_ = e.what();
        stop();
        return false;
    }
    return true;
}

bool PpoLab::watch_policy(const PackedFloat32Array& parameters) {
    stop();
    error_ = String();
    try {
        constexpr int expected = 64*(cudarl::cartpole::kObsDim+1)+64*65+cudarl::cartpole::kNumActions*65;
        if (parameters.size()!=expected) throw std::invalid_argument("CartPole policy architecture does not match");
        const std::vector<float> weights(parameters.ptr(), parameters.ptr() + parameters.size());
        for (float weight : weights) if (!std::isfinite(weight)) throw std::invalid_argument("Policy contains a non-finite weight");
        showcase_ = std::make_unique<cudarl::demos::CartPoleShowcase>(12, 64, 8);
        showcase_->set_policy(weights);
    } catch (const std::exception& e) {
        stop();
        error_ = e.what();
        return false;
    }
    return true;
}

void PpoLab::stop() {
    session_.reset();   // joins the training thread
    showcase_.reset();
    history_.clear();
    policy_version_ = 0;
}

String PpoLab::get_error() const {
    if (!error_.is_empty()) return error_;
    return session_ ? String(session_->error().c_str()) : String();
}

void PpoLab::set_updates_per_second(double ups) {
    if (session_) session_->set_updates_per_second(ups);
}

void PpoLab::set_paused(bool paused) {
    if (session_) session_->set_paused(paused);
}

bool PpoLab::is_finished() const {
    return session_ && session_->finished();
}

void PpoLab::sync_history() {
    if (session_) session_->history_since(history_.size(), history_);
}

Dictionary PpoLab::get_stats() {
    sync_history();
    Dictionary d;
    d["playback"] = showcase_ && !session_;
    d["policy_ready"] = showcase_ && (!session_ || policy_version_>0);
    const auto s = history_.empty() ? cudarl::rl::PpoStats{} : history_.back();
    d["update"]               = s.update;
    d["env_steps"]            = static_cast<int64_t>(s.env_steps);
    d["mean_episode_length"]  = s.mean_episode_length;
    d["entropy"]              = s.entropy;
    d["approx_kl"]            = s.approx_kl;
    d["clip_fraction"]        = s.clip_fraction;
    d["learning_rate"]        = s.learning_rate;
    d["update_ms"]            = s.update_ms;
    d["env_steps_per_second"] = s.env_steps_per_second;
    return d;
}

PackedVector2Array PpoLab::get_learning_curve() {
    sync_history();
    PackedVector2Array curve;
    for (const auto& s : history_) {
        if (s.episodes > 0) curve.push_back(Vector2(static_cast<real_t>(s.env_steps), s.mean_episode_length));
    }
    return curve;
}

PackedFloat32Array PpoLab::step_showcase(bool greedy) {
    PackedFloat32Array out;
    if (!showcase_) return out;
    if (session_) {
        if (session_->latest_policy(policy_version_, policy_)) showcase_->set_policy(policy_);
        if (policy_version_==0) return out;
    }
    showcase_->step(greedy);
    out.resize(static_cast<int64_t>(showcase_->count()) * 4);
    for (int i = 0; i < showcase_->count(); ++i) {
        const auto& st = showcase_->state(i);
        out.set(i * 4 + 0, st.x);
        out.set(i * 4 + 1, st.theta);
        out.set(i * 4 + 2, static_cast<float>(st.steps));
        out.set(i * 4 + 3, showcase_->just_failed(i) ? 1.f : 0.f);
    }
    return out;
}

void PpoLab::reset_showcase() {
    if (showcase_) showcase_->reset();
}

} // namespace godot
