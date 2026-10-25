#include "racing_lab.h"

#include <exception>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include "cuda/device.h"

namespace godot {

namespace {

// x, y, heading, speed, laps, steps, crashed, then the raycast distances.
constexpr int kCarStride = 7 + cudarl::racing::kRays;

PackedVector2Array closed_polyline(const cudarl::racing::Vec2* points, int count) {
    PackedVector2Array out;
    for (int k = 0; k <= count; ++k) {
        const auto& p = points[k % count];
        out.push_back(Vector2(p.x, p.y));
    }
    return out;
}

} // namespace

void RacingLab::_bind_methods() {
    ClassDB::bind_method(D_METHOD("start", "num_envs", "seed", "showcase_count"), &RacingLab::start);
    ClassDB::bind_method(D_METHOD("stop"), &RacingLab::stop);
    ClassDB::bind_method(D_METHOD("get_error"), &RacingLab::get_error);
    ClassDB::bind_method(D_METHOD("set_updates_per_second", "ups"), &RacingLab::set_updates_per_second);
    ClassDB::bind_method(D_METHOD("set_paused", "paused"), &RacingLab::set_paused);
    ClassDB::bind_method(D_METHOD("is_finished"), &RacingLab::is_finished);
    ClassDB::bind_method(D_METHOD("get_stats"), &RacingLab::get_stats);
    ClassDB::bind_method(D_METHOD("get_learning_curve"), &RacingLab::get_learning_curve);
    ClassDB::bind_method(D_METHOD("get_track_outer"), &RacingLab::get_track_outer);
    ClassDB::bind_method(D_METHOD("get_track_inner"), &RacingLab::get_track_inner);
    ClassDB::bind_method(D_METHOD("step_showcase"), &RacingLab::step_showcase);
    ClassDB::bind_method(D_METHOD("get_car_stride"), &RacingLab::get_car_stride);
    ClassDB::bind_method(D_METHOD("reset_showcase"), &RacingLab::reset_showcase);
}

bool RacingLab::start(int num_envs, int seed, int showcase_count) {
    stop();
    error_ = String();
    const cudarl::DeviceInfo device = cudarl::probe_device();
    if (!device.available) {
        error_ = String("CUDA unavailable: ") + device.error.c_str();
        return false;
    }
    try {
        cudarl::rl::RacingPpoConfig cfg;
        cfg.num_envs = num_envs;
        cfg.seed     = static_cast<uint32_t>(seed);
        showcase_ = std::make_unique<cudarl::demos::RacingShowcase>(showcase_count, cfg.hidden, cfg.seed + 7);
        session_  = std::make_unique<cudarl::demos::TrainingSession<cudarl::rl::RacingPpoTrainer>>(cfg);
    } catch (const std::exception& e) {
        error_ = e.what();
        stop();
        return false;
    }
    return true;
}

void RacingLab::stop() {
    session_.reset();   // joins the training thread
    showcase_.reset();
    history_.clear();
    policy_version_ = 0;
    policy_.clear();
}

String RacingLab::get_error() const {
    if (!error_.is_empty()) return error_;
    return session_ ? String(session_->error().c_str()) : String();
}

void RacingLab::set_updates_per_second(double ups) {
    if (session_) session_->set_updates_per_second(ups);
}

void RacingLab::set_paused(bool paused) {
    if (session_) session_->set_paused(paused);
}

bool RacingLab::is_finished() const {
    return session_ && session_->finished();
}

void RacingLab::sync_history() {
    if (session_) session_->history_since(history_.size(), history_);
}

Dictionary RacingLab::get_stats() {
    sync_history();
    Dictionary d;
    if (history_.empty()) return d;
    const auto& s = history_.back();
    d["update"]               = s.update;
    d["env_steps"]            = static_cast<int64_t>(s.env_steps);
    d["mean_episode_length"]  = s.mean_episode_length;
    d["mean_laps"]            = s.mean_laps;
    d["crash_fraction"]       = s.crash_fraction;
    d["entropy"]              = s.entropy;
    d["approx_kl"]            = s.approx_kl;
    d["clip_fraction"]        = s.clip_fraction;
    d["mean_std"]             = s.mean_std;
    d["learning_rate"]        = s.learning_rate;
    d["update_ms"]            = s.update_ms;
    d["env_steps_per_second"] = s.env_steps_per_second;
    return d;
}

PackedVector2Array RacingLab::get_learning_curve() {
    sync_history();
    PackedVector2Array curve;
    for (const auto& s : history_) {
        if (s.episodes > 0) curve.push_back(Vector2(static_cast<real_t>(s.env_steps), s.mean_laps));
    }
    return curve;
}

PackedVector2Array RacingLab::get_track_outer() const {
    const auto track = cudarl::racing::make_track();
    return closed_polyline(track.outer, cudarl::racing::kWallPoints);
}

PackedVector2Array RacingLab::get_track_inner() const {
    const auto track = cudarl::racing::make_track();
    return closed_polyline(track.inner, cudarl::racing::kWallPoints);
}

PackedFloat32Array RacingLab::step_showcase() {
    PackedFloat32Array out;
    if (!showcase_ || !session_) return out;
    if (session_->latest_policy(policy_version_, policy_)) showcase_->set_policy(policy_);
    showcase_->step();
    const int n = showcase_->count();
    out.resize(static_cast<int64_t>(n) * kCarStride);
    for (int i = 0; i < n; ++i) {
        const auto& s = showcase_->state(i);
        const int64_t base = static_cast<int64_t>(i) * kCarStride;
        out.set(base + 0, s.x);
        out.set(base + 1, s.y);
        out.set(base + 2, s.heading);
        out.set(base + 3, s.speed);
        out.set(base + 4, s.progress / cudarl::racing::kTwoPi);
        out.set(base + 5, static_cast<float>(s.steps));
        out.set(base + 6, showcase_->just_crashed(i) ? 1.f : 0.f);
        for (int r = 0; r < cudarl::racing::kRays; ++r) out.set(base + 7 + r, showcase_->ray(i, r));
    }
    return out;
}

int RacingLab::get_car_stride() const {
    return kCarStride;
}

void RacingLab::reset_showcase() {
    if (showcase_) showcase_->reset();
}

} // namespace godot
