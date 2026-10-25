#ifndef CUDARL_GODOT_RACING_LAB_H
#define CUDARL_GODOT_RACING_LAB_H

#include <memory>
#include <vector>

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include "demos/ppo_session.h"
#include "demos/racing_showcase.h"
#include "rl/racing_ppo.h"

namespace godot {

// GDScript wrapper for the live racing demo: a Gaussian PPO policy trains on a
// background thread (cudarl::demos::TrainingSession) while a few showcase cars
// drive the newest policy on the CPU. Nothing throws into Godot; failures (e.g.
// no CUDA device) come back through get_error().
class RacingLab : public RefCounted {
    GDCLASS(RacingLab, RefCounted);

protected:
    static void _bind_methods();

public:
    // Starts (or restarts) training with num_envs cars and a seed.
    bool   start(int num_envs, int seed, int showcase_count);
    void   stop();
    String get_error() const;

    void set_updates_per_second(double ups);   // <= 0: full speed
    void set_paused(bool paused);
    bool is_finished() const;

    // Latest update's stats: { update, env_steps, mean_episode_length, mean_laps,
    // crash_fraction, entropy, approx_kl, clip_fraction, mean_std, learning_rate,
    // update_ms, env_steps_per_second }.
    Dictionary get_stats();
    // (env_steps, mean laps per finished episode) for every update that finished episodes.
    PackedVector2Array get_learning_curve();

    // The two walls of the track as closed polylines (the first point is repeated at the end).
    PackedVector2Array get_track_outer() const;
    PackedVector2Array get_track_inner() const;

    // Steps every showcase car once (call at the physics rate) with the newest policy.
    // Returns get_car_stride() floats per car: x, y, heading, speed, laps, steps,
    // crashed (0 or 1), then the raycast distances in metres.
    PackedFloat32Array step_showcase();
    int                get_car_stride() const;
    void               reset_showcase();

private:
    void sync_history();

    std::unique_ptr<cudarl::demos::TrainingSession<cudarl::rl::RacingPpoTrainer>> session_;
    std::unique_ptr<cudarl::demos::RacingShowcase>                                showcase_;
    std::vector<cudarl::rl::RacingStats>                                          history_;
    uint64_t                                                                      policy_version_ = 0;
    std::vector<float>                                                            policy_;
    String                                                                        error_;
};

} // namespace godot

#endif // CUDARL_GODOT_RACING_LAB_H
