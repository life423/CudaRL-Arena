#ifndef CUDARL_GODOT_PPO_LAB_H
#define CUDARL_GODOT_PPO_LAB_H

#include <memory>
#include <vector>

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include "demos/cartpole_showcase.h"
#include "demos/ppo_session.h"

namespace godot {

// GDScript wrapper for the live CartPole demo: PPO trains on a background
// thread (cudarl::demos::PpoSession) while a few showcase carts run on the CPU
// with the latest policy. Nothing throws into Godot; failures (e.g. no CUDA
// device) come back through get_error().
class PpoLab : public RefCounted {
    GDCLASS(PpoLab, RefCounted);

protected:
    static void _bind_methods();

public:
    // Starts (or restarts) training with num_envs environments and a seed.
    bool   start(int num_envs, int seed, int showcase_count);
    void   stop();
    String get_error() const;

    void set_updates_per_second(double ups);   // <= 0: full speed
    void set_paused(bool paused);
    bool is_finished() const;

    // Latest update's stats: { update, env_steps, mean_episode_length, entropy,
    // approx_kl, clip_fraction, learning_rate, update_ms, env_steps_per_second }.
    Dictionary get_stats();
    // (env_steps, mean episode length) for every update so far.
    PackedVector2Array get_learning_curve();

    // Steps every showcase cart once (call at 50 Hz) with the newest policy.
    // Returns [x, theta, steps, failed] per cart.
    PackedFloat32Array step_showcase(bool greedy);
    void reset_showcase();

private:
    void sync_history();

    std::unique_ptr<cudarl::demos::PpoSession>       session_;
    std::unique_ptr<cudarl::demos::CartPoleShowcase> showcase_;
    std::vector<cudarl::rl::PpoStats>                history_;
    uint64_t                                         policy_version_ = 0;
    std::vector<float>                               policy_;
    String                                           error_;
};

} // namespace godot

#endif // CUDARL_GODOT_PPO_LAB_H
