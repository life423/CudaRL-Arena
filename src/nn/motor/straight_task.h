#ifndef CUDARL_STRAIGHT_TASK_H
#define CUDARL_STRAIGHT_TASK_H
#include "motor/joint_motor.h"
// Shared requested-speed task: pure observations/reward/acceptance, no ownership.
namespace cudarl::straight_motor {
namespace p = physics;
namespace joint = joint_motor;
namespace old = motor;
using World = joint::World;
using Action = joint::Action;
using Outcome = old::Outcome;
using joint::action_from_latent;
using joint::step_state;
constexpr int kTaskVersion = 6, kObservations = 40, kActions = 4;
constexpr int kRewardVersion = 4;
struct ForwardConfig {
    float target_speed_m_s = .008f, clearance_m = .01f;
    float minimum_speed_m_s = .0072f, maximum_speed_m_s = .0088f, speed_rmse_limit = .0025f;
    float lateral_limit_m = .002f, depth_limit_m = .001f, path_angle_limit_deg = 5, tilt_limit_deg = 10;
    float work_per_distance_limit = 0.001f; // J/m = 1 microJ/mm
    int maximum_controls = 160, measure_start = 40;
};
CUDARL_HD inline bool supported_speed(float speed) {
    return p::finite_number(speed) && speed >= .004f && speed <= .008f;
}
CUDARL_HD inline ForwardConfig config_for_speed(float speed) {
    ForwardConfig c;
    c.target_speed_m_s = speed;
    c.minimum_speed_m_s = .9f * speed;
    c.maximum_speed_m_s = 1.1f * speed;
    c.speed_rmse_limit = .3125f * speed;
    return c;
}
CUDARL_HD inline contact::Arena long_swim_arena() { return {.6f, .4f}; }
// One episode contract is passed unchanged to every training kernel. Evaluation
// retains its own existing ForwardConfig and is independent of training horizon.
struct TrainingEpisode {
    int controls = 160;
    bool tilt_mixture = false;
    CUDARL_HD bool valid() const { return controls == 160 || controls == 640; }
    CUDARL_HD contact::Arena arena() const { return controls == 640 ? long_swim_arena() : contact::Arena{}; }
    CUDARL_HD ForwardConfig task_config() const {
        auto c = config_for_speed(.008f);
        c.maximum_controls = controls;
        return c;
    }
};
CUDARL_HD inline World initial_world(int id) {
    World w;
    w.physical.paused = false;
    w.physical.body.orientation = p::axis_rotation({0, 1, 0}, (float(id % 4) - 1.5f) * .05f);
    w.physical.body.position.x = (float(id % 4) - 1.5f) * .02f;
    w.physical.body.angle[0] = id % 2 ? .02f : -.02f;
    w.physical.body.angle[2] = -w.physical.body.angle[0];
    for (int j = 0; j < 4; ++j)
        w.physical.held_targets[j] = w.physical.body.angle[j];
    return w;
}
// Training-only rigid pose variation: keep the level start's COM depth and
// all relative joint angles/targets. This is initialization, never a correction.
CUDARL_HD inline World initial_training_world(int environment, bool tilt_mixture) {
    auto w = initial_world(environment % 2 ? 3 : 0);
    if (tilt_mixture) {
        const auto target = old::motion(w.physical.body).com_model;
        const int group = (environment / 2) % 4;
        const float pitch = group == 1 ? 3.f : (group == 2 ? -3.f : 0.f);
        w.physical.body.orientation =
            w.physical.body.orientation * p::axis_rotation({0, 0, 1}, pitch * movement::pi / 180.f);
        const auto posed = old::motion(w.physical.body).com_model;
        w.physical.body.position.y += target.y - posed.y;
    }
    return w;
}
struct Metrics {
    float forward_m = 0, lateral_m = 0, depth_m = 0, speed_error_m_s = 0, head_yaw_deg = 0, tilt_deg = 0;
};
struct ForwardTask {
    simulation::MotorRequest request{};
    p::V start_com_model{}, steady_start_com_model{};
    uint64_t last_world_control = 0;
    float start_work_J = 0;
    int controls = 0, steady_samples = 0;
    double velocity_error_sq = 0, head_yaw_sq = 0;
    float steady_speed_m_s = 0, speed_rmse = 0, path_angle_deg = 0, maximum_lateral = 0, maximum_depth = 0,
          maximum_tilt = 0, maximum_head_yaw = 0;
    Outcome outcome = Outcome::Running;
};
// Dimensionless cost rates; multiply their sum by the physical control interval.
struct RewardCosts {
    float speed = 0, lane = 0, depth = 0, posture = 0, effort = 0;
    CUDARL_HD float total() const { return speed + lane + depth + posture + effort; }
};
struct Transition {
    Metrics metrics{};
    RewardCosts costs{};
    float reward = 0;
    bool accepted = false, done = false, truncated = false;
    Outcome outcome = Outcome::Rejected;
};
CUDARL_HD inline bool valid(const ForwardConfig &c) {
    const float fields[] = {c.target_speed_m_s,       c.clearance_m,          c.minimum_speed_m_s,
                            c.maximum_speed_m_s,      c.speed_rmse_limit,     c.lateral_limit_m,
                            c.depth_limit_m,          c.path_angle_limit_deg, c.tilt_limit_deg,
                            c.work_per_distance_limit};
    for (float x : fields)
        if (!p::finite_number(x) || x <= 0)
            return false;
    return supported_speed(c.target_speed_m_s) && c.maximum_speed_m_s >= c.minimum_speed_m_s &&
           c.clearance_m >= .01f && c.measure_start > 0 && c.maximum_controls > c.measure_start;
}
CUDARL_HD inline bool in_domain(const World &w, const ForwardConfig &c) {
    return valid(c) && joint::in_domain(w, c.clearance_m);
}
CUDARL_HD inline bool begin_forward(const World &w, const ForwardConfig &c, ForwardTask &output) {
    if (!in_domain(w, c))
        return false;
    const auto motion = old::motion(w.physical.body);
    auto heading = w.physical.body.orientation.rotate({1, 0, 0});
    heading = heading / p::norm(heading);
    ForwardTask next;
    next.start_com_model = motion.com_model;
    next.steady_start_com_model = motion.com_model;
    next.last_world_control = w.physical.control_steps;
    next.start_work_J = w.physical.body.work;
    next.request = {motion.com_model + heading * (.072f / p::scale), heading * c.target_speed_m_s, heading};
    output = next;
    return true;
}
CUDARL_HD inline bool begin_training_episode(int environment, bool tilt_mixture, const ForwardConfig &config,
                                             World &world, ForwardTask &task, contact::Arena arena = {}) {
    world = initial_training_world(environment, tilt_mixture);
    world.arena = arena;
    if (!begin_forward(world, config, task))
        return false;
    if (tilt_mixture) {
        // A tilted start must still request the SAME horizontal swim, not a
        // climb/dive along the tilted head's forward axis.
        const auto level = initial_world(environment % 2 ? 3 : 0);
        const auto heading = level.physical.body.orientation.rotate({1, 0, 0});
        task.request = {task.start_com_model + heading * (.072f / p::scale),
                        heading * config.target_speed_m_s, heading};
    }
    return true;
}
CUDARL_HD inline bool request_speed(const World &w, float speed, ForwardConfig &c, ForwardTask &t) {
    if (!supported_speed(speed) || !in_domain(w, c))
        return false;
    const auto heading = t.request.heading;
    const auto next = config_for_speed(speed);
    ForwardTask segment;
    if (!begin_forward(w, next, segment))
        return false;
    segment.request.heading = heading;
    segment.request.terminal_velocity = heading * speed;
    c = next;
    t = segment;
    return true;
}
CUDARL_HD inline bool observe(const ForwardTask &t, const World &w, const ForwardConfig &c, float *out) {
    if (!out || t.outcome != Outcome::Running || !in_domain(w, c))
        return false;
    const auto motion = old::motion(w.physical.body);
    const auto &q = w.physical.body.orientation;
    const auto displacement = (motion.com_model - t.start_com_model) * p::scale;
    const auto lane_error = t.request.heading * p::dot(displacement, t.request.heading) - displacement;
    old::vector_features(out, 0, q.inverse(lane_error) / old::kBodyLength);
    old::vector_features(out, 3, q.inverse(t.request.terminal_velocity) / old::kVelocityScale);
    old::vector_features(out, 6, q.inverse(t.request.heading));
    old::vector_features(out, 9, q.inverse({0, 1, 0}));
    old::vector_features(out, 12, q.inverse(motion.velocity_m_s) / old::kVelocityScale);
    old::vector_features(
        out, 15,
        q.inverse({w.physical.body.velocity[3], w.physical.body.velocity[4], w.physical.body.velocity[5]}) /
            old::kAngularScale);
    for (int j = 0; j < 4; ++j) {
        out[18 + j] = w.physical.body.angle[j] / p::limit;
        out[22 + j] = w.physical.body.velocity[6 + j] / 12.f;
        out[26 + j] = w.physical.last_actual_torques[j];
        out[30 + j] = w.physical.held_targets[j] / p::limit;
    }
    out[34] = p::clamp(1.f - float(t.controls) / float(c.measure_start), 0, 1);
    out[35] = w.physical.actuation == simulation::Actuation::JointTarget ? 1.f : 0.f;
    for (int j = 0; j < 4; ++j)
        out[36 + j] = w.reference_rate[j] / 12.f;
    for (int j = 0; j < 40; ++j)
        if (!p::finite_number(out[j]))
            return false;
    return true;
}
CUDARL_HD inline Metrics measure(const ForwardTask &t, const World &w) {
    const auto motion = old::motion(w.physical.body);
    const auto displacement = (motion.com_model - t.start_com_model) * p::scale;
    Metrics result;
    result.forward_m = p::dot(displacement, t.request.heading);
    const auto perpendicular = displacement - t.request.heading * result.forward_m;
    result.lateral_m = p::norm({perpendicular.x, 0, perpendicular.z});
    result.depth_m = fabsf(displacement.y);
    result.speed_error_m_s = p::norm(motion.velocity_m_s - t.request.terminal_velocity);
    const auto forward = w.physical.body.orientation.rotate({1, 0, 0});
    const auto up = w.physical.body.orientation.rotate({0, 1, 0});
    result.head_yaw_deg = fabsf(atan2f(p::dot(p::cross(t.request.heading, forward), {0, 1, 0}),
                                       p::dot(t.request.heading, forward))) *
                          180 / movement::pi;
    result.tilt_deg = acosf(p::clamp(p::dot(up, {0, 1, 0}), -1, 1)) * 180 / movement::pi;
    return result;
}
CUDARL_HD inline RewardCosts reward_costs(const Metrics &metrics, const World &world,
                                          const ForwardConfig &config, bool speed_scored) {
    const float velocity = metrics.speed_error_m_s / config.target_speed_m_s;
    const float lane = metrics.lateral_m / config.lateral_limit_m;
    const float depth = metrics.depth_m / config.depth_limit_m;
    // BodyState.orientation is link0 (the head), not an average of bent links.
    // World Y is up. This yaw-invariant tilt combines roll and pitch; no pose clamp.
    const float tilt = metrics.tilt_deg / config.tilt_limit_deg;
    return {speed_scored ? velocity * velocity : 0.f, .05f * lane * lane, .05f * depth * depth,
            .05f * tilt * tilt, .002f * world.physical.mean_squared_torque};
}
CUDARL_HD inline Transition record_control(ForwardTask &t, const World &w, const ForwardConfig &c) {
    Transition out;
    if (t.outcome != Outcome::Running) {
        out.done = true;
        out.outcome = t.outcome;
        return out;
    }
    if (!in_domain(w, c) || w.physical.control_steps != t.last_world_control + 1) {
        t.outcome = out.outcome = Outcome::Rejected;
        out.done = true;
        return out;
    }
    out.metrics = measure(t, w);
    // Interval40 ends at2s; intervals41..160 are the scored2..8s window.
    // ONLY speed has a startup window. Lane, depth, head posture and effort are
    // charged on the very first interval and all later intervals. Full trajectory
    // credit, physical work, safety and acceptance bookkeeping never reset.
    const bool scored = t.controls >= c.measure_start;
    out.costs = reward_costs(out.metrics, w, c, scored);
    out.reward = -p::dt * out.costs.total();
    if (!p::finite_number(out.reward)) {
        t.outcome = out.outcome = Outcome::Rejected;
        out.done = true;
        return out;
    }
    ++t.controls;
    t.last_world_control = w.physical.control_steps;
    t.maximum_lateral = fmaxf(t.maximum_lateral, out.metrics.lateral_m);
    t.maximum_depth = fmaxf(t.maximum_depth, out.metrics.depth_m);
    t.maximum_tilt = fmaxf(t.maximum_tilt, out.metrics.tilt_deg);
    t.maximum_head_yaw = fmaxf(t.maximum_head_yaw, out.metrics.head_yaw_deg);
    t.head_yaw_sq += out.metrics.head_yaw_deg * out.metrics.head_yaw_deg;
    const auto motion = old::motion(w.physical.body);
    if (t.controls == c.measure_start)
        t.steady_start_com_model = motion.com_model;
    if (t.controls > c.measure_start) {
        ++t.steady_samples;
        t.velocity_error_sq += out.metrics.speed_error_m_s * out.metrics.speed_error_m_s;
        const auto delta = (motion.com_model - t.steady_start_com_model) * p::scale;
        const float forward = p::dot(delta, t.request.heading);
        const auto perpendicular = delta - t.request.heading * forward;
        t.steady_speed_m_s = forward / (float(t.steady_samples) * p::dt);
        t.speed_rmse = sqrtf(static_cast<float>(t.velocity_error_sq / t.steady_samples));
        t.path_angle_deg =
            atan2f(p::norm({perpendicular.x, 0, perpendicular.z}), forward) * 180 / movement::pi;
    }
    if (t.controls >= c.maximum_controls) {
        const float work_per_m =
            (w.physical.body.work - t.start_work_J) / fmaxf(out.metrics.forward_m, 1e-9f);
        const bool success =
            t.steady_speed_m_s >= c.minimum_speed_m_s && t.steady_speed_m_s <= c.maximum_speed_m_s &&
            t.speed_rmse <= c.speed_rmse_limit && t.maximum_lateral <= c.lateral_limit_m &&
            t.maximum_depth <= c.depth_limit_m && t.path_angle_deg <= c.path_angle_limit_deg &&
            t.maximum_tilt <= c.tilt_limit_deg && work_per_m <= c.work_per_distance_limit;
        t.outcome = success ? Outcome::Success : Outcome::Timeout;
    }
    out.accepted = true;
    out.outcome = t.outcome;
    out.done = t.outcome != Outcome::Running;
    return out;
}
} // namespace cudarl::straight_motor
#endif
