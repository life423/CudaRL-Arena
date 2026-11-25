#ifndef CUDARL_STRAIGHT_TASK_H
#define CUDARL_STRAIGHT_TASK_H
#include "motor/joint_motor.h"
#include "motor/observation_contract.h"
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
constexpr int kRewardVersion = 5;
// Reward references only: these NEVER command a joint or body velocity.
constexpr float kDepthSettlingSeconds = 2.f, kRecoverySpeedLimit = .001f;
constexpr float kRollCost = .05f, kPitchCost = .01f;
struct ForwardConfig {
    uint32_t observation_schema = motor_observation::legacy;
    // Normally equal to the nonzero request. A stopping request retains this
    // lesson scale instead of dividing by zero or reweighting depth/posture.
    float velocity_reward_scale_m_s = .008f;
    bool stationary_request = false;
    bool horizontal_from_first_step = false;
    bool position_hold = false; // Research schema3; no physical state or joint constraints.
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
    c.velocity_reward_scale_m_s = speed;
    c.minimum_speed_m_s = .9f * speed;
    c.maximum_speed_m_s = 1.1f * speed;
    c.speed_rmse_limit = .3125f * speed;
    return c;
}
CUDARL_HD inline contact::Arena long_swim_arena() { return {.6f, .4f}; }
// A goal schedule only: it never supplies joint actions or modifies world state.
// Activation in the training runner requires a separately registered protocol.
// ForwardTask terminal gates remain swimming-only; stopping acceptance must be
// scored separately by the explicit stopping experiment.
struct StopSequence {
    static constexpr int request_control = 640, maximum_controls = 1280;
    CUDARL_HD ForwardConfig config(int completed_controls, uint32_t observation_schema) const {
        auto c = config_for_speed(.008f);
        c.maximum_controls = maximum_controls;
        c.observation_schema = observation_schema;
        if (completed_controls >= request_control) {
            c.target_speed_m_s = 0;
            c.stationary_request = true;
        }
        return c;
    }
};
// One episode contract is passed unchanged to every training kernel. Evaluation
// retains its own existing ForwardConfig and is independent of training horizon.
struct TrainingEpisode {
    uint32_t observation_schema = motor_observation::legacy;
    int controls = 160;
    bool tilt_mixture = false;
    bool fixed_depth_offsets = false;
    bool stopping = false;
    bool startup_speed = false;
    bool holding = false;
    bool appended_hold = false;
    // Explicit research protocol; historical training defaults are unchanged.
    bool balanced_hold = false;
    bool rest_hold = false; // Explicit approved hold-from-rest research mode.
    float holding_reward_gain = 1.f;
    CUDARL_HD bool hold_episode(int environment) const {
        return holding && (balanced_hold ? (environment / 2) % 2 : environment % 2);
    }
    CUDARL_HD float reward_gain(int completed_controls, int environment) const {
        return hold_episode(environment) && (rest_hold || completed_controls >= 640) ? holding_reward_gain : 1.f;
    }
    int offset_phase = 0;
    CUDARL_HD float initial_vertical_offset(int environment) const {
        if (!fixed_depth_offsets || (rest_hold && hold_episode(environment)))
            return 0;
        const int group = (environment / 2 + offset_phase) % 3;
        return group == 1 ? -.0005f : (group == 2 ? .0005f : 0.f);
    }
    CUDARL_HD bool valid() const {
        return motor_observation::valid(observation_schema) &&
               (controls == 160 || controls == 640 || controls == 1280 || (rest_hold && controls == 320)) && (!stopping || controls == 1280) &&
               (holding_reward_gain == 1.f || holding_reward_gain == 1.f / 1024) &&
               (!balanced_hold || (holding && appended_hold && (controls == 1280 || (rest_hold && controls == 320)))) &&
               (!rest_hold || (holding && balanced_hold && appended_hold && controls == 320 && !stopping && !startup_speed));
    }
    CUDARL_HD contact::Arena arena() const { return controls >= 640 || rest_hold ? long_swim_arena() : contact::Arena{}; }
    CUDARL_HD int reward_version() const { return appended_hold ? 9 : (holding ? 8 : (startup_speed ? 7 : (stopping ? 6 : kRewardVersion))); }
    CUDARL_HD ForwardConfig task_config(int completed_controls = 0, int environment = 0) const {
        if (rest_hold && hold_episode(environment)) {
            auto c = config_for_speed(.008f);c.maximum_controls = controls;
            c.observation_schema = observation_schema;c.target_speed_m_s = 0;
            c.stationary_request = true;c.position_hold = true;return c;
        }
        if (stopping || hold_episode(environment)) {
            auto c = StopSequence{}.config(completed_controls, observation_schema);
            c.position_hold = holding;
            return c;
        }
        auto c = config_for_speed(.008f);
        c.maximum_controls = controls;
        c.observation_schema = observation_schema;
        c.horizontal_from_first_step = startup_speed;
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
    float signed_depth_m = 0, vertical_velocity_m_s = 0, horizontal_speed_error_m_s = 0;
    float roll_deg = 0, pitch_deg = 0;
    p::V hold_error_m{}, velocity_m_s{};
};
struct ForwardTask {
    simulation::MotorRequest request{};
    p::V hold_anchor_model{};
    bool hold_anchor_set = false;
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
    // Details of speed and posture aggregates; do not add them a second time.
    float horizontal = 0, recovery = 0, roll = 0, pitch = 0;
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
    const float fields[] = {c.velocity_reward_scale_m_s,
                            c.clearance_m,
                            c.minimum_speed_m_s,
                            c.maximum_speed_m_s,
                            c.speed_rmse_limit,
                            c.lateral_limit_m,
                            c.depth_limit_m,
                            c.path_angle_limit_deg,
                            c.tilt_limit_deg,
                            c.work_per_distance_limit};
    for (float x : fields)
        if (!p::finite_number(x) || x <= 0)
            return false;
    const bool target_valid =
        supported_speed(c.target_speed_m_s) || (c.stationary_request && c.target_speed_m_s == 0);
    return target_valid && c.maximum_speed_m_s >= c.minimum_speed_m_s && c.clearance_m >= .01f &&
           c.measure_start > 0 && c.maximum_controls > c.measure_start;
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
// Initialization only: translate the whole creature, retaining the already chosen
// world-space target/lane. This never changes an ongoing episode or its deadline.
CUDARL_HD inline bool offset_episode_start(World &world, ForwardTask &task, const ForwardConfig &config,
                                           float vertical_offset_m) {
    if (!p::finite_number(vertical_offset_m) || task.controls != 0 || task.outcome != Outcome::Running ||
        world.physical.control_steps != 0 || world.attempted_substeps != 0 || !in_domain(world, config))
        return false;
    if (vertical_offset_m == 0)
        return true;
    auto shifted = world;
    shifted.physical.body.position.y += vertical_offset_m / p::scale;
    if (!in_domain(shifted, config))
        return false;
    // Speed displacement starts at the actual translated COM. Depth/lane and
    // reward still use start_com_model, which deliberately remains unchanged.
    const auto actual = old::motion(shifted.physical.body).com_model;
    task.steady_start_com_model = actual;
    task.maximum_depth = fabsf((actual.y - task.start_com_model.y) * p::scale);
    world = shifted;
    return true;
}
CUDARL_HD inline bool begin_training_episode(int environment, bool tilt_mixture, const ForwardConfig &config,
                                             World &world, ForwardTask &task, contact::Arena arena = {},
                                             float vertical_offset_m = 0) {
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
    return offset_episode_start(world, task, config, vertical_offset_m);
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
// Within-episode goal update. Unlike request_speed's segment reset, this keeps
// the fixed world target, heading, elapsed time, warmup and all audit state.
CUDARL_HD inline bool update_velocity_request(const World &w, const ForwardConfig &c, ForwardTask &t) {
    if (!in_domain(w, c) || t.outcome != Outcome::Running || w.physical.control_steps != t.last_world_control)
        return false;
    if (c.position_hold && c.stationary_request && !t.hold_anchor_set) {
        // One command target: actual horizontal COM at request, original depth.
        // Never teleport, recenter or change the original depth/lane reference.
        t.hold_anchor_model = old::motion(w.physical.body).com_model;
        t.hold_anchor_model.y = t.start_com_model.y;
        t.hold_anchor_set = true;
    }
    t.request.terminal_velocity = t.request.heading * c.target_speed_m_s;
    return true;
}
CUDARL_HD inline bool observe(const ForwardTask &t, const World &w, const ForwardConfig &c, float *out) {
    if (!out || !motor_observation::valid(c.observation_schema) || t.outcome != Outcome::Running ||
        !in_domain(w, c))
        return false;
    const auto motion = old::motion(w.physical.body);
    const auto &q = w.physical.body.orientation;
    const auto displacement = (motion.com_model - t.start_com_model) * p::scale;
    auto lane_error = t.request.heading * p::dot(displacement, t.request.heading) - displacement;
    if (c.position_hold && c.stationary_request) {
        if (c.observation_schema != motor_observation::position_hold || !t.hold_anchor_set)
            return false;
        lane_error = (t.hold_anchor_model - motion.com_model) * p::scale;
    }
    old::vector_features(out, 0,
                         q.inverse(lane_error) / motor_observation::path_scale_m(c.observation_schema));
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
// Schema4 adds exactly one scalar; the learned meanings of the first40 stay schema2.
CUDARL_HD inline bool observe_appended(const ForwardTask &t, const World &w,
                                      const ForwardConfig &config, float *out) {
    auto original = config;
    original.observation_schema = motor_observation::task_scale;
    original.position_hold = false;
    if (!observe(t, w, original, out))
        return false;
    out[40] = 0;
    if (config.position_hold && config.stationary_request) {
        if (!t.hold_anchor_set)
            return false;
        const auto error = (t.hold_anchor_model - old::motion(w.physical.body).com_model) * p::scale;
        out[40] = tanhf(p::dot(error, t.request.heading) / .02f);
    }
    return p::finite_number(out[40]);
}
CUDARL_HD inline Metrics measure(const ForwardTask &t, const World &w) {
    const auto motion = old::motion(w.physical.body);
    const auto displacement = (motion.com_model - t.start_com_model) * p::scale;
    Metrics result;
    result.velocity_m_s = motion.velocity_m_s;
    if (t.hold_anchor_set)
        result.hold_error_m = (motion.com_model - t.hold_anchor_model) * p::scale;
    result.forward_m = p::dot(displacement, t.request.heading);
    const auto perpendicular = displacement - t.request.heading * result.forward_m;
    result.lateral_m = p::norm({perpendicular.x, 0, perpendicular.z});
    result.depth_m = fabsf(displacement.y);
    result.signed_depth_m = displacement.y;
    result.vertical_velocity_m_s = motion.velocity_m_s.y;
    const auto velocity_error = motion.velocity_m_s - t.request.terminal_velocity;
    result.speed_error_m_s = p::norm(velocity_error); // Original acceptance, including vertical motion.
    result.horizontal_speed_error_m_s = p::norm({velocity_error.x, 0, velocity_error.z});
    const auto forward = w.physical.body.orientation.rotate({1, 0, 0});
    const auto up = w.physical.body.orientation.rotate({0, 1, 0});
    result.head_yaw_deg = fabsf(atan2f(p::dot(p::cross(t.request.heading, forward), {0, 1, 0}),
                                       p::dot(t.request.heading, forward))) *
                          180 / movement::pi;
    result.tilt_deg = acosf(p::clamp(p::dot(up, {0, 1, 0}), -1, 1)) * 180 / movement::pi;
    const auto local_up = w.physical.body.orientation.inverse({0, 1, 0});
    result.pitch_deg = atan2f(local_up.x, sqrtf(local_up.y * local_up.y + local_up.z * local_up.z)) * 180 /
                       movement::pi; // positive nose-up
    result.roll_deg = atan2f(-local_up.z, local_up.y) * 180 / movement::pi;
    return result;
}
CUDARL_HD inline float recovery_velocity(float signed_depth_m) {
    // Y up; below target -> positive reference, taper continuously to zero.
    return p::clamp(-signed_depth_m / kDepthSettlingSeconds, -kRecoverySpeedLimit, kRecoverySpeedLimit);
}
CUDARL_HD inline RewardCosts reward_costs(const Metrics &metrics, const World &world,
                                          const ForwardConfig &config, bool speed_scored) {
    const float horizontal = metrics.horizontal_speed_error_m_s / config.velocity_reward_scale_m_s;
    const float vertical = (metrics.vertical_velocity_m_s - recovery_velocity(metrics.signed_depth_m)) /
                           config.velocity_reward_scale_m_s;
    const float lane = metrics.lateral_m / config.lateral_limit_m;
    const float depth = metrics.depth_m / config.depth_limit_m;
    const float roll = metrics.roll_deg / config.tilt_limit_deg;
    const float pitch = metrics.pitch_deg / config.tilt_limit_deg;
    RewardCosts out;
    out.horizontal = (speed_scored || config.horizontal_from_first_step) ? horizontal * horizontal : 0.f;
    out.recovery = vertical * vertical; // From FIRST control, including startup.
    out.roll = kRollCost * roll * roll;
    out.pitch = kPitchCost * pitch * pitch;
    out.speed = out.horizontal + out.recovery;
    out.lane = .05f * lane * lane;
    out.depth = .05f * depth * depth;
    out.posture = out.roll + out.pitch;
    if (config.position_hold && config.stationary_request) {
        // Reward references only. The network still chooses all four joints.
        auto desired = p::V{-metrics.hold_error_m.x / 2.f, 0, -metrics.hold_error_m.z / 2.f};
        const float magnitude = p::norm(desired);
        if (magnitude > .002f)
            desired = desired * (.002f / magnitude);
        const auto error = p::V{metrics.velocity_m_s.x - desired.x, 0,
                                metrics.velocity_m_s.z - desired.z};
        out.horizontal = p::dot(error, error) / (.001f * .001f);
        const float vy_error = metrics.vertical_velocity_m_s - recovery_velocity(metrics.signed_depth_m);
        out.recovery = vy_error * vy_error / (.001f * .001f);
        out.speed = out.horizontal + out.recovery;
        const auto position = p::V{metrics.hold_error_m.x, 0, metrics.hold_error_m.z};
        out.lane += .05f * p::dot(position, position) / (.002f * .002f);
    }
    out.effort = .002f * world.physical.mean_squared_torque;
    return out;
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
    // Reward5 masks horizontal speed during startup; reward7 removes only that
    // mask. Acceptance still scores the original2..8s window. Recovery, lane, depth, head
    // posture and effort are charged from the first interval. Full trajectory
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
