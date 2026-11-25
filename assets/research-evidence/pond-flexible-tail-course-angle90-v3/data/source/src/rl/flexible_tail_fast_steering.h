#pragma once

#include "rl/flexible_tail_yaw_offset.h"
#include "rl/flexible_tail_beat_course.h"

#include <cfloat>

namespace cudarl::flexible_tail::fast {

inline constexpr int decision_controls = 4;
inline constexpr int turn_controls = 650;       // 13 seconds.
inline constexpr int episode_controls = 750;    // Two seconds of onward swimming.
inline constexpr float minimum_speed_m_s = .006f;
inline constexpr float envelope_tilt_deg = 10.f;
inline constexpr float envelope_depth_error_m = .001f;
inline constexpr float course_tolerance_rad = 5.f * pi / 180.f;
inline constexpr int completion_controls = 50; // One continuous second.
inline constexpr int critic_inputs = navigation_inputs + 5;

enum class FeedbackMode : uint32_t { HalfSecond = 0, WholeBeat = 1, WholeBeat90 = 2 };

CUDARL_HD inline bool is_whole_beat_feedback(FeedbackMode mode) {
    return mode == FeedbackMode::WholeBeat || mode == FeedbackMode::WholeBeat90;
}

inline constexpr float course_error90_scale_rad = pi / 2.f;

// Signed wrapped course error is already in [-pi, pi]. Scaling by 90 degrees
// gives [-1,1] through 90 degrees and retains magnitude information beyond it.
// This is an observation transformation, not an actuator or physical limit.
CUDARL_HD inline float course_error_input(float error_rad, FeedbackMode mode) {
    return mode == FeedbackMode::WholeBeat90 ? error_rad / course_error90_scale_rad
        : physics::clamp(error_rad / yaw_offset_error_scale, -4.f, 4.f);
}

enum Fault {
    Healthy = 0, NativePhysics = 1, Clearance = 2, Nonfinite = 3,
    PoseEnvelope = 5, InvalidUse = 6
};

CUDARL_HD inline bool is_global_fault(int fault) {
    return fault == NativePhysics || fault == Nonfinite || fault == InvalidUse;
}

struct World {
    FeedbackMode feedback_mode = FeedbackMode::HalfSecond;
    NavigationWorld navigation{};
    BeatCourseFeedback beat_feedback{};
    int feedback_callback_control = -1, feedback_callback_index = 0;
    Snapshot pose{};
    float applied_offset = 0.f;
    float targets[motor_outputs]{};
    float requested_angle_rad = 0.f;
    int horizon_controls = episode_controls;
    int fault = Healthy;
    bool done = false;
    bool truncated = false;
    uint64_t attempted_substeps = 0, accepted_substeps = 0;
    float episode_reward = 0.f;
    float min_gap_m = FLT_MAX;
    float peak_tilt_deg = 0.f, peak_depth_error_m = 0.f;
    float course_error_rad = 0.f, rolling_speed_m_s = 0.f;
    float projected_speed_m_s = 0.f;
    int aligned_controls = 0;
    int completion_control = -1;
    uint64_t reliable_course_controls = 0, low_course_controls = 0;
};

CUDARL_HD inline void set_feedback_mode(World& world, FeedbackMode mode) {
    if (mode != FeedbackMode::HalfSecond && !is_whole_beat_feedback(mode)) {
        world.fault = InvalidUse;
        world.done = true;
        return;
    }
    world.feedback_mode = mode;
    world.beat_feedback.initialize(world.navigation.motion);
    world.feedback_callback_control = -1;
    world.feedback_callback_index = 0;
}

CUDARL_HD inline void record_feedback_substep(World& world, const Snapshot& pose) {
    const auto& motion = world.navigation.motion;
    if (world.feedback_callback_control != motion.controls) {
        world.feedback_callback_control = motion.controls;
        world.feedback_callback_index = 0;
    } else {
        ++world.feedback_callback_index;
    }
    world.beat_feedback.add_substep(motion.body.velocity[6],
        motion.controls * turn_substeps_per_control + world.feedback_callback_index, pose.com_m);
}

CUDARL_HD inline void record_feedback_control(World& world) {
    world.beat_feedback.add_control(snapshot(world.navigation.motion.body).com_m);
}

CUDARL_HD inline BeatCourseSample whole_beat_sample(const World& world, int back_controls = 0) {
    return world.beat_feedback.sample(back_controls);
}

CUDARL_HD inline BeatCourseSample feedback_sample(const World& world, int back_controls = 0) {
    if (is_whole_beat_feedback(world.feedback_mode)) return whole_beat_sample(world, back_controls);
    BeatCourseSample sample{};
    sample.period_s = .5f;
    const auto& motion = world.navigation.motion;
    if (back_controls == 0) {
        sample.velocity_m_s = turn_course(motion, snapshot(motion.body).com_m, steering_history_controls) / .5f;
    } else {
        physics::V begin{}, end{};
        if (!world.beat_feedback.position(static_cast<float>(back_controls), end) ||
            !world.beat_feedback.position(static_cast<float>(back_controls + steering_history_controls), begin)) return sample;
        sample.velocity_m_s = (end - begin) / .5f;
        sample.velocity_m_s.y = 0;
    }
    sample.fresh = true;
    sample.reliable = physics::norm(sample.velocity_m_s) >= .003f;
    return sample;
}

struct Result {
    float reward = 0.f;
    int controls = 0, fault = Healthy;
    int attempted_substeps = 0, accepted_substeps = 0;
    bool done = false, truncated = false;
};

// Match the archived actor's three reachable phase paths exactly. The other
// hidden rows are disconnected from the readout, and pitch is identically zero.
inline constexpr int gait_hidden = 32;
inline constexpr int gait_layer1_bias = navigation_inputs * gait_hidden;
inline constexpr int gait_layer2_weights = gait_layer1_bias + gait_hidden;
inline constexpr int gait_layer2_bias = gait_layer2_weights + gait_hidden * gait_hidden;
inline constexpr int gait_readout = gait_layer2_bias + gait_hidden;
inline constexpr int gait_output_bias = gait_readout + gait_hidden * motor_outputs;
inline constexpr int phase_columns = 3;
inline constexpr int phase_row = gait_hidden - phase_columns;

inline void validate_frozen_gait(const std::vector<float>& gait) {
    validate_navigation_weights(gait);
    const auto spec = navigation_motor_spec();
    if (spec.in(0) != navigation_inputs || spec.out(0) != gait_hidden ||
        spec.in(1) != gait_hidden || spec.out(1) != gait_hidden ||
        spec.out(2) != motor_outputs || spec.bias_offset(2) != gait_output_bias) {
        throw std::runtime_error("Unsupported frozen fast-steering gait layout");
    }
    for (int axis = 0; axis < phase_columns; ++axis) {
        const int row = phase_row + axis;
        for (int i = 0; i < navigation_inputs; ++i) {
            const float expected = i == motor_inputs + axis ? gait_seed_hidden_gain : 0.f;
            if (gait[row * navigation_inputs + i] != expected)
                throw std::runtime_error("Frozen gait phase inputs changed");
        }
        for (int i = 0; i < gait_hidden; ++i) {
            const float expected = i == row ? 1.f : 0.f;
            if (gait[gait_layer2_weights + row * gait_hidden + i] != expected)
                throw std::runtime_error("Frozen gait phase activations changed");
        }
        if (gait[gait_layer1_bias + row] != 0.f || gait[gait_layer2_bias + row] != 0.f)
            throw std::runtime_error("Frozen gait phase intercept changed");
    }
    for (int i = 0; i < gait_hidden * motor_outputs + motor_outputs; ++i) {
        if ((i < phase_row || i >= gait_hidden) && gait[gait_readout + i] != 0.f)
            throw std::runtime_error("Unsupported frozen gait readout");
    }
}

inline void validate_actor(const std::vector<float>& actor) {
    validate_yaw_offset_weights(actor);
    for (int i = 4; i < navigation_inputs; ++i) {
        if (actor[i] != 0.f) throw std::runtime_error("Unused steering input changed");
    }
}

inline std::vector<float> initial_actor90(const std::vector<float>& source) {
    validate_actor(source);
    auto candidate = source;
    // The previous feature was error/5 degrees inside its 20 degree saturation.
    // Transfer its coefficient so small-angle behavior is preserved while
    // 30/45/60/90 degree errors become distinguishable to the same readout.
    candidate[0] *= 18.f;
    validate_actor(candidate);
    return candidate;
}

CUDARL_HD inline void frozen_gait_latent(const World& world, const float* weights,
                                         float (&latent)[motor_outputs]) {
    const auto& body = world.navigation.motion.body;
    const float yaw = body.angle[0];
    const float travel = body.velocity[6] / gait_nominal_rate;
    const float radius = sqrtf(yaw * yaw + travel * travel);
    const float scale = fmaxf(radius, gait_phase_floor_rad);
    const float phase[phase_columns]{yaw / scale, travel / scale,
        fmaxf(0.f, 1.f - radius / gait_phase_floor_rad)};
    latent[0] = latent[1] = 0.f;
    for (int i = 0; i < phase_columns; ++i) {
        const float hidden = nn::activate(nn::Activation::Tanh,
            nn::activate(nn::Activation::Tanh, gait_seed_hidden_gain * phase[i]));
        latent[0] += hidden * weights[gait_readout + phase_row + i];
    }
}

CUDARL_HD inline void update_motion_metrics(World& world) {
    if (is_whole_beat_feedback(world.feedback_mode)) {
        const auto sample = feedback_sample(world);
        world.rolling_speed_m_s = physics::norm(sample.velocity_m_s);
        world.projected_speed_m_s = physics::dot(sample.velocity_m_s, world.navigation.motion.request.heading);
        world.course_error_rad = sample.reliable
            ? angle_error(sample.velocity_m_s, world.navigation.motion.request.heading) : pi;
        return;
    }
    const auto& motion = world.navigation.motion;
    const auto course = turn_course(motion, world.pose.com_m, steering_history_controls);
    world.rolling_speed_m_s = physics::norm(course) / .5f;
    world.projected_speed_m_s = physics::dot(course, motion.request.heading) / .5f;
    world.course_error_rad = physics::norm(course) >= .0015f
        ? angle_error(course, motion.request.heading) : pi;
}

CUDARL_HD inline void initialize(World& world, const NavigationWorld& source,
                                  float angle_rad, FeedbackMode feedback_mode = FeedbackMode::HalfSecond) {
    world = {};
    if (source.motion.controls != 0 || source.leg_count < 1 ||
        !physics::finite_body(source.motion.body) || !physics::finite_number(angle_rad) ||
        fabsf(angle_rad) > pi / 2.f || fabsf(angle_rad) < .001f ||
        !physics::finite_vector(source.motion.entry_course) ||
        physics::norm(source.motion.entry_course) < .5f ||
        fabsf(source.motion.entry_course.y) > .001f ||
        !physics::finite_vector(source.legs[0].goal_m)) {
        world.fault = InvalidUse;
        world.done = true;
        return;
    }
    // These sources already carry the measured two-whole-beat incoming course.
    // Preserve body motion, servo state, history and previous actuator targets.
    world.navigation = source;
    auto& motion = world.navigation.motion;
    world.pose = snapshot(motion.body);
    world.requested_angle_rad = angle_rad;
    auto incoming = motion.entry_course;
    incoming.y = 0.f;
    incoming = incoming / physics::norm(incoming);
    const auto target = physics::axis_rotation({0.f, 1.f, 0.f}, angle_rad).rotate(incoming);
    auto origin = source.legs[0].goal_m;
    origin.x = world.pose.com_m.x;
    origin.z = world.pose.com_m.z;
    world.navigation.legs[0] = {origin, target, turn_controls,
        angle_rad > 0.f ? Choice::TurnLeft : Choice::TurnRight};
    world.navigation.legs[1] = {origin, target, episode_controls, Choice::Swim};
    world.navigation.leg_count = 2;
    world.navigation.active_leg = 0;
    motion.entry_course = incoming;
    motion.turn_radians = angle_rad;
    motion.turn_issued = true;
    motion.turn_controls = turn_controls;
    for (int i = 0; i < motor_outputs; ++i) world.targets[i] = motion.last_targets[i];
    prepare_navigation(world.navigation);
    set_feedback_mode(world, feedback_mode);
    update_motion_metrics(world);
}

CUDARL_HD inline void observe(const World& world, float (&out)[navigation_inputs]) {
    observe_yaw_offset(world.navigation, world.applied_offset, out);
    if (is_whole_beat_feedback(world.feedback_mode) && yaw_offset_enabled(world.navigation)) {
        const auto sample = feedback_sample(world);
        out[0] = sample.reliable ? course_error_input(yaw_offset_error(sample.velocity_m_s,
            world.navigation.motion.request.heading), world.feedback_mode) : 0.f;
    }
}

CUDARL_HD inline void observe_critic(const World& world, float (&out)[critic_inputs]) {
    float gait[navigation_inputs], steering[navigation_inputs];
    observe_gait(world.navigation, gait);
    observe(world, steering);
    for (int i = 0; i < navigation_inputs; ++i) out[i] = gait[i];
    for (int i = 0; i < 4; ++i) out[navigation_inputs + i] = steering[i];
    out[navigation_inputs + 4] = physics::clamp(
        static_cast<float>(world.horizon_controls - world.navigation.motion.controls) /
            static_cast<float>(world.horizon_controls), 0.f, 1.f);
}

// A sampled latent is held for one 80 ms decision. There is no stored offset
// policy inside this function: the PPO sample controls the actual actuator trim.
template <class ObserveSubstep = IgnoreTurnSubstep>
CUDARL_HD CUDARL_DEVICE_NOINLINE Result advance(World& world, float sampled_latent,
    const float* frozen_gait, const pond_geometry_v1::Parameters& pond,
    int max_controls = decision_controls, ObserveSubstep observe_substep = {}) {
    Result result{};
    if (world.done || world.fault != Healthy) {
        result.fault = world.fault;
        result.done = world.done;
        result.truncated = world.truncated;
        return result;
    }
    if (max_controls < 1 || max_controls > decision_controls ||
        world.horizon_controls < 1 || !physics::finite_number(sampled_latent)) {
        world.fault = !physics::finite_number(sampled_latent) ? Nonfinite : InvalidUse;
        world.done = true;
        result.fault = world.fault;
        result.done = true;
        return result;
    }
    prepare_navigation(world.navigation);
    float latent[motor_outputs];
    frozen_gait_latent(world, frozen_gait, latent);
    if (!physics::finite_number(latent[0]) || !physics::finite_number(latent[1])) {
        world.fault = Nonfinite;
        world.done = true;
        result.fault = Nonfinite;
        result.done = true;
        return result;
    }
    decode_yaw_offset(world.navigation, latent, sampled_latent,
        world.applied_offset, world.targets);
    for (int i = 0; i < max_controls && !world.done; ++i) {
        prepare_navigation(world.navigation);
        const auto& motion = world.navigation.motion;
        const auto before = turn_course(motion, world.pose.com_m, steering_history_controls);
        const float before_error = angle_error(before, motion.request.heading);
        const auto step = step_turn_control(world.navigation.motion, world.targets, pond,
            [&](const Snapshot& pose) {
                record_feedback_substep(world, pose);
                observe_substep(pose);
            });
        result.attempted_substeps += step.attempted_substeps;
        result.accepted_substeps += step.accepted_substeps;
        world.attempted_substeps += step.attempted_substeps;
        world.accepted_substeps += step.accepted_substeps;
        world.min_gap_m = fminf(world.min_gap_m, step.min_gap_m);
        world.peak_tilt_deg = fmaxf(world.peak_tilt_deg, step.max_tilt_deg);
        world.peak_depth_error_m = fmaxf(world.peak_depth_error_m, step.max_depth_error_m);
        if (step.fault != Healthy) {
            world.fault = step.fault;
            world.done = true;
            // Guard failures are preserved; no invalid control earns reward.
            result.reward -= is_global_fault(step.fault) ? 0.f : 2.f;
            break;
        }
        ++result.controls;
        world.pose = snapshot(world.navigation.motion.body);
        record_feedback_control(world);
        update_motion_metrics(world);
        const float tilt = step.max_tilt_deg / envelope_tilt_deg;
        const float depth = step.max_depth_error_m / envelope_depth_error_m;
        const bool reliable_course = is_whole_beat_feedback(world.feedback_mode)
            ? feedback_sample(world).reliable : world.rolling_speed_m_s >= .003f;
        if (reliable_course) ++world.reliable_course_controls;
        else ++world.low_course_controls;
        const float movement_quality = physics::clamp(
            world.rolling_speed_m_s / minimum_speed_m_s, 0.f, 1.f);
        const bool envelope = tilt <= 1.f && depth <= 1.f;
        const float error_scale = 15.f * pi / 180.f;
        const float error = world.course_error_rad / error_scale;
        const float alignment = expf(-error * error);
        const float transport = physics::clamp(world.projected_speed_m_s / requested_speed_m_s, -1.f, 1.5f);
        float correction = physics::norm(before) >= .0015f && reliable_course
            ? physics::clamp((before_error - world.course_error_rad) /
                (motor_control_s * error_scale), -1.f, 1.f) : 0.f;
        if (is_whole_beat_feedback(world.feedback_mode)) {
            // Compare both ends of this accepted control with the same newly
            // measured period. A changed period cannot manufacture progress.
            const auto previous = feedback_sample(world, 1);
            correction = previous.reliable && reliable_course
                ? physics::clamp((angle_error(previous.velocity_m_s, motion.request.heading)
                    - world.course_error_rad) / (motor_control_s * error_scale), -1.f, 1.f) : 0.f;
        }
        const float effort = world.applied_offset / yaw_offset_limit;
        // Per simulated second: transport .25, alignment 1.5, signed course
        // improvement .75, tilt 1, depth .75, offset effort .01. Positive
        // alignment credit is graded by actual COM travel. Slow recovery still
        // receives transport credit instead of encountering a hard speed gate.
        float reward = motor_control_s *
            ((envelope ? .25f * transport + 1.5f * movement_quality * alignment : 0.f)
                + .75f * correction - tilt * tilt - .75f * depth * depth
                - .01f * effort * effort);
        if (!physics::finite_number(reward) || !physics::finite_number(world.course_error_rad)) {
            world.fault = Nonfinite;
            world.done = true;
            break;
        }
        if (!envelope) {
            world.fault = PoseEnvelope;
            world.done = true;
            reward -= 2.f;
        }
        result.reward += reward;
        if (envelope && world.projected_speed_m_s >= minimum_speed_m_s &&
            world.course_error_rad <= course_tolerance_rad) {
            ++world.aligned_controls;
            if (world.aligned_controls >= completion_controls && world.completion_control < 0)
                world.completion_control = world.navigation.motion.controls;
        } else {
            world.aligned_controls = 0;
        }
        if (!world.done && world.navigation.motion.controls >= world.horizon_controls) {
            world.done = true;
            world.truncated = true;
        }
    }
    world.episode_reward += result.reward;
    result.fault = world.fault;
    result.done = world.done;
    result.truncated = world.truncated;
    return result;
}

} // namespace cudarl::flexible_tail::fast
