#pragma once

#include "rl/flexible_tail_yaw_offset.h"

#include <cfloat>
#include <climits>

namespace cudarl::flexible_tail::chooser {

inline constexpr int inputs = 32;
inline constexpr int actions = 3;
inline constexpr int decision_controls = 200;
inline constexpr int cue_limit_controls = 300;
inline constexpr float heading_step_rad = pi / 12.f;
inline constexpr float cell_size_m = .05f;
inline constexpr int cells_x = 48;
inline constexpr int cells_z = 32;
inline constexpr int recent_cells = 32;
inline constexpr float minimum_speed_m_s = .006f;
inline constexpr float envelope_tilt_deg = 10.f;
inline constexpr float envelope_depth_error_m = .001f;

enum Fault {
    Healthy = 0, NativePhysics = 1, Clearance = 2, Nonfinite = 3,
    CueUnavailable = 4, PoseEnvelope = 5, InvalidUse = 6
};

struct Crossing {
    physics::V com{};
    double sample = 0;
};

struct BeatCourse {
    Crossing positive[3]{}, candidate{};
    physics::V previous_com{};
    float previous_rate = 0;
    int previous_sample = -1;
    int direction = 0;
    bool pending = false;

    CUDARL_HD void add(float rate, int sample, physics::V com) {
        if (sample <= previous_sample) return;
        if (previous_sample >= 0 && previous_rate <= 0.f && rate > 0.f) {
            const float fraction = -previous_rate / (rate - previous_rate);
            candidate = {previous_com + (com - previous_com) * fraction,
                static_cast<double>(previous_sample) + fraction * static_cast<double>(sample - previous_sample)};
            pending = true;
        }
        constexpr float deadband_rad_s = .02f;
        const int next = rate > deadband_rad_s ? 1 : rate < -deadband_rad_s ? -1 : 0;
        if (next != 0) {
            if (direction < 0 && next > 0 && pending) {
                positive[0] = positive[1];
                positive[1] = positive[2];
                positive[2] = candidate;
            }
            if (direction != next) pending = false;
            direction = next;
        }
        previous_rate = rate;
        previous_sample = sample;
        previous_com = com;
    }

    CUDARL_HD physics::V course(int since_control) const {
        const double first = positive[0].sample;
        const double last = positive[2].sample;
        if (!(first > 0.) || !(positive[1].sample > first) || !(last > positive[1].sample) ||
            first < static_cast<double>(since_control * turn_substeps_per_control)) return {};
        auto travel = positive[2].com - positive[0].com;
        travel.y = 0;
        const float length = physics::norm(travel);
        if (length < minimum_speed_m_s * (last - first) * motor_substep_s) return {};
        return travel / length;
    }
};

struct RecentVisit {
    int cell = -1;
    int control = -1;
};

struct World {
    NavigationWorld navigation{};
    Snapshot pose{};
    BeatCourse beats{};
    float applied_offset = 0.f;
    float targets[motor_outputs]{};
    uint64_t visited[cells_x * cells_z / 64]{};
    RecentVisit recent[recent_cells]{};
    physics::V entry_position{};
    int entry_cell = -1, entry_control = 0;
    int recent_index = 0;
    bool entry_credited = false;
    int decision_count = 0, decision_control = 0, last_action = 0;
    int turns = 0, visits = 0;
    int fault = Healthy;
    uint64_t attempted_substeps = 0, accepted_substeps = 0;
    float gap_m = 0, min_gap_m = FLT_MAX, min_speed_m_s = FLT_MAX;
    float peak_tilt_deg = 0, peak_depth_error_m = 0;
    float stall_s = 0, max_stall_s = 0, transport_m = 0;
};

struct Result {
    float reward = 0;
    int controls = 0, fault = Healthy;
    int attempted_substeps = 0, accepted_substeps = 0;
};

CUDARL_HD inline physics::V horizontal(physics::V value) {
    value.y = 0;
    return value;
}

CUDARL_HD inline int cell_index(physics::V position) {
    const int x = static_cast<int>(floorf(position.x / cell_size_m)) + cells_x / 2;
    const int z = static_cast<int>(floorf(position.z / cell_size_m)) + cells_z / 2;
    return x >= 0 && x < cells_x && z >= 0 && z < cells_z ? z * cells_x + x : -1;
}

CUDARL_HD inline bool seen(const World& world, int cell) {
    return cell >= 0 && (world.visited[cell / 64] & (uint64_t{1} << (cell % 64))) != 0;
}

CUDARL_HD inline float revisit_fraction(const World& world, int cell, int control) {
    // The most recent entries are searched first. Long absences beyond the
    // bounded history earn a revisit value, never another first-visit credit.
    for (int i = 0; i < recent_cells; ++i) {
        const auto& visit = world.recent[(world.recent_index + recent_cells - 1 - i) % recent_cells];
        if (visit.cell == cell) {
            const float seconds = (control - visit.control) * motor_control_s;
            return physics::clamp((seconds - 90.f) / 210.f, 0.f, 1.f);
        }
    }
    // Evicted cells were last credited on or before the oldest retained
    // entry. Credit only an absence supported by that conservative bound.
    const int oldest_control = world.recent[world.recent_index].control;
    if (oldest_control < 0) return 0.f;
    const float seconds = (control - oldest_control) * motor_control_s;
    return physics::clamp((seconds - 90.f) / 210.f, 0.f, 1.f);
}

CUDARL_HD inline float novelty(const World& world, physics::V position) {
    const int cell = cell_index(position);
    if (cell < 0) return 0.f;
    return seen(world, cell)
        ? .25f * revisit_fraction(world, cell, world.navigation.motion.controls) : 1.f;
}

CUDARL_HD inline float visit_reward(World& world, float speed_m_s) {
    const int control = world.navigation.motion.controls;
    const int cell = cell_index(world.pose.com_m);
    if (cell < 0 || world.entry_cell != cell) {
        world.entry_cell = cell;
        world.entry_control = control;
        world.entry_position = world.pose.com_m;
        world.entry_credited = false;
        return 0.f;
    }
    if (world.entry_credited || (control - world.entry_control) * motor_control_s < 1.f ||
        speed_m_s < minimum_speed_m_s ||
        physics::norm(horizontal(world.pose.com_m - world.entry_position)) < .006f) return 0.f;
    const bool known = seen(world, cell);
    const float reward = known ? .15f * revisit_fraction(world, cell, control) : .6f;
    if (!known) {
        world.visited[cell / 64] |= uint64_t{1} << (cell % 64);
        ++world.visits;
    }
    world.recent[world.recent_index] = {cell, control};
    world.recent_index = (world.recent_index + 1) % recent_cells;
    world.entry_credited = true;
    return reward;
}

CUDARL_HD inline void initialize(World& world, const NavigationWorld& source,
                                 physics::V com_horizontal_m, float yaw_rad) {
    world = {};
    auto& motion = world.navigation.motion;
    motion = source.motion;
    if (motion.controls != 0 || source.leg_count < 1 || !physics::finite_body(motion.body) ||
        !physics::finite_vector(com_horizontal_m) || !physics::finite_number(yaw_rad)) {
        world.fault = InvalidUse;
        return;
    }
    const auto prior_pose = snapshot(motion.body);
    const auto origin = horizontal(prior_pose.com_m);
    const auto destination = horizontal(com_horizontal_m);
    const auto& first = source.legs[0];
    auto heading = yaw_offset_turn(first.choice) ? source.motion.entry_course : first.heading;
    auto goal = first.goal_m;
    const bool transformed = yaw_rad != 0.f || origin.x != destination.x || origin.z != destination.z;
    if (transformed) {
        const auto rotation = physics::axis_rotation({0, 1, 0}, yaw_rad);
        const auto position = [&](physics::V p) { return destination + rotation.rotate(p - origin); };
        motion.body.position = position(motion.body.position * physics::scale) / physics::scale;
        motion.body.orientation = rotation * motion.body.orientation;
        motion.body.orientation.normalize();
        for (int i = 0; i < 6; i += 3) {
            const auto v = rotation.rotate({motion.body.velocity[i], motion.body.velocity[i + 1],
                motion.body.velocity[i + 2]});
            motion.body.velocity[i] = v.x;
            motion.body.velocity[i + 1] = v.y;
            motion.body.velocity[i + 2] = v.z;
        }
        for (auto& sample : motion.history) sample = position(sample);
        heading = rotation.rotate(heading);
        goal = position(goal);
    }
    motion.entry_course = heading;
    world.navigation.legs[0] = {goal, heading, INT_MAX, Choice::Swim};
    world.navigation.leg_count = 1;
    for (int i = 0; i < motor_outputs; ++i) world.targets[i] = motion.last_targets[i];
    prepare_navigation(world.navigation);
    world.pose = transformed ? snapshot(motion.body) : prior_pose;
    world.beats.add(motion.body.velocity[6], 0, world.pose.com_m);
    world.entry_cell = cell_index(world.pose.com_m);
    world.entry_position = world.pose.com_m;
}

CUDARL_HD inline physics::V decision_course(const World& world) {
    return world.beats.course(world.decision_control);
}

CUDARL_HD inline bool ready(const World& world) {
    const int now = world.navigation.motion.controls;
    return world.fault == Healthy && now % launch_action_controls == 0 &&
        now - world.decision_control >= decision_controls &&
        physics::norm(decision_course(world)) > .5f;
}

CUDARL_HD inline physics::V action_direction(physics::V course, int action) {
    if (action == 0) return course;
    const float angle = action == 1 ? heading_step_rad : -heading_step_rad;
    return physics::axis_rotation({0, 1, 0}, angle).rotate(course);
}

CUDARL_HD inline bool accept(World& world, int action) {
    if (action < 0 || action >= actions || !ready(world)) return false;
    const auto course = decision_course(world);
    auto& navigation = world.navigation;
    const auto current = navigation.legs[navigation.active_leg];
    world.decision_control = navigation.motion.controls;
    ++world.decision_count;
    world.last_action = action;
    // Rollout can stop exactly on the end of a turn control. Synchronize its
    // held Swim request before the next decision to preserve the offset.
    prepare_navigation(navigation);
    if (action != 0) {
        const auto heading = action_direction(course, action);
        const auto choice = action == 1 ? Choice::TurnLeft : Choice::TurnRight;
        const physics::V goal{world.pose.com_m.x, current.goal_m.y, world.pose.com_m.z};
        navigation.legs[0] = {goal, heading, world.decision_control + decision_controls, choice};
        navigation.legs[1] = {goal, heading, INT_MAX, Choice::Swim};
        navigation.leg_count = 2;
        navigation.active_leg = 0;
        prepare_navigation(navigation);
        ++world.turns;
    }
    return true;
}

CUDARL_HD inline float lookahead_gap(const World& world, const pond_geometry_v1::Parameters& pond,
                                      physics::V displacement) {
    auto body = world.navigation.motion.body;
    auto pose = world.pose;
    body.position += displacement / physics::scale;
    for (auto& part : pose.spine_m) part += displacement;
    const float gap = static_cast<float>(submerged_gap_m(pond, body, pose));
    return physics::clamp(gap / .05f, -1.f, 1.f);
}

CUDARL_HD inline void observe(const World& world, const pond_geometry_v1::Parameters& pond,
                              float (&features)[inputs]) {
    for (auto& feature : features) feature = 0;
    if (world.fault != Healthy) return;
    const auto& motion = world.navigation.motion;
    const auto& pose = world.pose;
    const auto course = decision_course(world);
    if (physics::norm(course) < .5f) return;
    constexpr float ranges[] = {.04f, .085f, .13f};
    for (int action = 0; action < actions; ++action) {
        const auto direction = action_direction(course, action);
        for (int range = 0; range < 3; ++range) {
            features[3 * action + range] = lookahead_gap(world, pond, direction * ranges[range]);
        }
        features[21 + action] = novelty(world, pose.com_m + direction * .075f);
        features[24 + action] = novelty(world, pose.com_m + direction * .13f);
        features[27 + action] = world.last_action == action ? 1.f : 0.f;
    }
    for (int side = 0; side < 2; ++side) {
        const float angle = side == 0 ? pi / 4.f : -pi / 4.f;
        const auto ray = physics::axis_rotation({0, 1, 0}, angle).rotate(course);
        features[9 + 2 * side] = lookahead_gap(world, pond, ray * .055f);
        features[10 + 2 * side] = lookahead_gap(world, pond, ray * .115f);
    }
    features[13] = physics::clamp(world.gap_m / .05f, 0.f, 1.f);
    features[14] = pose.com_m.x / static_cast<float>(pond.water_radius_x_m);
    features[15] = pose.com_m.z / static_cast<float>(pond.water_radius_z_m);
    features[16] = pose.depth_m / .08f;
    features[17] = course.x;
    features[18] = course.z;
    features[19] = physics::norm(turn_course(motion, pose.com_m, steering_history_controls)) /
        (launch_course_window_s * requested_speed_m_s);
    features[20] = physics::clamp(yaw_offset_error(course, motion.request.heading) / heading_step_rad, -2.f, 2.f);
    features[30] = sqrtf(features[14] * features[14] + features[15] * features[15]);
    features[31] = novelty(world, pose.com_m);
}

inline constexpr int gait_hidden = 32;
inline constexpr int gait_layer1_bias = navigation_inputs * gait_hidden;
inline constexpr int gait_layer2_weights = gait_layer1_bias + gait_hidden;
inline constexpr int gait_layer2_bias = gait_layer2_weights + gait_hidden * gait_hidden;
inline constexpr int gait_readout = gait_layer2_bias + gait_hidden;
inline constexpr int gait_output_bias = gait_readout + gait_hidden * motor_outputs;
inline constexpr int phase_columns = 3;
inline constexpr int phase_row = gait_hidden - phase_columns;

// The archived frozen actor's reachable graph contains three isolated phase
// paths, a zero pitch output, and a zero output intercept. Reject any other
// layout before entering the physical evaluator rather than ignoring weights.
inline void validate_frozen_layout(const std::vector<float>& gait, const std::vector<float>& offset) {
    validate_navigation_weights(gait);
    validate_yaw_offset_weights(offset);
    const auto spec = navigation_motor_spec();
    if (spec.in(0) != navigation_inputs || spec.out(0) != gait_hidden ||
        spec.in(1) != gait_hidden || spec.out(1) != gait_hidden ||
        spec.out(2) != motor_outputs || spec.bias_offset(2) != gait_output_bias) {
        throw std::runtime_error("Frozen chooser motor has an unsupported layer layout");
    }
    for (int axis = 0; axis < phase_columns; ++axis) {
        const int row = phase_row + axis;
        for (int i = 0; i < navigation_inputs; ++i) {
            const float expected = i == motor_inputs + axis ? gait_seed_hidden_gain : 0.f;
            if (gait[row * navigation_inputs + i] != expected) {
                throw std::runtime_error("Frozen chooser gait no longer has isolated phase inputs");
            }
        }
        for (int i = 0; i < gait_hidden; ++i) {
            const float expected = i == row ? 1.f : 0.f;
            if (gait[gait_layer2_weights + row * gait_hidden + i] != expected) {
                throw std::runtime_error("Frozen chooser gait no longer has isolated phase activations");
            }
        }
        if (gait[gait_layer1_bias + row] != 0.f || gait[gait_layer2_bias + row] != 0.f) {
            throw std::runtime_error("Unexpected frozen phase intercept");
        }
    }
    for (int i = 0; i < gait_hidden * motor_outputs + motor_outputs; ++i) {
        if ((i < phase_row || i >= gait_hidden) && gait[gait_readout + i] != 0.f) {
            throw std::runtime_error("Unexpected frozen chooser gait readout");
        }
    }
    for (int i = 4; i < navigation_inputs; ++i) {
        if (offset[i] != 0.f) throw std::runtime_error("Unexpected frozen chooser yaw offset input");
    }
}

CUDARL_HD inline bool frozen_targets(World& world, const float* gait_weights, const float* yaw_weights) {
    const auto& motion = world.navigation.motion;
    const auto& body = motion.body;
    const float yaw = body.angle[0];
    const float travel = body.velocity[6] / gait_nominal_rate;
    const float radius = sqrtf(yaw * yaw + travel * travel);
    const float scale = fmaxf(radius, gait_phase_floor_rad);
    const float phase[3]{yaw / scale, travel / scale, fmaxf(0.f, 1.f - radius / gait_phase_floor_rad)};
    float gait[motor_outputs]{};
    for (int i = 0; i < phase_columns; ++i) {
        const float hidden = nn::activate(nn::Activation::Tanh,
            nn::activate(nn::Activation::Tanh, gait_seed_hidden_gain * phase[i]));
        gait[0] += hidden * gait_weights[gait_readout + phase_row + i];
    }
    float offset = 0;
    if (yaw_offset_enabled(world.navigation)) {
        const auto heading = motion.request.heading;
        const auto course = turn_course(motion, world.pose.com_m, steering_history_controls);
        const auto facing = body.orientation.rotate({1.f, 0.f, 0.f});
        const auto measured = physics::norm(course) >= .0015f ? course : facing;
        const float features[]{
            physics::clamp(yaw_offset_error(measured, heading) / yaw_offset_error_scale, -4.f, 4.f),
            physics::clamp(yaw_offset_error(facing, heading) / yaw_offset_error_scale, -4.f, 4.f),
            world.applied_offset / yaw_offset_limit,
            physics::clamp(body.velocity[4] / .5f, -3.f, 3.f)
        };
        for (int i = 0; i < 4; ++i) offset += features[i] * yaw_weights[i];
    }
    if (!physics::finite_number(gait[0]) || !physics::finite_number(offset)) return false;
    decode_yaw_offset(world.navigation, gait, offset, world.applied_offset, world.targets);
    return true;
}

struct TrackControl {
    World& world;
    int sample;

    CUDARL_HD void operator()(const Snapshot& pose) {
        world.pose = pose;
        world.beats.add(world.navigation.motion.body.velocity[6], sample++, pose.com_m);
    }
};

CUDARL_HD inline float clearance_risk(float gap_m) {
    const float risk = physics::clamp(1.f - gap_m / .025f, 0.f, 1.f);
    return risk * risk;
}

CUDARL_HD CUDARL_DEVICE_NOINLINE Result advance(World& world, const float* gait_weights,
                                                const float* yaw_weights,
                                                const pond_geometry_v1::Parameters& pond,
                                                int max_controls = cue_limit_controls) {
    Result result{};
    if (world.fault != Healthy) {
        result.fault = world.fault;
        return result;
    }
    auto& motion = world.navigation.motion;
    if (motion.controls > INT_MAX / turn_substeps_per_control - cue_limit_controls) world.fault = InvalidUse;
    if (world.fault == Healthy && motion.controls == 0) {
        world.gap_m = static_cast<float>(submerged_gap_m(pond, motion.body, world.pose));
        world.min_gap_m = world.gap_m;
        if (!(world.gap_m > 0.f)) world.fault = Clearance;
    }

    for (int i = 0; i < max_controls && world.fault == Healthy && !ready(world); ++i) {
        if (motion.controls - world.decision_control >= cue_limit_controls) {
            world.fault = CueUnavailable;
            break;
        }
        prepare_navigation(world.navigation);
        if (motion.controls % launch_action_controls == 0 &&
            !frozen_targets(world, gait_weights, yaw_weights)) {
            world.fault = Nonfinite;
            break;
        }

        const auto start = world.pose.com_m;
        const auto step = step_turn_control(motion, world.targets, pond,
            TrackControl{world, motion.controls * turn_substeps_per_control});
        result.attempted_substeps += step.attempted_substeps;
        result.accepted_substeps += step.accepted_substeps;
        world.min_gap_m = fminf(world.min_gap_m, step.min_gap_m);
        world.peak_tilt_deg = fmaxf(world.peak_tilt_deg, step.max_tilt_deg);
        world.peak_depth_error_m = fmaxf(world.peak_depth_error_m, step.max_depth_error_m);
        if (step.fault != Healthy) {
            world.fault = step.fault;
            break;
        }

        ++result.controls;
        const auto movement = horizontal(world.pose.com_m - start);
        const float speed = physics::norm(turn_course(motion, world.pose.com_m,
            steering_history_controls)) / launch_course_window_s;
        world.transport_m += physics::norm(movement);
        if (motion.controls >= steering_history_controls) {
            world.min_speed_m_s = fminf(world.min_speed_m_s, speed);
        }
        if (speed >= minimum_speed_m_s) {
            const float projected = physics::dot(movement, motion.request.heading) /
                (motor_control_s * requested_speed_m_s);
            result.reward += .035f * motor_control_s * physics::clamp(projected, 0.f, 1.25f);
            world.stall_s = 0.f;
        } else {
            world.stall_s += motor_control_s;
            world.max_stall_s = fmaxf(world.max_stall_s, world.stall_s);
        }
        result.reward += visit_reward(world, speed);
        result.reward -= .15f * fmaxf(0.f, clearance_risk(step.min_gap_m) - clearance_risk(world.gap_m));
        world.gap_m = step.min_gap_m;
        if (world.peak_tilt_deg > envelope_tilt_deg || world.peak_depth_error_m > envelope_depth_error_m) {
            world.fault = PoseEnvelope;
        }
    }
    if (world.fault == Healthy && !ready(world) &&
        motion.controls - world.decision_control >= cue_limit_controls) world.fault = CueUnavailable;
    if (world.fault == Clearance || world.fault == CueUnavailable || world.fault == PoseEnvelope) result.reward -= 4.f;
    world.attempted_substeps += result.attempted_substeps;
    world.accepted_substeps += result.accepted_substeps;
    result.fault = world.fault;
    return result;
}

} // namespace cudarl::flexible_tail::chooser

