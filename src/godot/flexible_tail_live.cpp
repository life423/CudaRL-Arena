#include "flexible_tail_live.h"

#include "habitat_config.h"
#include "nn/mlp_cpu.h"
#include "rl/flexible_tail_yaw_offset.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/quaternion.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace godot {
namespace {
namespace tail = cudarl::flexible_tail;
namespace physics = cudarl::physics;
namespace nn = cudarl::nn;

constexpr int turn_controls = 600;
constexpr int minimum_onward_controls = 100;
constexpr float turn_radians = tail::pi / 12.f;
constexpr float reversal_deadband_rad_s = .02f;
constexpr float minimum_steering_speed_m_s = .006f;

Vector3 point(physics::V value) { return {value.x, value.y, value.z}; }

nn::MlpSpec gait_spec() {
    auto spec = tail::navigation_motor_spec();
    spec.activations.back() = nn::Activation::Identity;
    return spec;
}

// Debounced root-yaw crossings mark the same stroke phase on successive
// beats. The cue uses COM travel between the first and third crossings.
struct CourseTracker {
    struct Crossing {
        physics::V com{};
        double sample = 0;
    };
    std::array<Crossing, 3> positive{};
    Crossing candidate{};
    physics::V previous_com{};
    float previous_rate = 0;
    int previous_sample = -1;
    int confirmed_direction = 0;
    bool positive_candidate = false;

    void observe(float rate, int sample, physics::V com) {
        if (sample <= previous_sample) return;
        if (previous_sample >= 0 && previous_rate <= 0.f && rate > 0.f) {
            const float fraction = -previous_rate / (rate - previous_rate);
            candidate = {previous_com + (com - previous_com) * fraction,
                previous_sample + double(fraction) * (sample - previous_sample)};
            positive_candidate = true;
        }
        const int direction = rate > reversal_deadband_rad_s ? 1 :
                              rate < -reversal_deadband_rad_s ? -1 : 0;
        if (direction != 0) {
            if (confirmed_direction < 0 && direction > 0 && positive_candidate) {
                positive[0] = positive[1];
                positive[1] = positive[2];
                positive[2] = candidate;
            }
            if (direction != confirmed_direction) positive_candidate = false;
            confirmed_direction = direction;
        }
        previous_rate = rate;
        previous_sample = sample;
        previous_com = com;
    }

    physics::V course(int onward_control) const {
        if (!(positive[1].sample > positive[0].sample) ||
            !(positive[0].sample > 0) ||
            positive[0].sample < double(onward_control) * tail::turn_substeps_per_control) return {};
        auto travel = positive[2].com - positive[0].com;
        travel.y = 0;
        const float distance = physics::norm(travel);
        const double seconds = (positive[2].sample - positive[0].sample) * tail::motor_substep_s;
        if (distance < minimum_steering_speed_m_s * seconds) return {};
        return travel / distance;
    }
};

const char* motion_label(tail::Choice choice) {
    if (choice == tail::Choice::TurnLeft) return "turn_left";
    if (choice == tail::Choice::TurnRight) return "turn_right";
    return "swim";
}
} // namespace

struct FlexibleTailLive::Impl {
    nn::CpuMlp<float> gait{gait_spec()};
    nn::CpuMlp<float> steering{tail::yaw_offset_spec()};
    tail::NavigationWorld start{}, world{};
    tail::Snapshot pose{};
    physics::V render_position{};
    physics::Q render_orientation{};
    CourseTracker tracker{};
    float targets[tail::motor_outputs]{};
    float applied_offset = 0;
    float clearance_m = 0;
    float rolling_speed_mm_s = 0;
    int accepted_substeps = 0;
    int onward_start = 0;
    int queued_direction = 0;
    int turn_count = 0;
    int fault = 0;
    bool loaded = false;
    std::string error;

    void observe(const tail::Snapshot& accepted_pose, int sample) {
        const auto& body = world.motion.body;
        pose = accepted_pose;
        render_position = body.position * physics::scale;
        render_orientation = body.orientation;
        tracker.observe(body.velocity[6], sample, pose.com_m);
    }

    void issue_queued_turn() {
        const int controls = world.motion.controls;
        const auto active = world.legs[world.active_leg];
        if (!queued_direction || active.choice != tail::Choice::Swim ||
            controls - onward_start < minimum_onward_controls) return;
        const auto approach = tracker.course(onward_start);
        if (physics::norm(approach) < .5f) return;

        // Keep the same native world, joints, actuator targets, trim, and
        // course history. Only the held user movement request changes.
        const auto heading = physics::axis_rotation({0, 1, 0},
            turn_radians * float(queued_direction)).rotate(approach);
        const auto choice = queued_direction > 0 ? tail::Choice::TurnLeft : tail::Choice::TurnRight;
        const auto goal = physics::V{pose.com_m.x, active.goal_m.y, pose.com_m.z};
        onward_start = controls + turn_controls;
        world.legs[0] = {goal, heading, onward_start, choice};
        world.legs[1] = {goal, heading, std::numeric_limits<int>::max(), tail::Choice::Swim};
        world.leg_count = 2;
        world.active_leg = 0;
        queued_direction = 0;
        ++turn_count;
        tail::prepare_navigation(world);
    }
};

FlexibleTailLive::FlexibleTailLive() : impl_(std::make_unique<Impl>()) {}
FlexibleTailLive::~FlexibleTailLive() = default;

void FlexibleTailLive::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_motor", "gait_path", "offset_path", "starts_path"), &FlexibleTailLive::load_motor);
    ClassDB::bind_method(D_METHOD("snapshot"), &FlexibleTailLive::snapshot);
    ClassDB::bind_method(D_METHOD("step_control"), &FlexibleTailLive::step_control);
    ClassDB::bind_method(D_METHOD("request_turn", "direction"), &FlexibleTailLive::request_turn);
    ClassDB::bind_method(D_METHOD("reset"), &FlexibleTailLive::reset);
    ClassDB::bind_method(D_METHOD("get_error"), &FlexibleTailLive::get_error);
}

bool FlexibleTailLive::load_motor(const String& gait_path, const String& offset_path, const String& starts_path) {
    auto& state = *impl_;
    state.loaded = false;
    try {
        auto gait = tail::read_navigation_weights(gait_path.utf8().get_data(), tail::NavigationMode::Gait);
        auto offset = tail::read_yaw_offset_weights(offset_path.utf8().get_data());
        const auto starts = tail::read_navigation_worlds(starts_path.utf8().get_data());
        const auto& start = starts.front();
        const auto pose = tail::snapshot(start.motion.body);
        if (starts.size() != 12 || start.leg_count != 1 || start.legs[0].choice != tail::Choice::Swim ||
            physics::norm(pose.com_velocity_m_s) >= .001f ||
            !(tail::submerged_gap_m(larger_pond, start.motion.body, pose) > 0)) {
            throw std::runtime_error("The qualified straight-swim start could not be loaded.");
        }
        state.gait.params() = std::move(gait);
        state.steering.params() = std::move(offset);
        state.start = start;
        state.loaded = true;
        return reset();
    } catch (const std::exception& error) {
        state.error = error.what();
        return false;
    }
}

bool FlexibleTailLive::reset() {
    auto& state = *impl_;
    if (!state.loaded) return false;
    state.world = state.start;
    state.world.legs[0].end_control = std::numeric_limits<int>::max();
    tail::prepare_navigation(state.world);
    state.tracker = {};
    state.applied_offset = 0;
    state.rolling_speed_mm_s = 0;
    state.accepted_substeps = 0;
    state.onward_start = 0;
    state.queued_direction = 0;
    state.turn_count = 0;
    state.fault = 0;
    state.error.clear();
    for (int axis = 0; axis < tail::motor_outputs; ++axis) {
        state.targets[axis] = state.world.motion.last_targets[axis];
    }
    const auto& body = state.world.motion.body;
    state.observe(tail::snapshot(body), 0);
    state.clearance_m = float(tail::submerged_gap_m(larger_pond, body, state.pose));
    return true;
}

bool FlexibleTailLive::request_turn(int direction) {
    auto& state = *impl_;
    if (!state.loaded || state.fault || state.queued_direction || (direction != 1 && direction != -1)) return false;
    state.queued_direction = direction;
    return true;
}

bool FlexibleTailLive::step_control() {
    auto& state = *impl_;
    if (!state.loaded || state.fault) return false;
    auto& world = state.world;
    const int controls = world.motion.controls;
    if (state.accepted_substeps >= std::numeric_limits<int>::max() - tail::turn_substeps_per_control) {
        state.fault = 4;
        state.error = "The native sample clock has reached its supported range.";
        return false;
    }
    tail::prepare_navigation(world);
    if (controls % tail::launch_action_controls == 0) {
        state.issue_queued_turn();
        float gait_observation[tail::navigation_inputs], offset_observation[tail::navigation_inputs];
        tail::observe_gait(world, gait_observation);
        tail::observe_yaw_offset(world, state.applied_offset, offset_observation);
        state.gait.forward(gait_observation, 1);
        state.steering.forward(offset_observation, 1);
        const float offset = state.steering.output()[0];
        const auto* gait = state.gait.output();
        if (!physics::finite_number(offset) || !physics::finite_number(gait[0]) || !physics::finite_number(gait[1])) {
            state.fault = 3;
            state.error = "The saved motor returned a non-finite actuator request.";
            return false;
        }
        tail::decode_yaw_offset(world, gait, offset, state.applied_offset, state.targets);
    }
    int sample = state.accepted_substeps;
    const auto result = tail::step_turn_control(world.motion, state.targets, larger_pond,
        [&](const tail::Snapshot& pose) { state.observe(pose, sample++); });
    state.accepted_substeps += result.accepted_substeps;
    state.clearance_m = result.min_gap_m;
    state.fault = result.fault;
    if (state.fault) {
        if (state.fault == 2 && state.clearance_m <= -.99f) {
            state.error = "The native guard could not certify whole-body clearance here.";
        } else if (state.fault == 2) {
            state.error = "The native whole-body clearance bound reached zero.";
        } else {
            state.error = "The native physics stopped after a numerical check failed.";
        }
        state.queued_direction = 0;
        return false;
    }
    const auto travelled = tail::turn_course(world.motion, state.pose.com_m, tail::steering_history_controls);
    state.rolling_speed_mm_s = 1000.f * physics::norm(travelled) /
        (tail::steering_history_controls * tail::motor_control_s);
    return true;
}

Dictionary FlexibleTailLive::snapshot() const {
    const auto& state = *impl_;
    Dictionary snapshot;
    if (!state.loaded) return snapshot;
    const auto& body_rotation = state.render_orientation;
    const auto& pose = state.pose;
    PackedVector3Array spine;
    for (const auto position : pose.spine_m) spine.append(point(position));
    snapshot["time_s"] = double(state.accepted_substeps) * tail::motor_substep_s;
    snapshot["body_position_m"] = point(state.render_position);
    snapshot["body_quaternion"] = Quaternion(body_rotation.x, body_rotation.y, body_rotation.z, body_rotation.w);
    snapshot["spine_m"] = spine;
    snapshot["com_m"] = point(pose.com_m);
    snapshot["com_velocity_m_s"] = point(pose.com_velocity_m_s);
    snapshot["rolling_speed_mm_s"] = state.rolling_speed_mm_s;
    snapshot["tilt_deg"] = pose.tilt_deg;
    snapshot["depth_error_mm"] = 1000.f * std::fabs(state.world.motion.request.origin_m.y - pose.com_m.y);
    snapshot["clearance_mm"] = state.clearance_m > -.99f ? Variant(1000.f * state.clearance_m) : Variant{};
    snapshot["heading"] = point(state.world.motion.request.heading);
    snapshot["motion"] = motion_label(state.world.motion.request.choice);
    snapshot["queued_direction"] = state.queued_direction;
    snapshot["fault"] = state.fault;
    snapshot["turn_count"] = state.turn_count;
    return snapshot;
}

String FlexibleTailLive::get_error() const { return String::utf8(impl_->error.c_str()); }
} // namespace godot

