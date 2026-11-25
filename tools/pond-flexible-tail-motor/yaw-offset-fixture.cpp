#include "rl/flexible_tail_yaw_offset.h"
#include "nn/mlp_cpu.h"
#include "habitat_config.h"
#include "bank-room-replay.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
namespace tail = cudarl::flexible_tail;
namespace physics = cudarl::physics;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Actor = cudarl::nn::CpuMlp<float>;
using physics::V;
using tail::NavigationWorld;

constexpr int settle_controls = 20;
constexpr int beat_capture_controls = 250;
constexpr int practice_turn_controls = 400;
constexpr int practice_controls = 512;
constexpr int assessment_turn_controls = 600;
constexpr int window_controls = 100;
constexpr int beat_trace_lead_controls = 150;
constexpr int beat_trace_extension_controls = window_controls;
constexpr double beat_sample_seconds = .00125;
static_assert(tail::motor_substep_s == static_cast<float>(beat_sample_seconds));
constexpr int straight_controls = 600;
constexpr float radians_per_degree = tail::pi / 180.f;
constexpr float window_seconds = window_controls * tail::motor_control_s;
constexpr float speed_gate_mm_s = 6.f;
constexpr float course_gate_deg = 5.f;
constexpr float tilt_gate_deg = 10.f;
constexpr float depth_gate_mm = 1.f;
constexpr float stroke_keep_ratio = .8f;
constexpr float reversal_deadband_rad_s = .02f;
constexpr std::array<const char*, 12> straight_labels{{
    "neutral_a", "neutral_b", "neutral_c", "neutral_d",
    "bent_left_a", "bent_right_a", "bent_left_b", "bent_right_b",
    "moving_a", "moving_b", "moving_c", "moving_d"
}};
constexpr std::array<const char*, 8> turn_labels{{
    "moving_a_left", "moving_a_right", "moving_b_left", "moving_b_right",
    "moving_c_left", "moving_c_right", "moving_d_left", "moving_d_right"
}};

void new_directory(const fs::path& output) {
    if (!fs::create_directory(output)) throw std::runtime_error("Output directory must be new");
}

NavigationWorld turn_world(NavigationWorld world, float angle_deg, int turn_controls, int total_controls) {
    if (world.leg_count != 1 || world.legs[0].choice != tail::Choice::Swim ||
        physics::norm(tail::snapshot(world.motion.body).com_velocity_m_s) < .006f) {
        throw std::runtime_error("A yaw-offset turn requires a saved moving swim start");
    }
    auto& first = world.legs[0];
    first.end_control = settle_controls;
    const V heading = physics::axis_rotation({0, 1, 0}, angle_deg * radians_per_degree).rotate(first.heading);
    const auto turn = angle_deg > 0 ? tail::Choice::TurnLeft : tail::Choice::TurnRight;
    world.legs[1] = {first.goal_m, heading, settle_controls + turn_controls, turn};
    world.legs[2] = {first.goal_m, heading, total_controls, tail::Choice::Swim};
    world.leg_count = 3;
    return world;
}

void capture(const std::string& prior_train, const std::string& prior_eval, const fs::path& output) {
    const auto old_train = tail::read_navigation_worlds(prior_train);
    const auto straight = tail::read_navigation_worlds(prior_eval);
    if (old_train.size() != 256 || straight.size() != straight_labels.size()) {
        throw std::runtime_error("Unexpected saved launch fixture counts");
    }

    std::vector<NavigationWorld> train, turns;
    train.reserve(old_train.size());
    for (int replica = 0; replica < 2; ++replica) {
        for (size_t index = 3; index < old_train.size(); index += 4) {
            for (const float angle : {15.f, -15.f}) {
                train.push_back(turn_world(old_train[index], angle, practice_turn_controls, practice_controls));
            }
        }
    }
    for (size_t index = 8; index < straight.size(); ++index) {
        for (const float angle : {15.f, -15.f}) {
            turns.push_back(turn_world(straight[index], angle, assessment_turn_controls,
                settle_controls + assessment_turn_controls + window_controls));
        }
    }
    new_directory(output);
    tail::write_navigation_worlds((output / "train-worlds.bin").u8string(), train);
    tail::write_navigation_worlds((output / "eval-turn-worlds.bin").u8string(), turns);
    tail::write_navigation_worlds((output / "eval-straight-worlds.bin").u8string(), straight);
    const std::vector<float> zero(static_cast<size_t>(tail::yaw_offset_spec().num_params()), 0.f);
    tail::write_yaw_offset_weights((output / "zero-offset.bin").u8string(), zero);
    std::cout << "train_worlds=" << train.size() << " turn_worlds=" << turns.size()
              << " straight_worlds=" << straight.size() << " physics_substeps=0\n";
}

Actor frozen_motor(const std::vector<float>& weights, bool latent) {
    auto spec = tail::navigation_motor_spec();
    if (latent) spec.activations.back() = cudarl::nn::Activation::Identity;
    Actor fitted(std::move(spec));
    fitted.params() = weights;
    return fitted;
}

Actor offset_motor(const std::vector<float>& weights) {
    Actor offset(tail::yaw_offset_spec());
    offset.params() = weights;
    return offset;
}

int verify_commands(const std::vector<float>& fitted, const std::vector<float>& offset_weights) {
    auto latent = frozen_motor(fitted, true);
    auto direct = frozen_motor(fitted, false);
    auto offset = offset_motor(offset_weights);
    auto zero = offset_motor(std::vector<float>(offset_weights.size(), 0.f));
    int compared = 0;
    for (const float bend : {-.6f, -.2f, 0.f, .2f, .6f}) {
        for (const float rate : {-4.f, 0.f, 4.f}) {
            for (const auto choice : {tail::Choice::Swim, tail::Choice::TurnLeft, tail::Choice::TurnRight}) {
                NavigationWorld world{};
                world.leg_count = 1;
                world.legs[0] = {{0, -.07f, 0}, {1, 0, 0}, 100, choice};
                world.motion.body.angle[0] = bend;
                world.motion.body.velocity[6] = rate;
                tail::prepare_navigation(world);
                if (tail::yaw_offset_enabled(world) != (choice != tail::Choice::Swim)) {
                    throw std::runtime_error("Unexpected yaw-offset activation");
                }
                float observation[tail::navigation_inputs], feedback[tail::navigation_inputs];
                tail::observe_gait(world, observation);
                tail::observe_yaw_offset(world, 0.f, feedback);
                latent.forward(observation, 1);
                direct.forward(observation, 1);
                if (choice == tail::Choice::Swim) offset.forward(feedback, 1);
                else zero.forward(feedback, 1);
                const float selected = choice == tail::Choice::Swim ? offset.output()[0] + 1.f : zero.output()[0];
                float applied = 0.f;
                float targets[tail::motor_outputs]{};
                tail::decode_yaw_offset(world, latent.output(), selected, applied, targets);
                if (applied != 0.f || std::memcmp(targets, direct.output(), sizeof targets) != 0) {
                    throw std::runtime_error("Yaw-offset command differs from the frozen actor at zero bias");
                }
                ++compared;
            }
        }
    }
    return compared;
}

std::vector<float> read_offset(const std::string& source) {
    if (source == "zero") return std::vector<float>(tail::yaw_offset_spec().num_params(), 0.f);
    return tail::read_yaw_offset_weights(source);
}

void verify(const std::string& gait, const std::string& offset, const fs::path& output) {
    const auto fitted = tail::read_navigation_weights(gait, tail::NavigationMode::Gait);
    const int checked = verify_commands(fitted, read_offset(offset));
    new_directory(output);
    std::ofstream receipt(output / "result.json");
    receipt.exceptions(std::ios::failbit | std::ios::badbit);
    receipt << "{\"exact_command_matches\":" << checked << ",\"physics_substeps\":0}\n";
    std::cout << "exact_command_matches=" << checked << " physics_substeps=0\n";
}

int reversal_direction(float rate) {
    return rate > reversal_deadband_rad_s ? 1 : (rate < -reversal_deadband_rad_s ? -1 : 0);
}

struct Oscillation {
    float low = std::numeric_limits<float>::infinity();
    float high = -std::numeric_limits<float>::infinity();
    int direction = 0;
    int reversals = 0;

    void observe(float angle, float rate) {
        low = std::min(low, angle);
        high = std::max(high, angle);
        const int next = reversal_direction(rate);
        if (next == 0) return;
        reversals += direction != 0 && direction != next;
        direction = next;
    }

    float swing() const { return std::max(0.f, high - low); }
};

struct Segment {
    V launch{}, finish{};
    Oscillation root, passive;
    double yaw_work_J = 0, target_sum = 0, offset_sum = 0;
    int controls = 0;
    bool complete = false;
};

struct BeatEvent {
    const char* axis;
    double confirmed_s, crossing_s;
    int sample, direction;
    V crossing_com{};
};

struct BeatDirection {
    float previous_rate = 0;
    double previous_s = 0;
    double crossing_s = std::numeric_limits<double>::quiet_NaN();
    int direction = 0, crossing_direction = 0;
    bool seen = false;
    V previous_com{}, crossing_com{};

    void observe(const char* axis, float rate, int sample, std::vector<BeatEvent>& events, V com = {}) {
        const double time_s = sample * beat_sample_seconds;
        if (seen && ((previous_rate <= 0.f && rate > 0.f) || (previous_rate >= 0.f && rate < 0.f))) {
            const double fraction = -double(previous_rate) / (rate - previous_rate);
            crossing_s = previous_s - double(previous_rate) * (time_s - previous_s) / (rate - previous_rate);
            crossing_com = previous_com + (com - previous_com) * static_cast<float>(fraction);
            crossing_direction = rate > 0 ? 1 : -1;
        }
        const int next = reversal_direction(rate);
        if (next != 0) {
            if (direction != 0 && direction != next) {
                events.push_back({axis, time_s, crossing_direction == next ? crossing_s :
                    std::numeric_limits<double>::quiet_NaN(), sample, next, crossing_com});
            }
            direction = next;
        }
        previous_rate = rate;
        previous_s = time_s;
        previous_com = com;
        seen = true;
    }
};

struct BeatTrace {
    std::vector<BeatEvent> events;
    float peak_tilt_deg = 0, peak_depth_mm = 0, min_gap_mm = std::numeric_limits<float>::infinity();
    float min_speed_mm_s = std::numeric_limits<float>::infinity();
    int attempted = 0, accepted = 0, fault = 0;
};

struct Score {
    V incoming{};
    std::array<Segment, 3> segments{};
    float measured_tilt_deg = 0, measured_depth_mm = 0, minimum_gap_mm = 1000;
    float minimum_moving_speed_mm_s = std::numeric_limits<float>::infinity();
    int controls = 0, attempted = 0, accepted = 0, fault = 0, exact_straight_commands = 0;
    BeatTrace beat_trace;
};

int sign_of_turn(size_t index) { return index % 2 == 0 ? 1 : -1; }

float signed_course_deg(V from, V toward) {
    from.y = toward.y = 0.f;
    if (physics::norm(from) < .001f || physics::norm(toward) < .001f) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return std::atan2(physics::cross(from, toward).y, physics::dot(from, toward)) / radians_per_degree;
}

struct BeatCrossing {
    double time_s = 0;
    V com_m{};
};

struct BeatCapture {
    NavigationWorld world;
    std::array<BeatCrossing, 3> crossings{};
    int count = 0, attempted = 0, accepted = 0, fault = 0;
};

BeatCapture measure_approach(NavigationWorld world, Actor& fitted, Clock::time_point deadline) {
    if (world.leg_count != 1 || world.legs[0].choice != tail::Choice::Swim ||
        world.legs[0].end_control < beat_capture_controls ||
        physics::norm(tail::snapshot(world.motion.body).com_velocity_m_s) < .006f) {
        throw std::runtime_error("A whole-beat cue requires a saved moving swim start");
    }
    BeatCapture result{world};
    auto& approach = result.world;
    auto& motion = approach.motion;
    tail::prepare_navigation(approach);
    float previous_yaw = motion.body.angle[0];
    V previous_com = tail::snapshot(motion.body).com_m;
    double previous_time = 0;
    float targets[tail::motor_outputs]{};
    while (motion.controls < beat_capture_controls) {
        if (Clock::now() >= deadline) {
            result.fault = 4;
            break;
        }
        if (motion.controls % tail::launch_action_controls == 0) {
            if (result.count == static_cast<int>(result.crossings.size())) break;
            float observations[tail::navigation_inputs];
            tail::observe_gait(approach, observations);
            fitted.forward(observations, 1);
            float applied = 0;
            tail::decode_yaw_offset(approach, fitted.output(), 0.f, applied, targets);
        }
        int sample = motion.controls * tail::turn_substeps_per_control;
        const auto step = tail::step_turn_control(motion, targets, larger_pond,
            [&](const tail::Snapshot& pose) {
                const double time = double(sample++) * tail::motor_substep_s;
                const float yaw = motion.body.angle[0];
                if (time > previous_time && result.count < static_cast<int>(result.crossings.size()) &&
                    previous_yaw < 0.f && yaw >= 0.f) {
                    const float fraction = -previous_yaw / (yaw - previous_yaw);
                    result.crossings[result.count++] = {
                        previous_time + fraction * (time - previous_time),
                        previous_com + (pose.com_m - previous_com) * fraction
                    };
                }
                previous_yaw = yaw;
                previous_com = pose.com_m;
                previous_time = time;
            });
        result.attempted += step.attempted_substeps;
        result.accepted += step.accepted_substeps;
        if (step.fault != 0) {
            result.fault = step.fault;
            break;
        }
    }
    if (!result.fault && (result.count != static_cast<int>(result.crossings.size()) ||
        motion.controls % tail::launch_action_controls != 0)) {
        result.fault = 6;
    }
    if (!result.fault) {
        V travel = result.crossings.back().com_m - result.crossings.front().com_m;
        travel.y = 0;
        const float distance = physics::norm(travel);
        const double elapsed = result.crossings.back().time_s - result.crossings.front().time_s;
        if (!(elapsed > 0 && distance / elapsed >= .006)) result.fault = 7;
        else motion.entry_course = travel / distance;
    }
    return result;
}

NavigationWorld beat_turn_world(NavigationWorld world, float degrees, int turn_controls, int total_controls) {
    const V measured = world.motion.entry_course;
    const V heading = physics::axis_rotation({0, 1, 0}, degrees * radians_per_degree).rotate(measured);
    if (std::abs(signed_course_deg(measured, heading) - degrees) > .001f) {
        throw std::runtime_error("Whole-beat fixture did not produce the requested signed turn");
    }
    const V goal = world.legs[0].goal_m;
    const auto turn = degrees > 0 ? tail::Choice::TurnLeft : tail::Choice::TurnRight;
    world.legs[0] = {goal, heading, turn_controls, turn};
    world.legs[1] = {goal, heading, total_controls, tail::Choice::Swim};
    world.leg_count = 2;
    // Preserve the captured body, actuator targets and course samples.
    world.motion.controls = 0;
    return world;
}

void capture_beats(const std::string& gait, const std::string& prior_train, const std::string& prior_eval,
                   const fs::path& output, Clock::time_point deadline) {
    const auto old_train = tail::read_navigation_worlds(prior_train);
    const auto straight = tail::read_navigation_worlds(prior_eval);
    if (old_train.size() != 256 || straight.size() != straight_labels.size()) {
        throw std::runtime_error("Unexpected saved launch fixture counts");
    }
    auto fitted = frozen_motor(tail::read_navigation_weights(gait, tail::NavigationMode::Gait), true);
    new_directory(output);
    std::ofstream cues(output / "beat-cues.csv");
    cues.exceptions(std::ios::failbit | std::ios::badbit);
    cues << std::setprecision(10) << "split,saved_index,approach_controls,approach_s,crossings,"
        "first_crossing_s,second_crossing_s,third_crossing_s,first_beat_s,second_beat_s,"
        "approach_heading_x,approach_heading_z,course_x,course_z,course_shift_deg,left_target_deg,right_target_deg,"
        "attempted_substeps,accepted_substeps,fault\n";
    std::vector<NavigationWorld> train, turns;
    train.reserve(old_train.size());
    int attempted = 0, accepted = 0, approaches = 0, faults = 0;
    const auto prepare = [&](const NavigationWorld& archived, const char* split, size_t index,
                             int turn_controls, int total_controls, std::vector<NavigationWorld>& staged) {
        const auto result = measure_approach(archived, fitted, deadline);
        ++approaches;
        attempted += result.attempted;
        accepted += result.accepted;
        faults += result.fault != 0;
        const auto& world = result.world;
        const auto saved_heading = archived.legs[0].heading;
        const auto course = world.motion.entry_course;
        cues << split << ',' << index << ',' << world.motion.controls << ','
             << double(world.motion.controls) * tail::motor_control_s << ',' << result.count;
        for (int i = 0; i < 3; ++i) {
            cues << ',';
            if (i < result.count) cues << result.crossings[i].time_s;
        }
        for (int i = 1; i < 3; ++i) {
            cues << ',';
            if (i < result.count) cues << result.crossings[i].time_s - result.crossings[i - 1].time_s;
        }
        cues << ',' << saved_heading.x << ',' << saved_heading.z;
        if (result.fault == 0) {
            const auto left = beat_turn_world(world, 15.f, turn_controls, total_controls);
            const auto right = beat_turn_world(world, -15.f, turn_controls, total_controls);
            cues << ',' << course.x << ',' << course.z << ',' << signed_course_deg(saved_heading, course)
                 << ',' << signed_course_deg(course, left.legs[0].heading)
                 << ',' << signed_course_deg(course, right.legs[0].heading);
            staged.push_back(left);
            staged.push_back(right);
        } else {
            cues << ",,,,,";
        }
        cues << ',' << result.attempted << ',' << result.accepted << ',' << result.fault << '\n';
    };
    for (size_t i = 3; i < old_train.size() && !faults; i += 4) {
        prepare(old_train[i], "train", i, practice_turn_controls, practice_controls, train);
    }
    for (size_t i = 8; i < straight.size() && !faults; ++i) {
        prepare(straight[i], "eval", i, assessment_turn_controls, assessment_turn_controls + window_controls, turns);
    }
    std::ofstream receipt(output / "result.json");
    receipt.exceptions(std::ios::failbit | std::ios::badbit);
    receipt << "{\"approaches\":" << approaches << ",\"attempted_substeps\":" << attempted
            << ",\"accepted_substeps\":" << accepted << ",\"faults\":" << faults << "}\n";
    receipt.flush();
    if (faults != 0) throw std::runtime_error("A fitted approach failed to capture two complete tail beats");
    const auto paired_train = train;
    train.insert(train.end(), paired_train.begin(), paired_train.end());
    tail::write_navigation_worlds((output / "train-worlds.bin").u8string(), train);
    tail::write_navigation_worlds((output / "eval-turn-worlds.bin").u8string(), turns);
    tail::write_navigation_worlds((output / "eval-straight-worlds.bin").u8string(), straight);
    const std::vector<float> zero(static_cast<size_t>(tail::yaw_offset_spec().num_params()), 0.f);
    tail::write_yaw_offset_weights((output / "zero-offset.bin").u8string(), zero);
    std::cout << "train_worlds=" << train.size() << " turn_worlds=" << turns.size()
              << " straight_worlds=" << straight.size() << " approach_attempted_substeps="
              << attempted << " approach_accepted_substeps=" << accepted << '\n';
}

float projected_speed(const Segment& segment, const V& heading) {
    if (!segment.complete) return std::numeric_limits<float>::quiet_NaN();
    return 1000.f * physics::dot(segment.finish - segment.launch, heading) /
        (segment.controls * tail::motor_control_s);
}

float rolling_speed_mm_s(V displacement) {
    return 1000.f * physics::norm(displacement) / (tail::steering_history_controls * tail::motor_control_s);
}

void sample_segment(Segment& segment, const physics::BodyState& body, double work_J) {
    segment.root.observe(body.angle[0], body.velocity[6]);
    segment.passive.observe(body.angle[2], body.velocity[8]);
    segment.yaw_work_J += work_J;
}

Score roll(NavigationWorld world, const std::vector<float>& gait, const std::vector<float>& offset_weights,
           bool straight, Clock::time_point deadline, bool trace_beats = false) {
    auto fitted = frozen_motor(gait, true);
    auto direct = frozen_motor(gait, false);
    auto offset = offset_motor(offset_weights);
    float applied_offset = 0;
    float targets[tail::motor_outputs]{};
    const bool whole_beats = !straight && world.leg_count == 2;
    const int turn_begin = whole_beats ? 0 : settle_controls;
    const int total = world.legs[world.leg_count - 1].end_control;
    const int turn_end = straight ? total - window_controls : world.legs[whole_beats ? 0 : 1].end_control;
    const int stop = total + (trace_beats ? beat_trace_extension_controls : 0);
    const std::array<int, 3> segment_begin{straight ? turn_end - window_controls : turn_begin,
        turn_end - window_controls, turn_end};
    const std::array<int, 3> segment_end{turn_end, turn_end, total};

    Score score;
    BeatDirection root_direction, passive_direction;
    auto& trace = score.beat_trace;
    tail::prepare_navigation(world);
    auto& motion = world.motion;
    auto pose = tail::snapshot(motion.body);
    score.measured_tilt_deg = pose.tilt_deg;
    score.measured_depth_mm = 1000.f * std::abs(motion.request.origin_m.y - pose.com_m.y);
    score.minimum_gap_mm = static_cast<float>(1000 * tail::submerged_gap_m(larger_pond, motion.body, pose));
    if (!(score.minimum_gap_mm > 0)) score.fault = 2;
    for (int control = 0; control < stop && score.fault == 0; ++control) {
        const bool extension = control >= total;
        auto& fault = extension ? trace.fault : score.fault;
        auto& attempted = extension ? trace.attempted : score.attempted;
        auto& accepted = extension ? trace.accepted : score.accepted;
        if ((control & 15) == 0 && Clock::now() >= deadline) {
            fault = 4;
            break;
        }
        tail::prepare_navigation(world);
        if (!straight && control == turn_begin) {
            const auto rolling = tail::turn_course(motion, pose.com_m, tail::steering_history_controls);
            score.incoming = whole_beats ? motion.entry_course : rolling;
            score.minimum_moving_speed_mm_s = rolling_speed_mm_s(rolling);
        }
        if (control % tail::launch_action_controls == 0) {
            float gait_observation[tail::navigation_inputs], offset_observation[tail::navigation_inputs];
            tail::observe_gait(world, gait_observation);
            tail::observe_yaw_offset(world, applied_offset, offset_observation);
            fitted.forward(gait_observation, 1);
            offset.forward(offset_observation, 1);
            if (!physics::finite_number(offset.output()[0]) || !physics::finite_number(fitted.output()[0]) ||
                !physics::finite_number(fitted.output()[1])) {
                fault = 3;
                break;
            }
            tail::decode_yaw_offset(world, fitted.output(), offset.output()[0], applied_offset, targets);
            if (straight) {
                direct.forward(gait_observation, 1);
                if (applied_offset != 0.f || std::memcmp(targets, direct.output(), sizeof targets) != 0) {
                    score.fault = 5;
                    break;
                }
                ++score.exact_straight_commands;
            }
        }

        // Observe the same native substeps as the trainer with physical torque
        // telemetry. The evaluation does not need to recompute PPO reward.
        for (size_t s = 0; s < score.segments.size(); ++s) {
            if (control == segment_begin[s]) {
                score.segments[s].launch = pose.com_m;
                sample_segment(score.segments[s], motion.body, 0);
            }
        }
        tail::record_turn_position(motion, pose.com_m);
        for (int substep = 0; substep < tail::turn_substeps_per_control; ++substep) {
            ++attempted;
            tail::StepInfo info;
            if (!tail::step(motion.body, targets, tail::motor_substep_s, info)) {
                fault = 1;
                break;
            }
            pose = tail::snapshot(motion.body);
            const float gap_mm = static_cast<float>(1000 * tail::submerged_gap_m(larger_pond, motion.body, pose));
            auto& minimum_gap_mm = extension ? trace.min_gap_mm : score.minimum_gap_mm;
            minimum_gap_mm = std::min(minimum_gap_mm, gap_mm);
            if (!(gap_mm > 0)) {
                fault = 2;
                break;
            }
            auto& tilt_deg = extension ? trace.peak_tilt_deg : score.measured_tilt_deg;
            auto& depth_mm = extension ? trace.peak_depth_mm : score.measured_depth_mm;
            tilt_deg = std::max(tilt_deg, pose.tilt_deg);
            depth_mm = std::max(depth_mm, 1000.f * std::abs(motion.request.origin_m.y - pose.com_m.y));
            if (trace_beats && control >= turn_end - beat_trace_lead_controls) {
                const int sample = (control - turn_begin) * tail::turn_substeps_per_control + substep + 1;
                root_direction.observe("root", motion.body.velocity[6], sample, trace.events);
                passive_direction.observe("passive", motion.body.velocity[8], sample, trace.events);
            }
            const double work_J = std::max(0., double(info.torque_Nm[0]) *
                motion.body.velocity[6] * tail::motor_substep_s);
            for (size_t s = 0; s < score.segments.size(); ++s) {
                if (control >= segment_begin[s] && control < segment_end[s]) {
                    sample_segment(score.segments[s], motion.body, work_J);
                }
            }
            ++accepted;
        }
        if (fault != 0) break;
        std::copy_n(targets, tail::motor_outputs, motion.last_targets);
        ++motion.controls;
        if (!extension) ++score.controls;
        for (size_t s = 0; s < score.segments.size(); ++s) {
            if (control >= segment_begin[s] && control < segment_end[s]) {
                auto& segment = score.segments[s];
                ++segment.controls;
                segment.target_sum += targets[0];
                segment.offset_sum += applied_offset;
                if (control + 1 == segment_end[s]) {
                    segment.finish = pose.com_m;
                    segment.complete = true;
                }
            }
        }
        if (!straight && control >= turn_begin) {
            const auto rolling = tail::turn_course(motion, pose.com_m, tail::steering_history_controls);
            auto& minimum_speed_mm_s = extension ? trace.min_speed_mm_s : score.minimum_moving_speed_mm_s;
            minimum_speed_mm_s = std::min(minimum_speed_mm_s, rolling_speed_mm_s(rolling));
        }
    }
    return score;
}

bool stroke_pass(const Segment& trial, const Segment& base) {
    return trial.complete && base.complete && base.root.swing() > 0 && base.passive.swing() > 0 &&
        base.root.reversals > 0 && base.passive.reversals > 0 &&
        trial.root.swing() >= stroke_keep_ratio * base.root.swing() &&
        trial.passive.swing() >= stroke_keep_ratio * base.passive.swing() &&
        trial.root.reversals >= stroke_keep_ratio * base.root.reversals &&
        trial.passive.reversals >= stroke_keep_ratio * base.passive.reversals;
}

bool motion_pass(const Segment& segment, const V& heading) {
    return projected_speed(segment, heading) >= speed_gate_mm_s &&
        std::abs(signed_course_deg(heading, segment.finish - segment.launch)) <= course_gate_deg;
}

bool envelope_pass(const Score& score) {
    return score.fault == 0 && score.measured_tilt_deg <= tilt_gate_deg &&
        score.measured_depth_mm <= depth_gate_mm && score.minimum_gap_mm > 0;
}

void number(std::ostream& out, double value) {
    if (std::isfinite(value)) out << value;
}

void write_segments(std::ostream& out, const char* label, const char* policy, const Score& score,
                    const Score& base, const V& heading, bool straight) {
    const char* names[3]{straight ? "retention_previous" : "turn",
        "late_turn", straight ? "retention_final" : "onward"};
    for (size_t i = 0; i < score.segments.size(); ++i) {
        if (straight && i == 1) continue;
        const auto& segment = score.segments[i];
        const bool stroke = !straight && stroke_pass(segment, base.segments[i]);
        out << label << ',' << policy << ',' << names[i] << ',' << segment.complete << ',';
        number(out, projected_speed(segment, heading));
        out << ',';
        if (segment.complete) number(out, std::abs(signed_course_deg(heading, segment.finish - segment.launch)));
        out << ',' << segment.root.swing() << ',' << segment.passive.swing() << ','
            << segment.root.reversals << ',' << segment.passive.reversals << ','
            << 1e6 * segment.yaw_work_J << ',';
        if (segment.controls > 0) out << segment.target_sum / segment.controls;
        out << ',';
        if (segment.controls > 0) out << segment.offset_sum / segment.controls;
        out << ',';
        if (!straight) out << stroke;
        out << '\n';
    }
}

void evaluate(const std::string& gait, const std::string& offset, const fs::path& fixtures,
              const fs::path& output, Clock::time_point deadline, bool trace_beats = false) {
    const auto frozen = tail::read_navigation_weights(gait, tail::NavigationMode::Gait);
    const auto policy = read_offset(offset);
    const auto turns = tail::read_navigation_worlds((fixtures / "eval-turn-worlds.bin").u8string());
    const auto straight = tail::read_navigation_worlds((fixtures / "eval-straight-worlds.bin").u8string());
    const std::vector<float> baseline(policy.size(), 0.f);
    if (turns.size() != turn_labels.size() || straight.size() != straight_labels.size()) {
        throw std::runtime_error("Unexpected yaw-offset assessment fixture counts");
    }
    const bool whole_beats = turns.front().leg_count == 2;
    const int turn_leg = whole_beats ? 0 : 1;
    const int approach_controls = whole_beats ? 0 : settle_controls;
    for (size_t i = 0; i < turns.size(); ++i) {
        const auto& world = turns[i];
        if (world.leg_count != (whole_beats ? 2 : 3) ||
            (!whole_beats && world.legs[0].end_control != settle_controls) ||
            world.legs[turn_leg].end_control != approach_controls + assessment_turn_controls ||
            world.legs[turn_leg + 1].end_control != approach_controls + assessment_turn_controls + window_controls) {
            throw std::runtime_error("Unexpected turn schedule");
        }
        if (whole_beats) {
            const float cue = signed_course_deg(world.motion.entry_course, world.legs[turn_leg].heading);
            if (!std::isfinite(cue) || std::abs(cue - sign_of_turn(i) * 15.f) > .001f) {
                throw std::runtime_error("Whole-beat cue does not match its signed turn");
            }
        }
    }
    for (const auto& world : straight) {
        if (world.leg_count != 1 || world.legs[0].choice != tail::Choice::Swim ||
            world.legs[0].end_control != straight_controls) {
            throw std::runtime_error("Unexpected retention schedule");
        }
    }
    const int matched = verify_commands(frozen, policy);
    new_directory(output);
    std::vector<std::future<Score>> base_runs, turn_runs, straight_runs;
    for (const auto& world : turns) {
        base_runs.push_back(std::async(std::launch::async, [&, world] {
            return roll(world, frozen, baseline, false, deadline, trace_beats);
        }));
        turn_runs.push_back(std::async(std::launch::async, [&, world] {
            return roll(world, frozen, policy, false, deadline, trace_beats);
        }));
    }
    for (const auto& world : straight) {
        straight_runs.push_back(std::async(std::launch::async, [&, world] { return roll(world, frozen, policy, true, deadline); }));
    }

    std::ofstream cases(output / "cases.csv"), segments(output / "segments.csv"), result(output / "result.json");
    for (auto* out : {&cases, &segments, &result}) out->exceptions(std::ios::failbit | std::ios::badbit);
    std::ofstream events, extensions;
    if (trace_beats) {
        events.exceptions(std::ios::failbit | std::ios::badbit);
        extensions.exceptions(std::ios::failbit | std::ios::badbit);
        events.open(output / "beat-events.csv");
        extensions.open(output / "beat-extensions.csv");
        events << std::setprecision(10) << "case,policy,axis,confirmed_sample,confirmed_s,crossing_s,direction\n";
        extensions << std::setprecision(8) << "case,policy,attempted_substeps,accepted_substeps,fault,peak_tilt_deg,"
            "peak_depth_error_mm,min_gap_mm,min_rolling_speed_mm_s\n";
    }
    cases << std::setprecision(8) << "case,policy,target_turn_deg,measured_target_turn_deg,achieved_turn_deg,passed,stroke_pass,"
        "controls,attempted_substeps,accepted_substeps,fault,peak_tilt_deg,peak_depth_error_mm,min_gap_mm,"
        "min_rolling_speed_mm_s,exact_straight_commands\n";
    segments << std::setprecision(8) << "case,policy,segment,complete,speed_mm_s,course_error_deg,root_swing_rad,"
        "passive_swing_rad,root_reversals,passive_reversals,positive_yaw_work_uJ,"
        "mean_yaw_target_rad,mean_applied_offset,stroke_pass\n";
    int base_passes = 0, left = 0, right = 0, retention = 0, rest = 0, faults = 0, exact = matched;
    int attempted = 0, accepted = 0;
    int extended_attempted = 0, extended_accepted = 0, extended_faults = 0;
    const auto record = [&](const char* label, const char* controller, int requested,
                            const V& heading, const Score& score, const Score& base) {
        const bool turning = requested != 0;
        const auto& late = score.segments[turning ? 1 : 0];
        const float achieved = signed_course_deg(score.incoming, late.finish - late.launch);
        const bool stroke = turning && stroke_pass(score.segments[0], base.segments[0]) &&
            stroke_pass(late, base.segments[1]) && stroke_pass(score.segments[2], base.segments[2]);
        const bool turn_pass = !turning || (stroke && std::isfinite(achieved) &&
            achieved * requested > 0 && score.minimum_moving_speed_mm_s >= speed_gate_mm_s);
        const bool pass = envelope_pass(score) && turn_pass &&
            motion_pass(late, heading) && motion_pass(score.segments[2], heading);
        cases << label << ',' << controller << ',' << requested << ',';
        if (turning) number(cases, signed_course_deg(score.incoming, heading));
        cases << ',';
        if (turning && late.complete) number(cases, achieved);
        cases << ',' << pass << ',';
        if (turning) cases << stroke;
        cases << ',' << score.controls << ',' << score.attempted << ',' << score.accepted << ','
            << score.fault << ',' << score.measured_tilt_deg << ',' << score.measured_depth_mm << ','
            << score.minimum_gap_mm << ',';
        number(cases, score.minimum_moving_speed_mm_s);
        cases << ',' << score.exact_straight_commands << '\n';
        write_segments(segments, label, controller, score, base, heading, !turning);
        if (trace_beats && turning) {
            const auto& trace = score.beat_trace;
            for (const auto& event : trace.events) {
                events << label << ',' << controller << ',' << event.axis << ',' << event.sample << ','
                       << event.confirmed_s << ',';
                number(events, event.crossing_s);
                events << ',' << event.direction << '\n';
            }
            extensions << label << ',' << controller << ',' << trace.attempted << ',' << trace.accepted << ','
                       << trace.fault << ',' << trace.peak_tilt_deg << ',' << trace.peak_depth_mm << ',';
            number(extensions, trace.min_gap_mm);
            extensions << ',';
            number(extensions, trace.min_speed_mm_s);
            extensions << '\n';
            extended_attempted += trace.attempted;
            extended_accepted += trace.accepted;
            extended_faults += trace.fault != 0;
        }
        attempted += score.attempted;
        accepted += score.accepted;
        faults += score.fault != 0;
        exact += score.exact_straight_commands;
        return pass;
    };

    for (size_t i = 0; i < turns.size(); ++i) {
        const auto base = base_runs[i].get();
        const auto candidate = turn_runs[i].get();
        const int sign = sign_of_turn(i);
        const auto heading = turns[i].legs[turn_leg].heading;
        base_passes += record(turn_labels[i], "zero", sign * 15, heading, base, base);
        const bool pass = record(turn_labels[i], "offset", sign * 15, heading, candidate, base);
        if (sign > 0) left += pass;
        else right += pass;
    }
    for (size_t i = 0; i < straight.size(); ++i) {
        const auto score = straight_runs[i].get();
        const bool pass = record(straight_labels[i], "offset", 0, straight[i].legs[0].heading, score, score);
        retention += pass;
        rest += i < 8 && pass;
    }
    result << "{\"zero_turn_passes\":" << base_passes << ",\"left_turn_passes\":" << left
           << ",\"right_turn_passes\":" << right << ",\"straight_passes\":" << retention
           << ",\"rest_passes\":" << rest << ",\"faults\":" << faults << ",\"attempted_substeps\":"
           << attempted << ",\"accepted_substeps\":" << accepted << ",\"exact_command_matches\":" << exact;
    if (trace_beats) {
        result << ",\"extension_attempted_substeps\":" << extended_attempted
               << ",\"extension_accepted_substeps\":" << extended_accepted
               << ",\"extension_faults\":" << extended_faults;
    }
    result << "}\n";
    std::cout << "zero_turn_passes=" << base_passes << "/8 left_turn_passes=" << left << "/4 right_turn_passes="
              << right << "/4 straight_passes=" << retention << "/12 rest_passes=" << rest << "/8"
              << " exact_command_matches=" << exact << " attempted_substeps=" << attempted
              << " accepted_substeps=" << accepted << " faults=" << faults << '\n';
    if (trace_beats) {
        std::cout << "extension_attempted_substeps=" << extended_attempted
                  << " extension_accepted_substeps=" << extended_accepted
                  << " extension_faults=" << extended_faults << '\n';
    }
    if (faults != 0) throw std::runtime_error("A held-out yaw-offset case stopped early");
    if (extended_faults != 0) throw std::runtime_error("A whole-beat diagnostic extension stopped early");
}

constexpr int handoff_fixture_index = 1;
constexpr int handoff_cue_limit = 800;
constexpr int handoff_branch_controls = assessment_turn_controls + 2 * window_controls;
constexpr int handoff_substep_cap =
    (handoff_cue_limit + 2 * handoff_branch_controls) * tail::turn_substeps_per_control;
static_assert(handoff_substep_cap == 38400);

struct HandoffState {
    NavigationWorld world;
    float applied_offset = 0, targets[tail::motor_outputs]{};
    BeatDirection root, passive;
    std::vector<BeatEvent> events;
    int last_sample = -1;
};

struct HandoffLeg {
    std::array<Segment, 2> segments{};
    float peak_tilt = 0.f, peak_depth_mm = 0.f;
    float min_gap_mm = std::numeric_limits<float>::infinity();
    float min_speed_mm_s = std::numeric_limits<float>::infinity();
    int attempted = 0, accepted = 0, fault = 0;
};

uint64_t handoff_digest(const HandoffState& state) {
    cudarl::checkpoint::Writer out;
    const auto& motion = state.world.motion;
    out.value(motion.body);
    out.value(motion.request.choice);
    out.value(motion.request.origin_m);
    out.value(motion.request.heading);
    out.value(motion.request.terminal_velocity_m_s);
    out.value(motion.request.initialized);
    out.value(motion.history);
    out.value(motion.history_next);
    out.value(motion.controls);
    out.value(motion.last_targets);
    out.value(state.world.active_leg);
    out.value(state.applied_offset);
    out.value(state.targets);
    out.value(state.last_sample);
    for (const auto* tracker : {&state.root, &state.passive}) {
        out.value(tracker->previous_rate);
        out.value(tracker->previous_s);
        out.value(tracker->crossing_s);
        out.value(tracker->direction);
        out.value(tracker->crossing_direction);
        out.value(tracker->seen);
        out.value(tracker->previous_com);
        out.value(tracker->crossing_com);
    }
    for (const auto& event : state.events) {
        out.value(event.axis[0]);
        out.value(event.confirmed_s);
        out.value(event.crossing_s);
        out.value(event.sample);
        out.value(event.direction);
        out.value(event.crossing_com);
    }
    return cudarl::checkpoint::checksum(out.data);
}

void record_bank_sample(std::vector<bank_replay::Sample>& samples, int sample,
                        const physics::BodyState& body, const tail::Snapshot& pose) {
    bank_replay::Sample row{};
    row.sample = sample;
    row.com_m = pose.com_m;
    row.head_m = body.position * physics::scale;
    std::copy(std::begin(pose.spine_m), std::end(pose.spine_m), row.spine_m.begin());
    row.root_yaw_rad = body.angle[0];
    row.passive_yaw_rad = body.angle[2];
    samples.push_back(row);
}

void handoff_step(HandoffState& state, HandoffLeg& leg, Actor& fitted, Actor& offset,
                  const char* label, std::ostream& decisions, int scoring_boundary,
                  Clock::time_point deadline, std::vector<bank_replay::Sample>* bank_samples = nullptr) {
    auto& world = state.world;
    auto& motion = world.motion;
    const int control = motion.controls;
    if (Clock::now() >= deadline) {
        leg.fault = 4;
        return;
    }
    tail::prepare_navigation(world);
    if (control % tail::launch_action_controls == 0) {
        float gait_observation[tail::navigation_inputs], offset_observation[tail::navigation_inputs];
        tail::observe_gait(world, gait_observation);
        tail::observe_yaw_offset(world, state.applied_offset, offset_observation);
        fitted.forward(gait_observation, 1);
        offset.forward(offset_observation, 1);
        const float latent = offset.output()[0];
        if (!physics::finite_number(latent) || !physics::finite_number(fitted.output()[0]) ||
            !physics::finite_number(fitted.output()[1])) {
            leg.fault = 3;
            return;
        }
        const bool enabled = tail::yaw_offset_enabled(world);
        const float before = state.applied_offset;
        const float requested = enabled ? tail::yaw_offset_limit * tanhf(latent) : 0.f;
        tail::decode_yaw_offset(world, fitted.output(), latent, state.applied_offset, state.targets);
        decisions << label << ',' << control << ',' << control * double(tail::motor_control_s) << ','
                  << enabled << ',' << requested << ',' << before << ',' << state.applied_offset << ','
                  << state.targets[0] << '\n';
    }
    const int relative = control - scoring_boundary;
    Segment* segment = relative >= -window_controls && relative < window_controls
        ? &leg.segments[relative < 0 ? 0 : 1] : nullptr;
    if (segment && segment->controls == 0) segment->launch = tail::snapshot(motion.body).com_m;
    int sample = control * tail::turn_substeps_per_control;
    const auto step = tail::step_turn_control(motion, state.targets, larger_pond,
        [&](const tail::Snapshot& pose) {
            if (sample > state.last_sample) {
                state.root.observe("root", motion.body.velocity[6], sample, state.events, pose.com_m);
                state.passive.observe("passive", motion.body.velocity[8], sample, state.events, pose.com_m);
                state.last_sample = sample;
                if (bank_samples) record_bank_sample(*bank_samples, sample, motion.body, pose);
            }
            ++sample;
            if (segment) sample_segment(*segment, motion.body, 0.);
        });
    leg.attempted += step.attempted_substeps;
    leg.accepted += step.accepted_substeps;
    leg.fault = step.fault;
    leg.peak_tilt = std::max(leg.peak_tilt, step.max_tilt_deg);
    leg.peak_depth_mm = std::max(leg.peak_depth_mm, step.max_depth_error_m * 1000.f);
    leg.min_gap_mm = std::min(leg.min_gap_mm, step.min_gap_m * 1000.f);
    if (leg.fault != 0) return;
    const auto com = tail::snapshot(motion.body).com_m;
    const auto rolling = tail::turn_course(motion, com, tail::steering_history_controls);
    leg.min_speed_mm_s = std::min(leg.min_speed_mm_s, rolling_speed_mm_s(rolling));
    if (segment && ++segment->controls == window_controls) {
        segment->finish = com;
        segment->complete = true;
    }
}

void handoff_pair(const std::string& gait, const std::string& offset_file, const fs::path& fixtures,
                  const fs::path& output, Clock::time_point deadline) {
    const auto frozen = tail::read_navigation_weights(gait, tail::NavigationMode::Gait);
    const auto policy = tail::read_yaw_offset_weights(offset_file);
    auto fitted = frozen_motor(frozen, true);
    auto offset = offset_motor(policy);
    const auto turns = tail::read_navigation_worlds((fixtures / "eval-turn-worlds.bin").u8string());
    if (turns.size() != turn_labels.size()) throw std::runtime_error("Unexpected handoff fixture bundle");
    HandoffState shared{turns.at(handoff_fixture_index)};
    const auto right_leg = shared.world.legs[1];
    if (shared.world.leg_count != 2 || shared.world.legs[0].choice != tail::Choice::TurnRight ||
        shared.world.legs[0].end_control != assessment_turn_controls ||
        right_leg.choice != tail::Choice::Swim ||
        right_leg.end_control != assessment_turn_controls + window_controls ||
        std::abs(signed_course_deg(shared.world.motion.entry_course, right_leg.heading) + 15.f) > .001f) {
        throw std::runtime_error("Expected the archived whole-beat moving_a_right fixture");
    }

    new_directory(output);
    std::ofstream decisions(output / "decisions.csv"), events(output / "beat-events.csv"),
        segments(output / "segments.csv"), result(output / "result.json");
    for (auto* stream : {&decisions, &events, &segments, &result}) {
        stream->exceptions(std::ios::failbit | std::ios::badbit);
        *stream << std::setprecision(10);
    }
    decisions << "branch,control,time_s,offset_enabled,requested_offset,applied_before,applied_after,yaw_target_rad\n";
    events << "branch,axis,confirmed_sample,confirmed_s,crossing_s,direction\n";
    segments << "branch,segment,complete,signed_course_change_deg,course_error_deg,projected_speed_mm_s,"
        "root_swing_rad,passive_swing_rad,root_reversals,passive_reversals,motion_pass\n";

    HandoffLeg prefix, held, returning;
    std::array<BeatEvent, 3> crossings{};
    size_t count = 0;
    shared.world.legs[1].end_control = handoff_cue_limit;
    while (shared.world.motion.controls < handoff_cue_limit && prefix.fault == 0) {
        const auto previous_event = shared.events.size();
        handoff_step(shared, prefix, fitted, offset, "shared", decisions, assessment_turn_controls, deadline);
        for (size_t i = previous_event; i < shared.events.size() && count < crossings.size(); ++i) {
            const auto& event = shared.events[i];
            if (event.axis[0] == 'r' && event.direction > 0 &&
                event.crossing_s >= assessment_turn_controls * double(tail::motor_control_s)) crossings[count++] = event;
        }
        if (count == crossings.size() && shared.world.motion.controls % tail::launch_action_controls == 0) break;
    }
    int fork_control = -1;
    uint64_t digest = 0;
    V measured{}, left_target{};
    auto control = shared;
    auto test = shared;
    if (prefix.fault == 0) {
        if (count != crossings.size() || shared.world.motion.controls % tail::launch_action_controls != 0) {
            prefix.fault = 6;
        } else {
            auto delta = crossings.back().crossing_com - crossings.front().crossing_com;
            delta.y = 0;
            const float length = physics::norm(delta);
            if (!(length >= .001f)) prefix.fault = 7;
            else measured = delta / length;
        }
    }
    if (prefix.fault == 0) {
        digest = handoff_digest(shared);
        if (handoff_digest(control) != digest || handoff_digest(test) != digest) {
            throw std::runtime_error("The two branches do not start from the exact shared state");
        }
        fork_control = shared.world.motion.controls;
        const int comparison_end = fork_control + assessment_turn_controls;
        const int end = fork_control + handoff_branch_controls;
        left_target = physics::axis_rotation({0, 1, 0}, 15.f * radians_per_degree).rotate(measured);
        control.world.legs[1].end_control = end;
        test.world.legs[1].end_control = fork_control;
        test.world.legs[2] = {right_leg.goal_m, left_target, comparison_end, tail::Choice::TurnLeft};
        test.world.legs[3] = {right_leg.goal_m, left_target, end, tail::Choice::Swim};
        test.world.leg_count = 4;
        while (control.world.motion.controls < end && held.fault == 0) {
            handoff_step(control, held, fitted, offset, "control", decisions, comparison_end, deadline);
        }
        while (test.world.motion.controls < end && returning.fault == 0) {
            handoff_step(test, returning, fitted, offset, "test", decisions, comparison_end, deadline);
        }
    }

    const auto write_events = [&](const char* label, const HandoffState& state, size_t first) {
        for (size_t i = first; i < state.events.size(); ++i) {
            const auto& event = state.events[i];
            events << label << ',' << event.axis << ',' << event.sample << ',' << event.confirmed_s << ',';
            number(events, event.crossing_s);
            events << ',' << event.direction << '\n';
        }
    };
    write_events("shared", shared, 0);
    write_events("control", control, shared.events.size());
    write_events("test", test, shared.events.size());
    const auto write_segments = [&](const char* label, const HandoffLeg& leg, V incoming, V target, bool shared_leg) {
        for (size_t i = 0; i < leg.segments.size(); ++i) {
            const auto& segment = leg.segments[i];
            const auto moved = segment.finish - segment.launch;
            const char* name = i == 0 ? (shared_leg ? "right_turn" : "late_comparison") :
                (shared_leg ? "right_onward" : "onward");
            segments << label << ',' << name << ',' << segment.complete << ',';
            if (segment.complete) number(segments, signed_course_deg(incoming, moved));
            segments << ',';
            if (segment.complete) number(segments, std::abs(signed_course_deg(target, moved)));
            segments << ',';
            number(segments, projected_speed(segment, target));
            segments << ',' << segment.root.swing() << ',' << segment.passive.swing() << ','
                     << segment.root.reversals << ',' << segment.passive.reversals << ','
                     << motion_pass(segment, target) << '\n';
        }
    };
    write_segments("shared", prefix, shared.world.motion.entry_course, right_leg.heading, true);
    write_segments("control", held, measured, right_leg.heading, false);
    write_segments("test", returning, measured, left_target, false);

    const int attempted = prefix.attempted + held.attempted + returning.attempted;
    const int accepted = prefix.accepted + held.accepted + returning.accepted;
    if (attempted > handoff_substep_cap) throw std::runtime_error("Handoff exceeded its physical substep cap");
    const auto json_number = [&](double value) {
        if (std::isfinite(value)) result << value;
        else result << "null";
    };
    const auto write_run = [&](const char* label, const HandoffLeg& leg, const HandoffState& state) {
        result << ",\"" << label << "\":{\"attempted_substeps\":" << leg.attempted
               << ",\"accepted_substeps\":" << leg.accepted << ",\"fault\":" << leg.fault
               << ",\"end_control\":" << state.world.motion.controls << ",\"peak_tilt_deg\":" << leg.peak_tilt
               << ",\"peak_depth_error_mm\":" << leg.peak_depth_mm << ",\"min_gap_mm\":";
        json_number(leg.min_gap_mm);
        result << ",\"min_rolling_speed_mm_s\":";
        json_number(leg.min_speed_mm_s);
        result << '}';
    };
    result << "{\"case\":\"moving_a_right\",\"fixture_index\":" << handoff_fixture_index
           << ",\"fork_control\":" << fork_control << ",\"fork_s\":"
           << fork_control * double(tail::motor_control_s)
           << ",\"shared_fork_state_verified\":" << (fork_control >= 0 ? "true" : "false")
           << ",\"fork_state_checksum\":\"" << std::hex << digest << std::dec
           << "\",\"fork_applied_offset\":" << shared.applied_offset
           << ",\"incoming_heading_x\":" << measured.x << ",\"incoming_heading_z\":" << measured.z
           << ",\"control_heading_x\":" << right_leg.heading.x << ",\"control_heading_z\":" << right_leg.heading.z
           << ",\"test_heading_x\":" << left_target.x << ",\"test_heading_z\":" << left_target.z
           << ",\"cue_crossing_s\":[";
    for (size_t i = 0; i < count; ++i) result << (i ? "," : "") << crossings[i].crossing_s;
    result << "],\"attempted_substeps\":" << attempted << ",\"accepted_substeps\":" << accepted;
    write_run("shared", prefix, shared);
    write_run("control", held, control);
    write_run("test", returning, test);
    result << "}\n";
    std::cout << "fork_control=" << fork_control << " attempted_substeps=" << attempted
              << " accepted_substeps=" << accepted << " shared_fault=" << prefix.fault
              << " control_fault=" << held.fault << " test_fault=" << returning.fault << '\n';
    if (prefix.fault || held.fault || returning.fault) throw std::runtime_error("The matched handoff stopped early");
}

constexpr int alternating_substep_cap =
    (handoff_cue_limit + 4 * handoff_branch_controls) * tail::turn_substeps_per_control;
static_assert(alternating_substep_cap == 64000);

struct AlternatingFork {
    HandoffState control{}, test{};
    HandoffLeg held{}, turning{};
    int control_index = -1;
    V approach{}, held_heading{}, turn_heading{};
};

struct AlternatingCue {
    HandoffState state{};
    std::array<BeatEvent, 3> crossings{};
    V approach{};
    bool ready = false;
};

struct AlternatingReceipt {
    int attempted = 0, accepted = 0;
    bool failed = false;
    bool bank_ready = false;
    V bank_approach{};
    float bank_turn_deg = 0;
    std::vector<bank_replay::Sample> bank_samples;
};

V crossing_course(const std::array<BeatEvent, 3>& crossings) {
    auto delta = crossings.back().crossing_com - crossings.front().crossing_com;
    delta.y = 0.f;
    const auto length = physics::norm(delta);
    return length >= .001f ? delta / length : V{};
}

bool positive_root_crossing(const BeatEvent& event, int earliest_control) {
    return event.axis[0] == 'r' && event.direction > 0 &&
        event.crossing_s >= earliest_control * double(tail::motor_control_s);
}

void capture_alternating_cue(const HandoffState& state, int onward_start, AlternatingCue& cue) {
    const int control = state.world.motion.controls;
    if (cue.ready || control < onward_start + window_controls || control > onward_start + 2 * window_controls ||
        control % tail::launch_action_controls != 0) return;
    size_t count = 0;
    for (auto i = state.events.rbegin(); i != state.events.rend() && count < cue.crossings.size(); ++i) {
        if (positive_root_crossing(*i, onward_start)) {
            cue.crossings[cue.crossings.size() - 1 - count] = *i;
            ++count;
        }
    }
    if (count != cue.crossings.size()) return;
    cue.approach = crossing_course(cue.crossings);
    if (physics::norm(cue.approach) == 0.f) return;
    cue.state = state;
    cue.ready = true;
}

// Only elapsed request legs are discarded. Keep the previous Turn and the
// active Swim, because together they keep learned offset inference enabled.
void keep_active_swim(HandoffState& state) {
    auto& world = state.world;
    if (world.active_leg < 1 || world.active_leg != world.leg_count - 1 ||
        world.legs[world.active_leg].choice != tail::Choice::Swim ||
        !tail::yaw_offset_enabled(world)) {
        throw std::runtime_error("A consecutive-turn fork requires an active onward swim with learned offset");
    }
    const auto prior = world.legs[world.active_leg - 1];
    const auto active = world.legs[world.active_leg];
    world.legs[0] = prior;
    world.legs[1] = active;
    world.active_leg = 1;
    world.leg_count = 2;
}

void compare_alternating_fork(const HandoffState& source, const std::array<BeatEvent, 3>& crossings,
                              V approach, float degrees, const char* stage, const char* held_label,
                              const char* turn_label, Actor& fitted, Actor& offset, std::ostream& decisions,
                              std::ostream& forks, Clock::time_point deadline, AlternatingFork& comparison,
                              AlternatingCue* next = nullptr, std::vector<bank_replay::Sample>* bank_samples = nullptr) {
    comparison.control = source;
    comparison.test = source;
    const auto digest = handoff_digest(source);
    if (handoff_digest(comparison.control) != digest || handoff_digest(comparison.test) != digest) {
        throw std::runtime_error("Consecutive-turn forks do not share the exact same state");
    }
    keep_active_swim(comparison.control);
    keep_active_swim(comparison.test);
    comparison.control_index = source.world.motion.controls;
    comparison.approach = approach;
    const auto held_leg = comparison.control.world.legs[1];
    comparison.held_heading = held_leg.heading;
    comparison.turn_heading =
        physics::axis_rotation({0, 1, 0}, degrees * radians_per_degree).rotate(approach);
    const auto target_change = signed_course_deg(approach, comparison.turn_heading);
    if (!std::isfinite(target_change) || std::abs(target_change - degrees) > .001f) {
        throw std::runtime_error("Consecutive-turn request has an incorrect course change");
    }
    forks << stage << ',' << comparison.control_index << ','
          << comparison.control_index * double(tail::motor_control_s) << ','
          << std::hex << digest << std::dec << ',' << source.applied_offset;
    for (const auto& event : crossings) forks << ',' << event.crossing_s;
    forks << ',' << approach.x << ',' << approach.z << ','
          << held_leg.heading.x << ',' << held_leg.heading.z << ','
          << comparison.turn_heading.x << ',' << comparison.turn_heading.z << ',' << target_change << '\n';

    const int boundary = comparison.control_index + assessment_turn_controls;
    const int end = comparison.control_index + handoff_branch_controls;
    comparison.control.world.legs[1].end_control = end;
    auto& test = comparison.test.world;
    test.legs[1].end_control = comparison.control_index;
    const auto choice = degrees < 0 ? tail::Choice::TurnRight : tail::Choice::TurnLeft;
    test.legs[2] = {held_leg.goal_m, comparison.turn_heading, boundary, choice};
    test.legs[3] = {held_leg.goal_m, comparison.turn_heading, end, tail::Choice::Swim};
    test.leg_count = 4;
    if (bank_samples) {
        bank_samples->reserve(handoff_branch_controls * tail::turn_substeps_per_control + 1);
        record_bank_sample(*bank_samples, comparison.control_index * tail::turn_substeps_per_control,
            test.motion.body, tail::snapshot(test.motion.body));
    }
    while (comparison.control.world.motion.controls < end && comparison.held.fault == 0) {
        handoff_step(comparison.control, comparison.held, fitted, offset, held_label, decisions, boundary, deadline);
    }
    while (test.motion.controls < end && comparison.turning.fault == 0) {
        handoff_step(comparison.test, comparison.turning, fitted, offset, turn_label, decisions, boundary, deadline,
            bank_samples);
        if (next && comparison.turning.fault == 0) capture_alternating_cue(comparison.test, boundary, *next);
    }
}

AlternatingReceipt run_alternating_sequence(const NavigationWorld& saved, int fixture_index,
    Actor& fitted, Actor& offset, const fs::path& output, Clock::time_point deadline) {
    const bool left = fixture_index == 0;
    const auto first_choice = left ? tail::Choice::TurnLeft : tail::Choice::TurnRight;
    const float first_degrees = left ? 15.f : -15.f;
    HandoffState prefix{saved};
    const auto first_swim = prefix.world.legs[1];
    if (prefix.world.leg_count != 2 || prefix.world.legs[0].choice != first_choice ||
        prefix.world.legs[0].end_control != assessment_turn_controls || first_swim.choice != tail::Choice::Swim ||
        first_swim.end_control != assessment_turn_controls + window_controls ||
        std::abs(signed_course_deg(prefix.world.motion.entry_course, first_swim.heading) - first_degrees) > .001f) {
        throw std::runtime_error("Unexpected archived whole-beat alternating fixture");
    }
    new_directory(output);
    std::ofstream decisions(output / "decisions.csv"), events(output / "beat-events.csv"),
        segments(output / "segments.csv"), forks(output / "forks.csv"), result(output / "result.json");
    for (auto* stream : {&decisions, &events, &segments, &forks, &result}) {
        stream->exceptions(std::ios::failbit | std::ios::badbit);
        *stream << std::setprecision(10);
    }
    decisions << "branch,control,time_s,offset_enabled,requested_offset,applied_before,applied_after,yaw_target_rad\n";
    events << "branch,axis,confirmed_sample,confirmed_s,crossing_s,direction\n";
    segments << "branch,segment,complete,signed_course_change_deg,course_error_deg,projected_speed_mm_s,"
        "root_swing_rad,passive_swing_rad,root_reversals,passive_reversals,motion_pass,"
        "window_start_s,window_end_s,target_heading_x,target_heading_z\n";
    forks << "stage,control,time_s,checksum,applied_offset,crossing_1_s,crossing_2_s,crossing_3_s,"
        "approach_heading_x,approach_heading_z,control_heading_x,control_heading_z,"
        "test_heading_x,test_heading_z,target_change_deg\n";

    AlternatingReceipt receipt;
    HandoffLeg opening;
    AlternatingCue first_cue, second_cue;
    prefix.world.legs[1].end_control = handoff_cue_limit;
    while (prefix.world.motion.controls < handoff_cue_limit && opening.fault == 0 && !first_cue.ready) {
        handoff_step(prefix, opening, fitted, offset, "prefix", decisions, assessment_turn_controls, deadline);
        if (opening.fault == 0) capture_alternating_cue(prefix, assessment_turn_controls, first_cue);
    }
    if (opening.fault == 0 && !first_cue.ready) opening.fault = 6;

    AlternatingFork second, third;
    int third_cue_fault = 0;
    if (opening.fault == 0) {
        receipt.bank_approach = first_cue.approach;
        receipt.bank_turn_deg = -first_degrees;
        compare_alternating_fork(first_cue.state, first_cue.crossings, first_cue.approach, -first_degrees,
            "stage2", "stage2_control", "stage2_test", fitted, offset, decisions, forks, deadline, second,
            &second_cue, &receipt.bank_samples);
        receipt.bank_ready = second.turning.fault == 0;
        if (second.held.fault == 0 && second.turning.fault == 0 && !second_cue.ready) third_cue_fault = 6;
        if (second.held.fault == 0 && second_cue.ready) {
            compare_alternating_fork(second_cue.state, second_cue.crossings, second_cue.approach, first_degrees,
                "stage3", "stage3_control", "stage3_test", fitted, offset, decisions, forks, deadline, third);
        }
    }

    const auto write_events = [&](const char* label, const HandoffState& state, size_t first) {
        for (size_t i = first; i < state.events.size(); ++i) {
            const auto& event = state.events[i];
            events << label << ',' << event.axis << ',' << event.sample << ',' << event.confirmed_s << ',';
            number(events, event.crossing_s);
            events << ',' << event.direction << '\n';
        }
    };
    const auto write_segments = [&](const char* label, const HandoffLeg& leg, V incoming, V target, int boundary,
                                    const char* late_label, const char* onward_label) {
        for (size_t i = 0; i < leg.segments.size(); ++i) {
            const auto& segment = leg.segments[i];
            const auto moved = segment.finish - segment.launch;
            const int start = i == 0 ? boundary - window_controls : boundary;
            segments << label << ',' << (i == 0 ? late_label : onward_label) << ',' << segment.complete << ',';
            if (segment.complete) number(segments, signed_course_deg(incoming, moved));
            segments << ',';
            if (segment.complete) number(segments, std::abs(signed_course_deg(target, moved)));
            segments << ',';
            number(segments, projected_speed(segment, target));
            segments << ',' << segment.root.swing() << ',' << segment.passive.swing() << ','
                     << segment.root.reversals << ',' << segment.passive.reversals << ','
                     << motion_pass(segment, target) << ',' << start * double(tail::motor_control_s) << ','
                     << (start + window_controls) * double(tail::motor_control_s) << ','
                     << target.x << ',' << target.z << '\n';
        }
    };
    const auto write_fork = [&](const AlternatingFork& branch, size_t inherited, const char* control,
                                const char* test) {
        if (branch.control_index < 0) return;
        write_events(control, branch.control, inherited);
        write_events(test, branch.test, inherited);
        const int boundary = branch.control_index + assessment_turn_controls;
        write_segments(control, branch.held, branch.approach, branch.held_heading, boundary, "late_comparison", "onward");
        write_segments(test, branch.turning, branch.approach, branch.turn_heading, boundary, "late_comparison", "onward");
    };
    write_events("prefix", prefix, 0);
    write_segments("prefix", opening, prefix.world.motion.entry_course, first_swim.heading,
        assessment_turn_controls, left ? "left_turn" : "right_turn", left ? "left_onward" : "right_onward");
    write_fork(second, prefix.events.size(), "stage2_control", "stage2_test");
    write_fork(third, second_cue.state.events.size(), "stage3_control", "stage3_test");

    const int attempted = opening.attempted + second.held.attempted + second.turning.attempted +
        third.held.attempted + third.turning.attempted;
    const int accepted = opening.accepted + second.held.accepted + second.turning.accepted +
        third.held.accepted + third.turning.accepted;
    if (attempted > alternating_substep_cap) throw std::runtime_error("Alternating handoff exceeded its substep cap");
    const auto json_number = [&](double value) {
        if (std::isfinite(value)) result << value;
        else result << "null";
    };
    const auto write_run = [&](const char* label, const HandoffLeg& leg, const HandoffState& state) {
        result << ",\"" << label << "\":{\"attempted_substeps\":" << leg.attempted
               << ",\"accepted_substeps\":" << leg.accepted << ",\"fault\":" << leg.fault
               << ",\"end_control\":" << state.world.motion.controls << ",\"peak_tilt_deg\":" << leg.peak_tilt
               << ",\"peak_depth_error_mm\":" << leg.peak_depth_mm << ",\"min_gap_mm\":";
        json_number(leg.min_gap_mm);
        result << ",\"min_rolling_speed_mm_s\":";
        json_number(leg.min_speed_mm_s);
        result << '}';
    };
    result << "{\"case\":\"" << turn_labels.at(fixture_index) << "\",\"fixture_index\":" << fixture_index
           << ",\"stage2_fork_control\":" << second.control_index
           << ",\"stage3_fork_control\":" << third.control_index
           << ",\"stage3_cue_fault\":" << third_cue_fault << ",\"attempted_substeps\":" << attempted
           << ",\"accepted_substeps\":" << accepted << ",\"substep_cap\":" << alternating_substep_cap
           << ",\"bank_samples\":" << receipt.bank_samples.size() << ",\"bank_replay_ready\":"
           << (receipt.bank_ready ? "true" : "false");
    write_run("prefix", opening, prefix);
    write_run("stage2_control", second.held, second.control);
    write_run("stage2_test", second.turning, second.test);
    write_run("stage3_control", third.held, third.control);
    write_run("stage3_test", third.turning, third.test);
    result << "}\n";
    std::cout << "case=" << turn_labels.at(fixture_index) << " stage2_fork_control=" << second.control_index
              << " stage3_fork_control=" << third.control_index
              << " attempted_substeps=" << attempted << " accepted_substeps=" << accepted << '\n';
    receipt.attempted = attempted;
    receipt.accepted = accepted;
    receipt.failed = opening.fault || second.held.fault || second.turning.fault || third_cue_fault ||
        third.held.fault || third.turning.fault;
    return receipt;
}

void alternating_handoff(const std::string& gait, const std::string& offset_file, const fs::path& fixtures,
                         const fs::path& output, Clock::time_point deadline) {
    const auto turns = tail::read_navigation_worlds((fixtures / "eval-turn-worlds.bin").u8string());
    if (turns.size() != turn_labels.size()) throw std::runtime_error("Unexpected alternating fixture bundle");
    auto fitted = frozen_motor(tail::read_navigation_weights(gait, tail::NavigationMode::Gait), true);
    auto offset = offset_motor(tail::read_yaw_offset_weights(offset_file));
    new_directory(output);
    const std::array<fs::path, 2> sequence_paths{output / "left_start", output / "right_start"};
    const auto left = run_alternating_sequence(turns.at(0), 0, fitted, offset, sequence_paths[0], deadline);
    const auto right = run_alternating_sequence(turns.at(1), 1, fitted, offset, sequence_paths[1], deadline);
    std::ofstream result(output / "result.json");
    result.exceptions(std::ios::failbit | std::ios::badbit);
    result << "{\"attempted_substeps\":" << left.attempted + right.attempted
           << ",\"accepted_substeps\":" << left.accepted + right.accepted
           << ",\"substep_cap\":" << 2 * alternating_substep_cap
           << ",\"left_start_failed\":" << (left.failed ? "true" : "false")
           << ",\"right_start_failed\":" << (right.failed ? "true" : "false") << "}\n";
    result.close();
    const std::array<const AlternatingReceipt*, 2> receipts{&left, &right};
    for (size_t i = 0; i < receipts.size(); ++i) {
        const auto& receipt = *receipts[i];
        if (receipt.bank_ready) bank_replay::write_estimate(sequence_paths[i], larger_pond, receipt.bank_samples,
            receipt.bank_approach, receipt.bank_turn_deg);
    }
    if (left.failed || right.failed) throw std::runtime_error("An alternating handoff stopped early");
}
constexpr int live_bank_substep_cap = 2 * (handoff_cue_limit + handoff_branch_controls) *
    tail::turn_substeps_per_control;
static_assert(live_bank_substep_cap == 51200);

struct BankStart {
    V incoming{}, outward{}, cue_m{};
};

BankStart plant_east_bank(HandoffState& state, V approach, float degrees, int setback_mm) {
    const auto east = cudarl::pond_geometry_v1::outline(larger_pond, 0.);
    if (!east.valid) throw std::runtime_error("Invalid east-bank geometry");
    V tangent{float(larger_pond.water_radius_x_m * cudarl::pond_geometry_v1::edge_derivative(0.)), 0.f,
        float(larger_pond.water_radius_z_m * east.edge_factor)};
    tangent = tangent / physics::norm(tangent);
    const V outward{tangent.z, 0.f, -tangent.x};
    if (degrees > 0.f) tangent = tangent * -1.f;
    constexpr float angle = float(7.5 * cudarl::pond_geometry_v1::pi / 180.);
    const V incoming = tangent * cosf(angle) + outward * sinf(angle);
    const auto rotation = physics::axis_rotation({0, 1, 0},
        atan2f(physics::cross(approach, incoming).y, physics::dot(approach, incoming)));

    auto& motion = state.world.motion;
    const auto start = tail::snapshot(motion.body).com_m;
    const V origin{start.x, 0.f, start.z};
    const V shore{float(east.point_m.x), 0.f, 0.f};
    const auto point = [&](V p) {
        return shore + rotation.rotate(p - origin) + outward * (-float(setback_mm) / 1000.f);
    };
    motion.body.position = point(motion.body.position * physics::scale) / physics::scale;
    motion.body.orientation = rotation * motion.body.orientation;
    motion.body.orientation.normalize();
    for (int i : {0, 3}) {
        const auto v = rotation.rotate({motion.body.velocity[i], motion.body.velocity[i + 1],
            motion.body.velocity[i + 2]});
        motion.body.velocity[i] = v.x;
        motion.body.velocity[i + 1] = v.y;
        motion.body.velocity[i + 2] = v.z;
    }
    for (auto& position : motion.history) position = point(position);
    motion.entry_course = rotation.rotate(motion.entry_course);
    motion.request.origin_m = point(motion.request.origin_m);
    motion.request.heading = rotation.rotate(motion.request.heading);
    motion.request.terminal_velocity_m_s = rotation.rotate(motion.request.terminal_velocity_m_s);
    for (int i = 0; i < state.world.leg_count; ++i) {
        auto& leg = state.world.legs[i];
        leg.goal_m = point(leg.goal_m);
        leg.heading = rotation.rotate(leg.heading);
    }
    for (auto* tracker : {&state.root, &state.passive}) {
        tracker->previous_com = point(tracker->previous_com);
        tracker->crossing_com = point(tracker->crossing_com);
    }
    for (auto& event : state.events) event.crossing_com = point(event.crossing_com);
    return {incoming, outward, tail::snapshot(motion.body).com_m};
}

struct LiveBankCase {
    int prefix_attempted = 0, prefix_accepted = 0;
    int bank_attempted = 0, bank_accepted = 0;
    bool motion_passed = false;
};

LiveBankCase run_live_bank_case(const NavigationWorld& saved, size_t index, int expected_control,
                               int setback_mm, Actor& fitted, Actor& offset, const fs::path& output,
                               Clock::time_point deadline) {
    const bool left = index == 0;
    const float degrees = left ? -15.f : 15.f;
    HandoffState state{saved};
    std::copy(std::begin(saved.motion.last_targets), std::end(saved.motion.last_targets), state.targets);
    const auto before_cue = saved.legs[1];
    if (saved.leg_count != 2 || saved.motion.controls != 0 ||
        saved.legs[0].choice != (left ? tail::Choice::TurnLeft : tail::Choice::TurnRight) ||
        saved.legs[0].end_control != assessment_turn_controls || before_cue.choice != tail::Choice::Swim) {
        throw std::runtime_error("Unexpected archived east-bank fixture");
    }
    new_directory(output);
    std::ofstream decisions(output / "decisions.csv"), events(output / "beat-events.csv"),
        segments(output / "segments.csv"), result(output / "result.json");
    for (auto* stream : {&decisions, &events, &segments, &result}) {
        stream->exceptions(std::ios::failbit | std::ios::badbit);
        *stream << std::setprecision(10);
    }
    decisions << "branch,control,time_s,offset_enabled,requested_offset,applied_before,applied_after,yaw_target_rad\n";
    events << "axis,confirmed_sample,confirmed_s,crossing_s,direction\n";
    segments << "segment,complete,signed_course_change_deg,course_error_deg,projected_speed_mm_s,"
        "root_swing_rad,passive_swing_rad,root_reversals,passive_reversals,motion_pass,"
        "window_start_s,window_end_s\n";
    state.world.legs[1].end_control = handoff_cue_limit;
    HandoffLeg prefix, live;
    AlternatingCue cue;
    while (state.world.motion.controls < handoff_cue_limit && prefix.fault == 0 && !cue.ready) {
        handoff_step(state, prefix, fitted, offset, "prefix", decisions, assessment_turn_controls, deadline);
        if (prefix.fault == 0) capture_alternating_cue(state, assessment_turn_controls, cue);
    }
    const int cue_control = state.world.motion.controls;
    const int end_turn = cue_control + assessment_turn_controls;
    const int end_case = cue_control + handoff_branch_controls;
    const auto source_digest = handoff_digest(state);
    BankStart planted;
    float initial_gap_mm = std::numeric_limits<float>::quiet_NaN();
    float inward_mm = std::numeric_limits<float>::quiet_NaN();
    V target{};
    if (!prefix.fault && (!cue.ready || cue_control != expected_control)) prefix.fault = 6;
    if (!prefix.fault) {
        planted = plant_east_bank(state, cue.approach, degrees, setback_mm);
        auto& world = state.world;
        auto& motion = world.motion;
        initial_gap_mm = static_cast<float>(1000 * tail::submerged_gap_m(
            larger_pond, motion.body, tail::snapshot(motion.body)));
        if (!(initial_gap_mm > 0)) {
            live.fault = 2;
        } else {
            target = physics::axis_rotation({0, 1, 0}, degrees * radians_per_degree).rotate(planted.incoming);
            world.legs[1].end_control = cue_control;
            const auto goal = world.legs[1].goal_m;
            world.legs[2] = {goal, target, end_turn, left ? tail::Choice::TurnRight : tail::Choice::TurnLeft};
            world.legs[3] = {goal, target, end_case, tail::Choice::Swim};
            world.leg_count = 4;
            live.min_gap_mm = initial_gap_mm;
            while (motion.controls < end_case && live.fault == 0) {
                handoff_step(state, live, fitted, offset, "east_bank", decisions, end_turn, deadline);
            }
            if (live.segments[1].complete) {
                inward_mm = -1000.f * physics::dot(live.segments[1].finish - live.segments[1].launch, planted.outward);
            }
        }
    }

    bool passed = prefix.fault == 0 && live.fault == 0 && live.peak_tilt <= tilt_gate_deg &&
        live.peak_depth_mm <= depth_gate_mm && live.min_gap_mm > 0 &&
        live.min_speed_mm_s >= speed_gate_mm_s && inward_mm > 0;
    for (size_t i = 0; i < live.segments.size(); ++i) {
        const auto& segment = live.segments[i];
        const V movement = segment.finish - segment.launch;
        const auto change = signed_course_deg(planted.incoming, movement);
        const bool window_pass = segment.complete && change * degrees > 0 && motion_pass(segment, target);
        passed = passed && window_pass;
        const int begin = end_turn + (i == 0 ? -window_controls : 0);
        segments << (i == 0 ? "late_comparison" : "onward") << ',' << segment.complete << ',';
        number(segments, change);
        segments << ',';
        number(segments, std::abs(signed_course_deg(target, movement)));
        segments << ',';
        number(segments, projected_speed(segment, target));
        segments << ',' << segment.root.swing() << ',' << segment.passive.swing() << ',' <<
            segment.root.reversals << ',' << segment.passive.reversals << ',' << window_pass << ',' <<
            begin * double(tail::motor_control_s) << ',' << (begin + window_controls) * double(tail::motor_control_s) << '\n';
    }
    for (const auto& event : state.events) {
        events << event.axis << ',' << event.sample << ',' << event.confirmed_s << ',';
        number(events, event.crossing_s);
        events << ',' << event.direction << '\n';
    }
    const auto json_number = [&](double value) {
        if (std::isfinite(value)) result << value;
        else result << "null";
    };
    result << "{\"case\":\"" << turn_labels.at(index) << "\",\"turn_deg\":" << degrees
           << ",\"cue_control\":" << cue_control << ",\"source_digest\":\"" << std::hex << source_digest << std::dec
           << "\",\"setback_mm\":" << setback_mm << ",\"prefix_fault\":" << prefix.fault << ",\"fault\":" << live.fault
           << ",\"prefix_attempted\":" << prefix.attempted << ",\"prefix_accepted\":" << prefix.accepted
           << ",\"bank_attempted\":" << live.attempted << ",\"bank_accepted\":" << live.accepted
           << ",\"peak_tilt_deg\":" << live.peak_tilt << ",\"peak_depth_error_mm\":" << live.peak_depth_mm
           << ",\"start_gap_mm\":";
    json_number(initial_gap_mm);
    result << ",\"minimum_gap_mm\":";
    json_number(live.min_gap_mm);
    result << ",\"minimum_rolling_speed_mm_s\":";
    json_number(live.min_speed_mm_s);
    result << ",\"cue_depth_mm\":" << -1000 * planted.cue_m.y << ",\"onward_inward_mm\":";
    json_number(inward_mm);
    result << ",\"motion_pass\":" << (passed ? "true" : "false") << "}\n";
    return {prefix.attempted, prefix.accepted, live.attempted, live.accepted, passed};
}

void live_east_bank(const std::string& gait, const std::string& offset_file, const fs::path& fixtures,
                    const fs::path& output, Clock::time_point deadline) {
    const auto turns = tail::read_navigation_worlds((fixtures / "eval-turn-worlds.bin").u8string());
    if (turns.size() != turn_labels.size()) throw std::runtime_error("Unexpected live east-bank fixture bundle");
    auto fitted = frozen_motor(tail::read_navigation_weights(gait, tail::NavigationMode::Gait), true);
    auto offset = offset_motor(tail::read_yaw_offset_weights(offset_file));
    new_directory(output);
    const auto left = run_live_bank_case(turns[0], 0, 756, 407, fitted, offset, output / "left_start", deadline);
    const auto right = run_live_bank_case(turns[1], 1, 760, 316, fitted, offset, output / "right_start", deadline);
    const auto attempted = left.prefix_attempted + right.prefix_attempted + left.bank_attempted + right.bank_attempted;
    const auto accepted = left.prefix_accepted + right.prefix_accepted + left.bank_accepted + right.bank_accepted;
    if (attempted > live_bank_substep_cap) throw std::runtime_error("Live east-bank work exceeded its substep cap");
    std::ofstream result(output / "result.json");
    result.exceptions(std::ios::failbit | std::ios::badbit);
    result << "{\"attempted_substeps\":" << attempted << ",\"accepted_substeps\":" << accepted
           << ",\"substep_cap\":" << live_bank_substep_cap
           << ",\"left_start_motion_pass\":" << (left.motion_passed ? "true" : "false")
           << ",\"right_start_motion_pass\":" << (right.motion_passed ? "true" : "false") << "}\n";
}

}

int main(int argc, char** argv) {
    try {
        const auto start = Clock::now();
        const auto deadline = start + std::chrono::milliseconds(3500);
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "capture" && argc == 5) capture(argv[2], argv[3], argv[4]);
        else if (mode == "capture-beats" && argc == 6) capture_beats(argv[2], argv[3], argv[4], argv[5], deadline);
        else if (mode == "verify" && argc == 5) verify(argv[2], argv[3], argv[4]);
        else if (mode == "evaluate" && argc == 6) evaluate(argv[2], argv[3], argv[4], argv[5], deadline);
        else if (mode == "diagnose-beats" && argc == 6) evaluate(argv[2], argv[3], argv[4], argv[5], deadline, true);
        else if (mode == "handoff-pair" && argc == 6) handoff_pair(argv[2], argv[3], argv[4], argv[5], deadline);
        else if (mode == "live-east-bank" && argc == 6) live_east_bank(argv[2], argv[3], argv[4], argv[5], deadline);
        else if (mode == "alternating-handoff" && argc == 6) alternating_handoff(argv[2], argv[3], argv[4], argv[5], deadline);
        else throw std::runtime_error("Usage: yaw-offset-fixture capture PRIOR_TRAIN PRIOR_EVAL NEW_DIR | "
            "capture-beats FITTED_GAIT PRIOR_TRAIN PRIOR_EVAL NEW_DIR | "
            "verify FITTED_GAIT OFFSET_OR_ZERO NEW_DIR | evaluate FITTED_GAIT OFFSET_OR_ZERO FIXTURE_DIR NEW_DIR | "
            "diagnose-beats FITTED_GAIT OFFSET_OR_ZERO FIXTURE_DIR NEW_DIR | "
            "handoff-pair FITTED_GAIT OFFSET FIXTURE_DIR NEW_DIR | "
            "alternating-handoff FITTED_GAIT OFFSET FIXTURE_DIR NEW_DIR | "
            "live-east-bank FITTED_GAIT OFFSET FIXTURE_DIR NEW_DIR");
        std::cout << "elapsed_s=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
