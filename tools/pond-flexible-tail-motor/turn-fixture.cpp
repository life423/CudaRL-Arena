#include "rl/flexible_tail_transition.h"
#include "nn/mlp_cpu.h"
#include "habitat_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace tail = cudarl::flexible_tail;
namespace physics = cudarl::physics;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using physics::V;
using Actor = cudarl::nn::CpuMlp<float>;

constexpr float degrees_to_radians = tail::pi / 180.f;
constexpr int physics_steps = tail::turn_substeps_per_control;
constexpr int warm_shape_controls = 336;
constexpr int evaluation_turn_controls = 600;
constexpr int follow_controls = 100;
constexpr int training_world_count = 256;
constexpr int replicas = 8;
constexpr float minimum_projection_mm_s = 6.f;
constexpr float maximum_course_error_deg = 5.f;
constexpr float maximum_tilt_deg = 10.f;
constexpr float maximum_depth_error_mm = 1.f;
static_assert(tail::motor_control_s == .02f && tail::motor_substep_s == .00125f);
static_assert(tail::turn_history_controls == follow_controls);

struct Start {
    float heading, x, z, initial_yaw, phase_cycles;
};
constexpr Start train_starts[] = {
    {.09f, .025f, 0, .02f, 0},
    {-.09f, -.025f, 0, -.02f, .5f},
    {.37f, -.037f, .028f, .02f, .125f},
    {-.37f, .037f, -.028f, -.02f, .625f}
};
constexpr int train_captures[] = {300, 312, 324, warm_shape_controls};
struct EvalStart { const char* name; Start pose; float turn_degrees; int controls; };
constexpr EvalStart eval_starts[] = {
    {"left_15", {-.51f, -.014f, -.026f, -.015f, .25f}, 15, 306},
    {"right_15", {.61f, .01f, .027f, .015f, .75f}, -15, 306},
    {"left_30", {.87f, -.022f, .008f, .015f, .375f}, 30, 330},
    {"right_30", {-.82f, -.011f, -.017f, -.015f, .875f}, -30, 330}
};
static_assert(sizeof(train_starts) / sizeof(Start) * sizeof(train_captures) / sizeof(int) * 2 * replicas == training_world_count);

V direction(float yaw) { return {std::cos(yaw), 0, -std::sin(yaw)}; }

float bearing(V course) { return std::atan2(-course.z, course.x); }

float signed_angle_degrees(V from, V toward) {
    return std::remainder(bearing(toward) - bearing(from), 2 * tail::pi) / degrees_to_radians;
}

struct MotionMetrics {
    float projected_mm_s = 0, course_error_deg = 180, course_change_deg = 0;
};

MotionMetrics motion_metrics(V displacement, V heading, V entry_course) {
    displacement.y = 0;
    MotionMetrics out;
    out.projected_mm_s = 500.f * physics::dot(displacement, heading);
    if (physics::norm(displacement) >= .001f) {
        out.course_error_deg = std::abs(signed_angle_degrees(heading, displacement));
        out.course_change_deg = signed_angle_degrees(entry_course, displacement);
    }
    return out;
}

bool movement_pass(const MotionMetrics& movement) {
    return movement.projected_mm_s >= minimum_projection_mm_s && movement.course_error_deg <= maximum_course_error_deg;
}

struct CaptureMetrics {
    int attempted = 0, accepted = 0;
    float peak_tilt_deg = 0, peak_depth_mm = 0, min_gap_mm = 1000;
};

// The reference action is used only to prepare known steady-motion starts.
// Captures copy every physical field and the entire preceding COM history.
std::vector<tail::TurnWorld> capture_swim(Start start, const std::vector<int>& at_controls,
                                         CaptureMetrics& metrics, Clock::time_point deadline) {
    tail::TurnWorld world{};
    world.body.position = V{start.x, -.08f, start.z} / physics::scale;
    world.body.orientation = physics::axis_rotation({0, 1, 0}, start.heading);
    world.body.angle[0] = start.initial_yaw;
    auto pose = tail::snapshot(world.body);
    const V heading = direction(start.heading);
    world.request = {tail::Choice::Swim, pose.com_m, heading, heading * tail::requested_speed_m_s, true};
    std::vector<tail::TurnWorld> snapshots;
    snapshots.reserve(at_controls.size());

    for (int control = 0; control < at_controls.back(); ++control) {
        if ((control & 15) == 0 && Clock::now() >= deadline) throw std::runtime_error("Capture deadline reached");
        tail::record_turn_position(world, pose.com_m);
        const float phase = (control * tail::motor_control_s + start.phase_cycles) * (2 * tail::pi);
        world.last_targets[0] = .55f * std::sin(phase);
        for (int step = 0; step < physics_steps; ++step) {
            ++metrics.attempted;
            tail::StepInfo info;
            if (!tail::step(world.body, world.last_targets, tail::motor_substep_s, info)) {
                throw std::runtime_error("Capture has invalid dynamics");
            }
            pose = tail::snapshot(world.body);
            const auto gap = static_cast<float>(tail::submerged_gap_m(larger_pond, world.body, pose) * 1000);
            metrics.min_gap_mm = std::min(metrics.min_gap_mm, gap);
            metrics.peak_tilt_deg = std::max(metrics.peak_tilt_deg, pose.tilt_deg);
            metrics.peak_depth_mm = std::max(metrics.peak_depth_mm, 1000.f * std::abs(pose.com_m.y - world.request.origin_m.y));
            if (!(gap > 0)) throw std::runtime_error("Capture left certified water");
            ++metrics.accepted;
        }
        if (control + 1 != at_controls[snapshots.size()]) continue;
        const V displacement = tail::turn_course(world, pose.com_m, follow_controls);
        if (!movement_pass(motion_metrics(displacement, heading, heading)) ||
            metrics.peak_tilt_deg > maximum_tilt_deg || metrics.peak_depth_mm > maximum_depth_error_mm) {
            throw std::runtime_error("Captured swim did not meet the motion and posture gates");
        }
        auto captured = world;
        // At the start of the lesson, both arms inherit this heading and
        // associate the eventual signed cue with the two-second body course.
        captured.request.heading = displacement / physics::norm(displacement);
        captured.request.terminal_velocity_m_s = captured.request.heading * tail::requested_speed_m_s;
        snapshots.push_back(captured);
    }
    return snapshots;
}

void make_directory(const fs::path& path) {
    if (!fs::create_directories(path)) throw std::runtime_error("Output directory must be new");
}

void capture(const fs::path& destination, Clock::time_point deadline) {
    make_directory(destination);
    CaptureMetrics metrics;
    std::vector<tail::TurnWorld> training, evaluation;
    try {
        for (const auto& start : train_starts) {
            const auto captures = capture_swim(start, {std::begin(train_captures), std::end(train_captures)}, metrics, deadline);
            for (const auto& frame : captures) {
                for (int replica = 0; replica < replicas; ++replica) {
                    for (float sign : {1.f, -1.f}) {
                        auto world = frame;
                        world.turn_radians = sign * 15.f * degrees_to_radians;
                        world.turn_controls = tail::turn_training_controls;
                        training.push_back(world);
                    }
                }
            }
        }
        for (const auto& item : eval_starts) {
            auto world = capture_swim(item.pose, {item.controls}, metrics, deadline).front();
            world.turn_radians = item.turn_degrees * degrees_to_radians;
            world.turn_controls = evaluation_turn_controls;
            evaluation.push_back(world);
        }
    } catch (...) {
        std::cout << "capture_attempted_substeps=" << metrics.attempted
                  << " capture_accepted_substeps=" << metrics.accepted << '\n';
        throw;
    }
    tail::write_turn_worlds((destination / "train-worlds.bin").string(), training);
    tail::write_turn_worlds((destination / "eval-worlds.bin").string(), evaluation);
    std::cout << std::setprecision(8)
              << "train_worlds=" << training.size() << " eval_worlds=" << evaluation.size()
              << " capture_attempted_substeps=" << metrics.attempted << " capture_accepted_substeps=" << metrics.accepted
              << " capture_peak_tilt_deg=" << metrics.peak_tilt_deg
              << " capture_peak_depth_mm=" << metrics.peak_depth_mm
              << " capture_min_gap_mm=" << metrics.min_gap_mm << '\n';
}

struct Score {
    MotionMetrics turn, onward;
    float peak_tilt_deg = 0, peak_depth_mm = 0, min_gap_mm = 1000;
    int attempted = 0, accepted = 0, complete_controls = 0, fault = 0;
    bool turn_pass = false, onward_pass = false;
};

Score roll(tail::TurnWorld world, Actor& actor, Clock::time_point deadline) {
    const int turn_end = tail::turn_cue_control + world.turn_controls;
    const int final_control = turn_end + follow_controls;
    V turn_window_start{}, follow_window_start{};
    Score score;
    while (world.controls < final_control) {
        if ((world.controls & 15) == 0 && Clock::now() >= deadline) throw std::runtime_error("Evaluation deadline reached");
        if (world.controls == turn_end - follow_controls) turn_window_start = tail::snapshot(world.body).com_m;
        tail::prepare_turn_request(world);
        float observed[tail::motor_inputs];
        tail::observe_turn(world, observed);
        actor.forward(observed, 1);
        const float targets[tail::motor_outputs]{actor.output()[0], actor.output()[1]};
        const auto step = tail::step_turn_control(world, targets, larger_pond);
        score.attempted += step.attempted_substeps;
        score.accepted += step.accepted_substeps;
        score.peak_tilt_deg = std::max(score.peak_tilt_deg, step.max_tilt_deg);
        score.peak_depth_mm = std::max(score.peak_depth_mm, step.max_depth_error_m * 1000.f);
        score.min_gap_mm = std::min(score.min_gap_mm, step.min_gap_m * 1000.f);
        score.fault = step.fault;
        if (step.fault) break;
        score.complete_controls = world.controls;
        if (world.controls == turn_end) {
            follow_window_start = tail::snapshot(world.body).com_m;
            score.turn = motion_metrics(follow_window_start - turn_window_start, world.request.heading, world.entry_course);
            score.turn_pass = movement_pass(score.turn) && score.peak_tilt_deg <= maximum_tilt_deg &&
                score.peak_depth_mm <= maximum_depth_error_mm && score.min_gap_mm > 0;
        } else if (world.controls == final_control) {
            score.onward = motion_metrics(tail::snapshot(world.body).com_m - follow_window_start, world.request.heading, world.entry_course);
            score.onward_pass = movement_pass(score.onward) && score.peak_tilt_deg <= maximum_tilt_deg &&
                score.peak_depth_mm <= maximum_depth_error_mm && score.min_gap_mm > 0;
        }
    }
    return score;
}

void evaluate(const std::string& base_path, const std::string& candidate_path, const std::string& starts_path,
              const fs::path& destination, Clock::time_point deadline) {
    const auto starts = tail::read_turn_worlds(starts_path);
    if (starts.size() != sizeof(eval_starts) / sizeof(EvalStart)) throw std::runtime_error("Invalid held-out state count");
    Actor base(tail::motor_spec()), candidate(tail::motor_spec());
    base.params() = tail::read_motor_weights(base_path);
    candidate.params() = tail::read_motor_weights(candidate_path);
    make_directory(destination);
    std::ofstream csv(destination / "cases.csv");
    csv.exceptions(std::ios::failbit | std::ios::badbit);
    csv << "case,controller,signed_request_deg,turn_pass,onward_pass,full_pass,turn_speed_mm_s,turn_course_error_deg,"
           "turn_change_deg,onward_speed_mm_s,onward_course_error_deg,onward_change_deg,peak_tilt_deg,peak_depth_mm,min_gap_mm,"
           "complete_controls,attempted_substeps,accepted_substeps,fault\n";
    int counted_attempts = 0, counted_accepted = 0, base_passes = 0, candidate_passes = 0;
    for (size_t i = 0; i < starts.size(); ++i) {
        if (starts[i].controls != 0 || starts[i].turn_issued || starts[i].turn_controls != evaluation_turn_controls ||
            starts[i].turn_radians != eval_starts[i].turn_degrees * degrees_to_radians) {
            throw std::runtime_error("Unexpected held-out request");
        }
        for (const auto& label : {"saved", "candidate"}) {
            const bool comparing_base = std::string(label) == "saved";
            const auto score = roll(starts[i], comparing_base ? base : candidate, deadline);
            const bool pass = score.turn_pass && score.onward_pass && !score.fault;
            if (pass) ++(comparing_base ? base_passes : candidate_passes);
            counted_attempts += score.attempted;
            counted_accepted += score.accepted;
            csv << std::setprecision(8) << eval_starts[i].name << ',' << label << ',' << eval_starts[i].turn_degrees
                << ',' << score.turn_pass << ',' << score.onward_pass << ',' << pass
                << ',' << score.turn.projected_mm_s << ',' << score.turn.course_error_deg << ',' << score.turn.course_change_deg
                << ',' << score.onward.projected_mm_s << ',' << score.onward.course_error_deg << ',' << score.onward.course_change_deg
                << ',' << score.peak_tilt_deg << ',' << score.peak_depth_mm << ',' << score.min_gap_mm
                << ',' << score.complete_controls << ',' << score.attempted << ',' << score.accepted << ',' << score.fault << '\n';
            csv.flush();
        }
    }
    std::cout << "base_passes=" << base_passes << '/' << starts.size()
              << " candidate_passes=" << candidate_passes << '/' << starts.size()
              << " eval_attempted_substeps=" << counted_attempts << " eval_accepted_substeps=" << counted_accepted << '\n';
}
}

int main(int argc, char** argv) {
    try {
        const auto deadline = Clock::now() + std::chrono::milliseconds(3500);
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "capture" && argc == 3) capture(argv[2], deadline);
        else if (mode == "evaluate" && argc == 6) evaluate(argv[2], argv[3], argv[4], argv[5], deadline);
        else throw std::runtime_error("Usage: turn-fixture capture <new output folder> | evaluate <saved actor> <PPO actor> <eval worlds> <new output folder>");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
