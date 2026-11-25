#include "rl/flexible_tail_gait.h"
#include "nn/mlp_cpu.h"
#include "habitat_config.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace tail = cudarl::flexible_tail;
namespace physics = cudarl::physics;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Actor = cudarl::nn::CpuMlp<float>;
using physics::V;
using tail::NavigationWorld;

constexpr int training_worlds = 256;
constexpr int training_controls = 1024;
constexpr int window_controls = 100;
constexpr int evaluation_windows = 6;
constexpr int evaluation_controls = evaluation_windows * window_controls;
constexpr float window_seconds = window_controls * tail::motor_control_s;
constexpr float radians_per_degree = tail::pi / 180.f;
constexpr float speed_gate_mm_s = 6.f;
constexpr float course_gate_deg = 5.f;
constexpr float tilt_gate_deg = 10.f;
constexpr float depth_gate_mm = 1.f;

struct StartLabel { const char* name; const char* start; };
constexpr std::array<StartLabel, 12> evaluation_labels{{
    {"neutral_a", "neutral"}, {"neutral_b", "neutral"}, {"neutral_c", "neutral"}, {"neutral_d", "neutral"},
    {"bent_left_a", "bent"}, {"bent_right_a", "bent"}, {"bent_left_b", "bent"}, {"bent_right_b", "bent"},
    {"moving_a", "moving"}, {"moving_b", "moving"}, {"moving_c", "moving"}, {"moving_d", "moving"}
}};

NavigationWorld straight(const tail::TurnWorld& start, int controls) {
    NavigationWorld world{};
    world.motion = start;
    const auto origin = tail::snapshot(start.body).com_m;
    world.legs[0] = {origin, start.request.heading, controls, tail::Choice::Swim};
    world.leg_count = 1;
    tail::prepare_navigation(world);
    return world;
}

tail::TurnWorld stationary(tail::TurnWorld start, float yaw_degrees, float yaw_bend, V position_delta_m) {
    const auto position = start.body.position;
    const auto heading = start.request.heading;
    const float heading_yaw = std::atan2(-heading.z, heading.x);
    start.body = physics::BodyState{};
    start.body.position = position + position_delta_m / physics::scale;
    start.body.orientation = physics::axis_rotation({0, 1, 0}, heading_yaw + yaw_degrees * radians_per_degree);
    start.body.angle[0] = yaw_bend;
    start.body.angle[2] = -.4f * yaw_bend;
    const auto pose = tail::snapshot(start.body);
    start.history_next = 0;
    for (auto& point : start.history) point = pose.com_m;
    for (auto& last : start.last_targets) last = 0.f;
    return start;
}

void new_directory(const fs::path& output) {
    if (!fs::create_directories(output)) throw std::runtime_error("Output directory must be new");
}

void capture(const fs::path& source, const fs::path& output) {
    const auto archives = tail::read_turn_worlds((source / "train-worlds.bin").u8string());
    const auto held_out = tail::read_turn_worlds((source / "eval-worlds.bin").u8string());
    if (archives.size() != training_worlds || held_out.size() != 4) {
        throw std::runtime_error("Expected the preserved 256 training and 4 held-out swim starts");
    }

    std::vector<NavigationWorld> training, evaluation;
    for (int pair = 0; pair < training_worlds / 4; ++pair) {
        const auto& moving = archives[pair * 4];
        const float fraction = static_cast<float>(pair) / (training_worlds / 4 - 1);
        const float bend = .01f + .05f * fraction;
        const float yaw = .5f + 2.f * fraction;
        const V offset{0, .004f * (2.f * fraction - 1.f), 0};
        training.push_back(straight(stationary(moving, 0, 0, offset), training_controls));
        training.push_back(straight(stationary(moving, yaw, bend, offset), training_controls));
        training.push_back(straight(stationary(moving, -yaw, -bend, offset), training_controls));
        training.push_back(straight(moving, training_controls));
    }
    constexpr V neutral_offsets[]{{-.011f, .008f, .009f}, {.009f, -.009f, .012f},
        {.014f, .007f, -.015f}, {-.008f, -.006f, -.012f}};
    for (size_t i = 0; i < held_out.size(); ++i) {
        evaluation.push_back(straight(stationary(held_out[i], 0, 0, neutral_offsets[i]), evaluation_controls));
    }
    for (int i = 0; i < 2; ++i) {
        const float bend = i == 0 ? .017f : .041f;
        const float yaw = i == 0 ? 1.4f : 2.3f;
        const V offset = i == 0 ? V{.01f, -.005f, .003f} : V{-.01f, .005f, -.003f};
        for (float sign : {1.f, -1.f}) {
            evaluation.push_back(straight(stationary(held_out[2 * i], sign * yaw, sign * bend, offset),
                evaluation_controls));
        }
    }
    for (const auto& moving : held_out) evaluation.push_back(straight(moving, evaluation_controls));

    new_directory(output);
    tail::write_navigation_worlds((output / "train-worlds.bin").u8string(), training);
    tail::write_navigation_worlds((output / "eval-worlds.bin").u8string(), evaluation);
    std::cout << "train_worlds=" << training.size() << " eval_worlds=" << evaluation.size()
              << " training_neutral=64 training_bent=128 training_moving=64 new_physics_substeps=0\n";
}

struct Window {
    float projected_mm_s = 0.f;
    float course_deg = 180.f;
    bool complete = false;
};

struct Score {
    std::array<Window, evaluation_windows> windows{};
    float initial_mm_s = 0.f, peak_tilt_deg = 0.f, peak_depth_mm = 0.f, minimum_gap_mm = 1e3f;
    float peak_cross_track_mm = 0.f, progress_mm = 0.f, reward = 0.f;
    int controls = 0, attempted = 0, accepted = 0, fault = 0;
    bool passed = false, final_window_passed = false;
};

bool forward_window_pass(const Window& window) {
    return window.complete && window.projected_mm_s >= speed_gate_mm_s && window.course_deg <= course_gate_deg;
}

Score roll(NavigationWorld world, Actor& actor, tail::NavigationMode mode, Clock::time_point deadline) {
    if (world.leg_count != 1 || world.legs[0].end_control != evaluation_controls ||
        world.legs[0].choice != tail::Choice::Swim) throw std::runtime_error("Unexpected launch evaluation request");
    tail::prepare_navigation(world);
    auto pose = tail::snapshot(world.motion.body);
    const auto origin = pose.com_m;
    const auto heading = world.motion.request.heading;
    V window_start = origin;
    Score score;
    score.initial_mm_s = 1000.f * physics::norm(pose.com_velocity_m_s);
    score.peak_tilt_deg = pose.tilt_deg;
    score.peak_depth_mm = 1000.f * std::abs(world.motion.request.origin_m.y - pose.com_m.y);
    score.minimum_gap_mm = static_cast<float>(1000 * tail::submerged_gap_m(larger_pond, world.motion.body, pose));
    if (!(score.minimum_gap_mm > 0)) score.fault = 2;
    float targets[tail::motor_outputs]{};
    for (int control = 0; control < evaluation_controls && !score.fault; ++control) {
        if ((control & 15) == 0 && Clock::now() >= deadline) {
            score.fault = 4;
            break;
        }
        if (control % tail::launch_action_controls == 0) {
            float observations[tail::navigation_inputs];
            if (mode == tail::NavigationMode::Gait) {
                tail::observe_gait(world, observations);
            } else {
                tail::observe_launch(world, observations);
            }
            actor.forward(observations, 1);
            for (int axis = 0; axis < tail::motor_outputs; ++axis) targets[axis] = actor.output()[axis];
        }
        const auto step = tail::step_launch(world, targets, larger_pond);
        score.attempted += step.attempted_substeps;
        score.accepted += step.accepted_substeps;
        score.peak_tilt_deg = std::max(score.peak_tilt_deg, step.max_tilt_deg);
        score.peak_depth_mm = std::max(score.peak_depth_mm, 1000.f * step.max_depth_error_m);
        score.minimum_gap_mm = std::min(score.minimum_gap_mm, 1000.f * step.min_gap_m);
        score.fault = step.fault;
        if (score.fault) break;
        ++score.controls;
        score.reward += step.reward;
        pose = tail::snapshot(world.motion.body);
        const auto offset = pose.com_m - origin;
        score.progress_mm = 1000.f * physics::dot(offset, heading);
        score.peak_cross_track_mm = std::max(score.peak_cross_track_mm,
            1000.f * std::abs(physics::cross(heading, offset).y));
        if (score.controls % window_controls != 0) continue;
        auto displacement = pose.com_m - window_start;
        displacement.y = 0.f;
        auto& window = score.windows[score.controls / window_controls - 1];
        window.complete = true;
        window.projected_mm_s = 1000.f * physics::dot(displacement, heading) / window_seconds;
        if (physics::norm(displacement) >= .001f) {
            window.course_deg = std::abs(std::atan2(physics::cross(heading, displacement).y,
                physics::dot(heading, displacement))) / radians_per_degree;
        }
        window_start = pose.com_m;
    }
    score.final_window_passed = score.fault == 0 && forward_window_pass(score.windows.back()) &&
        score.peak_tilt_deg <= tilt_gate_deg &&
        score.peak_depth_mm <= depth_gate_mm && score.minimum_gap_mm > 0;
    score.passed = score.final_window_passed && forward_window_pass(score.windows[evaluation_windows - 2]);
    return score;
}

void evaluate(tail::NavigationMode mode, const std::string& actor_path, const std::string& starts_path,
              const fs::path& output, Clock::time_point deadline) {
    const auto starts = tail::read_navigation_worlds(starts_path);
    const auto weights = tail::read_navigation_weights(actor_path, mode);
    if (starts.size() != evaluation_labels.size()) throw std::runtime_error("Unexpected held-out launch count");
    new_directory(output);

    std::vector<std::future<Score>> runs;
    for (const auto& start : starts) {
        runs.push_back(std::async(std::launch::async, [&, start, mode] {
            Actor actor(tail::navigation_motor_spec());
            actor.params() = weights;
            return roll(start, actor, mode, deadline);
        }));
    }
    std::ofstream cases(output / "cases.csv"), windows(output / "windows.csv");
    cases.exceptions(std::ios::failbit | std::ios::badbit);
    windows.exceptions(std::ios::failbit | std::ios::badbit);
    cases << std::setprecision(8) <<
        "case,start,pass,final_window_pass,complete_controls,attempted_substeps,accepted_substeps,fault,initial_mm_s,"
        "speed_mm_s,course_error_deg,peak_tilt_deg,peak_depth_error_mm,min_gap_mm,peak_cross_track_mm,progress_mm,reward\n";
    windows << std::setprecision(8) << "case,end_s,complete,speed_mm_s,course_error_deg\n";
    int passed = 0, rest_passed = 0, attempted = 0, accepted = 0, faults = 0;
    for (size_t i = 0; i < runs.size(); ++i) {
        const auto score = runs[i].get();
        const auto& label = evaluation_labels[i];
        const auto& final = score.windows.back();
        cases << label.name << ',' << label.start << ',' << score.passed << ',' << score.final_window_passed << ','
            << score.controls << ','
            << score.attempted << ',' << score.accepted << ',' << score.fault << ',' << score.initial_mm_s << ',';
        if (final.complete) cases << final.projected_mm_s << ',' << final.course_deg;
        else cases << ',';
        cases << ',' << score.peak_tilt_deg << ',' << score.peak_depth_mm << ',' << score.minimum_gap_mm << ','
            << score.peak_cross_track_mm << ',' << score.progress_mm << ',' << score.reward << '\n';
        for (size_t j = 0; j < score.windows.size(); ++j) {
            const auto& window = score.windows[j];
            windows << label.name << ',' << (j + 1) * window_seconds << ',' << window.complete << ',';
            if (window.complete) windows << window.projected_mm_s << ',' << window.course_deg;
            else windows << ',';
            windows << '\n';
        }
        cases.flush();
        windows.flush();
        passed += score.passed;
        rest_passed += i < 8 && score.passed;
        attempted += score.attempted;
        accepted += score.accepted;
        faults += score.fault != 0;
    }
    std::cout << "passes=" << passed << "/12 rest_passes=" << rest_passed << "/8"
              << " attempted_substeps=" << attempted << " accepted_substeps=" << accepted
              << " faults=" << faults << '\n';
    if (faults != 0) throw std::runtime_error("A held-out launch case stopped early");
}
}

int main(int argc, char** argv) {
    try {
        const auto start = Clock::now();
        const auto deadline = start + std::chrono::milliseconds(3500);
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "capture" && argc == 4) capture(argv[2], argv[3]);
        else if (mode == "evaluate" && argc == 5) {
            evaluate(tail::NavigationMode::Launch, argv[2], argv[3], argv[4], deadline);
        } else if (mode == "evaluate-gait" && argc == 5) {
            evaluate(tail::NavigationMode::Gait, argv[2], argv[3], argv[4], deadline);
        }
        else throw std::runtime_error("Usage: launch-fixture capture <archived capture folder> <new folder> | "
            "evaluate <launch actor> <eval worlds> <new folder> | evaluate-gait <gait actor> <eval worlds> <new folder>");
        std::cout << "elapsed_s=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
