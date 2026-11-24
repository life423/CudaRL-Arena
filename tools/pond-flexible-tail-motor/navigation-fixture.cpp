#include "rl/flexible_tail_navigation.h"
#include "nn/mlp_cpu.h"
#include "habitat_config.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
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
using tail::Choice;
using tail::NavigationMode;
using tail::NavigationWorld;

constexpr float radians_per_degree = tail::pi / 180.f;
constexpr int controls_per_second = 50;
constexpr int scoring_controls = 2 * controls_per_second;
constexpr int braking_controls = 4 * controls_per_second;
constexpr int training_controls = 512;
constexpr int training_worlds = 256;
constexpr float minimum_speed_mm_s = 6.f;
constexpr float maximum_course_error_deg = 5.f;
constexpr float maximum_tilt_deg = 10.f;
constexpr float maximum_depth_error_mm = 1.f;
constexpr float maximum_hold_speed_mm_s = .5f;
constexpr float maximum_hold_error_mm = 2.f;
static_assert(controls_per_second * tail::motor_control_s == 1.f);

enum class Skill { Straight, Turn, Alternating, Brake, Rest, Recovery };
struct Fixture { const char* name; Skill skill; };
constexpr Fixture evaluation_fixtures[] = {
    {"straight_a", Skill::Straight}, {"straight_b", Skill::Straight},
    {"left_15", Skill::Turn}, {"right_15", Skill::Turn},
    {"left_30", Skill::Turn}, {"right_30", Skill::Turn},
    {"alternate_left", Skill::Alternating}, {"alternate_right", Skill::Alternating},
    {"brake_hold", Skill::Brake}, {"rest_hold", Skill::Rest},
    {"recover_stationary", Skill::Recovery}, {"recover_misoriented", Skill::Recovery}
};

struct Leg { int end_control; float turn_degrees = 0; Choice choice = Choice::Swim; };

V horizontal(V vector) {
    vector.y = 0;
    return vector;
}

V rotated(V heading, float degrees) {
    return physics::axis_rotation({0, 1, 0}, degrees * radians_per_degree).rotate(heading);
}

float signed_angle_deg(V from, V toward) {
    return std::atan2(physics::cross(from, toward).y, physics::dot(from, toward)) / radians_per_degree;
}

Choice turn_choice(float degrees) { return degrees > 0 ? Choice::TurnLeft : Choice::TurnRight; }

NavigationWorld path(tail::TurnWorld motion, std::initializer_list<Leg> legs) {
    NavigationWorld world{};
    world.motion = motion;
    const V origin = tail::snapshot(motion.body).com_m;
    V nominal_start = origin;
    V heading = motion.request.heading;
    V previous_goal{};
    int start_control = 0;
    for (const auto& leg : legs) {
        heading = rotated(heading, leg.turn_degrees);
        const float duration = (leg.end_control - start_control) * tail::motor_control_s;
        const bool holding = leg.choice == Choice::Hold;
        V goal = holding ? origin : nominal_start + heading * (tail::requested_speed_m_s * duration + .06f);
        if (world.leg_count > 0 && leg.choice == Choice::Swim && leg.turn_degrees == 0 &&
            world.legs[world.leg_count - 1].choice != Choice::Hold) {
            // The onward leg keeps the preceding request and its endpoint.
            goal = previous_goal;
        }
        world.legs[world.leg_count++] = {goal, heading, leg.end_control, leg.choice};
        nominal_start += heading * (tail::requested_speed_m_s * duration);
        previous_goal = goal;
        start_control = leg.end_control;
    }
    return world;
}

tail::TurnWorld stationary(tail::TurnWorld world, float yaw_offset_deg, float depth_delta_m) {
    const V facing = rotated(world.request.heading, yaw_offset_deg);
    const float yaw = std::atan2(-facing.z, facing.x);
    const V position = world.body.position;
    world.body = physics::BodyState{};
    world.body.position = position + V{0, depth_delta_m / physics::scale, 0};
    world.body.orientation = physics::axis_rotation({0, 1, 0}, yaw);
    const V com = tail::snapshot(world.body).com_m;
    world.request.origin_m = com;
    world.history_next = 0;
    for (auto& point : world.history) point = com;
    for (auto& target : world.last_targets) target = 0;
    return world;
}

NavigationWorld training_path(tail::TurnWorld start, int scenario) {
    const float sign = scenario % 2 == 0 ? 1.f : -1.f;
    switch (scenario / 2) {
    case 0: return path(start, {{20}, {420, sign * 15, turn_choice(sign)}, {training_controls}});
    case 1: return path(start, {{20}, {420, sign * 30, turn_choice(sign)}, {training_controls}});
    case 2: return path(start, {{20}, {165, sign * 20, turn_choice(sign)},
        {340, -sign * 30, turn_choice(-sign)}, {training_controls}});
    case 3: return path(start, {{training_controls}});
    case 4: return path(start, {{training_controls, 0, Choice::Hold}});
    case 5: return path(stationary(start, 0, .008f * sign), {{training_controls, 0, Choice::Hold}});
    case 6: return path(stationary(start, 0, .008f * sign), {{training_controls, sign * 20}});
    case 7: return path(stationary(start, sign * 55, .008f * sign), {{training_controls}});
    default: throw std::runtime_error("Invalid training scenario");
    }
}

std::vector<NavigationWorld> evaluation_paths(const std::vector<tail::TurnWorld>& starts) {
    constexpr int turn_end = 20 + 12 * controls_per_second;
    constexpr int straight_end = 12 * controls_per_second;
    constexpr int alternating_leg = 12 * controls_per_second;
    return {
        path(starts[0], {{straight_end}}),
        path(starts[1], {{straight_end}}),
        path(starts[0], {{20}, {turn_end, 15, Choice::TurnLeft}, {turn_end + scoring_controls}}),
        path(starts[1], {{20}, {turn_end, -15, Choice::TurnRight}, {turn_end + scoring_controls}}),
        path(starts[2], {{20}, {turn_end, 30, Choice::TurnLeft}, {turn_end + scoring_controls}}),
        path(starts[3], {{20}, {turn_end, -30, Choice::TurnRight}, {turn_end + scoring_controls}}),
        path(starts[2], {{alternating_leg, 30, Choice::TurnLeft}, {2 * alternating_leg, -30, Choice::TurnRight},
            {3 * alternating_leg, 30, Choice::TurnLeft}, {3 * alternating_leg + scoring_controls}}),
        path(starts[3], {{alternating_leg, -30, Choice::TurnRight}, {2 * alternating_leg, 30, Choice::TurnLeft},
            {3 * alternating_leg, -30, Choice::TurnRight}, {3 * alternating_leg + scoring_controls}}),
        path(starts[0], {{straight_end, 0, Choice::Hold}}),
        path(stationary(starts[2], 0, .012f), {{straight_end, 0, Choice::Hold}}),
        path(stationary(starts[0], 0, -.014f), {{straight_end, -25}}),
        path(stationary(starts[1], 75, .014f), {{straight_end}})
    };
}

void new_directory(const fs::path& output) {
    if (!fs::create_directories(output)) throw std::runtime_error("Output directory must be new");
}

void verify_observation_and_progress(NavigationWorld probe) {
    const V progress_goal{.12f, 0, 0};
    const V previous{.035f, 0, 0};
    constexpr float distance = .0001f * tail::motor_control_s;
    const float approaching = tail::navigation_progress_m(previous, previous + V{distance, 0, 0}, progress_goal);
    const float retreating = tail::navigation_progress_m(previous, previous - V{distance, 0, 0}, progress_goal);
    const float still = tail::navigation_progress_m(previous, previous, progress_goal);
    if (std::abs(approaching - distance) > 2e-8f || std::abs(retreating + distance) > 2e-8f || still != 0) {
        throw std::runtime_error("Signed progress at 0.1 mm/s did not match measured motion");
    }

    probe.motion.body.orientation = physics::axis_rotation({0, 1, 0}, tail::pi / 2);
    const auto pose = tail::snapshot(probe.motion.body);
    const V local{.12f, -.06f, .09f};
    probe.legs[0].goal_m = pose.com_m + probe.motion.body.orientation.rotate(local);
    for (auto& point : probe.motion.history) point = pose.com_m;
    tail::prepare_navigation(probe);
    float heading[tail::navigation_inputs], destination[tail::navigation_inputs];
    tail::observe_navigation(probe, NavigationMode::Heading, heading);
    tail::observe_navigation(probe, NavigationMode::Destination, destination);
    const std::array<float, 3> components{local.x, local.y, local.z};
    if (!std::equal(heading, heading + tail::motor_inputs, destination) ||
        heading[19] != 0.f || heading[20] != 0.f) {
        throw std::runtime_error("Navigation arms disagree on shared inputs or stationary course");
    }
    for (size_t axis = 0; axis < components.size(); ++axis) {
        const size_t column = tail::motor_inputs + axis;
        if (heading[column] != 0.f ||
            std::abs(destination[column] - components[axis] / tail::navigation_position_scale_m) > 1e-6f) {
            throw std::runtime_error("Destination did not transform into the body frame");
        }
    }
    std::cout << "slow_progress_m=" << approaching << " slow_retreat_m=" << retreating
              << " stationary_progress_m=" << still << " observation_checks=pass\n";
}

void capture(const fs::path& source, const fs::path& output) {
    const auto train_starts = tail::read_turn_worlds((source / "train-worlds.bin").string());
    const auto eval_starts = tail::read_turn_worlds((source / "eval-worlds.bin").string());
    if (train_starts.size() != training_worlds || eval_starts.size() != 4) {
        throw std::runtime_error("Unexpected archived fixture dimensions");
    }
    new_directory(output);
    std::vector<NavigationWorld> training;
    training.reserve(train_starts.size());
    for (size_t i = 0; i < train_starts.size(); ++i) {
        training.push_back(training_path(train_starts[i], static_cast<int>(i % 16)));
    }
    const auto evaluation = evaluation_paths(eval_starts);
    verify_observation_and_progress(evaluation.front());
    tail::write_navigation_worlds((output / "train-worlds.bin").string(), training);
    tail::write_navigation_worlds((output / "eval-worlds.bin").string(), evaluation);
    std::cout << "train_worlds=" << training.size() << " eval_worlds=" << evaluation.size()
              << " new_physics_substeps=0\n";
}

constexpr float missing = std::numeric_limits<float>::quiet_NaN();

struct LegScore {
    bool started = false, completed = false, passed = false;
    float speed_mm_s = missing, course_error_deg = missing, actual_turn_deg = missing, radius_mm = missing;
    float goal_progress_mm = missing, goal_distance_mm = missing, minimum_goal_distance_mm = missing;
    float peak_tilt_deg = 0, peak_depth_mm = 0, min_gap_mm = 1000;
    float max_hold_speed_mm_s = 0, max_anchor_error_mm = 0, max_lateral_error_mm = 0;
    float min_axial_mm = 0, max_axial_mm = 0, reward = 0;
};

struct Score {
    std::array<LegScore, tail::navigation_max_legs> legs{};
    int attempted = 0, accepted = 0, complete_controls = 0, fault = 0;
    bool passed = false;
};

const char* choice_name(Choice choice) {
    switch (choice) {
    case Choice::Swim: return "swim";
    case Choice::TurnLeft: return "left";
    case Choice::TurnRight: return "right";
    case Choice::Hold: return "hold";
    }
    throw std::runtime_error("Invalid navigation choice");
}

const char* skill_name(Skill skill) {
    switch (skill) {
    case Skill::Straight: return "straight";
    case Skill::Turn: return "turn_and_onward";
    case Skill::Alternating: return "alternating_and_onward";
    case Skill::Brake: return "brake_and_hold";
    case Skill::Rest: return "stationary_hold";
    case Skill::Recovery: return "recovery";
    }
    throw std::runtime_error("Invalid evaluation skill");
}

bool moving_pass(const LegScore& score) {
    return score.speed_mm_s >= minimum_speed_mm_s && score.course_error_deg <= maximum_course_error_deg;
}

bool hold_pass(const LegScore& score) {
    return score.max_hold_speed_mm_s <= maximum_hold_speed_mm_s &&
        score.max_anchor_error_mm <= maximum_hold_error_mm &&
        score.max_lateral_error_mm <= maximum_hold_error_mm &&
        score.max_axial_mm - score.min_axial_mm <= maximum_hold_error_mm;
}

Score roll(NavigationWorld world, Actor& actor, NavigationMode mode, Clock::time_point deadline) {
    Score score;
    auto pose = tail::snapshot(world.motion.body);
    for (int leg_index = 0; leg_index < world.leg_count; ++leg_index) {
        auto& measured = score.legs[leg_index];
        const auto& leg = world.legs[leg_index];
        const int start_control = world.motion.controls;
        const int duration = leg.end_control - start_control;
        const bool holding = leg.choice == Choice::Hold;
        const bool turning = leg.choice == Choice::TurnLeft || leg.choice == Choice::TurnRight;
        const V initial_com = pose.com_m;
        const V incoming_course = tail::turn_course(world.motion, initial_com, tail::steering_history_controls);
        const float initial_distance = physics::norm(leg.goal_m - initial_com);
        measured.started = true;
        measured.goal_distance_mm = measured.minimum_goal_distance_mm = initial_distance * 1000.f;
        measured.goal_progress_mm = 0;
        measured.peak_tilt_deg = pose.tilt_deg;
        measured.peak_depth_mm = 1000.f * std::abs(leg.goal_m.y - initial_com.y);
        measured.min_gap_mm = static_cast<float>(1000 * tail::submerged_gap_m(larger_pond, world.motion.body, pose));
        V start_window = initial_com;
        V curve_sample = initial_com;
        float curve_length_m = 0;
        bool recorded_axial_range = false;
        while (world.motion.controls < leg.end_control) {
            const int control = world.motion.controls;
            if ((control & 15) == 0 && Clock::now() >= deadline) {
                score.fault = 4;
                break;
            }
            if (control == leg.end_control - scoring_controls) start_window = pose.com_m;
            tail::prepare_navigation(world);
            float observations[tail::navigation_inputs];
            tail::observe_navigation(world, mode, observations);
            actor.forward(observations, 1);
            const float targets[tail::motor_outputs]{actor.output()[0], actor.output()[1]};
            const auto step = tail::step_navigation(world, mode, targets, larger_pond);
            score.attempted += step.attempted_substeps;
            score.accepted += step.accepted_substeps;
            measured.reward += step.reward;
            measured.peak_tilt_deg = std::max(measured.peak_tilt_deg, step.max_tilt_deg);
            measured.peak_depth_mm = std::max(measured.peak_depth_mm, step.max_depth_error_m * 1000.f);
            measured.min_gap_mm = std::min(measured.min_gap_mm, step.min_gap_m * 1000.f);
            if (holding) {
                measured.max_lateral_error_mm = std::max(measured.max_lateral_error_mm,
                    1000.f * step.hold.peak_lateral_error_m);
                if (control - start_control >= braking_controls) {
                    const float min_axial_mm = 1000.f * step.hold.min_axial_error_m;
                    const float max_axial_mm = 1000.f * step.hold.max_axial_error_m;
                    measured.max_hold_speed_mm_s = std::max(measured.max_hold_speed_mm_s,
                        1000.f * step.hold.peak_speed_m_s);
                    measured.max_anchor_error_mm = std::max(measured.max_anchor_error_mm,
                        1000.f * step.hold.peak_anchor_error_m);
                    if (!recorded_axial_range) {
                        measured.min_axial_mm = min_axial_mm;
                        measured.max_axial_mm = max_axial_mm;
                        recorded_axial_range = true;
                    }
                    measured.min_axial_mm = std::min(measured.min_axial_mm, min_axial_mm);
                    measured.max_axial_mm = std::max(measured.max_axial_mm, max_axial_mm);
                }
            }
            if (step.fault) {
                score.fault = step.fault;
                break;
            }
            score.complete_controls = world.motion.controls;
            pose = tail::snapshot(world.motion.body);
            const V error = pose.com_m - leg.goal_m;
            measured.goal_distance_mm = 1000.f * physics::norm(error);
            measured.minimum_goal_distance_mm = std::min(measured.minimum_goal_distance_mm, measured.goal_distance_mm);
            measured.goal_progress_mm = 1000.f * initial_distance - measured.goal_distance_mm;
            if (turning && ((world.motion.controls - start_control) % tail::steering_history_controls == 0 ||
                            world.motion.controls == leg.end_control)) {
                curve_length_m += physics::norm(horizontal(pose.com_m - curve_sample));
                curve_sample = pose.com_m;
            }
        }
        if (score.fault) break;
        measured.completed = true;
        if (!holding && duration >= scoring_controls) {
            const V displacement = horizontal(pose.com_m - start_window);
            measured.speed_mm_s = 500.f * physics::dot(displacement, leg.heading);
            if (physics::norm(displacement) >= .001f) {
                measured.course_error_deg = std::abs(signed_angle_deg(leg.heading, displacement));
                if (turning && physics::norm(incoming_course) >= .0015f) {
                    measured.actual_turn_deg = signed_angle_deg(incoming_course, displacement);
                    // Mean radius implied by the 0.5 s chord path and signed net course change.
                    if (std::abs(measured.actual_turn_deg) >= 5.f) {
                        measured.radius_mm = 1000.f * curve_length_m /
                            std::abs(measured.actual_turn_deg * radians_per_degree);
                    }
                }
            }
        }
        const bool posture = measured.peak_tilt_deg <= maximum_tilt_deg &&
            measured.peak_depth_mm <= maximum_depth_error_mm && measured.min_gap_mm > 0;
        const bool turned_as_requested = !turning ||
            (leg.choice == Choice::TurnLeft ? measured.actual_turn_deg > 0.f : measured.actual_turn_deg < 0.f);
        const bool movement = duration < scoring_controls || (moving_pass(measured) && turned_as_requested);
        measured.passed = posture && (holding ? hold_pass(measured) : movement);
    }
    score.passed = score.fault == 0 &&
        std::all_of(score.legs.begin(), score.legs.begin() + world.leg_count,
            [](const LegScore& leg) { return leg.completed && leg.passed; });
    return score;
}

void csv_value(std::ostream& stream, float value) {
    if (std::isfinite(value)) stream << value;
    stream << ',';
}

void evaluate(const std::string& heading_path, const std::string& destination_path, const std::string& starts_path,
              const fs::path& output, Clock::time_point deadline) {
    const auto starts = tail::read_navigation_worlds(starts_path);
    const auto heading_weights = tail::read_navigation_weights(heading_path, NavigationMode::Heading);
    const auto destination_weights = tail::read_navigation_weights(destination_path, NavigationMode::Destination);
    if (starts.size() != std::size(evaluation_fixtures)) throw std::runtime_error("Unexpected held-out fixture count");
    new_directory(output);
    std::ofstream cases(output / "cases.csv"), legs(output / "legs.csv");
    cases.exceptions(std::ios::failbit | std::ios::badbit);
    legs.exceptions(std::ios::failbit | std::ios::badbit);
    cases << "case,skill,mode,pass,complete_controls,attempted_substeps,accepted_substeps,fault\n";
    legs << std::setprecision(8)
        << "case,mode,leg,request,goal_x_m,goal_y_m,goal_z_m,requested_delta_deg,scored,start,complete,pass,"
           "speed_mm_s,course_error_deg,actual_turn_deg,estimated_turn_radius_mm,progress_mm,final_goal_distance_mm,"
           "min_goal_distance_mm,peak_tilt_deg,peak_depth_error_mm,min_gap_mm,hold_peak_speed_mm_s,hold_peak_anchor_mm,"
           "hold_peak_lateral_mm,hold_axial_excursion_mm,reward\n";

    // Each worker has its own policy and world copy. Results are emitted in fixture order.
    std::vector<std::future<Score>> results;
    results.reserve(starts.size() * 2);
    for (const auto& start : starts) {
        for (const auto mode : {NavigationMode::Heading, NavigationMode::Destination}) {
            results.push_back(std::async(std::launch::async, [&, start, mode] {
                Actor actor(tail::navigation_motor_spec());
                actor.params() = mode == NavigationMode::Heading ? heading_weights : destination_weights;
                return roll(start, actor, mode, deadline);
            }));
        }
    }
    int attempted = 0, accepted = 0, faults = 0;
    std::array<int, 2> passes{};
    for (size_t i = 0; i < starts.size(); ++i) {
        const auto& start = starts[i];
        const auto& fixture = evaluation_fixtures[i];
        for (size_t arm = 0; arm < 2; ++arm) {
            const char* mode = arm == 0 ? "heading" : "destination";
            const auto score = results[i * 2 + arm].get();
            attempted += score.attempted;
            accepted += score.accepted;
            faults += score.fault != 0;
            passes[arm] += score.passed;
            cases << fixture.name << ',' << skill_name(fixture.skill) << ',' << mode << ',' << score.passed
                << ',' << score.complete_controls << ',' << score.attempted << ',' << score.accepted
                << ',' << score.fault << '\n';
            int start_control = 0;
            V previous_heading = start.motion.request.heading;
            for (int j = 0; j < start.leg_count; ++j) {
                const auto& request = start.legs[j];
                const auto& leg = score.legs[j];
                const bool scored = request.end_control - start_control >= scoring_controls;
                legs << fixture.name << ',' << mode << ',' << j << ',' << choice_name(request.choice)
                    << ',' << request.goal_m.x << ',' << request.goal_m.y << ',' << request.goal_m.z << ','
                    << signed_angle_deg(previous_heading, request.heading) << ',' << scored << ',' << leg.started
                    << ',' << leg.completed << ',' << leg.passed << ',';
                for (const float value : {leg.speed_mm_s, leg.course_error_deg, leg.actual_turn_deg, leg.radius_mm,
                         leg.goal_progress_mm, leg.goal_distance_mm, leg.minimum_goal_distance_mm}) {
                    csv_value(legs, value);
                }
                const bool holding = request.choice == Choice::Hold && leg.started;
                for (const float value : {leg.started ? leg.peak_tilt_deg : missing, leg.started ? leg.peak_depth_mm : missing,
                         leg.started ? leg.min_gap_mm : missing, holding ? leg.max_hold_speed_mm_s : missing,
                         holding ? leg.max_anchor_error_mm : missing, holding ? leg.max_lateral_error_mm : missing,
                         holding ? leg.max_axial_mm - leg.min_axial_mm : missing}) {
                    csv_value(legs, value);
                }
                if (leg.started) legs << leg.reward;
                legs << '\n';
                start_control = request.end_control;
                previous_heading = request.heading;
            }
            cases.flush();
            legs.flush();
        }
    }
    std::cout << "heading_passes=" << passes[0] << '/' << starts.size()
              << " destination_passes=" << passes[1] << '/' << starts.size()
              << " attempted_substeps=" << attempted << " accepted_substeps=" << accepted
              << " faults=" << faults << '\n';
    if (faults != 0) throw std::runtime_error("A held-out case stopped before finishing");
}
}

int main(int argc, char** argv) {
    try {
        const auto start = Clock::now();
        const auto deadline = start + std::chrono::milliseconds(3500);
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "capture" && argc == 4) capture(argv[2], argv[3]);
        else if (mode == "evaluate" && argc == 6) evaluate(argv[2], argv[3], argv[4], argv[5], deadline);
        else throw std::runtime_error("Usage: navigation-fixture capture <archived capture folder> <new folder> | "
            "evaluate <heading actor> <destination actor> <eval worlds> <new folder>");
        std::cout << "elapsed_s=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
