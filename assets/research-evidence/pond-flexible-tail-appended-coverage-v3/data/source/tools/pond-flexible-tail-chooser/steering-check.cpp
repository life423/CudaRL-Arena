#include "habitat_config.h"
#include "rl/flexible_tail_chooser.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace tail = cudarl::flexible_tail;
namespace chooser = tail::chooser;
namespace physics = cudarl::physics;
namespace fs = std::filesystem;
using physics::V;
using Clock = std::chrono::steady_clock;

constexpr int saved_starts = 8;
constexpr int horizon_controls = 48 * 50;
constexpr int repeated_requests = 6;
constexpr int sample_controls = 25;
constexpr int substep_cap = 2 * 2 * saved_starts * horizon_controls * tail::turn_substeps_per_control;
static_assert(substep_cap == 1228800);

const char* name(chooser::SteeringMode mode) {
    return mode == chooser::SteeringMode::PersistentHeading ? "held-heading" : "measured-course";
}

float degrees(float radians) { return radians * (180.f / tail::pi); }

float signed_angle(V from, V toward) {
    from.y = toward.y = 0.f;
    if (physics::norm(from) < .001f || physics::norm(toward) < .001f)
        return std::numeric_limits<float>::quiet_NaN();
    return degrees(std::atan2(physics::cross(from, toward).y, physics::dot(from, toward)));
}

void number(std::ostream& out, double value) {
    if (std::isfinite(value)) out << value;
}

uint64_t physical_digest(const chooser::World& world) {
    cudarl::checkpoint::Writer out;
    out.value(world.navigation);
    out.value(world.beats);
    out.value(world.targets);
    out.value(world.applied_offset);
    return cudarl::checkpoint::checksum(out.data);
}

struct Case {
    chooser::SteeringMode mode{};
    int row = 0, sign = 0, controls = 0, attempted = 0, accepted = 0, fault = 0;
    int requests = 0, decisions = 0;
    double cue_s = 0, turn_end_s = 0;
    uint64_t cue_digest = 0;
    float minimum_gap_m = 1, maximum_tilt_deg = 0, maximum_depth_m = 0;
    float minimum_speed_m_s = 1;
    float path_m = 0, turning_path_m = 0, turning_course_deg = 0;
    float achieved_deg = 0, target_deg = 0, course_error_deg = 0, final_speed_m_s = 0;
    float first_60_s = -1, first_90_s = -1;
    bool complete = false;
};

Case run_case(const tail::NavigationWorld& source, int row, int sign, chooser::SteeringMode mode,
              const std::vector<float>& gait, const std::vector<float>& offset, const fs::path& output,
              Clock::time_point deadline) {
    Case result{};
    result.mode = mode;
    result.row = row;
    result.sign = sign;
    const std::string id = std::string(name(mode)) + "-row-" + std::to_string(row) +
        (sign > 0 ? "-left" : "-right");
    std::ofstream trace(output / "traces" / (id + ".csv"));
    trace.exceptions(std::ios::failbit | std::ios::badbit);
    trace << std::setprecision(10)
          << "control,time_s,event,action,requests,x_m,y_m,z_m,heading_x,heading_z,"
             "actual_course_change_deg,requested_course_change_deg,course_error_deg,"
             "rolling_speed_mm_s,clearance_bound_mm,tilt_deg,depth_error_mm,applied_yaw_trim,fault\n";

    chooser::World world{};
    chooser::initialize(world, source, chooser::horizontal(tail::snapshot(source.motion.body).com_m), 0.f, mode);
    const float depth_y = world.navigation.motion.request.origin_m.y;
    V approach{}, previous_position = world.pose.com_m;
    const auto course_now = [&]() {
        return tail::turn_course(world.navigation.motion, world.pose.com_m, tail::turn_history_controls);
    };
    const auto speed_now = [&]() {
        return physics::norm(tail::turn_course(world.navigation.motion, world.pose.com_m,
            tail::steering_history_controls)) / tail::launch_course_window_s;
    };
    const auto write_trace = [&](const char* event, int action = -1) {
        const auto& motion = world.navigation.motion;
        const auto heading = motion.request.heading;
        const auto com = world.pose.com_m;
        trace << result.controls << ',' << double(result.accepted) * tail::motor_substep_s << ',' << event << ',';
        if (action >= 0) trace << action;
        trace << ',' << result.requests << ',' << com.x << ',' << com.y << ',' << com.z << ','
              << heading.x << ',' << heading.z << ',';
        number(trace, signed_angle(approach, course_now()));
        trace << ',';
        number(trace, signed_angle(approach, heading));
        trace << ',';
        number(trace, std::abs(signed_angle(course_now(), heading)));
        trace << ',' << speed_now() * 1000.f << ',' << world.gap_m * 1000.f << ',' << world.pose.tilt_deg << ','
              << std::abs(depth_y - com.y) * 1000.f << ',' << world.applied_offset << ',' << world.fault << '\n';
    };

    world.gap_m = static_cast<float>(tail::submerged_gap_m(larger_pond, world.navigation.motion.body, world.pose));
    result.minimum_gap_m = world.gap_m;
    if (!(world.gap_m > 0)) world.fault = chooser::Clearance;
    write_trace("start");
    while (result.controls < horizon_controls && world.fault == chooser::Healthy) {
        if (Clock::now() >= deadline) {
            result.fault = 9;
            break;
        }
        if (chooser::ready(world)) {
            if (result.decisions == 0) {
                approach = chooser::decision_course(world);
                previous_position = world.pose.com_m;
                result.cue_s = result.controls * double(tail::motor_control_s);
                result.cue_digest = physical_digest(world);
            }
            // Explicit diagnostic stimuli. These action labels are never used
            // by PPO training or deployed autonomous navigation.
            const int action = result.requests < repeated_requests ? (sign > 0 ? 1 : 2) : 0;
            if (result.requests == repeated_requests && result.turn_end_s == 0.) {
                result.turn_end_s = result.controls * double(tail::motor_control_s);
                result.turning_course_deg = signed_angle(approach, course_now());
                result.turning_path_m = result.path_m;
            }
            if (!chooser::accept(world, action)) {
                world.fault = chooser::InvalidUse;
                break;
            }
            ++result.decisions;
            if (action != 0) ++result.requests;
            write_trace("decision", action);
        }
        const auto step = chooser::advance(world, gait.data(), offset.data(), larger_pond, 1);
        result.attempted += step.attempted_substeps;
        result.accepted += step.accepted_substeps;
        result.controls += step.controls;
        result.minimum_gap_m = std::min(result.minimum_gap_m, world.min_gap_m);
        result.maximum_tilt_deg = std::max(result.maximum_tilt_deg, world.peak_tilt_deg);
        result.maximum_depth_m = std::max(result.maximum_depth_m, world.peak_depth_error_m);
        if (step.fault != chooser::Healthy || step.controls != 1) {
            result.fault = step.fault ? step.fault : chooser::InvalidUse;
            break;
        }
        if (result.requests > 0) {
            const float speed = speed_now();
            result.minimum_speed_m_s = std::min(result.minimum_speed_m_s, speed);
            const float achieved = signed_angle(approach, course_now()) * float(sign);
            const float seconds = float(result.controls) * tail::motor_control_s - float(result.cue_s);
            if (result.first_60_s < 0 && achieved >= 60.f) result.first_60_s = seconds;
            if (result.first_90_s < 0 && achieved >= 90.f) result.first_90_s = seconds;
            if (result.controls % sample_controls == 0) {
                result.path_m += physics::norm(chooser::horizontal(world.pose.com_m - previous_position));
                previous_position = world.pose.com_m;
            }
        }
        if (result.controls % sample_controls == 0) write_trace("sample");
    }
    if (result.fault == 0) result.fault = world.fault;
    if (result.turn_end_s == 0.) {
        result.turning_path_m = result.path_m;
        result.turning_course_deg = signed_angle(approach, course_now());
    }
    result.achieved_deg = signed_angle(approach, course_now());
    result.target_deg = signed_angle(approach, world.navigation.motion.request.heading);
    result.course_error_deg = std::abs(signed_angle(course_now(), world.navigation.motion.request.heading));
    result.final_speed_m_s = speed_now();
    result.complete = result.controls == horizon_controls && result.fault == 0;
    write_trace(result.complete ? "complete" : "stop");
    return result;
}

void write_case(std::ostream& out, const Case& row) {
    const float radius = std::abs(row.turning_course_deg) >= 5.f
        ? row.turning_path_m / (std::abs(row.turning_course_deg) * tail::pi / 180.f)
        : std::numeric_limits<float>::quiet_NaN();
    out << name(row.mode) << ',' << row.row << ',' << (row.sign > 0 ? "left" : "right") << ','
        << row.controls * double(tail::motor_control_s) << ',' << row.complete << ',' << row.fault << ','
        << row.attempted << ',' << row.accepted << ',' << row.requests << ',' << row.decisions << ','
        << row.cue_s << ',' << std::hex << row.cue_digest << std::dec << ',';
    number(out, row.achieved_deg);
    out << ',';
    number(out, row.target_deg);
    out << ',';
    number(out, row.course_error_deg);
    out << ',';
    if (row.first_60_s >= 0) out << row.first_60_s;
    out << ',';
    if (row.first_90_s >= 0) out << row.first_90_s;
    out << ',' << row.path_m * 1000.f << ',' << row.turning_course_deg << ',';
    number(out, radius * 1000.f);
    out << ',' << row.minimum_speed_m_s * 1000.f << ',' << row.final_speed_m_s * 1000.f << ','
        << row.maximum_tilt_deg << ',' << row.maximum_depth_m * 1000.f << ','
        << row.minimum_gap_m * 1000.f << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto begin = Clock::now();
        if (argc != 5) throw std::runtime_error("Usage: steering_check FITTED_GAIT UPDATE8_OFFSET EVAL_TURN_WORLDS NEW_OUTPUT");
        const auto gait = tail::read_navigation_weights(argv[1], tail::NavigationMode::Gait);
        const auto offset = tail::read_yaw_offset_weights(argv[2]);
        const auto starts = tail::read_navigation_worlds(argv[3]);
        chooser::validate_frozen_layout(gait, offset);
        if (starts.size() != saved_starts) throw std::runtime_error("Expected eight archived moving turn starts");
        for (const auto& start : starts) {
            if (physics::norm(tail::snapshot(start.motion.body).com_velocity_m_s) < .006f)
                throw std::runtime_error("A diagnostic start is not a moving swimming state");
        }
        const auto output = fs::u8path(argv[4]);
        if (!fs::create_directory(output)) throw std::runtime_error("Steering diagnostic output already exists");
        fs::create_directory(output / "traces");
        const auto deadline = begin + std::chrono::seconds(55);
        std::vector<std::future<Case>> pending;
        for (const auto mode : {chooser::SteeringMode::MeasuredCourse, chooser::SteeringMode::PersistentHeading}) {
            for (int row = 0; row < saved_starts; ++row) {
                for (const int sign : {1, -1}) {
                    pending.push_back(std::async(std::launch::async, [&, row, sign, mode] {
                        return run_case(starts[row], row, sign, mode, gait, offset, output, deadline);
                    }));
                }
            }
        }
        std::ofstream table(output / "cases.csv"), summary(output / "summary.json");
        table.exceptions(std::ios::failbit | std::ios::badbit);
        summary.exceptions(std::ios::failbit | std::ios::badbit);
        table << std::setprecision(10)
            << "interface,row,direction,simulated_s,complete,fault,attempted_substeps,accepted_substeps,"
               "turn_requests,decisions,first_cue_s,first_cue_state_checksum,actual_final_course_deg,"
               "final_requested_course_deg,final_course_error_deg,first_actual_60_s,first_actual_90_s,"
               "distance_after_cue_mm,turning_course_deg,effective_turning_radius_mm,minimum_rolling_speed_mm_s,"
               "final_rolling_speed_mm_s,peak_tilt_deg,peak_depth_error_mm,minimum_clearance_mm\n";
        int attempted = 0, accepted = 0, measured_60 = 0, persistent_60 = 0, faults = 0, incomplete = 0;
        std::array<uint64_t, 2 * saved_starts> matched_digests{};
        size_t index = 0;
        for (auto& future : pending) {
            const auto row = future.get();
            write_case(table, row);
            attempted += row.attempted;
            accepted += row.accepted;
            if (row.fault != 0) ++faults;
            if (!row.complete) ++incomplete;
            if (row.first_60_s >= 0) {
                if (row.mode == chooser::SteeringMode::MeasuredCourse) ++measured_60;
                else ++persistent_60;
            }
            const size_t pair = index % matched_digests.size();
            if (index < matched_digests.size()) matched_digests[pair] = row.cue_digest;
            else if (row.cue_digest != matched_digests[pair])
                throw std::runtime_error("Steering arms reached different physical first-cue states");
            ++index;
        }
        if (attempted > substep_cap) throw std::runtime_error("Steering diagnostic substep cap exceeded");
        summary << std::setprecision(10) << "{\"trajectories\":" << pending.size()
                << ",\"attempted_substeps\":" << attempted << ",\"accepted_substeps\":" << accepted
                << ",\"substep_cap\":" << substep_cap << ",\"measured_actual_60\":" << measured_60
                << ",\"held_actual_60\":" << persistent_60 << ",\"faulted_trajectories\":" << faults
                << ",\"incomplete_trajectories\":" << incomplete << ",\"first_cue_state_pairs_matched\":16"
                << ",\"optimizer_updates\":0,\"runtime_s\":"
                << std::chrono::duration<double>(Clock::now() - begin).count() << "}\n";
        std::cout << "measured_actual_60=" << measured_60 << "/16 held_actual_60=" << persistent_60
                  << "/16 attempted_substeps=" << attempted << " accepted_substeps=" << accepted << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
