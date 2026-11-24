#include "rl/flexible_tail_launch.h"
#include "nn/mlp_cpu.h"
#include "habitat_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
namespace tail = cudarl::flexible_tail;
namespace physics = cudarl::physics;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr int controls_per_second = 50;
constexpr int controls_per_window = 2 * controls_per_second;
constexpr int total_controls = 6 * controls_per_window;
constexpr int trailing_controls = controls_per_second / 2;
constexpr float minimum_speed_mm_s = 6.f;
constexpr float reference_amplitude_rad = .55f;

enum class Policy { Learned, Reference80, Reference20 };
struct Start {
    const char* name;
    int index;
    float reference_phase_cycles;
    bool moving;
};
// Moving reference phases continue the 306-control captures in turn-fixture.cpp.
constexpr Start fixtures[] {
    {"neutral_a", 0, 0.f, false}, {"bent_left_a", 4, 0.f, false},
    {"moving_a", 8, 306 * tail::motor_control_s + .25f, true},
    {"moving_b", 9, 306 * tail::motor_control_s + .75f, true}
};
constexpr Policy policies[]{Policy::Learned, Policy::Reference80, Policy::Reference20};

const char* name(Policy policy) {
    switch (policy) {
    case Policy::Learned: return "learned_80ms";
    case Policy::Reference80: return "reference_80ms";
    case Policy::Reference20: return "reference_20ms";
    }
    throw std::logic_error("Invalid yaw trace policy");
}

struct ControlMetrics {
    int attempted = 0, accepted = 0, fault = 0;
    float maximum_tilt_deg = 0, maximum_depth_m = 0, minimum_gap_m = 1;
    double sum_abs_yaw_torque_Nm = 0, positive_yaw_work_J = 0;
};

// Diagnostic instrumentation around the same native substep, clearance test,
// and held request as step_turn_control. StepInfo supplies physical torque.
ControlMetrics trace_control(tail::TurnWorld& world, const float (&targets)[tail::motor_outputs]) {
    ControlMetrics result;
    auto pose = tail::snapshot(world.body);
    tail::record_turn_position(world, pose.com_m);
    for (int i = 0; i < tail::turn_substeps_per_control; ++i) {
        ++result.attempted;
        tail::StepInfo info;
        if (!tail::step(world.body, targets, tail::motor_substep_s, info)) {
            result.fault = 1;
            return result;
        }
        pose = tail::snapshot(world.body);
        const float gap = static_cast<float>(tail::submerged_gap_m(larger_pond, world.body, pose));
        result.minimum_gap_m = std::min(result.minimum_gap_m, gap);
        if (!(gap > 0)) {
            result.fault = 2;
            return result;
        }
        result.maximum_tilt_deg = std::max(result.maximum_tilt_deg, pose.tilt_deg);
        result.maximum_depth_m = std::max(result.maximum_depth_m,
            std::abs(world.request.origin_m.y - pose.com_m.y));
        result.sum_abs_yaw_torque_Nm += std::abs(info.torque_Nm[0]);
        result.positive_yaw_work_J += std::max(0., double(info.torque_Nm[0]) *
            world.body.velocity[6] * tail::motor_substep_s);
        ++result.accepted;
    }
    for (int axis = 0; axis < tail::motor_outputs; ++axis) world.last_targets[axis] = targets[axis];
    ++world.controls;
    return result;
}

struct Series {
    float low = 0, high = 0;
    double sum = 0, squares = 0;
    int count = 0;

    void add(float x) {
        low = count ? std::min(low, x) : x;
        high = count ? std::max(high, x) : x;
        sum += x;
        squares += double(x) * x;
        ++count;
    }

    float swing() const { return high - low; }
    double rms() const { return count ? std::sqrt(squares / count) : 0; }
    double mean() const { return count ? sum / count : 0; }
};

struct Outcome {
    int attempted = 0, accepted = 0, controls = 0, fault = 0;
};

Outcome compare(const Start& start, Policy policy, tail::NavigationWorld world,
                const std::vector<float>& weights, std::ofstream& trace, std::ofstream& windows,
                std::ofstream& summary, Clock::time_point deadline) {
    tail::prepare_navigation(world);
    auto& motion = world.motion;
    const auto heading = motion.request.heading;
    auto previous_com = tail::snapshot(motion.body).com_m;
    auto window_start = previous_com;
    cudarl::nn::CpuMlp<float> actor(tail::navigation_motor_spec());
    actor.params() = weights;
    const int hold = policy == Policy::Reference20 ? 1 : tail::launch_action_controls;
    float targets[tail::motor_outputs]{};
    Outcome result;
    int below = 0, first_sustained_mm_s_loss_control = 0;
    bool reached_speed = start.moving;
    Series target_yaw, root_yaw, distal_yaw, torque;
    double window_work = 0;
    float peak_tilt = 0, peak_depth_m = 0, minimum_gap_m = 1;
    trace << start.name << ',' << name(policy) << ",0,"
          << motion.last_targets[0] << ',' << motion.last_targets[1] << ',' << motion.body.angle[0] << ','
          << motion.body.velocity[6] << ',' << motion.body.angle[2] << ',' << motion.body.velocity[8]
          << ",0,0,0," << 1000.f * physics::dot(tail::snapshot(motion.body).com_velocity_m_s, heading)
          << ',' << tail::snapshot(motion.body).tilt_deg << ",0,"
          << 1000.f * tail::submerged_gap_m(larger_pond, motion.body, tail::snapshot(motion.body)) << '\n';
    for (int control = 0; control < total_controls; ++control) {
        if ((control & 15) == 0 && Clock::now() >= deadline) {
            result.fault = 4;
            break;
        }
        if (control % hold == 0) {
            if (policy == Policy::Learned) {
                float observations[tail::navigation_inputs];
                tail::observe_launch(world, observations);
                actor.forward(observations, 1);
                for (int axis = 0; axis < tail::motor_outputs; ++axis) targets[axis] = actor.output()[axis];
            } else {
                const float phase = (start.reference_phase_cycles + control * tail::motor_control_s) * 2 * tail::pi;
                targets[0] = reference_amplitude_rad * std::sin(phase);
            }
        }
        const auto step = trace_control(motion, targets);
        result.attempted += step.attempted;
        result.accepted += step.accepted;
        if (step.fault) {
            result.fault = step.fault;
            break;
        }
        peak_tilt = std::max(peak_tilt, step.maximum_tilt_deg);
        peak_depth_m = std::max(peak_depth_m, step.maximum_depth_m);
        minimum_gap_m = std::min(minimum_gap_m, step.minimum_gap_m);
        const auto pose = tail::snapshot(motion.body);
        result.controls = motion.controls;
        const float seconds = result.controls * tail::motor_control_s;
        const float forward_mm_s = 1000.f * physics::dot(pose.com_m - previous_com, heading) / tail::motor_control_s;
        const float trailing_mm_s = 2000.f * physics::dot(tail::turn_course(motion, pose.com_m, trailing_controls), heading);
        if (result.controls >= trailing_controls || start.moving) {
            if (trailing_mm_s >= minimum_speed_mm_s) {
                reached_speed = true;
                below = 0;
            } else if (reached_speed) {
                ++below;
                if (below == trailing_controls && first_sustained_mm_s_loss_control == 0)
                    first_sustained_mm_s_loss_control = result.controls - below + 1;
            }
        }
        const float mean_abs_yaw_torque_uNm = static_cast<float>(step.sum_abs_yaw_torque_Nm / step.accepted * 1e6);
        target_yaw.add(targets[0]);
        root_yaw.add(motion.body.angle[0]);
        distal_yaw.add(motion.body.angle[2]);
        torque.add(mean_abs_yaw_torque_uNm);
        window_work += step.positive_yaw_work_J;
        trace << start.name << ',' << name(policy) << ',' << seconds << ',' << targets[0] << ','
              << targets[1] << ',' << motion.body.angle[0] << ',' << motion.body.velocity[6] << ','
              << motion.body.angle[2] << ',' << motion.body.velocity[8] << ',' << mean_abs_yaw_torque_uNm
              << ',' << 1e6 * step.positive_yaw_work_J << ',' << forward_mm_s << ',' << trailing_mm_s
              << ',' << step.maximum_tilt_deg << ',' << 1000.f * step.maximum_depth_m << ','
              << 1000.f * step.minimum_gap_m << '\n';
        if (result.controls % controls_per_window == 0) {
            const auto displacement = pose.com_m - window_start;
            const float course_deg = std::abs(std::atan2(physics::cross(heading, displacement).y,
                physics::dot(heading, displacement))) * (180.f / tail::pi);
            windows << start.name << ',' << name(policy) << ',' << seconds << ','
                    << 500.f * physics::dot(displacement, heading) << ',' << course_deg << ','
                    << target_yaw.swing() << ',' << target_yaw.rms() << ',' << root_yaw.swing() << ','
                    << root_yaw.rms() << ',' << distal_yaw.swing() << ',' << distal_yaw.rms() << ','
                    << torque.mean() << ',' << 1e6 * window_work << '\n';
            target_yaw = {};
            root_yaw = {};
            distal_yaw = {};
            torque = {};
            window_work = 0;
            window_start = pose.com_m;
        }
        previous_com = pose.com_m;
    }
    summary << start.name << ',' << name(policy) << ',' << result.controls << ',' << result.fault << ','
            << result.attempted << ',' << result.accepted << ',' << reached_speed << ',';
    if (first_sustained_mm_s_loss_control)
        summary << first_sustained_mm_s_loss_control * tail::motor_control_s;
    summary << ',' << peak_tilt << ',' << 1000.f * peak_depth_m << ',' << 1000.f * minimum_gap_m << '\n';
    summary.flush();
    windows.flush();
    trace.flush();
    return result;
}
}

int main(int argc, char** argv) {
    try {
        const auto start = Clock::now();
        const auto deadline = start + std::chrono::milliseconds(3500);
        if (argc != 4) throw std::runtime_error("Usage: yaw-trace <launch actor> <eval worlds> <new folder>");
        const auto weights = tail::read_navigation_weights(argv[1], tail::NavigationMode::Launch);
        const auto worlds = tail::read_navigation_worlds(argv[2]);
        if (worlds.size() != 12) throw std::runtime_error("Expected the original 12 held-out worlds");
        const fs::path output(argv[3]);
        if (!fs::create_directory(output)) throw std::runtime_error("Output directory must be new");
        std::ofstream trace(output / "trace.csv"), windows(output / "windows.csv"), summary(output / "cases.csv");
        for (auto* csv : {&trace, &windows, &summary}) {
            csv->exceptions(std::ios::failbit | std::ios::badbit);
            *csv << std::setprecision(8);
        }
        trace << "case,policy,time_s,target_yaw_rad,target_pitch_rad,root_yaw_rad,root_yaw_rate_rad_s,"
                 "passive_yaw_rad,passive_yaw_rate_rad_s,abs_yaw_torque_mean_uNm,positive_yaw_work_uJ,"
                 "forward_mm_s,trailing_half_second_mm_s,peak_tilt_deg,peak_depth_error_mm,min_gap_mm\n";
        windows << "case,policy,end_s,forward_mm_s,course_error_deg,target_yaw_swing_rad,target_yaw_rms_rad,"
                   "root_yaw_swing_rad,root_yaw_rms_rad,passive_yaw_swing_rad,passive_yaw_rms_rad,"
                   "abs_yaw_torque_mean_uNm,positive_yaw_work_uJ\n";
        summary << "case,policy,controls,fault,attempted_substeps,accepted_substeps,reached_6_mm_s,"
                   "first_sustained_speed_loss_s,peak_tilt_deg,peak_depth_error_mm,min_gap_mm\n";
        int attempted = 0, accepted = 0, faults = 0;
        for (const auto& fixture : fixtures) {
            for (const auto policy : policies) {
                const auto run = compare(fixture, policy, worlds[fixture.index], weights, trace, windows, summary, deadline);
                attempted += run.attempted;
                accepted += run.accepted;
                faults += run.fault != 0;
            }
        }
        std::cout << "cases=12 attempted_substeps=" << attempted << " accepted_substeps=" << accepted
                  << " faults=" << faults << " elapsed_s=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
        return faults ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
