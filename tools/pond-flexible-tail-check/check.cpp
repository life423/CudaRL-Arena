#include "environment/pond_geometry_v1.h"
#include "environment/flexible_tail_clearance.h"
#include "habitat_config.h"
#include "physics/flexible_tail.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace physics = cudarl::physics;
namespace tail = cudarl::flexible_tail;
namespace geometry = cudarl::pond_geometry_v1;
namespace fs = std::filesystem;

namespace {
using Clock = std::chrono::steady_clock;
using physics::BodyState;
using physics::V;
constexpr float pi = 3.14159265358979323846f;
constexpr float rad = pi / 180;
constexpr float drive_s = 6;
constexpr float turn_s = 8;
constexpr float rest_s = 6;
constexpr float step_s = .005f;
constexpr float comparison_step_s = .00125f;
constexpr float course_speed_floor_m_s = .006f;
constexpr float max_depth_error_m = .001f;
constexpr float max_tilt_deg = 10;
constexpr auto pond = geometry::current_visual_parameters();

struct Stroke {
    float frequency_hz, amplitude_rad, pitch_rad;
};

struct Frame {
    float time_s;
    V head_m;
    physics::Q head;
    tail::Snapshot pose;
};

struct Outcome {
    const char* stop = "";
    float projected_speed_m_s = 0;
    float course_error_deg = 180;
    float measured_course_rad = 0;
    float course_change_deg = 0;
    float max_tilt = 0;
    float max_depth_error = 0;
    float clearance_m = 1;
    float distance_m = 0;
    float max_yaw_bend_rad = 0;
    float max_torque_nm = 0;
    double work_j = 0;
    int steps = 0;
    BodyState state;
    std::vector<Frame> frames;
};

float course(V v) { return std::atan2(-v.z, v.x); }

float wrap(float angle) { return std::remainder(angle, 2 * pi); }

float horizontal_speed(V v) { return std::hypot(v.x, v.z); }

float forward(V v, float angle) {
    return v.x * std::cos(angle) - v.z * std::sin(angle);
}

// The central bed is flat and its ellipse is convex. An envelope about each
// tail endpoint also encloses the connecting stations and their cross sections.
// A fixture leaving this envelope cannot establish clear-water locomotion.
float clearance(const BodyState& body, const tail::Snapshot& pose) {
    float minimum = 1;
    const auto test = [&](V point, float radius) {
        const double far_x = (std::abs(point.x) + radius) / pond.flat_radius_x_m;
        const double far_z = (std::abs(point.z) + radius) / pond.flat_radius_z_m;
        if (std::hypot(far_x, far_z) >= 1) return false;
        const float bed_gap = point.y - radius - static_cast<float>(pond.floor_y_m);
        const float surface_gap = -point.y - radius;
        minimum = std::min({minimum, bed_gap, surface_gap});
        return minimum > 0;
    };
    if (!test(body.position * physics::scale, .011f)) return -1;
    for (const V point : pose.spine_m) {
        if (!test(point, .012f)) return -1;
    }
    return minimum;
}

void record(Outcome& result, const BodyState& body, const tail::Snapshot& pose, float time_s) {
    result.frames.push_back({time_s, body.position * physics::scale, body.orientation, pose});
}

enum class Lesson { Forward, Turn, Coast };

struct SteeringHistory {
    std::array<V, 10> positions{};
    std::size_t samples = 0;
    float bias = 0;
};

Outcome simulate(BodyState body, const Stroke stroke, Lesson lesson, float target_course,
                 float duration, float timestep, double start_s, bool save_frames, float entry_course,
                 float steering_gain,
                 Clock::time_point deadline, const geometry::Parameters* habitat = nullptr,
                 SteeringHistory* continued_steering = nullptr) {
    Outcome result;
    result.state = body;
    auto pose = tail::snapshot(body);
    const V entry_com = pose.com_m;
    if (save_frames) record(result, body, pose, 0);
    const int total = static_cast<int>(std::lround(duration / timestep));
    const int window = static_cast<int>(std::lround(2 / timestep));
    const int samples_per_frame = static_cast<int>(std::lround(.05f / timestep));
    V final_window_start = entry_com;
    int final_window_tick = 0;
    SteeringHistory local_steering;
    auto& steering = continued_steering ? *continued_steering : local_steering;
    if (steering.samples == 0) steering.positions.fill(entry_com);
    for (int tick = 0; tick < total; ++tick) {
        if ((tick & 63) == 0 && Clock::now() >= deadline) {
            result.stop = "time_budget";
            break;
        }
        if (tick == total - window) {
            final_window_start = pose.com_m;
            final_window_tick = tick;
        }
        if (lesson == Lesson::Turn && tick % samples_per_frame == 0) {
            auto& recent = steering.positions[steering.samples++ % steering.positions.size()];
            const V displacement = pose.com_m - recent;
            recent = pose.com_m;
            const float measured_course = horizontal_speed(displacement) >= .0015f ?
                course(displacement) : entry_course;
            const float error = wrap(target_course - measured_course);
            const float alpha = std::min(1.f, .05f / .5f);
            // Diagnostic feedback on the root target exercises physical
            // steering authority. The learner and live app never call this.
            steering.bias += alpha * (std::clamp(steering_gain * error, -.45f, .45f) - steering.bias);
        }
        const double time_s = start_s + tick * static_cast<double>(timestep);
        const float drive = lesson == Lesson::Coast ? 0.f :
            stroke.amplitude_rad * static_cast<float>(std::sin(2 * pi * stroke.frequency_hz * time_s));
        const float pitch = lesson == Lesson::Coast ? 0.f : stroke.pitch_rad;
        const float targets[]{std::clamp(drive + steering.bias, -physics::limit, physics::limit), pitch};
        tail::StepInfo info;
        if (!tail::step(body, targets, timestep, info)) {
            result.stop = "invalid_dynamics";
            break;
        }
        ++result.steps;
        result.state = body;
        pose = tail::snapshot(body);
        result.max_yaw_bend_rad = std::max(result.max_yaw_bend_rad, std::abs(body.angle[2]));
        result.max_torque_nm = std::max({result.max_torque_nm, std::abs(info.torque_Nm[0]),
            std::abs(info.torque_Nm[1])});
        result.work_j += info.root_work_J;
        result.max_tilt = std::max(result.max_tilt, pose.tilt_deg);
        result.max_depth_error = std::max(result.max_depth_error, std::abs(pose.depth_m - .08f));
        const float gap = habitat ? static_cast<float>(tail::submerged_gap_m(*habitat, body, pose)) :
            clearance(body, pose);
        result.clearance_m = std::min(result.clearance_m, gap);
        if (!(result.clearance_m > 0)) {
            result.stop = habitat ? "submergence_not_certified" : "left_clear_water_fixture";
            break;
        }
        if (save_frames && (tick + 1) % samples_per_frame == 0) {
            record(result, body, pose, (tick + 1) * timestep);
        }
    }
    const V delta = pose.com_m - final_window_start;
    result.distance_m = horizontal_speed(pose.com_m - entry_com);
    const float measured_s = static_cast<float>(result.steps - final_window_tick) * timestep;
    if (measured_s > 0) result.projected_speed_m_s = forward(delta, target_course) / measured_s;
    if (horizontal_speed(delta) >= .001f) {
        result.measured_course_rad = course(delta);
        result.course_error_deg = std::abs(wrap(result.measured_course_rad - target_course)) / rad;
        result.course_change_deg = wrap(course(delta) - entry_course) / rad;
    }
    return result;
}

bool completed(const Outcome& run) { return !*run.stop; }

bool movement_pass(const Outcome& run) {
    return completed(run) && run.projected_speed_m_s >= course_speed_floor_m_s &&
        run.course_error_deg <= 5 && run.max_depth_error <= max_depth_error_m &&
        run.max_tilt <= max_tilt_deg;
}

void write_case(std::ostream& out, const char* mode, const Stroke stroke,
                float degrees, const Outcome& x, bool pass) {
    out << mode << ',' << stroke.frequency_hz << ',' << stroke.amplitude_rad << ',' << stroke.pitch_rad
        << ',' << degrees << ',' << x.steps << ',' << x.stop << ',' << pass
        << ',' << x.projected_speed_m_s * 1000 << ',' << x.course_error_deg
        << ',' << x.course_change_deg << ',' << x.max_tilt << ',' << x.max_depth_error * 1000
        << ',' << x.clearance_m * 1000 << ',' << x.distance_m * 1000
        << ',' << x.max_yaw_bend_rad << ',' << x.max_torque_nm * 1e6 << ',' << x.work_j << '\n';
}

void write_movement_columns(std::ostream& csv) {
    csv << "mode,frequency_Hz,amplitude_rad,pitch_rad,relative_target_deg,steps,stop,pass,"
        << "projected_speed_mm_s,course_error_deg,course_change_deg,peak_tilt_deg,max_depth_error_mm,"
        << "min_clearance_mm,horizontal_displacement_mm,max_yaw_bend_rad,max_torque_uNm,root_work_J\n";
}

std::ostream& xyz(std::ostream& out, V v) {
    return out << '[' << v.x << ',' << v.y << ',' << v.z << ']';
}

void write_trace(const fs::path& file, const std::vector<Frame>& frames,
                 const std::vector<Frame>& continuation, const V* anchor = nullptr,
                 float brake_at = 0, float hold_at = 0, const char* hold_label = "Hold check") {
    std::ofstream out(file);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::setprecision(9) << "{\"schema\":1,\"scope\":\"diagnostic_root_joint\"";
    if (anchor) {
        out << ",\"anchor_m\":";
        xyz(out, *anchor);
        out << ",\"phases\":[";
        if (brake_at > 0) out << "{\"time_s\":0,\"label\":\"Swim\"},";
        if (hold_at > brake_at) {
            out << "{\"time_s\":" << brake_at << ",\"label\":\"Brake check\"},";
        }
        out << "{\"time_s\":" << hold_at << ",\"label\":\"" << hold_label << "\"}]";
    }
    out << ",\"frames\":[\n";
    const auto write_frame = [&](const Frame& frame, float time_s) {
        out << "{\"time_s\":" << time_s << ",\"body_position_m\":";
        xyz(out, frame.head_m);
        const auto q = frame.head;
        out << ",\"body_quaternion\":[" << q.x << ',' << q.y << ',' << q.z << ',' << q.w << "],\"spine_m\":[";
        for (int point = 0; point < 7; ++point) {
            if (point) out << ',';
            xyz(out, frame.pose.spine_m[point]);
        }
        out << "],\"com_m\":";
        xyz(out, frame.pose.com_m);
        out << ",\"com_velocity_m_s\":";
        xyz(out, frame.pose.com_velocity_m_s);
        out << '}';
    };
    for (std::size_t frame = 0; frame < frames.size(); ++frame) {
        if (frame) out << ",\n";
        write_frame(frames[frame], frames[frame].time_s);
    }
    for (std::size_t frame = 1; frame < continuation.size(); ++frame) {
        out << ",\n";
        write_frame(continuation[frame], frames.back().time_s + continuation[frame].time_s);
    }
    out << "\n]}\n";
}

struct PhysicsChecks {
    bool valid = false;
    float rest_travel_mm = 0;
    float passive_bend_remaining = 0;
    double passive_energy_ratio = 0;
    double max_linear_momentum = 0;
    double max_angular_momentum = 0;
};

PhysicsChecks check_physics(Clock::time_point deadline) {
    PhysicsChecks checks;
    BodyState rest;
    const auto initial = tail::snapshot(rest);
    tail::StepInfo info;
    for (int i = 0; i < 400; ++i) {
        if (((i & 63) == 0 && Clock::now() >= deadline) ||
            !tail::step(rest, nullptr, step_s, info) || info.root_work_J != 0) return checks;
    }
    checks.rest_travel_mm = 1000 * physics::norm(tail::snapshot(rest).com_m - initial.com_m);
    BodyState bend;
    bend.angle[2] = .4f;
    const double initial_energy = tail::elastic_energy(bend);
    for (int i = 0; i < 400; ++i) {
        if (((i & 63) == 0 && Clock::now() >= deadline) ||
            !tail::step(bend, nullptr, step_s, info, {0, 0, 0}) || info.root_work_J != 0) return checks;
        const auto pose = tail::snapshot(bend);
        checks.max_linear_momentum = std::max(checks.max_linear_momentum,
            static_cast<double>(physics::norm(pose.linear_momentum_kg_m_s)));
        checks.max_angular_momentum = std::max(checks.max_angular_momentum,
            static_cast<double>(physics::norm(pose.angular_momentum_about_com_kg_m2_s)));
    }
    checks.passive_energy_ratio = (tail::elastic_energy(bend) + tail::kinetic_energy(bend)) / initial_energy;
    checks.passive_bend_remaining = std::abs(bend.angle[2]);
    checks.valid = checks.rest_travel_mm <= .5f && checks.passive_bend_remaining <= .04f &&
        checks.passive_energy_ratio < .1 && checks.max_linear_momentum < 5e-7 &&
        checks.max_angular_momentum < 5e-9;
    return checks;
}

constexpr float brake_time_s = 4;
constexpr float quiet_time_s = 8;
constexpr float hold_time_s = brake_time_s + quiet_time_s;
constexpr float quiet_speed_m_s = .0005f;
constexpr float anchor_radius_m = .002f;

enum class Brake { Disconnected, Neutral, Fold, Flick };

struct BrakeStroke {
    Brake mode;
    float yaw = 0, outward_s = 0, return_s = 0;
};

struct HoldScore {
    const char* fault = "";
    int substeps = 0;
    float brake_speed = 0, end_speed = 0, peak_quiet_speed = 0;
    float peak_anchor_error = 0, quiet_anchor_error = 0, hold_axial_excursion = 0;
    float depth_error = 0, peak_tilt = 0, lateral_error = 0, quiet_streak_s = 0;
    float min_clearance = 1, peak_torque = 0;
    double positive_work_j = 0;
    Outcome replay;
};

bool hold_pass(const HoldScore& score) {
    return !*score.fault && score.peak_quiet_speed <= quiet_speed_m_s &&
        score.quiet_anchor_error <= anchor_radius_m && score.hold_axial_excursion <= anchor_radius_m &&
        score.depth_error <= max_depth_error_m && score.peak_tilt <= max_tilt_deg &&
        score.lateral_error <= anchor_radius_m;
}

float braking_yaw(const BrakeStroke& stroke, const float time) {
    if (stroke.mode == Brake::Fold) return stroke.yaw;
    if (stroke.mode != Brake::Flick) return 0;
    const float cycle = stroke.outward_s + stroke.return_s;
    const auto sweep = static_cast<int>(time / cycle);
    const float phase = time - static_cast<float>(sweep) * cycle;
    const float return_fraction = std::clamp((phase - stroke.outward_s) / stroke.return_s, 0.f, 1.f);
    return (sweep % 2 ? -stroke.yaw : stroke.yaw) * (1 - return_fraction);
}

template <class Command>
HoldScore simulate_station(BodyState body, V anchor, float axis, V current, Command request,
                           bool save, Clock::time_point deadline,
                           float dt = comparison_step_s, float duration = hold_time_s) {
    HoldScore score;
    const float initial_work = body.work;
    auto pose = tail::snapshot(body);
    if (save) record(score.replay, body, pose, 0);
    const int ticks = static_cast<int>(std::lround(duration / dt));
    const int brake_ticks = static_cast<int>(std::lround(brake_time_s / dt));
    const int per_frame = static_cast<int>(std::lround(.05f / dt));
    float low_axial = 1, high_axial = -1;
    for (int tick = 0; tick < ticks; ++tick) {
        if ((tick & 63) == 0 && Clock::now() >= deadline) {
            score.fault = "time_budget";
            break;
        }
        float targets[2]{};
        const bool powered = request(static_cast<float>(tick) * dt, pose, targets);
        tail::StepInfo info;
        if (!tail::step(body, powered ? targets : nullptr, dt, info, {}, current)) {
            score.fault = "invalid_dynamics";
            break;
        }
        ++score.substeps;
        pose = tail::snapshot(body);
        score.min_clearance = std::min(score.min_clearance, clearance(body, pose));
        if (!(score.min_clearance > 0)) {
            score.fault = "left_clear_water_fixture";
            break;
        }
        const V error = pose.com_m - anchor;
        const float radius = physics::norm(error);
        const float speed = physics::norm(pose.com_velocity_m_s);
        score.end_speed = speed;
        if (tick + 1 == brake_ticks) score.brake_speed = speed;
        score.peak_anchor_error = std::max(score.peak_anchor_error, radius);
        score.depth_error = std::max(score.depth_error, std::abs(error.y));
        score.peak_tilt = std::max(score.peak_tilt, pose.tilt_deg);
        score.lateral_error = std::max(score.lateral_error, std::abs(forward(error, axis + pi * .5f)));
        score.positive_work_j = body.work - initial_work;
        score.peak_torque = std::max({score.peak_torque, std::abs(info.torque_Nm[0]),
            std::abs(info.torque_Nm[1])});
        score.quiet_streak_s = radius <= anchor_radius_m && speed <= quiet_speed_m_s ?
            score.quiet_streak_s + dt : 0.f;
        if (tick + 1 >= brake_ticks) {
            const float axial = forward(error, axis);
            low_axial = std::min(low_axial, axial);
            high_axial = std::max(high_axial, axial);
            score.peak_quiet_speed = std::max(score.peak_quiet_speed, speed);
            score.quiet_anchor_error = std::max(score.quiet_anchor_error, radius);
            score.hold_axial_excursion = high_axial - low_axial;
        }
        if (save && (tick + 1) % per_frame == 0) {
            record(score.replay, body, pose, static_cast<float>(tick + 1) * dt);
        }
    }
    return score;
}

HoldScore simulate_hold(BodyState body, V anchor, float axis, BrakeStroke stroke, bool save,
                        Clock::time_point deadline) {
    const auto request = [&](float time, const tail::Snapshot&, float (&targets)[2]) {
        if (time < brake_time_s) targets[0] = braking_yaw(stroke, time);
        return stroke.mode != Brake::Disconnected;
    };
    return simulate_station(body, anchor, axis, {}, request, save, deadline);
}

void write_hold_metrics(std::ostream& csv, const HoldScore& s) {
    csv << s.substeps << ',' << s.fault << ','
        << hold_pass(s) << ',' << s.brake_speed * 1000 << ',' << s.end_speed * 1000 << ','
        << s.peak_quiet_speed * 1000 << ',' << s.peak_anchor_error * 1000 << ','
        << s.quiet_anchor_error * 1000 << ',' << s.hold_axial_excursion * 1000 << ','
        << s.depth_error * 1000 << ',' << s.lateral_error * 1000 << ',' << s.peak_tilt << ','
        << s.min_clearance * 1000 << ',' << s.quiet_streak_s << ',' << s.peak_torque * 1e6 << ','
        << s.positive_work_j << '\n';
}

void write_hold_columns(std::ostream& csv) {
    csv << "substeps,fault,pass,speed_at_4s_mm_s,end_speed_mm_s,peak_hold_speed_mm_s,"
        << "peak_anchor_mm,peak_hold_anchor_mm,hold_axial_excursion_mm,depth_error_mm,"
        << "lateral_error_mm,peak_tilt_deg,min_clearance_mm,final_quiet_streak_s,peak_torque_uNm,positive_work_J\n";
}

void write_hold_row(std::ostream& csv, const char* start, const BrakeStroke& stroke, const HoldScore& s) {
    const char* modes[]{"disconnected", "neutral", "fold", "flick"};
    csv << start << ',' << modes[static_cast<int>(stroke.mode)] << ',' << stroke.yaw << ','
        << stroke.outward_s << ',' << stroke.return_s << ',';
    write_hold_metrics(csv, s);
}

int run_hold(const fs::path& folder) {
    const auto start = Clock::now();
    const auto deadline = start + std::chrono::milliseconds(3500);
    if (!fs::create_directory(folder)) throw std::runtime_error("Output folder must be new");
    const Stroke swimming{1, .55f, 0};
    const auto swim = simulate({}, swimming, Lesson::Forward, 0, drive_s,
        comparison_step_s, 0, true, 0, 0, deadline);
    if (!movement_pass(swim)) {
        std::cerr << "Held entry failed the forward fixture: " << swim.stop << '\n';
        return 2;
    }
    const auto entry = tail::snapshot(swim.state);
    const auto anchor = entry.com_m;
    const float axis = swim.measured_course_rad;
    std::vector<BrakeStroke> settings{{Brake::Disconnected}, {Brake::Neutral},
        {Brake::Fold, -.9f}, {Brake::Fold, .9f}};
    for (float sign : {-1.f, 1.f}) {
        for (float yaw : {.5f, .75f, .95f}) {
            for (float outward : {.16f, .28f}) {
                for (float recovery : {.6f, 1.2f}) {
                    settings.push_back({Brake::Flick, sign * yaw, outward, recovery});
                }
            }
        }
    }
    std::vector<HoldScore> scores(settings.size());
    std::atomic<std::size_t> next = 0;
    std::vector<std::thread> workers;
    for (int i = 0; i < 8; ++i) {
        workers.emplace_back([&] {
            for (auto index = next++; index < settings.size(); index = next++) {
                scores[index] = simulate_hold(swim.state, anchor, axis, settings[index], false, deadline);
            }
        });
    }
    for (auto& worker : workers) worker.join();
    std::ofstream csv(folder / "hold.csv");
    csv.exceptions(std::ios::failbit | std::ios::badbit);
    csv << std::setprecision(8) << "start,command,yaw_rad,outward_s,return_s,";
    write_hold_columns(csv);
    std::size_t best = scores.size();
    int swim_hold_passes = 0;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        write_hold_row(csv, "swimming", settings[i], scores[i]);
        swim_hold_passes += hold_pass(scores[i]);
        if (!*scores[i].fault && (best == scores.size() ||
                (hold_pass(scores[i]) && !hold_pass(scores[best])) ||
                (hold_pass(scores[i]) == hold_pass(scores[best]) &&
                    scores[i].quiet_anchor_error < scores[best].quiet_anchor_error))) best = i;
    }
    if (best == scores.size()) return 3;
    const auto best_run = simulate_hold(swim.state, anchor, axis, settings[best], true, deadline);
    if (best_run.replay.frames.size() > 1) {
        write_trace(folder / "swim-to-brake.json", swim.frames, best_run.replay.frames,
            &anchor, drive_s, drive_s + brake_time_s);
    }
    BodyState stationary;
    const auto rest_anchor = tail::snapshot(stationary).com_m;
    const auto rest = simulate_hold(stationary, rest_anchor, 0, {Brake::Disconnected}, true, deadline);
    write_hold_row(csv, "centered_rest", {Brake::Disconnected}, rest);
    if (rest.replay.frames.size() > 1) {
        write_trace(folder / "centered-rest.json", rest.replay.frames, {}, &rest_anchor);
    }
    for (float sign : {-1.f, 1.f}) {
        BodyState moving;
        moving.velocity[0] = sign * .00075f;
        const char* label = sign < 0 ? "drift_negative" : "drift_positive";
        const auto neutral = simulate_hold(moving, rest_anchor, 0, {Brake::Neutral}, false, deadline);
        write_hold_row(csv, label, {Brake::Neutral}, neutral);
        if (settings[best].mode != Brake::Neutral) {
            const auto tested = simulate_hold(moving, rest_anchor, 0, settings[best], false, deadline);
            write_hold_row(csv, label, settings[best], tested);
        }
    }
    csv.close();
    std::cout << "entry_mm_s=" << physics::norm(entry.com_velocity_m_s) * 1000
        << " swim_holds=" << swim_hold_passes << '/' << scores.size() << " best_row=" << best
        << " passive_rest=" << hold_pass(rest) << " elapsed_s="
        << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    return 0;
}

// Exercise upstream propulsion with the same fixed world-space anchor.
// Only the root actuator targets depend on the anchor error.
HoldScore run_current_case(Stroke stroke, float heading, float downstream_offset,
                           float drift, bool save, Clock::time_point deadline,
                           float dt = comparison_step_s, float duration = hold_time_s) {
    BodyState body;
    body.orientation = physics::axis_rotation({0, 1, 0}, heading);
    const V anchor = tail::snapshot(body).com_m;
    const V upstream = body.orientation.rotate({1, 0, 0});
    const V current = upstream * -.00075f;
    body.position = body.position - upstream * (downstream_offset / physics::scale);
    body.velocity[0] = -upstream.x * drift;
    body.velocity[2] = -upstream.z * drift;
    const auto request = [&](float time, const tail::Snapshot& pose, float (&targets)[2]) {
        if (stroke.amplitude_rad <= 0) return false;
        const float downstream = physics::dot(anchor - pose.com_m, upstream);
        const float amplitude = std::clamp(stroke.amplitude_rad + 35.f * downstream, 0.f, .26f);
        targets[0] = amplitude * std::sin(2 * pi * stroke.frequency_hz * time);
        return true;
    };
    return simulate_station(body, anchor, heading, current, request, save, deadline, dt, duration);
}

int run_current_hold(const fs::path& folder, bool confirm) {
    const auto start = Clock::now();
    const auto deadline = start + std::chrono::milliseconds(3500);
    if (!fs::create_directory(folder)) throw std::runtime_error("Output folder must be new");
    std::vector<Stroke> strokes{{0, 0, 0}};
    if (confirm) {
        strokes.push_back({3, .05f, 0});
    } else {
        for (float frequency : {1.f, 1.5f, 2.f, 2.5f, 3.f}) {
            for (float amplitude : {.025f, .05f, .075f, .1f, .125f, .15f, .18f, .22f, .26f}) {
                strokes.push_back({frequency, amplitude, 0});
            }
        }
    }
    const float duration = confirm ? 20.f : hold_time_s;
    std::vector<HoldScore> scores(strokes.size());
    std::atomic<std::size_t> next = 0;
    std::vector<std::thread> workers;
    for (int i = 0; i < 8; ++i) {
        workers.emplace_back([&] {
            for (auto index = next++; index < strokes.size(); index = next++) {
                scores[index] = run_current_case(strokes[index], 0, 0, 0, false, deadline,
                    comparison_step_s, duration);
            }
        });
    }
    for (auto& worker : workers) worker.join();
    std::ofstream csv(folder / "current-hold.csv");
    csv.exceptions(std::ios::failbit | std::ios::badbit);
    csv << std::setprecision(8) << "start,frequency_Hz,yaw_rad,";
    write_hold_columns(csv);
    const auto write = [&](const char* start_name, Stroke stroke, const HoldScore& score) {
        csv << start_name << ',' << stroke.frequency_hz << ',' << stroke.amplitude_rad << ',';
        write_hold_metrics(csv, score);
    };
    std::size_t best = strokes.size();
    float best_score = 0;
    int passes = 0;
    for (std::size_t i = 0; i < strokes.size(); ++i) {
        const auto& row = scores[i];
        write("calm_start", strokes[i], row);
        passes += i > 0 && hold_pass(row);
        if (i == 0 || *row.fault) continue;
        const float quality = std::max({row.quiet_anchor_error / anchor_radius_m,
            row.peak_quiet_speed / quiet_speed_m_s, row.hold_axial_excursion / anchor_radius_m,
            row.lateral_error / anchor_radius_m, row.depth_error / max_depth_error_m,
            row.peak_tilt / max_tilt_deg});
        if (best == strokes.size() || quality < best_score) {
            best = i;
            best_score = quality;
        }
    }
    if (best == strokes.size()) return 2;
    const auto stroke = strokes[best];
    const auto controlled = run_current_case(stroke, pi, .0005f, .00025f, true, deadline,
        comparison_step_s, duration);
    const auto passive = run_current_case(strokes[0], pi, .0005f, .00025f, false, deadline,
        comparison_step_s, duration);
    write("mirrored_disturbed", stroke, controlled);
    write("mirrored_disturbed", strokes[0], passive);
    if (confirm) {
        const auto fine = run_current_case(stroke, pi, .0005f, .00025f, false, deadline,
            comparison_step_s / 2, duration);
        write("mirrored_disturbed_half_step", stroke, fine);
        std::cout << "half_step_pass=" << hold_pass(fine) << ' ';
    }
    BodyState centered;
    centered.orientation = physics::axis_rotation({0, 1, 0}, pi);
    const V anchor = tail::snapshot(centered).com_m;
    if (controlled.replay.frames.size() > 1) {
        write_trace(folder / "current-station.json", controlled.replay.frames, {}, &anchor, 0, 0,
            "Holding against current");
    }
    csv.close();
    std::cout << "active_passes=" << passes << '/' << strokes.size() - 1
        << " chosen_Hz=" << stroke.frequency_hz << " chosen_yaw_rad=" << stroke.amplitude_rad
        << " held_out=" << hold_pass(controlled) << " passive_held_out=" << hold_pass(passive)
        << " elapsed_s=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    return 0;
}

enum class Fixture { Fastest, PondSpeed, Refined, Extended };

int run_habitat(const fs::path& folder) {
    const auto start = Clock::now();
    const auto deadline = start + std::chrono::milliseconds(3500);
    if (!fs::create_directory(folder)) throw std::runtime_error("Output folder must be new");
    const auto test_gap = [](V position) {
        BodyState body;
        body.position = position / physics::scale;
        return tail::submerged_gap_m(larger_pond, body, tail::snapshot(body));
    };
    const bool bounds_pass = test_gap({0, -.08f, 0}) > 0 &&
        test_gap({.56f, -.08f, 0}) > 0 && test_gap({0, -.08f, .38f}) > 0 &&
        test_gap({0, -.006f, 0}) <= 0 && test_gap({0, -.18f, 0}) <= 0 &&
        test_gap({.88f, -.08f, 0}) <= 0;
    std::cout << "clearance_controls=" << bounds_pass << '\n';
    if (!bounds_pass) return 2;

    const Stroke stroke{1, .55f, 0};
    struct Start { const char* name; V position; float yaw, turn; };
    const Start fixtures[]{
        {"east_shoal", {.45f, -.08f, 0}, 0, 30},
        {"west_shoal", {-.45f, -.08f, 0}, pi, -30},
        {"north_shoal", {0, -.08f, -.24f}, pi/2, 30},
        {"south_shoal", {0, -.08f, .24f}, -pi/2, -30}
    };
    std::ofstream csv(folder / "cases.csv");
    csv.exceptions(std::ios::failbit | std::ios::badbit);
    csv << std::setprecision(8);
    write_movement_columns(csv);
    int passes = 0;
    for (const auto& fixture : fixtures) {
        BodyState body;
        body.position = fixture.position / physics::scale;
        body.orientation = physics::axis_rotation({0, 1, 0}, fixture.yaw);
        const auto swim = simulate(body, stroke, Lesson::Forward, fixture.yaw, drive_s, comparison_step_s,
            0, true, fixture.yaw, 0, deadline, &larger_pond);
        write_case(csv, fixture.name, stroke, 0, swim, movement_pass(swim));
        if (!movement_pass(swim)) continue;
        const float course_at_request = swim.measured_course_rad;
        const auto turn = simulate(swim.state, stroke, Lesson::Turn, course_at_request + fixture.turn * rad,
            12, comparison_step_s, drive_s, true, course_at_request, -4.2f, deadline, &larger_pond);
        const auto& end = turn.frames.empty() ? swim.frames.back() : turn.frames.back();
        const auto place = geometry::sample(larger_pond, end.pose.com_m.x, end.pose.com_m.z);
        const auto old = geometry::sample(pond, end.pose.com_m.x, end.pose.com_m.z);
        const bool pass = movement_pass(turn) && place.region == geometry::SubmergedSlope;
        passes += pass;
        write_case(csv, fixture.name, stroke, fixture.turn, turn, pass);
        if (turn.frames.size() > 1) write_trace(folder / (std::string(fixture.name) + ".json"), swim.frames, turn.frames);
        std::cout << fixture.name << " pass=" << pass << " shoal_at_end="
            << (place.region == geometry::SubmergedSlope) << " outside_previous_water="
            << (old.region == geometry::DryLand) << " minimum_gap_mm=" << turn.clearance_m * 1000 << '\n';
    }
    csv.close();
    std::cout << "shoal_turns=" << passes << "/4 elapsed_s="
        << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    return passes == 4 ? 0 : 3;
}

int run_repeated_habitat_turns(const fs::path& folder) {
    const auto start = Clock::now();
    const auto deadline = start + std::chrono::milliseconds(3500);
    if (!fs::create_directory(folder)) throw std::runtime_error("Output folder must be new");
    constexpr Stroke stroke{1, .55f, 0};
    constexpr float leg_s = 12;
    struct Sequence {
        const char* name;
        V initial_position_m;
        std::vector<float> turn_degrees;
    };
    // Early and late same-sign fixtures differ only in 150 mm of initial eastward progress.
    const Sequence sequences[]{
        {"alternating_interior", {-.32f, -.08f, .23f}, {30, -30, 30, -30}},
        {"same_sign_early", {.30f, -.08f, .17f}, {30, 30, 30, 30, 30, 30, 30}},
        {"same_sign_late", {.45f, -.08f, .17f}, {30, 30, 30, 30, 30, 30, 30}},
    };
    std::ofstream csv(folder / "cases.csv");
    csv.exceptions(std::ios::failbit | std::ios::badbit);
    csv << std::setprecision(9)
        << "sequence,leg,start_s,entry_course_deg,target_world_course_deg,end_course_deg,"
           "cumulative_course_change_deg,end_com_x_mm,end_com_z_mm,";
    write_movement_columns(csv);
    int total_attempted = 0, total_passed = 0, total_steps = 0;
    for (const auto& sequence : sequences) {
        BodyState initial;
        initial.position = sequence.initial_position_m / physics::scale;
        const auto warm = simulate(initial, stroke, Lesson::Forward, 0, drive_s, comparison_step_s,
            0, true, 0, 0, deadline, &larger_pond);
        total_steps += warm.steps;
        const auto warm_pose = tail::snapshot(warm.state);
        csv << sequence.name << ",0,0,0,0," << warm.measured_course_rad / rad
            << ",0," << warm_pose.com_m.x * 1000 << ',' << warm_pose.com_m.z * 1000 << ',';
        write_case(csv, "warmup", stroke, 0, warm, movement_pass(warm));
        if (!movement_pass(warm)) {
            std::cout << sequence.name << " warmup_failed " << warm.stop << '\n';
            continue;
        }

        auto body = warm.state;
        auto frames = warm.frames;
        SteeringHistory steering;
        float entry_course = warm.measured_course_rad;
        float total_course_change_deg = 0;
        float elapsed_s = drive_s;
        float min_gap_m = warm.clearance_m;
        int passed = 0, attempted = 0;
        for (const float degrees : sequence.turn_degrees) {
            ++attempted;
            const float target_course = wrap(entry_course + degrees * rad);
            auto turn = simulate(body, stroke, Lesson::Turn, target_course, leg_s, comparison_step_s,
                elapsed_s, true, entry_course, -4.2f, deadline, &larger_pond, &steering);
            total_steps += turn.steps;
            min_gap_m = std::min(min_gap_m, turn.clearance_m);
            for (std::size_t i = 1; i < turn.frames.size(); ++i) {
                auto frame = turn.frames[i];
                frame.time_s += elapsed_s;
                frames.push_back(frame);
            }
            body = turn.state;
            const auto end = tail::snapshot(body);
            if (completed(turn)) total_course_change_deg += turn.course_change_deg;
            const bool pass = movement_pass(turn);
            passed += pass;
            csv << sequence.name << ',' << attempted << ',' << elapsed_s << ','
                << entry_course / rad << ',' << target_course / rad << ',' << turn.measured_course_rad / rad << ','
                << total_course_change_deg << ',' << end.com_m.x * 1000 << ',' << end.com_m.z * 1000 << ',';
            write_case(csv, "turn", stroke, degrees, turn, pass);
            if (!completed(turn)) break;
            entry_course = turn.measured_course_rad;
            elapsed_s += leg_s;
        }
        if (frames.size() >= 2) write_trace(folder / (std::string(sequence.name) + ".json"), frames, {});
        total_attempted += attempted;
        total_passed += passed;
        std::cout << sequence.name << " passed=" << passed << '/' << sequence.turn_degrees.size()
            << " attempted=" << attempted << " accumulated_turn_deg=" << total_course_change_deg
            << " minimum_wet_gap_mm=" << min_gap_m * 1000 << '\n';
    }
    csv.close();
    std::cout << "repeated_turns=" << total_passed << '/' << total_attempted << " attempted; planned=18"
        << " substeps=" << total_steps << " elapsed_s="
        << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    return total_passed == 18 ? 0 : 3;
}

int run(const fs::path& folder, Fixture fixture) {
    const auto start = Clock::now();
    // Reserve a small part of the four-second local budget for output.
    const auto deadline = start + std::chrono::milliseconds(3500);
    if (!fs::create_directory(folder)) throw std::runtime_error("Output folder must be new");
    std::cout << std::setprecision(7);
    const auto checks = check_physics(deadline);
    std::cout << "preflight=" << checks.valid << " rest_travel_mm=" << checks.rest_travel_mm
        << " passive_bend_rad=" << checks.passive_bend_remaining << " passive_energy_ratio="
        << checks.passive_energy_ratio << " max_linear_momentum=" << checks.max_linear_momentum
        << " max_angular_momentum=" << checks.max_angular_momentum << std::endl;
    if (!checks.valid) return 2;

    const bool refined = fixture == Fixture::Refined || fixture == Fixture::Extended;
    const float dt = refined ? comparison_step_s : step_s;
    const float turn_duration = fixture == Fixture::Extended ? 12.f : turn_s;
    const bool pond_speed_fixture = fixture != Fixture::Fastest;
    std::vector<Stroke> strokes;
    if (fixture == Fixture::Extended) {
        strokes.push_back({1, .55f, 0});
    } else {
        for (float frequency : {1.f, 2.f, 3.f, 4.f}) {
            for (float amplitude : {.3f, .55f, .8f}) {
                for (float pitch : {0.f, -.12f, .12f}) {
                    strokes.push_back({frequency, amplitude, pitch});
                }
            }
        }
    }
    std::vector<Outcome> drives(strokes.size());
    std::atomic<std::size_t> next = 0;
    std::vector<std::thread> workers;
    for (int index = 0; index < 8; ++index) {
        workers.emplace_back([&] {
            for (auto i = next++; i < strokes.size(); i = next++) {
                drives[i] = simulate({}, strokes[i], Lesson::Forward, 0, drive_s, dt, 0, false, 0, 0, deadline);
            }
        });
    }
    for (auto& worker : workers) worker.join();

    std::ofstream csv(folder / "cases.csv");
    csv.exceptions(std::ios::failbit | std::ios::badbit);
    csv << std::setprecision(8);
    write_movement_columns(csv);
    std::size_t best = strokes.size();
    int forward_passes = 0;
    for (std::size_t i = 0; i < strokes.size(); ++i) {
        const auto& drive = drives[i];
        const bool pass = movement_pass(drive);
        forward_passes += pass;
        write_case(csv, "forward_sweep", strokes[i], 0, drive, pass);
        if (completed(drive) && (best == strokes.size() ||
                (pass && !movement_pass(drives[best])) ||
                (pass == movement_pass(drives[best]) &&
                    (pond_speed_fixture ?
                        std::abs(drive.projected_speed_m_s - .008f) <
                            std::abs(drives[best].projected_speed_m_s - .008f) :
                        drive.projected_speed_m_s > drives[best].projected_speed_m_s)))) best = i;
    }
    std::cout << "swims=" << forward_passes << '/' << strokes.size() << std::endl;
    if (best == strokes.size()) return 3;
    const Stroke stroke = strokes[best];
    const auto straight = simulate({}, stroke, Lesson::Forward, 0, drive_s, dt, 0, true, 0, 0, deadline);
    const auto half_step = simulate({}, stroke, Lesson::Forward, 0, drive_s, .5f * dt, 0, false, 0, 0, deadline);
    write_case(csv, "forward_replay", stroke, 0, straight, movement_pass(straight));
    write_case(csv, "forward_half_step", stroke, 0, half_step, movement_pass(half_step));
    if (straight.frames.size() >= 2) write_trace(folder / "forward.json", straight.frames, {});
    const double timestep_delta_mm = 1000 * physics::norm(
        tail::snapshot(straight.state).com_m - tail::snapshot(half_step.state).com_m);
    std::cout << "best=" << best << " forward_mm_s=" << straight.projected_speed_m_s * 1000
        << " dt_ms=" << 1000 * dt << " half_step_endpoint_delta_mm=" << timestep_delta_mm << std::endl;
    if (!completed(straight) || !completed(half_step)) return 3;

    const float entry_course = straight.measured_course_rad;
    constexpr std::array<float, 4> degrees{-30.f, -15.f, 15.f, 30.f};
    std::array<Outcome, degrees.size()> turns;
    const float steering_gain = refined ? -4.2f :
        (pond_speed_fixture ? -1.4f : 1.4f);
    workers.clear();
    for (std::size_t i = 0; i < turns.size(); ++i) {
        workers.emplace_back([&, i] {
            turns[i] = simulate(straight.state, stroke, Lesson::Turn, entry_course + degrees[i] * rad,
                turn_duration, dt, drive_s, true, entry_course, steering_gain, deadline);
        });
    }
    std::array<Outcome, degrees.size()> reference;
    if (fixture == Fixture::Refined) {
        for (std::size_t i = 0; i < reference.size(); ++i) {
            workers.emplace_back([&, i] {
                reference[i] = simulate(straight.state, stroke, Lesson::Turn,
                    entry_course + degrees[i] * rad, turn_s, dt, drive_s, false, entry_course, -1.4f, deadline);
            });
        }
    }
    auto coast = simulate(straight.state, stroke, Lesson::Coast, entry_course,
        rest_s, dt, drive_s, true, entry_course, 0, deadline);
    for (auto& worker : workers) worker.join();
    int turn_passes = 0;
    for (std::size_t i = 0; i < turns.size(); ++i) {
        const bool pass = movement_pass(straight) && movement_pass(turns[i]);
        turn_passes += pass;
        write_case(csv, "turn", stroke, degrees[i], turns[i], pass);
        if (fixture == Fixture::Refined) {
            write_case(csv, "turn_lower_gain", stroke, degrees[i], reference[i],
                movement_pass(straight) && movement_pass(reference[i]));
        }
        if (turns[i].frames.size() >= 2) write_trace(folder / ("turn-" + std::to_string(
            static_cast<int>(degrees[i])) + ".json"), straight.frames, turns[i].frames);
    }
    const auto coast_pose = tail::snapshot(coast.state);
    const bool coast_pass = movement_pass(straight) && completed(coast) &&
        horizontal_speed(coast_pose.com_velocity_m_s) < .0005f &&
        coast.max_depth_error <= max_depth_error_m && coast.max_tilt <= max_tilt_deg;
    write_case(csv, "coast", stroke, 0, coast, coast_pass);
    if (coast.frames.size() >= 2) write_trace(folder / "coast.json", straight.frames, coast.frames);
    csv.close();
    std::cout << "turns=" << turn_passes << '/' << turns.size() << " coast_pass=" << coast_pass
        << " coast_end_mm_s=" << 1000 * horizontal_speed(coast_pose.com_velocity_m_s)
        << " elapsed_s=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    const std::string mode = argc == 3 ? argv[2] : "";
    if (argc < 2 || argc > 3 ||
        (argc == 3 && mode != "--pond-speed" && mode != "--refined" && mode != "--extended" &&
         mode != "--hold" && mode != "--current-hold" && mode != "--verify-current-hold" &&
         mode != "--habitat" && mode != "--repeated-habitat-turns")) {
        std::cerr << "Usage: pond_flexible_tail_check <new-output-folder> "
            "[--pond-speed|--refined|--extended|--hold|--current-hold|--verify-current-hold|"
            "--habitat|--repeated-habitat-turns]\n";
        return 1;
    }
    try {
        if (mode == "--repeated-habitat-turns") return run_repeated_habitat_turns(argv[1]);
        if (mode == "--habitat") return run_habitat(argv[1]);
        if (mode == "--hold") return run_hold(argv[1]);
        if (mode == "--current-hold" || mode == "--verify-current-hold") {
            return run_current_hold(argv[1], mode == "--verify-current-hold");
        }
        Fixture fixture = Fixture::Fastest;
        if (mode == "--pond-speed") fixture = Fixture::PondSpeed;
        if (mode == "--refined") fixture = Fixture::Refined;
        if (mode == "--extended") fixture = Fixture::Extended;
        return run(argv[1], fixture);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
