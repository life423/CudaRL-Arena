#include "rl/flexible_tail_motor.h"
#include "environment/flexible_tail_clearance.h"
#include "nn/mlp_cpu.h"
#include "habitat_config.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace physics = cudarl::physics;
namespace tail = cudarl::flexible_tail;
namespace checkpoint_fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using physics::V;
using Actor = cudarl::nn::CpuMlp<float>;

constexpr float degrees_to_radians = tail::pi / 180.f;
constexpr int ticks_per_second = 50;
constexpr int physics_per_tick = 16;
constexpr int warmup_ticks = 6 * ticks_per_second;
constexpr int course_history_ticks = ticks_per_second / 2;
static_assert(tail::motor_control_s == .02f && tail::motor_substep_s == .00125f);

enum class Task { Swim, Turn, Calm, Current };

struct Case {
    const char* name;
    Task task;
    int seconds;
    float heading, turn_degrees, x_m, z_m, root_yaw, phase_cycles;
};

constexpr Case training[] = {
    {"swim_a", Task::Swim, 8, -.16f, 0, -.035f, .02f, .02f, 0},
    {"swim_b", Task::Swim, 8, .34f, 0, .02f, -.025f, -.02f, .5f},
    {"left_15_a", Task::Turn, 18, .09f, 15, .025f, 0, .02f, 0},
    {"right_15_a", Task::Turn, 18, -.09f, -15, -.025f, 0, -.02f, .5f},
    {"left_30_a", Task::Turn, 18, .18f, 30, -.04f, .02f, .02f, 0},
    {"right_30_a", Task::Turn, 18, -.24f, -30, -.04f, -.02f, -.02f, .5f},
    {"left_30_b", Task::Turn, 18, -.37f, 30, .03f, -.025f, -.02f, .5f},
    {"right_30_b", Task::Turn, 18, .43f, -30, .03f, .025f, .02f, 0},
    {"calm_a", Task::Calm, 12, .2f, 0, -.03f, .015f, 0, 0},
    {"calm_b", Task::Calm, 12, -1.1f, 0, .025f, -.015f, 0, 0},
    {"current_a", Task::Current, 20, .32f, 0, .02f, .01f, 0, 0},
    {"current_b", Task::Current, 20, 2.67f, 0, -.027f, -.015f, 0, .5f},
};

constexpr Case evaluation[] = {
    {"swim", Task::Swim, 8, -.27f, 0, .005f, .024f, .015f, 0},
    {"left_15", Task::Turn, 18, -.51f, 15, -.014f, -.026f, -.015f, 0},
    {"right_15", Task::Turn, 18, .61f, -15, .01f, .027f, .015f, 0},
    {"left_30", Task::Turn, 18, .87f, 30, -.022f, .008f, .015f, 0},
    {"right_30", Task::Turn, 18, -.82f, -30, -.011f, -.017f, -.015f, 0},
    {"calm_hold", Task::Calm, 12, 1.15f, 0, -.019f, .044f, 0, 0},
    {"current_hold", Task::Current, 20, -2.22f, 0, .026f, -.049f, 0, 0},
};

constexpr int example_count() {
    int rows = 0;
    for (const auto& item : training) rows += item.seconds * ticks_per_second;
    return rows;
}

struct Dataset {
    std::vector<float> observations, targets;
};

struct Metrics {
    const char* fault = "";
    int substeps = 0;
    bool passed = false;
    float progress_mm_s = 0, course_error_deg = 0, turn_deg = 0, warmup_mm_s = 0;
    float peak_tilt_deg = 0, peak_depth_mm = 0, minimum_gap_mm = 1000;
    float peak_hold_mm_s = 0, peak_anchor_mm = 0, peak_lateral_mm = 0, axial_excursion_mm = 0;
};

V horizontal(V value) { return {value.x, 0, value.z}; }

V direction(float heading) { return {std::cos(heading), 0, -std::sin(heading)}; }

float bearing(V value) { return std::atan2(-value.z, value.x); }

float wrap(float angle) { return std::remainder(angle, 2 * tail::pi); }

bool motion_gate(float speed_mm_s, float error_deg) {
    return speed_mm_s >= 6.f && error_deg <= 5.f;
}

Metrics roll(const Case& item, Clock::time_point deadline, Dataset* data, Actor* actor) {
    const bool holding = item.task == Task::Calm || item.task == Task::Current;
    const V initial_heading = direction(item.heading);
    const V current = item.task == Task::Current ? initial_heading * -.00075f : V{};
    physics::BodyState body;
    body.position = V{item.x_m, -.08f, item.z_m} / physics::scale;
    body.orientation = physics::axis_rotation({0, 1, 0}, item.heading);
    body.angle[0] = item.root_yaw;
    const V anchor = tail::snapshot(body).com_m;
    if (item.task == Task::Current) {
        body.position = body.position - initial_heading * (.0005f / physics::scale);
        body.velocity[0] = initial_heading.x * -.00025f;
        body.velocity[2] = initial_heading.z * -.00025f;
    }
    auto pose = tail::snapshot(body);
    float held_heading = item.heading;
    float entry_course = item.heading;
    tail::MovementRequest request;
    request.choice = holding ? tail::Choice::Hold : tail::Choice::Swim;
    request.origin_m = anchor;
    request.heading = initial_heading;
    request.terminal_velocity_m_s = holding ? V{} : initial_heading * tail::requested_speed_m_s;
    request.initialized = true;

    std::array<V, course_history_ticks> positions{};
    positions.fill(pose.com_m);
    const V side = direction(item.heading + tail::pi / 2);
    const int controls = item.seconds * ticks_per_second;
    float last[tail::motor_outputs]{};
    float steering_bias = 0;
    V final_start = pose.com_m, warmup_start = pose.com_m;
    float axial_min = 1, axial_max = -1;
    float warmup_error = 0;
    Metrics result;

    for (int i = 0; i < controls; ++i) {
        if ((i & 15) == 0 && Clock::now() >= deadline) {
            result.fault = "process_deadline";
            break;
        }
        V& historical_position = positions[static_cast<std::size_t>(i) % positions.size()];
        const V travel = horizontal(pose.com_m - historical_position);
        historical_position = pose.com_m;
        const V measured_course = physics::norm(travel) >= .0015f ? travel : request.heading;
        if (i == controls - 2 * ticks_per_second) final_start = pose.com_m;
        if (i == warmup_ticks - 2 * ticks_per_second) warmup_start = pose.com_m;
        if (item.task == Task::Turn && i == warmup_ticks) {
            const V warmup_displacement = horizontal(pose.com_m - warmup_start);
            result.warmup_mm_s = physics::dot(warmup_displacement, initial_heading) * 500;
            warmup_error = physics::norm(warmup_displacement) >= .001f ?
                std::abs(wrap(bearing(warmup_displacement) - item.heading)) / degrees_to_radians : 180.f;
            // Cue the turn from the settled course. The shorter observation
            // window stays in the turning feedback so it can react sooner.
            entry_course = physics::norm(warmup_displacement) >= .001f ?
                bearing(warmup_displacement) : item.heading;
            held_heading = entry_course + item.turn_degrees * degrees_to_radians;
            request.choice = item.turn_degrees > 0 ? tail::Choice::TurnLeft : tail::Choice::TurnRight;
            request.heading = direction(held_heading);
            request.terminal_velocity_m_s = request.heading * tail::requested_speed_m_s;
            request.origin_m = {pose.com_m.x, anchor.y, pose.com_m.z};
        }

        float observations[tail::motor_inputs];
        tail::observe_motor(body, pose, request, measured_course, current, last, observations);
        float targets[tail::motor_outputs]{};
        if (actor) {
            actor->forward(observations, 1);
            std::copy_n(actor->output(), tail::motor_outputs, targets);
        } else {
            // Fixture-only references supply labels. Every validation tick
            // instead follows the candidate's two learned actuator targets.
            const float seconds = static_cast<float>(i) * tail::motor_control_s;
            float amplitude = .55f;
            float frequency = 1.f;
            if (holding) {
                const float downstream = physics::dot(anchor - pose.com_m, initial_heading);
                amplitude = item.task == Task::Current ? std::clamp(.05f + 35.f * downstream, 0.f, .26f) : 0.f;
                frequency = 3.f;
            } else if (item.task == Task::Turn && i >= warmup_ticks) {
                const float error = wrap(held_heading - bearing(measured_course));
                const float goal_bias = std::clamp(-4.2f * error, -.45f, .45f);
                steering_bias += (tail::motor_control_s / .5f) * (goal_bias - steering_bias);
            }
            targets[0] = std::clamp(amplitude * std::sin(2 * tail::pi *
                (frequency * seconds + item.phase_cycles)) + steering_bias,
                -physics::limit, physics::limit);
            data->observations.insert(data->observations.end(), std::begin(observations), std::end(observations));
            data->targets.insert(data->targets.end(), std::begin(targets), std::end(targets));
        }
        std::copy_n(targets, tail::motor_outputs, last);
        for (int substep = 0; substep < physics_per_tick; ++substep) {
            tail::StepInfo info;
            if (!tail::step(body, targets, tail::motor_substep_s, info, {}, current)) {
                result.fault = "invalid_dynamics";
                return result;
            }
            ++result.substeps;
            pose = tail::snapshot(body);
            const auto gap = static_cast<float>(tail::submerged_gap_m(larger_pond, body, pose) * 1000);
            result.minimum_gap_mm = std::min(result.minimum_gap_mm, gap);
            if (!(gap > 0)) {
                result.fault = "submergence_not_certified";
                return result;
            }
            const V difference = pose.com_m - anchor;
            result.peak_depth_mm = std::max(result.peak_depth_mm, 1000 * std::abs(difference.y));
            result.peak_tilt_deg = std::max(result.peak_tilt_deg, pose.tilt_deg);
            if (holding) result.peak_lateral_mm = std::max(result.peak_lateral_mm, 1000 * std::abs(physics::dot(difference, side)));
            if (holding && i >= 4 * ticks_per_second) {
                result.peak_hold_mm_s = std::max(result.peak_hold_mm_s, 1000 * physics::norm(pose.com_velocity_m_s));
                result.peak_anchor_mm = std::max(result.peak_anchor_mm, 1000 * physics::norm(difference));
                const float axial = physics::dot(difference, initial_heading);
                axial_min = std::min(axial_min, axial);
                axial_max = std::max(axial_max, axial);
            }
        }
    }
    if (*result.fault) return result;
    const V final_displacement = horizontal(pose.com_m - final_start);
    result.progress_mm_s = 500 * physics::dot(final_displacement, request.heading);
    result.course_error_deg = physics::norm(final_displacement) >= .001f ?
        std::abs(wrap(bearing(final_displacement) - held_heading)) / degrees_to_radians : 180.f;
    result.turn_deg = wrap(bearing(final_displacement) - entry_course) / degrees_to_radians;
    result.axial_excursion_mm = holding ? 1000 * (axial_max - axial_min) : 0;

    const bool posture = result.peak_depth_mm <= 1.f && result.peak_tilt_deg <= 10.f;
    if (holding) {
        result.passed = posture && result.peak_hold_mm_s <= .5f && result.peak_anchor_mm <= 2.f &&
            result.axial_excursion_mm <= 2.f && result.peak_lateral_mm <= 2.f;
    } else {
        result.passed = posture && motion_gate(result.progress_mm_s, result.course_error_deg) &&
            (item.task != Task::Turn || motion_gate(result.warmup_mm_s, warmup_error));
    }
    return result;
}

void metrics_heading() {
    std::cout << "case,pass,substeps,stop,progress_mm_s,course_error_deg,actual_turn_deg,warmup_mm_s,"
                 "peak_tilt_deg,peak_depth_mm,min_gap_mm,peak_hold_mm_s,peak_anchor_mm,"
                 "peak_lateral_mm,axial_excursion_mm\n";
}

void print(const Case& item, const Metrics& m) {
    std::cout << item.name << ',' << m.passed << ',' << m.substeps << ',' << m.fault << ','
        << m.progress_mm_s << ',' << m.course_error_deg << ',' << m.turn_deg << ',' << m.warmup_mm_s << ','
        << m.peak_tilt_deg << ',' << m.peak_depth_mm << ',' << m.minimum_gap_mm << ',' << m.peak_hold_mm_s
        << ',' << m.peak_anchor_mm << ',' << m.peak_lateral_mm << ',' << m.axial_excursion_mm << '\n';
}

void write_dataset(const checkpoint_fs::path& path, const Dataset& data) {
    const auto pending = path.string() + ".pending";
    if (checkpoint_fs::exists(path) || checkpoint_fs::exists(pending))
        throw std::runtime_error("Output dataset path must be new");
    if (data.observations.size() != example_count() * tail::motor_inputs ||
        data.targets.size() != example_count() * tail::motor_outputs)
        throw std::runtime_error("Dataset does not match the fixed collection size");
    constexpr uint32_t header[]{example_count(), tail::motor_inputs};
    std::ofstream file(pending, std::ios::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write(reinterpret_cast<const char*>(header), sizeof(header));
    file.write(reinterpret_cast<const char*>(data.observations.data()),
        static_cast<std::streamsize>(data.observations.size() * sizeof(float)));
    file.write(reinterpret_cast<const char*>(data.targets.data()),
        static_cast<std::streamsize>(data.targets.size() * sizeof(float)));
    file.close();
    checkpoint_fs::rename(pending, path);
}

int collect(const std::string& path, Clock::time_point deadline) {
    if (checkpoint_fs::exists(path) || checkpoint_fs::exists(path + ".pending"))
        throw std::runtime_error("Output dataset path must be new");
    Dataset data;
    data.observations.reserve(example_count() * tail::motor_inputs);
    data.targets.reserve(example_count() * tail::motor_outputs);
    int passed = 0, substeps = 0;
    metrics_heading();
    for (const auto& item : training) {
        const auto result = roll(item, deadline, &data, nullptr);
        print(item, result);
        substeps += result.substeps;
        if (*result.fault) return 2;
        passed += result.passed;
    }
    std::cout << "teacher_passes=" << passed << '/' << std::size(training)
              << " substeps=" << substeps << '\n';
    if (passed != std::size(training)) return 3;
    write_dataset(path, data);
    std::cout << "dataset_rows=" << example_count() << '\n';
    return 0;
}

int evaluate(const std::string& path, Clock::time_point deadline) {
    Actor actor(tail::motor_spec());
    actor.params() = tail::read_motor_weights(path);
    int passed = 0, substeps = 0;
    metrics_heading();
    for (const auto& item : evaluation) {
        const auto result = roll(item, deadline, nullptr, &actor);
        print(item, result);
        substeps += result.substeps;
        if (*result.fault) return 2;
        passed += result.passed;
    }
    std::cout << "evaluation_passes=" << passed << '/' << std::size(evaluation)
              << " substeps=" << substeps << '\n';
    return passed == std::size(evaluation) ? 0 : 3;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3 || (std::string(argv[1]) != "data" && std::string(argv[1]) != "evaluate")) {
        std::cerr << "Usage: pond_flex_motor_fixture data <examples.bin> | evaluate <actor.bin>\n";
        return 1;
    }
    try {
        std::cout << std::setprecision(9);
        const auto deadline = Clock::now() + std::chrono::milliseconds(3500);
        return std::string(argv[1]) == "data" ? collect(argv[2], deadline) : evaluate(argv[2], deadline);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
