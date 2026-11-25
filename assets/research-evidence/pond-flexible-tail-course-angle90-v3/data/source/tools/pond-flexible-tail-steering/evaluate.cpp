#include "rl/flexible_tail_fast_steering.h"
#include "nn/mlp_cpu.h"
#include "habitat_config.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
namespace tail = cudarl::flexible_tail;
namespace fast = tail::fast;
namespace physics = cudarl::physics;
namespace fs = std::filesystem;
using physics::V;
using Clock = std::chrono::steady_clock;
using Actor = cudarl::nn::CpuMlp<float>;
constexpr double control_s = .02;
constexpr double substep_s = .00125;
constexpr int turn_controls = 650;
constexpr int onward_controls = 100;
constexpr int closure_controls = 100;
constexpr int turn_total_controls = turn_controls + onward_controls + closure_controls;
constexpr int retention_controls = 600;
constexpr float radians_per_degree = tail::pi / 180.f;
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr const char* retention_names[12] = {
    "neutral_a", "neutral_b", "neutral_c", "neutral_d", "bent_left_a", "bent_right_a",
    "bent_left_b", "bent_right_b", "moving_a", "moving_b", "moving_c", "moving_d"};
static_assert(tail::turn_substeps_per_control == 16);
static_assert(tail::motor_control_s == static_cast<float>(control_s));
static_assert(tail::motor_substep_s == static_cast<float>(substep_s));

double signed_angle(V from, V to) {
    from.y = to.y = 0;
    if (physics::norm(from) < 1e-9f || physics::norm(to) < 1e-9f) return nan;
    return std::atan2(physics::cross(from, to).y, physics::dot(from, to)) / radians_per_degree;
}

double horizontal_length(V value) { value.y = 0; return physics::norm(value); }
void number(std::ostream& out, double value) { if (std::isfinite(value)) out << value; }
double env_number(const char* name, double fallback, double minimum, double maximum) {
    const char* raw = std::getenv(name);
    if (!raw) return fallback;
    char* end = nullptr;
    const double value = std::strtod(raw, &end);
    if (!end || *end || !std::isfinite(value) || value < minimum || value > maximum)
        throw std::runtime_error(std::string("Invalid environment setting: ") + name);
    return value;
}

struct Sample {
    double time = 0;
    V com{};
    float root = 0, root_rate = 0, passive = 0, passive_rate = 0;
    float tilt = 0, depth_mm = 0, clearance_mm = 0;
};
struct Control {
    int control = 0;
    V com{};
    double course_error = nan, rolling_speed_mm_s = 0, projected_rolling_mm_s = 0;
    float requested_offset = 0, applied_offset = 0, yaw_target = 0, pitch_target = 0;
    int fault = 0;
    double feedback_error_deg = nan, feedback_speed_mm_s = 0, feedback_period_s = 0;
    bool feedback_period_measured = false, feedback_fresh = false, feedback_reliable = false;
};
struct Crossing {
    double time = 0;
    int direction = 0;
    V com{};
};
struct Beat {
    double begin = 0, end = 0, swing = 0;
    V begin_com{}, end_com{};
};
struct Rhythm {
    int beats = 0, reversals = 0;
    double frequency = nan, swing = nan;
};
struct Window {
    bool complete = false;
    double projected_mm_s = nan, course_error_deg = nan, achieved_deg = nan;
    double minimum_rolling_mm_s = nan;
    Rhythm root, passive;
};
struct Job {
    std::string name;
    int arm = 0, phase = 0;
    float angle_deg = 0;
    bool retention = false;
    tail::NavigationWorld start{};
};
struct Result {
    int attempted = 0, accepted = 0, controls = 0, fault = 0, exact_bypass = 0;
    bool bypass_exact = true;
    double peak_tilt = 0, peak_depth_mm = 0, minimum_clearance_mm = 1e6;
    double time_to_target_s = nan, response_s = nan, effective_radius_mm = nan;
    V incoming{}, target{};
    std::vector<Sample> samples;
    std::vector<Control> trace;
    std::array<Window, 2> windows;
    fast::World first_failure_before{}, first_failure_after{};
    tail::NavigationWorld final_state{};
    std::string exception;
};

// Confirm a crossing only after its new rate reaches the same 0.02 rad/s
// deadband as the archived beat diagnostic. Its time remains the interpolated
// zero crossing, rather than the later confirmation sample.
std::vector<Crossing> crossings(const std::vector<Sample>& samples, bool root) {
    std::vector<Crossing> result;
    int direction = 0, pending_direction = 0;
    Crossing pending{};
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        const float rate = root ? sample.root_rate : sample.passive_rate;
        if (i) {
            const auto& previous = samples[i - 1];
            const float old_rate = root ? previous.root_rate : previous.passive_rate;
            if ((old_rate <= 0 && rate > 0) || (old_rate >= 0 && rate < 0)) {
                const double fraction = -double(old_rate) / (rate - old_rate);
                pending_direction = rate > 0 ? 1 : -1;
                pending = {previous.time + fraction * (sample.time - previous.time), pending_direction,
                    previous.com + (sample.com - previous.com) * static_cast<float>(fraction)};
            }
        }
        const int next = rate > .02f ? 1 : rate < -.02f ? -1 : 0;
        if (next) {
            if (direction && next != direction && pending_direction == next) result.push_back(pending);
            direction = next;
        }
    }
    return result;
}

std::vector<Beat> complete_beats(const std::vector<Sample>& samples, bool root) {
    const auto events = crossings(samples, root);
    std::vector<Beat> result;
    // Positive-to-positive intervals include a confirmed negative crossing.
    // Counting only one crossing direction avoids counting every beat twice.
    for (size_t i = 2; i < events.size(); ++i) {
        if (events[i - 2].direction != 1 || events[i - 1].direction != -1 || events[i].direction != 1) continue;
        const double begin = events[i - 2].time, end = events[i].time;
        double low = 1e30, high = -1e30;
        auto first = std::lower_bound(samples.begin(), samples.end(), begin,
            [](const Sample& sample, double time) { return sample.time < time; });
        for (auto sample = first; sample != samples.end() && sample->time <= end; ++sample) {
            const double angle = root ? sample->root : sample->passive;
            low = std::min(low, angle);
            high = std::max(high, angle);
        }
        if (end > begin && high >= low)
            result.push_back({begin, end, high - low, events[i - 2].com, events[i].com});
    }
    return result;
}

Rhythm rhythm(const std::vector<Sample>& samples, bool root, double begin, double end) {
    Rhythm result;
    double periods = 0, swings = 0;
    for (const auto& beat : complete_beats(samples, root)) {
        const double midpoint = .5 * (beat.begin + beat.end);
        if (midpoint < begin || midpoint >= end) continue;
        ++result.beats;
        periods += beat.end - beat.begin;
        swings += beat.swing;
    }
    if (result.beats) {
        result.frequency = result.beats / periods;
        result.swing = swings / result.beats;
    }
    for (const auto& event : crossings(samples, root))
        result.reversals += event.time >= begin && event.time < end;
    return result;
}

const Sample* at_time(const std::vector<Sample>& samples, double time) {
    const auto found = std::lower_bound(samples.begin(), samples.end(), time - 1e-7,
        [](const Sample& sample, double target) { return sample.time < target; });
    return found != samples.end() && std::abs(found->time - time) < 1e-5 ? &*found : nullptr;
}

void summarize(Result& result, const Job& job) {
    const double starts[2]{job.retention ? 8. : 11., job.retention ? 10. : 13.};
    for (int w = 0; w < 2; ++w) {
        auto& window = result.windows[w];
        const double begin = starts[w], end = begin + 2.;
        const Sample* first = at_time(result.samples, begin);
        const Sample* last = at_time(result.samples, end);
        if (!first || !last) continue;
        window.complete = true;
        const V displacement = last->com - first->com;
        window.projected_mm_s = 1000 * physics::dot(displacement, result.target) / (end - begin);
        window.course_error_deg = std::abs(signed_angle(result.target, displacement));
        window.achieved_deg = signed_angle(result.incoming, displacement);
        window.minimum_rolling_mm_s = 1e30;
        for (const auto& control : result.trace) {
            const double time = control.control * control_s;
            if (time > begin && time <= end + 1e-7)
                window.minimum_rolling_mm_s = std::min(window.minimum_rolling_mm_s, control.rolling_speed_mm_s);
        }
        window.root = rhythm(result.samples, true, begin, end);
        window.passive = rhythm(result.samples, false, begin, end);
    }
    if (!job.retention) {
        const auto beats = complete_beats(result.samples, true);
        int stable = 0;
        double stable_begin = nan;
        for (const auto& control : result.trace) {
            const double time = control.control * control_s;
            if (time > 13. + 1e-6) break;
            if (!std::isfinite(result.response_s)) {
                for (const auto& beat : beats) {
                    if (beat.end > time || beat.end < .5) continue;
                    const double achieved = signed_angle(result.incoming, beat.end_com - beat.begin_com);
                    if (achieved * job.angle_deg > 0 && std::abs(achieved) >= 2.) {
                        result.response_s = beat.end;
                        break;
                    }
                }
            }
            if (control.course_error <= 5 && control.projected_rolling_mm_s >= 6 &&
                control.rolling_speed_mm_s >= 6) {
                if (!stable) stable_begin = time;
                ++stable;
                if (stable >= 50 && !std::isfinite(result.time_to_target_s)) result.time_to_target_s = stable_begin;
            } else stable = 0;
        }
        // Effective average curvature from successive complete-beat COM courses.
        // This deliberately excludes rapid head/tail yaw and lateral beat jitter.
        double distance = 0, rotation = 0;
        V previous_course{};
        bool seen = false;
        for (const auto& beat : beats) {
            if (beat.end > 13.) break;
            const V travel = beat.end_com - beat.begin_com;
            if (seen) {
                const double angle = signed_angle(previous_course, travel);
                if (std::isfinite(angle)) {
                    distance += horizontal_length(travel);
                    rotation += std::abs(angle) * radians_per_degree;
                }
            }
            previous_course = travel;
            seen = true;
        }
        if (rotation > 1e-4) result.effective_radius_mm = 1000 * distance / rotation;
    }
}

Result run(const Job& job, const std::vector<float>& gait, const std::vector<float>& weights,
           const std::vector<float>& original, Clock::time_point deadline, fast::FeedbackMode candidate_mode) {
    Result result;
    fast::World world{};
    const auto baseline_mode = candidate_mode == fast::FeedbackMode::WholeBeat90
        ? fast::FeedbackMode::WholeBeat : fast::FeedbackMode::HalfSecond;
    if (job.retention) world.navigation = job.start;
    else fast::initialize(world, job.start, job.angle_deg * radians_per_degree,
        job.arm ? candidate_mode : baseline_mode);
    if (world.fault) {
        result.fault = world.fault;
        result.first_failure_before = result.first_failure_after = world;
        return result;
    }
    world.horizon_controls = job.retention ? retention_controls : turn_total_controls;
    if (!job.retention) {
        world.navigation.legs[0].end_control = turn_controls;
        world.navigation.legs[1].end_control = turn_total_controls;
    }
    auto& navigation = world.navigation;
    auto& motion = navigation.motion;
    tail::prepare_navigation(navigation);
    if (job.retention) fast::set_feedback_mode(world,
        job.arm ? candidate_mode : baseline_mode);
    result.incoming = job.retention ? navigation.legs[0].heading : motion.entry_course;
    result.target = navigation.legs[0].heading;
    Actor actor(tail::yaw_offset_spec()), baseline(tail::yaw_offset_spec());
    actor.params() = weights;
    baseline.params() = original;
    result.samples.reserve(world.horizon_controls * tail::turn_substeps_per_control + 1);
    result.trace.reserve(world.horizon_controls);
    const auto capture = [&](double time, const tail::Snapshot& pose) {
        const auto& body = motion.body;
        // The native step already certifies every accepted substep. Avoid
        // duplicating terrain work just to retain joint/COM beat samples.
        const float gap_mm = time == 0
            ? static_cast<float>(1000 * tail::submerged_gap_m(larger_pond, body, pose)) : 0.f;
        const float depth_mm = 1000 * std::abs(motion.request.origin_m.y - pose.com_m.y);
        result.samples.push_back({time, pose.com_m, body.angle[0], body.velocity[6], body.angle[2],
            body.velocity[8], pose.tilt_deg, depth_mm, gap_mm});
        result.peak_tilt = std::max(result.peak_tilt, double(pose.tilt_deg));
        result.peak_depth_mm = std::max(result.peak_depth_mm, double(depth_mm));
        if (time == 0) result.minimum_clearance_mm = std::min(result.minimum_clearance_mm, double(gap_mm));
    };
    capture(0, tail::snapshot(motion.body));
    if (!(result.minimum_clearance_mm > 0)) result.fault = 2;
    for (int control = 0; control < world.horizon_controls && !result.fault; ++control) {
        if ((control & 15) == 0 && Clock::now() >= deadline) {
            result.fault = 4;
            result.first_failure_before = result.first_failure_after = world;
            break;
        }
        tail::prepare_navigation(navigation);
        if (control % tail::launch_action_controls == 0) {
            float observation[tail::navigation_inputs], latent[tail::motor_outputs];
            if (world.feedback_mode == fast::FeedbackMode::HalfSecond)
                tail::observe_yaw_offset(navigation, world.applied_offset, observation);
            else fast::observe(world, observation);
            actor.forward(observation, 1);
            fast::frozen_gait_latent(world, gait.data(), latent);
            const float requested = actor.output()[0];
            if (!std::isfinite(requested) || !std::isfinite(latent[0]) || !std::isfinite(latent[1])) {
                result.fault = 3;
                result.first_failure_before = result.first_failure_after = world;
                break;
            }
            tail::decode_yaw_offset(navigation, latent, requested, world.applied_offset, world.targets);
            if (job.retention) {
                float original_observation[tail::navigation_inputs], reference[tail::motor_outputs];
                tail::observe_yaw_offset(navigation, 0, original_observation);
                baseline.forward(original_observation, 1);
                float original_applied = 0;
                tail::decode_yaw_offset(navigation, latent, baseline.output()[0], original_applied, reference);
                const float direct[tail::motor_outputs]{
                    cudarl::nn::activate(cudarl::nn::Activation::Tanh, latent[0]),
                    cudarl::nn::activate(cudarl::nn::Activation::Tanh, latent[1])};
                if (world.applied_offset != 0 || original_applied != 0 ||
                    std::memcmp(reference, world.targets, sizeof reference) ||
                    std::memcmp(direct, world.targets, sizeof direct)) {
                    result.bypass_exact = false; result.fault = 6;
                    result.first_failure_before = result.first_failure_after = world;
                    break;
                }
                ++result.exact_bypass;
            }
        }
        world.pose = tail::snapshot(motion.body);
        const auto before = world;
        int observer_call = 0;
        const auto step = tail::step_turn_control(motion, world.targets, larger_pond,
            [&](const tail::Snapshot& pose) {
                fast::record_feedback_substep(world, pose);
                // Native control reports its initial pose once. Adjacent controls
                // share that sample; retain only the 16 newly accepted substeps.
                if (observer_call++) capture(control * control_s + (observer_call - 1) * substep_s, pose);
            });
        result.attempted += step.attempted_substeps;
        result.accepted += step.accepted_substeps;
        result.minimum_clearance_mm = std::min(result.minimum_clearance_mm, double(step.min_gap_m) * 1000);
        result.peak_tilt = std::max(result.peak_tilt, double(step.max_tilt_deg));
        result.peak_depth_mm = std::max(result.peak_depth_mm, double(step.max_depth_error_m) * 1000);
        result.fault = step.fault;
        if (!result.fault && (result.peak_tilt > 10 || result.peak_depth_mm > 1)) result.fault = 5;
        const auto pose = tail::snapshot(motion.body);
        world.pose = pose;
        if (!step.fault) fast::record_feedback_control(world);
        if (result.fault) {
            result.first_failure_before = before;
            world.pose = pose;
            world.fault = result.fault;
            world.done = true;
            result.first_failure_after = world;
        } else ++result.controls;
        const V rolling = tail::turn_course(motion, pose.com_m, tail::steering_history_controls);
        result.trace.push_back({control + 1, pose.com_m,
            std::abs(signed_angle(result.target, rolling)), 1000 * horizontal_length(rolling) / .5,
            1000 * physics::dot(rolling, result.target) / .5,
            0, world.applied_offset, world.targets[0], world.targets[1], result.fault});
        const auto feedback = fast::whole_beat_sample(world);
        auto& row = result.trace.back();
        row.feedback_error_deg = feedback.reliable
            ? std::abs(signed_angle(result.target, feedback.velocity_m_s)) : nan;
        row.feedback_speed_mm_s = 1000 * horizontal_length(feedback.velocity_m_s);
        row.feedback_period_s = feedback.period_s;
        row.feedback_period_measured = feedback.period_measured;
        row.feedback_fresh = feedback.fresh;
        row.feedback_reliable = feedback.reliable;
    }
    result.final_state = navigation;
    summarize(result, job);
    return result;
}

bool motion_pass(const Window& window) {
    return window.complete && window.projected_mm_s >= 6 && window.course_error_deg <= 5 &&
        window.minimum_rolling_mm_s >= 6;
}
bool stroke_pass(const Rhythm& rhythm, const Rhythm& baseline) {
    return rhythm.beats > 0 && baseline.beats > 0 && baseline.frequency > 0 && baseline.swing > 0 &&
        rhythm.frequency >= .8 * baseline.frequency && rhythm.swing >= .8 * baseline.swing;
}
bool pass(const Job& job, const Result& result, const Result& baseline) {
    if (result.fault || result.controls != (job.retention ? retention_controls : turn_total_controls) ||
        result.peak_tilt > 10 || result.peak_depth_mm > 1 || !(result.minimum_clearance_mm > 0) ||
        !result.bypass_exact) return false;
    for (int i = 0; i < 2; ++i) {
        if (!motion_pass(result.windows[i])) return false;
        if (!job.retention && (!(result.windows[i].achieved_deg * job.angle_deg > 0) ||
            !stroke_pass(result.windows[i].root, baseline.windows[i].root) ||
            !stroke_pass(result.windows[i].passive, baseline.windows[i].passive))) return false;
    }
    return true;
}

void evaluate(int argc, char** argv) {
    if (argc != 7 && argc != 8) throw std::runtime_error(
        "Usage: pond_flexible_tail_steering_evaluate FITTED BASELINE_OFFSET CANDIDATE_OFFSET EVALTURN EVALSTRAIGHT OUT [half-second|whole-beat|whole-beat-90]");
    const std::string feedback_name = argc == 8 ? argv[7] : "half-second";
    const auto feedback_mode = feedback_name == "whole-beat-90" ? fast::FeedbackMode::WholeBeat90
        : feedback_name == "whole-beat" ? fast::FeedbackMode::WholeBeat
        : feedback_name == "half-second" ? fast::FeedbackMode::HalfSecond
        : throw std::runtime_error("Unknown candidate feedback mode");
    const bool angle90 = feedback_mode == fast::FeedbackMode::WholeBeat90;
    const int turn_cases_per_arm = angle90 ? 32 : 24;
    const int maximum_substeps = 2 * (turn_cases_per_arm * turn_total_controls + 12 * retention_controls)
        * tail::turn_substeps_per_control;
    const char* baseline_feedback = angle90 ? "whole-beat" : "half-second";
    const char* baseline_label = angle90 ? "wholebeat56_baseline" : "update8_original";
    const auto started = Clock::now();
    const auto gait = tail::read_navigation_weights(argv[1], tail::NavigationMode::Gait);
    const auto original = tail::read_yaw_offset_weights(argv[2]);
    const auto candidate = tail::read_yaw_offset_weights(argv[3]);
    fast::validate_frozen_gait(gait);
    fast::validate_actor(original);
    fast::validate_actor(candidate);
    const auto turns = tail::read_navigation_worlds(argv[4]);
    const auto straight = tail::read_navigation_worlds(argv[5]);
    if (turns.size() != 8 || straight.size() != 12) throw std::runtime_error("Expected eight turn and twelve retention starts");
    const fs::path output(argv[6]);
    if (!fs::create_directory(output)) throw std::runtime_error("Output directory must be new");
    const double timeout = env_number("CUDARL_STEERING_EVAL_TIMEOUT_S", 60, .1, 60);
    const auto deadline = started + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(timeout));
    const unsigned automatic_threads = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    const int worker_count = static_cast<int>(env_number("CUDARL_STEERING_EVAL_THREADS", automatic_threads, 1, 64));
    std::vector<Job> jobs;
    for (int phase = 0; phase < 4; ++phase) {
        if (turns[phase * 2].leg_count != 2 || turns[phase * 2].motion.controls != 0)
            throw std::runtime_error("Expected archived whole-beat starts");
        const std::vector<int> magnitudes = angle90 ? std::vector<int>{15, 30, 60, 90} : std::vector<int>{15, 30, 60};
        for (const int magnitude : magnitudes) for (const int sign : {-1, 1}) for (int arm = 0; arm < 2; ++arm)
            jobs.push_back({"phase_" + std::to_string(phase) + (sign < 0 ? "_right_" : "_left_") +
                std::to_string(magnitude), arm, phase, float(sign * magnitude), false, turns[phase * 2]});
    }
    for (int index = 0; index < 12; ++index) {
        if (straight[index].leg_count != 1 || straight[index].legs[0].choice != tail::Choice::Swim ||
            straight[index].legs[0].end_control != retention_controls)
            throw std::runtime_error("Unexpected original retention schedule");
        for (int arm = 0; arm < 2; ++arm)
            jobs.push_back({retention_names[index], arm, index, 0, true, straight[index]});
    }
    std::vector<Result> results(jobs.size());
    std::atomic<size_t> next{0};
    std::vector<std::thread> workers;
    const auto physics_started = Clock::now();
    for (int worker = 0; worker < worker_count; ++worker) workers.emplace_back([&] {
        while (true) {
            const auto index = next.fetch_add(1);
            if (index >= jobs.size()) return;
            try { results[index] = run(jobs[index], gait, jobs[index].arm ? candidate : original, original, deadline,
                feedback_mode); }
            catch (const std::exception& error) { results[index].fault = 6; results[index].exception = error.what(); }
        }
    });
    for (auto& worker : workers) worker.join();
    const double physics_seconds = std::chrono::duration<double>(Clock::now() - physics_started).count();
    const auto writing_started = Clock::now();
    std::ofstream cases(output / "cases.csv"), windows(output / "windows.csv"), controls(output / "controls.csv"),
        beats(output / "beats.csv"), summary(output / "result.json");
    for (auto* out : {&cases, &windows, &controls, &beats, &summary}) {
        out->exceptions(std::ios::failbit | std::ios::badbit); *out << std::setprecision(10);
    }
    cases << "case,policy,kind,target_deg,passed,fault,controls,attempted_substeps,accepted_substeps,"
        "peak_tilt_deg,peak_depth_error_mm,min_certified_clearance_mm,response_s,target_1s_begin_s,"
        "effective_complete_beat_radius_mm,exact_bypass_decisions\n";
    windows << "case,policy,window,complete,projected_mm_s,course_error_deg,achieved_deg,min_rolling_mm_s,"
        "root_complete_beats,root_frequency_hz,root_swing_rad,root_fixed_reversals,passive_complete_beats,"
        "passive_frequency_hz,passive_swing_rad,passive_fixed_reversals,stroke_pass\n";
    controls << "case,policy,control,time_s,com_x_m,com_y_m,com_z_m,course_error_deg,rolling_speed_mm_s,"
        "projected_rolling_mm_s,applied_yaw_offset,yaw_target,pitch_target,fault,"
        "whole_beat_course_error_deg,whole_beat_speed_mm_s,whole_beat_period_s,"
        "whole_beat_period_measured,whole_beat_fresh,whole_beat_reliable\n";
    beats << "case,policy,axis,begin_s,end_s,period_s,swing_rad,com_x_start,com_z_start,com_x_end,com_z_end\n";
    int turning[2]{}, retention[2]{}, from_rest[2]{}, left[2]{}, right[2]{}, by_angle[2][4]{}, faults = 0, global_faults = 0;
    long long attempted = 0, accepted = 0, exact = 0;
    bool all_bypass_exact = true, retention_final_state_exact = true;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const auto& job = jobs[i]; const auto& result = results[i];
        const auto& baseline = results[job.arm ? i - 1 : i];
        const char* policy = job.arm ? "candidate" : baseline_label;
        const bool passed = pass(job, result, baseline);
        if (job.retention) {
            retention[job.arm] += passed;
            if (job.phase < 8) from_rest[job.arm] += passed;
            if (job.arm) retention_final_state_exact = retention_final_state_exact &&
                std::memcmp(&result.final_state, &baseline.final_state, sizeof result.final_state) == 0;
        }
        else {
            turning[job.arm] += passed;
            (job.angle_deg > 0 ? left[job.arm] : right[job.arm]) += passed;
            by_angle[job.arm][std::abs(job.angle_deg) == 15 ? 0 : std::abs(job.angle_deg) == 30 ? 1
                : std::abs(job.angle_deg) == 60 ? 2 : 3] += passed;
        }
        attempted += result.attempted; accepted += result.accepted; exact += result.exact_bypass;
        faults += result.fault != 0;
        global_faults += result.fault == 1 || result.fault == 3 || result.fault == 4 || result.fault == 6;
        all_bypass_exact = all_bypass_exact && result.bypass_exact;
        cases << job.name << ',' << policy << ',' << (job.retention ? "straight_start_retention" : "turn") << ','
              << job.angle_deg << ',' << passed << ',' << result.fault << ',' << result.controls << ','
              << result.attempted << ',' << result.accepted << ',' << result.peak_tilt << ',' << result.peak_depth_mm << ',';
        number(cases, result.minimum_clearance_mm); cases << ','; number(cases, result.response_s); cases << ',';
        number(cases, result.time_to_target_s); cases << ','; number(cases, result.effective_radius_mm);
        cases << ',' << result.exact_bypass << '\n';
        for (int w = 0; w < 2; ++w) {
            const auto& window = result.windows[w];
            windows << job.name << ',' << policy << ',' << (w ? "onward_or_final" : "late_turn_or_previous") << ','
                << window.complete << ','; number(windows, window.projected_mm_s); windows << ',';
            number(windows, window.course_error_deg); windows << ','; number(windows, window.achieved_deg); windows << ',';
            number(windows, window.minimum_rolling_mm_s); windows << ',' << window.root.beats << ',';
            number(windows, window.root.frequency); windows << ','; number(windows, window.root.swing);
            windows << ',' << window.root.reversals << ',' << window.passive.beats << ',';
            number(windows, window.passive.frequency); windows << ','; number(windows, window.passive.swing);
            windows << ',' << window.passive.reversals << ',';
            if (!job.retention) windows << (stroke_pass(window.root, baseline.windows[w].root) &&
                stroke_pass(window.passive, baseline.windows[w].passive));
            windows << '\n';
        }
        for (const auto& control : result.trace) {
            controls << job.name << ',' << policy << ',' << control.control << ',' << control.control * control_s << ','
                << control.com.x << ',' << control.com.y << ',' << control.com.z << ',';
            number(controls, control.course_error); controls << ',' << control.rolling_speed_mm_s << ','
                << control.projected_rolling_mm_s << ',' << control.applied_offset << ',' << control.yaw_target << ','
                << control.pitch_target << ',' << control.fault << ',';
            number(controls, control.feedback_error_deg);
            controls << ',' << control.feedback_speed_mm_s << ',' << control.feedback_period_s << ','
                << control.feedback_period_measured << ',' << control.feedback_fresh << ','
                << control.feedback_reliable << '\n';
        }
        for (int axis = 0; axis < 2; ++axis) for (const auto& beat : complete_beats(result.samples, axis == 0))
            beats << job.name << ',' << policy << ',' << (axis == 0 ? "root" : "passive") << ',' << beat.begin << ','
                << beat.end << ',' << beat.end - beat.begin << ',' << beat.swing << ',' << beat.begin_com.x << ','
                << beat.begin_com.z << ',' << beat.end_com.x << ',' << beat.end_com.z << '\n';
        if (result.fault) {
            cudarl::checkpoint::Writer failure;
            failure.value<uint32_t>(0x46534631); failure.value<uint32_t>(1);
            failure.value<int>(result.fault); failure.value(result.first_failure_before); failure.value(result.first_failure_after);
            cudarl::checkpoint::write_atomic((output / (job.name + "-" + policy + "-failure.bin")).u8string(), failure.data);
            if (!result.exception.empty()) {
                std::ofstream failure_text(output / (job.name + "-" + policy + "-error.txt"));
                failure_text << result.exception << '\n';
            }
        }
    }
    for (auto* out : {&cases, &windows, &controls, &beats}) out->flush();
    const double writing_seconds = std::chrono::duration<double>(Clock::now() - writing_started).count();
    summary << "{\"schema\":1,\"turn_cases_per_arm\":" << turn_cases_per_arm
        << ",\"retention_cases_per_arm\":12,\"total_trajectories\":" << jobs.size()
        << ",\"baseline_feedback\":\"" << baseline_feedback << "\",\"candidate_feedback\":\"" << feedback_name << "\","
        "\"turn_control_cap\":850,\"retention_control_cap\":600,\"maximum_substeps\":" << maximum_substeps << ','
        << "\"baseline_turn_passes\":" << turning[0] << ",\"candidate_turn_passes\":" << turning[1]
        << ",\"baseline_left_passes\":" << left[0] << ",\"baseline_right_passes\":" << right[0]
        << ",\"candidate_left_passes\":" << left[1] << ",\"candidate_right_passes\":" << right[1]
        << ",\"baseline_15_passes\":" << by_angle[0][0] << ",\"candidate_15_passes\":" << by_angle[1][0]
        << ",\"baseline_30_passes\":" << by_angle[0][1] << ",\"candidate_30_passes\":" << by_angle[1][1]
        << ",\"baseline_60_passes\":" << by_angle[0][2] << ",\"candidate_60_passes\":" << by_angle[1][2]
        << ",\"baseline_90_passes\":" << by_angle[0][3] << ",\"candidate_90_passes\":" << by_angle[1][3]
        << ",\"baseline_straight_retention\":" << retention[0] << ",\"candidate_straight_retention\":" << retention[1]
        << ",\"baseline_from_rest_passes\":" << from_rest[0] << ",\"candidate_from_rest_passes\":" << from_rest[1]
        << ",\"all_inactive_commands_exact\":" << (all_bypass_exact && exact == 3600 ? "true" : "false")
        << ",\"paired_retention_final_states_exact\":" << (retention_final_state_exact && exact == 3600 ? "true" : "false")
        << ",\"exact_bypass_decisions\":" << exact << ",\"station_holding_tested\":false,\"faulted_trajectories\":" << faults
        << ",\"global_faults\":" << global_faults
        << ",\"attempted_substeps\":" << attempted << ",\"accepted_substeps\":" << accepted
        << ",\"optimizer_updates\":0,\"cpu_workers\":" << worker_count << ",\"physics_and_scoring_s\":" << physics_seconds
        << ",\"evidence_writing_s\":" << writing_seconds << ",\"total_s\":"
        << std::chrono::duration<double>(Clock::now() - started).count() << "}\n";
    summary.flush();
    std::cout << "baseline_turns=" << turning[0] << '/' << turn_cases_per_arm << " candidate_turns=" << turning[1]
        << '/' << turn_cases_per_arm << " baseline_straight=" << retention[0] << "/12 candidate_straight=" << retention[1]
        << "/12 attempted_substeps=" << attempted << " faults=" << faults
        << " evaluation_seconds=" << std::chrono::duration<double>(Clock::now() - started).count() << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try { evaluate(argc, argv); return 0; }
    catch (const std::exception& error) { std::cerr << "steering evaluation: " << error.what() << '\n'; return 2; }
}
