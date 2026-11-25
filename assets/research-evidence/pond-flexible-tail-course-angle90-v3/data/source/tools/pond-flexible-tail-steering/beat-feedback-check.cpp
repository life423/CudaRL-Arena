#include "rl/flexible_tail_fast_steering.h"
#include "nn/mlp_cpu.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ft = cudarl::flexible_tail;
namespace fast = ft::fast;
namespace p = cudarl::physics;
namespace nn = cudarl::nn;
namespace fs = std::filesystem;

struct Exact {
    int values = 0, differences = 0;
    float maximum = 0;
    void add(float a, float b) {
        ++values;
        differences += std::memcmp(&a, &b, sizeof a) != 0;
        maximum = std::fmax(maximum, std::fabs(a - b));
    }
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

double angle_deg(p::V v) { return std::atan2(v.z, v.x) * 180. / ft::pi; }

struct Synthetic {
    int assertions = 0, samples = 0;
    double half_square = 0, beat_square = 0, max_beat_angle = 0, max_speed_error = 0;
    double period = 0;
    bool stopped_unreliable = false, stale_unreliable = false, mirrored_exact = true;
};

Synthetic synthetic(const ft::NavigationWorld& source) {
    Synthetic result;
    auto motion = source.motion;
    const auto origin = ft::snapshot(motion.body).com_m;
    motion.controls = 0;
    motion.history_next = 0;
    motion.body.velocity[6] = 1.f;
    const auto point = [&](double t, bool mirror = false) {
        return origin + p::V{float(.008 * t), 0, float((mirror ? -1 : 1) * .001 * std::sin(2 * ft::pi * t))};
    };
    for (int i = 0; i < ft::turn_history_controls; ++i)
        motion.history[i] = point((i - ft::turn_history_controls) * double(ft::motor_control_s));
    ft::BeatCourseFeedback tracker, mirrored;
    tracker.initialize(motion);
    for (int i = 0; i < ft::turn_history_controls; ++i)
        motion.history[i] = point((i - ft::turn_history_controls) * double(ft::motor_control_s), true);
    mirrored.initialize(motion);
    for (int step = 1; step <= 4800; ++step) {
        const double time = step * double(ft::motor_substep_s);
        const float rate = float(std::cos(2 * ft::pi * time));
        tracker.add_substep(rate, step, point(time));
        mirrored.add_substep(rate, step, point(time, true));
        if (step % ft::turn_substeps_per_control) continue;
        tracker.add_control(point(time));
        mirrored.add_control(point(time, true));
        const auto beat = tracker.sample(), other = mirrored.sample();
        if (time < 3.) continue;
        require(beat.period_measured && beat.reliable && beat.fresh, "Full measured beat unavailable in synthetic straight swim");
        result.assertions += 3;
        p::V half_start{}, current{};
        require(tracker.position(25, half_start) && tracker.position(0, current), "Synthetic history unavailable");
        const auto half = (current - half_start) / .5f;
        const double half_error = angle_deg(half), beat_error = angle_deg(beat.velocity_m_s);
        result.half_square += half_error * half_error;
        result.beat_square += beat_error * beat_error;
        result.max_beat_angle = std::fmax(result.max_beat_angle, std::abs(beat_error));
        result.max_speed_error = std::fmax(result.max_speed_error, std::abs(double(beat.velocity_m_s.x) - .008));
        result.period = beat.period_s;
        result.mirrored_exact = result.mirrored_exact &&
            std::abs(beat.velocity_m_s.x - other.velocity_m_s.x) < 1e-7f &&
            std::abs(beat.velocity_m_s.z + other.velocity_m_s.z) < 1e-7f;
        ++result.samples;
    }
    require(result.samples > 100 && result.max_beat_angle < .1 && result.max_speed_error < 1e-6,
        "Whole-beat feedback failed synthetic constant-course accuracy");
    require(result.beat_square * 100 < result.half_square && result.mirrored_exact,
        "Whole-beat feedback failed noise/reflection check");
    result.assertions += 5;
    // Retain real time and motion history while the synthetic body stops.
    // A fixed request or head direction must not invent continuing travel.
    const auto stopped = point(6.);
    for (int step = 4801; step <= 8000; ++step) {
        tracker.add_substep(0, step, stopped);
        if (step % ft::turn_substeps_per_control == 0) tracker.add_control(stopped);
    }
    const auto stop = tracker.sample();
    result.stopped_unreliable = !stop.reliable && p::norm(stop.velocity_m_s) < 1e-8f;
    result.stale_unreliable = !stop.fresh;
    require(result.stopped_unreliable && result.stale_unreliable,
        "Stopped swimmer retained stale or invented course");
    result.assertions += 2;
    // The full tracker is value state: byte-copying and continuing the same
    // next sample must preserve every history/crossing/interpolation field.
    auto restored = tracker;
    tracker.add_substep(.5f, 8001, stopped);
    restored.add_substep(.5f, 8001, stopped);
    require(std::memcmp(&tracker, &restored, sizeof tracker) == 0,
        "Tracker copy did not preserve exact continuation state");
    ++result.assertions;
    return result;
}

int main(int argc, char** argv) {
    try {
        const auto started = std::chrono::steady_clock::now();
        if (argc != 7 && argc != 8) throw std::runtime_error(
            "Usage: beat_feedback_check FITTED OFFSET TRAIN_STARTS TURN_STARTS STRAIGHT_STARTS OUT [whole-beat|whole-beat-90]");
        const std::string mode_name = argc == 8 ? argv[7] : "whole-beat";
        require(mode_name == "whole-beat" || mode_name == "whole-beat-90", "Unknown feedback-check mode");
        const bool angle90 = mode_name == "whole-beat-90";
        const auto gait = ft::read_navigation_weights(argv[1], ft::NavigationMode::Gait);
        const auto offset = ft::read_yaw_offset_weights(argv[2]);
        const auto training = ft::read_navigation_worlds(argv[3]);
        const auto heldout = ft::read_navigation_worlds(argv[4]);
        const auto straight = ft::read_navigation_worlds(argv[5]);
        fast::validate_frozen_gait(gait); fast::validate_actor(offset);
        require(training.size() == 256 && heldout.size() == 8 && straight.size() == 12,
            "Unexpected frozen fixture counts");
        const fs::path output(argv[6]);
        require(fs::create_directory(output), "Diagnostic output must be new");
        nn::CpuMlp<float> full_gait(ft::navigation_motor_spec()), actor(ft::yaw_offset_spec());
        full_gait.params() = gait; actor.params() = offset;
        Exact sparse, historical, actor_mean, critic, bypass, unchanged_features;
        double transfer_maximum = 0, feature90_maximum = 0;
        int transfer_values = 0, feature90_values = 0;
        bool angle90_monotonic = true, angle90_reflection = true, angle90_distinct = true;
        const auto transferred = angle90 ? fast::initial_actor90(offset) : offset;
        nn::CpuMlp<float> actor90(ft::yaw_offset_spec());
        actor90.params() = transferred;
        int requests = 0, bodies_exact = 0, histories_exact = 0, finite_critic = 0;
        float request_error = 0;
        std::ofstream observations(output / "observations.csv");
        observations << "source,angle,half_second_feature,whole_beat_feature,measured_period,period_s,reliable\n";
        for (size_t index = 0; index < training.size(); ++index) for (int magnitude : {15,30,45,60}) for (int sign : {-1,1}) {
            ++requests;
            const auto& source = training[index];
            fast::World old{}, beat{};
            const float angle = sign * magnitude * ft::pi / 180.f;
            fast::initialize(old, source, angle, fast::FeedbackMode::HalfSecond);
            fast::initialize(beat, source, angle, fast::FeedbackMode::WholeBeat);
            require(!old.fault && !beat.fault, "Invalid initialized diagnostic world");
            bodies_exact += std::memcmp(&beat.navigation.motion.body, &source.motion.body, sizeof source.motion.body) == 0;
            histories_exact += std::memcmp(beat.navigation.motion.history, source.motion.history, sizeof source.motion.history) == 0;
            request_error = std::fmax(request_error, std::fabs(ft::yaw_offset_error(beat.navigation.motion.entry_course,
                beat.navigation.motion.request.heading) - angle));
            float gait_observation[ft::navigation_inputs], latent[ft::motor_outputs];
            ft::observe_gait(beat.navigation, gait_observation);
            full_gait.forward(gait_observation, 1);
            fast::frozen_gait_latent(beat, gait.data(), latent);
            for (int axis = 0; axis < ft::motor_outputs; ++axis)
                sparse.add(nn::activate(nn::Activation::Tanh, latent[axis]), full_gait.output()[axis]);
            old.applied_offset = beat.applied_offset = (int(index) % 3 - 1) * .03125f;
            float original[ft::navigation_inputs], old_observation[ft::navigation_inputs], beat_observation[ft::navigation_inputs];
            ft::observe_yaw_offset(old.navigation, old.applied_offset, original);
            fast::observe(old, old_observation); fast::observe(beat, beat_observation);
            for (int i = 0; i < ft::navigation_inputs; ++i) {
                historical.add(original[i], old_observation[i]);
                if (i) unchanged_features.add(old_observation[i], beat_observation[i]);
            }
            actor.forward(beat_observation, 1);
            float direct = 0;
            for (int i = 0; i < 4; ++i) direct += beat_observation[i] * offset[i];
            actor_mean.add(actor.output()[0], direct);
            float critic_observation[fast::critic_inputs];
            fast::observe_critic(beat, critic_observation);
            bool finite = true;
            for (float value : critic_observation) finite = finite && std::isfinite(value);
            finite_critic += finite;
            for (int i = 0; i < 4; ++i) critic.add(critic_observation[ft::navigation_inputs + i], beat_observation[i]);
            const auto measurement = fast::whole_beat_sample(beat);
            observations << index << ',' << sign * magnitude << ',' << old_observation[0] << ',' << beat_observation[0]
                << ',' << measurement.period_measured << ',' << measurement.period_s << ',' << measurement.reliable << '\n';
        }
        if (angle90) {
            std::ofstream table(output / "course-input-table.csv");
            table << std::setprecision(12) << "signed_course_error_deg,historical_feature,new_feature\n";
            float previous_positive = -1.f;
            for (const int magnitude : {0,5,15,20,30,45,60,90}) {
                const float radians = magnitude * ft::pi / 180.f;
                const float positive = fast::course_error_input(radians, fast::FeedbackMode::WholeBeat90);
                const float negative = fast::course_error_input(-radians, fast::FeedbackMode::WholeBeat90);
                angle90_monotonic = angle90_monotonic && positive > previous_positive;
                angle90_distinct = angle90_distinct && positive != previous_positive;
                angle90_reflection = angle90_reflection && positive == -negative;
                previous_positive = positive;
                for (int sign : {-1,1}) table << sign * magnitude << ','
                    << fast::course_error_input(sign * radians, fast::FeedbackMode::WholeBeat) << ','
                    << fast::course_error_input(sign * radians, fast::FeedbackMode::WholeBeat90) << '\n';
            }
            for (const auto& source : training) {
                fast::World base{}, changed{};
                fast::initialize(base, source, 15.f * ft::pi / 180.f, fast::FeedbackMode::WholeBeat);
                fast::initialize(changed, source, 15.f * ft::pi / 180.f, fast::FeedbackMode::WholeBeat90);
                const auto measured = fast::feedback_sample(base);
                require(measured.reliable, "Initial actual COM course unavailable for angle90 feature check");
                for (int magnitude : {5,15,20,30,45,60,90}) for (int sign : {-1,1}) {
                    const float radians = sign * magnitude * ft::pi / 180.f;
                    const auto heading = p::axis_rotation({0,1,0}, radians).rotate(
                        measured.velocity_m_s / p::norm(measured.velocity_m_s));
                    base.navigation.motion.request.heading = changed.navigation.motion.request.heading = heading;
                    base.applied_offset = changed.applied_offset = .03125f;
                    float old_observation[ft::navigation_inputs], full_observation[ft::navigation_inputs];
                    fast::observe(base, old_observation); fast::observe(changed, full_observation);
                    const double difference = std::abs(double(full_observation[0]) - double(radians / (ft::pi / 2.f)));
                    feature90_maximum = std::fmax(feature90_maximum, difference);
                    ++feature90_values;
                    for (int i = 1; i < ft::navigation_inputs; ++i)
                        unchanged_features.add(old_observation[i], full_observation[i]);
                    if (magnitude <= 20) {
                        actor.forward(old_observation, 1); actor90.forward(full_observation, 1);
                        transfer_maximum = std::fmax(transfer_maximum,
                            std::abs(double(actor.output()[0]) - double(actor90.output()[0])));
                        ++transfer_values;
                    }
                }
            }
            require(angle90_monotonic && angle90_distinct && angle90_reflection,
                "Course input fails distinct/reflected monotonic mapping through90degrees");
            require(feature90_maximum <= 1e-6 && transfer_maximum <= 1e-6,
                "Actual course observer or small-angle actor transfer exceeded1e-6");
        }
        const std::vector<fast::FeedbackMode> bypass_modes = angle90
            ? std::vector<fast::FeedbackMode>{fast::FeedbackMode::HalfSecond, fast::FeedbackMode::WholeBeat, fast::FeedbackMode::WholeBeat90}
            : std::vector<fast::FeedbackMode>{fast::FeedbackMode::HalfSecond, fast::FeedbackMode::WholeBeat};
        for (const auto& source : straight) for (auto mode : bypass_modes)
            for (auto choice : {ft::Choice::Swim, ft::Choice::Hold}) for (float sampled : {-100.f, 0.f, 100.f}) {
                fast::World world{};
                world.navigation = source;
                world.navigation.leg_count = 1; world.navigation.active_leg = 0;
                world.navigation.motion.request.choice = choice;
                fast::set_feedback_mode(world, mode);
                float observation[ft::navigation_inputs], latent[ft::motor_outputs], targets[ft::motor_outputs];
                fast::observe(world, observation);
                for (float value : observation) bypass.add(value, 0.f);
                fast::frozen_gait_latent(world, gait.data(), latent);
                float applied = .2f;
                ft::decode_yaw_offset(world.navigation, latent, sampled, applied, targets);
                bypass.add(applied, 0.f);
                for (int axis = 0; axis < ft::motor_outputs; ++axis)
                    bypass.add(targets[axis], nn::activate(nn::Activation::Tanh, latent[axis]));
            }
        const auto syn = synthetic(training.front());
        const bool pass = sparse.differences == 0 && historical.differences == 0 && actor_mean.differences == 0 &&
            critic.differences == 0 && bypass.differences == 0 && unchanged_features.differences == 0 &&
            bodies_exact == requests && histories_exact == requests && finite_critic == requests && request_error < 1e-5f;
        if (pass && angle90)
            ft::write_yaw_offset_weights((output / "initial-actor90.bin").u8string(), transferred);
        std::ofstream report(output / "report.json");
        report << std::setprecision(12) << "{\"schema\":1,\"pass\":" << (pass ? "true" : "false")
            << ",\"physics_steps\":0,\"optimizer_calls\":0,\"request_cases\":" << requests
            << ",\"body_copies_exact\":" << bodies_exact << ",\"history_copies_exact\":" << histories_exact
            << ",\"finite_critic_rows\":" << finite_critic << ",\"maximum_request_error_rad\":" << request_error << ',';
        const auto emit = [&](const char* name, const Exact& check) {
            report << '"' << name << "\":{\"values\":" << check.values << ",\"bit_differences\":" << check.differences
                << ",\"maximum_absolute_difference\":" << check.maximum << "},";
        };
        emit("sparse_gait", sparse); emit("historical_observer", historical); emit("actor_mean", actor_mean);
        emit("critic_steering", critic); emit("inactive_bypass", bypass); emit("unchanged_actor_features", unchanged_features);
        report << "\"angle90_enabled\":" << (angle90 ? "true" : "false")
            << ",\"angle90_monotonic\":" << (angle90_monotonic ? "true" : "false")
            << ",\"angle90_distinct\":" << (angle90_distinct ? "true" : "false")
            << ",\"angle90_reflection\":" << (angle90_reflection ? "true" : "false")
            << ",\"angle90_observer_values\":" << feature90_values
            << ",\"angle90_observer_maximum_difference\":" << feature90_maximum
            << ",\"small_angle_transfer_values\":" << transfer_values
            << ",\"small_angle_transfer_maximum_difference\":" << transfer_maximum
            << ",\"small_angle_transfer_tolerance\":0.000001,\"synthetic_assertions\":" << syn.assertions << ",\"synthetic_scored_samples\":" << syn.samples
            << ",\"half_second_course_rms_deg\":" << std::sqrt(syn.half_square / syn.samples)
            << ",\"whole_beat_course_rms_deg\":" << std::sqrt(syn.beat_square / syn.samples)
            << ",\"whole_beat_maximum_course_error_deg\":" << syn.max_beat_angle
            << ",\"whole_beat_maximum_speed_error_m_s\":" << syn.max_speed_error
            << ",\"measured_period_s\":" << syn.period
            << ",\"stopped_unreliable\":" << (syn.stopped_unreliable ? "true" : "false")
            << ",\"stale_unreliable\":" << (syn.stale_unreliable ? "true" : "false")
            << ",\"reflection_matches\":" << (syn.mirrored_exact ? "true" : "false")
            << ",\"elapsed_s\":" << std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count() << "}\n";
        std::cout << "pass=" << pass << " synthetic_assertions=" << syn.assertions << " requests=" << requests << '\n';
        return pass ? 0 : 2;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
