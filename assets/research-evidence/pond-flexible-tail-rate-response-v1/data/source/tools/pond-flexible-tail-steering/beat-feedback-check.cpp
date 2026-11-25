#include "rl/flexible_tail_fast_steering.h"
#include "rl/flexible_tail_stroke_metrics.h"
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

p::V reflect_vector(p::V v) { v.z = -v.z; return v; }
ft::Choice reflect_choice(ft::Choice choice) {
    if (choice == ft::Choice::TurnLeft) return ft::Choice::TurnRight;
    if (choice == ft::Choice::TurnRight) return ft::Choice::TurnLeft;
    return choice;
}
ft::NavigationWorld reflect_world(ft::NavigationWorld world) {
    auto& motion = world.motion;
    auto& body = motion.body;
    body.position.z = -body.position.z;
    body.orientation.x = -body.orientation.x;
    body.orientation.y = -body.orientation.y;
    for (const int axis : {0, 2}) body.angle[axis] = -body.angle[axis];
    for (const int index : {2, 3, 4, 6, 8}) body.velocity[index] = -body.velocity[index];
    for (auto& position : motion.history) position.z = -position.z;
    motion.last_targets[0] = -motion.last_targets[0];
    motion.entry_course = reflect_vector(motion.entry_course);
    motion.turn_radians = -motion.turn_radians;
    motion.request.origin_m = reflect_vector(motion.request.origin_m);
    motion.request.heading = reflect_vector(motion.request.heading);
    motion.request.terminal_velocity_m_s = reflect_vector(motion.request.terminal_velocity_m_s);
    motion.request.choice = reflect_choice(motion.request.choice);
    for (auto& leg : world.legs) {
        leg.goal_m = reflect_vector(leg.goal_m);
        leg.heading = reflect_vector(leg.heading);
        leg.choice = reflect_choice(leg.choice);
    }
    return world;
}

struct PhaseChecks {
    Exact prefix, initial_mean, initial_targets, checkpoint;
    int states = 0, reflected_features = 0;
    float reflection_maximum = 0, feature90_maximum = 0;
    bool phase_changes = false, speed_changes = false, legacy_rejected = false;
};

struct MainStrokeChecks {
    int clean_cycles = 0, ripple_cycles = 0, tiny_cycles = 0;
    double clean_hz = 0, ripple_hz = 0, clean_swing = 0, ripple_swing = 0;
    bool pass = false;
};

MainStrokeChecks main_stroke_checks() {
    MainStrokeChecks result;
    ft::stroke::MainStrokeTracker clean, ripple, tiny;
    ft::stroke::Cycle cycle;
    double clean_period = 0, ripple_period = 0;
    for (int i = 0; i <= 9600; ++i) {
        const double t = i * .00125;
        const float stroke = .35f * std::sin(float(2 * ft::pi * t));
        const float noise = .008f * std::sin(float(48 * ft::pi * t));
        const p::V com{float(.008 * t), 0, 0};
        if (clean.add(t, stroke, com, cycle)) {
            ++result.clean_cycles; clean_period += cycle.end_s - cycle.begin_s;
            result.clean_swing += cycle.swing_rad;
        }
        if (ripple.add(t, stroke + noise, com, cycle)) {
            ++result.ripple_cycles; ripple_period += cycle.end_s - cycle.begin_s;
            result.ripple_swing += cycle.swing_rad;
        }
        if (tiny.add(t, noise, com, cycle)) ++result.tiny_cycles;
    }
    result.clean_hz = result.clean_cycles / clean_period;
    result.ripple_hz = result.ripple_cycles / ripple_period;
    result.clean_swing /= result.clean_cycles;
    result.ripple_swing /= result.ripple_cycles;
    p::BodyState body{};
    const float zero_quality = ft::stroke::passive_quality(body);
    body.angle[2] = .2f;
    const float amplitude_quality = ft::stroke::passive_quality(body);
    body.angle[2] = 0;
    body.velocity[8] = .2f * 4.f * ft::pi;
    const float velocity_quality = ft::stroke::passive_quality(body);
    result.pass = result.clean_cycles >= 10 && result.clean_cycles == result.ripple_cycles &&
        result.tiny_cycles == 0 && std::abs(result.clean_hz - 1) < .02 &&
        std::abs(result.ripple_hz - result.clean_hz) < .02 &&
        std::abs(result.ripple_swing - result.clean_swing) < .03 &&
        zero_quality == 0 && amplitude_quality == 1 && velocity_quality >= .99999f;
    require(result.pass, "Main-stroke prominence or causal passive quality check failed");
    return result;
}

struct ProgramChecks {
    Exact initial_means, initial_commands, inactive_commands, checkpoint;
    int request_rows = 0;
    double request_feature_error = 0;
    bool passive_inputs_measured = false;
    bool request_integration_preserves_body = false;
};

ProgramChecks program_checks(const std::string& source_path, ft::steering::Study study,
    const std::vector<float>& gait, const std::vector<ft::NavigationWorld>& starts, const fs::path& output,
    const std::string& material_name = {}) {
    ProgramChecks result;
    const auto source = ft::steering::read_model(source_path);
    auto candidate = ft::steering::initial_program_model(source, study);
    if (!material_name.empty()) candidate.material = ft::steering::read_material(material_name);
    ft::steering::Config before_config; before_config.study = source.study; before_config.material=source.material;
    ft::steering::Config after_config; after_config.study = study; after_config.material=candidate.material;
    for (size_t index = 0; index < starts.size(); ++index)
        for (int degrees : {-90,-60,-30,-15,15,30,60,90}) {
            fast::World before{}, after{};
            fast::initialize(before, starts[index], degrees * ft::pi / 180.f, fast::FeedbackMode::PhaseAware);
            after = before;
            float old_obs[ft::navigation_inputs], new_obs[ft::navigation_inputs], old_mean[3]{}, new_mean[3]{};
            if (source.kind >= ft::steering::Kind::ProgramScalar) fast::observe_program(before, before_config, old_obs);
            else fast::observe(before, old_obs);
            fast::observe_program(after, after_config, new_obs);
            ft::steering::mean(source, old_obs, old_mean);
            ft::steering::mean(candidate, new_obs, new_mean);
            for (int axis = 0; axis < 3; ++axis) result.initial_means.add(old_mean[axis], new_mean[axis]);
            fast::decode_program(before, old_mean, gait.data(), before_config);
            fast::decode_program(after, new_mean, gait.data(), after_config);
            result.initial_commands.add(before.applied_offset, after.applied_offset);
            for (int axis = 0; axis < ft::motor_outputs; ++axis)
                result.initial_commands.add(before.targets[axis], after.targets[axis]);
            require(std::memcmp(&before.navigation.motion.body, &starts[index].motion.body,
                sizeof(p::BodyState)) == 0, "Initial program inference moved a physical body");
            require(std::memcmp(before.navigation.motion.history, starts[index].motion.history,
                sizeof before.navigation.motion.history) == 0, "Initial program inference changed course history");
            for (float speed : {.006f,.008f,.012f}) for (int rate : {-8,-4,-2,0,2,4,8}) {
                auto config = after_config;
                config.target_speed_m_s = speed;
                config.desired_course_rate_rad_s = rate * ft::pi / 180.f;
                require(ft::steering::valid_config(config), "Declared speed/rate request invalid");
                fast::observe_program(after, config, new_obs);
                result.request_feature_error = std::fmax(result.request_feature_error,
                    std::fabs(new_obs[ft::steering::RequestedSpeed] - (speed / .008f - 1)));
                result.request_feature_error = std::fmax(result.request_feature_error,
                    std::fabs(new_obs[ft::steering::RequestedCourseRate] - rate / 8.f));
                ++result.request_rows;
            }
        }
    fast::World measured{};
    fast::initialize(measured, starts.front(), ft::pi / 6.f, fast::FeedbackMode::PhaseAware);
    float values[ft::navigation_inputs];
    measured.navigation.motion.body.angle[2] = .07f;
    measured.navigation.motion.body.velocity[8] = -.8f;
    fast::observe_program(measured, after_config, values);
    result.passive_inputs_measured = values[ft::steering::PassiveYawPosition] == .07f / .2f &&
        values[ft::steering::PassiveYawVelocity] == -.8f / (4.f * ft::pi * .2f);
    auto held = measured;
    const auto native = held.navigation.motion.body;
    const auto history = held.navigation.motion;
    const float offset = held.applied_offset;
    const auto incoming = held.navigation.motion.request.heading;
    auto rate_config = after_config;
    rate_config.desired_course_rate_rad_s = 8.f * ft::pi / 180.f;
    for (int tick = 0; tick < 50; ++tick) fast::prepare_program_request(held, rate_config);
    result.request_feature_error = std::fmax(result.request_feature_error,
        std::fabs(ft::yaw_offset_error(incoming, held.navigation.motion.request.heading) - 8.f * ft::pi / 180.f));
    result.request_integration_preserves_body =
        std::memcmp(&native, &held.navigation.motion.body, sizeof native) == 0 &&
        std::memcmp(history.history, held.navigation.motion.history, sizeof history.history) == 0 &&
        std::memcmp(history.last_targets, held.navigation.motion.last_targets, sizeof history.last_targets) == 0 &&
        held.applied_offset == offset && held.navigation.motion.controls == history.controls;
    for (auto choice : {ft::Choice::Swim,ft::Choice::Hold}) {
        auto inactive = measured;
        inactive.navigation.active_leg=0;inactive.navigation.leg_count=1;
        inactive.navigation.legs[0].choice=choice;
        ft::prepare_navigation(inactive.navigation);
        if(ft::steering::program_enabled(inactive.navigation,after_config))continue;
        fast::observe_program(inactive,after_config,values);
        float means[3]{};ft::steering::mean(candidate,values,means);
        float gait_latent[ft::motor_outputs];fast::frozen_gait_latent(inactive,gait.data(),gait_latent);
        inactive.applied_offset=.2f;
        fast::decode_program(inactive,means,gait.data(),after_config);
        result.inactive_commands.add(inactive.applied_offset,0.f);
        for(int axis=0;axis<ft::motor_outputs;++axis)
            result.inactive_commands.add(inactive.targets[axis],nn::activate(nn::Activation::Tanh,gait_latent[axis]));
    }
    const auto path = output / "initial-program-actor.bin";
    ft::steering::write_model(path.u8string(), candidate);
    const auto restored = ft::steering::read_model(path.u8string());
    require(restored.kind == candidate.kind && restored.study == candidate.study && restored.material==candidate.material &&
        restored.frozen_base == candidate.frozen_base && restored.frozen_skip == candidate.frozen_skip,
        "Program checkpoint layout, study or skip changed");
    for (size_t i = 0; i < candidate.weights.size(); ++i) result.checkpoint.add(candidate.weights[i], restored.weights[i]);
    require(result.initial_means.differences == 0 && result.initial_commands.differences == 0 &&
        result.inactive_commands.differences == 0 && result.checkpoint.differences == 0 &&
        result.request_feature_error <= 1e-6 && result.passive_inputs_measured && result.request_integration_preserves_body,
        "Program initialization, request semantics or checkpoint equivalence failed");
    return result;
}

PhaseChecks phase_checks(const std::vector<float>& offset,
    const std::vector<float>& gait, const std::vector<ft::NavigationWorld>& training,
    const fs::path& output) {
    PhaseChecks result;
    const auto initial = ft::steering::initial_phase_model(offset);
    nn::CpuMlp<float> base(ft::steering::spec()), phase(ft::steering::spec());
    base.params() = offset;
    phase.params() = initial.weights;
    std::ofstream features(output / "phase-features.csv");
    features << std::setprecision(12) << "source,target_deg,full_course_error,root_phase_position,root_phase_velocity,"
        "lateral_speed_feature,error_speed_feature,error_phase_cos2,error_phase_sin2,error2_phase_position,error2_phase_velocity\n";
    for (size_t source = 0; source < training.size(); ++source)
        for (int magnitude : {5,15,30,45,60,90}) for (int sign : {-1,1}) {
            const float angle = sign * magnitude * ft::pi / 180.f;
            fast::World old{}, changed{}, mirrored{};
            fast::initialize(old, training[source], angle, fast::FeedbackMode::WholeBeat);
            fast::initialize(changed, training[source], angle, fast::FeedbackMode::PhaseAware);
            fast::initialize(mirrored, reflect_world(training[source]), -angle, fast::FeedbackMode::PhaseAware);
            require(!old.fault && !changed.fault && !mirrored.fault, "Phase diagnostic world initialization failed");
            old.applied_offset = changed.applied_offset = (int(source) % 3 - 1) * .03125f;
            mirrored.applied_offset = -changed.applied_offset;
            float legacy[ft::navigation_inputs], values[ft::navigation_inputs], reflected[ft::navigation_inputs];
            fast::observe(old, legacy); fast::observe(changed, values); fast::observe(mirrored, reflected);
            for (int i = 0; i < 4; ++i) result.prefix.add(legacy[i], values[i]);
            for (int i = 4; i <= 12; ++i) {
                result.reflection_maximum = std::fmax(result.reflection_maximum, std::fabs(values[i] + reflected[i]));
                ++result.reflected_features;
                require(std::isfinite(values[i]), "Nonfinite phase-aware feature");
            }
            base.forward(legacy, 1); phase.forward(values, 1);
            result.initial_mean.add(base.output()[0], phase.output()[0]);
            float old_latent[ft::motor_outputs], changed_latent[ft::motor_outputs];
            fast::frozen_gait_latent(old, gait.data(), old_latent);
            fast::frozen_gait_latent(changed, gait.data(), changed_latent);
            for (float noise : {-.3f,0.f,.3f}) {
                float old_applied = old.applied_offset, new_applied = changed.applied_offset;
                float old_targets[ft::motor_outputs], new_targets[ft::motor_outputs];
                ft::decode_yaw_offset(old.navigation, old_latent, base.output()[0] + noise, old_applied, old_targets);
                ft::decode_yaw_offset(changed.navigation, changed_latent, phase.output()[0] + noise, new_applied, new_targets);
                result.initial_targets.add(old_applied, new_applied);
                for (int axis = 0; axis < ft::motor_outputs; ++axis)
                    result.initial_targets.add(old_targets[axis], new_targets[axis]);
            }
            features << source << ',' << sign * magnitude;
            for (int i = 4; i <= 12; ++i) features << ',' << values[i];
            features << '\n';
            ++result.states;
        }
    fast::World sample{};
    fast::initialize(sample, training.front(), ft::pi / 6.f, fast::FeedbackMode::PhaseAware);
    const auto measurement = fast::feedback_sample(sample);
    float original[ft::navigation_inputs], altered[ft::navigation_inputs];
    ft::steering::observe_phase(sample.navigation, 0, measurement, original);
    auto body_phase = sample.navigation;
    body_phase.motion.body.angle[0] += .13f;
    ft::steering::observe_phase(body_phase, 0, measurement, altered);
    result.phase_changes = original[5] != altered[5] || original[6] != altered[6];
    auto slower = measurement;
    slower.velocity_m_s = slower.velocity_m_s * .5f;
    ft::steering::observe_phase(sample.navigation, 0, slower, altered);
    result.speed_changes = std::fabs(altered[8] - original[8] * .5f) < 1e-6f &&
        std::fabs(altered[8] - original[8]) > 1e-4f;
    for (const int magnitude : {5,15,20,30,45,60,90}) for (int sign : {-1,1}) {
        const float angle = sign * magnitude * ft::pi / 180.f;
        sample.navigation.motion.request.heading = p::axis_rotation({0,1,0}, angle).rotate(
            measurement.velocity_m_s / p::norm(measurement.velocity_m_s));
        ft::steering::observe_phase(sample.navigation, 0, measurement, altered);
        result.feature90_maximum = std::fmax(result.feature90_maximum, std::fabs(altered[4] - angle / (ft::pi / 2.f)));
    }
    for (int index = 0; index < ft::steering::spec().num_params(); ++index)
        require(ft::steering::trainable_parameter(index, ft::steering::Kind::PhaseAware13) == (index >= 4 && index <= 12),
            "Phase trainable mask differs from declared nine added coefficients");
    require(result.prefix.differences == 0 && result.initial_mean.differences == 0 && result.initial_targets.differences == 0,
        "Phase initialization changed the frozen baseline commands");
    require(result.reflection_maximum <= 1e-6f && result.feature90_maximum <= 1e-6f &&
        result.phase_changes && result.speed_changes, "Phase/speed/full-angle feature semantics failed");
    const auto path = output / "initial-phase-actor.bin";
    ft::steering::write_model(path.u8string(), initial);
    const auto roundtrip = ft::steering::read_model(path.u8string());
    require(roundtrip.kind == ft::steering::Kind::PhaseAware13 && roundtrip.frozen_base == initial.frozen_base,
        "Phase-aware checkpoint layout or frozen-base identity changed");
    for (size_t i = 0; i < initial.weights.size(); ++i) result.checkpoint.add(initial.weights[i], roundtrip.weights[i]);
    try { (void)ft::read_yaw_offset_weights(path.u8string()); }
    catch (const std::exception&) { result.legacy_rejected = true; }
    require(result.checkpoint.differences == 0 && result.legacy_rejected,
        "Phase-aware checkpoint failed roundtrip or legacy rejection");
    return result;
}

int main(int argc, char** argv) {
    try {
        const auto started = std::chrono::steady_clock::now();
        if (argc < 7 || argc > 11) throw std::runtime_error(
            "Usage: beat_feedback_check FITTED OFFSET TRAIN_STARTS TURN_STARTS STRAIGHT_STARTS OUT [feedback] [study SOURCE_MODEL [baseline|yaw-half-stiffness]]");
        const std::string mode_name = argc >= 8 ? argv[7] : "whole-beat";
        const auto study = argc >= 9 ? ft::steering::read_study(argv[8]) : ft::steering::Study::None;
        require(study == ft::steering::Study::None || argc >= 10, "Program diagnostic requires the declared source model");
        require(mode_name == "whole-beat" || mode_name == "whole-beat-90" || mode_name == "phase-aware", "Unknown feedback-check mode");
        const bool angle90 = mode_name == "whole-beat-90";
        const bool phase_aware = mode_name == "phase-aware";
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
        std::vector<fast::FeedbackMode> bypass_modes = angle90
            ? std::vector<fast::FeedbackMode>{fast::FeedbackMode::HalfSecond, fast::FeedbackMode::WholeBeat, fast::FeedbackMode::WholeBeat90}
            : std::vector<fast::FeedbackMode>{fast::FeedbackMode::HalfSecond, fast::FeedbackMode::WholeBeat};
        if (phase_aware) bypass_modes.push_back(fast::FeedbackMode::PhaseAware);
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
        const auto principal = main_stroke_checks();
        const auto phase_result = phase_aware ? phase_checks(offset, gait, training, output) : PhaseChecks{};
        const auto program_result = study == ft::steering::Study::None ? ProgramChecks{}
            : program_checks(argv[9], study, gait, training, output, argc >= 11 ? argv[10] : "");
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
        emit("phase_initial_prefix", phase_result.prefix); emit("phase_initial_mean", phase_result.initial_mean);
        emit("phase_initial_commands", phase_result.initial_targets); emit("phase_checkpoint_roundtrip", phase_result.checkpoint);
        emit("program_initial_means", program_result.initial_means);
        emit("program_initial_commands", program_result.initial_commands);
        emit("program_inactive_commands", program_result.inactive_commands);
        emit("program_checkpoint_roundtrip", program_result.checkpoint);
        report << "\"phase_aware_enabled\":" << (phase_aware ? "true" : "false")
            << ",\"program_study\":\"" << ft::steering::study_name(study) << '"'
            << ",\"program_request_rows\":" << program_result.request_rows
            << ",\"program_request_feature_error\":" << program_result.request_feature_error
            << ",\"program_passive_inputs_measured\":" << (program_result.passive_inputs_measured ? "true" : "false")
            << ",\"program_request_integration_preserves_body\":" << (program_result.request_integration_preserves_body ? "true" : "false")
            << ",\"phase_request_states\":" << phase_result.states
            << ",\"phase_reflected_features\":" << phase_result.reflected_features
            << ",\"phase_reflection_maximum_difference\":" << phase_result.reflection_maximum
            << ",\"phase_full_angle_maximum_difference\":" << phase_result.feature90_maximum
            << ",\"phase_changes_observation\":" << (phase_result.phase_changes ? "true" : "false")
            << ",\"speed_changes_observation\":" << (phase_result.speed_changes ? "true" : "false")
            << ",\"legacy_loader_rejects_phase_checkpoint\":" << (phase_result.legacy_rejected ? "true" : "false")
            << ",\"angle90_enabled\":" << (angle90 ? "true" : "false")
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
            << ",\"main_stroke_schema\":" << ft::stroke::metric_schema
            << ",\"main_stroke_prominence_rad\":" << ft::stroke::prominence_rad
            << ",\"main_stroke_synthetic_pass\":" << (principal.pass ? "true" : "false")
            << ",\"main_clean_cycles\":" << principal.clean_cycles
            << ",\"main_ripple_cycles\":" << principal.ripple_cycles
            << ",\"main_tiny_cycles\":" << principal.tiny_cycles
            << ",\"main_clean_hz\":" << principal.clean_hz
            << ",\"main_ripple_hz\":" << principal.ripple_hz
            << ",\"stopped_unreliable\":" << (syn.stopped_unreliable ? "true" : "false")
            << ",\"stale_unreliable\":" << (syn.stale_unreliable ? "true" : "false")
            << ",\"reflection_matches\":" << (syn.mirrored_exact ? "true" : "false")
            << ",\"elapsed_s\":" << std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count() << "}\n";
        std::cout << "pass=" << pass << " synthetic_assertions=" << syn.assertions << " requests=" << requests << '\n';
        return pass ? 0 : 2;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
