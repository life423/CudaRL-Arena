#pragma once

#include "rl/flexible_tail_fast_steering.h"
#include "rl/flexible_tail_chooser.h"
#include "rl/flexible_tail_bank_observation.h"
#include "rl/flexible_tail_distant_coverage.h"
#include "rl/flexible_tail_curved_bank.h"
#include "core/checkpoint.h"
#include <climits>

namespace cudarl::flexible_tail::rate_chooser {

inline constexpr int local_inputs=32, decision_controls = 40, cue_limit_controls = 40;
#if defined(CUDARL_CURVED_BANK)
inline constexpr int inputs=39, default_observation_mode=3;
#elif defined(CUDARL_APPENDED_COVERAGE)
inline constexpr int inputs=35, default_observation_mode=2;
#else
inline constexpr int inputs=32;
#if defined(CUDARL_DISTANT_COVERAGE)
inline constexpr int default_observation_mode=1;
#else
inline constexpr int default_observation_mode=0;
#endif
#endif
#if defined(CUDARL_DISTANT_COVERAGE)
inline constexpr bool default_distant = true;
#else
inline constexpr bool default_distant = false;
#endif
#if defined(CUDARL_RELIABLE_CHOOSER)
inline constexpr int rate_count = 5, actions = 15;
inline constexpr uint32_t schema = 2, model_version = 3;
#else
inline constexpr int rate_count = 3, actions = 9;
inline constexpr uint32_t schema = 1, model_version = 2;
#endif
CUDARL_HD inline float action_rate_deg_s(int action) {
    return (action % rate_count - rate_count / 2) * (8.f / (rate_count / 2));
}
inline const char* mode_name() { return rate_count == 5 ? "learned-fifteen-action-speed-course-rate" : "learned-nine-action-speed-course-rate"; }
enum Fault { Healthy = 0, NativePhysics = 1, Clearance = 2, Nonfinite = 3,
    CueUnavailable = 4, PoseEnvelope = 5, InvalidUse = 6 };

// One native body owner. The visitation memory belongs to this episode and is
// saved with it; none of the sensory features chooses or overrides an action.
struct World {
    fast::World motor{};
    steering::Kind motor_kind = steering::Kind::ProgramNonlinear;
    steering::Config config{};
    Snapshot pose{};
    uint64_t visited[chooser::cells_x * chooser::cells_z / 64]{};
    chooser::RecentVisit recent[chooser::recent_cells]{};
    physics::V entry_position{};
    int entry_cell = -1, entry_control = 0, recent_index = 0;
    bool entry_credited = false;
    int decision_count = 0, decision_control = 0, last_action = 4, turns = 0, visits = 0;
    int fault = Healthy;
    uint64_t attempted_substeps = 0, accepted_substeps = 0;
    float gap_m = 0, min_gap_m = FLT_MAX, min_speed_m_s = FLT_MAX;
    float peak_tilt_deg = 0, peak_depth_error_m = 0, stall_s = 0, max_stall_s = 0, transport_m = 0;
};
struct Result { float reward = 0; int controls = 0, fault = Healthy;
    int attempted_substeps = 0, accepted_substeps = 0; };

CUDARL_HD inline bool ready(const World& w) {
    return w.fault == Healthy && w.motor.navigation.motion.controls - w.decision_control >= decision_controls;
}

CUDARL_HD inline void initialize(World& w, const NavigationWorld& source,
    physics::V horizontal_position, float yaw, steering::Kind kind, steering::Config config) {
    w = {};
    if (!steering::valid_config(config) || kind < steering::Kind::ProgramScalar) {
        w.fault = InvalidUse; return;
    }
    // Reuse the existing, measured fixture placement transform. This sets the
    // initial condition only; no deployed continuation resets or teleports.
    chooser::World placed{};
    chooser::initialize(placed, source, horizontal_position, yaw);
    if (placed.fault) { w.fault = placed.fault; return; }
    w.motor.navigation = placed.navigation;
    w.motor.pose = placed.pose;
    w.motor.horizon_controls = INT_MAX / 32;
    w.motor.feedback_mode = fast::FeedbackMode::PhaseAware;
    w.motor.beat_feedback.initialize(w.motor.navigation.motion);
    for (int i = 0; i < motor_outputs; ++i) w.motor.targets[i] = placed.targets[i];
    w.motor_kind = kind; w.config = config; w.config.study = steering::Study::Rate;
    w.pose = placed.pose;
    w.entry_position = w.pose.com_m; w.entry_cell = chooser::cell_index(w.pose.com_m);
    // A persistent course request keeps learned steering active when the
    // chooser subsequently requests zero rate. It does not rotate the body.
    auto& nav = w.motor.navigation;
    nav.legs[0].choice = Choice::TurnLeft;
    nav.legs[0].end_control = INT_MAX / 32;
    nav.leg_count = 1; nav.active_leg = 0;
    prepare_navigation(nav);
}

CUDARL_HD inline bool accept(World& w, int action) {
    if (!ready(w) || action < 0 || action >= actions) return false;
    constexpr float speeds[3]{.006f, .008f, .012f};
    const float rate = action_rate_deg_s(action);
    w.config.desired_course_rate_rad_s = rate * pi / 180.f;
    w.config.target_speed_m_s = speeds[action / rate_count];
    w.decision_control = w.motor.navigation.motion.controls;
    w.last_action = action; ++w.decision_count;
    if (rate) ++w.turns;
    return true;
}

CUDARL_HD inline void observe(const World& w, const pond_geometry_v1::Parameters& pond,
    float (&out)[inputs], int observation_mode=default_observation_mode) {
    for (auto& v : out) v = 0;
    const auto sample = fast::feedback_sample(w.motor);
    float bank_features[bank::inputs];
    bank::observe(w.motor.navigation.motion.body, w.pose, pond, sample.velocity_m_s,
        sample.reliable, bank_features);
    for (int i = 0; i < bank::inputs; ++i) out[i] = bank_features[i];
    const auto& nav = w.motor.navigation;
    const auto& body = nav.motion.body;
    auto course = sample.reliable ? sample.velocity_m_s : body.orientation.rotate({1,0,0});
    course.y = 0; const float length = physics::norm(course);
    if (length > .0001f) course = course / length;
    for (int side = 0; side < 3; ++side) {
        const auto direction = physics::axis_rotation({0,1,0}, (side - 1) * pi / 4.f).rotate(course);
        out[18 + side] = chooser::novelty_at(w, w.pose.com_m + direction * .08f, nav.motion.controls);
        out[21 + side] = chooser::novelty_at(w, w.pose.com_m + direction * .24f, nav.motion.controls);
    }
    if(observation_mode==1||((observation_mode==2||observation_mode==3)&&inputs>=35)){
        float field[3];distant::observe(w,w.pose.com_m,course,nav.motion.request.origin_m.y,nav.motion.controls,field);
        if(observation_mode==1){for(int side=0;side<3;++side)out[21+side]=field[side];}
        if constexpr(inputs>=35){if(observation_mode>=2){for(int side=0;side<3;++side)out[32+side]=field[side];}}
    }
    out[24] = w.pose.com_m.x / float(pond.water_radius_x_m);
    out[25] = w.pose.com_m.z / float(pond.water_radius_z_m);
    out[26] = w.config.target_speed_m_s / .008f - 1.f;
    out[27] = w.config.desired_course_rate_rad_s / (8.f * pi / 180.f);
    out[28] = sample.reliable ? yaw_offset_error(sample.velocity_m_s, nav.motion.request.heading) / pi : 0.f;
    out[29] = chooser::novelty_at(w, w.pose.com_m, nav.motion.controls);
    out[30] = body.angle[2] / .2f;
    out[31] = physics::clamp(w.stall_s / 2.f, 0.f, 1.f);
    if constexpr(inputs==39){if(observation_mode==3){const auto curve=curved_bank::observe(w.motor,pond);
        if(curve.valid){out[35]=curve.loss_s/curved_bank::horizon_s;out[36]=physics::clamp(curve.min_gap_m/.05f,-1.f,1.f);
            out[37]=curve.rate_rad_s/(8*pi/180.f);out[38]=1;}}}
}

// packed_motor contains the model's parameters, followed by its frozen skip.
// Both are immutable throughout chooser PPO and saved in its input bindings.
CUDARL_HD inline Result advance(World& w, const float* gait, const float* packed_motor,
    const pond_geometry_v1::Parameters& pond, int max_controls = decision_controls) {
    Result out{};
    while (out.controls < max_controls && w.fault == Healthy && !ready(w)) {
        auto& motor = w.motor;
        const auto before = motor.pose.com_m;
        float observation[navigation_inputs], mean[3];
        fast::observe_program(motor, w.config, observation);
        const int count = w.motor_kind == steering::Kind::ProgramNonlinear ? 563 :
            w.motor_kind == steering::Kind::ProgramLinear ? 96 : 32;
        steering::infer(w.motor_kind, packed_motor,
            w.motor_kind == steering::Kind::ProgramNonlinear ? packed_motor + count : nullptr,
            observation, mean);
        const auto step = fast::advance_program(motor, mean, gait, pond, w.config,
            (max_controls - out.controls < 4 ? max_controls - out.controls : 4));
        out.attempted_substeps += step.attempted_substeps;
        out.accepted_substeps += step.accepted_substeps;
        out.controls += step.controls;
        w.pose = motor.pose;
        w.gap_m = motor.min_gap_m; w.min_gap_m = fminf(w.min_gap_m, motor.min_gap_m);
        w.peak_tilt_deg = fmaxf(w.peak_tilt_deg, motor.peak_tilt_deg);
        w.peak_depth_error_m = fmaxf(w.peak_depth_error_m, motor.peak_depth_error_m);
        if (step.fault) { w.fault = step.fault; break; }
        const auto sample = fast::feedback_sample(motor);
        const float speed = physics::norm(sample.velocity_m_s);
        w.min_speed_m_s = fminf(w.min_speed_m_s, speed);
        w.transport_m += physics::norm(chooser::horizontal(w.pose.com_m - before));
        const float seconds = step.controls * motor_control_s;
        if (speed >= .006f) {
            w.stall_s = 0;
#if defined(CUDARL_COVERAGE_CHOOSER)
            // Transport into new water keeps full value. Safe circles through
            // recently credited cells retain only a small maintenance reward.
            // This scores actual movement; it neither selects nor alters actions.
            const float novelty = chooser::novelty_at(w, w.pose.com_m, motor.navigation.motion.controls);
            out.reward += seconds * .04f * (.05f + .95f * novelty);
#else
            out.reward += seconds * .04f;
#endif
        }
        else { w.stall_s += seconds; out.reward -= seconds * .10f; }
        w.max_stall_s = fmaxf(w.max_stall_s, w.stall_s);
        out.reward += chooser::visit_reward_at(w, w.pose.com_m, motor.navigation.motion.controls, speed);
#if !defined(CUDARL_CURVED_BANK)
        float senses[bank::inputs];
        bank::observe(motor.navigation.motion.body,w.pose,pond,sample.velocity_m_s,sample.reliable,senses);
        // Penalize an approaching loss of clearance, not proximity alone.
        if (senses[bank::TransportTimeValid] > 0.f) {
            const float remaining_s = senses[bank::ConstantTransportTime] * bank::time_scale_s;
            out.reward -= seconds * .10f * physics::clamp((12.f - remaining_s) / 12.f, 0.f, 1.f);
        }
#endif
        if (step.controls == 0) { w.fault = InvalidUse; break; }
    }
    w.attempted_substeps += out.attempted_substeps;
    w.accepted_substeps += out.accepted_substeps;
    if (w.fault == Clearance || w.fault == PoseEnvelope || w.fault == CueUnavailable) out.reward -= 5.f;
    out.fault = w.fault;
    return out;
}

inline nn::MlpSpec actor_spec() { return {{inputs,32,actions},{nn::Activation::Tanh,nn::Activation::Identity}}; }
// First-layer matrices are output-major. Biases and all later layers keep
// their exact bits; newly appended columns (including Adam moments) are zero.
inline std::vector<float> resize_input_parameters(const std::vector<float>& p,int from,int to){
    if(from==to)return p;
    if((from!=32&&from!=35&&from!=39)||(to!=32&&to!=35&&to!=39)||p.size()<size_t(from*32))
        throw std::runtime_error("Unsupported chooser input migration");
    std::vector<float> result(size_t(int64_t(p.size())+32*(to-from)),0.f);
    for(int row=0;row<32;++row){
        for(int column=0;column<std::min(from,to);++column)result[row*to+column]=p[row*from+column];
        for(int column=to;column<from;++column)if(p[row*from+column]!=0.f)
            throw std::runtime_error("Cannot discard a learned appended column");
    }
    std::copy(p.begin()+32*from,p.end(),result.begin()+32*to);
    return result;
}
inline void validate_weights(const std::vector<float>& weights) {
    if (weights.size() != size_t(actor_spec().num_params())) throw std::runtime_error("Invalid rate chooser model");
    for (float x : weights) if (!physics::finite_number(x)) throw std::runtime_error("Nonfinite rate chooser model");
}
inline void write_weights(const std::string& path, const std::vector<float>& weights) {
    validate_weights(weights); checkpoint::Writer out;
    out.value<uint32_t>(0x48435446); out.value(inputs==39?uint32_t(6):(inputs==35?uint32_t(5):(default_distant?uint32_t(4):model_version))); out.value(inputs==39?uint32_t(5):(inputs==35?uint32_t(4):(default_distant?uint32_t(3):schema)));
    out.value(inputs); out.value(actions); out.value(decision_controls);
    if constexpr(default_distant||inputs>=35)out.value(distant_map::identity);
    if constexpr(inputs==39)out.value(curved_bank::identity);
    out.array(weights);
    checkpoint::write_atomic(path,out.data);
}
inline std::vector<float> read_weights(const std::string& path, int* observation_mode=nullptr) {
    auto bytes=checkpoint::read(path); checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x48435446)throw std::runtime_error("Rate chooser magic mismatch");
    const auto version=in.value<uint32_t>(),request_schema=in.value<uint32_t>();
    const bool is_distant=version==4&&request_schema==3&&actions==15;
    const bool is_appended=version==5&&request_schema==4&&actions==15;
    const bool is_curved=version==6&&request_schema==5&&actions==15;
    const int saved_inputs=in.value<int>();
    if((!is_distant&&!is_appended&&!is_curved&&(version!=model_version||request_schema!=schema)) ||
        saved_inputs!=(is_curved?39:(is_appended?35:32)) || saved_inputs>inputs || in.value<int>()!=actions || in.value<int>()!=decision_controls)
        throw std::runtime_error("Rate chooser schema mismatch");
    if((is_distant||is_appended||is_curved)&&in.value<uint64_t>()!=distant_map::identity)throw std::runtime_error("Distant coverage map binding mismatch");
    if(is_curved&&in.value<uint64_t>()!=curved_bank::identity)throw std::runtime_error("Curved forecast binding mismatch");
    if((is_distant||is_appended||is_curved)&&!observation_mode&&!default_distant)throw std::runtime_error("Chooser needs an observation-mode consumer");
    if(observation_mode)*observation_mode=is_curved?3:(is_appended?2:(is_distant?1:0));
    auto weights=in.array<float>(actor_spec().num_params()); in.finish();
    const nn::MlpSpec saved_spec{{saved_inputs,32,actions},{nn::Activation::Tanh,nn::Activation::Identity}};
    if(weights.size()!=size_t(saved_spec.num_params()))throw std::runtime_error("Saved chooser parameter count differs");
    weights=resize_input_parameters(weights,saved_inputs,inputs);validate_weights(weights);return weights;
}
} // namespace cudarl::flexible_tail::rate_chooser
