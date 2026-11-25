#ifndef CUDARL_RL_POND_PPO_CONTRACT_H
#define CUDARL_RL_POND_PPO_CONTRACT_H

#include "core/checkpoint.h"
#include "nn/adam_config.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

// This is a new pond-runtime candidate format. It intentionally does not use
// ppo_checkpoint.h: that historical reader owns a different 40-input topology.
// Parsing these bytes constructs no world, CUDA allocation or optimizer.
namespace cudarl::rl::pond_ppo {
inline constexpr char default_study_protocol_id[]="pond-motor-ppo-v2-2026-11-23";
inline constexpr char curriculum_study_protocol_id[]="pond-motor-ppo-v3-curriculum-2026-11-23";
inline constexpr char repeated_study_protocol_id[]="pond-motor-ppo-v4-repeated-2026-11-24";
inline constexpr char half_second_study_protocol_id[]="pond-motor-ppo-v5-half-second-2026-11-24";
inline constexpr char signed_course_study_protocol_id[]="pond-motor-ppo-v6-signed-course-2026-11-24";
inline constexpr char direct_retrain_protocol_id[]="pond-motor-336-direct-retrain-v1-2026-11-24";
inline constexpr char course_context_protocol_id[]="pond-motor-336-course-context-v1-2026-11-24";
inline constexpr char medium_turn_protocol_id[]="pond-motor-336-medium-turn-v1-2026-11-24";
inline constexpr char fresh_motor_protocol_id[]="pond-motor-fresh-v1-2026-11-24";
inline constexpr char correlated_motor_protocol_id[]="pond-motor-fresh-correlated-v1-2026-11-24";
inline constexpr char propulsion_motor_protocol_id[]="pond-motor-propulsion-v1-2026-11-24";
inline constexpr char course_study_protocol_id[]="pond-motor-ppo-v3-course-2026-11-23";

inline constexpr uint32_t candidate_magic = 0x31505044u;
inline constexpr uint32_t candidate_schema = 1;
inline constexpr uint32_t runtime_version = 4, envelope_version = 2;
inline constexpr uint32_t query_arithmetic_version = 2;
inline constexpr uint32_t observation_schema = 5, action_mapping = 3;
inline constexpr uint32_t parent_package_version = 2;
inline constexpr int observations = 41, hidden = 32, actions = 4;
inline constexpr int actor_weights = 2532, critic_weights = 2433;
inline constexpr uint32_t signed_inputs = observations + 1;
inline constexpr uint32_t actor_parameter_count(uint32_t inputs) { return actor_weights + (inputs - observations) * hidden; }
inline constexpr uint32_t actor_schema(uint32_t inputs) { return inputs == signed_inputs ? 6u : observation_schema; }
static_assert(actor_weights == observations * hidden + hidden + hidden * hidden + hidden + hidden * actions + actions);
static_assert(critic_weights == observations * hidden + hidden + hidden * hidden + hidden + hidden + 1);

inline constexpr uint32_t updates = 32, worlds = 512, controls_per_world = 64;
inline constexpr uint32_t epochs = 4, minibatch = 4096;
inline constexpr uint32_t controls_per_update = worlds * controls_per_world;
inline constexpr uint32_t minibatches_per_epoch = controls_per_update / minibatch;
inline constexpr uint32_t adam_calls_per_update = epochs * minibatches_per_epoch;
inline constexpr uint64_t total_controls = uint64_t{updates} * controls_per_update;
inline constexpr uint64_t total_adam_calls = uint64_t{updates} * adam_calls_per_update * 2;
inline constexpr uint64_t substeps_per_control = 10;
static_assert(controls_per_update % minibatch == 0 && adam_calls_per_update == 32 && total_adam_calls == 2048);

inline constexpr float learning_rate = 1e-5f, clip = .1f, gamma = .995f, lambda = .95f;
inline constexpr float direct_retrain_actor_learning_rate = 3e-5f;
inline constexpr float fresh_actor_learning_rate = .00025f, fresh_critic_learning_rate = .001f;
inline constexpr float fresh_sigma = .30f, fresh_holding_gain = 1.f;
inline constexpr float correlated_rho = .8f;
inline constexpr float sigma = .10f, gradient_clip = 1.f;
inline constexpr float frozen_holding_gain = .26211655139923096f;
inline constexpr double maximum_wall_seconds = 1200;
inline constexpr uint64_t maximum_evidence_bytes = 4ull * 1024 * 1024 * 1024;
inline constexpr uint64_t maximum_state_bytes = 16ull * 1024 * 1024;
inline constexpr uint64_t maximum_adam_bytes = 256ull * 1024;
inline constexpr std::string_view canonical_parent_sha256 =
    "810dbd593ce39475cdd6e95b709bef3fdb34d50c9fcbbb2c345c7a8882a0bef0";
inline constexpr std::string_view parent_package_sha256 =
    "4b6400c4db65873bd346c3e0c4fed384e3a77f9e5174d0dcea5d74073b94616d";

inline float fixed_log_std() { return std::log(sigma); }
inline float fresh_sampling_log_std() { return std::log(fresh_sigma); }

struct Counters {
    uint64_t completed_updates = 0;
    uint64_t attempted_controls = 0, attempted_substeps = 0;
    uint64_t actor_adam_calls = 0, critic_adam_calls = 0;
};

struct CandidateState {
    uint64_t parent_package_checksum = 0, protocol_checksum = 0;
    uint64_t source_identity_checksum = 0, sampler_decision_serial = 0;
    double wall_seconds_used = 0;
    uint64_t evidence_bytes_used = 0;
    uint32_t actor_inputs = observations, seed = 0, world_stride = 0, task_stride = 0;
    Counters counters{};
    std::vector<float> actor, critic, log_std;
    float frozen_gain = frozen_holding_gain;
    std::vector<uint8_t> actor_adam, critic_adam, world_state, task_state;
};

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// First-layer weights are output-major. Insert one zero in every row while
// preserving the original weights, biases and all subsequent layers.
inline std::vector<float> initial_actor(const std::vector<float>& parent, uint32_t inputs) {
    require(parent.size() == actor_weights, "Invalid parent actor shape");
    if (inputs == observations) return parent;
    require(inputs == signed_inputs, "Unknown candidate actor shape");
    std::vector<float> expanded(actor_parameter_count(inputs), 0.f);
    for (int row = 0; row < hidden; ++row)
        std::copy_n(parent.begin() + row * observations, observations, expanded.begin() + row * inputs);
    std::copy(parent.begin() + hidden * observations, parent.end(), expanded.begin() + hidden * inputs);
    return expanded;
}

inline void validate_counters(const Counters& counters) {
    require(counters.completed_updates <= updates, "Pond PPO update budget exceeded");
    const uint64_t completed_controls = counters.completed_updates * controls_per_update;
    const uint64_t expected_adam = counters.completed_updates * adam_calls_per_update;
    require(counters.attempted_controls == completed_controls && counters.attempted_controls <= total_controls,
            "Invalid pond PPO attempted-control count");
    require(counters.attempted_substeps == completed_controls * substeps_per_control,
            "Invalid pond PPO attempted-substep count");
    require(counters.actor_adam_calls == expected_adam && counters.critic_adam_calls == expected_adam,
            "Pond PPO optimizer/update counters disagree");
}

// GpuAdam::save's existing byte layout is checked without GpuAdam construction
// or restore. Fresh candidate moments have their own age, never the parent's.
inline void validate_adam(const std::vector<uint8_t>& bytes, size_t weights, uint64_t expected_steps,
                          float expected_learning_rate = learning_rate) {
    static_assert(sizeof(nn::AdamConfig) == 20 && sizeof(int) == 4, "Pond Adam ABI changed");
    require(bytes.size() <= maximum_adam_bytes, "Pond Adam state exceeds size limit");
    checkpoint::Reader in{bytes};
    const auto config = in.value<nn::AdamConfig>();
    const int steps = in.value<int>();
    require(config.lr == expected_learning_rate && config.beta1 == .9f && config.beta2 == .999f &&
                config.eps == 1e-8f && config.max_grad_norm == gradient_clip,
            "Wrong pond PPO Adam configuration");
    require(steps >= 0 && static_cast<uint64_t>(steps) == expected_steps, "Pond Adam age disagrees with counters");
    const auto first = in.array<float>(weights), second = in.array<float>(weights);
    require(first.size() == weights && second.size() == weights, "Wrong pond Adam topology");
    for (size_t i = 0; i < weights; ++i) {
        require(std::isfinite(first[i]) && std::isfinite(second[i]) && second[i] >= 0,
                "Invalid pond Adam moments");
        require(expected_steps != 0 || (first[i] == 0 && second[i] == 0), "Fresh pond Adam moments must be zero");
    }
    in.finish();
}

inline void validate_state(const CandidateState& state, float actor_learning_rate = learning_rate, bool fresh_motor = false) {
    require(state.parent_package_checksum != 0 && state.protocol_checksum != 0 && state.source_identity_checksum != 0,
            "Missing pond PPO source identity");
    require(std::isfinite(state.wall_seconds_used) && state.wall_seconds_used >= 0 &&
                state.wall_seconds_used <= maximum_wall_seconds,
            "Invalid pond PPO cumulative wall-time accounting");
    require(state.evidence_bytes_used <= maximum_evidence_bytes, "Pond PPO cumulative evidence-byte budget exceeded");
    validate_counters(state.counters);
    require(state.sampler_decision_serial==state.counters.completed_updates*controls_per_world,
            "Saved sampler serial does not match completed collection");
    require((state.actor_inputs == observations || state.actor_inputs == signed_inputs) &&
            state.actor.size() == actor_parameter_count(state.actor_inputs) && state.critic.size() == critic_weights && state.log_std.size() == actions,
            "Wrong pond PPO network topology");
    for (const auto* values : {&state.actor, &state.critic})
        for (float value : *values) require(std::isfinite(value), "Nonfinite pond PPO network parameter");
    const float expected_log_std = fresh_motor ? fresh_sampling_log_std() : fixed_log_std();
    for (float value : state.log_std) require(value == expected_log_std, "Pond PPO sampling scale changed");
    require(state.frozen_gain == (fresh_motor ? fresh_holding_gain : frozen_holding_gain), "Holding amplitude differs from the study");
    require(!fresh_motor || actor_learning_rate == fresh_actor_learning_rate, "Fresh motor requires its sealed actor rate");
    require(state.world_stride != 0 && state.task_stride != 0 &&
                uint64_t{state.world_stride} * worlds <= maximum_state_bytes &&
                uint64_t{state.task_stride} * worlds <= maximum_state_bytes,
            "Invalid pond PPO carried-state layout");
    require(state.world_state.size() == uint64_t{state.world_stride} * worlds &&
                state.task_state.size() == uint64_t{state.task_stride} * worlds,
            "Wrong pond PPO carried-state array size");
    validate_adam(state.actor_adam, actor_parameter_count(state.actor_inputs), state.counters.actor_adam_calls, actor_learning_rate);
    validate_adam(state.critic_adam, critic_weights, state.counters.critic_adam_calls,
                  fresh_motor ? fresh_critic_learning_rate : learning_rate);
}

inline std::vector<uint8_t> identity_bytes(std::string_view value) {
    return {value.begin(), value.end()};
}

inline void seal(checkpoint::Writer& out) { out.value(checkpoint::checksum(out.data)); }

inline std::vector<uint8_t> checked_body(const std::vector<uint8_t>& bytes, uint64_t maximum) {
    require(bytes.size() >= sizeof(uint64_t) && bytes.size() <= maximum, "Invalid pond PPO envelope length");
    std::vector<uint8_t> body(bytes.begin(), bytes.end() - static_cast<std::ptrdiff_t>(sizeof(uint64_t)));
    checkpoint::Reader trailer{bytes, body.size()};
    require(trailer.value<uint64_t>() == checkpoint::checksum(body), "Pond PPO payload checksum failed");
    trailer.finish();
    return body;
}

inline std::vector<uint8_t> write_candidate(const CandidateState& state, float actor_learning_rate = learning_rate, bool fresh_motor = false) {
    validate_state(state, actor_learning_rate, fresh_motor);
    checkpoint::Writer out;
    for (uint32_t value : {candidate_magic, candidate_schema, runtime_version, envelope_version,
                          actor_schema(state.actor_inputs), action_mapping, parent_package_version,
                          state.actor_inputs, uint32_t{actions}, actor_parameter_count(state.actor_inputs), uint32_t{critic_weights}})
        out.value(value);
    out.array(identity_bytes(canonical_parent_sha256));
    out.array(identity_bytes(parent_package_sha256));
    out.value(state.parent_package_checksum); out.value(state.protocol_checksum);
    out.value(state.source_identity_checksum); out.value(state.sampler_decision_serial);
    out.value(state.wall_seconds_used); out.value(state.evidence_bytes_used);
    out.value(state.seed); out.value(state.world_stride); out.value(state.task_stride);
    const auto& count = state.counters;
    for (uint64_t value : {count.completed_updates, count.attempted_controls, count.attempted_substeps,
                           count.actor_adam_calls, count.critic_adam_calls}) out.value(value);
    out.array(state.actor); out.array(state.critic); out.array(state.log_std); out.value(state.frozen_gain);
    out.array(state.actor_adam); out.array(state.critic_adam);
    out.array(state.world_state); out.array(state.task_state);
    seal(out);
    return out.data;
}

inline CandidateState read_candidate(const std::vector<uint8_t>& bytes, uint64_t expected_parent,
                                     uint64_t expected_protocol, uint64_t expected_source,
                                     uint32_t expected_world_stride, uint32_t expected_task_stride,
                                     uint32_t expected_inputs = observations,
                                     float expected_actor_learning_rate = learning_rate,
                                     bool fresh_motor = false) {
    require(expected_inputs == observations || expected_inputs == signed_inputs, "Unknown expected candidate actor shape");
    const auto body = checked_body(bytes, 2 * maximum_state_bytes + 2 * maximum_adam_bytes + 65536);
    checkpoint::Reader in{body};
    for (uint32_t expected : {candidate_magic, candidate_schema, runtime_version, envelope_version,
                             actor_schema(expected_inputs), action_mapping, parent_package_version,
                             expected_inputs, uint32_t{actions}, actor_parameter_count(expected_inputs), uint32_t{critic_weights}})
        require(in.value<uint32_t>() == expected, "Wrong pond PPO schema/runtime/topology");
    require(in.array<uint8_t>(64) == identity_bytes(canonical_parent_sha256) &&
                in.array<uint8_t>(64) == identity_bytes(parent_package_sha256), "Wrong immutable pond PPO parent identity");
    CandidateState state;
    state.actor_inputs = expected_inputs;
    state.parent_package_checksum = in.value<uint64_t>(); state.protocol_checksum = in.value<uint64_t>();
    state.source_identity_checksum = in.value<uint64_t>(); state.sampler_decision_serial = in.value<uint64_t>();
    state.wall_seconds_used = in.value<double>(); state.evidence_bytes_used = in.value<uint64_t>();
    state.seed = in.value<uint32_t>(); state.world_stride = in.value<uint32_t>(); state.task_stride = in.value<uint32_t>();
    require(state.parent_package_checksum == expected_parent && state.protocol_checksum == expected_protocol &&
                state.source_identity_checksum == expected_source, "Pond PPO checkpoint source/protocol mismatch");
    require(state.world_stride == expected_world_stride && state.task_stride == expected_task_stride,
            "Pond PPO checkpoint native layout mismatch");
    auto& count = state.counters;
    count.completed_updates = in.value<uint64_t>(); count.attempted_controls = in.value<uint64_t>();
    count.attempted_substeps = in.value<uint64_t>(); count.actor_adam_calls = in.value<uint64_t>();
    count.critic_adam_calls = in.value<uint64_t>();
    state.actor = in.array<float>(actor_parameter_count(expected_inputs)); state.critic = in.array<float>(critic_weights);
    state.log_std = in.array<float>(actions); state.frozen_gain = in.value<float>();
    state.actor_adam = in.array<uint8_t>(static_cast<size_t>(maximum_adam_bytes));
    state.critic_adam = in.array<uint8_t>(static_cast<size_t>(maximum_adam_bytes));
    state.world_state = in.array<uint8_t>(static_cast<size_t>(maximum_state_bytes));
    state.task_state = in.array<uint8_t>(static_cast<size_t>(maximum_state_bytes));
    in.finish();
    validate_state(state, expected_actor_learning_rate, fresh_motor);
    return state;
}

inline void save_candidate(const std::string& path, const CandidateState& state) {
    checkpoint::write_atomic(path, write_candidate(state));
}

inline CandidateState load_candidate(const std::string& path, uint64_t expected_parent, uint64_t expected_protocol,
                                     uint64_t expected_source, uint32_t expected_world_stride, uint32_t expected_task_stride) {
    return read_candidate(checkpoint::read(path), expected_parent, expected_protocol, expected_source,
                          expected_world_stride, expected_task_stride);
}

inline constexpr uint32_t retention_magic = 0x31525044u, retention_schema = 3;
struct RetentionReceipt {
    uint64_t checkpoint_checksum = 0, parent_package_checksum = 0, protocol_checksum = 0, eval_source_identity = 0;
    uint32_t cases_passed = 0, total = 0, completed_cases = 0, physical_failures = 0;
    uint64_t controls = 0, substeps = 0;
    bool passed = false, baseline_only = false;
};

inline void validate_retention_receipt(const RetentionReceipt& receipt) {
    require(receipt.checkpoint_checksum != 0 && receipt.parent_package_checksum != 0 &&
                receipt.protocol_checksum != 0 && receipt.eval_source_identity != 0,
            "Missing pond retention receipt identity");
    require(receipt.total==20&&receipt.completed_cases==20&&receipt.cases_passed<=20&&receipt.physical_failures==0,
            "Pond assessment must complete20 cases without physical faults");
    // v2 baseline_only identifies candidate0; it does not exempt it from the
    // exact inherited-skill gate required before the first optimizer call.
    require(receipt.passed&&receipt.cases_passed==20,"Pond retention receipt does not report a full pass");
    require(receipt.controls==30400&&receipt.substeps==304000,"Invalid pond retention receipt counters");
}

inline std::vector<uint8_t> write_retention_receipt(const RetentionReceipt& receipt) {
    validate_retention_receipt(receipt);
    checkpoint::Writer out;
    for (uint32_t value : {retention_magic, retention_schema, runtime_version, envelope_version}) out.value(value);
    for (uint64_t value : {receipt.checkpoint_checksum, receipt.parent_package_checksum, receipt.protocol_checksum,
                           receipt.eval_source_identity}) out.value(value);
    out.value(receipt.cases_passed); out.value(receipt.total);out.value(receipt.completed_cases);out.value(receipt.physical_failures);
    out.value(receipt.controls); out.value(receipt.substeps);out.value(static_cast<uint32_t>(receipt.passed));out.value(static_cast<uint32_t>(receipt.baseline_only));
    seal(out);
    return out.data;
}

inline RetentionReceipt read_retention_receipt(const std::vector<uint8_t>& bytes, uint64_t expected_checkpoint,
                                               uint64_t expected_parent, uint64_t expected_protocol,
                                               uint64_t expected_eval_source,bool allow_baseline=false) {
    const auto body = checked_body(bytes, 128);
    checkpoint::Reader in{body};
    for (uint32_t expected : {retention_magic, retention_schema, runtime_version, envelope_version})
        require(in.value<uint32_t>() == expected, "Wrong pond retention receipt schema/runtime");
    RetentionReceipt receipt;
    receipt.checkpoint_checksum = in.value<uint64_t>(); receipt.parent_package_checksum = in.value<uint64_t>();
    receipt.protocol_checksum = in.value<uint64_t>(); receipt.eval_source_identity = in.value<uint64_t>();
    receipt.cases_passed = in.value<uint32_t>(); receipt.total = in.value<uint32_t>();receipt.completed_cases=in.value<uint32_t>();receipt.physical_failures=in.value<uint32_t>();
    receipt.controls = in.value<uint64_t>(); receipt.substeps = in.value<uint64_t>();
    const auto passed=in.value<uint32_t>(),baseline=in.value<uint32_t>();
    require(passed<=1&&baseline<=1,"Invalid assessment flags");receipt.passed=passed!=0;receipt.baseline_only=baseline!=0;
    require(receipt.baseline_only==allow_baseline,"Baseline assessment cannot release a trained checkpoint");
    in.finish();
    validate_retention_receipt(receipt);
    require(receipt.checkpoint_checksum == expected_checkpoint && receipt.parent_package_checksum == expected_parent &&
                receipt.protocol_checksum == expected_protocol && receipt.eval_source_identity == expected_eval_source,
            "Pond retention receipt checkpoint/source mismatch");
    return receipt;
}

} // namespace cudarl::rl::pond_ppo
#endif
