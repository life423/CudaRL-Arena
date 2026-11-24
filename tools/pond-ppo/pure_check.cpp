#include "rl/pond_ppo_contract.h"
#include "rl/gaussian_math.h"
#include "admission.h"

#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <system_error>

namespace {
namespace contract = cudarl::rl::pond_ppo;
namespace cp = cudarl::checkpoint;
namespace fs = std::filesystem;

int checks = 0;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
    ++checks;
}
void rejects(const std::function<void()>& action, const char* message) {
    bool rejected = false;
    try { action(); } catch (const std::runtime_error&) { rejected = true; }
    check(rejected, message);
}
bool near(double left, double right, double tolerance = 1e-12) { return std::abs(left - right) <= tolerance; }

std::vector<uint8_t> adam_bytes(size_t count, uint64_t steps) {
    cp::Writer out;
    cudarl::nn::AdamConfig config;
    config.lr = contract::learning_rate; config.max_grad_norm = contract::gradient_clip;
    out.value(config); out.value(static_cast<int>(steps));
    out.array(std::vector<float>(count, 0.f)); out.array(std::vector<float>(count, 0.f));
    return out.data;
}

contract::CandidateState candidate(uint64_t update = 8) {
    contract::CandidateState state;
    state.parent_package_checksum = 0x1020304050607080ull;
    state.protocol_checksum = 0x0123456789abcdefull;
    state.source_identity_checksum = 0xfedcba9876543210ull;
    state.seed = 4813;
    state.world_stride = 1; state.task_stride = 1;
    state.counters.completed_updates = update;
    state.counters.attempted_controls = update * contract::controls_per_update;
    state.counters.attempted_substeps = state.counters.attempted_controls * contract::substeps_per_control;
    state.counters.actor_adam_calls = update * contract::adam_calls_per_update;
    state.counters.critic_adam_calls = state.counters.actor_adam_calls;
    state.sampler_decision_serial = update * contract::controls_per_world;
    state.wall_seconds_used = 12.5; state.evidence_bytes_used = 8192;
    state.actor.assign(contract::actor_weights, .05f); state.critic.assign(contract::critic_weights, -.025f);
    state.log_std.assign(contract::actions, contract::fixed_log_std());
    state.actor_adam = adam_bytes(contract::actor_weights, state.counters.actor_adam_calls);
    state.critic_adam = adam_bytes(contract::critic_weights, state.counters.critic_adam_calls);
    // Opaque dummy bytes exercise bounded carried-state persistence only. They
    // are not WorldState/Task objects and never reach a simulator or optimizer.
    state.world_state.assign(contract::worlds, uint8_t{0xa5});
    state.task_state.assign(contract::worlds, uint8_t{0x5a});
    return state;
}

contract::CandidateState decode(const std::vector<uint8_t>& bytes, const contract::CandidateState& expected) {
    return contract::read_candidate(bytes, expected.parent_package_checksum, expected.protocol_checksum,
                                    expected.source_identity_checksum, expected.world_stride, expected.task_stride);
}

template<class T> void patch(std::vector<uint8_t>& bytes, size_t offset, T value) {
    check(offset <= bytes.size() && sizeof(T) <= bytes.size() - offset, "test patch out of range");
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

void reseal(std::vector<uint8_t>& bytes) {
    std::vector<uint8_t> body(bytes.begin(), bytes.end() - static_cast<std::ptrdiff_t>(sizeof(uint64_t)));
    patch(bytes, body.size(), cp::checksum(body));
}

struct TemporaryFile {
    fs::path directory, file;
    TemporaryFile() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto base = fs::temp_directory_path();
        for (int attempt = 0; attempt < 64; ++attempt) {
            const auto path = base / ("cudarl-pond-ppo-pure-" + std::to_string(stamp) + "-" + std::to_string(attempt));
            if (fs::create_directory(path)) { directory = path; file = directory / "candidate.bin"; return; }
        }
        throw std::runtime_error("Could not create unique pure-check temporary directory");
    }
    ~TemporaryFile() {
        // Remove only these explicitly created files and the empty directory;
        // never recursively remove a computed filesystem path.
        std::error_code ignored;
        if (!file.empty()) {
            fs::remove(file, ignored); fs::remove(fs::path(file.string() + ".tmp"), ignored);
            fs::remove(fs::path(file.string() + ".lock"), ignored);
        }
        if (!directory.empty()) fs::remove(directory, ignored);
    }
};

void checkpoint_checks() {
    const auto state = candidate();
    const auto bytes = contract::write_candidate(state);
    const auto restored = decode(bytes, state);
    check(contract::write_candidate(restored) == bytes, "candidate roundtrip changed bytes");
    check(restored.actor.size() == 2532 && restored.critic.size() == 2433, "candidate topology lost");
    check(restored.world_state == state.world_state && restored.task_state == state.task_state &&
              restored.sampler_decision_serial == state.sampler_decision_serial, "carried replay state lost");
    check(restored.wall_seconds_used == state.wall_seconds_used && restored.evidence_bytes_used == state.evidence_bytes_used,
          "cumulative resource accounting changed");
    check(restored.actor_adam == state.actor_adam && restored.critic_adam == state.critic_adam,
          "serialized optimizer state changed");
    check(restored.counters.completed_updates == 8 && restored.counters.actor_adam_calls == 256 &&
              restored.counters.critic_adam_calls == 256 && restored.counters.attempted_controls == 262144 &&
              restored.counters.attempted_substeps == 2621440, "replay counters changed");
    check(contract::total_controls == 1048576 && contract::total_adam_calls == 2048, "finite budget math changed");
    check(contract::runtime_version == 4 && contract::envelope_version == 2 && contract::query_arithmetic_version == 2, "candidate runtime/envelope version changed");
    for (uint64_t update : {uint64_t{0}, uint64_t{16}, uint64_t{24}, uint64_t{32}}) {
        const auto boundary = candidate(update);
        check(contract::write_candidate(decode(contract::write_candidate(boundary), boundary)) ==
                  contract::write_candidate(boundary), "chunk boundary checkpoint failed");
    }
    auto corrupted = bytes; corrupted[corrupted.size() / 2] ^= uint8_t{1};
    rejects([&] { (void)decode(corrupted, state); }, "candidate checksum corruption accepted");
    corrupted = bytes; corrupted.pop_back();
    rejects([&] { (void)decode(corrupted, state); }, "candidate truncation accepted");
    corrupted = bytes; corrupted.push_back(0);
    rejects([&] { (void)decode(corrupted, state); }, "candidate trailing bytes accepted");
    corrupted=bytes;patch(corrupted,size_t{16},uint32_t{4});reseal(corrupted);
    rejects([&]{(void)decode(corrupted,state);},"Exercise-v1 observation schema accepted by v2");
    for (size_t offset : {size_t{0}, size_t{4}, size_t{8}, size_t{12}, size_t{16}, size_t{28}, size_t{36}, size_t{40}}) {
        corrupted = bytes; patch(corrupted, offset, uint32_t{999}); reseal(corrupted);
        rejects([&] { (void)decode(corrupted, state); }, "candidate schema/runtime/topology mutation accepted");
    }
    corrupted = bytes; corrupted[52] ^= uint8_t{1}; reseal(corrupted);
    rejects([&] { (void)decode(corrupted, state); }, "canonical parent identity mutation accepted");
    auto expected = state; ++expected.parent_package_checksum;
    rejects([&] { (void)decode(bytes, expected); }, "wrong parent checksum accepted");
    expected = state; ++expected.protocol_checksum;
    rejects([&] { (void)decode(bytes, expected); }, "wrong protocol checksum accepted");
    expected = state; ++expected.source_identity_checksum;
    rejects([&] { (void)decode(bytes, expected); }, "wrong source identity accepted");
    expected = state; ++expected.world_stride;
    rejects([&] { (void)decode(bytes, expected); }, "wrong native world layout accepted");
    auto bad = state; bad.actor.pop_back();
    rejects([&] { (void)contract::write_candidate(bad); }, "wrong actor shape accepted");
    bad = state; bad.wall_seconds_used = -1;
    rejects([&] { (void)contract::write_candidate(bad); }, "negative cumulative wall time accepted");
    bad = state; bad.wall_seconds_used = std::numeric_limits<double>::infinity();
    rejects([&] { (void)contract::write_candidate(bad); }, "nonfinite cumulative wall time accepted");
    bad = state; bad.wall_seconds_used = contract::maximum_wall_seconds + 1;
    rejects([&] { (void)contract::write_candidate(bad); }, "cumulative wall-time budget extension accepted");
    bad = state; bad.evidence_bytes_used = contract::maximum_evidence_bytes + 1;
    rejects([&] { (void)contract::write_candidate(bad); }, "cumulative evidence-byte budget extension accepted");
    bad = state; bad.wall_seconds_used = contract::maximum_wall_seconds;
    bad.evidence_bytes_used = contract::maximum_evidence_bytes;
    check(decode(contract::write_candidate(bad), bad).evidence_bytes_used == contract::maximum_evidence_bytes,
          "exact cumulative resource budget boundary rejected");
    bad = state; bad.critic[0] = std::numeric_limits<float>::infinity();
    rejects([&] { (void)contract::write_candidate(bad); }, "nonfinite critic accepted");
    bad = state; bad.log_std[0] += .1f;
    rejects([&] { (void)contract::write_candidate(bad); }, "scaled or changed Gaussian sigma accepted");
    bad = state; bad.frozen_gain = 1.f;
    rejects([&] { (void)contract::write_candidate(bad); }, "changed saved holding amplitude accepted");
    bad = state; bad.task_state.pop_back();
    rejects([&] { (void)contract::write_candidate(bad); }, "truncated carried task state accepted");
    bad = state; bad.world_stride = std::numeric_limits<uint32_t>::max();
    rejects([&] { (void)contract::write_candidate(bad); }, "oversized carried world layout accepted");
    bad = state; ++bad.counters.actor_adam_calls;
    rejects([&] { (void)contract::write_candidate(bad); }, "optimizer/update count mismatch accepted");
    bad = state; --bad.counters.attempted_controls;
    rejects([&] { (void)contract::write_candidate(bad); }, "completed update with missing controls accepted");
    bad = state; bad.counters.completed_updates = 33;
    rejects([&] { (void)contract::write_candidate(bad); }, "extra update accepted");
    bad = state; bad.counters.attempted_controls = contract::total_controls + 1;
    rejects([&] { (void)contract::write_candidate(bad); }, "extended control budget accepted");
    bad = state; ++bad.counters.attempted_substeps;
    rejects([&] { (void)contract::write_candidate(bad); }, "impossible physical substep count accepted");
    bad = state; ++bad.counters.attempted_controls; bad.counters.attempted_substeps += 3;
    rejects([&] { (void)contract::write_candidate(bad); }, "rejected partial collection masqueraded as a good candidate");
    bad=state;++bad.sampler_decision_serial;rejects([&]{(void)contract::write_candidate(bad);},"sampler replay counter mutation accepted");
    bad = state; bad.actor_adam = adam_bytes(contract::critic_weights, state.counters.actor_adam_calls);
    rejects([&] { (void)contract::write_candidate(bad); }, "wrong Adam topology accepted");
    bad = state; patch(bad.actor_adam, sizeof(cudarl::nn::AdamConfig), int{255});
    rejects([&] { (void)contract::write_candidate(bad); }, "Adam byte age mismatch accepted");
    bad = state; patch(bad.actor_adam, size_t{0}, float{3e-4f});
    rejects([&] { (void)contract::write_candidate(bad); }, "wrong Adam learning rate accepted");
    bad = candidate(0);
    patch(bad.actor_adam, sizeof(cudarl::nn::AdamConfig) + sizeof(int) + sizeof(uint64_t), float{.1f});
    rejects([&] { (void)contract::write_candidate(bad); }, "inherited nonzero fresh moments accepted");

    TemporaryFile scratch;
    contract::save_candidate(scratch.file.string(), state);
    const auto disk = contract::load_candidate(scratch.file.string(), state.parent_package_checksum,
        state.protocol_checksum, state.source_identity_checksum, state.world_stride, state.task_stride);
    check(contract::write_candidate(disk) == bytes, "atomic outer-envelope roundtrip failed");
    {
        std::fstream file(scratch.file, std::ios::binary | std::ios::in | std::ios::out);
        check(static_cast<bool>(file), "could not open temporary outer envelope");
        file.seekp(24); const char wrong = 'x'; file.write(&wrong, 1); file.flush();
        check(static_cast<bool>(file), "could not mutate temporary outer envelope");
    }
    rejects([&] { (void)cp::read(scratch.file.string()); }, "outer checkpoint checksum corruption accepted");
}

void retention_checks() {
    contract::RetentionReceipt receipt;
    receipt.checkpoint_checksum = 11; receipt.parent_package_checksum = 12;
    receipt.protocol_checksum = 13; receipt.eval_source_identity = 14;
    receipt.cases_passed = 20; receipt.total = 20;receipt.completed_cases=20; receipt.controls = 30400; receipt.substeps = 304000; receipt.passed = true;
    const auto bytes = contract::write_retention_receipt(receipt);
    auto read = [&](const std::vector<uint8_t>& value) { return contract::read_retention_receipt(value, 11, 12, 13, 14); };
    check(contract::write_retention_receipt(read(bytes)) == bytes, "retention receipt roundtrip failed");
    rejects([&] { (void)contract::read_retention_receipt(bytes, 21, 12, 13, 14); }, "stale checkpoint retention accepted");
    rejects([&] { (void)contract::read_retention_receipt(bytes, 11, 12, 13, 24); }, "wrong evaluator source accepted");
    auto corrupted = bytes; corrupted[17] ^= uint8_t{1};
    rejects([&] { (void)read(corrupted); }, "corrupted retention receipt accepted");
    auto bad = receipt; bad.passed = false;
    rejects([&] { (void)contract::write_retention_receipt(bad); }, "failed retention released continuation");
    bad = receipt; bad.cases_passed = 9;
    rejects([&] { (void)contract::write_retention_receipt(bad); }, "partial retention released continuation");
    bad = receipt; bad.cases_passed = 0; bad.total = 0;
    rejects([&] { (void)contract::write_retention_receipt(bad); }, "empty retention released continuation");
    bad = receipt; bad.substeps = 1001;
    rejects([&] { (void)contract::write_retention_receipt(bad); }, "invalid retention substep accounting accepted");
    auto baseline=receipt;baseline.baseline_only=true;
    const auto baseline_bytes=contract::write_retention_receipt(baseline);
    check(contract::read_retention_receipt(baseline_bytes,11,12,13,14,true).passed,"Passing baseline receipt rejected");
    auto obsolete=baseline_bytes;patch(obsolete,size_t{4},uint32_t{2});reseal(obsolete);
    rejects([&]{(void)contract::read_retention_receipt(obsolete,11,12,13,14,true);},"Old baseline-exemption receipt schema admitted");
    rejects([&]{(void)read(baseline_bytes);},"Baseline receipt released a trained checkpoint");
    baseline.passed=false;rejects([&]{(void)contract::write_retention_receipt(baseline);},"Failed baseline released first optimizer call");
    baseline.passed=true;baseline.cases_passed=18;rejects([&]{(void)contract::write_retention_receipt(baseline);},"Partial inherited-skill baseline released training");
    baseline.cases_passed=20;baseline.controls=30399;rejects([&]{(void)contract::write_retention_receipt(baseline);},"Short baseline released training");
    baseline.controls=30400;
    baseline.physical_failures=1;rejects([&]{(void)contract::write_retention_receipt(baseline);},"Physical baseline fault released training");
    baseline.physical_failures=0;baseline.completed_cases=19;rejects([&]{(void)contract::write_retention_receipt(baseline);},"Incomplete baseline assessment released training");
}

void gaussian_checks() {
    const double stddev = static_cast<double>(contract::sigma);
    const double log_std = std::log(stddev);
    const std::array<double, 4> base_mean{.6, -.2, .1, .3};
    std::array<double, 4> mean{}, log_stds{}, action{};
    double reference = 0, scaled_reference = 0, normalization = 1;
    for (size_t j = 0; j < mean.size(); ++j) {
        mean[j] = static_cast<double>(contract::frozen_holding_gain) * base_mean[j];
        log_stds[j] = log_std;
        action[j] = mean[j] + (static_cast<double>(j) - 1.5) * .03;
        const double z = (action[j] - mean[j]) / stddev;
        reference += -.5 * z * z - log_std - .5 * cudarl::rl::kLog2Pi;
        const double scaled_std = stddev * static_cast<double>(contract::frozen_holding_gain);
        const double scaled_z = (action[j] - mean[j]) / scaled_std;
        scaled_reference += -.5 * scaled_z * scaled_z - std::log(scaled_std) - .5 * cudarl::rl::kLog2Pi;
        constexpr int intervals = 20000;
        const double width = 16 * stddev / intervals;
        double integral = 0;
        for (int i = 0; i < intervals; ++i) {
            const double point = mean[j] - 8 * stddev + (static_cast<double>(i) + .5) * width;
            integral += std::exp(cudarl::rl::gaussian_logp(&mean[j], &log_stds[j], &point, 1)) * width;
        }
        check(near(integral, 1, 1e-10), "Gaussian coordinate density is not normalized");
        normalization *= integral;
    }
    const double logp = cudarl::rl::gaussian_logp(mean.data(), log_stds.data(), action.data(), 4);
    check(near(logp, reference), "unscaled latent Gaussian log probability mismatch");
    check(std::abs(logp - scaled_reference) > .1, "holding gain accidentally scaled Gaussian noise");
    check(near(normalization, 1, 1e-10), "four-output Gaussian normalization failed");
    check(near(std::exp(logp - reference), 1), "unchanged-policy PPO likelihood ratio is not one");
}

void terminal_checks() {
    // Two independent synthetic episodes, stored [T][N]. The large rewards
    // after the t=1 boundaries must not leak back across either boundary.
    const std::array<double, 6> rewards{1, 2, 2, 3, 1000, 2000};
    const std::array<double, 6> values{.2, .3, .4, .6, .5, .7};
    const std::array<double, 6> dones{0, 0, 1, 1, 0, 0};
    const std::array<double, 6> truncated{0, 0, 0, 1, 0, 0};
    const std::array<double, 6> final_values{0, 0, 99, 5, 0, 0};
    const std::array<double, 2> last_values{8, 9};
    std::array<double, 6> advantages{}, returns{};
    const double discount = static_cast<double>(contract::gamma), trace = static_cast<double>(contract::lambda);
    for (int env = 0; env < 2; ++env)
        cudarl::rl::gae_one_env(3, 2, env, rewards.data(), values.data(), dones.data(), truncated.data(),
            final_values.data(), last_values.data(), discount, trace, advantages.data(), returns.data());
    check(near(returns[2], 2), "true terminal incorrectly bootstrapped");
    check(near(returns[3], 3 + discount * 5), "truncation missed final-observation bootstrap");
    check(near(returns[4], 1000 + discount * 8) && near(returns[5], 2000 + discount * 9),
          "nonterminal rollout boundary missed last-value bootstrap");
    check(near(advantages[0], 1 + discount * .4 - .2 + discount * trace * (2 - .4)),
          "true-terminal GAE leaked across reset");
    check(near(advantages[1], 2 + discount * .6 - .3 + discount * trace * (3 + discount * 5 - .6)),
          "truncated GAE leaked across reset");
    auto changed = rewards; changed[4] = -9000; changed[5] = -8000;
    std::array<double, 6> changed_advantages{}, changed_returns{};
    for (int env = 0; env < 2; ++env)
        cudarl::rl::gae_one_env(3, 2, env, changed.data(), values.data(), dones.data(), truncated.data(),
            final_values.data(), last_values.data(), discount, trace, changed_advantages.data(), changed_returns.data());
    check(changed_returns[0] == returns[0] && changed_returns[1] == returns[1] &&
              changed_returns[2] == returns[2] && changed_returns[3] == returns[3], "future episode changed past GAE");
}
void admission_checks() {
    namespace host=cudarl::rl::pond_ppo_host;
    const std::string cpu=R"({"schema":1,"protocol_id":"shoreline-product-v4-2026-11-23","runtime_version":4,"query_arithmetic_version":2,"preflight_only":false,"passed":true,"optimizer_steps":0,"reason":"product_physics_and_retention_passed_navigation_unqualified","completed_cases":36,"failed_cases":0,"skipped_cases":0,"retention_old_accepted_controls":30400,"retention_new_accepted_controls":30400,"bank_accepted_controls":1600,"half_step_accepted_controls":800,"package_checksum_fnv1a64":11})";
    const std::string gpu=R"({"schema":1,"protocol_id":"shoreline-product-v4-cuda-2026-11-23","query_arithmetic_version":2,"preflight_only":false,"passed":true,"optimizer_steps":0,"runtime_version":4,"arithmetic_profile":"cpu_compatible_no_implicit_fma_v1","canonical_radius_version":1,"reason":"matched_fixed_input_dynamics_passed","controls_completed_per_case":128,"CUDA_substep_readback_complete":true,"CPU_attempted_controls":1024,"CUDA_attempted_controls":1024,"CPU_attempted_substeps":10240,"CUDA_attempted_substeps":10240,"saved_payload_checksum_fnv1a64":11,"cpu_gate_checksum_fnv1a64":12})";
    host::validate_gate_documents(cpu,gpu,11,12);check(true,"correct current optimized gate rejected");
    const auto changed=[](std::string value,const std::string& old,const std::string& replacement){const auto i=value.find(old);if(i==std::string::npos)throw std::runtime_error("bad pure fixture");value.replace(i,old.size(),replacement);return value;};
    rejects([&]{host::validate_gate_documents(cpu,changed(gpu,"shoreline-product-v4-cuda-2026-11-23","shoreline-product-v3-optimized-2026-11-23"),11,12);},"closed unoptimized CUDA diagnostic admitted");
    rejects([&]{host::validate_gate_documents(cpu,changed(gpu,"\"preflight_only\":false","\"preflight_only\":true"),11,12);},"zero-integration CUDA preflight admitted");
    rejects([&]{host::validate_gate_documents(changed(cpu,"\"completed_cases\":36","\"completed_cases\":0"),gpu,11,12);},"incomplete CPU qualification admitted");
    rejects([&]{host::validate_gate_documents(cpu,changed(gpu,"\"CUDA_attempted_controls\":1024","\"CUDA_attempted_controls\":16"),11,12);},"short diagnostic physics admitted");
    rejects([&]{host::validate_gate_documents(cpu,changed(gpu,"\"CUDA_substep_readback_complete\":true","\"CUDA_substep_readback_complete\":false"),11,12);},"unknown CUDA physical counts admitted");
    rejects([&]{host::validate_gate_documents(cpu,gpu,11,13);},"CUDA tested another CPU gate");
    rejects([&]{host::validate_gate_documents(cpu,gpu,13,12);},"gate tested another original motor");
    rejects([&]{host::validate_gate_documents(cpu,changed(gpu,"\"optimizer_steps\":0","\"optimizer_steps\":1"),11,12);},"optimizer-modified gate admitted");
    rejects([&]{host::validate_gate_documents(changed(cpu,"\"runtime_version\":4","\"runtime_version\":3"),gpu,11,12);},"obsolete runtime receipt admitted");
    rejects([&]{host::validate_gate_documents(cpu,changed(gpu,"\"query_arithmetic_version\":2","\"query_arithmetic_version\":1"),11,12);},"uncorrected support arithmetic admitted");
    check(host::scalar(R"({"diagnostics":{"passed":true},"passed":false})","passed")=="false","nested passed masked failed root");
    check(host::sha256({})=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","CNG SHA256 empty reference mismatch");
}
} // namespace

int main(int argc,char** argv) {
    try {
        namespace host=cudarl::rl::pond_ppo_host;
        if(argc==2&&std::string(argv[1])=="--describe-host") {
            std::cout<<"{\"source_sha256\":\""<<host::source_sha256()<<"\",\"source_identity_fnv1a64\":\""<<host::source_identity()
                <<"\",\"qualified_physics_source_sha256\":\""<<host::physics_source_sha256()<<"\",\"physics_controls\":0,\"optimizer_calls\":0,\"CUDA_calls\":0}\n";return 0;
        }
        if(argc==5&&std::string(argv[1])=="--check-gates") {
            const auto cpu=host::whole_file(argv[2]);host::validate_gate_documents({cpu.begin(),cpu.end()},host::text_file(argv[3]),std::stoull(argv[4]),cp::checksum(cpu));
            std::cout<<"PASS exact corrected CPU-v4 and CUDA-v4 dynamics receipts physics_controls=0 optimizer_calls=0 CUDA_calls=0\n";return 0;
        }
        if(argc==7&&std::string(argv[1])=="--check-protocol") {
            host::prerequisites(argv[2],argv[3],argv[4],std::stoull(argv[5]),std::stoull(argv[6]));
            std::cout<<"PASS sealed machine protocol and physics bindings physics_controls=0 optimizer_calls=0 CUDA_calls=0\n";return 0;
        }
        if(argc!=1)throw std::runtime_error("Use no arguments, --describe-host, --check-gates CPU GPU PARENT_FNV or --check-protocol PROTOCOL CPU GPU EVAL_FNV PARENT_FNV");
        checkpoint_checks(); retention_checks(); gaussian_checks(); terminal_checks();admission_checks();
        std::cout << "PASS pond_ppo_pure checks=" << checks
                  << " physics_controls=0 physics_substeps=0 optimizer_calls=0 cuda_calls=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL pond_ppo_pure: " << error.what() << '\n';
        return 1;
    }
}
