// Read the recorded v5 rollout. This executable has no simulator, device or optimizer.
#include "core/checkpoint.h"
#include "nn/mlp_cpu.h"
#include "rl/pond_ppo_contract.h"
#include "rl/ppo_math.h"
#include <array>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

namespace cp = cudarl::checkpoint;
namespace pp = cudarl::rl::pond_ppo;
namespace nn = cudarl::nn;
namespace fs = std::filesystem;
constexpr int worlds = pp::worlds, steps = pp::controls_per_world, rows = worlds * steps;
constexpr int updates = 8, horizon = 300, trace_bytes = 192;
constexpr uint64_t v5_source = 14100886521839488064ull;
constexpr std::array<uint64_t, updates> v5_batch_checksums{
    0x3859bb0d0051cf03ull, 0xb8fffc601fc27035ull, 0xb670efd290657c13ull, 0x48429e8c7be7d25dull,
    0x44aa46f1289e96a6ull, 0xefd497fae4209c87ull, 0x144fd419fdd1ca7eull, 0xee52f5de74838a44ull};
static_assert(worlds == 512 && steps == 64);
void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
struct Row {
    float value, reward, next, gae, target, x, z, hx, hz, angle;
    uint32_t kind, episode, control, serial, age, events;
    bool done, truncated;
    float final_critic = 0;
};
using Histories = std::array<std::vector<Row>, worlds>;
template <class T> T field(const uint8_t *row, size_t byte) {
    T result;
    std::memcpy(&result, row + byte, sizeof(T));
    return result;
}
template <class T> std::vector<T> array(cp::Reader &input, size_t count) {
    auto result = input.array<T>(count);
    check(result.size() == count, "Unexpected v5 batch dimensions");
    return result;
}
#include "actor_probe.h"
nn::CpuMlp<float> reference_critic(const std::vector<float> &weights) {
    nn::CpuMlp<float> model({{pp::observations, 32, 32, 1},
                             {nn::Activation::Tanh, nn::Activation::Tanh, nn::Activation::Identity}});
    model.params() = weights;
    return model;
}
void read_batch(const fs::path &path, int update, Histories &histories, nn::CpuMlp<float> &final_critic,
                double &replay_max_abs, ActorProbe &probe) {
    const auto bytes = cp::read(path.string());
    check(cp::checksum(bytes) == v5_batch_checksums.at(update - 1), "Rollout differs from archived v5 study");
    cp::Reader in{bytes};
    check(in.value<uint32_t>() == 0x31425450 && in.value<uint32_t>() == 2 &&
              in.value<uint32_t>() == uint32_t(update) && in.value<uint32_t>() == steps,
          "Unexpected v5 header");
    in.value<uint64_t>(); // sampler serial
    check(in.value<uint64_t>() == v5_source, "Batch does not bind v5 source");
    array<uint8_t>(in, 1384 * worlds);
    array<uint8_t>(in, 360 * worlds); // starting world/task
    auto actor = array<float>(in, pp::actor_weights);
    if (update == 1)
        check(actor == probe.model.params(), "First collection actor differs from original 336");
    auto collection_critic = reference_critic(array<float>(in, pp::critic_weights));
    auto observations = array<float>(in, rows * pp::observations);
    array<float>(in, rows * pp::actions);
    array<float>(in, rows); // action log probabilities
    auto values = array<float>(in, rows), rewards = array<float>(in, rows);
    auto done = array<float>(in, rows), truncated = array<float>(in, rows);
    auto final = array<float>(in, rows), last = array<float>(in, worlds);
    auto attempted = array<uint64_t>(in, rows), substeps = array<uint64_t>(in, rows);
    array<int>(in, rows); // termination reason
    check(in.value<uint32_t>() == trace_bytes, "Unexpected trace ABI");
    auto traces = array<uint8_t>(in, rows * trace_bytes);
    in.finish();
    std::vector<float> gae(rows), targets(rows);
    for (int world = 0; world < worlds; ++world) {
        cudarl::rl::gae_one_env(steps, worlds, world, rewards.data(), values.data(), done.data(),
                                truncated.data(), final.data(), last.data(), pp::gamma, pp::lambda,
                                gae.data(), targets.data());
        for (int t = 0; t < steps; ++t) {
            int i = t * worlds + world;
            check(attempted[i] == 1 && substeps[i] == 10, "Unaccepted physical control");
            const auto *tr = traces.data() + i * trace_bytes;
            float next = done[i] ? (truncated[i] ? final[i] : 0.f)
                                 : (t + 1 == steps ? last[world] : values[i + worlds]);
            Row row{values[i],
                    rewards[i],
                    next,
                    gae[i],
                    targets[i],
                    field<float>(tr, 0),
                    field<float>(tr, 4),
                    field<float>(tr, 100),
                    field<float>(tr, 104),
                    field<float>(tr, 108),
                    field<uint32_t>(tr, 48),
                    field<uint32_t>(tr, 40),
                    field<uint32_t>(tr, 44),
                    field<uint32_t>(tr, 112),
                    field<uint32_t>(tr, 120),
                    field<uint32_t>(tr, 128),
                    done[i] != 0,
                    truncated[i] != 0};
            if (update == 1 && row.kind == 1 && row.control >= 41 && row.control <= 60)
                probe.observe(observations.data() + size_t(i) * pp::observations);
            if (row.control == 41 || row.control == 41 + horizon) {
                const auto *input = observations.data() + size_t(i) * pp::observations;
                collection_critic.forward(input, 1);
                replay_max_abs =
                    std::max(replay_max_abs, std::abs(double(*collection_critic.output()) - row.value));
                final_critic.forward(input, 1);
                row.final_critic = *final_critic.output();
            }
            check(std::isfinite(row.value) && std::isfinite(row.reward) && std::isfinite(row.next) &&
                      std::isfinite(row.gae) && row.kind < 3 && (!row.truncated || row.done),
                  "Invalid rollout row");
            const auto &h = histories[world];
            if (!h.empty()) {
                bool since_reset = h.back().done;
                check(row.episode == h.back().episode + unsigned(since_reset) &&
                          row.control == (since_reset ? 1u : h.back().control + 1),
                      "Episode discontinuity");
            }
            histories[world].push_back(row);
        }
    }
}
struct PairStats {
    uint64_t n = 0;
    double x = 0, y = 0, xx = 0, xy = 0, yy = 0;
    void add(double a, double b) {
        ++n;
        x += a;
        y += b;
        xx += a * a;
        xy += a * b;
        yy += b * b;
    }
    void print(bool predictions) const {
        double vx = xx - x * x / n, vy = yy - y * y / n, covariance = xy - x * y / n;
        std::cout << "{\"n\":" << n << ",\"mean_x\":" << x / n << ",\"mean_y\":" << y / n << ",\"r\":";
        if (vx > 0 && vy > 0)
            std::cout << covariance / std::sqrt(vx * vy);
        else
            std::cout << "null";
        if (predictions) {
            std::cout << ",\"prediction_bias\":" << (x - y) / n << ",\"explained_variance\":";
            if (vy > 0)
                std::cout << 1 - (vx + vy - 2 * covariance) / vy;
            else
                std::cout << "null";
        }
        std::cout << "}";
    }
};
std::string group(const Row &row) {
    if (row.kind == 0)
        return "rest";
    if (row.kind == 1)
        return "straight";
    if (row.serial == 0)
        return "turn_settle";
    return row.angle > 0 ? "left" : "right";
}
// Half-second course ending at the post-action COM sample end.
double error(const std::vector<Row> &h, int end, const Row &target) {
    check(end >= 10 && h[end].episode == h[end - 10].episode, "Missing course history");
    const double dx = h[end].x - h[end - 10].x, dz = h[end].z - h[end - 10].z;
    check(std::hypot(dx, dz) > 1e-9, "Undefined course");
    return std::atan2(std::abs(dx * target.hz - dz * target.hx), dx * target.hx + dz * target.hz) * 180 /
           3.141592653589793;
}
struct Audit {
    std::map<std::string, PairStats> lambda_targets, final_rollout_lambda, cash, targets, progress;
    std::map<std::string, PairStats> final_cash, final_targets;
    std::map<std::string, PairStats> onset_outcomes;
    uint64_t done = 0, truncated = 0, switched = 0;
};
Audit summarize(const Histories &histories) {
    Audit result;
    const double discount = std::pow(double(pp::gamma), horizon);
    for (const auto &h : histories) {
        const int length = static_cast<int>(h.size());
        for (int t = 0; t < length; ++t) {
            const Row &row = h[t];
            const auto key = group(row);
            result.lambda_targets[key].add(row.value, row.target);
            if (t >= (updates - 1) * steps)
                result.final_rollout_lambda[key].add(row.value, row.target);
            result.done += row.done;
            result.truncated += row.truncated;
            if (row.age == 1 && row.kind == 2) {
                bool succeeded = false, resolved = false;
                for (int j = t; j < length && h[j].episode == row.episode && h[j].serial == row.serial; ++j) {
                    if (h[j].events & 2u)
                        succeeded = true;
                    if (h[j].events & (2u | 4u | 8u | 16u))
                        resolved = true;
                }
                if (resolved)
                    result.onset_outcomes[key + (succeeded ? "_completed" : "_unsuccessful")].add(row.gae,
                                                                                                  row.value);
            }
            const int end = t + horizon - 1;
            if (end >= length || h[end].episode != row.episode || h[end].done)
                continue;
            // One origin per world for comparable 15-second evaluation. Turning origins
            // are the first requested action; straight and rest use the same control.
            if (row.control != 41)
                continue;
            double observed = 0;
            for (int j = end; j >= t; --j)
                observed = h[j].reward + pp::gamma * observed;
            result.cash[key].add(row.value, observed);
            result.targets[key].add(row.value, observed + discount * h[end].next);
            check(end + 1 < length && h[end + 1].control == 41 + horizon,
                  "Final critic endpoint observation missing");
            result.final_cash[key].add(row.final_critic, observed);
            result.final_targets[key].add(row.final_critic, observed + discount * h[end + 1].final_critic);
            if (row.kind != 2)
                continue;
            if (h[end].serial != row.serial) {
                ++result.switched;
                continue;
            }
            result.progress[key].add(row.gae, error(h, t - 1, row) - error(h, end, row));
        }
    }
    return result;
}
void emit(const char *name, const std::map<std::string, PairStats> &values, bool predictions) {
    std::cout << ",\n\"" << name << "\":{";
    bool comma = false;
    for (const auto &[key, stats] : values) {
        if (comma)
            std::cout << ',';
        std::cout << '\n' << '"' << key << "\":";
        stats.print(predictions);
        comma = true;
    }
    std::cout << "}";
}
#include "v6_ablation.h"
int main(int argc, char **argv) try {
    if (argc == 3 && std::string(argv[1]) == "--v6-ablation") {
        v6_ablation(argv[2]);
        return 0;
    }
    check(argc == 4,
          "Pass the v5 batch directory, original-amplitude-package.bin and historical ppo-update-336.bin");
    Histories histories;
    ActorProbe probe{argv[2], argv[3]};
    auto saved_candidate = cp::read((fs::path(argv[1]) / "candidate-8.bin").string());
    check(cp::checksum(saved_candidate) == 0x0c332aebf3460643ull,
          "Checkpoint differs from archived v5 update 8");
    auto candidate = pp::read_candidate(saved_candidate, 4517507952728352494ull, 8973322001545550014ull,
                                        v5_source, 1384, 360);
    check(candidate.counters.completed_updates == updates, "Wrong v5 final checkpoint");
    auto final_critic = reference_critic(candidate.critic);
    double replay_max_abs = 0;
    for (int i = 1; i <= updates; ++i)
        read_batch(fs::path(argv[1]) / ("batch-" + std::to_string(i) + ".bin"), i, histories, final_critic,
                   replay_max_abs, probe);
    auto result = summarize(histories);
    std::cout << std::setprecision(9)
              << "{\"protocol\":\"pond-motor-ppo-v5-half-second-2026-11-24\",\"training_updates_analyzed\":8,"
              << "\"recorded_transitions\":" << rows * updates
              << ",\"new_physics_steps\":0,\"new_updates\":0,"
              << "\"terminal_rows\":" << result.done << ",\"time_limit_rows\":" << result.truncated
              << ",\"gamma\":" << pp::gamma << ",\"lambda\":" << pp::lambda << ",\"rollout_ticks\":" << steps
              << ",\"gamma_300\":" << std::pow(double(pp::gamma), 300)
              << ",\"gae_64\":" << std::pow(double(pp::gamma) * pp::lambda, 64)
              << ",\"gae_300_unbounded\":" << std::pow(double(pp::gamma) * pp::lambda, 300)
              << ",\"different_request_at_15s_excluded_from_progress\":" << result.switched
              << ",\"final_critic_sha256_of_checkpoint\":"
                 "\"8eeae7a3d4bd4533af7bfea940005efa9c3799a570ac575e5770e6553f81a0b5\""
              << ",\"cpu_critic_replay_max_abs_error_on_1024_origin_and_endpoint_values\":" << replay_max_abs
              << ",\"source_hashes\":\"Eight raw batch SHA-256s in "
                 "pond-turning-v5-half-second/v5-training-summary.json\""
              << ",\"interpretation\":\"x is pre-action collection V for value tables. GAE lambda targets "
                 "are reconstructed with the trainer function, before normalization. Latest collection uses "
                 "checkpoint 7, before update 8. At control 41 there is one origin per world, held through "
                 "300 realized rewards across changing collection policies. Raw reward sum omits the "
                 "continuing tail. Bootstrapped 15s target adds gamma^300 times the recorded next V. These "
                 "are descriptive training trajectories, not a frozen held-out critic test. Progress x is "
                 "raw onset GAE, y is absolute half-second course error reduction in degrees at 15s. "
                 "Resolved-onset x is raw GAE and y is V. Unresolved requests are excluded from the latter. "
                 "Within-task aggregates avoid pooling task reward scales.\"";
    emit("collection_V_vs_GAE_target", result.lambda_targets, true);
    emit("last_collection_V_vs_GAE_target", result.final_rollout_lambda, true);
    emit("V_vs_observed_15s_reward_sum_at_control41", result.cash, false);
    emit("V_vs_bootstrapped_15s_target_at_control41", result.targets, true);
    emit("final_cp8_V_vs_same_observed_15s_reward_at_control41", result.final_cash, false);
    emit("final_cp8_V_vs_same_realized_rewards_plus_cp8_endpoint_bootstrap", result.final_targets, true);
    emit("onset_GAE_vs_15s_same_request_course_progress", result.progress, false);
    emit("resolved_request_onset_GAE_and_V", result.onset_outcomes, false);
    probe.print();
    std::cout << "\n}\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
