#include "check.h"
#include "nn/mlp_cpu.h"
#include "rl/ppo_checkpoint.h"
#include "rl/ppo_diagnostics.h"
#include <filesystem>
namespace {
namespace cp = cudarl::checkpoint;
std::vector<uint8_t> fixture(bool trained = false) {
    cp::Writer out;
    out.value(uint32_t{0x31525050});
    out.value(uint32_t{2});
    out.value(uint32_t{1247037233});
    out.value(uint64_t{99});
    out.value(trained ? 128 : 0);
    for (uint32_t v : {6u, 3u, 6u, 4u, 40u, 4u})
        out.value(v);
    out.array(std::vector<float>(2500, .01f));
    out.array(std::vector<float>(2401, .02f));
    out.array(std::vector<float>(4, -1));
    for (int count : {2500, 2401}) {
        auto config = cudarl::nn::AdamConfig{};
        if (trained)
            config.lr = .0007f;
        out.value(config);
        out.value(trained ? 5120 : 0);
        out.array(std::vector<float>(count, trained ? .125f : 0));
        out.array(std::vector<float>(count, trained ? .25f : 0));
    }
    return out.data;
}
TEST_CASE("ppo_diagnostics: same-action ratios inverse permutation and value summaries are independent") {
    namespace rl = cudarl::rl;
    const auto identity = rl::diagnostic_ratios({-2, -1}, {-2, -1}, .2);
    CHECK(identity.k1 == 0 && identity.k3 == 0 && identity.clipped_fraction == 0);
    const auto changed = rl::diagnostic_ratios({0, 0}, {.3f, -.3f}, .2);
    CHECK(fabs(changed.k1) < 1e-12 && fabs(changed.k3 - (cosh(.3) - 1)) < 1e-7 &&
          changed.clipped_fraction == 1);
    const auto negative_sample = rl::diagnostic_ratios({0}, {.1f}, .2);
    CHECK(negative_sample.k1 < 0 && negative_sample.k3 >= 0);
    const auto perm = rl::make_permutation(37, 123);
    std::vector<float> shuffled(37);
    for (uint32_t i = 0; i < 37; ++i)
        shuffled[i] = float(rl::permute(perm, i));
    const auto ordered = rl::diagnostic_original_order(shuffled, perm);
    for (int i = 0; i < 37; ++i)
        CHECK(ordered[i] == float(i));
    const auto moments = rl::diagnostic_moments({1, 2, 3});
    CHECK(moments.mean == 2 && fabs(moments.variance - 2. / 3) < 1e-12);
    const auto exact = rl::diagnostic_value({1, 2, 3}, {1, 2, 3});
    CHECK(exact.half_mse == 0 && exact.explained_variance == 1 && exact.variance_defined);
    const auto constant = rl::diagnostic_value({2, 2, 2}, {1, 2, 3});
    CHECK(fabs(constant.half_mse - 1. / 3) < 1e-12 && constant.explained_variance == 0);
    CHECK(!rl::diagnostic_value({1, 1}, {1, 1}).variance_defined);
}
} // namespace
TEST_CASE("ppo_checkpoint: full resume retains critic both Adam states sigma and counters") {
    const auto bytes = fixture(true);
    const auto state = cudarl::rl::read_ppo_training_state(bytes, 99);
    CHECK(state.actor.seed == 1247037233 && state.actor.update == 128 && state.critic.size() == 2401);
    CHECK(state.actor.parameters.size() == 2500 && state.actor.log_sigma == std::vector<float>(4, -1));
    for (const auto *optimizer : {&state.actor_adam, &state.critic_adam}) {
        CHECK(optimizer->steps == 5120 && optimizer->config.lr == .0007f);
        CHECK(optimizer->first.front() == .125f && optimizer->second.back() == .25f);
        const auto payload = cudarl::rl::adam_payload(*optimizer);
        cp::Reader in{payload};
        CHECK(in.value<cudarl::nn::AdamConfig>().lr == .0007f);
        CHECK(in.value<int>() == 5120);
        CHECK(in.array<float>() == optimizer->first);
        CHECK(in.array<float>() == optimizer->second);
        in.finish();
    }
    CHECK(cudarl::rl::read_ppo_actor(bytes, 99, 1247037233).parameters == state.actor.parameters);
}
TEST_CASE("path_scale: checkpoint conversion transforms columns and cotangents without resets") {
    namespace rl = cudarl::rl;
    const auto original = rl::read_ppo_training_state(fixture(true), 99);
    const auto scaled = rl::rescale_ppo_path_inputs(original);
    CHECK(scaled.actor.observation_schema == 2 && original.actor.observation_schema == 1);
    CHECK(scaled.actor.update == original.actor.update);
    for (int network = 0; network < 2; ++network) {
        const auto &a = network ? original.critic : original.actor.parameters;
        const auto &b = network ? scaled.critic : scaled.actor.parameters;
        const auto &oa = network ? original.critic_adam : original.actor_adam;
        const auto &ob = network ? scaled.critic_adam : scaled.actor_adam;
        CHECK(oa.steps == ob.steps && oa.config.lr == ob.config.lr && oa.config.eps == ob.config.eps);
        for (size_t i = 0; i < a.size(); ++i) {
            const bool changed = i < 1280 && i % 40 < 3;
            CHECK(b[i] == (changed ? a[i] / 36.f : a[i]));
            CHECK(ob.first[i] == (changed ? oa.first[i] * 36.f : oa.first[i]));
            CHECK(ob.second[i] == (changed ? oa.second[i] * 1296.f : oa.second[i]));
        }
    }
    const auto bytes = rl::write_ppo_training_state(scaled);
    const auto back = rl::read_ppo_training_state(bytes, 99, original.actor.seed, 2);
    CHECK(back.actor.parameters == scaled.actor.parameters && back.critic == scaled.critic);
    CHECK(back.actor_adam.first == scaled.actor_adam.first &&
          back.critic_adam.second == scaled.critic_adam.second);
    bool rejected = false;
    try {
        (void)rl::read_ppo_actor(bytes, 99, original.actor.seed);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    CHECK(rejected); // Legacy observation evaluator must not silently consume schema2.
    CHECK(rl::read_ppo_actor(bytes, 99, original.actor.seed, 2).observation_schema == 2);
    rejected = false;
    try {
        (void)rl::rescale_ppo_path_inputs(scaled);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    CHECK(rejected);
}
TEST_CASE("ppo_checkpoint: four-axis schema preserves actor without optimizer updates") {
    const auto bytes = fixture();
    const auto actor = cudarl::rl::read_ppo_actor(bytes, 99, 1247037233);
    CHECK(actor.parameters.size() == 2500 && actor.update == 0 && actor.actor_steps == 0 &&
          actor.critic_steps == 0);
    CHECK(bytes == fixture());
}
TEST_CASE("ppo_checkpoint: reward4 and reward5 retain explicit provenance without changing weights") {
    const auto old = fixture(true);
    auto revised = old;
    revised[36] = 5; // Existing on-disk reward-version field, no schema conversion.
    const auto a = cudarl::rl::read_ppo_training_state(old, 99);
    const auto b = cudarl::rl::read_ppo_training_state(revised, 99);
    CHECK(a.actor.reward_version == 4 && b.actor.reward_version == 5);
    CHECK(a.actor.parameters == b.actor.parameters && a.critic == b.critic);
    CHECK(a.actor_adam.first == b.actor_adam.first && a.actor_adam.steps == b.actor_adam.steps);
    CHECK(a.critic_adam.second == b.critic_adam.second);
}
TEST_CASE("ppo_checkpoint: incompatible schema dimensions seed protocol nonfinite and trailing data reject") {
    for (int fault = 0; fault < 9; ++fault) {
        auto bytes = fixture();
        if (fault < 6)
            bytes[fault == 0   ? 4
                  : fault == 1 ? 8
                  : fault == 2 ? 12
                  : fault == 3 ? 24
                  : fault == 4 ? 40
                               : 44] ^= 1;
        if (fault == 6) {
            const uint32_t nan = 0x7fc00000;
            std::memcpy(bytes.data() + 56, &nan, 4);
        }
        if (fault == 8)
            bytes[36] = 3;
        if (fault == 7)
            bytes.push_back(0);
        bool rejected = false;
        try {
            (void)cudarl::rl::read_ppo_actor(bytes, 99, 1247037233);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        CHECK(rejected);
    }
}
TEST_CASE("ppo_checkpoint: historical two-output actor rejects without conversion") {
    const auto path =
        std::filesystem::path(__FILE__).parent_path().parent_path() / "assets/ppo-forward-v1/checkpoint.bin";
    bool rejected = false;
    try {
        (void)cudarl::rl::read_ppo_actor(cp::read(path.string()), 0, 1347440433);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    CHECK(rejected);
}
TEST_CASE("ppo_checkpoint: schema3 records correlation and rejects invalid stochastic metadata") {
    for (float rho : {0.f, .8f}) {
        auto bytes = fixture();
        bytes[4] = 3;
        const uint32_t stochastic = 1;
        std::vector<uint8_t> addition(8);
        std::memcpy(addition.data(), &stochastic, 4);
        std::memcpy(addition.data() + 4, &rho, 4);
        bytes.insert(bytes.begin() + 48, addition.begin(), addition.end());
        const auto actor = cudarl::rl::read_ppo_actor(bytes, 99, 1247037233);
        CHECK(actor.correlation == rho && actor.parameters.size() == 2500);
        bytes[48] = 2;
        bool rejected = false;
        try {
            (void)cudarl::rl::read_ppo_actor(bytes, 99, 1247037233);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        CHECK(rejected);
    }
}
