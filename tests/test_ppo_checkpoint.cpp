#include "check.h"
#include "rl/ppo_checkpoint.h"
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
TEST_CASE("ppo_checkpoint: four-axis schema preserves actor without optimizer updates") {
    const auto bytes = fixture();
    const auto actor = cudarl::rl::read_ppo_actor(bytes, 99, 1247037233);
    CHECK(actor.parameters.size() == 2500 && actor.update == 0 && actor.actor_steps == 0 &&
          actor.critic_steps == 0);
    CHECK(bytes == fixture());
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
