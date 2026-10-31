#include "check.h"
#include "rl/ppo_checkpoint.h"
#include <filesystem>
namespace {
namespace cp = cudarl::checkpoint;
std::vector<uint8_t> fixture() {
    cp::Writer out;
    out.value(uint32_t{0x31525050});
    out.value(uint32_t{2});
    out.value(uint32_t{1247037233});
    out.value(uint64_t{99});
    out.value(int{0});
    for (uint32_t v : {6u, 3u, 6u, 3u, 40u, 4u})
        out.value(v);
    out.array(std::vector<float>(2500, .01f));
    out.array(std::vector<float>(2401, .02f));
    out.array(std::vector<float>(4, -1));
    for (int count : {2500, 2401}) {
        out.value(cudarl::nn::AdamConfig{});
        out.value(int{0});
        out.array(std::vector<float>(count, 0));
        out.array(std::vector<float>(count, 0));
    }
    return out.data;
}
} // namespace
TEST_CASE("ppo_checkpoint: four-axis schema preserves actor without optimizer updates") {
    const auto bytes = fixture();
    const auto actor = cudarl::rl::read_ppo_actor(bytes, 99, 1247037233);
    CHECK(actor.parameters.size() == 2500 && actor.update == 0 && actor.actor_steps == 0 &&
          actor.critic_steps == 0);
    CHECK(bytes == fixture());
}
TEST_CASE("ppo_checkpoint: incompatible schema dimensions seed protocol nonfinite and trailing data reject") {
    for (int fault = 0; fault < 8; ++fault) {
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
            std::memcpy(bytes.data() + 52, &nan, 4);
        }
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
