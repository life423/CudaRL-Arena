#include "check.h"
#include "rl/ppo_checkpoint.h"
#include <filesystem>
namespace {
namespace cp = cudarl::checkpoint;
std::filesystem::path asset() {
    return std::filesystem::path(__FILE__).parent_path().parent_path() / "assets/ppo-forward-v1";
}
} // namespace
TEST_CASE("ppo_checkpoint: frozen loading preserves exact accepted actor and validates auxiliary state "
          "without updates") {
    const auto protocol = cp::checksum(cp::read((asset() / "protocol.bin").string()));
    const auto bytes = cp::read((asset() / "checkpoint.bin").string());
    const auto initial_bytes = cp::read((asset() / "initial-checkpoint.bin").string());
    const auto actor = cudarl::rl::read_ppo_actor(bytes, protocol, 1347440433),
               initial = cudarl::rl::read_ppo_actor(initial_bytes, protocol, 1347440433);
    CHECK(actor.parameters.size() == 2498 && actor.update == 64 && actor.actor_steps == 1210 &&
          actor.critic_steps == 1210);
    CHECK(initial.update == 0 && initial.actor_steps == 0 && initial.critic_steps == 0 &&
          actor.parameters != initial.parameters);
    CHECK(cp::read((asset() / "checkpoint.bin").string()) == bytes);
}
TEST_CASE("ppo_checkpoint: wrong schema seed protocol shape nonfinite parameters and trailing data reject") {
    const auto protocol = cp::checksum(cp::read((asset() / "protocol.bin").string()));
    const auto valid = cp::read((asset() / "checkpoint.bin").string());
    for (int fault = 0; fault < 7; ++fault) {
        auto bytes = valid;
        if (fault == 0)
            bytes[0] ^= 1;
        if (fault == 1)
            bytes[8] ^= 1;
        if (fault == 2)
            bytes[12] ^= 1;
        if (fault == 3)
            bytes[24] ^= 1;
        if (fault == 4)
            bytes[32] ^= 1;
        if (fault == 5) {
            const uint32_t nan = 0x7fc00000;
            std::memcpy(bytes.data() + 40, &nan, 4);
        }
        if (fault == 6)
            bytes.push_back(0);
        bool rejected = false;
        try {
            (void)cudarl::rl::read_ppo_actor(bytes, protocol, 1347440433);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        CHECK(rejected);
    }
}
