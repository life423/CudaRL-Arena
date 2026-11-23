#ifndef CUDARL_REQUESTED_TURN_PPO_AUTHORIZATION_V1_H
#define CUDARL_REQUESTED_TURN_PPO_AUTHORIZATION_V1_H
#include "core/checkpoint.h"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace cudarl::requested_turn_ppo_authorization_v1 {
// A separate grant for this proposed full-motor lesson. Neither the held
// continuous-pond V2 grant nor the completed frozen probe grant is compatible.
inline uint64_t whole_file(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);
    if(!input)throw std::runtime_error("Cannot read required requested-turn frozen input");
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),{});
    return checkpoint::checksum(bytes);
}
inline void require_approved(const std::filesystem::path& root,uint64_t protocol) {
    if(std::filesystem::exists(root/"design-hold.json"))
        throw std::runtime_error("The requested-turn PPO proposal is on design hold");
    if(!std::filesystem::is_regular_file(root/"authorization.bin")||
       !std::filesystem::is_regular_file(root/"frozen-inputs.json"))
        throw std::runtime_error("Fresh explicit approval of this frozen requested-turn PPO lesson is required");
    const auto bytes=checkpoint::read((root/"authorization.bin").string());checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x31555452u||in.value<uint32_t>()!=1||
       in.value<uint64_t>()!=protocol||in.value<uint64_t>()!=whole_file(root/"frozen-inputs.json"))
        throw std::runtime_error("Approval belongs to a different requested-turn protocol or source freeze");
    const auto message=in.array<uint8_t>(65536);in.finish();
    if(message.empty())throw std::runtime_error("The actual fresh user grant text is missing");
}
}
#endif
