#ifndef CUDARL_FROZEN_REQUESTED_TURN_AUTHORIZATION_H
#define CUDARL_FROZEN_REQUESTED_TURN_AUTHORIZATION_H
#include "core/checkpoint.h"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
namespace cudarl::frozen_requested_turn_authorization {
inline uint64_t whole_file(const std::filesystem::path& path) {
    std::ifstream stream(path,std::ios::binary);
    if(!stream)throw std::runtime_error("Cannot read required frozen input");
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(stream)),{});
    return checkpoint::checksum(bytes);
}
inline void require_approved(const std::filesystem::path& root,uint64_t protocol) {
    if(std::filesystem::exists(root/"design-hold.json"))
        throw std::runtime_error("This probe is on design hold");
    if(!std::filesystem::is_regular_file(root/"authorization.bin")||!std::filesystem::is_regular_file(root/"frozen-inputs.json"))
        throw std::runtime_error("Fresh approval and the frozen input manifest are required");
    const auto bytes=checkpoint::read((root/"authorization.bin").string());checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x31555446u||in.value<uint32_t>()!=1||in.value<uint64_t>()!=protocol||
       in.value<uint64_t>()!=whole_file(root/"frozen-inputs.json"))
        throw std::runtime_error("Approval belongs to a different probe or freeze");
    const auto text=in.array<uint8_t>(65536);in.finish();
    if(text.empty())throw std::runtime_error("Actual fresh approval text is missing");
}
}
#endif
