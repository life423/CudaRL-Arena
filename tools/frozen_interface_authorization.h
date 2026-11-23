#ifndef CUDARL_FROZEN_INTERFACE_AUTHORIZATION_H
#define CUDARL_FROZEN_INTERFACE_AUTHORIZATION_H
#include "core/checkpoint.h"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
namespace cudarl::frozen_interface_authorization {
inline uint64_t whole_file(const std::filesystem::path& path) {
    std::ifstream stream(path,std::ios::binary);
    if(!stream)throw std::runtime_error("Cannot read required comparison input");
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(stream)),{});
    return checkpoint::checksum(bytes);
}
inline void require_approved(const std::filesystem::path& root,uint64_t protocol) {
    if(std::filesystem::exists(root/"design-hold.json"))
        throw std::runtime_error("The comparison is on design hold");
    if(!std::filesystem::is_regular_file(root/"authorization.bin") ||
       !std::filesystem::is_regular_file(root/"frozen-inputs.json") ||
       !std::filesystem::is_regular_file(root/"independent-preparation-verification.json"))
        throw std::runtime_error("Comparison approval, freeze and preparation proof are required");
    const auto bytes=checkpoint::read((root/"authorization.bin").string());checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x31554946u || in.value<uint32_t>()!=1 ||
       in.value<uint64_t>()!=protocol || in.value<uint64_t>()!=whole_file(root/"frozen-inputs.json") ||
       in.value<uint64_t>()!=whole_file(root/"independent-preparation-verification.json"))
        throw std::runtime_error("Approval belongs to a different comparison or preparation");
    const auto text=in.array<uint8_t>(65536);in.finish();
    if(text.empty())throw std::runtime_error("The actual current approval text is missing");
}
}
#endif
