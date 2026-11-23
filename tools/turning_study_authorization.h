#ifndef CUDARL_TURNING_STUDY_AUTHORIZATION_H
#define CUDARL_TURNING_STUDY_AUTHORIZATION_H
#include "core/checkpoint.h"
#include <filesystem>
#include <stdexcept>

namespace cudarl::study_authorization {
// Pure preparation never requires this marker. Only the root execution owner
// creates it after an explicit user grant bound to this new frozen protocol.
inline void require_approved(const std::filesystem::path& root,uint64_t protocol) {
    if(!std::filesystem::is_regular_file(root/"authorization.bin"))
        throw std::runtime_error("Fresh user approval is required before physical execution");
    const auto bytes=checkpoint::read((root/"authorization.bin").string());
    checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x32554150u || in.value<uint32_t>()!=1 || in.value<uint64_t>()!=protocol)
        throw std::runtime_error("Fresh user approval is absent or bound to a different study");
    const auto message=in.array<uint8_t>(65536);in.finish();
    if(message.empty())throw std::runtime_error("The actual user approval text is missing");
}
}
#endif
