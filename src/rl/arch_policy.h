#ifndef CUDARL_ARCH_POLICY_H
#define CUDARL_ARCH_POLICY_H
#include "progress_selector_policy.h"
#include "envs/swimmer/arch.h"
#include <limits>

namespace cudarl::rl {
// The inherited actor is the validated outer snapshot of the feeding learner.
// The mutable, unvalidated PPO actor inside that file is deliberately ignored.
struct ArchSource {
    ProgressPolicy policy;
    std::vector<float> motor;
    int parent_update=0;
    uint64_t fingerprint=0;
    void validate() const {
        if(parent_update<0 || motor.size()!=static_cast<size_t>(swimmer_specification(4,feeding::kObs).num_params()))
            throw std::runtime_error("Invalid inherited feeding motor");
        for(float weight:motor) if(!std::isfinite(weight)) throw std::runtime_error("Non-finite inherited feeding motor");
    }
    void write(checkpoint::Writer& out) const {
        validate();policy.write(out);out.array(motor);out.value(parent_update);out.value(fingerprint);
    }
    static ArchSource read(checkpoint::Reader& in) {
        ArchSource source;source.policy=ProgressPolicy::read(in);
        source.motor=in.array<float>(swimmer_specification(4,feeding::kObs).num_params());
        source.parent_update=in.value<int>();source.fingerprint=in.value<uint64_t>();source.validate();return source;
    }
};
inline std::vector<float> arch_motor_parameters(const std::vector<float>& original) {
    const auto old=swimmer_specification(4,feeding::kObs),next=swimmer_specification(4,arch::kObs);
    if(original.size()!=static_cast<size_t>(old.num_params())) throw std::runtime_error("Invalid arch source dimensions");
    std::vector<float> weights(static_cast<size_t>(next.num_params()),0);
    for(int layer=0;layer<old.num_layers();++layer) {
        for(int row=0;row<old.out(layer);++row) for(int col=0;col<old.in(layer);++col)
            weights[next.weight_offset(layer)+row*next.in(layer)+col]=original[old.weight_offset(layer)+row*old.in(layer)+col];
        for(int row=0;row<old.out(layer);++row) weights[next.bias_offset(layer)+row]=original[old.bias_offset(layer)+row];
    }
    return weights;
}
inline ArchSource load_arch_source(const std::string& path) {
    const auto bytes=checkpoint::read(path);checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x46454544u || in.value<uint32_t>()!=2u || in.value<uint32_t>()!=sizeof(feeding::State))
        throw std::runtime_error("Arch learning needs a compatible full CUDA feeding save");
    const int worlds=in.value<int>();
    if(worlds<16 || worlds>2048 || worlds%8) throw std::runtime_error("Invalid source feeding population");
    ArchSource source;source.policy=ProgressPolicy::read(in);
    (void)in.value<feeding::State>();const auto context=in.value<feeding::Context>();
    if(context.count<0 || context.count>32) throw std::runtime_error("Invalid source feeding context");
    source.motor=in.array<float>(swimmer_specification(4,feeding::kObs).num_params());source.parent_update=in.value<int>();source.validate();
    // Validate framing and the publication counter without copying the entire
    // nested optimizer/world snapshot a second time. It remains untouched.
    const auto size=in.value<uint64_t>();
    if(size<sizeof(int) || size>128000000u || size>bytes.size()-in.position)
        throw std::runtime_error("Invalid source feeding learner payload");
    int learner_update=0;std::memcpy(&learner_update,bytes.data()+in.position,sizeof(learner_update));
    if(learner_update<source.parent_update) throw std::runtime_error("Invalid source feeding publication counter");
    in.position+=static_cast<size_t>(size);in.finish();
    // Stable provenance identifier for the complete checksummed input payload.
    // This is an identifier, not a cryptographic integrity claim.
    source.fingerprint=14695981039346656037ull;
    for(uint8_t byte:bytes) {source.fingerprint^=byte;source.fingerprint*=1099511628211ull;}
    return source;
}
}
#endif
