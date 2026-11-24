#pragma once
#include "rl/pond_ppo_contract.h"
#include <algorithm>

namespace cudarl::residual_four_ppo {
namespace cp=checkpoint;
namespace pp=rl::pond_ppo;
inline constexpr char protocol_id[]="pond-motor-residual-four-ppo-v1-2026-11-24";
inline constexpr char course_rate_protocol_id[]="pond-motor-residual-four-course-rate-v1-2026-11-24";
inline constexpr char posture_protocol_id[]="pond-motor-residual-four-posture-v1-2026-11-24";
inline constexpr uint32_t inputs=42, hidden=16, outputs=4;
inline constexpr uint32_t parameters=inputs*hidden+hidden+hidden*outputs+outputs;
inline constexpr uint32_t output_offset=(inputs+1)*hidden;
static_assert(parameters==756);
inline bool output_layer_zero(const std::vector<float>& weights){
    return weights.size()==parameters && std::all_of(weights.begin()+output_offset,weights.end(),[](float v){return v==0.f;});
}
struct Actor{
    uint32_t update=0;
    uint64_t parent=0,protocol=0,training_controls=0,training_substeps=0,actor_adam_calls=0,critic_adam_calls=0;
    std::vector<float> weights;
};
inline void validate(const Actor& a){
    pp::require(a.parent&&a.protocol&&a.update<=32&&a.update%8==0,"Invalid residual identity/update");
    pp::require(a.training_controls==uint64_t(a.update)*pp::controls_per_update&&a.training_substeps==a.training_controls*10&&
                a.actor_adam_calls==uint64_t(a.update)*pp::adam_calls_per_update&&a.critic_adam_calls==a.actor_adam_calls,"Residual physical/optimizer count mismatch");
    pp::require(a.weights.size()==parameters&&std::all_of(a.weights.begin(),a.weights.end(),[](float x){return std::isfinite(x);}),"Invalid residual actor parameters");
    pp::require(a.update||output_layer_zero(a.weights),"Initial residual output must be identically zero");
}
inline std::vector<uint8_t> write_actor(const Actor& a){
    validate(a);cp::Writer out;
    for(uint32_t v:{0x31504c52u,1u,inputs,hidden,outputs,parameters,a.update})out.value(v);
    for(uint64_t v:{a.parent,a.protocol,a.training_controls,a.training_substeps,a.actor_adam_calls,a.critic_adam_calls})out.value(v);
    out.array(a.weights);pp::seal(out);return out.data;
}
inline Actor read_actor(const std::vector<uint8_t>& payload,uint64_t parent,uint64_t protocol){
    const auto inner=pp::checked_body(payload,16384);cp::Reader in{inner};
    for(uint32_t v:{0x31504c52u,1u,inputs,hidden,outputs,parameters})pp::require(in.value<uint32_t>()==v,"Wrong residual actor format");
    Actor a;a.update=in.value<uint32_t>();a.parent=in.value<uint64_t>();a.protocol=in.value<uint64_t>();
    pp::require(a.parent==parent&&a.protocol==protocol,"Residual actor source mismatch");
    a.training_controls=in.value<uint64_t>();a.training_substeps=in.value<uint64_t>();a.actor_adam_calls=in.value<uint64_t>();a.critic_adam_calls=in.value<uint64_t>();
    a.weights=in.array<float>(parameters);in.finish();validate(a);return a;
}
} // namespace cudarl::residual_four_ppo
