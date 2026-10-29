#ifndef CUDARL_BOULDER_SOURCE_H
#define CUDARL_BOULDER_SOURCE_H
#include "arch_policy.h"

namespace cudarl::rl {
// Read-only import of the actually applied actor, not an unapproved learner.
// Keep the entire parent payload so the earlier motor/Adam/world history remains
// recoverable even though this experiment trains only the goal chooser.
struct BoulderSource {
    ArchSource ancestry;
    arch::State frame{};
    arch::Context context{};
    std::vector<float> motor;
    std::vector<uint8_t> parent;
    int motor_update=0,learner_update=0,worlds=0;
    uint64_t fingerprint=0;
};
inline BoulderSource read_boulder_source(const std::vector<uint8_t>& bytes) {
    checkpoint::Reader in{bytes};BoulderSource out;
    if(in.value<uint32_t>()!=0x41524348u || in.value<uint32_t>()!=3u ||
       in.value<uint32_t>()!=sizeof(arch::State))
        throw std::runtime_error("Shelter practice requires a full v3 boulder save");
    out.worlds=in.value<int>();out.ancestry=ArchSource::read(in);out.frame=in.value<arch::State>();
    const auto count=static_cast<size_t>(swimmer_specification(4,arch::kObs).num_params());
    const auto approved=in.array<float>(count);const int approved_update=in.value<int>();
    out.motor=in.array<float>(count);out.motor_update=in.value<int>();
    out.context=in.value<arch::Context>();const auto practice=in.value<arch::Context>();
    const auto calm=in.array<float>(count);(void)in.value<int>();(void)in.value<bool>();(void)in.value<uint64_t>();
    const auto training=in.array<uint8_t>(128000000);in.finish();
    checkpoint::Reader learner{training};out.learner_update=learner.value<int>();
    if(out.worlds<16 || out.worlds>32768 || out.worlds%8 || out.context.basin.layout!=2 || practice.basin.layout!=2 ||
       approved.size()!=count || out.motor.size()!=count || calm.size()!=count || approved_update<out.motor_update ||
       out.motor_update<0 || out.learner_update<approved_update || !std::isfinite(deep::norm(out.frame.pond.body.position)) ||
       out.frame.pond.invalid)
        throw std::runtime_error("Invalid full boulder source");
    const std::vector<float>* snapshots[]={&approved,&out.motor,&calm};
    for(const auto* weights:snapshots)
        for(float x:*weights) if(!std::isfinite(x)) throw std::runtime_error("Non-finite boulder source actor");
    out.fingerprint=14695981039346656037ull;
    for(uint8_t x:bytes) {out.fingerprint^=x;out.fingerprint*=1099511628211ull;}
    out.parent=bytes;return out;
}
inline BoulderSource load_boulder_source(const std::string& path) {
    return read_boulder_source(checkpoint::read(path));
}
}
#endif
