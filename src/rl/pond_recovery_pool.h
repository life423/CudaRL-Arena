#ifndef CUDARL_POND_RECOVERY_POOL_H
#define CUDARL_POND_RECOVERY_POOL_H
#include "progress_selector_policy.h"
#include <set>
namespace cudarl::rl {
inline void validate_recovery(const progress_selector::RecoveryStart& r) {
    if(r.target<0 || r.target>2 || r.state.steps<0 || r.state.since_food<0 || !std::isfinite(deep::norm(r.current)) || deep::norm(r.current)>.01001f)
        throw std::runtime_error("Invalid recovery-state counters or current");
    float obs[40];progress_selector::observe(r.state,r.target,r.history,obs);
    for(float x:obs) if(!std::isfinite(x)) throw std::runtime_error("Non-finite recovery observation");
    for(int f=0;f<3;++f) {
        if(!deep_pond::water(r.state.food[f])) throw std::runtime_error("Recovery food must remain underwater with bottom clearance");
        if(r.history.placements[f]!=r.state.placements[f] || !std::isfinite(r.history.distance[f]) || !std::isfinite(r.history.progress[f])) throw std::runtime_error("Invalid recovery progress history");
    }
    for(float v:r.state.body.velocity) if(!std::isfinite(v)) throw std::runtime_error("Non-finite recovery velocity");
}
struct RecoveryPool {
    SelectorPolicy baseline;std::vector<progress_selector::RecoveryStart> starts;
    void save(const std::string& path) const {
        checkpoint::Writer out;out.value(uint32_t{0x42525031});out.value(uint32_t{1});baseline.write(out);out.array(starts);checkpoint::write_atomic(path,out.data);
    }
    static RecoveryPool load(const std::string& path) {
        const auto bytes=checkpoint::read(path);checkpoint::Reader in{bytes};
        if(in.value<uint32_t>()!=0x42525031u || in.value<uint32_t>()!=1) throw std::runtime_error("Unsupported recovery pool");
        RecoveryPool out;out.baseline=SelectorPolicy::read(in);out.starts=in.array<progress_selector::RecoveryStart>(4096);in.finish();
        if(out.starts.empty()) throw std::runtime_error("Recovery pool is empty");
        std::set<std::pair<uint32_t,uint32_t>> ids;
        for(const auto& r:out.starts) {validate_recovery(r);if(!ids.emplace(r.state.seed,r.state.environment).second) throw std::runtime_error("Repeated recovery source world");}
        return out;
    }
};
}
#endif
