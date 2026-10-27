#ifndef CUDARL_RL_POND_SELECTOR_POLICY_H
#define CUDARL_RL_POND_SELECTOR_POLICY_H
#include "swimmer_spec.h"
#include "envs/swimmer/pond_selector.h"
#include "core/checkpoint.h"
namespace cudarl::rl {
namespace ps=cudarl::pond_selector;
inline nn::MlpSpec selector_spec() {return swimmer_specification(1,ps::kFeatures);}
struct SelectorPolicy {
    int update=0,motor_update=0,decision_steps=20,foods=-1,trials=64;
    std::vector<float> selector,motor;
    double food_per_minute() const {return trials>0?double(foods)/trials:0;}
    void write(checkpoint::Writer& out) const {
        out.value(update);out.value(motor_update);out.value(decision_steps);out.value(foods);out.value(trials);out.array(selector);out.array(motor);
    }
    static SelectorPolicy read(checkpoint::Reader& in) {
        SelectorPolicy p;p.update=in.value<int>();p.motor_update=in.value<int>();p.decision_steps=in.value<int>();p.foods=in.value<int>();p.trials=in.value<int>();
        p.selector=in.array<float>(selector_spec().num_params());p.motor=in.array<float>(swimmer_specification(4,33).num_params());
        if(p.update<0 || p.motor_update<0 || p.decision_steps<1 || 1200%p.decision_steps!=0 || p.foods<0 || p.trials<1 ||
            p.selector.size()!=static_cast<size_t>(selector_spec().num_params()) || p.motor.size()!=static_cast<size_t>(swimmer_specification(4,33).num_params())) throw std::runtime_error("Invalid selector policy dimensions");
        for(const auto& values:{p.selector,p.motor}) for(float v:values) if(!std::isfinite(v)) throw std::runtime_error("Non-finite selector policy");return p;
    }
    void save(const std::string& path) const {checkpoint::Writer out;out.value(uint32_t{0x53504F31});out.value(uint32_t{2});write(out);checkpoint::write_atomic(path,out.data);}
    static SelectorPolicy load(const std::string& path) {
        const auto bytes=checkpoint::read(path);checkpoint::Reader in{bytes};
        if(in.value<uint32_t>()!=0x53504F31u || in.value<uint32_t>()!=2) throw std::runtime_error("Unsupported selector policy");
        auto p=read(in);in.finish();return p;
    }
};
// Same orthogonal initialization and output gain as the PPO actors. This is
// fixed, deterministic inference from random weights, not sampled random torque.
inline SelectorPolicy initial_pond_policy(uint32_t seed=7) {
    SelectorPolicy policy;
    const std::vector<float> gains{std::sqrt(2.f),std::sqrt(2.f),.01f};
    policy.motor=nn::orthogonal_params(swimmer_specification(4,33),seed,gains);
    policy.selector=nn::orthogonal_params(selector_spec(),seed+1,gains);
    return policy;
}
}
#endif
