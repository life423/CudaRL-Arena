#ifndef CUDARL_PROGRESS_SELECTOR_POLICY_H
#define CUDARL_PROGRESS_SELECTOR_POLICY_H
#include "pond_selector_policy.h"
#include "envs/swimmer/progress_selector.h"
namespace cudarl::rl {
inline nn::MlpSpec progress_selector_spec() {return swimmer_specification(1,progress_selector::kFeatures);}
inline std::vector<float> transfer_progress_actor(const std::vector<float>& weights) {
    const auto old=selector_spec(),next=progress_selector_spec();
    if(weights.size()!=static_cast<size_t>(old.num_params())) throw std::invalid_argument("Invalid source selector");
    std::vector<float> result(static_cast<size_t>(next.num_params()),0);
    for(int l=0;l<old.num_layers();++l) {
        for(int row=0;row<old.out(l);++row) for(int column=0;column<old.in(l);++column)
            result[next.weight_offset(l)+row*next.in(l)+column]=weights[old.weight_offset(l)+row*old.in(l)+column];
        for(int row=0;row<old.out(l);++row) result[next.bias_offset(l)+row]=weights[old.bias_offset(l)+row];
    }
    return result;
}
struct ProgressPolicy {
    int update=0;SelectorPolicy baseline;std::vector<float> actor;
    void write(checkpoint::Writer& out) const {out.value(update);baseline.write(out);out.array(actor);}
    static ProgressPolicy read(checkpoint::Reader& in) {
        ProgressPolicy result;result.update=in.value<int>();result.baseline=SelectorPolicy::read(in);result.actor=in.array<float>(progress_selector_spec().num_params());
        if(result.update<0 || result.actor.size()!=static_cast<size_t>(progress_selector_spec().num_params())) throw std::runtime_error("Invalid progress policy");
        for(float x:result.actor) if(!std::isfinite(x)) throw std::runtime_error("Non-finite progress policy");return result;
    }
    void save(const std::string& path) const {checkpoint::Writer out;out.value(uint32_t{0x50534731});out.value(uint32_t{1});write(out);checkpoint::write_atomic(path,out.data);}
    static ProgressPolicy load(const std::string& path) {const auto bytes=checkpoint::read(path);checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=0x50534731u || in.value<uint32_t>()!=1) throw std::runtime_error("Unsupported progress policy");auto result=read(in);in.finish();return result;}
};
}
#endif
