#ifndef CUDARL_APPEND_HOLD_CHECKPOINT_H
#define CUDARL_APPEND_HOLD_CHECKPOINT_H
#include "rl/ppo_checkpoint.h"
#include <algorithm>
#include <stdexcept>
namespace cudarl::rl {
struct AppendedHoldState {
    PpoTrainingState ppo;
    int actor_column_age = 0, critic_column_age = 0;
};
inline std::vector<float> append_zero_input_column(const std::vector<float>& old) {
    if (old.size() != 2500 && old.size() != 2401)
        throw std::runtime_error("Append migration requires original40-input network");
    std::vector<float> next(old.size() + 32, 0.f);
    for (int r=0;r<32;++r)
        std::copy_n(old.begin()+r*40,40,next.begin()+r*41);
    std::copy(old.begin()+1280,old.end(),next.begin()+1312);
    return next;
}
inline AppendedHoldState migrate_appended_hold(const PpoTrainingState& original) {
    if (original.actor.observation_schema != motor_observation::task_scale ||
        original.actor.reward_version != 5 || original.actor.update != 336)
        throw std::runtime_error("Append migration requires preserved336 schema2/reward5");
    AppendedHoldState out;
    out.ppo=original;
    out.ppo.actor.parameters=append_zero_input_column(original.actor.parameters);
    out.ppo.critic=append_zero_input_column(original.critic);
    for (auto* adam : {&out.ppo.actor_adam,&out.ppo.critic_adam}) {
        adam->first=append_zero_input_column(adam->first);
        adam->second=append_zero_input_column(adam->second);
    }
    out.ppo.actor.observation_schema=motor_observation::appended_hold;
    out.ppo.actor.reward_version=9;
    return out;
}
inline PpoAdamState read_append_adam(checkpoint::Reader& in,size_t count) {
    PpoAdamState out;
    out.config=in.value<nn::AdamConfig>();out.steps=in.value<int>();
    out.first=in.array<float>(count);out.second=in.array<float>(count);
    const auto& c=out.config;
    if (out.first.size()!=count || out.second.size()!=count || out.steps<0 ||
        !std::isfinite(c.lr) || c.lr<=0 || c.beta1!=.9f || c.beta2!=.999f ||
        c.eps!=.00001f || c.max_grad_norm!=.5f)
        throw std::runtime_error("Invalid append Adam metadata");
    for(size_t i=0;i<count;++i)
        if(!std::isfinite(out.first[i]) || !std::isfinite(out.second[i]) || out.second[i]<0)
            throw std::runtime_error("Invalid append Adam moments");
    return out;
}
inline AppendedHoldState read_appended_hold(const std::vector<uint8_t>& bytes,uint64_t protocol) {
    checkpoint::Reader in{bytes};AppendedHoldState out;auto& s=out.ppo;
    if(in.value<uint32_t>()!=0x35415050 || in.value<uint32_t>()!=5)
        throw std::runtime_error("Wrong appended checkpoint schema");
    s.source_protocol=in.value<uint64_t>();s.actor.seed=in.value<uint32_t>();
    s.actor.update=in.value<int>();
    if(s.source_protocol!=protocol || s.actor.seed!=1247037243 || s.actor.update<336)
        throw std::runtime_error("Appended identity mismatch");
    for(uint32_t expected : {6u,6u,3u,9u,4u,41u,4u,2532u,2433u})
        if(in.value<uint32_t>()!=expected)throw std::runtime_error("Appended shape/task mismatch");
    s.actor.observation_schema=4;s.actor.reward_version=9;
    s.actor.correlation=in.value<float>();
    if(s.actor.correlation!=.8f)throw std::runtime_error("Appended stochastic mismatch");
    s.actor.log_sigma=in.array<float>(4);
    s.actor.parameters=in.array<float>(2532);s.critic=in.array<float>(2433);
    if(s.actor.log_sigma!=std::vector<float>(4,-1.f) || s.actor.parameters.size()!=2532 || s.critic.size()!=2433)
        throw std::runtime_error("Appended array shape mismatch");
    for(const auto* v:{&s.actor.parameters,&s.critic})for(float x:*v)
        if(!std::isfinite(x))throw std::runtime_error("Nonfinite appended network");
    s.actor_adam=read_append_adam(in,2532);s.critic_adam=read_append_adam(in,2433);
    out.actor_column_age=in.value<int>();out.critic_column_age=in.value<int>();
    if(out.actor_column_age<0 || out.critic_column_age<0 ||
        out.actor_column_age>s.actor_adam.steps || out.critic_column_age>s.critic_adam.steps)
        throw std::runtime_error("Invalid appended Adam age");
    s.actor.actor_steps=s.actor_adam.steps;s.actor.critic_steps=s.critic_adam.steps;
    in.finish();return out;
}
inline std::vector<uint8_t> write_appended_hold(const AppendedHoldState& state,uint64_t protocol) {
    checkpoint::Writer out;const auto& s=state.ppo;
    out.value(uint32_t{0x35415050});out.value(uint32_t{5});out.value(protocol);
    out.value(s.actor.seed);out.value(s.actor.update);
    for(uint32_t v:{6u,6u,3u,9u,4u,41u,4u,2532u,2433u})out.value(v);
    out.value(s.actor.correlation);out.array(s.actor.log_sigma);
    out.array(s.actor.parameters);out.array(s.critic);
    for(const auto* adam:{&s.actor_adam,&s.critic_adam}){
        auto bytes=adam_payload(*adam);out.data.insert(out.data.end(),bytes.begin(),bytes.end());
    }
    out.value(state.actor_column_age);out.value(state.critic_column_age);
    (void)read_appended_hold(out.data,protocol);return out.data;
}
} // namespace cudarl::rl
#endif
