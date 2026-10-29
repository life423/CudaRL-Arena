#ifndef CUDARL_SHELTER_PPO_H
#define CUDARL_SHELTER_PPO_H
#include "swimmer_ppo.h"
#include "envs/swimmer/shelter_batch.h"

namespace cudarl::rl {
inline nn::MlpSpec shelter_selector_spec() {return swimmer_specification(1,shelter::kFeatures);}
inline std::vector<float> transfer_shelter_actor(const std::vector<float>& source) {
    const auto old=progress_selector_spec(),next=shelter_selector_spec();
    if(source.size()!=static_cast<size_t>(old.num_params())) throw std::invalid_argument("Shelter selector requires the inherited shared food scorer");
    for(float value:source) if(!std::isfinite(value)) throw std::invalid_argument("Non-finite source goal scorer");
    std::vector<float> result(static_cast<size_t>(next.num_params()),0.f);
    for(int l=0;l<old.num_layers();++l) {
        for(int r=0;r<old.out(l);++r) for(int c=0;c<old.in(l);++c)
            result[next.weight_offset(l)+r*next.in(l)+c]=source[old.weight_offset(l)+r*old.in(l)+c];
        for(int r=0;r<old.out(l);++r) result[next.bias_offset(l)+r]=source[old.bias_offset(l)+r];
    }
    return result;
}
// PPO remains categorical over a shared per-goal scorer. This extends the first
// controller; its compatible 59-input body controller is frozen for each rollout.
struct ShelterTrainer {
    static constexpr int horizon=32,epochs=4,minibatches=8,inputs=shelter::kObs,choices=shelter::kChoices;
    int n,batch,mb,update=0;uint32_t seed;arch::Context context;
    std::vector<float> source_actor,frozen_motor;
    shelter_motor::ActuatorScale action_scales{};
    SwimmerStream stream;shelter::Batch env;
    nn::GpuMlp actor,critic,motor;nn::GpuAdam actor_opt,critic_opt;
    device_ptr<float> obs,old_logp,values,rewards,done,truncated,final_obs,final_values,last_values,advantages,returns,
        mb_obs,mb_logp,mb_adv,mb_ret,dlogits,dvalues,adv_stats,row_stats,stats,feature_rows,masked_logits,motor_means;
    device_ptr<int> actions,mb_actions;
    static nn::AdamConfig optimizer() {auto out=DeepPondTrainer::optimizer();out.lr=.0003f;return out;}
    ShelterTrainer(const std::vector<float>& shared34,const std::vector<float>& frozen59,int environments,uint32_t key,arch::Context current,
                   shelter_motor::ActuatorScale scales={}):
        n(environments),batch(n*horizon),mb(batch/minibatches),seed(key),context(current),source_actor(shared34),frozen_motor(frozen59),action_scales(scales),
        env(n,seed,context),actor(shelter_selector_spec(),choices*std::max({n,mb,64}),seed),
        critic(swimmer_specification(1,inputs),std::max(batch,64),seed+1),
        motor(swimmer_specification(4,shelter_motor::kObs),std::max(n,64),7),
        actor_opt(actor.num_params(),optimizer()),critic_opt(critic.num_params(),optimizer()) {
        if(n<16 || n>2048 || batch%minibatches) throw std::invalid_argument("Invalid shelter training population");
        if(frozen_motor.size()!=static_cast<size_t>(motor.num_params())) throw std::invalid_argument("Shelter PPO requires a compatible 59-input swimming controller");
        for(float v:frozen_motor) if(!std::isfinite(v)) throw std::invalid_argument("Non-finite frozen swimming actor");
        for(float v:action_scales.value) if(!std::isfinite(v) || v<1.f || v>1048576.f) throw std::invalid_argument("Invalid frozen motor normalization");
        actor.set_params(transfer_shelter_actor(source_actor));motor.set_params(frozen_motor);
        critic.set_params(nn::orthogonal_params(critic.spec(),seed+1,{std::sqrt(2.f),std::sqrt(2.f),1.f}));
        obs.allocate(static_cast<size_t>(batch)*inputs);final_obs.allocate(obs.size());CUDA_CHECK_THROW(cudaMemset(final_obs.get(),0,final_obs.size_bytes()));
        actions.allocate(batch);mb_actions.allocate(mb);
        for(auto* p:{&old_logp,&values,&rewards,&done,&truncated,&final_values,&advantages,&returns}) p->allocate(batch);
        last_values.allocate(n);mb_obs.allocate(static_cast<size_t>(mb)*inputs);
        for(auto* p:{&mb_logp,&mb_adv,&mb_ret,&dvalues}) p->allocate(mb);
        dlogits.allocate(static_cast<size_t>(mb)*choices);adv_stats.allocate(2);row_stats.allocate(static_cast<size_t>(mb)*kPpoStats);stats.allocate(kPpoStats);
        feature_rows.allocate(static_cast<size_t>(choices*std::max({n,mb,64}))*shelter::kFeatures);
        masked_logits.allocate(static_cast<size_t>(choices*std::max({n,mb,64})));CUDA_CHECK_THROW(cudaDeviceSynchronize());
        motor_means.allocate(static_cast<size_t>(n)*shelter_motor::kAct);
    }
    const float* score(const float* input,int count) {
        shelter::make_features(input,feature_rows.get(),count,stream.value);actor.forward(feature_rows.get(),count*choices,stream.value);
        swimmer_copy(masked_logits.get(),actor.output(),static_cast<size_t>(count)*choices,stream.value);
        shelter::mask_logits(input,masked_logits.get(),count,stream.value);return masked_logits.get();
    }
    void train_update() {
        const auto s=stream.value;CUDA_CHECK_THROW(cudaMemsetAsync(stats.get(),0,stats.size_bytes(),s));
        for(int t=0;t<horizon;++t) {
            const int row=t*n;swimmer_copy(obs.get()+row*inputs,env.obs(),static_cast<size_t>(n)*inputs,s);
            sample_actions(score(obs.get()+row*inputs,n),n,choices,seed,static_cast<uint32_t>(update*horizon+t),false,actions.get()+row,old_logp.get()+row,s);
            critic.forward(obs.get()+row*inputs,n,s);swimmer_copy(values.get()+row,critic.output(),n,s);
            swimmer::Outputs out{rewards.get()+row,done.get()+row,truncated.get()+row,final_obs.get()+row*inputs};env.begin(actions.get()+row,out,s);
            for(int tick=0;tick<shelter::kPeriod;++tick) {
                const auto* motor_input=env.motor_obs(s);motor.forward(motor_input,n,s);
                shelter_motor::policy_means(motor.output(),motor_input,motor_means.get(),n,action_scales,s);
                env.step(motor_means.get(),out,s);
            }
        }
        critic.forward(env.obs(),n,s);swimmer_copy(last_values.get(),critic.output(),n,s);
        critic.forward(final_obs.get(),batch,s);swimmer_copy(final_values.get(),critic.output(),batch,s);
        compute_gae(horizon,n,rewards.get(),values.get(),done.get(),truncated.get(),final_values.get(),last_values.get(),.99f,.95f,advantages.get(),returns.get(),s);
        for(int epoch=0;epoch<epochs;++epoch) {
            const auto perm=make_permutation(static_cast<uint32_t>(batch),rng::bits(seed,static_cast<uint32_t>(update),static_cast<uint32_t>(epoch)));
            for(int m=0;m<minibatches;++m) {
                gather_minibatch(perm,m*mb,mb,inputs,obs.get(),actions.get(),old_logp.get(),advantages.get(),returns.get(),mb_obs.get(),mb_actions.get(),mb_logp.get(),mb_adv.get(),mb_ret.get(),s);
                mean_std(mb_adv.get(),mb,adv_stats.get(),s);const auto logits=score(mb_obs.get(),mb);critic.forward(mb_obs.get(),mb,s);
                ppo_loss_grad(logits,critic.output(),mb,choices,mb_actions.get(),mb_logp.get(),mb_adv.get(),mb_ret.get(),adv_stats.get(),{.2f,.01f,.5f},dlogits.get(),dvalues.get(),row_stats.get(),stats.get(),s);
                actor.zero_grad(s);actor.backward(dlogits.get(),mb*choices,nullptr,s);actor_opt.step(actor.params(),actor.grads(),s);
                critic.zero_grad(s);critic.backward(dvalues.get(),mb,nullptr,s);critic_opt.step(critic.params(),critic.grads(),s);
            }
        }
        CUDA_CHECK_THROW(cudaStreamSynchronize(s));++update;
    }
    std::vector<float> policy() const {return actor.get_params();}
    std::vector<uint8_t> snapshot() const {
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));checkpoint::Writer out;
        out.value(uint32_t{0x534C5452u});out.value(uint32_t{1});out.value(uint32_t{sizeof(shelter::World)});
        out.value(n);out.value(seed);out.value(update);out.value(context);
        const std::vector<float> protocol={float(inputs),float(shelter::kFeatures),float(choices),float(shelter_motor::kObs),float(horizon),float(epochs),float(minibatches),float(shelter::kPeriod),.99f,.95f,.2f,.01f,.5f,.0003f,action_scales.value[0],action_scales.value[1],action_scales.value[2],action_scales.value[3]};out.array(protocol);
        out.array(source_actor);out.array(frozen_motor);out.array(actor.get_params());out.array(critic.get_params());
        actor_opt.save(out);critic_opt.save(out);env.save(out);return out.data;
    }
    void restore(const std::vector<uint8_t>& bytes) {
        checkpoint::Reader in{bytes};
        if(in.value<uint32_t>()!=0x534C5452u || in.value<uint32_t>()!=1 || in.value<uint32_t>()!=sizeof(shelter::World))
            throw std::runtime_error("Unsupported shelter PPO checkpoint");
        if(in.value<int>()!=n || in.value<uint32_t>()!=seed) throw std::runtime_error("Shelter PPO configuration mismatch");
        const int saved_update=in.value<int>();const auto saved_context=in.value<arch::Context>();
        if(saved_update<0 || saved_context.basin.layout!=context.basin.layout || saved_context.version!=context.version ||
           saved_context.current.x!=context.current.x || saved_context.current.y!=context.current.y || saved_context.current.z!=context.current.z)
            throw std::runtime_error("Shelter PPO context mismatch");
        const std::vector<float> protocol={float(inputs),float(shelter::kFeatures),float(choices),float(shelter_motor::kObs),float(horizon),float(epochs),float(minibatches),float(shelter::kPeriod),.99f,.95f,.2f,.01f,.5f,.0003f,action_scales.value[0],action_scales.value[1],action_scales.value[2],action_scales.value[3]};
        if(in.array<float>(protocol.size())!=protocol || in.array<float>(source_actor.size())!=source_actor || in.array<float>(frozen_motor.size())!=frozen_motor)
            throw std::runtime_error("Shelter PPO protocol or frozen source changed");
        const auto saved_actor=in.array<float>(actor.num_params()),saved_critic=in.array<float>(critic.num_params());
        if(saved_actor.size()!=static_cast<size_t>(actor.num_params()) || saved_critic.size()!=static_cast<size_t>(critic.num_params()))
            throw std::runtime_error("Shelter PPO network dimensions changed");
        for(const auto* parameters:{&saved_actor,&saved_critic}) for(float v:*parameters) if(!std::isfinite(v)) throw std::runtime_error("Non-finite shelter PPO parameters");
        actor.set_params(saved_actor);critic.set_params(saved_critic);actor_opt.load(in);critic_opt.load(in);env.load(in);in.finish();
        if(actor_opt.steps()!=saved_update*epochs*minibatches || critic_opt.steps()!=actor_opt.steps())
            throw std::runtime_error("Shelter PPO optimizer counter mismatch");
        update=saved_update;
    }
};
}
#endif
