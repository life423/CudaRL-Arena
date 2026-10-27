#ifndef CUDARL_RL_POND_SELECTOR_PPO_H
#define CUDARL_RL_POND_SELECTOR_PPO_H
#include "swimmer_ppo.h"
#include "pond_selector_policy.h"
#include "envs/swimmer/pond_selector.h"
namespace cudarl::rl {

struct SelectorConfig {
    int environments=1024,decision_steps=20;uint32_t seed=11;bool recovery=false;
    void write(checkpoint::Writer& out) const {out.value(environments);out.value(decision_steps);out.value(seed);out.value(recovery);}
    static SelectorConfig read(checkpoint::Reader& in) {SelectorConfig c;c.environments=in.value<int>();c.decision_steps=in.value<int>();c.seed=in.value<uint32_t>();c.recovery=in.value<bool>();return c;}
};
struct SelectorTrainer {
    static constexpr int horizon=32,num_epochs=4,num_minibatches=8;
    SelectorConfig cfg;int update=0,n,batch,mb,motor_update;
    SwimmerStream stream;ps::Batch env;nn::GpuMlp actor,critic,motor;
    nn::GpuAdam actor_opt,critic_opt;SelectorPolicy best;
    device_ptr<float> obs,old_logp,values,rewards,done,truncated,final_obs,final_values,last_values,advantages,returns,
        mb_obs,mb_logp,mb_adv,mb_ret,dlogits,dvalues,adv_stats,row_stats,stats,feature_rows;
    device_ptr<int> actions,mb_actions;
    std::vector<float> frozen_motor;
    static nn::AdamConfig optimizer() {auto a=DeepPondTrainer::optimizer();a.lr=.0003f;return a;}
    SelectorTrainer(SelectorConfig config,const std::vector<float>& motor_weights,int source_update):cfg(config),n(cfg.environments),batch(n*horizon),mb(batch/num_minibatches),motor_update(source_update),
        env(n,cfg.seed,cfg.decision_steps,cfg.recovery),actor(selector_spec(),3*std::max({n,mb,64}),cfg.seed),critic(swimmer_specification(1,ps::kObs),batch,cfg.seed+1),
        motor(swimmer_specification(4,33),std::max(n,64),7),actor_opt(actor.num_params(),optimizer()),critic_opt(critic.num_params(),optimizer()),frozen_motor(motor_weights) {
        if(n<1 || n>32768 || batch%num_minibatches!=0) throw std::invalid_argument("Invalid selector population");
        actor.set_params(nn::orthogonal_params(actor.spec(),cfg.seed,{std::sqrt(2.f),std::sqrt(2.f),.01f}));
        critic.set_params(nn::orthogonal_params(critic.spec(),cfg.seed+1,{std::sqrt(2.f),std::sqrt(2.f),1.f}));motor.set_params(motor_weights);
        obs.allocate(static_cast<size_t>(batch)*36);final_obs.allocate(obs.size());CUDA_CHECK_THROW(cudaMemset(final_obs.get(),0,final_obs.size_bytes()));actions.allocate(batch);mb_actions.allocate(mb);
        for(auto* p:{&old_logp,&values,&rewards,&done,&truncated,&final_values,&advantages,&returns}) p->allocate(batch);
        last_values.allocate(n);mb_obs.allocate(static_cast<size_t>(mb)*36);
        for(auto* p:{&mb_logp,&mb_adv,&mb_ret,&dvalues}) p->allocate(mb);dlogits.allocate(static_cast<size_t>(mb)*3);adv_stats.allocate(2);row_stats.allocate(static_cast<size_t>(mb)*kPpoStats);stats.allocate(kPpoStats);feature_rows.allocate(static_cast<size_t>(3*std::max({n,mb,64}))*ps::kFeatures);
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
    }
    void score(const float* input,int count) {ps::make_features(input,feature_rows.get(),count,stream.value);actor.forward(feature_rows.get(),count*3,stream.value);}
    void train_update() {
        const auto s=stream.value;CUDA_CHECK_THROW(cudaMemsetAsync(stats.get(),0,stats.size_bytes(),s));
        for(int t=0;t<horizon;++t) {
            const int row=t*n;swimmer_copy(obs.get()+row*36,env.obs(),static_cast<size_t>(n)*36,s);
            score(obs.get()+row*36,n);sample_actions(actor.output(),n,3,cfg.seed,static_cast<uint32_t>(update*horizon+t),false,actions.get()+row,old_logp.get()+row,s);
            critic.forward(obs.get()+row*36,n,s);swimmer_copy(values.get()+row,critic.output(),n,s);
            sw::Outputs out{rewards.get()+row,done.get()+row,truncated.get()+row,final_obs.get()+row*36};env.begin(actions.get()+row,out,s);
            for(int tick=0;tick<cfg.decision_steps;++tick) {motor.forward(env.motor_obs(s),n,s);env.step(motor.output(),out,s);}
        }
        critic.forward(env.obs(),n,s);swimmer_copy(last_values.get(),critic.output(),n,s);critic.forward(final_obs.get(),batch,s);swimmer_copy(final_values.get(),critic.output(),batch,s);
        compute_gae(horizon,n,rewards.get(),values.get(),done.get(),truncated.get(),final_values.get(),last_values.get(),.99f,.95f,advantages.get(),returns.get(),s);
        for(int epoch=0;epoch<num_epochs;++epoch) {
            const auto perm=make_permutation(static_cast<uint32_t>(batch),rng::bits(cfg.seed,static_cast<uint32_t>(update),static_cast<uint32_t>(epoch)));
            for(int m=0;m<num_minibatches;++m) {
                gather_minibatch(perm,m*mb,mb,36,obs.get(),actions.get(),old_logp.get(),advantages.get(),returns.get(),mb_obs.get(),mb_actions.get(),mb_logp.get(),mb_adv.get(),mb_ret.get(),s);
                mean_std(mb_adv.get(),mb,adv_stats.get(),s);score(mb_obs.get(),mb);critic.forward(mb_obs.get(),mb,s);
                ppo_loss_grad(actor.output(),critic.output(),mb,3,mb_actions.get(),mb_logp.get(),mb_adv.get(),mb_ret.get(),adv_stats.get(),{.2f,.01f,.5f},dlogits.get(),dvalues.get(),row_stats.get(),stats.get(),s);
                actor.zero_grad(s);actor.backward(dlogits.get(),mb*3,nullptr,s);actor_opt.step(actor.params(),actor.grads(),s);
                critic.zero_grad(s);critic.backward(dvalues.get(),mb,nullptr,s);critic_opt.step(critic.params(),critic.grads(),s);
            }
        }
        CUDA_CHECK_THROW(cudaStreamSynchronize(s));++update;
    }
    int evaluate(uint32_t seed=0x6A91B23Du,bool select_best=true) {
        constexpr int count=64;ps::Batch trials(count,seed,cfg.decision_steps);device_ptr<int> selected(count);device_ptr<float> logp(count);
        CUDA_CHECK_THROW(cudaDeviceSynchronize());const auto s=stream.value;
        for(int decision=0;decision<1200/cfg.decision_steps;++decision) {
            score(trials.obs(),count);sample_actions(actor.output(),count,3,0,0,true,selected.get(),logp.get(),s);trials.begin(selected.get(),{},s);
            for(int tick=0;tick<cfg.decision_steps;++tick) {motor.forward(trials.motor_obs(s),count,s);trials.step(motor.output(),{},s,false);}
        }
        std::vector<deep_pond::State> states(count);CUDA_CHECK_THROW(cudaMemcpyAsync(states.data(),trials.states(),states.size()*sizeof(deep_pond::State),cudaMemcpyDeviceToHost,s));CUDA_CHECK_THROW(cudaStreamSynchronize(s));
        int foods=0;for(const auto& state:states) foods+=state.eaten;
        if(select_best && seed==0x6A91B23Du && foods>best.foods) best={update,motor_update,cfg.decision_steps,foods,count,actor.get_params(),frozen_motor};return foods;
    }
    std::vector<uint8_t> snapshot() const {
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));checkpoint::Writer out;out.value(uint32_t{0x53454C31});out.value(uint32_t{3});cfg.write(out);out.value(update);out.value(motor_update);
        const std::vector<float> protocol={36,32,3,float(horizon),float(num_epochs),float(num_minibatches),.99f,.95f,.2f,.01f,.5f,.0003f,32,1};out.array(protocol);
        out.array(frozen_motor);out.array(actor.get_params());out.array(critic.get_params());actor_opt.save(out);critic_opt.save(out);env.save(out);
        out.value(best.foods>=0);if(best.foods>=0) best.write(out);return out.data;
    }
    void restore(const std::vector<uint8_t>& bytes) {
        checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=0x53454C31u) throw std::runtime_error("Unsupported selector checkpoint");
        const auto version=in.value<uint32_t>();if(version!=2 && version!=3) throw std::runtime_error("Unsupported selector checkpoint");
        SelectorConfig config;config.environments=in.value<int>();config.decision_steps=in.value<int>();config.seed=in.value<uint32_t>();if(version==3) config.recovery=in.value<bool>();
        if(config.environments!=n || config.seed!=cfg.seed || config.decision_steps!=cfg.decision_steps || config.recovery!=cfg.recovery) throw std::runtime_error("Selector configuration mismatch");
        update=in.value<int>();if(in.value<int>()!=motor_update || update<0) throw std::runtime_error("Selector motor source mismatch");
        const std::vector<float> expected={36,32,3,float(horizon),float(num_epochs),float(num_minibatches),.99f,.95f,.2f,.01f,.5f,.0003f,32,1};
        if(in.array<float>(expected.size())!=expected || in.array<float>(frozen_motor.size())!=frozen_motor) throw std::runtime_error("Selector protocol or frozen motor changed");
        actor.set_params(in.array<float>(actor.num_params()));critic.set_params(in.array<float>(critic.num_params()));actor_opt.load(in);critic_opt.load(in);env.load(in);
        best={};if(in.value<bool>()) best=SelectorPolicy::read(in);in.finish();
        if(actor_opt.steps()!=update*num_epochs*num_minibatches || critic_opt.steps()!=actor_opt.steps()) throw std::runtime_error("Selector optimizer counter mismatch");
    }
};
}
#endif
