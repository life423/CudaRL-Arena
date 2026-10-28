#ifndef CUDARL_PROGRESS_SELECTOR_PPO_H
#define CUDARL_PROGRESS_SELECTOR_PPO_H
#include "swimmer_ppo.h"
#include "progress_selector_policy.h"
namespace cudarl::rl {
namespace pg=cudarl::progress_selector;
struct ProgressInstrumentation {
    void (*phase)(void*,int)=nullptr;
    void* context=nullptr;
};
// Isolated experimental orchestration; the categorical PPO, GAE, gradient and
// Adam kernels are the same as SelectorTrainer. Existing save formats stay intact.
struct ProgressTrainer {
    static constexpr int horizon=32,epochs=4,minibatches=8,inputs=pg::kObs;
    int n,batch,mb,update=0;uint32_t seed;float maximum;
    SelectorPolicy baseline;SwimmerStream stream;pg::Batch env;
    std::vector<pg::RecoveryStart> recovery;
    nn::GpuMlp actor,critic,motor;nn::GpuAdam actor_opt,critic_opt;
    device_ptr<float> obs,old_logp,values,rewards,done,truncated,final_obs,final_values,last_values,advantages,returns,
        mb_obs,mb_logp,mb_adv,mb_ret,dlogits,dvalues,adv_stats,row_stats,stats,feature_rows;
    device_ptr<int> actions,mb_actions;
    static nn::AdamConfig optimizer() {auto out=DeepPondTrainer::optimizer();out.lr=.0003f;return out;}
    ProgressTrainer(const SelectorPolicy& source,int environments,uint32_t key,float max_current=.01f):n(environments),batch(n*horizon),mb(batch/minibatches),seed(key),maximum(max_current),baseline(source),
        env(n,seed,source.decision_steps,maximum),actor(progress_selector_spec(),3*std::max({n,mb,64}),seed),
        critic(swimmer_specification(1,inputs),std::max(batch,64),seed+1),motor(swimmer_specification(4,33),std::max(n,64),7),
        actor_opt(actor.num_params(),optimizer()),critic_opt(critic.num_params(),optimizer()) {
        if(n<16 || n>32768 || batch%minibatches) throw std::invalid_argument("Invalid progress training batch");
        actor.set_params(transfer_progress_actor(baseline.selector));motor.set_params(baseline.motor);
        critic.set_params(nn::orthogonal_params(critic.spec(),seed+1,{std::sqrt(2.f),std::sqrt(2.f),1.f}));
        obs.allocate(static_cast<size_t>(batch)*inputs);final_obs.allocate(obs.size());CUDA_CHECK_THROW(cudaMemset(final_obs.get(),0,final_obs.size_bytes()));
        actions.allocate(batch);mb_actions.allocate(mb);
        for(auto* p:{&old_logp,&values,&rewards,&done,&truncated,&final_values,&advantages,&returns}) p->allocate(batch);
        last_values.allocate(n);mb_obs.allocate(static_cast<size_t>(mb)*inputs);
        for(auto* p:{&mb_logp,&mb_adv,&mb_ret,&dvalues}) p->allocate(mb);
        dlogits.allocate(static_cast<size_t>(mb)*3);adv_stats.allocate(2);row_stats.allocate(static_cast<size_t>(mb)*kPpoStats);stats.allocate(kPpoStats);
        feature_rows.allocate(static_cast<size_t>(3*std::max({n,mb,64}))*pg::kFeatures);CUDA_CHECK_THROW(cudaDeviceSynchronize());
    }
    void score(const float* in,int count) {pg::make_features(in,feature_rows.get(),count,stream.value);actor.forward(feature_rows.get(),count*3,stream.value);}
    void train_update(ProgressInstrumentation* instrument=nullptr) {
        const auto mark=[&](int phase) {if(instrument && instrument->phase) instrument->phase(instrument->context,phase);};
        mark(0);
        const auto s=stream.value;CUDA_CHECK_THROW(cudaMemsetAsync(stats.get(),0,stats.size_bytes(),s));
        for(int t=0;t<horizon;++t) {
            const int row=t*n;swimmer_copy(obs.get()+row*inputs,env.obs(),static_cast<size_t>(n)*inputs,s);
            score(obs.get()+row*inputs,n);sample_actions(actor.output(),n,3,seed,static_cast<uint32_t>(update*horizon+t),false,actions.get()+row,old_logp.get()+row,s);
            critic.forward(obs.get()+row*inputs,n,s);swimmer_copy(values.get()+row,critic.output(),n,s);
            sw::Outputs out{rewards.get()+row,done.get()+row,truncated.get()+row,final_obs.get()+row*inputs};env.begin(actions.get()+row,out,s);
            for(int tick=0;tick<baseline.decision_steps;++tick) {motor.forward(env.motor_obs(s),n,s);env.step(motor.output(),out,s);}
        }
        mark(1);
        critic.forward(env.obs(),n,s);swimmer_copy(last_values.get(),critic.output(),n,s);critic.forward(final_obs.get(),batch,s);swimmer_copy(final_values.get(),critic.output(),batch,s);
        compute_gae(horizon,n,rewards.get(),values.get(),done.get(),truncated.get(),final_values.get(),last_values.get(),.99f,.95f,advantages.get(),returns.get(),s);
        mark(2);
        for(int epoch=0;epoch<epochs;++epoch) {
            const auto perm=make_permutation(static_cast<uint32_t>(batch),rng::bits(seed,static_cast<uint32_t>(update),static_cast<uint32_t>(epoch)));
            for(int m=0;m<minibatches;++m) {
                gather_minibatch(perm,m*mb,mb,inputs,obs.get(),actions.get(),old_logp.get(),advantages.get(),returns.get(),mb_obs.get(),mb_actions.get(),mb_logp.get(),mb_adv.get(),mb_ret.get(),s);
                mean_std(mb_adv.get(),mb,adv_stats.get(),s);score(mb_obs.get(),mb);critic.forward(mb_obs.get(),mb,s);
                ppo_loss_grad(actor.output(),critic.output(),mb,3,mb_actions.get(),mb_logp.get(),mb_adv.get(),mb_ret.get(),adv_stats.get(),{.2f,.01f,.5f},dlogits.get(),dvalues.get(),row_stats.get(),stats.get(),s);
                actor.zero_grad(s);actor.backward(dlogits.get(),mb*3,nullptr,s);actor_opt.step(actor.params(),actor.grads(),s);
                critic.zero_grad(s);critic.backward(dvalues.get(),mb,nullptr,s);critic_opt.step(critic.params(),critic.grads(),s);
            }
        }
        mark(3);
        CUDA_CHECK_THROW(cudaStreamSynchronize(s));mark(4);++update;
    }
    ProgressPolicy policy() const {return {update,baseline,actor.get_params()};}
    void set_recovery_starts(const std::vector<pg::RecoveryStart>& starts,bool reset_worlds=true) {env.set_recovery_starts(starts,reset_worlds);recovery=starts;}
    std::vector<double> evaluate(uint32_t key=0xA7D13101u) {
        constexpr int count=64;std::vector<double> results;
        const deep::V current[3]={{},{.01f,0,0},{-.007071067579f,0,-.007071067579f}};
        for(auto water:current) {
            pg::Batch trials(count,key,baseline.decision_steps,maximum,false,water);device_ptr<int> selected(count);device_ptr<float> logp(count);
            CUDA_CHECK_THROW(cudaDeviceSynchronize());const auto s=stream.value;
            for(int tick=0;tick<1200;tick+=baseline.decision_steps) {
                score(trials.obs(),count);sample_actions(actor.output(),count,3,0,0,true,selected.get(),logp.get(),s);trials.begin(selected.get(),{},s);
                for(int j=0;j<baseline.decision_steps;++j) {motor.forward(trials.motor_obs(s),count,s);trials.step(motor.output(),{},s,false);}
            }
            std::vector<deep_pond::State> states(count);CUDA_CHECK_THROW(cudaMemcpyAsync(states.data(),trials.states(),count*sizeof(deep_pond::State),cudaMemcpyDeviceToHost,s));CUDA_CHECK_THROW(cudaStreamSynchronize(s));
            int foods=0;for(const auto& state:states) foods+=state.eaten;results.push_back(double(foods)/count);
        }
        return results;
    }
    std::vector<uint8_t> snapshot() const {
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));checkpoint::Writer out;out.value(uint32_t{0x50535431});out.value(uint32_t{recovery.empty()?1u:2u});out.value(n);out.value(seed);out.value(maximum);out.value(update);baseline.write(out);
        out.array(actor.get_params());out.array(critic.get_params());actor_opt.save(out);critic_opt.save(out);env.save(out);if(!recovery.empty()) out.array(recovery);return out.data;
    }
    void restore(const std::vector<uint8_t>& bytes) {
        checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=0x50535431u) throw std::runtime_error("Unsupported progress training save");
        const auto version=in.value<uint32_t>();if(version!=1 && version!=2) throw std::runtime_error("Unsupported progress training version");
        if(in.value<int>()!=n || in.value<uint32_t>()!=seed || in.value<float>()!=maximum) throw std::runtime_error("Progress training configuration mismatch");
        update=in.value<int>();const auto parent=SelectorPolicy::read(in);
        if(update<0 || parent.motor!=baseline.motor || parent.selector!=baseline.selector || parent.decision_steps!=baseline.decision_steps) throw std::runtime_error("Frozen baseline mismatch");
        actor.set_params(in.array<float>(actor.num_params()));critic.set_params(in.array<float>(critic.num_params()));actor_opt.load(in);critic_opt.load(in);env.load(in);
        set_recovery_starts(version==2?in.array<pg::RecoveryStart>(4096):std::vector<pg::RecoveryStart>{},false);in.finish();
        if(actor_opt.steps()!=update*epochs*minibatches || critic_opt.steps()!=actor_opt.steps()) throw std::runtime_error("Progress optimizer counter mismatch");
    }
};
}
#endif
