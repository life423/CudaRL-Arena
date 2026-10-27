#ifndef CUDARL_RL_SWIMMER_PPO_H
#define CUDARL_RL_SWIMMER_PPO_H
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#include "envs/swimmer/swimmer.h"
#include "envs/swimmer/foraging.h"
#include "nn/adam.h"
#include "nn/mlp_gpu.h"
#include "rl/gaussian_kernels.h"
#include "rl/ppo_kernels.h"
// Shared, unchanged PPO update path used by the CLI experiment and the live
// background session. No Godot dependency and no saved policy initialization.
namespace cudarl::rl {
namespace sw=cudarl::swimmer;
using Clock=std::chrono::steady_clock;
constexpr int hidden=32,rollout=64,epochs=4,minibatches=8;
inline nn::MlpSpec swimmer_specification(int outputs,int inputs=sw::kObs) {
    return nn::MlpSpec({inputs,hidden,hidden,outputs},
        {nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity});
}
struct SwimmerStream {
    cudaStream_t value=nullptr;
    SwimmerStream() { CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking)); }
    ~SwimmerStream() { if(value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
};
inline void swimmer_copy(float* dst,const float* src,size_t count,cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaMemcpyAsync(dst,src,count*sizeof(float),cudaMemcpyDeviceToDevice,stream));
}
struct SwimmerConfig {
    int num_envs=1024;
    uint32_t seed=1;
    int total_updates=300;
    int evaluation_every=0; // Optional fixed, separate comparison trials for the live graph.
    sw::Body body=sw::body_preset(6);
};
struct SwimmerStats {
    int update=0;
    long long env_steps=0;
    double update_ms=0;
    int evaluation_update=-1;
    int evaluation_successes=0;
    int evaluation_count=0;
    int evaluation_foods=0;
    double evaluation_seconds=10;
};
struct SingleFoodTask {
    using State=sw::State;using Batch=sw::Batch;
    static constexpr int inputs=sw::kObs,evaluation_steps=200;
    static constexpr float entropy=0.f;
    static int collected(const State& state) {return sw::distance(state)<=sw::kFoodRadius?1:0;}
};
struct ThreeFoodTask {
    using State=foraging::State;using Batch=foraging::Batch;
    static constexpr int inputs=foraging::kObs,evaluation_steps=foraging::kEpisodeSteps;
    static constexpr float entropy=0.005f;
    static int collected(const State& state) {return state.eaten;}
};
template<class Task>
struct BasicSwimmerTrainer {
    using Config=SwimmerConfig;
    using Stats=SwimmerStats;
    static constexpr int kInputs=Task::inputs;
    explicit BasicSwimmerTrainer(const Config& cfg):BasicSwimmerTrainer(cfg.num_envs,cfg.seed,cfg.body) {
        evaluation_every=cfg.evaluation_every;
        evaluation_body=cfg.body;
        if(evaluation_every>0) evaluate();
    }
    std::vector<float> actor_params() const { return actor.get_params(); }
    static std::vector<float> initial_actor_params(uint32_t key) {
        const auto spec=swimmer_specification(sw::kAct,kInputs);
        return nn::orthogonal_params(spec,key,{std::sqrt(2.f),std::sqrt(2.f),0.01f});
    }
    Stats update() {
        const double elapsed=update_seconds();
        if(evaluation_every>0 && update_id % evaluation_every==0) evaluate();
        return {update_id,static_cast<long long>(update_id)*batch,elapsed*1000.0,
                evaluation_update,evaluation_successes,evaluation_every>0?evaluation_count:0,
                evaluation_foods,evaluation_steps*static_cast<double>(sw::kDt)};
    }
    static constexpr int evaluation_count=64,evaluation_steps=Task::evaluation_steps;
    int evaluation_every=0,evaluation_update=-1,evaluation_successes=0,evaluation_foods=0;
    sw::Body evaluation_body{};
    // Deterministic mean actions, fixed starts and food, no exploration and no
    // gradients. Separate environments never feed these trials into PPO.
    void evaluate(uint32_t evaluation_seed=0x51A7E123u) {
        typename Task::Batch trials(evaluation_count,evaluation_seed,evaluation_body);
        CUDA_CHECK_THROW(cudaDeviceSynchronize()); // Batch initialization uses the default stream.
        for(int t=0;t<evaluation_steps;++t) {
            actor.forward(trials.obs(),evaluation_count,stream.value);
            trials.step(actor.output(),{nullptr,nullptr,nullptr,nullptr},stream.value,false);
        }
        std::vector<typename Task::State> states(evaluation_count);
        CUDA_CHECK_THROW(cudaMemcpyAsync(states.data(),trials.states(),states.size()*sizeof(typename Task::State),cudaMemcpyDeviceToHost,stream.value));
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
        evaluation_successes=0;evaluation_foods=0;
        for(const auto& state:states) {
            const int foods=Task::collected(state);
            evaluation_foods+=foods;if(foods>0) ++evaluation_successes;
        }
        evaluation_update=update_id;
    }
    SwimmerStream stream;
    int n,batch,mb,update_id=0;
    uint32_t seed;
    typename Task::Batch env;
    nn::GpuMlp actor,critic;
    nn::GpuAdam actor_opt,critic_opt,std_opt;
    device_ptr<float> log_std,std_grad,obs,actions,logp,values,rewards,done,truncated,final_obs,
        final_values,last_values,adv,returns,mb_obs,mb_actions,mb_logp,mb_adv,mb_ret,
        dmean,dstd,dvalue,row_stats,adv_stats,stats;
    static nn::AdamConfig optimizer() { nn::AdamConfig c; c.lr=1e-3f;c.eps=1e-5f;c.max_grad_norm=0.5f;return c; }
    BasicSwimmerTrainer(int count,uint32_t key,sw::Body body={}):n(count),batch(count*rollout),mb(batch/minibatches),seed(key),env(n,seed,body),
        actor(swimmer_specification(sw::kAct,kInputs),std::max({n,mb,evaluation_count}),seed),critic(swimmer_specification(1,kInputs),batch,seed+1),
        actor_opt(actor.num_params(),optimizer()),critic_opt(critic.num_params(),optimizer()),std_opt(sw::kAct,optimizer()) {
        if(count<=0 || batch%minibatches!=0) throw std::invalid_argument("Invalid batch size");
        actor.set_params(nn::orthogonal_params(actor.spec(),seed,{std::sqrt(2.f),std::sqrt(2.f),0.01f}));
        critic.set_params(nn::orthogonal_params(critic.spec(),seed+1,{std::sqrt(2.f),std::sqrt(2.f),1.f}));
        log_std.allocate(sw::kAct);std_grad.allocate(sw::kAct);
        const float initial_std[2]={-0.5f,-0.5f}; log_std.copy_from_host(initial_std,2);
        obs.allocate(static_cast<size_t>(batch)*kInputs);final_obs.allocate(obs.size());
        CUDA_CHECK_THROW(cudaMemset(final_obs.get(),0,final_obs.size_bytes()));
        actions.allocate(static_cast<size_t>(batch)*sw::kAct);
        for(auto* p:{&logp,&values,&rewards,&done,&truncated,&final_values,&adv,&returns}) p->allocate(batch);
        last_values.allocate(n);mb_obs.allocate(static_cast<size_t>(mb)*kInputs);
        mb_actions.allocate(static_cast<size_t>(mb)*sw::kAct);
        for(auto* p:{&mb_logp,&mb_adv,&mb_ret,&dvalue}) p->allocate(mb);
        dmean.allocate(static_cast<size_t>(mb)*sw::kAct);dstd.allocate(dmean.size());
        row_stats.allocate(static_cast<size_t>(mb)*rl::kGaussianStats);adv_stats.allocate(2);stats.allocate(rl::kGaussianStats);
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
    }
    double update_seconds() {
        const auto start=Clock::now();const auto s=stream.value;
        for(int t=0;t<rollout;++t) {
            const size_t row=static_cast<size_t>(t)*n;
            swimmer_copy(obs.get()+row*kInputs,env.obs(),static_cast<size_t>(n)*kInputs,s);
            actor.forward(env.obs(),n,s);
            rl::sample_gaussian(actor.output(),log_std.get(),n,sw::kAct,seed,
                static_cast<uint32_t>(update_id*rollout+t),actions.get()+row*sw::kAct,logp.get()+row,s);
            critic.forward(env.obs(),n,s);swimmer_copy(values.get()+row,critic.output(),n,s);
            env.step(actions.get()+row*sw::kAct,{rewards.get()+row,done.get()+row,truncated.get()+row,final_obs.get()+row*kInputs},s);
        }
        critic.forward(env.obs(),n,s);swimmer_copy(last_values.get(),critic.output(),n,s);
        critic.forward(final_obs.get(),batch,s);swimmer_copy(final_values.get(),critic.output(),batch,s);
        rl::compute_gae(rollout,n,rewards.get(),values.get(),done.get(),truncated.get(),final_values.get(),last_values.get(),
            0.99f,0.95f,adv.get(),returns.get(),s);
        CUDA_CHECK_THROW(cudaMemsetAsync(stats.get(),0,stats.size_bytes(),s));
        for(int epoch=0;epoch<epochs;++epoch) {
            const auto perm=rl::make_permutation(static_cast<uint32_t>(batch),rng::bits(seed,static_cast<uint32_t>(update_id),static_cast<uint32_t>(epoch)));
            for(int m=0;m<minibatches;++m) {
                const int first=m*mb;
                rl::gather_rows(perm,first,mb,kInputs,obs.get(),mb_obs.get(),s);
                rl::gather_rows(perm,first,mb,sw::kAct,actions.get(),mb_actions.get(),s);
                rl::gather_rows(perm,first,mb,1,logp.get(),mb_logp.get(),s);
                rl::gather_rows(perm,first,mb,1,adv.get(),mb_adv.get(),s);
                rl::gather_rows(perm,first,mb,1,returns.get(),mb_ret.get(),s);
                rl::mean_std(mb_adv.get(),mb,adv_stats.get(),s);
                actor.forward(mb_obs.get(),mb,s);critic.forward(mb_obs.get(),mb,s);
                rl::gaussian_ppo_loss_grad(actor.output(),log_std.get(),critic.output(),mb,sw::kAct,
                    mb_actions.get(),mb_logp.get(),mb_adv.get(),mb_ret.get(),adv_stats.get(),{0.2f,Task::entropy,0.5f},
                    dmean.get(),dstd.get(),dvalue.get(),row_stats.get(),stats.get(),s);
                actor.zero_grad(s);actor.backward(dmean.get(),mb,nullptr,s);actor_opt.step(actor.params(),actor.grads(),s);
                CUDA_CHECK_THROW(cudaMemsetAsync(std_grad.get(),0,std_grad.size_bytes(),s));
                rl::sum_columns(dstd.get(),mb,sw::kAct,std_grad.get(),s);std_opt.step(log_std.get(),std_grad.get(),s);
                critic.zero_grad(s);critic.backward(dvalue.get(),mb,nullptr,s);critic_opt.step(critic.params(),critic.grads(),s);
            }
        }
        CUDA_CHECK_THROW(cudaStreamSynchronize(s));++update_id;return std::chrono::duration<double>(Clock::now()-start).count();
    }
};
using SwimmerTrainer=BasicSwimmerTrainer<SingleFoodTask>;
using ForagingTrainer=BasicSwimmerTrainer<ThreeFoodTask>;
} // namespace cudarl::rl
#endif
