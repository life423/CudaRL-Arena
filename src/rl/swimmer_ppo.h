#ifndef CUDARL_RL_SWIMMER_PPO_H
#define CUDARL_RL_SWIMMER_PPO_H
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "envs/swimmer/swimmer.h"
#include "envs/swimmer/foraging.h"
#include "nn/adam.h"
#include "nn/mlp_gpu.h"
#include "rl/gaussian_kernels.h"
#include "rl/ppo_kernels.h"
#include "rl/evaluated_policy.h"
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
    bool shuffle_food_inputs=false;
    std::string training_checkpoint_path;
    bool resume_training=true;
    EvaluatedPolicy import_policy;
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
    int best_evaluation_foods=-1,best_evaluation_update=-1;
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
        shuffle_inputs=cfg.shuffle_food_inputs && kInputs==foraging::kObs;
        checkpoint_path_=cfg.training_checkpoint_path;
        if constexpr(kInputs==foraging::kObs) {
            if(!checkpoint_path_.empty()) {
                try {checkpoint_lease_=std::make_unique<checkpoint::Lease>(checkpoint_path_);}
                catch(const checkpoint::Busy&) {throw;}
                catch(const std::exception& e) {
                    if(checkpoint::exists(checkpoint_path_)) throw;
                    persistence_.error=e.what();
                }
                if(checkpoint_lease_ && cfg.resume_training && checkpoint::exists(checkpoint_path_)) {
                    restore_training(checkpoint::read(checkpoint_path_));persistence_.resumed=true;
                    persistence_.loaded_update=update_id;persistence_.saved_update=update_id;return;
                }
                if(checkpoint_lease_ && !cfg.resume_training && checkpoint::exists(checkpoint_path_))
                    checkpoint::archive_before_reset(checkpoint_path_);
                if(cfg.resume_training && cfg.import_policy.valid()) {
                    if(cfg.import_policy.parameters.size()!=actor_params().size() || cfg.import_policy.shuffled_inputs!=shuffle_inputs)
                        throw std::runtime_error("Best policy does not match the requested training variant");
                    actor.set_params(cfg.import_policy.parameters);update_id=cfg.import_policy.update;origin_update_=update_id;
                    best_policy_=cfg.import_policy;persistence_.imported=true;persistence_.loaded_update=update_id;
                }
            }
        }
        if(evaluation_every>0) evaluate();
        training_history_.push_back(current_stats(0));
        if(!checkpoint_path_.empty()) save_checkpoint();
    }
    std::vector<float> actor_params() const { return actor.get_params(); }
    const EvaluatedPolicy& best_policy() const {return best_policy_;}
    static std::vector<float> initial_actor_params(uint32_t key) {
        const auto spec=swimmer_specification(sw::kAct,kInputs);
        return nn::orthogonal_params(spec,key,{std::sqrt(2.f),std::sqrt(2.f),0.01f});
    }
    Stats update() {
        const double elapsed=update_seconds();
        if(evaluation_every>0 && update_id % evaluation_every==0) evaluate();
        const auto result=current_stats(elapsed*1000.0);
        training_history_.push_back(result);
        if(!checkpoint_path_.empty()) {
            if(Clock::now()-last_save_>=std::chrono::seconds(30)) save_checkpoint();
        }
        return result;
    }
    Stats current_stats(double milliseconds=0) const {
        return {update_id,static_cast<long long>(update_id)*batch,milliseconds,
                evaluation_update,evaluation_successes,evaluation_every>0?evaluation_count:0,
                evaluation_foods,evaluation_steps*static_cast<double>(sw::kDt),
                best_policy_.valid()?best_policy_.foods:-1,best_policy_.update};
    }
    const std::vector<Stats>& training_history() const {return training_history_;}
    const checkpoint::Status& checkpoint_status() const {return persistence_;}
    void save_checkpoint() {
        if(checkpoint_path_.empty() || !checkpoint_lease_) return;
        try {
            checkpoint::write_atomic(checkpoint_path_,training_state());
            persistence_.saved_update=update_id;persistence_.error.clear();
        } catch(const std::exception& e) {persistence_.error=e.what();}
        last_save_=Clock::now();
    }
    std::vector<uint8_t> training_state() const {
        if constexpr(kInputs==foraging::kObs) {
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));checkpoint::Writer out;
            out.value(uint32_t{0x46504731});out.value(uint32_t{1});
            out.value(static_cast<uint32_t>(sizeof(typename Task::State)));out.value(static_cast<uint32_t>(sizeof(sw::Body)));
            out.value(n);out.value(seed);out.value(evaluation_every);out.value(shuffle_inputs);out.value(evaluation_body);
            const std::vector<float> protocol={static_cast<float>(hidden),static_cast<float>(rollout),static_cast<float>(epochs),static_cast<float>(minibatches),
                0.99f,0.95f,0.2f,Task::entropy,0.5f,sw::kDt,foraging::kFoodRadiusX,foraging::kFoodRadiusY,foraging::kFoodSeparation,foraging::kRespawnDistance};
            out.array(protocol);out.value(update_id);out.value(origin_update_);out.value(evaluation_update);out.value(evaluation_successes);out.value(evaluation_foods);
            out.value(best_policy_.update);out.value(best_policy_.foods);out.value(best_policy_.count);out.value(best_policy_.steps);
            out.value(best_policy_.evaluation_seed);out.value(best_policy_.training_seed);out.value(best_policy_.shuffled_inputs);out.array(best_policy_.parameters);
            out.array(actor.get_params());out.array(critic.get_params());
            std::vector<float> deviation(sw::kAct);log_std.copy_to_host(deviation.data(),deviation.size());out.array(deviation);
            actor_opt.save(out);critic_opt.save(out);std_opt.save(out);env.save(out);out.array(training_history_);
            return out.data;
        } else throw std::runtime_error("Full training checkpoints require the three-food task");
    }
    void restore_training(const std::vector<uint8_t>& bytes) {
        if constexpr(kInputs==foraging::kObs) {
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));checkpoint::Reader in{bytes};
            if(in.value<uint32_t>()!=0x46504731u || in.value<uint32_t>()!=1u || in.value<uint32_t>()!=sizeof(typename Task::State) || in.value<uint32_t>()!=sizeof(sw::Body))
                throw std::runtime_error("Incompatible training checkpoint format or state layout");
            if(in.value<int>()!=n || in.value<uint32_t>()!=seed || in.value<int>()!=evaluation_every || in.value<bool>()!=shuffle_inputs)
                throw std::runtime_error("Training save has different population, seed, evaluation interval or input ordering");
            const auto body=in.value<sw::Body>();
            if(body.head!=evaluation_body.head || body.proximal!=evaluation_body.proximal || body.distal!=evaluation_body.distal || body.fin_normal!=evaluation_body.fin_normal ||
               body.motor_speed!=evaluation_body.motor_speed || body.tapered!=evaluation_body.tapered || body.eel!=evaluation_body.eel || body.inertial!=evaluation_body.inertial)
                throw std::runtime_error("Training save uses different body physics");
            const auto protocol=in.array<float>(14);
            const std::vector<float> expected={static_cast<float>(hidden),static_cast<float>(rollout),static_cast<float>(epochs),static_cast<float>(minibatches),
                0.99f,0.95f,0.2f,Task::entropy,0.5f,sw::kDt,foraging::kFoodRadiusX,foraging::kFoodRadiusY,foraging::kFoodSeparation,foraging::kRespawnDistance};
            if(protocol!=expected) throw std::runtime_error("Training save uses different PPO or food rules");
            update_id=in.value<int>();origin_update_=in.value<int>();evaluation_update=in.value<int>();evaluation_successes=in.value<int>();evaluation_foods=in.value<int>();
            if(update_id<0 || origin_update_<0 || origin_update_>update_id || evaluation_update>update_id) throw std::runtime_error("Invalid training counters");
            best_policy_.update=in.value<int>();best_policy_.foods=in.value<int>();best_policy_.count=in.value<int>();best_policy_.steps=in.value<int>();
            best_policy_.evaluation_seed=in.value<uint32_t>();best_policy_.training_seed=in.value<uint32_t>();best_policy_.shuffled_inputs=in.value<bool>();
            best_policy_.parameters=in.array<float>(actor.num_params());
            auto parameters=[&](int count) {
                auto p=in.array<float>(count);if(p.size()!=static_cast<size_t>(count)) throw std::runtime_error("Invalid training parameter count");
                for(float v:p) if(!std::isfinite(v)) throw std::runtime_error("Invalid training parameter");return p;
            };
            actor.set_params(parameters(actor.num_params()));critic.set_params(parameters(critic.num_params()));const auto deviation=parameters(sw::kAct);
            log_std.copy_from_host(deviation.data(),deviation.size());actor_opt.load(in);critic_opt.load(in);std_opt.load(in);env.load(in);
            training_history_=in.array<Stats>();in.finish();
            const auto expected_steps=static_cast<long long>(update_id-origin_update_)*epochs*minibatches;
            if(actor_opt.steps()!=expected_steps || critic_opt.steps()!=expected_steps || std_opt.steps()!=expected_steps || training_history_.empty() || training_history_.back().update!=update_id)
                throw std::runtime_error("Training save has inconsistent optimizer/history counters");
        } else throw std::runtime_error("Full training checkpoints require the three-food task");
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
        // Select only on the fixed comparison suite, never a separately seeded
        // holdout. Ties retain the earlier actor; weights match this exact score.
        if(evaluation_seed==0x51A7E123u && (!best_policy_.valid() || evaluation_foods>best_policy_.foods))
            best_policy_={update_id,evaluation_foods,evaluation_count,evaluation_steps,evaluation_seed,seed,actor_params(),shuffle_inputs};
    }
    EvaluatedPolicy best_policy_;
    SwimmerStream stream;
    int n,batch,mb,update_id=0;
    uint32_t seed;
    bool shuffle_inputs=false;
    int origin_update_=0;
    std::string checkpoint_path_;
    std::unique_ptr<checkpoint::Lease> checkpoint_lease_;
    checkpoint::Status persistence_;
    std::vector<Stats> training_history_;
    Clock::time_point last_save_=Clock::now();
    typename Task::Batch env;
    nn::GpuMlp actor,critic;
    nn::GpuAdam actor_opt,critic_opt,std_opt;
    device_ptr<float> log_std,std_grad,obs,actions,logp,values,rewards,done,truncated,final_obs,
        final_values,last_values,adv,returns,mb_obs,mb_actions,mb_logp,mb_adv,mb_ret,
        dmean,dstd,dvalue,row_stats,adv_stats,stats,bootstrap_obs;
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
        bootstrap_obs.allocate(static_cast<size_t>(n)*kInputs);
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
            const float* input=env.obs();
            if constexpr(kInputs==foraging::kObs) if(shuffle_inputs) {
                foraging::shuffle_food_inputs(obs.get()+row*kInputs,n,n,seed,static_cast<uint32_t>(update_id*rollout+t),s);
                input=obs.get()+row*kInputs;
            }
            // Keep this exact observation for every PPO epoch: old logp and
            // new logp must refer to the same food ordering.
            actor.forward(input,n,s);
            rl::sample_gaussian(actor.output(),log_std.get(),n,sw::kAct,seed,
                static_cast<uint32_t>(update_id*rollout+t),actions.get()+row*sw::kAct,logp.get()+row,s);
            critic.forward(input,n,s);swimmer_copy(values.get()+row,critic.output(),n,s);
            env.step(actions.get()+row*sw::kAct,{rewards.get()+row,done.get()+row,truncated.get()+row,final_obs.get()+row*kInputs},s);
        }
        const float* bootstrap=env.obs();
        if constexpr(kInputs==foraging::kObs) if(shuffle_inputs) {
            swimmer_copy(bootstrap_obs.get(),env.obs(),static_cast<size_t>(n)*kInputs,s);
            foraging::shuffle_food_inputs(bootstrap_obs.get(),n,n,seed,static_cast<uint32_t>((update_id+1)*rollout),s);
            foraging::shuffle_food_inputs(final_obs.get(),batch,n,seed,static_cast<uint32_t>(update_id*rollout+1),s);
            bootstrap=bootstrap_obs.get();
        }
        critic.forward(bootstrap,n,s);swimmer_copy(last_values.get(),critic.output(),n,s);
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
