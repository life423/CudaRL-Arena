#ifndef CUDARL_FEEDING_CPU_H
#define CUDARL_FEEDING_CPU_H
#include "feeding_policy.h"
#include "progress_selector_policy.h"
#include "gaussian_math.h"
#include "nn/mlp_cpu.h"
#include <numeric>

namespace cudarl::feeding {
// The same task, selector features, motor observations and force integration as CUDA.
// Only active pellets needing a goal decision are batched for CPU inference.
class CpuBatch {
    nn::CpuMlp<float> selector_{rl::progress_selector_spec()};
public:
    std::vector<State> states;
    std::vector<float> observations;
    Context context;
    uint32_t seed;
    CpuBatch(int n,uint32_t key,const std::vector<float>& selector,const Context& c={}):
        states(static_cast<size_t>(n)),observations(static_cast<size_t>(n)*kObs),context(c),seed(key) {
        if(n<1 || n>2048) throw std::invalid_argument("Invalid CPU feeding population");
        selector_.params()=selector; reset();
    }
    void reset() {
        for(size_t i=0;i<states.size();++i) states[i]=initial(seed,static_cast<uint32_t>(i),0,context,i%4==0);
        prepare();
    }
    void prepare() {
        std::vector<float> input; std::vector<std::pair<size_t,int>> rows;
        for(size_t i=0;i<states.size();++i) if(states[i].decision)
            for(int f=0;f<capacity;++f) if(states[i].pellets[f].active) {
                const auto begin=input.size();input.resize(begin+progress_selector::kFeatures);
                features(states[i],f,input.data()+begin);rows.emplace_back(i,f);
            }
        for(auto& s:states) if(s.decision) s.target=-1;
        if(!rows.empty()) {
            selector_.forward(input.data(),static_cast<int>(rows.size()));
            std::vector<float> best(states.size(),-1e30f);
            for(size_t r=0;r<rows.size();++r) {
                const auto [world,food]=rows[r];
                if(selector_.output()[r]>best[world]) {best[world]=selector_.output()[r];states[world].target=food;}
            }
        }
        for(size_t i=0;i<states.size();++i) {states[i].decision=false;motor_observe(states[i],observations.data()+i*kObs);}
    }
    void step(const float* actions,float* rewards=nullptr,float* done=nullptr,float* final_obs=nullptr,bool autoreset=true,bool practice_events=false) {
        for(size_t i=0;i<states.size();++i) {
            const auto current=(autoreset||practice_events)?practice_current(states[i],context):context.current;
            const float reward=advance(states[i],actions+i*kAct,current);
            const bool timeout=states[i].steps>=kEpisodeSteps;
            if(rewards) rewards[i]=reward;if(done) done[i]=timeout?1.f:0.f;
            if((autoreset||practice_events) && (!autoreset||!timeout)) replenish(states[i],context);
        }
        prepare(); // The time-limit bootstrap uses the selected terminal observation.
        if(autoreset) {
            bool reset_any=false;
            for(size_t i=0;i<states.size();++i) if(states[i].steps>=kEpisodeSteps) {
                if(final_obs) std::copy_n(observations.data()+i*kObs,kObs,final_obs+i*kObs);
                const auto old=states[i];states[i]=initial(old.seed,old.environment,old.episode+1,context,old.retention);reset_any=true;
            }
            if(reset_any) prepare();
        }
    }
};
}
namespace cudarl::rl {
class CpuFeedingTrainer {
    static nn::AdamConfig optimizer(float lr) {nn::AdamConfig c;c.lr=lr;c.eps=1e-5f;c.max_grad_norm=.5f;return c;}
public:
    static constexpr int rollout_steps=64,ppo_epochs=4,ppo_minibatches=8;
    feeding::CpuBatch env;
    nn::CpuMlp<float> actor{swimmer_specification(4,feeding::kObs)},critic{swimmer_specification(1,feeding::kObs)};
    nn::CpuAdam<float> actor_opt{actor.spec().num_params(),optimizer(.0001f)},
        critic_opt{critic.spec().num_params(),optimizer(.0003f)},std_opt{4,optimizer(.0001f)};
    std::vector<float> log_std=std::vector<float>(4,-1.5f);
    int update_id=0;
    CpuFeedingTrainer(int n,const ProgressPolicy& source,const feeding::Context& context):env(n,71,source.actor,context) {
        if(n<8 || n%8) throw std::invalid_argument("CPU population must be divisible by eight");
        actor.params()=feeding_motor_parameters(source.baseline.motor);
        critic.params()=nn::orthogonal_params(critic.spec(),72,{std::sqrt(2.f),std::sqrt(2.f),1.f});
    }
    void update() {
        constexpr int inputs=feeding::kObs,actions=4;
        const int n=static_cast<int>(env.states.size()),batch=n*rollout_steps,mb=batch/ppo_minibatches;
        std::vector<float> obs(static_cast<size_t>(batch)*inputs),act(static_cast<size_t>(batch)*actions),
            old_logp(batch),values(batch),rewards(batch),done(batch),final_obs(obs.size(),0),final_values(batch),adv(batch),returns(batch);
        for(int t=0;t<rollout_steps;++t) {
            const int row=t*n;std::copy(env.observations.begin(),env.observations.end(),obs.data()+row*inputs);
            actor.forward(env.observations.data(),n);critic.forward(env.observations.data(),n);
            for(int i=0;i<n;++i) {
                const auto* mean=actor.output()+i*actions;auto* a=act.data()+(row+i)*actions;
                for(int j=0;j<actions;++j) a[j]=mean[j]+std::exp(log_std[j])*rng::normal(71,static_cast<uint32_t>(update_id*rollout_steps+t),static_cast<uint32_t>(i*actions+j));
                old_logp[row+i]=gaussian_logp(mean,log_std.data(),a,actions);values[row+i]=critic.output()[i];
            }
            env.step(act.data()+row*actions,rewards.data()+row,done.data()+row,final_obs.data()+row*inputs);
        }
        critic.forward(env.observations.data(),n);const std::vector<float> last(critic.output(),critic.output()+n);
        critic.forward(final_obs.data(),batch);std::copy_n(critic.output(),batch,final_values.data());
        for(int i=0;i<n;++i) gae_one_env(rollout_steps,n,i,rewards.data(),values.data(),done.data(),done.data(),final_values.data(),last.data(),.99f,.95f,adv.data(),returns.data());
        std::vector<float> x(static_cast<size_t>(mb)*inputs),a(static_cast<size_t>(mb)*actions),lp(mb),advantages(mb),ret(mb),dmean(a.size()),dv(mb),ds(4),row_std(4);
        for(int epoch=0;epoch<ppo_epochs;++epoch) {
            const auto permutation=make_permutation(static_cast<uint32_t>(batch),rng::bits(71,static_cast<uint32_t>(update_id),static_cast<uint32_t>(epoch)));
            for(int m=0;m<ppo_minibatches;++m) {
                for(int i=0;i<mb;++i) {
                    const auto row=permute(permutation,static_cast<uint32_t>(m*mb+i));
                    std::copy_n(obs.data()+row*inputs,inputs,x.data()+i*inputs);std::copy_n(act.data()+row*actions,actions,a.data()+i*actions);
                    lp[i]=old_logp[row];advantages[i]=adv[row];ret[i]=returns[row];
                }
                double mean=0,square=0;for(float v:advantages) {mean+=v;square+=static_cast<double>(v)*v;}
                mean/=mb;const float deviation=static_cast<float>(std::sqrt(std::max(0.,square/mb-mean*mean)));
                actor.forward(x.data(),mb);critic.forward(x.data(),mb);std::fill(ds.begin(),ds.end(),0.f);
                for(int i=0;i<mb;++i) {
                    const float normalized=(advantages[i]-static_cast<float>(mean))/(deviation+1e-8f);
                    ppo_gaussian_row(actor.output()+i*actions,log_std.data(),actions,a.data()+i*actions,lp[i],normalized,critic.output()[i],ret[i],{.2f,.002f,.5f},1.f/mb,dmean.data()+i*actions,row_std.data(),dv.data()+i);
                    for(int j=0;j<actions;++j) ds[j]+=row_std[j];
                }
                actor.zero_grad();actor.backward(dmean.data(),mb);actor_opt.step(actor.params(),actor.grads());
                critic.zero_grad();critic.backward(dv.data(),mb);critic_opt.step(critic.params(),critic.grads());std_opt.step(log_std,ds);
            }
        }
        ++update_id;
    }
    void save(checkpoint::Writer& out) const {
        out.value(update_id);out.array(actor.params());out.array(critic.params());out.array(log_std);
        actor_opt.save(out);critic_opt.save(out);std_opt.save(out);out.array(env.states);out.value(env.context);
    }
    void load(checkpoint::Reader& in) {
        update_id=in.value<int>();
        auto weights=[&](size_t n) {auto v=in.array<float>(n);if(v.size()!=n) throw std::runtime_error("Invalid CPU learner parameter count");for(float p:v) if(!std::isfinite(p)) throw std::runtime_error("Non-finite CPU learner save");return v;};
        actor.params()=weights(actor.params().size());critic.params()=weights(critic.params().size());log_std=weights(4);
        actor_opt.load(in);critic_opt.load(in);std_opt.load(in);
        const auto worlds=in.array<feeding::State>(env.states.size());const auto context=in.value<feeding::Context>();
        if(worlds.size()!=env.states.size() || context.count<0 || context.count>32 || update_id<0 ||
            actor_opt.steps()!=static_cast<long long>(update_id)*ppo_epochs*ppo_minibatches || critic_opt.steps()!=actor_opt.steps() || std_opt.steps()!=actor_opt.steps()) throw std::runtime_error("Inconsistent CPU learner save");
        for(const auto& s:worlds) if(s.steps<0 || s.steps>=feeding::kEpisodeSteps || s.target>=feeding::capacity || !std::isfinite(s.body.position.x)) throw std::runtime_error("Invalid CPU practice world");
        env.states=worlds;env.context=context;env.prepare();
    }
};
}
#endif
