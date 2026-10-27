#include <cmath>
#include "check.h"
#include "rl/pond_selector_ppo.h"
#include "nn/mlp_cpu.h"
namespace ps=cudarl::pond_selector;
namespace dp=cudarl::deep_pond;
namespace d=cudarl::deep;
namespace rl=cudarl::rl;
TEST_CASE("pond selector: all six input orders preserve each food's shared features") {
    const int orders[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    for(uint32_t trial=0;trial<64;++trial) {
        const auto world=dp::initial(191,trial);float original[36];ps::observe(world,static_cast<int>(trial%3),original);
        for(int order=0;order<6;++order) {
            float shuffled[36];std::copy(original,original+36,shuffled);dp::permute(shuffled,order);
            for(int j=0;j<3;++j) shuffled[33+j]=original[33+orders[order][j]];
            for(int j=0;j<3;++j) {
                float before[32],after[32];ps::features(original,orders[order][j],before);ps::features(shuffled,j,after);
                for(int k=0;k<32;++k) CHECK(before[k]==after[k]);
            }
        }
    }
}
GPU_TEST_CASE("pond selector: GPU motor inputs preserve physical slots and CPU physics") {
    constexpr int n=8;ps::Batch world(n,31,20);std::vector<dp::State> cpu(n),gpu(n);std::vector<float> obs(n*33),torques(n*4,.2f),rewards(n),final_obs(n*36);
    std::vector<int> targets(n);for(int i=0;i<n;++i) {targets[i]=i%3;cpu[i]=dp::initial(31,static_cast<uint32_t>(i));}
    device_ptr<int> selected(n);device_ptr<float> actions(n*4),r(n),final(n*36),done(n),truncated(n);selected.copy_from_host(targets.data(),targets.size());actions.copy_from_host(torques.data(),torques.size());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());cudarl::swimmer::Outputs out{r.get(),done.get(),truncated.get(),final.get()};
    world.begin(selected.get(),out,nullptr);
    for(int t=0;t<40;++t) {
        CUDA_CHECK_THROW(cudaMemcpy(obs.data(),world.motor_obs(nullptr),obs.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for(int i=0;i<n;++i) {
            float reference[33];dp::observe(cpu[i],reference);cudarl::demos::primary_pond_food(reference,targets[i]);
            for(int k=0;k<33;++k) CHECK(std::fabs(reference[k]-obs[i*33+k])<.004f);
            dp::advance(cpu[i],torques.data()+i*4);
        }
        world.step(actions.get(),out,nullptr);
    }
    CUDA_CHECK_THROW(cudaMemcpy(gpu.data(),world.states(),gpu.size()*sizeof(dp::State),cudaMemcpyDeviceToHost));
    r.copy_to_host(rewards.data(),rewards.size());
    for(int i=0;i<n;++i) {
        CHECK(d::norm(cpu[i].body.position-gpu[i].body.position)<.003f);CHECK(cpu[i].eaten==gpu[i].eaten);CHECK(rewards[i]==float(gpu[i].eaten));
        for(int f=0;f<3;++f) CHECK(d::norm(cpu[i].food[f]-gpu[i].food[f])<.003f);
    }
}
GPU_TEST_CASE("pond selector: PPO freezes motor, resumes exactly across timeout and preserves policy bundle") {
    const auto weights=rl::DeepPondTrainer::initial_actor_params(7);rl::SelectorConfig cfg;cfg.environments=16;cfg.seed=19;
    rl::SelectorTrainer a(cfg,weights,42);a.train_update();const auto saved=a.snapshot();
    std::vector<float> observations(16*36),scores(16*3);CUDA_CHECK_THROW(cudaMemcpy(observations.data(),a.env.obs(),observations.size()*sizeof(float),cudaMemcpyDeviceToHost));
    a.score(a.env.obs(),16);CUDA_CHECK_THROW(cudaMemcpyAsync(scores.data(),a.actor.output(),scores.size()*sizeof(float),cudaMemcpyDeviceToHost,a.stream.value));CUDA_CHECK_THROW(cudaStreamSynchronize(a.stream.value));
    cudarl::nn::CpuMlp<float> cpu(rl::selector_spec());cpu.params()=a.actor.get_params();
    for(int row=0;row<16;++row) for(int food=0;food<3;++food) {
        float features[32];ps::features(observations.data()+row*36,food,features);cpu.forward(features,1);CHECK(std::fabs(cpu.output()[0]-scores[row*3+food])<3e-6f);
    }
    rl::SelectorTrainer b(cfg,weights,42);b.restore(saved);
    a.train_update();b.train_update();a.train_update();b.train_update();
    CHECK(a.actor.get_params()==b.actor.get_params());CHECK(a.critic.get_params()==b.critic.get_params());CHECK(a.snapshot()==b.snapshot());
    CHECK(a.motor.get_params()==weights);CHECK(b.motor.get_params()==weights);
    for(float x:a.actor.get_params()) CHECK(std::isfinite(x));
    a.evaluate();cudarl::checkpoint::Writer out;a.best.write(out);cudarl::checkpoint::Reader in{out.data};const auto best=rl::SelectorPolicy::read(in);in.finish();
    CHECK(best.motor==weights);CHECK(best.selector==a.best.selector);CHECK(best.motor_update==42);CHECK(best.decision_steps==20);
    rl::SelectorTrainer c(cfg,weights,43);bool rejected=false;
    try {c.restore(saved);} catch(const std::runtime_error&) {rejected=true;}CHECK(rejected);
}
GPU_TEST_CASE("pond selector: recovery failures are terminal, training resets never alter deployment physics") {
    ps::Batch world(1,73,20,true);device_ptr<int> target(1);const int slot=0;target.copy_from_host(&slot,1);
    device_ptr<float> actions(4),reward(1),done(1),truncated(1),final(36);const float zero[4]{};actions.copy_from_host(zero,4);
    auto state=dp::initial(73,0);state.steps=19;state.since_food=399;state.body.position={0,-2,0};
    CUDA_CHECK_THROW(cudaDeviceSynchronize());CUDA_CHECK_THROW(cudaMemcpy(const_cast<dp::State*>(world.states()),&state,sizeof(state),cudaMemcpyHostToDevice));
    cudarl::swimmer::Outputs out{reward.get(),done.get(),truncated.get(),final.get()};world.begin(target.get(),out,nullptr);world.step(actions.get(),out,nullptr);
    float r=0,dn=0,tr=0;reward.copy_to_host(&r,1);done.copy_to_host(&dn,1);truncated.copy_to_host(&tr,1);
    dp::State after;CUDA_CHECK_THROW(cudaMemcpy(&after,world.states(),sizeof(after),cudaMemcpyDeviceToHost));
    CHECK(r==-5.f && dn==1.f && tr==0.f);CHECK(after.episode==1 && after.steps==0 && after.since_food==0);
    // The same physical state in normal inference continues without a rescue/reset.
    dp::advance(state,zero);CHECK(state.steps==20 && state.since_food==400 && state.episode==0);
}
GPU_TEST_CASE("pond selector: recovery training resumes bit identically through a five-minute timeout") {
    const auto weights=rl::DeepPondTrainer::initial_actor_params(7);rl::SelectorConfig cfg;cfg.environments=8;cfg.seed=61;cfg.recovery=true;
    rl::SelectorTrainer a(cfg,weights,42);std::vector<dp::State> states(8);
    CUDA_CHECK_THROW(cudaMemcpy(states.data(),a.env.states(),states.size()*sizeof(dp::State),cudaMemcpyDeviceToHost));
    for(auto& state:states) state.steps=5760;
    CUDA_CHECK_THROW(cudaMemcpy(const_cast<dp::State*>(a.env.states()),states.data(),states.size()*sizeof(dp::State),cudaMemcpyHostToDevice));
    const auto saved=a.snapshot();rl::SelectorTrainer b(cfg,weights,42);b.restore(saved);a.train_update();b.train_update();
    CHECK(a.snapshot()==b.snapshot());CHECK(a.motor.get_params()==weights);
}
