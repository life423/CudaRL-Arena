#include <cmath>
#include <chrono>
#include <filesystem>
#include "check.h"
#include "envs/swimmer/deep_pond.h"
#include "rl/swimmer_ppo.h"
#include "nn/mlp_cpu.h"
namespace dp=cudarl::deep_pond;
namespace dw=cudarl::deep;
TEST_CASE("deep pond: three separated depths, relocation and consumed-only replacement") {
    float minimum_depth=99,maximum_depth=-99;int quadrants[4]{};float shortest=100;
    for(uint32_t i=0;i<350;++i) {
        auto s=dp::initial(7,i);
        for(int round=0;round<8;++round) {
            for(int f=0;f<3;++f) {
                const auto p=s.food[f];CHECK(dp::water(p));
                minimum_depth=std::min(minimum_depth,-p.y);maximum_depth=std::max(maximum_depth,-p.y);
                ++quadrants[(p.x>=0?1:0)+(p.z>=0?2:0)];
                for(int j=0;j<f;++j) {CHECK(dw::norm(p-s.food[j])>=dp::separation-1e-5f);CHECK(std::fabs(p.y-s.food[j].y)>=dp::vertical_gap-1e-5f);}
            }
            const int slot=round%3;const auto old=s;
            // Place a resting, non-penetrating body at the food; its tail points
            // toward deeper water. Teleporting accumulated velocity or a tail
            // through the bank would test contact impulses, not replacement.
            s.body=dw::BodyState{};s.body.position=s.food[slot];
            s.body.orientation=dw::axis_rotation({0,1,0},std::atan2(-s.food[slot].z,s.food[slot].x));
            const float zero[4]{};const auto result=dp::advance(s,zero);
            CHECK(result.eaten_slot==slot);CHECK(s.eaten==old.eaten+1);CHECK(dp::water(s.food[slot]));
            const float distance=dw::norm(s.food[slot]-old.food[slot]);shortest=std::min(shortest,distance);CHECK(distance>=dp::relocation-1e-5f);
            for(int other=0;other<3;++other) if(other!=slot) CHECK(dw::norm(s.food[other]-old.food[other])==0);
        }
    }
    CHECK(minimum_depth<.8f && maximum_depth>3.2f);for(int count:quadrants) CHECK(count>1000);
    std::printf("  DEEP_POND_PLACEMENT trials=2800 relocation=%.3f depth=%.3f..%.3f\n",shortest,minimum_depth,maximum_depth);
}
TEST_CASE("deep pond: swimming forces are retained away from the bank and the basin resists penetration") {
    auto pond=dp::initial(7,0);dw::State original;original.body=pond.body;
    original.food=pond.food[0];
    for(int t=0;t<20;++t) {
        float a[4]={.4f*std::sin(t*.3f),.3f*std::cos(t*.25f),.5f*std::sin(t*.3f-1),.2f*std::cos(t*.25f-1)};
        dw::advance(original,a);dp::advance(pond,a);
        CHECK(dw::norm(original.body.position-pond.body.position)<1e-6f);
        for(int j=0;j<4;++j) CHECK(std::fabs(original.body.angle[j]-pond.body.angle[j])<1e-6f);
    }
    dw::BodyState body;body.position={1,-4,0};dw::Physics physics;physics.pond=true;const float zero[4]{};
    for(int i=0;i<60;++i) dw::advance_physics(body,zero,physics);
    CHECK(body.position.y>dw::pond_floor(body.position.x,body.position.z));
    auto state=dp::initial(4,1);state.body.position={0,-2,0};state.food[0]={0,-1,0};
    CHECK(dp::advance(state,zero).eaten_slot<0); // Same X/Z, wrong depth is not collection.
}
GPU_TEST_CASE("deep pond: transfer preserves motor outputs, Adam, source save and exact resumed learning") {
    const auto path=std::filesystem::temp_directory_path()/std::filesystem::path("cudarl-deep-transfer-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
    cudarl::rl::SwimmerConfig cfg;cfg.num_envs=16;cfg.seed=7;cfg.evaluation_every=25;cfg.training_checkpoint_path=(path/"source.bin").string();
    std::vector<float> actor,critic;std::vector<uint8_t> adam;int inherited=0;
    {
        cudarl::rl::DeepWaterTrainer original(cfg);original.update();original.update();original.save_checkpoint();
        actor=original.actor_params();critic=original.critic.get_params();inherited=original.current_stats().update;
        cudarl::checkpoint::Writer saved;original.actor_opt.save(saved);adam=saved.data;
    }
    const auto source_bytes=cudarl::checkpoint::read(cfg.training_checkpoint_path);
    cfg.deep_source_checkpoint=cfg.training_checkpoint_path;cfg.training_checkpoint_path=(path/"pond.bin").string();cfg.shuffle_food_inputs=true;
    std::vector<float> upgraded;
    {
        cudarl::rl::DeepPondTrainer next(cfg);upgraded=next.actor_params();
        CHECK(next.checkpoint_status().imported);CHECK(next.current_stats().update==inherited);
        CHECK(next.actor_opt.steps()==inherited*32);CHECK(next.critic_opt.steps()==inherited*32);CHECK(next.std_opt.steps()==inherited*32);
        CHECK(next.critic.get_params()==cudarl::rl::expand_deep_parameters(critic,1));
        cudarl::nn::CpuMlp<float> before(cudarl::rl::swimmer_specification(4,27)),after(cudarl::rl::swimmer_specification(4,33));
        before.params()=actor;after.params()=upgraded;
        for(uint32_t i=0;i<80;++i) {
            auto scene=dp::initial(7,i);dw::State old;old.body=scene.body;old.food=scene.food[0];
            float x[27],y[33];dw::observe(old,x);dp::observe(scene,y);before.forward(x,1);after.forward(y,1);
            for(int k=0;k<4;++k) CHECK(std::fabs(before.output()[k]-after.output()[k])<2e-6f);
        }
        cudarl::checkpoint::Reader old_adam{adam};cudarl::checkpoint::Writer new_adam;next.actor_opt.save(new_adam);cudarl::checkpoint::Reader mapped{new_adam.data};
        CHECK(old_adam.value<cudarl::nn::AdamConfig>().lr==mapped.value<cudarl::nn::AdamConfig>().lr);CHECK(old_adam.value<int>()==mapped.value<int>());
        CHECK(cudarl::rl::expand_deep_parameters(old_adam.array<float>(),4)==mapped.array<float>());
        CHECK(cudarl::rl::expand_deep_parameters(old_adam.array<float>(),4)==mapped.array<float>());
        next.update();const auto snapshot=next.training_state();auto resume_cfg=cfg;resume_cfg.training_checkpoint_path.clear();resume_cfg.deep_source_checkpoint.clear();
        cudarl::rl::DeepPondTrainer resumed(resume_cfg);resumed.restore_training(snapshot);
        for(int t=0;t<19;++t) {next.update();resumed.update();}CHECK(next.actor_params()==resumed.actor_params());
        for(float p:next.actor_params()) CHECK(std::isfinite(p));
    }
    CHECK(cudarl::checkpoint::read(cfg.deep_source_checkpoint)==source_bytes);
    {
        cudarl::rl::DeepPondTrainer restored(cfg);CHECK(restored.checkpoint_status().resumed);CHECK(!restored.checkpoint_status().imported);
        CHECK(restored.current_stats().update==inherited);CHECK(restored.actor_params()==upgraded);
    }
    std::filesystem::remove_all(path);
}
GPU_TEST_CASE("deep pond: CPU CUDA agreement and observation shuffling keep physical food unchanged") {
    constexpr int n=16;dp::Batch batch(n,7);std::vector<dp::State> cpu(n),gpu(n);std::vector<float> actions(n*4),obs(n*33),shuffled(n*33);
    for(int i=0;i<n;++i) cpu[i]=dp::initial(7,static_cast<uint32_t>(i));device_ptr<float> motor(n*4),inputs(n*33);
    for(int t=0;t<30;++t) {
        for(int i=0;i<n*4;++i) actions[i]=.7f*std::sin(t*.2f+i*.7f);motor.copy_from_host(actions.data(),actions.size());batch.step(motor.get(),{},nullptr);
        for(int i=0;i<n;++i) dp::advance(cpu[i],actions.data()+i*4);
    }
    CUDA_CHECK_THROW(cudaDeviceSynchronize());CUDA_CHECK_THROW(cudaMemcpy(gpu.data(),batch.states(),n*sizeof(dp::State),cudaMemcpyDeviceToHost));
    for(int i=0;i<n;++i) {CHECK(dw::norm(cpu[i].body.position-gpu[i].body.position)<.003f);CHECK(cpu[i].eaten==gpu[i].eaten);}
    CUDA_CHECK_THROW(cudaMemcpy(obs.data(),batch.obs(),obs.size()*sizeof(float),cudaMemcpyDeviceToHost));inputs.copy_from_host(obs.data(),obs.size());
    dp::shuffle_food_inputs(inputs.get(),n,n,7,123,nullptr);CUDA_CHECK_THROW(cudaDeviceSynchronize());inputs.copy_to_host(shuffled.data(),shuffled.size());
    for(int i=0;i<n;++i) {
        const int order=cudarl::rng::bits(7u^0xA6F029B3u,static_cast<uint32_t>(i),123u)%6u;
        dp::permute(obs.data()+i*33,order);
    }
    CHECK(shuffled==obs);
}
