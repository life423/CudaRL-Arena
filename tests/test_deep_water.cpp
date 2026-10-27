#include <cmath>
#include <filesystem>
#include "check.h"
#include "envs/swimmer/deep.h"
#include "rl/swimmer_ppo.h"
namespace dw=cudarl::deep;
TEST_CASE("deep water: neutral rest, passive drag, internal motor momentum and 3D response") {
    dw::BodyState s;const float zero[4]{};const auto position=s.position;
    for(int i=0;i<200;++i) dw::advance_physics(s,zero);
    CHECK(dw::norm(s.position-position)<1e-5f);CHECK(dw::energy(s)<1e-12f);
    s.velocity[0]=.04f;s.velocity[1]=-.02f;s.velocity[2]=.01f;
    const float initial=dw::energy(s);
    for(int i=0;i<80;++i) dw::advance_physics(s,zero);
    CHECK(dw::energy(s)<initial*.2f);
    dw::Physics vacuum;vacuum.density=0;vacuum.viscosity=0;vacuum.gravity=0;vacuum.boundaries=false;
    dw::BodyState body;const float active[4]={.5f,.4f,-.3f,.2f};
    dw::substep(body,active,.0001f,vacuum);const auto shape=dw::shape(body);dw::V momentum;
    for(const auto& l:shape.link) momentum+=dw::multiply(l.linear,body.velocity)*l.mass;
    CHECK(dw::norm(momentum)<1e-8f); // Internal torques cannot create net momentum.
    dw::BodyState swimming;float min_y=swimming.position.y,max_y=min_y,min_z=0,max_z=0;
    for(int i=0;i<300;++i) {
        const float phase=i*.05f*7;const float a[4]={sinf(phase),.5f*sinf(phase),sinf(phase-1.2f),.5f*sinf(phase-1.2f)};
        dw::advance_physics(swimming,a);min_y=std::min(min_y,swimming.position.y);max_y=std::max(max_y,swimming.position.y);
        min_z=std::min(min_z,swimming.position.z);max_z=std::max(max_z,swimming.position.z);
        CHECK(std::isfinite(dw::energy(swimming)));
    }
    CHECK(max_y-min_y>.05f);CHECK(max_z-min_z>.05f);
    std::printf("  DEEP_3D y_span=%.3f z_span=%.3f x=%.3f\n",max_y-min_y,max_z-min_z,swimming.position.x);
}
TEST_CASE("deep water: food varies vertically, collection uses 3D distance and boundary forces recover") {
    int above=0,below=0;float min_y=0,max_y=-10;
    for(uint32_t i=0;i<400;++i) {
        const auto s=dw::initial(7,i);if(s.food.y>s.body.position.y) ++above;else ++below;
        min_y=std::min(min_y,s.food.y);max_y=std::max(max_y,s.food.y);
        CHECK(dw::norm(s.food-s.body.position)>=1.2f);CHECK(fabsf(s.food.y-s.body.position.y)>=.45f);
    }
    CHECK(above>100 && below>100);CHECK(min_y< -3.1f && max_y>-.9f);
    auto s=dw::initial(7,0);s.body=dw::BodyState{};s.food=s.body.position+dw::V{0,.5f,0};const float zero[4]{};
    CHECK(!dw::advance(s,zero).ate);s.food=s.body.position;CHECK(dw::advance(s,zero).ate);CHECK(s.eaten==1);
    dw::BodyState wall;wall.position.y=-4.f;
    for(int i=0;i<80;++i) dw::advance_physics(wall,zero);
    CHECK(wall.position.y> -3.9f);CHECK(std::isfinite(wall.position.y));
}
GPU_TEST_CASE("deep water: shared CPU CUDA dynamics, PPO finite and exact checkpoint continuation") {
    constexpr int n=16;dw::Batch batch(n,7);std::vector<dw::State> cpu(n),gpu(n);
    std::vector<float> actions(n*4);for(int i=0;i<n;++i) cpu[i]=dw::initial(7,static_cast<uint32_t>(i));
    device_ptr<float> action(n*4);
    for(int t=0;t<20;++t) {
        for(int i=0;i<n*4;++i) actions[i]=.5f*std::sin(t*.13f+i*.73f);
        action.copy_from_host(actions.data(),actions.size());batch.step(action.get(),{},nullptr);
        for(int i=0;i<n;++i) dw::advance(cpu[i],actions.data()+i*4);
    }
    CUDA_CHECK_THROW(cudaDeviceSynchronize());CUDA_CHECK_THROW(cudaMemcpy(gpu.data(),batch.states(),n*sizeof(dw::State),cudaMemcpyDeviceToHost));
    for(int i=0;i<n;++i) {CHECK(dw::norm(cpu[i].body.position-gpu[i].body.position)<.002f);CHECK(cpu[i].eaten==gpu[i].eaten);}
    cudarl::rl::SwimmerConfig cfg;cfg.num_envs=16;cfg.seed=7;cfg.evaluation_every=0;
    cudarl::rl::DeepWaterTrainer trainer(cfg);trainer.update();
    for(float p:trainer.actor_params()) CHECK(std::isfinite(p));
    const auto snapshot=trainer.training_state();cudarl::rl::DeepWaterTrainer restored(cfg);restored.restore_training(snapshot);
    CHECK(trainer.actor_params()==restored.actor_params());trainer.update();restored.update();CHECK(trainer.actor_params()==restored.actor_params());
    CHECK(trainer.best_policy().parameters.empty());
    bool rejected=false;
    try {cudarl::rl::ForagingTrainer shallow(cfg);shallow.restore_training(snapshot);} catch(const std::exception&) {rejected=true;}
    CHECK(rejected);
}
