#include "check.h"
#include "rl/shelter_motor_ppo.h"
#include "nn/mlp_cpu.h"
#include "rl/pond_selector_policy.h"

using namespace cudarl;
TEST_CASE("shelter motor transfer preserves every food input and action") {
    const auto old=nn::orthogonal_params(rl::swimmer_specification(4,arch::kObs),73,{std::sqrt(2.f),std::sqrt(2.f),.01f});
    nn::CpuMlp<float> before(rl::swimmer_specification(4,arch::kObs)),after(rl::swimmer_specification(4,shelter_motor::kObs));
    before.params()=old;after.params()=rl::transfer_shelter_motor(old);
    for(uint32_t i=0;i<16;++i) {
        shelter_motor::State world;world.state=arch::initial(73,i,0,false,{2});world.state.pond.target=0;
        float original[arch::kObs],next[shelter_motor::kObs];arch::observe(world.state,original,{2});shelter_motor::observe(world,next,{2});
        for(int j=0;j<arch::kObs;++j) CHECK(original[j]==next[j]);for(int j=55;j<59;++j) CHECK(next[j]==0);
        before.forward(original,1);after.forward(next,1);for(int j=0;j<4;++j) CHECK(fabsf(before.output()[j]-after.output()[j])<1e-6f);
        const auto physical=world.state;const auto point=predator::cover(0,world.request.eye,world.state.habitat,{2});
        world.request.cover=true;world.request.goal={point.position,{},-1,0,point.valid};
        world.request.heading=deep::axis_rotation({0,1,0},point.yaw).rotate({1,0,0});shelter_motor::observe(world,next,{2});
        CHECK(next[55]==1);CHECK(world.state.pond.body.position.x==physical.pond.body.position.x);
        CHECK(world.state.pond.eaten==physical.pond.eaten);for(int f=0;f<feeding::capacity;++f) CHECK(world.state.pond.pellets[f].active==physical.pond.pellets[f].active);
    }
}
GPU_TEST_CASE("shelter motor native PPO saves actor critic deviations Adam and worlds exactly") {
    const auto parent=rl::initial_pond_policy();const auto shared=rl::transfer_progress_actor(parent.selector);
    const auto old=nn::orthogonal_params(rl::swimmer_specification(4,arch::kObs),73,{std::sqrt(2.f),std::sqrt(2.f),.01f});
    rl::ShelterMotorTrainer a(16,73);a.actor.set_params(rl::transfer_shelter_motor(old));
    const arch::Context context{{.0075f,0,0},1,{2}};a.env.configure(shared,context,a.stream.value,true);
    auto initial=a.env.host_states(a.stream.value);int cover=0,food=0;
    for(const auto& state:initial) {
        CHECK(!state.state.pond.invalid);
        if(state.request.cover) {++cover;CHECK(predator::body_clearance(state.state.pond.body,state.state.habitat,{2})>=.08f);}
        else ++food;
    }
    CHECK(cover==8);CHECK(food==8);a.update_seconds();const auto saved=rl::shelter_motor_snapshot(a);
    rl::ShelterMotorTrainer b(16,73);b.env.configure(shared,context,b.stream.value,true);rl::restore_shelter_motor(b,saved);
    CHECK(a.actor.get_params()==b.actor.get_params());CHECK(a.critic.get_params()==b.critic.get_params());CHECK(rl::shelter_motor_snapshot(b)==saved);
    a.update_seconds();b.update_seconds();CHECK(rl::shelter_motor_snapshot(a)==rl::shelter_motor_snapshot(b));
    CHECK(a.actor.get_params()!=rl::transfer_shelter_motor(old));CHECK(a.actor_opt.steps()==2*rl::epochs*rl::minibatches);
    for(float v:a.actor.get_params()) CHECK(std::isfinite(v));
    auto corrupt=saved;corrupt[8]^=1;bool rejected=false;try {rl::restore_shelter_motor(b,corrupt);} catch(const std::exception&) {rejected=true;}CHECK(rejected);
}
