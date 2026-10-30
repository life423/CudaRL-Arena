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
TEST_CASE("shelter motor parameter reconditioning preserves physical food and initial cover commands") {
    const auto spec=rl::swimmer_specification(4,arch::kObs);
    auto parent=nn::orthogonal_params(spec,73,{std::sqrt(2.f),std::sqrt(2.f),.01f});
    // Exercise a complete inherited parameter vector with the large final-row
    // gain seen in the protected swimmer, rather than only a scalar identity.
    const int final=spec.num_layers()-1;
    for(int column=0;column<spec.in(final);++column) parent[static_cast<size_t>(spec.weight_offset(final)+2*spec.in(final)+column)]*=2048.f;
    parent[static_cast<size_t>(spec.bias_offset(final)+2)]=128.f;
    shelter_motor::ActuatorScale scale;scale.value[0]=2;scale.value[1]=1;scale.value[2]=2048;scale.value[3]=4;
    const auto legacy=rl::transfer_shelter_motor(parent),conditioned=rl::normalized_shelter_actor(legacy,scale);
    nn::CpuMlp<float> before(rl::swimmer_specification(4,shelter_motor::kObs)),after(rl::swimmer_specification(4,shelter_motor::kObs));before.params()=legacy;after.params()=conditioned;
    for(uint32_t i=0;i<32;++i) {
        shelter_motor::State world;world.state=arch::initial(73,i,0,false,{2});world.state.pond.target=0;
        float input[shelter_motor::kObs];shelter_motor::observe(world,input,{2});before.forward(input,1);after.forward(input,1);
        for(int j=0;j<4;++j) CHECK(shelter_motor::action_mean(after.output()[j],false,j,scale)==before.output()[j]);
        const int goal=static_cast<int>(i%4);const auto point=predator::cover(goal,world.request.eye,world.state.habitat,{2});
        world.request.cover=true;world.request.goal={point.position,{},-1,goal,point.valid};world.state.pond.target=-1;
        world.request.heading=deep::axis_rotation({0,1,0},point.yaw).rotate({1,0,0});
        shelter_motor::observe(world,input,{2});before.forward(input,1);after.forward(input,1);
        for(int j=0;j<4;++j) {
            CHECK(after.output()[j]==before.output()[j]/scale.value[j]);
            CHECK(shelter_motor::action_mean(after.output()[j],true,j,scale)==before.output()[j]/scale.value[j]);
        }
    }
}
GPU_TEST_CASE("shelter motor native PPO saves actor critic deviations Adam and worlds exactly") {
    const auto parent=rl::initial_pond_policy();const auto shared=rl::transfer_progress_actor(parent.selector);
    const auto old=nn::orthogonal_params(rl::swimmer_specification(4,arch::kObs),73,{std::sqrt(2.f),std::sqrt(2.f),.01f});
    shelter_motor::ActuatorScale scale;scale.value[0]=2;scale.value[1]=1;scale.value[2]=128;scale.value[3]=4;
    const auto initial_actor=rl::normalized_shelter_actor(rl::transfer_shelter_motor(old),scale);
    rl::ShelterMotorTrainer a(16,73);a.actor.set_params(initial_actor);a.env.set_action_scales(scale);
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
    for(int j=0;j<4;++j) CHECK(b.env.action_scales().value[j]==scale.value[j]);
    a.update_seconds();b.update_seconds();CHECK(rl::shelter_motor_snapshot(a)==rl::shelter_motor_snapshot(b));
    CHECK(a.actor.get_params()!=initial_actor);CHECK(a.actor_opt.steps()==2*rl::epochs*rl::minibatches);
    for(float v:a.actor.get_params()) CHECK(std::isfinite(v));
    auto corrupt=saved;corrupt[8]^=1;bool rejected=false;try {rl::restore_shelter_motor(b,corrupt);} catch(const std::exception&) {rejected=true;}CHECK(rejected);
}
