#include "check.h"
#include "nn/mlp_cpu.h"
#include "rl/shelter_motor_ppo.h"
#include "rl/shelter_ppo.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
namespace sh=cudarl::shelter;
namespace pr=cudarl::predator;
namespace ar=cudarl::arch;
namespace dp=cudarl::deep;
namespace cp=cudarl::checkpoint;
namespace rl=cudarl::rl;
namespace nn=cudarl::nn;

ar::Context shelter_test_context() {return {{.005f,0,0},3,{2}};}
std::vector<float> shelter_test_source() {return nn::init_params(rl::progress_selector_spec(),71);}
std::vector<float> shelter_test_motor() {
    return rl::transfer_shelter_motor(nn::init_params(rl::swimmer_specification(4,ar::kObs),7));
}
template<class F> bool shelter_rejected(F&& action) {
    try {action();} catch(const std::exception&) {return true;}return false;
}
template<class T> void shelter_overwrite(std::vector<uint8_t>& bytes,size_t offset,const T& value) {
    if(offset>bytes.size() || sizeof(value)>bytes.size()-offset) throw std::runtime_error("Invalid test checkpoint offset");
    std::memcpy(bytes.data()+offset,&value,sizeof(value));
}
struct ShelterSnapshotOffsets {
    size_t update=0,actor=0,actor_optimizer_steps=0,actor_first_moment=0,world=0;
};
// Decode a valid snapshot rather than guessing network sizes or byte offsets.
ShelterSnapshotOffsets shelter_snapshot_offsets(const std::vector<uint8_t>& bytes) {
    cp::Reader in{bytes};ShelterSnapshotOffsets result;
    for(int i=0;i<3;++i) (void)in.value<uint32_t>();
    (void)in.value<int>();(void)in.value<uint32_t>();
    result.update=in.position;(void)in.value<int>();(void)in.value<ar::Context>();
    for(int i=0;i<3;++i) (void)in.array<float>(); // protocol, original scorer, frozen body
    result.actor=in.position+sizeof(uint64_t);(void)in.array<float>();(void)in.array<float>();
    (void)in.value<nn::AdamConfig>();result.actor_optimizer_steps=in.position;(void)in.value<int>();
    result.actor_first_moment=in.position+sizeof(uint64_t);(void)in.array<float>();(void)in.array<float>();
    (void)in.value<nn::AdamConfig>();(void)in.value<int>();(void)in.array<float>();(void)in.array<float>();
    (void)in.value<ar::Context>();result.world=in.position+sizeof(uint64_t);(void)in.array<sh::World>();
    in.finish();return result;
}
}

TEST_CASE("shelter selector: expanded scorer preserves both real foods through threat cues and slot swaps") {
    const auto source=shelter_test_source();nn::CpuMlp<float> before(rl::progress_selector_spec()),after(rl::shelter_selector_spec());
    before.params()=source;after.params()=rl::transfer_shelter_actor(source);
    for(uint32_t trial=0;trial<24;++trial) {
        sh::World world;world.state=ar::initial(71,trial,0,false,{2});world.predator.seed=trial+1;
        world.predator.manual_age=trial%3==0?2.f:(trial%3==1?12.f:30.f);
        world.state.pond.target=-1;world.state.pond.since_food=static_cast<int>(trial)*13;
        world.state.pond.pellets[0].progress=.4f;world.state.pond.pellets[1].progress=-.3f;
        world.exposure=pr::exposure(world.state.pond.body,sh::event(world).eye,world.state.habitat,{2});
        float observation[sh::kObs]{};sh::observe(world,observation,{2});float scores[sh::kFoods]{};
        for(int food=0;food<sh::kFoods;++food) {
            CHECK(observation[sh::kChoices*sh::kFeatures+food]==1.f);
            float inherited[34]{};cudarl::feeding::features(world.state.pond,food,inherited);
            for(int k=0;k<34;++k) CHECK(observation[food*sh::kFeatures+k]==inherited[k]);
            before.forward(inherited,1);after.forward(observation+food*sh::kFeatures,1);
            scores[food]=after.output()[0];CHECK(std::fabs(before.output()[0]-scores[food])<1e-6f);
        }
        std::swap(world.state.pond.pellets[0],world.state.pond.pellets[1]);
        sh::observe(world,observation,{2});
        for(int food=0;food<sh::kFoods;++food) {
            after.forward(observation+food*sh::kFeatures,1);
            CHECK(std::fabs(after.output()[0]-scores[1-food])<1e-6f);
        }
    }
    auto invalid=source;invalid[0]=std::numeric_limits<float>::quiet_NaN();
    CHECK(shelter_rejected([&] {rl::transfer_shelter_actor(invalid);}));
    invalid=source;invalid.pop_back();CHECK(shelter_rejected([&] {rl::transfer_shelter_actor(invalid);}));
}

TEST_CASE("shelter selector: expanded motor retains inherited torque outputs with new hold observations") {
    const auto source=nn::init_params(rl::swimmer_specification(4,ar::kObs),7);
    nn::CpuMlp<float> before(rl::swimmer_specification(4,ar::kObs));
    nn::CpuMlp<float> after(rl::swimmer_specification(4,cudarl::shelter_motor::kObs));
    before.params()=source;after.params()=rl::transfer_shelter_motor(source);
    for(uint32_t trial=0;trial<24;++trial) {
        auto state=ar::initial(93,trial,0,false,{2});state.pond.target=static_cast<int>(trial%2);
        const auto location=pr::cover(static_cast<int>(trial%pr::kCoverCount),{-8,4,-1.25f},state.habitat,{2});
        sh::Goal goal;goal.position=location.position;goal.cover=static_cast<int>(trial%pr::kCoverCount);goal.valid=location.valid;
        if(trial%2==0) {goal.cover=-1;goal.food=state.pond.target;goal.position=state.pond.pellets[goal.food].position;goal.valid=true;}
        float old_obs[ar::kObs]{},next_obs[cudarl::shelter_motor::kObs]{};
        sh::motor_observe(state,goal,old_obs,{2});
        sh::motor_observe59(state,goal,{0,0,1},next_obs,{2});
        for(int k=0;k<ar::kObs;++k) CHECK(old_obs[k]==next_obs[k]);
        CHECK(next_obs[55]==(goal.cover>=0 && goal.valid?1.f:0.f));
        before.forward(old_obs,1);after.forward(next_obs,1);
        for(int action=0;action<4;++action) CHECK(std::fabs(before.output()[action]-after.output()[action])<1e-6f);
    }
}

GPU_TEST_CASE("shelter selector: categorical sampling respects replayed food and cover masks") {
    constexpr int count=32;std::vector<float> observation(count*sh::kObs),logits(count*sh::kChoices),copied(count*sh::kChoices*sh::kFeatures);
    for(int row=0;row<count;++row) for(int choice=0;choice<sh::kChoices;++choice) {
        const bool valid=choice==row%sh::kChoices || (row%3==0 && choice==(row+1)%sh::kChoices);
        observation[row*sh::kObs+sh::kChoices*sh::kFeatures+choice]=valid?1.f:0.f;
        // Invalid options deliberately begin with the largest logit.
        logits[row*sh::kChoices+choice]=valid?float(choice)*.3f:1000.f;
        for(int k=0;k<sh::kFeatures;++k) observation[row*sh::kObs+choice*sh::kFeatures+k]=float(row*1000+choice*100+k)*.001f;
    }
    device_ptr<float> obs(observation.size()),scores(logits.size()),features(copied.size()),logp(count);device_ptr<int> actions(count);
    obs.copy_from_host(observation.data(),observation.size());scores.copy_from_host(logits.data(),logits.size());
    sh::make_features(obs.get(),features.get(),count,nullptr);sh::mask_logits(obs.get(),scores.get(),count,nullptr);
    features.copy_to_host(copied.data(),copied.size());std::vector<float> masked(logits.size());scores.copy_to_host(masked.data(),masked.size());
    for(int row=0;row<count;++row) for(int choice=0;choice<sh::kChoices;++choice) {
        const bool valid=observation[row*sh::kObs+sh::kChoices*sh::kFeatures+choice]>.5f;
        CHECK(masked[row*sh::kChoices+choice]==(valid?logits[row*sh::kChoices+choice]:-1e9f));
        for(int k=0;k<sh::kFeatures;++k) CHECK(copied[(row*sh::kChoices+choice)*sh::kFeatures+k]==observation[row*sh::kObs+choice*sh::kFeatures+k]);
    }
    std::vector<int> selected(count);std::vector<float> probabilities(count);
    for(uint32_t draw=0;draw<32;++draw) {
        rl::sample_actions(scores.get(),count,sh::kChoices,71,draw,draw==31,actions.get(),logp.get(),nullptr);
        actions.copy_to_host(selected.data(),selected.size());logp.copy_to_host(probabilities.data(),probabilities.size());
        for(int row=0;row<count;++row) {
            CHECK(selected[row]>=0 && selected[row]<sh::kChoices);
            if(selected[row]>=0 && selected[row]<sh::kChoices)
                CHECK(observation[row*sh::kObs+sh::kChoices*sh::kFeatures+selected[row]]==1.f);
            CHECK(std::isfinite(probabilities[row]) && probabilities[row]<=0.f);
        }
    }
}

GPU_TEST_CASE("shelter selector: cover goals never create food and many-pellet imports are rejected explicitly") {
    const auto context=shelter_test_context();rl::SwimmerStream stream;
    sh::Batch batch(1,71,context,sh::kPeriod,{false,true,true,true});const auto before=batch.state(stream.value);
    CHECK(cudarl::feeding::count(before.state.pond)==sh::kFoods);
    int selected=-1;for(int choice=sh::kFoods;choice<sh::kChoices;++choice) if(sh::option(before,choice,context.basin).valid) {selected=choice;break;}
    CHECK(selected>=sh::kFoods);if(selected<sh::kFoods) return;
    device_ptr<int> action(1);action.copy_from_host(&selected,1);batch.begin(action.get(),{},stream.value);
    const auto after=batch.state(stream.value);
    CHECK(after.goal.valid && after.goal.cover==selected-sh::kFoods && after.goal.food==-1);
    CHECK(after.state.pond.target==-1 && after.state.pond.eaten==before.state.pond.eaten);
    CHECK(after.state.pond.next_id==before.state.pond.next_id);
    CHECK(std::memcmp(after.state.pond.pellets,before.state.pond.pellets,sizeof(before.state.pond.pellets))==0);
    CHECK(std::memcmp(&after.state.pond.body,&before.state.pond.body,sizeof(before.state.pond.body))==0);
    std::vector<float> observed(cudarl::shelter_motor::kObs);CUDA_CHECK_THROW(cudaMemcpyAsync(observed.data(),batch.motor_obs(stream.value),observed.size()*sizeof(float),cudaMemcpyDeviceToHost,stream.value));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
    const auto cover=pr::cover(after.goal.cover,sh::event(after).eye,after.state.habitat,context.basin);
    const auto heading=dp::axis_rotation({0,1,0},cover.yaw).rotate({1,0,0});
    float reference[cudarl::shelter_motor::kObs]{};sh::motor_observe59(after.state,after.goal,heading,reference,context.basin);
    for(size_t k=0;k<observed.size();++k) CHECK(std::fabs(observed[k]-reference[k])<1e-5f);
    const auto after_motor_inputs=batch.state(stream.value);
    CHECK(std::memcmp(after_motor_inputs.state.pond.pellets,before.state.pond.pellets,sizeof(before.state.pond.pellets))==0);
    CHECK(after_motor_inputs.state.pond.eaten==before.state.pond.eaten && after_motor_inputs.state.pond.next_id==before.state.pond.next_id);
    // This is a bounded two-food research task, not a new limit on the player
    // feeding pond. Never discard an imported food merely because its slot is high.
    auto many=after;many.state.pond.pellets[cudarl::feeding::capacity-1]=many.state.pond.pellets[0];
    CHECK(shelter_rejected([&] {batch.set_state(many,stream.value);}));
    const auto unchanged=batch.state(stream.value);
    CHECK(std::memcmp(unchanged.state.pond.pellets,after.state.pond.pellets,sizeof(after.state.pond.pellets))==0);
}

GPU_TEST_CASE("shelter selector: PPO changes weights and resumes actor optimizer and all worlds through timeout") {
    const auto source=shelter_test_source(),motor=shelter_test_motor();const auto context=shelter_test_context();
    rl::ShelterTrainer a(source,motor,16,71,context);auto starts=a.env.states(a.stream.value);
    for(auto& world:starts) {world.state.pond.steps=ar::kEpisodeSteps-sh::kPeriod;world.predator.elapsed=8.f;}
    cp::Writer fixture;fixture.value(context);fixture.array(starts);cp::Reader load{fixture.data};a.env.load(load);load.finish();
    const auto actor_before=a.actor.get_params(),critic_before=a.critic.get_params();a.train_update();
    CHECK(a.update==1);CHECK(a.actor.get_params()!=actor_before);CHECK(a.critic.get_params()!=critic_before);
    CHECK(a.actor_opt.steps()==rl::ShelterTrainer::epochs*rl::ShelterTrainer::minibatches);
    CHECK(a.critic_opt.steps()==a.actor_opt.steps());CHECK(a.motor.get_params()==motor);
    for(float value:a.policy()) CHECK(std::isfinite(value));
    for(float value:a.critic.get_params()) CHECK(std::isfinite(value));
    std::vector<float> rolled_observations(static_cast<size_t>(a.batch)*sh::kObs),rewards(a.batch);std::vector<int> decisions(a.batch);
    a.obs.copy_to_host(rolled_observations.data(),rolled_observations.size());a.actions.copy_to_host(decisions.data(),decisions.size());a.rewards.copy_to_host(rewards.data(),rewards.size());
    for(int row=0;row<a.batch;++row) {
        CHECK(std::isfinite(rewards[row]));CHECK(decisions[row]>=0 && decisions[row]<sh::kChoices);
        if(decisions[row]>=0 && decisions[row]<sh::kChoices)
            CHECK(rolled_observations[row*sh::kObs+sh::kChoices*sh::kFeatures+decisions[row]]==1.f);
    }
    auto worlds=a.env.states(a.stream.value);bool practiced_threat=false;
    for(const auto& world:worlds) {CHECK(world.state.pond.episode>=1);CHECK(world.state.pond.steps%sh::kPeriod==0);practiced_threat|=world.metrics.threat_seconds>0;}
    CHECK(practiced_threat);
    const auto saved=a.snapshot();rl::ShelterTrainer b(source,motor,16,71,context);b.restore(saved);
    CHECK(b.snapshot()==saved);CHECK(a.actor.get_params()==b.actor.get_params());CHECK(a.critic.get_params()==b.critic.get_params());
    CHECK(a.actor_opt.steps()==b.actor_opt.steps());CHECK(a.critic_opt.steps()==b.critic_opt.steps());
    const auto restored=b.env.states(b.stream.value);CHECK(worlds.size()==restored.size());
    CHECK(std::memcmp(worlds.data(),restored.data(),worlds.size()*sizeof(sh::World))==0);
    // These are random-initialization continuation checks, not evidence that
    // sheltering has been learned or that this actor is suitable for deployment.
    a.train_update();b.train_update();CHECK(a.snapshot()==b.snapshot());
    CHECK(a.motor.get_params()==motor && b.motor.get_params()==motor);
}

GPU_TEST_CASE("shelter selector: malformed full checkpoints and changed sources fail closed") {
    const auto source=shelter_test_source(),motor=shelter_test_motor();const auto context=shelter_test_context();
    rl::ShelterTrainer original(source,motor,16,71,context);const auto saved=original.snapshot();const auto offsets=shelter_snapshot_offsets(saved);
    rl::ShelterTrainer target(source,motor,16,71,context);
    auto reject=[&](const std::vector<uint8_t>& bytes) {CHECK(shelter_rejected([&] {target.restore(bytes);}));target.restore(saved);};
    auto damaged=saved;damaged.pop_back();reject(damaged);
    damaged=saved;damaged[0]^=0x80;reject(damaged);
    damaged=saved;damaged.push_back(0);reject(damaged);
    damaged=saved;shelter_overwrite(damaged,offsets.update,-1);reject(damaged);
    damaged=saved;shelter_overwrite(damaged,offsets.actor,std::numeric_limits<float>::quiet_NaN());reject(damaged);
    damaged=saved;shelter_overwrite(damaged,offsets.actor_first_moment,std::numeric_limits<float>::infinity());reject(damaged);
    damaged=saved;shelter_overwrite(damaged,offsets.actor_optimizer_steps,1);reject(damaged);
    sh::World world;std::memcpy(&world,saved.data()+offsets.world,sizeof(world));
    auto invalid=world;invalid.state.pond.body.position.x=std::numeric_limits<float>::quiet_NaN();
    damaged=saved;shelter_overwrite(damaged,offsets.world,invalid);reject(damaged);
    invalid=world;invalid.predator.elapsed=std::numeric_limits<float>::infinity();
    damaged=saved;shelter_overwrite(damaged,offsets.world,invalid);reject(damaged);
    invalid=world;invalid.state.pond.steps=1;
    damaged=saved;shelter_overwrite(damaged,offsets.world,invalid);reject(damaged);
    auto changed_source=source;changed_source[0]+=.01f;rl::ShelterTrainer wrong_source(changed_source,motor,16,71,context);
    CHECK(shelter_rejected([&] {wrong_source.restore(saved);}));
    auto changed_motor=motor;changed_motor[0]+=.01f;rl::ShelterTrainer wrong_motor(source,changed_motor,16,71,context);
    CHECK(shelter_rejected([&] {wrong_motor.restore(saved);}));
    auto changed_context=context;++changed_context.version;rl::ShelterTrainer wrong_context(source,motor,16,71,changed_context);
    CHECK(shelter_rejected([&] {wrong_context.restore(saved);}));
    rl::ShelterTrainer wrong_seed(source,motor,16,72,context);CHECK(shelter_rejected([&] {wrong_seed.restore(saved);}));
    CHECK(original.snapshot()==saved);
}
