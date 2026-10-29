#include "check.h"
#include "demos/pond_reliability.h"
#include "demos/pond_current_schedule.h"
#include "demos/pond_bank_monitor.h"
#include "nn/mlp_cpu.h"
#include <limits>
#include "envs/swimmer/feeding.h"
#include "envs/swimmer/feeding_batch.h"
#include "rl/feeding_policy.h"
#include "rl/swimmer_ppo.h"
#include "rl/feeding_cpu.h"
TEST_CASE("sinking food: added velocity inputs preserve the inherited motor before training") {
    const auto source=cudarl::rl::initial_pond_policy();const auto state=cudarl::deep_pond::initial(7,0);
    float old[33],expanded[cudarl::feeding::kObs]{};cudarl::deep_pond::observe(state,old);
    for(int i=0;i<33;++i) expanded[i]=old[i];expanded[33]=.7f;expanded[34]=-.9f;expanded[35]=.3f;expanded[36]=1;
    cudarl::nn::CpuMlp<float> a(cudarl::rl::swimmer_specification(4,33)),b(cudarl::rl::swimmer_specification(4,cudarl::feeding::kObs));
    a.params()=source.motor;b.params()=cudarl::rl::feeding_motor_parameters(source.motor);a.forward(old,1);b.forward(expanded,1);
    for(int j=0;j<4;++j) CHECK(std::fabs(a.output()[j]-b.output()[j])<1e-7f);
}
TEST_CASE("sinking food: physical settling, drift, bed contact and a many-pellet pool") {
    namespace fd=cudarl::feeding;
    fd::State pond;CHECK(fd::drop(pond,{0,0,0})==0);auto& food=pond.pellets[0];
    const float surface=food.position.y;
    for(int tick=0;tick<200;++tick) fd::move(food,{.005f,0,0});
    CHECK(food.position.y<surface-.1f);CHECK(food.position.x>.8f);CHECK(food.velocity.y<0);
    for(int tick=0;tick<3000;++tick) fd::move(food,{});
    CHECK(std::isfinite(food.position.y));CHECK(food.position.y>=cudarl::deep::pond_floor(food.position.x,food.position.z));
    const float height=food.position.y;for(int tick=0;tick<100;++tick) fd::move(food,{});CHECK(std::fabs(food.position.y-height)<.002f);
    for(int i=1;i<fd::capacity;++i) CHECK(fd::drop(pond,{0,0,0})==i);
    CHECK(fd::count(pond)==fd::capacity);CHECK(fd::drop(pond,{0,0,0})<0);CHECK(fd::drop(pond,{100,0,0})<0);
    CHECK(pond.pellets[0].id==1 && pond.pellets[0].position.y==food.position.y);
}
TEST_CASE("sinking food: generalized scoring preserves the three-food features exactly") {
    namespace fd=cudarl::feeding;const auto old=cudarl::deep_pond::initial(7,0);fd::State state;state.body=old.body;state.target=1;state.since_food=40;
    for(int i=0;i<3;++i) {state.pellets[i].active=true;state.pellets[i].position=old.food[i];}
    float obs[40],reference[34],actual[34];auto sample=old;sample.since_food=40;
    cudarl::progress_selector::observe(sample,1,cudarl::progress_selector::initial_history(sample),obs);
    for(int food=0;food<3;++food) {
        cudarl::progress_selector::features(obs,food,reference);fd::features(state,food,actual);
        for(int i=0;i<34;++i) CHECK(std::fabs(reference[i]-actual[i])<1e-5f);
    }
}
GPU_TEST_CASE("sinking food: CUDA scores candidates beyond three and shares native particle physics") {
    namespace fd=cudarl::feeding;fd::Batch batch(1,7);cudarl::rl::SwimmerStream stream;
    const auto spec=cudarl::rl::progress_selector_spec();std::vector<float> weights(spec.num_params(),0);
    for(int layer=0;layer<spec.num_layers();++layer) weights[spec.weight_offset(layer)]=1;
    batch.configure(weights,{},stream.value);fd::State reference;
    for(int i=0;i<16;++i) fd::drop(reference,{-2.f+i*.25f,0,0});
    batch.set_state(reference,stream.value);reference=batch.state(stream.value);CHECK(reference.target==15);CHECK(fd::count(reference)==16);
    device_ptr<float> actions(4);const std::vector<float> zeros(4,0);actions.copy_from_host(zeros.data(),4);
    for(int tick=0;tick<12;++tick) {fd::advance(reference,zeros.data(),{});batch.step(actions.get(),{},stream.value,false);}
    const auto actual=batch.state(stream.value);CHECK(cudarl::deep::norm(actual.body.position-reference.body.position)<.003f);
    for(int i=0;i<16;++i) CHECK(cudarl::deep::norm(actual.pellets[i].position-reference.pellets[i].position)<1e-5f);
    // Bootstrap the fully selected terminal observation, not a stale target or reset world.
    reference.steps=1199;reference.decision=true;batch.set_state(reference,stream.value);
    device_ptr<float> reward(1),done(1),truncated(1),final_obs(fd::kObs);
    fd::advance(reference,zeros.data(),{});
    batch.step(actions.get(),{reward.get(),done.get(),truncated.get(),final_obs.get()},stream.value,true);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));float expected[fd::kObs],terminal[fd::kObs],ended=0;
    fd::motor_observe(reference,expected);final_obs.copy_to_host(terminal,fd::kObs);done.copy_to_host(&ended,1);
    CHECK(ended==1 && batch.state(stream.value).steps==0);
    for(int i=0;i<fd::kObs;++i) CHECK(std::fabs(terminal[i]-expected[i])<.003f);
}
TEST_CASE("practice feeding: timed arrivals repeat exactly and do not immediately replace eaten food") {
    namespace fd=cudarl::feeding;
    fd::Context context;context.count=1;context.drops[0]={2.f,0,-1.f};
    auto first=fd::initial(771,3,0,context,false),second=first;
    for(auto& pellet:first.pellets) pellet.active=false;
    for(auto& pellet:second.pellets) pellet.active=false;
    const auto body=first.body.position;
    int arrivals=0,previous_tick=-120;
    for(int tick=1;tick<=1200;++tick) {
        first.steps=second.steps=tick;
        fd::replenish(first,context);fd::replenish(second,context);
        CHECK(first.next_id==second.next_id);
        CHECK(fd::count(first)==fd::count(second));
        const int added=fd::count(first);
        CHECK(added<=2);
        if(added) {
            // A seeded arrival is scheduled inside each six-second window.
            CHECK(tick%120>=20 && tick%120<=100);
            CHECK(tick-previous_tick>1);
            previous_tick=tick;arrivals+=added;
        }
        for(int i=0;i<fd::capacity;++i) {
            CHECK(first.pellets[i].active==second.pellets[i].active);
            if(first.pellets[i].active) {
                CHECK(cudarl::deep::norm(first.pellets[i].position-second.pellets[i].position)==0);
                CHECK(fd::valid_drop(first.pellets[i].position));
                CHECK(first.pellets[i].position.y==-.2f);
                first.pellets[i].active=second.pellets[i].active=false;
            }
        }
        CHECK(cudarl::deep::norm(first.body.position-body)==0);
    }
    CHECK(arrivals>=10 && arrivals<=20);
    auto full=fd::initial(771,3,0,context,false);
    for(auto& pellet:full.pellets) pellet.active=false;
    for(int i=0;i<16;++i) CHECK(fd::drop(full,{2.f,0,-1.f})==i);
    const auto before=full;
    for(int tick=1;tick<=120;++tick) {full.steps=tick;fd::replenish(full,context);}
    CHECK(fd::count(full)==16 && full.next_id==before.next_id);
    for(int i=0;i<16;++i) CHECK(full.pellets[i].id==before.pellets[i].id && cudarl::deep::norm(full.pellets[i].position-before.pellets[i].position)==0);
}
TEST_CASE("practice feeding: saved player locations supplement broad automatic coverage") {
    namespace fd=cudarl::feeding;
    fd::Context context;context.count=1;context.drops[0]={0,0,0};
    auto state=fd::initial(901,3,0,context,false);
    int contextual=0,broad=0,outer=0;
    float minimum_x=100,maximum_x=-100,minimum_z=100,maximum_z=-100;
    for(uint32_t event=0;event<1024;++event) {
        const auto point=fd::sample_practice_drop(state,context,event);
        CHECK(fd::valid_drop(point));
        CHECK(cudarl::deep::norm(point-fd::sample_practice_drop(state,context,event))==0);
        if(cudarl::deep::norm(point-context.drops[0])==0) ++contextual;
        else {
            ++broad;if(fd::radial(point)>.8f*.8f) ++outer;
            minimum_x=std::fmin(minimum_x,point.x);maximum_x=std::fmax(maximum_x,point.x);
            minimum_z=std::fmin(minimum_z,point.z);maximum_z=std::fmax(maximum_z,point.z);
        }
    }
    CHECK(contextual>256 && contextual<768);
    CHECK(broad>256 && outer>10);
    CHECK(minimum_x<-2.5f && maximum_x>2.5f && minimum_z<-1.8f && maximum_z>1.8f);
    context.count=0;
    for(uint32_t event=0;event<64;++event) CHECK(fd::valid_drop(fd::sample_practice_drop(state,context,event)));
}
TEST_CASE("practice currents: calm, steady and smoothly fluctuating speeds stay bounded and deterministic") {
    namespace fd=cudarl::feeding;
    fd::Context context;context.current={-.003f,0,.004f};
    int calm_nonretention=0,fluctuating_worlds=0,fluctuating_calm_ticks=0;
    for(uint32_t environment=0;environment<32;++environment) {
        auto state=fd::initial(1801,environment,0,context,environment%4==0);
        auto previous=fd::practice_current(state,context);
        const float initial_speed=cudarl::deep::norm(previous);
        float minimum=initial_speed,maximum=initial_speed;
        if(!state.retention && environment%4==1) ++calm_nonretention;
        for(int tick=0;tick<=1200;++tick) {
            state.steps=tick;
            const auto flow=fd::practice_current(state,context);
            const float speed=cudarl::deep::norm(flow);
            CHECK(std::isfinite(speed) && speed<=.0075001f && flow.y==0);
            CHECK(cudarl::deep::norm(flow-fd::practice_current(state,context))==0);
            CHECK(std::fabs(flow.x*context.current.z-flow.z*context.current.x)<1e-9f);
            if(state.retention || environment%4==1) CHECK(speed==0);
            if(environment%4==2) CHECK(std::fabs(speed-initial_speed)<1e-8f);
            if(environment%4==3) {
                CHECK(cudarl::deep::norm(flow-previous)<=.0002f);
                if(speed==0) ++fluctuating_calm_ticks;
            }
            minimum=std::fmin(minimum,speed);maximum=std::fmax(maximum,speed);previous=flow;
        }
        if(environment%4==3 && maximum-minimum>.0005f) ++fluctuating_worlds;
    }
    CHECK(calm_nonretention==8 && fluctuating_worlds==8 && fluctuating_calm_ticks>0);
    auto retained=fd::initial(1801,0,0,context,true);
    const auto retained_position=retained.body.position;retained.pellets[1].active=false;fd::replenish(retained,context);
    CHECK(fd::count(retained)==3 && !retained.pellets[1].settling);
    CHECK(cudarl::deep::norm(retained.body.position-retained_position)==0 && cudarl::deep::norm(fd::practice_current(retained,context))==0);
    context.current={};auto state=fd::initial(1801,3,0,context,false);
    for(int tick=0;tick<600;++tick) {state.steps=tick;const auto flow=fd::practice_current(state,context);CHECK(flow.x>=0 && flow.y==0 && flow.z==0);}
}
TEST_CASE("practice feeding: CPU optimizer and scheduled worlds continue exactly after a full save") {
    namespace fd=cudarl::feeding;namespace rl=cudarl::rl;
    const auto parent=rl::initial_pond_policy();
    const rl::ProgressPolicy source{0,parent,rl::transfer_progress_actor(parent.selector)};
    fd::Context context;context.count=1;context.drops[0]={2.f,0,-1.f};context.current={.005f,0,0};
    rl::CpuFeedingTrainer uninterrupted(8,source,context);
    for(int i=0;i<2;++i) uninterrupted.update();
    cudarl::checkpoint::Writer saved;uninterrupted.save(saved);
    rl::CpuFeedingTrainer restored(8,source,{});cudarl::checkpoint::Reader reader{saved.data};restored.load(reader);reader.finish();
    for(int i=0;i<2;++i) {uninterrupted.update();restored.update();}
    cudarl::checkpoint::Writer original_state,restored_state;uninterrupted.save(original_state);restored.save(restored_state);
    CHECK(original_state.data==restored_state.data);
    CHECK(uninterrupted.actor.params()!=rl::feeding_motor_parameters(parent.motor));
    CHECK(source.baseline.motor==parent.motor && source.actor==rl::transfer_progress_actor(parent.selector));
}
GPU_TEST_CASE("practice feeding: CUDA and CPU share timed events and checkpointed current phases") {
    namespace fd=cudarl::feeding;
    fd::Context context;context.count=1;context.drops[0]={2.f,0,-1.f};context.current={.006f,0,0};
    const std::vector<float> selector(cudarl::rl::progress_selector_spec().num_params(),0);
    fd::CpuBatch cpu(1,77,selector,context);cpu.states[0]=fd::initial(77,3,0,context,false);cpu.prepare();
    fd::Batch gpu(1,77);cudarl::rl::SwimmerStream stream;gpu.configure(selector,context,stream.value);gpu.set_state(cpu.states[0],stream.value);
    const float zeros[4]={};device_ptr<float> actions(4);actions.copy_from_host(zeros,4);
    for(int tick=0;tick<240;++tick) {
        cpu.step(zeros,nullptr,nullptr,nullptr,false,true);gpu.step(actions.get(),{},stream.value,false,true);
        if(tick%30==0 || tick==239) {
            const auto actual=gpu.state(stream.value);const auto& expected=cpu.states[0];
            CHECK(actual.steps==expected.steps && actual.next_id==expected.next_id && actual.target==expected.target);
            CHECK(cudarl::deep::norm(actual.body.position-expected.body.position)<.003f);
            CHECK(fd::count(actual)==fd::count(expected));
            for(int i=0;i<fd::capacity;++i) if(actual.pellets[i].active || expected.pellets[i].active) {
                CHECK(actual.pellets[i].active==expected.pellets[i].active);
                CHECK(cudarl::deep::norm(actual.pellets[i].position-expected.pellets[i].position)<.0002f);
            }
        }
    }
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
    cudarl::checkpoint::Writer saved;gpu.save(saved);
    fd::Batch restored(1,91);restored.configure(selector,{},stream.value);
    cudarl::checkpoint::Reader reader{saved.data};restored.load(reader);reader.finish();
    for(int tick=0;tick<160;++tick) {
        gpu.step(actions.get(),{},stream.value,false,true);restored.step(actions.get(),{},stream.value,false,true);
    }
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
    cudarl::checkpoint::Writer first,second;gpu.save(first);restored.save(second);CHECK(first.data==second.data);
}
namespace dm=cudarl::demos;
TEST_CASE("changing currents: constant speed, mirrored directions and a physical two-minute period") {
    const dm::PondCurrentSchedule steady({.005f,0,0}),positive({.005f,0,0},120),negative({.005f,0,0},-120);
    for(int tick=0;tick<=6000;tick+=7) {
        const auto a=positive.at_step(tick),b=negative.at_step(tick);
        CHECK(std::fabs(cudarl::deep::norm(a)-.005f)<1e-8f);CHECK(a.y==0);
        CHECK(std::fabs(a.x-b.x)<1e-8f && std::fabs(a.z+b.z)<1e-8f);
        CHECK(steady.at_step(tick).x==.005f && steady.at_step(tick).z==0);
        CHECK(cudarl::deep::norm(positive.at_step(tick+1)-a)<1.4e-5f);
    }
    CHECK(cudarl::deep::norm(positive.at_step(600)-cudarl::deep::V{0,0,.005f})<1e-8f);
    CHECK(cudarl::deep::norm(positive.at_step(1200)-cudarl::deep::V{-.005f,0,0})<1e-8f);
    CHECK(cudarl::deep::norm(positive.at_step(2400)-cudarl::deep::V{.005f,0,0})<1e-8f);
    bool rejected=false;try {dm::PondCurrentSchedule invalid({.005f,0,0},std::numeric_limits<double>::quiet_NaN());} catch(const std::invalid_argument&) {rejected=true;}CHECK(rejected);
}
TEST_CASE("changing currents: bank observations distinguish clearing, feeding and censored visits") {
    auto state=cudarl::deep_pond::initial(7,0);state.eaten=10;state.body=cudarl::deep::BodyState{};
    const cudarl::deep::V bank{3.5f,cudarl::deep::pond_floor(3.5f,0)+.1f,0},free{0,-2,0};
    state.body.position=bank;CHECK(dm::pond_near_bank(state));state.body.position=free;CHECK(!dm::pond_near_bank(state));
    dm::PondBankMonitor monitor;
    for(int tick=0;tick<=50;++tick) {
        state.steps=tick;state.body.position=tick<=30?bank:free;if(tick==40) ++state.eaten;
        monitor.observe(state);
        if(tick==0) CHECK(monitor.visits[0].first_food<0);
    }
    CHECK(monitor.visits.size()==1);CHECK(monitor.contact_steps==30);
    CHECK(monitor.visits[0].start==0 && monitor.visits[0].clear==50);
    CHECK(monitor.visits[0].first_food==40 && monitor.visits[0].slow_steps==30);
    dm::PondBankMonitor unfinished;state.eaten=20;state.body.position=bank;
    for(int tick=0;tick<=10;++tick) {state.steps=tick;unfinished.observe(state);}
    CHECK(unfinished.visits.size()==1 && unfinished.visits[0].clear<0 && unfinished.visits[0].first_food<0);
}
GPU_TEST_CASE("changing currents: live CUDA playback agrees with CPU while flow turns without resets") {
    const auto policy=cudarl::rl::initial_pond_policy();const dm::PondCurrentSchedule schedule({.005f,0,0},120);
    auto cpu=dm::create_pond_playback(policy,"cpu"),gpu=dm::create_pond_playback(policy,"cuda",CUDARL_POND_MODULE);
    cpu->set_food_drift(true);gpu->set_food_drift(true);
    for(int tick=0;tick<40;++tick) {
        const auto before=gpu->frame();const auto current=schedule.at_step(tick);
        cpu->set_current(current);gpu->set_current(current);
        CHECK(gpu->frame().state.steps==before.state.steps);
        CHECK(cudarl::deep::norm(gpu->frame().state.body.position-before.state.body.position)==0);
        const auto& a=cpu->step();const auto& b=gpu->step();
        CHECK(a.selected==b.selected);CHECK(cudarl::deep::norm(a.state.body.position-b.state.body.position)<.003f);
        for(int f=0;f<3;++f) CHECK(cudarl::deep::norm(a.state.food[f]-b.state.food[f])<2e-5f);
    }
}
TEST_CASE("playback reliability: stationary, moving and short pauses remain distinct") {
    dm::PondFrame frame;frame.state=cudarl::deep_pond::initial(7,0);const auto origin=frame.state.body.position;
    dm::PondReliability stopped,moving,brief;
    for(int tick=0;tick<=200;++tick) {
        frame.state.steps=tick;frame.state.since_food=tick;frame.state.body.position=origin;stopped.observe(frame);
        frame.state.body.position=origin+cudarl::deep::V{tick*.02f,0,0};moving.observe(frame);
        const int position=tick<70?0:tick-70;frame.state.body.position=origin+cudarl::deep::V{position*.02f,0,0};brief.observe(frame);
    }
    stopped.finish();moving.finish();brief.finish();CHECK(!stopped.passes());CHECK(stopped.confirmed_events()==1);CHECK(stopped.events.size()==1);
    CHECK(stopped.max_immobility_seconds>=10);CHECK(moving.passes());CHECK(brief.passes());
}
TEST_CASE("playback reliability: invalid food, state and duplicate steps fail visibly") {
    dm::PondFrame frame;frame.state=cudarl::deep_pond::initial(7,0);dm::PondReliability monitor;monitor.observe(frame);monitor.observe(frame);
    frame.state.steps=1;frame.state.food[0].y=1;frame.state.body.position.x=std::numeric_limits<float>::quiet_NaN();monitor.observe(frame);
    CHECK(!monitor.passes());CHECK(monitor.invalid_states==1);CHECK(monitor.invalid_foods==1);CHECK(monitor.nonsequential_steps==1);
}
TEST_CASE("progress playback: transferred selector keeps the legacy CPU trajectory and tracks history") {
    auto original=cudarl::rl::initial_pond_policy();original.foods=0;
    cudarl::rl::ProgressPolicy progress{0,original,cudarl::rl::transfer_progress_actor(original.selector)};
    auto a=dm::create_pond_playback(original,"cpu"),b=dm::create_pond_playback(progress,"cpu");
    for(int tick=1;tick<=80;++tick) {
        const auto& x=a->step();const auto& y=b->step();CHECK(cudarl::deep::norm(x.state.body.position-y.state.body.position)<1e-6f);CHECK(x.selected==y.selected);
        if(tick%original.decision_steps==0) for(int f=0;f<3;++f) {
            CHECK(y.history.placements[f]==y.state.placements[f]);CHECK(std::fabs(y.history.distance[f]-cudarl::deep::norm(y.state.food[f]-y.state.body.position))<1e-6f);
        }
    }
    auto fallback=dm::create_pond_playback(progress,"auto","missing-module");CHECK(std::string(fallback->backend())=="CPU");
    auto bad=progress;bad.actor.resize(3);bool rejected=false;try {dm::create_pond_playback(bad,"cpu");} catch(const std::exception&) {rejected=true;}CHECK(rejected);
}
TEST_CASE("playback currents: switching preserves state and uses the shared water-relative forces") {
    const auto policy=cudarl::rl::initial_pond_policy();auto player=dm::create_pond_playback(policy,"cpu");
    for(int tick=0;tick<23;++tick) player->step();
    const auto before=player->frame();player->set_current({.005f,0,0});
    CHECK(player->frame().state.steps==before.state.steps);CHECK(player->frame().selected==before.selected);
    CHECK(cudarl::deep::norm(player->frame().state.body.position-before.state.body.position)==0);
    for(int f=0;f<3;++f) CHECK(cudarl::deep::norm(player->frame().state.food[f]-before.state.food[f])==0);
    const auto next=player->step();auto expected=before.state;float observation[33];
    cudarl::deep_pond::observe(expected,observation);cudarl::demos::primary_pond_food(observation,next.selected);
    cudarl::nn::CpuMlp<float> motor(cudarl::rl::swimmer_specification(4,33));motor.params()=policy.motor;motor.forward(observation,1);
    cudarl::deep_pond::advance(expected,motor.output(),{.005f,0,0});
    CHECK(cudarl::deep::norm(next.state.body.position-expected.body.position)<1e-7f);
    for(int j=0;j<cudarl::deep::dofs;++j) CHECK(std::fabs(next.state.body.velocity[j]-expected.body.velocity[j])<1e-7f);
    player->set_current({});CHECK(player->frame().state.steps==next.state.steps);CHECK(cudarl::deep::norm(player->current())==0);
    const cudarl::deep::V invalid[]={{.1f,0,0},{0,.001f,0},{std::numeric_limits<float>::quiet_NaN(),0,0}};
    for(auto value:invalid) {bool rejected=false;try {player->set_current(value);} catch(const std::invalid_argument&) {rejected=true;}CHECK(rejected);CHECK(cudarl::deep::norm(player->current())==0);}
}
GPU_TEST_CASE("playback currents: optional CUDA module preserves continuity and agrees with CPU after a live switch") {
    const auto policy=cudarl::rl::initial_pond_policy();
    auto cpu=dm::create_pond_playback(policy,"cpu"),gpu=dm::create_pond_playback(policy,"cuda",CUDARL_POND_MODULE);
    CHECK(std::string(gpu->backend())=="CUDA");
    for(int tick=0;tick<50;++tick) {
        if(tick==13 || tick==34) {
            const auto before=gpu->frame();const cudarl::deep::V flow=tick==13?cudarl::deep::V{.005f,0,0}:cudarl::deep::V{};
            cpu->set_current(flow);gpu->set_current(flow);
            CHECK(gpu->frame().state.steps==before.state.steps);CHECK(gpu->frame().selected==before.selected);
            CHECK(cudarl::deep::norm(gpu->frame().state.body.position-before.state.body.position)==0);
        }
        const auto& a=cpu->step();const auto& b=gpu->step();
        CHECK(cudarl::deep::norm(a.state.body.position-b.state.body.position)<.003f);CHECK(a.selected==b.selected);
        CHECK(cudarl::deep::norm(cpu->current()-gpu->current())==0);
    }
}
TEST_CASE("playback food drift: opt-in preserves state, advances targets and switches back without resets") {
    const auto policy=cudarl::rl::initial_pond_policy();auto player=dm::create_pond_playback(policy,"cpu");
    const auto before=player->frame();player->set_food_drift(true);player->set_current({.005f,0,0});
    CHECK(player->frame().food_drift);CHECK(player->frame().state.steps==before.state.steps);
    for(int f=0;f<3;++f) CHECK(cudarl::deep::norm(player->frame().state.food[f]-before.state.food[f])==0);
    const auto next=player->step();
    for(int f=0;f<3;++f) {CHECK(cudarl::deep::norm(next.state.food[f]-before.state.food[f])>0);CHECK(cudarl::deep_pond::water(next.state.food[f]));}
    player->set_current({});const auto calm=player->frame();player->step();
    for(int f=0;f<3;++f) CHECK(cudarl::deep::norm(player->frame().state.food[f]-calm.state.food[f])==0);
    player->set_food_drift(false);CHECK(!player->frame().food_drift);
}
GPU_TEST_CASE("playback food drift: CUDA and CPU agree on opt-in targets and live toggles") {
    const auto policy=cudarl::rl::initial_pond_policy();
    auto cpu=dm::create_pond_playback(policy,"cpu"),gpu=dm::create_pond_playback(policy,"cuda",CUDARL_POND_MODULE);
    cpu->set_food_drift(true);gpu->set_food_drift(true);
    for(int tick=0;tick<80;++tick) {
        if(tick==0 || tick==40) {const cudarl::deep::V flow=tick==0?cudarl::deep::V{.005f,0,0}:cudarl::deep::V{};cpu->set_current(flow);gpu->set_current(flow);}
        const auto& a=cpu->step();const auto& b=gpu->step();
        CHECK(a.food_drift && b.food_drift);CHECK(a.selected==b.selected);
        CHECK(cudarl::deep::norm(a.state.body.position-b.state.body.position)<.003f);
        for(int f=0;f<3;++f) CHECK(cudarl::deep::norm(a.state.food[f]-b.state.food[f])<2e-5f);
    }
}
