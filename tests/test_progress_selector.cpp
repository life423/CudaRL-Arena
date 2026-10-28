#include "check.h"
#include "rl/progress_selector_ppo.h"
#include "nn/mlp_cpu.h"
#include "rl/pond_recovery_pool.h"
namespace pg=cudarl::progress_selector;
namespace p=cudarl::deep_pond;
namespace rl=cudarl::rl;
TEST_CASE("progress selector: zero-column transfer preserves learned scores and progress is food-specific") {
    auto parent=rl::initial_pond_policy();parent.foods=0;
    cudarl::nn::CpuMlp<float> old(rl::selector_spec()),next(rl::progress_selector_spec());
    old.params()=parent.selector;next.params()=rl::transfer_progress_actor(parent.selector);
    for(uint32_t trial=0;trial<32;++trial) {
        auto state=p::initial(71,trial);auto history=pg::initial_history(state);history.progress[0]=.4f;history.progress[1]=-.8f;state.since_food=200;
        float obs[40];pg::observe(state,1,history,obs);CHECK(obs[36]==.5f);
        for(int f=0;f<3;++f) {
            float a[32],b[34];cudarl::pond_selector::features(obs,f,a);pg::features(obs,f,b);
            CHECK(b[32]==.5f);CHECK(b[33]==history.progress[f]);old.forward(a,1);next.forward(b,1);
            CHECK(std::fabs(old.output()[0]-next.output()[0])<1e-6f);
        }
        auto before=history;state.food[0].x+=1;++state.placements[0];pg::update_history(state,history,20);
        CHECK(history.progress[0]==0);CHECK(history.placements[0]!=before.placements[0]);
    }
    int calm=0,quadrant[4]{};
    for(uint32_t i=0;i<1000;++i) {
        const auto current=pg::sample_current(71,i,0,.01f);const auto speed=cudarl::deep::norm(current);
        CHECK(speed<=.010001f);CHECK(current.y==0);if(speed==0) ++calm;else ++quadrant[(current.x<0?1:0)+(current.z<0?2:0)];
    }
    CHECK(calm>200 && calm<300);for(int n:quadrant) CHECK(n>100);
}
GPU_TEST_CASE("progress selector: finite PPO, frozen motor and exact full-state continuation") {
    auto parent=rl::initial_pond_policy();parent.foods=0;
    rl::ProgressTrainer a(parent,16,71);a.train_update();const auto saved=a.snapshot();
    CHECK(a.motor.get_params()==parent.motor);CHECK(a.actor.get_params()!=rl::transfer_progress_actor(parent.selector));
    for(float x:a.actor.get_params()) CHECK(std::isfinite(x));
    rl::ProgressTrainer b(parent,16,71);b.restore(saved);CHECK(a.actor.get_params()==b.actor.get_params());
    int phases=0;rl::ProgressInstrumentation instrument{[](void* count,int phase) {auto& value=*static_cast<int*>(count);CHECK(phase==value);++value;},&phases};
    a.train_update();b.train_update(&instrument);CHECK(phases==5);CHECK(a.actor.get_params()==b.actor.get_params());CHECK(a.critic.get_params()==b.critic.get_params());
    CHECK(a.snapshot()==b.snapshot());CHECK(a.motor.get_params()==parent.motor);
    cudarl::checkpoint::Writer out;a.policy().write(out);cudarl::checkpoint::Reader in{out.data};const auto restored=rl::ProgressPolicy::read(in);in.finish();
    CHECK(restored.actor==a.actor.get_params());CHECK(restored.baseline.motor==parent.motor);
}
GPU_TEST_CASE("progress selector: bank curriculum preserves underwater food and exact resumable learning") {
    auto parent=rl::initial_pond_policy();parent.foods=0;rl::ProgressTrainer a(parent,16,71);a.train_update();
    pg::RecoveryStart start;start.state=p::initial(81,4);start.state.since_food=500;start.target=1;start.history=pg::initial_history(start.state);start.current={.005f,0,0};
    const auto actor=a.actor.get_params(),critic=a.critic.get_params();a.set_recovery_starts({start});
    CHECK(a.actor.get_params()==actor);CHECK(a.critic.get_params()==critic);CHECK(a.motor.get_params()==parent.motor);
    const auto save=a.snapshot();rl::ProgressTrainer b(parent,16,71);b.restore(save);CHECK(b.recovery.size()==1);CHECK(b.snapshot()==save);
    a.train_update();b.train_update();CHECK(a.snapshot()==b.snapshot());CHECK(a.motor.get_params()==parent.motor);
    auto bad=start;bad.state.food[0].y=1;bool rejected=false;try {a.set_recovery_starts({bad});} catch(const std::exception&) {rejected=true;}CHECK(rejected);
    rejected=false;try {rl::validate_recovery(bad);} catch(const std::exception&) {rejected=true;}CHECK(rejected);
    cudarl::checkpoint::Writer world;a.env.save(world);cudarl::checkpoint::Reader reader{world.data};const auto states=reader.array<p::State>();
    for(const auto& state:states) for(auto food:state.food) CHECK(p::water(food));
}
