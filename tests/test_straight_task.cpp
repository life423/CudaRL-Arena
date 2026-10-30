#include "check.h"
#include "motor/straight_task.h"
#include "rl/ppo_math.h"
namespace st=cudarl::straight_motor;namespace p=cudarl::physics;
TEST_CASE("straight_reward2: interval40 is warmup and41 is scored, effort always charged (zero integration)") {
    st::World world;st::ForwardConfig config;st::ForwardTask task;CHECK(st::begin_forward(world,config,task));
    task.controls=39;task.last_world_control=39;world.physical.control_steps=40;world.physical.mean_squared_torque=.5f;
    const auto warmup=st::record_control(task,world,config);CHECK(warmup.accepted&&task.steady_samples==0);CHECK(std::fabs(warmup.reward+.00005f)<1e-8f);
    world.physical.control_steps=41;const auto scored=st::record_control(task,world,config);CHECK(scored.accepted&&task.steady_samples==1);CHECK(std::fabs(scored.reward-warmup.reward+.05f)<1e-7f);
    CHECK(world.attempted_substeps==0);
}
TEST_CASE("straight_reward2: warmup safety and work are not discarded (zero integration)") {
    st::World bad;st::ForwardConfig config;st::ForwardTask rejected;CHECK(st::begin_forward(bad,config,rejected));bad.physical.invalid=true;bad.physical.control_steps=1;
    CHECK(!st::record_control(rejected,bad,config).accepted);
    st::World world;st::ForwardTask task;CHECK(st::begin_forward(world,config,task));world.physical.body.velocity[0]=config.target_speed_m_s;
    world.physical.body.work=.0001f; // already spent in warmup; must count at acceptance
    for(int n=1;n<=80;++n){world.physical.body.position.x=config.target_speed_m_s*p::dt*float(n)/p::scale;world.physical.control_steps=n;CHECK(st::record_control(task,world,config).accepted);}
    CHECK(task.outcome==st::Outcome::Timeout);CHECK(world.attempted_substeps==0&&bad.attempted_substeps==0);
}
TEST_CASE("straight_reward2: later scored costs retain full-return credit to warmup (zero integration)") {
    float rewards[80]{},values[80]{},done[80]{},zero[80]{},advantage[80]{},returns[80]{};const float last[1]{99};
    for(int t=40;t<80;++t)rewards[t]=-.01f;done[79]=1;
    cudarl::rl::gae_one_env(80,1,0,rewards,values,done,zero,zero,last,.99f,1.f,advantage,returns);
    CHECK(rewards[0]==0&&advantage[0]<-.1f);CHECK(returns[39]<0&&std::fabs(returns[39]-.99f*returns[40])<1e-6f);
}
TEST_CASE("straight_learning: full-episode lambda1 returns ignore bad terminal bootstrap (zero integration)") {
    float rewards[80]{},values[80]{},dones[80]{},zeros[80]{},advantage[80]{},returns[80]{};const float last[1]{999};
    for(int t=0;t<80;++t){rewards[t]=t<20?-.05f:-.002f;values[t]=.3f*std::cos(float(t));}dones[79]=1;
    cudarl::rl::gae_one_env(80,1,0,rewards,values,dones,zeros,zeros,last,.99f,1.f,advantage,returns);
    double reference=0;for(int t=79;t>=0;--t){reference=rewards[t]+double(.99f)*reference;CHECK(std::fabs(returns[t]-reference)<2e-6);CHECK(std::fabs(advantage[t]+values[t]-reference)<2e-6);}
}
TEST_CASE("straight_task4: forward travel beyond old destination has no arrival penalty (zero integration)") {
    st::World world;st::ForwardConfig config;st::ForwardTask task;CHECK(st::begin_forward(world,config,task));world.physical.body.velocity[0]=config.target_speed_m_s;
    auto near=world,far=world;auto near_task=task,far_task=task;near.physical.body.position.x+=.020f/p::scale;far.physical.body.position.x+=.040f/p::scale;
    near.physical.control_steps=far.physical.control_steps=1;const auto a=st::record_control(near_task,near,config),b=st::record_control(far_task,far,config);
    CHECK(a.accepted&&b.accepted&&!a.done&&!b.done);CHECK(std::fabs(a.reward-b.reward)<1e-7f);CHECK(world.attempted_substeps==0);
}
TEST_CASE("straight_task4: fixed window accepts straight speed and reports head yaw independently (zero integration)") {
    st::World world;st::ForwardConfig config;st::ForwardTask task;CHECK(st::begin_forward(world,config,task));const auto start=task.start_com_model;
    world.physical.body.orientation=p::axis_rotation({0,1,0},.4f);world.physical.body.velocity[0]=config.target_speed_m_s;st::Transition transition;
    for(int n=1;n<=80;++n){const auto now=cudarl::motor::motion(world.physical.body).com_model;const auto desired=start+p::V{config.target_speed_m_s*p::dt*float(n)/p::scale,0,0};world.physical.body.position+=desired-now;world.physical.control_steps=n;transition=st::record_control(task,world,config);CHECK(transition.accepted);}
    CHECK(transition.done&&transition.outcome==st::Outcome::Success);CHECK(task.maximum_head_yaw>20);CHECK(std::fabs(task.steady_speed_m_s-.008f)<1e-6f);CHECK(world.attempted_substeps==0);
}
TEST_CASE("straight_task4: cross-track drift fails despite correct forward speed (zero integration)") {
    st::World world;st::ForwardConfig config;st::ForwardTask task;CHECK(st::begin_forward(world,config,task));world.physical.body.velocity[0]=config.target_speed_m_s;
    for(int n=1;n<=80;++n){world.physical.body.position.x=config.target_speed_m_s*p::dt*float(n)/p::scale;world.physical.body.position.z=.003f/p::scale;world.physical.control_steps=n;const auto result=st::record_control(task,world,config);CHECK(result.accepted);}
    CHECK(task.outcome==st::Outcome::Timeout&&task.maximum_lateral>.002f);CHECK(world.attempted_substeps==0);
}
