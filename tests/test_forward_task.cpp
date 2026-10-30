// Synthetic states only: no force solve, integration, CUDA launch or learning.
#include <limits>
#include "check.h"
#include "motor/forward_task.h"
#include "rl/gaussian_math.h"

namespace m=cudarl::motor;
namespace p=cudarl::physics;
namespace s=cudarl::simulation;
namespace {
bool near(float a,float b,float tolerance=2e-6f){return std::fabs(a-b)<=tolerance;}
s::WorldState sample(){s::WorldState w;w.telemetry.minimum_floor_gap=.05f;w.telemetry.minimum_other_gap=.05f;return w;}
}
TEST_CASE("forward_task: COM request and body-frame observations preserve units") {
    auto w=sample();m::ForwardTask task;const m::ForwardConfig cfg;
    CHECK(m::begin_forward(w,cfg,task));
    const auto initial=m::motion(w.body);
    CHECK(near((task.request.position.x-initial.com_model.x)*p::scale,.02f));
    CHECK(near(task.request.terminal_velocity.x,.01f));
    float observation[m::kObservations]{};CHECK(m::observe(task,w,cfg,observation));
    CHECK(near(observation[m::Goal],.02f/.072f));CHECK(near(observation[m::TerminalVelocity],.5f));
    CHECK(observation[m::Heading]==1&&observation[m::Up+1]==1&&observation[m::TimeRemaining]==1);
    w.body.orientation=p::axis_rotation({0,1,0},1.57079632679f);
    CHECK(m::begin_forward(w,cfg,task));CHECK(m::observe(task,w,cfg,observation));
    CHECK(near(observation[m::Goal],.02f/.072f));CHECK(near(observation[m::Goal+2],0));
    CHECK(near(task.request.heading.z,-1));
}
TEST_CASE("forward_task: actual torque and held targets have distinct features") {
    auto w=sample();w.last_actual_torques[0]=-.25f;w.held_targets[0]=.75f;
    m::ForwardTask task;const m::ForwardConfig cfg;CHECK(m::begin_forward(w,cfg,task));
    float observation[m::kObservations]{};CHECK(m::observe(task,w,cfg,observation));
    CHECK(observation[m::ActualTorque]==-.25f&&observation[m::HeldTargets]==.75f);
    CHECK(observation[m::ActuationMode]==1);
    w.actuation=s::Actuation::Torque;CHECK(m::observe(task,w,cfg,observation));CHECK(observation[m::ActuationMode]==0);
}
TEST_CASE("forward_task: head displacement alone is not COM travel") {
    auto w=sample();m::ForwardTask task;const m::ForwardConfig cfg;CHECK(m::begin_forward(w,cfg,task));
    w.body.angle[0]=.8f;const auto changed=m::motion(w.body);
    const auto shift=task.start_com_model-changed.com_model;w.body.position+=shift;
    CHECK(p::norm(w.body.position-p::BodyState{}.position)>1e-3f);
    const auto metrics=m::measure(task,w);CHECK(near(metrics.forward_m,0));CHECK(near(metrics.lateral_m,0));
}
TEST_CASE("forward_task: persistent request, progress reward and success need consecutive controls") {
    auto w=sample();m::ForwardTask task;const m::ForwardConfig cfg;CHECK(m::begin_forward(w,cfg,task));
    const auto request=task.request;
    w.body.position.x+=.01f/p::scale;w.body.velocity[0]=.01f;++w.control_steps;
    auto transition=m::record_control(task,w,cfg);CHECK(transition.accepted&&!transition.done&&transition.reward>0);
    w.body.position.x+=.0101f/p::scale;
    for(int i=0;i<cfg.success_controls;++i){++w.control_steps;transition=m::record_control(task,w,cfg);CHECK(transition.accepted);}
    CHECK(transition.done&&!transition.truncated&&transition.outcome==m::Outcome::Success);
    CHECK(task.request.position.x==request.position.x&&task.request.heading.x==request.heading.x&&
        task.request.terminal_velocity.x==request.terminal_velocity.x);
    const int controls=task.controls;transition=m::record_control(task,w,cfg);CHECK(!transition.accepted&&task.controls==controls);
}
TEST_CASE("forward_task: effort uses applied torque and finite horizon is terminal") {
    const m::ForwardConfig cfg;auto w=sample();m::ForwardTask a,b;CHECK(m::begin_forward(w,cfg,a));b=a;
    ++w.control_steps;const auto unforced=m::record_control(a,w,cfg);
    w.mean_squared_torque=1;const auto effort=m::record_control(b,w,cfg);CHECK(effort.reward<unforced.reward);
    m::ForwardConfig short_cfg;short_cfg.maximum_controls=short_cfg.success_controls=1;
    w=sample();CHECK(m::begin_forward(w,short_cfg,a));++w.control_steps;
    const auto timeout=m::record_control(a,w,short_cfg);CHECK(timeout.accepted&&timeout.done&&!timeout.truncated&&timeout.outcome==m::Outcome::Timeout);
}
TEST_CASE("forward_task: bad interval, current, contact or duplicate clock rejects without reward") {
    const m::ForwardConfig cfg;
    for(int defect=0;defect<5;++defect){auto w=sample();m::ForwardTask task;CHECK(m::begin_forward(w,cfg,task));++w.control_steps;
        if(defect==0)w.current.x=.001f;
        if(defect==1)w.telemetry.minimum_floor_gap=.005f; // unsafe INTERVAL, even if final pose is clear
        if(defect==2)w.body.position.y=-3.84f;
        if(defect==3)w.last_actual_torques[0]=std::numeric_limits<float>::quiet_NaN();
        if(defect==4)--w.control_steps;
        const auto result=m::record_control(task,w,cfg);CHECK(!result.accepted&&result.done&&result.reward==0&&result.outcome==m::Outcome::Rejected&&task.controls==0);
    }
}
TEST_CASE("forward_task: latent Gaussian samples stay separate from bounded commands") {
    const float latent[4]{2,-2,0,.5f};m::Action action;CHECK(m::action_from_latent(latent,action));
    CHECK(action.latent[0]==2&&action.command[0]<1&&action.command[0]>.9f);
    const float mean[4]{},log_std[4]{};
    const auto latent_logp=cudarl::rl::gaussian_logp(mean,log_std,action.latent,4);
    const auto command_logp=cudarl::rl::gaussian_logp(mean,log_std,action.command,4);
    CHECK(std::fabs(latent_logp-command_logp)>1);
    auto invalid=action;invalid.latent[1]=std::numeric_limits<float>::quiet_NaN();
    const auto prior=action;CHECK(!m::action_from_latent(invalid.latent,action));
    CHECK(action.latent[0]==prior.latent[0]&&action.command[0]==prior.command[0]);
}
