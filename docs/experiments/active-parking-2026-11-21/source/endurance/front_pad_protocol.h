#ifndef FRONT_PAD_PROTOCOL_H
#define FRONT_PAD_PROTOCOL_H
#include "motor/hold_assessment.h"
#include "nn/mlp_cpu.h"
#include "rl/append_hold_checkpoint.h"
#include "rl/ar_gaussian_math.h"
#include <array>
#include <cmath>
#include <stdexcept>

namespace pilot {
namespace m=cudarl::straight_motor;
namespace p=cudarl::physics;
namespace nn=cudarl::nn;
namespace rl=cudarl::rl;
constexpr int Inputs=42, BaseInputs=40, Outputs=4, Hidden=32;
constexpr int Env=32, LaneTicks=320, Episode=80, MaxUpdates=4;
constexpr int HeldCount=2, ShortControls=640, LongControls=2400, PadControls=HeldCount*Episode;
constexpr int TrainingUpdatesControls=Env*LaneTicks*MaxUpdates;
constexpr int TotalControls=TrainingUpdatesControls+(1+MaxUpdates)*(ShortControls+LongControls+PadControls);
static_assert(TotalControls==56960);
constexpr uint32_t Seed=1692487673u;
constexpr float Gamma=.99f, Rho=.8f, LogSigma=-1.f;
constexpr float GoalRadius=.001f, SettleSpeed=.0005f, ApproachSpeed=.003f, ApproachRadius=.006f;
constexpr float PadAngleLimit=10.f, PadDepthLimit=.001f, PadLaneLimit=.002f;
constexpr int SettleFirst=60;
constexpr int Episodes=64, Groups=4, Peers=Episodes/Groups;
constexpr int Mini=256, Epochs=2;
constexpr float ActorLR=.001f, PpoClip=.2f, MaxKl=.01f;
inline void need(bool x,const char* e) {if(!x)throw std::runtime_error(e);}
struct Start {int pose;float distance;};
inline constexpr Start train[Groups]={{0,.001875f},{3,.00195f},{0,.00205f},{3,.002125f}};
inline constexpr Start held[HeldCount]={{1,.002f},{2,.002f}};
inline nn::MlpSpec spec(int n=Inputs){
    return {{n,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}};
}
inline int front_group(int episode){return episode%Groups;}
inline bool updated_index(int i){return i<Inputs*Hidden && i%Inputs>=BaseInputs;}
inline std::vector<float> adapt(const std::vector<float>& parent){
    need(parent.size()==2500,"expected original40 actor");
    std::vector<float> next(size_t(spec().num_params()),0.f);
    for(int j=0;j<Hidden;++j) std::copy_n(parent.begin()+j*BaseInputs,BaseInputs,next.begin()+j*Inputs);
    std::copy(parent.begin()+Hidden*BaseInputs,parent.end(),next.begin()+Hidden*Inputs);
    return next;
}
inline m::ForwardConfig swim(int horizon) {
    auto c=m::config_for_speed(.008f);
    c.observation_schema=cudarl::motor_observation::task_scale;
    c.maximum_controls=horizon; return c;
}
inline m::ForwardConfig hold() {
    auto c=swim(Episode);
    c.observation_schema=cudarl::motor_observation::appended_hold;
    c.target_speed_m_s=0; c.stationary_request=true; c.position_hold=true; return c;
}
inline m::ForwardTask setup_front(Start start,m::World& w){
    need(start.distance>GoalRadius,"no centered/trivial front start");
    w=m::initial_world(start.pose);
    w.arena=m::long_swim_arena();
    m::ForwardTask t; const auto c=hold();
    need(m::begin_forward(w,c,t),"front initialization");
    // Match the diagnostic world-space anchor and initial state; no motion
    // override after initialization, and no fixed-target action in PPO.
    const auto pad=t.start_com_model+p::V{.013f/p::scale,0,.009f/p::scale};
    const auto h=t.request.heading;
    w.physical.body.position=w.physical.body.position+(pad-t.start_com_model)+h*(-start.distance/p::scale);
    t.hold_anchor_model=pad;
    t.hold_anchor_set=true;
    t.start_com_model=pad;
    t.steady_start_com_model=m::old::motion(w.physical.body).com_model;
    constexpr float initial_speed=.0005f;
    w.physical.body.velocity[0]=h.x*initial_speed;
    w.physical.body.velocity[1]=0;
    w.physical.body.velocity[2]=h.z*initial_speed;
    w.external_work=cudarl::movement::kinetic(w.physical.body);
    const auto z=m::measure(t,w);
    need(p::norm(z.velocity_m_s-h*initial_speed)<1e-7f &&
         std::fabs(p::dot(z.hold_error_m,h)+start.distance)<5e-7f &&
         w.external_work>0 && w.attempted_substeps==0,"invalid slow initial state");
    need(m::in_domain(w,c) && p::norm(w.physical.current)==0,"front state invalid");
    return t;
}
inline bool observation(const m::ForwardTask& task,const m::World& world,const m::ForwardConfig& conf,float* out) {
    if (conf.position_hold && conf.stationary_request) {
        if(!m::observe_appended(task,world,conf,out))return false;
        out[41]=1.f; return true;
    }
    if(!m::observe(task,world,conf,out))return false;
    out[40]=out[41]=0.f; return true;
}
inline bool in_pad(const m::Metrics& z) {
    return std::hypot(z.hold_error_m.x,z.hold_error_m.z)<=GoalRadius && p::norm(z.velocity_m_s)<=SettleSpeed;
}
inline bool safe(const m::Metrics& z) {
    return std::isfinite(z.tilt_deg) && z.tilt_deg>=0 && z.tilt_deg<=PadAngleLimit &&
           std::isfinite(z.depth_m) && z.depth_m>=0 && z.depth_m<=PadDepthLimit &&
           std::isfinite(z.lateral_m) && z.lateral_m>=0 && z.lateral_m<=PadLaneLimit &&
           p::norm(z.velocity_m_s)<=ApproachSpeed && std::hypot(z.hold_error_m.x,z.hold_error_m.z)<=ApproachRadius;
}
// Use the existing position/desired-return velocity reward only. Approach
// safety failures are measured in the gate, not treated as numerical faults.
inline float pad_reward(const m::Transition& t) {return t.reward;}
struct EpisodeReturns {
    std::array<float,Episode> reward{};
    std::array<double,Episode> value{};
};
inline void returns(EpisodeReturns& a) {
    double future=0.;
    for(int t=Episode-1;t>=0;--t){future=double(a.reward[t])+Gamma*future;a.value[t]=future;}
}
inline std::array<std::array<float,Episode>,Episodes> loo(std::array<EpisodeReturns,Episodes>& eps){
    auto out=std::array<std::array<float,Episode>,Episodes>{};
    double sq=0.;
    for(auto& e:eps)returns(e);
    for(int j=0;j<Episodes;++j) for(int t=0;t<Episode;++t){
        double baseline=0.;int n=0;
        // Sum only peers; excluded episode must never be visited.
        for(int peer=front_group(j);peer<Episodes;peer+=Groups) if(peer!=j){baseline+=eps[peer].value[t];++n;}
        need(n==Peers-1,"LOO cohort mismatch");
        double value=eps[j].value[t]-baseline/n;
        need(std::isfinite(value),"Nonfinite front advantage");
        out[j][t]=float(value);sq+=value*value;
    }
    const auto scale=std::sqrt(sq/(Episodes*Episode));
    need(std::isfinite(scale)&&scale>1e-8,"No meaningful front advantage signal");
    for(auto& e:out)for(auto& v:e)v=float(v/scale);
    return out;
}
}
#endif
