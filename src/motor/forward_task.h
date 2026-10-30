#ifndef CUDARL_MOTOR_FORWARD_TASK_H
#define CUDARL_MOTOR_FORWARD_TASK_H
#include "simulation/world.h"

// Pure task math/bookkeeping. No stepping, reset, inference, optimizer or world
// ownership. A future rollout adapter must reject unsafe intervals BEFORE PPO.
namespace cudarl::motor {
namespace p=physics;
namespace sim=simulation;
constexpr int kTaskVersion=1,kActions=4,kObservations=36;
constexpr float kBodyLength=.072f,kVelocityScale=.02f,kAngularScale=4.f;
enum ObservationIndex {Goal=0,TerminalVelocity=3,Heading=6,Up=9,Velocity=12,
    AngularVelocity=15,JointAngles=18,JointRates=22,ActualTorque=26,
    HeldTargets=30,TimeRemaining=34,ActuationMode=35};
struct ForwardConfig {
    float distance_m=.02f,terminal_speed_m_s=.01f,clearance_m=.01f;
    float lateral_tolerance_m=.002f,speed_tolerance_m_s=.005f,heading_cos=.98480775f;
    int maximum_controls=80,success_controls=3;
};
enum class Outcome {Running,Success,Timeout,Rejected};
struct ForwardTask {
    sim::MotorRequest request{}; // position is WHOLE-BODY COM in model coordinates
    p::V start_com_model{};
    uint64_t last_world_control=0;
    int controls=0,consecutive_success=0;
    float previous_error_m=0;
    Outcome outcome=Outcome::Running;
};
struct Motion {p::V com_model{},velocity_m_s{};};
struct Metrics {
    float error_m=0,forward_m=0,lateral_m=0,speed_error_m_s=0,heading_cos=0;
};
struct Transition {
    Metrics metrics{};
    float reward=0;
    bool accepted=false,done=false,truncated=false;
    Outcome outcome=Outcome::Rejected;
};
struct Action {float latent[kActions]{},command[kActions]{};};

CUDARL_HD inline bool valid(const ForwardConfig& c) {
    const float values[]{c.distance_m,c.terminal_speed_m_s,c.clearance_m,c.lateral_tolerance_m,c.speed_tolerance_m_s};
    for(float x:values)
        if(!p::finite_number(x)||x<=0)return false;
    return p::finite_number(c.heading_cos)&&c.heading_cos>=-1&&c.heading_cos<=1&&
        c.maximum_controls>0&&c.success_controls>0&&c.success_controls<=c.maximum_controls;
}
CUDARL_HD inline Motion motion(const p::BodyState& body) {
    const auto shape=p::shape(body);Motion result;float mass=0;
    for(const auto& link:shape.link){mass+=link.mass;result.com_model+=link.center*link.mass;
        result.velocity_m_s+=p::multiply(link.linear,body.velocity)*link.mass;}
    result.com_model=result.com_model/(mass*p::scale);result.velocity_m_s=result.velocity_m_s/mass;
    return result;
}
CUDARL_HD inline bool in_domain(const sim::WorldState& w,const ForwardConfig& c) {
    if(!valid(c)||!sim::finite_state(w)||w.unsafe||w.invalid||w.current.x!=0||w.current.y!=0||w.current.z!=0)return false;
    const p::Parameters calm;
    if(w.fluid.density!=calm.density||w.fluid.viscosity!=calm.viscosity||w.fluid.gravity!=calm.gravity)return false;
    if(!p::finite_number(w.mean_squared_torque)||w.mean_squared_torque<0||w.mean_squared_torque>1)return false;
    for(int j=0;j<kActions;++j)
        if(!p::finite_number(w.last_actual_torques[j])||fabsf(w.last_actual_torques[j])>1||
            !p::finite_number(w.held_targets[j])||fabsf(w.held_targets[j])>p::limit)return false;
    const auto gap=contact::inspect(w.body);
    return !gap.invalid&&!gap.escape&&!gap.excessive&&gap.min_gap>=c.clearance_m&&
        p::finite_number(w.telemetry.minimum_floor_gap)&&p::finite_number(w.telemetry.minimum_other_gap)&&
        w.telemetry.minimum_floor_gap>=c.clearance_m&&w.telemetry.minimum_other_gap>=c.clearance_m;
}
CUDARL_HD inline bool begin_forward(const sim::WorldState& w,const ForwardConfig& c,ForwardTask& output) {
    if(!in_domain(w,c))return false;
    const auto m=motion(w.body);auto heading=w.body.orientation.rotate({1,0,0});heading=heading/p::norm(heading);
    auto target_pose=w.body;target_pose.position+=heading*(c.distance_m/p::scale);
    const auto target_gap=contact::inspect(target_pose); // geometry query only
    if(target_gap.invalid||target_gap.escape||target_gap.excessive||target_gap.min_gap<c.clearance_m)return false;
    ForwardTask next;next.start_com_model=m.com_model;next.last_world_control=w.control_steps;
    next.request={m.com_model+heading*(c.distance_m/p::scale),heading*c.terminal_speed_m_s,heading};
    next.previous_error_m=c.distance_m;output=next;return true;
}
CUDARL_HD inline Metrics measure(const ForwardTask& t,const sim::WorldState& w) {
    const auto m=motion(w.body);const auto displacement=(m.com_model-t.start_com_model)*p::scale;
    Metrics out;out.error_m=p::norm((t.request.position-m.com_model)*p::scale);
    out.forward_m=p::dot(displacement,t.request.heading);
    out.lateral_m=p::norm(displacement-t.request.heading*out.forward_m);
    out.speed_error_m_s=p::norm(m.velocity_m_s-t.request.terminal_velocity);
    const auto forward=w.body.orientation.rotate({1,0,0});
    out.heading_cos=p::clamp(p::dot(forward,t.request.heading)/p::norm(forward),-1,1);return out;
}
CUDARL_HD inline void vector_features(float* out,int index,p::V v) {out[index]=v.x;out[index+1]=v.y;out[index+2]=v.z;}
CUDARL_HD inline bool observe(const ForwardTask& t,const sim::WorldState& w,const ForwardConfig& c,float* output) {
    if(!output||t.outcome!=Outcome::Running||!in_domain(w,c))return false;
    float out[kObservations]{};const auto m=motion(w.body);const auto q=w.body.orientation;
    vector_features(out,Goal,q.inverse((t.request.position-m.com_model)*p::scale)/kBodyLength);
    vector_features(out,TerminalVelocity,q.inverse(t.request.terminal_velocity)/kVelocityScale);
    vector_features(out,Heading,q.inverse(t.request.heading));vector_features(out,Up,q.inverse({0,1,0}));
    vector_features(out,Velocity,q.inverse(m.velocity_m_s)/kVelocityScale);
    vector_features(out,AngularVelocity,q.inverse({w.body.velocity[3],w.body.velocity[4],w.body.velocity[5]})/kAngularScale);
    for(int j=0;j<kActions;++j){out[JointAngles+j]=w.body.angle[j]/p::limit;out[JointRates+j]=w.body.velocity[6+j]/12.f;
        out[ActualTorque+j]=w.last_actual_torques[j];out[HeldTargets+j]=w.held_targets[j]/p::limit;}
    out[TimeRemaining]=p::clamp(1.f-float(t.controls)/float(c.maximum_controls),0,1);
    out[ActuationMode]=w.actuation==sim::Actuation::JointTarget?1.f:0.f;
    for(float x:out)if(!p::finite_number(x))return false;
    for(int i=0;i<kObservations;++i)output[i]=out[i];return true;
}
// PPO stores/replays latent and its Gaussian logp. command is the deterministic
// tanh transform supplied to the chosen existing actuator interface, not a logp
// sample and not the resulting physical torque. No interface is selected here.
CUDARL_HD inline bool action_from_latent(const float* latent,Action& output) {
    if(!latent)return false;Action next;
    for(int j=0;j<kActions;++j){if(!p::finite_number(latent[j]))return false;next.latent[j]=latent[j];next.command[j]=tanhf(latent[j]);}
    output=next;return true;
}
CUDARL_HD inline Transition record_control(ForwardTask& t,const sim::WorldState& w,const ForwardConfig& c) {
    Transition out;out.outcome=t.outcome;
    if(t.outcome!=Outcome::Running){out.done=true;return out;}
    if(!in_domain(w,c)||w.control_steps<=t.last_world_control||w.control_steps-t.last_world_control!=1){
        t.outcome=out.outcome=Outcome::Rejected;out.done=true;return out;}
    out.metrics=measure(t,w);
    const float velocity_cost=out.metrics.speed_error_m_s/kVelocityScale;
    const float lane_cost=out.metrics.lateral_m/c.distance_m;
    out.reward=(t.previous_error_m-out.metrics.error_m)/c.distance_m-p::dt*(
        .25f*velocity_cost*velocity_cost+.1f*lane_cost*lane_cost+.1f*(1-out.metrics.heading_cos)+.002f*w.mean_squared_torque);
    if(!p::finite_number(out.reward)){t.outcome=out.outcome=Outcome::Rejected;out.done=true;return out;}
    ++t.controls;t.last_world_control=w.control_steps;t.previous_error_m=out.metrics.error_m;
    const bool reached=out.metrics.forward_m>=c.distance_m&&out.metrics.lateral_m<=c.lateral_tolerance_m&&
        out.metrics.speed_error_m_s<=c.speed_tolerance_m_s&&out.metrics.heading_cos>=c.heading_cos;
    t.consecutive_success=reached?t.consecutive_success+1:0;
    if(t.consecutive_success>=c.success_controls)t.outcome=Outcome::Success;
    else if(t.controls>=c.maximum_controls)t.outcome=Outcome::Timeout;
    out.accepted=true;out.outcome=t.outcome;out.done=t.outcome!=Outcome::Running;
    // This finite, observed horizon is terminal, not a bootstrap truncation.
    return out;
}
}
#endif
