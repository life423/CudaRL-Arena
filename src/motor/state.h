#ifndef CUDARL_MOTOR_STATE_H
#define CUDARL_MOTOR_STATE_H
#include "simulation/world.h"
namespace cudarl::motor {
namespace p=physics; namespace sim=simulation;
constexpr float kBodyLength=.072f,kVelocityScale=.02f,kAngularScale=4.f;
enum class Outcome {Running,Success,Timeout,Rejected};
enum ObservationIndex {Goal=0,TerminalVelocity=3,Heading=6,Up=9,Velocity=12,
    AngularVelocity=15,JointAngles=18,JointRates=22,ActualTorque=26,
    HeldTargets=30,TimeRemaining=34,ActuationMode=35};
struct Motion {p::V com_model{},velocity_m_s{};};
CUDARL_HD inline Motion motion(const p::BodyState& body) {
    const auto shape=p::shape(body);Motion result;float mass=0;
    for(const auto& link:shape.link){mass+=link.mass;result.com_model+=link.center*link.mass;
        result.velocity_m_s+=p::multiply(link.linear,body.velocity)*link.mass;}
    result.com_model=result.com_model/(mass*p::scale);result.velocity_m_s=result.velocity_m_s/mass;
    return result;
}
CUDARL_HD inline bool in_domain(const sim::WorldState& w,float clearance_m=.01f) {
    if(!p::finite_number(clearance_m)||clearance_m<.01f||!sim::finite_state(w)||w.unsafe||w.invalid||w.current.x!=0||w.current.y!=0||w.current.z!=0)return false;
    const p::Parameters calm;
    if(w.fluid.density!=calm.density||w.fluid.viscosity!=calm.viscosity||w.fluid.gravity!=calm.gravity)return false;
    if(!p::finite_number(w.mean_squared_torque)||w.mean_squared_torque<0||w.mean_squared_torque>1)return false;
    for(int j=0;j<4;++j)
        if(!p::finite_number(w.last_actual_torques[j])||fabsf(w.last_actual_torques[j])>1||
            !p::finite_number(w.held_targets[j])||fabsf(w.held_targets[j])>p::limit)return false;
    const auto gap=contact::inspect(w.body);
    return !gap.invalid&&!gap.escape&&!gap.excessive&&gap.min_gap>=clearance_m&&
        p::finite_number(w.telemetry.minimum_floor_gap)&&p::finite_number(w.telemetry.minimum_other_gap)&&
        w.telemetry.minimum_floor_gap>=clearance_m&&w.telemetry.minimum_other_gap>=clearance_m;
}
CUDARL_HD inline void vector_features(float* out,int i,p::V v){out[i]=v.x;out[i+1]=v.y;out[i+2]=v.z;}
}
#endif
