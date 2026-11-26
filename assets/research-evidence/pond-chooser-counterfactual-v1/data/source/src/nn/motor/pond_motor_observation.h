#ifndef CUDARL_POND_MOTOR_OBSERVATION_H
#define CUDARL_POND_MOTOR_OBSERVATION_H
#include "physics/dynamics.h"
#include <cstdint>

// Current saved motor's input/measurement adapter. This owns no physical world,
// force, domain, scorer, deadline or optimizer. Schema 4 retains the exact input
// arithmetic used to train the saved 41-input motor; world validity belongs to
// the versioned pond runtime, not an old benchmark's rectangular arena.
namespace cudarl::pond_motor_observation {
namespace p=physics;
constexpr uint32_t version=1,observation_schema=4;
constexpr int observations=41,measure_start=40;
constexpr float velocity_scale_m_s=.02f,angular_scale_rad_s=4.f,path_scale_m=.002f;
constexpr float request_distance_m=.072f,swim_speed_m_s=.008f;
constexpr float pi=3.14159265358979323846f;
struct Motion {p::V com_model{},velocity_m_s{};};
CUDARL_HD inline Motion motion(const p::BodyState& body) {
    const auto shape=p::shape(body);Motion result;float mass=0;
    for(const auto& link:shape.link) {
        mass+=link.mass;result.com_model+=link.center*link.mass;
        result.velocity_m_s+=p::multiply(link.linear,body.velocity)*link.mass;
    }
    result.com_model=result.com_model/(mass*p::scale);result.velocity_m_s=result.velocity_m_s/mass;
    return result;
}
struct Request {p::V position{},terminal_velocity{},heading{1,0,0};};
struct RequestState {
    Request request{};
    p::V start_com_model{},hold_anchor_model{};
    bool hold_anchor_set=false,holding=false;
    uint64_t controls=0;
};
struct Metrics {
    float forward_m=0,lateral_m=0,depth_m=0,speed_error_m_s=0,head_yaw_deg=0,tilt_deg=0;
    float signed_depth_m=0,vertical_velocity_m_s=0,horizontal_speed_error_m_s=0;
    float roll_deg=0,pitch_deg=0;
    p::V hold_error_m{},velocity_m_s{};
};
// Fixture construction only. An ongoing world is never moved or reset by
// observe()/measure(). Hold ids 0..5 and Swim ids 0..3 retain the saved starts.
CUDARL_HD inline bool initialize(int case_id,bool holding,p::BodyState& body,float* held_targets,
                                 float* reference_rate,float* last_actual_torques,RequestState& request) {
    if(case_id<0||case_id>(holding?5:3)||!held_targets||!reference_rate||!last_actual_torques)return false;
    const int pose=holding?(case_id%2?3:0):case_id;
    p::BodyState next;
    next.orientation=p::axis_rotation({0,1,0},(float(pose%4)-1.5f)*.05f);
    next.position.x=(float(pose%4)-1.5f)*.02f;
    next.angle[0]=pose%2?.02f:-.02f;next.angle[2]=-next.angle[0];
    const auto initial=motion(next);
    auto heading=next.orientation.rotate({1,0,0});heading=heading/p::norm(heading);
    RequestState state;state.holding=holding;state.start_com_model=initial.com_model;
    state.request={initial.com_model+heading*(request_distance_m/p::scale),
        heading*(holding?0.f:swim_speed_m_s),heading};
    if(holding) {
        const float offset=case_id/2==1?-.0005f:case_id/2==2?.0005f:0.f;
        if(offset!=0)next.position.y+=offset/p::scale;
        state.hold_anchor_model=motion(next).com_model;
        state.hold_anchor_model.y=state.start_com_model.y;
        state.hold_anchor_set=true;
    }
    body=next;request=state;
    for(int j=0;j<4;++j) {
        held_targets[j]=next.angle[j];reference_rate[j]=0;last_actual_torques[j]=0;
    }
    return true;
}
CUDARL_HD inline void vector_features(float* output,int offset,p::V value) {
    output[offset]=value.x;output[offset+1]=value.y;output[offset+2]=value.z;
}
CUDARL_HD inline bool observe(const RequestState& state,const p::BodyState& body,const float* held_targets,
                              const float* reference_rate,const float* last_actual_torques,float* output,
                              bool joint_target_actuation=true) {
    if(!output||!held_targets||!reference_rate||!last_actual_torques||!p::finite_body(body)||
       !p::finite_vector(state.start_com_model)||!p::finite_vector(state.request.heading)||
       !p::finite_vector(state.request.terminal_velocity)||
       (state.holding&&(!state.hold_anchor_set||!p::finite_vector(state.hold_anchor_model))))return false;
    const auto current=motion(body);const auto& q=body.orientation;
    const auto displacement=(current.com_model-state.start_com_model)*p::scale;
    // Schema4 keeps schema2's original lane/depth channels even while holding.
    // Its appended channel40 supplies longitudinal fixed-anchor error.
    const auto lane_error=state.request.heading*p::dot(displacement,state.request.heading)-displacement;
    vector_features(output,0,q.inverse(lane_error)/path_scale_m);
    vector_features(output,3,q.inverse(state.request.terminal_velocity)/velocity_scale_m_s);
    vector_features(output,6,q.inverse(state.request.heading));
    vector_features(output,9,q.inverse({0,1,0}));
    vector_features(output,12,q.inverse(current.velocity_m_s)/velocity_scale_m_s);
    vector_features(output,15,q.inverse({body.velocity[3],body.velocity[4],body.velocity[5]})/angular_scale_rad_s);
    for(int j=0;j<4;++j) {
        output[18+j]=body.angle[j]/p::limit;
        output[22+j]=body.velocity[6+j]/12.f;
        output[26+j]=last_actual_torques[j];
        output[30+j]=held_targets[j]/p::limit;
    }
    // This is the inherited warmup input feature, not a movement constraint.
    output[34]=p::clamp(1.f-float(state.controls)/float(measure_start),0,1);
    output[35]=joint_target_actuation?1.f:0.f;
    for(int j=0;j<4;++j)output[36+j]=reference_rate[j]/12.f;
    output[40]=0;
    if(state.holding) {
        const auto error=(state.hold_anchor_model-current.com_model)*p::scale;
        output[40]=tanhf(p::dot(error,state.request.heading)/.02f);
    }
    for(int j=0;j<observations;++j)if(!p::finite_number(output[j]))return false;
    return true;
}
CUDARL_HD inline Metrics measure(const RequestState& state,const p::BodyState& body) {
    const auto current=motion(body);
    const auto displacement=(current.com_model-state.start_com_model)*p::scale;
    Metrics result;result.velocity_m_s=current.velocity_m_s;
    if(state.hold_anchor_set)result.hold_error_m=(current.com_model-state.hold_anchor_model)*p::scale;
    result.forward_m=p::dot(displacement,state.request.heading);
    const auto perpendicular=displacement-state.request.heading*result.forward_m;
    result.lateral_m=p::norm({perpendicular.x,0,perpendicular.z});
    result.depth_m=fabsf(displacement.y);result.signed_depth_m=displacement.y;
    result.vertical_velocity_m_s=current.velocity_m_s.y;
    const auto velocity_error=current.velocity_m_s-state.request.terminal_velocity;
    result.speed_error_m_s=p::norm(velocity_error);
    result.horizontal_speed_error_m_s=p::norm({velocity_error.x,0,velocity_error.z});
    const auto forward=body.orientation.rotate({1,0,0}),up=body.orientation.rotate({0,1,0});
    result.head_yaw_deg=fabsf(atan2f(p::dot(p::cross(state.request.heading,forward),{0,1,0}),
        p::dot(state.request.heading,forward)))*180/pi;
    result.tilt_deg=acosf(p::clamp(p::dot(up,{0,1,0}),-1,1))*180/pi;
    const auto local_up=body.orientation.inverse({0,1,0});
    result.pitch_deg=atan2f(local_up.x,sqrtf(local_up.y*local_up.y+local_up.z*local_up.z))*180/pi;
    result.roll_deg=atan2f(-local_up.z,local_up.y)*180/pi;
    return result;
}
}
#endif
