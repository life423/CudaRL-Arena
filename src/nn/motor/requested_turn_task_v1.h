#ifndef CUDARL_REQUESTED_TURN_TASK_V1_H
#define CUDARL_REQUESTED_TURN_TASK_V1_H
#include "motor/turning_task_v2.h"

// Preparation-only requested-motion lesson. This is not a bank-navigation
// policy or an unlimited viewer mode. All physical forces remain Model6.
namespace cudarl::requested_turn_task_v1 {
namespace p=physics;namespace old=straight_motor;namespace joint=joint_motor;
namespace model=movement;namespace bank=turning_task_v2;
using World=joint::World;
constexpr unsigned task_version=1,observation_version=5,reward_version=1,intent_version=1;
constexpr int observations=42,critic_observations=43,actions=4,controls=240;
constexpr int cases_per_pose=12,case_count=24,request_control=80,stall_first_control=140,motion_window_controls=40;
constexpr double duration_seconds=12.,return_normalizer_seconds=12.;
constexpr float reference_speed_m_s=.008f,velocity_width_m_s=.0025f;
enum Kind {Rest=0,Swim=1,Turn=2};
enum Reason {Running=0,Boundary=1,Deadline=2,HardFault=3,Depth=4,Tilt=5,Stalled=6};
enum EnergyIndex {SignedWork=0,PositiveWork=1,FluidLoss=2,NumericalLoss=3,ExternalWork=4,EnergyError=5};
struct Case {int id=0,pose=0,kind=Rest;float nominal_offset_deg=0,effective_offset_deg=0,angle_cap=90;bool withheld=false;};
struct Task {
    old::ForwardTask forward{};
    p::V original_heading{},last_com_model{};
    uint64_t controls=0,last_world_control=0;
    int maximum_controls=240,case_id=0,pose=0,kind=Rest,reason=Running;
    float nominal_offset_deg=0,effective_offset_deg=0,angle_cap=90;
    float maximum_depth_m=0,maximum_tilt_deg=0,minimum_clearance_m=1e6f;
    float motion_window_arc[motion_window_controls]{},motion_window_dt[motion_window_controls]{};
    int motion_window_next=0,motion_window_count=0;
    float rolling_horizontal_speed_m_s=0;
    double elapsed_seconds=0,reward_sum=0,quality_seconds=0,path_length_m=0;
    bool holding=false,heading_only=false,request_latched=false,done=false;
    bool boundary_terminal=false,behavior_terminal=false,withheld=false;
};
struct StepInfo {
    int substeps=0,attempted_substeps=0,accepted_substeps=0,failure_code=0,terminal_reason=Running;
    float seconds=0,dt=0,maximum_depth_m=0,maximum_tilt_deg=0,minimum_clearance_m=1e6f;
    double positive_work_before=0,positive_work_after=0,positive_work_delta=0;
    double energy_before[6]{},energy_after[6]{};
    bool boundary_terminal=false,behavior_terminal=false,hard_fault=false,valid=false;
    p::BodyState before{},candidate{};
    model::Telemetry telemetry{};
};
struct Transition {
    old::Metrics metrics{};
    float reward=0,seconds=0,dt=0;
    double quality=0,positive_power=0;
    int substeps=0,reason=Running;
    bool accepted=false,done=false,truncated=false,boundary_terminal=false,behavior_terminal=false;
};

CUDARL_HD inline bool finite_double(double x){return x==x&&fabs(x)<1e100;}
CUDARL_HD inline float curriculum_limit(int update) {
    return update==0?90.f:update<1||update>32?0.f:update<=8?15.f:update<=16?30.f:update<=24?60.f:90.f;
}
CUDARL_HD inline Case case_for(int id,float angle_cap=90,bool withheld=false) {
    Case out;out.id=id;out.pose=id<cases_per_pose?(withheld?1:0):(withheld?2:3);out.withheld=withheld;out.angle_cap=angle_cap;
    const int item=id%cases_per_pose;out.kind=item==0?Rest:item==1?Swim:Turn;
    if(item>=2){const int pair=(item-2)/2;const float angle=pair==0?5.f:pair==1?15.f:pair==2?30.f:pair==3?60.f:90.f;
        out.nominal_offset_deg=(item%2==0?1.f:-1.f)*angle;
        out.effective_offset_deg=(item%2==0?1.f:-1.f)*fminf(angle,angle_cap);}
    return out;
}
CUDARL_HD inline old::ForwardConfig config(bool holding) {
    auto out=old::config_for_speed(reference_speed_m_s);out.observation_schema=4;out.maximum_controls=controls;
    if(holding)out.target_speed_m_s=0;
    out.stationary_request=holding;out.position_hold=holding;return out;
}
CUDARL_HD inline bool in_domain(const World& world){return bank::in_domain(world);}
CUDARL_HD inline bool initialize(int id,World& world,Task& task,float angle_cap=90,bool withheld=false) {
    if(id<0||id>=case_count||(angle_cap!=15&&angle_cap!=30&&angle_cap!=60&&angle_cap!=90))return false;
    const auto fixture=case_for(id,angle_cap,withheld);auto next=old::initial_world(fixture.pose);next.arena=old::long_swim_arena();
    // Initial translation only. Keep the original vertical state and attitude.
    const auto original=cudarl::motor::motion(next.physical.body).com_model;
    next.physical.body.position.x-=original.x;next.physical.body.position.z-=original.z;
    if(!requested_turn_task_v1::in_domain(next)||bank::bounds(next.physical.body).inset_radius_squared>=1)return false;
    Task t;t.case_id=id;t.pose=fixture.pose;t.kind=fixture.kind;t.holding=fixture.kind==Rest;t.withheld=withheld;
    t.nominal_offset_deg=fixture.nominal_offset_deg;t.effective_offset_deg=fixture.effective_offset_deg;t.angle_cap=angle_cap;
    if(!old::begin_forward(next,config(t.holding),t.forward))return false;
    if(t.holding&&!old::update_velocity_request(next,config(true),t.forward))return false;
    t.original_heading=t.forward.request.heading;t.last_com_model=cudarl::motor::motion(next.physical.body).com_model;
    t.last_world_control=next.physical.control_steps;t.minimum_clearance_m=bank::bounds(next.physical.body).clearance_m;
    t.maximum_tilt_deg=old::measure(t.forward,next).tilt_deg;
    world=next;task=t;return true;
}
CUDARL_HD inline bool update_request(Task& task,const World& world) {
    if(task.done||!requested_turn_task_v1::in_domain(world)||world.physical.control_steps!=task.last_world_control||task.forward.controls!=int(task.controls))return false;
    if(task.holding)return old::update_velocity_request(world,config(true),task.forward);
    if(task.kind==Turn&&!task.request_latched&&task.controls>=request_control){
        if(task.controls!=request_control||task.effective_offset_deg==0)return false;
        const auto heading=p::axis_rotation({0,1,0},task.effective_offset_deg*model::pi/180.f).rotate(task.original_heading);
        task.forward.request.heading=heading;task.forward.request.terminal_velocity=heading*reference_speed_m_s;
        task.forward.request.position=task.forward.start_com_model+heading*(.072f/p::scale);
        task.heading_only=true;task.request_latched=true;
    }
    return true;
}
CUDARL_HD inline bool observe(const Task& task,const World& world,float* out) {
    if(!out||task.done||!requested_turn_task_v1::in_domain(world))return false;
    if(task.kind==Turn&&task.controls>=request_control&&!task.request_latched)return false;
    auto view=task.forward;
    if(task.heading_only){const auto com=cudarl::motor::motion(world.physical.body).com_model;
        // Remove WORLD horizontal lane error before inverse rotation. The
        // persistent heading/velocity, original depth and actual Task stay put.
        view.start_com_model.x=com.x;view.start_com_model.z=com.z;}
    if(!old::observe_appended(view,world,config(task.holding),out))return false;
    out[41]=task.heading_only?1.f:0.f;return true;
}
CUDARL_HD inline double remaining_seconds(const Task& task){return fmax(0.,duration_seconds-task.elapsed_seconds);}
CUDARL_HD inline void energy(const World& w,double* out) {
    out[SignedWork]=w.signed_work;out[PositiveWork]=w.positive_work;out[FluidLoss]=w.fluid_loss;
    out[NumericalLoss]=w.numerical_loss;out[ExternalWork]=w.external_work;out[EnergyError]=w.last_energy_error;
}
CUDARL_HD inline bool close_step(const World& w,StepInfo& info,bool valid) {
    energy(w,info.energy_after);info.positive_work_after=w.positive_work;
    info.positive_work_delta=info.positive_work_after-info.positive_work_before;
    info.valid=valid;if(!valid)info.hard_fault=true;return valid;
}
CUDARL_HD inline int behavior_reason(float absolute_depth_m,float tilt_deg,const bank::Bounds& boundary) {
    return boundary.inset_radius_squared>=1?Boundary:absolute_depth_m>.005f?Depth:tilt_deg>20?Tilt:Running;
}
CUDARL_HD inline bool step_control(World& w,const Task& task,const float* command,StepInfo& info,p::BodyState* saved_substeps=nullptr) {
    info={};info.before=info.candidate=w.physical.body;energy(w,info.energy_before);info.positive_work_before=w.positive_work;
    const auto initial=bank::bounds(w.physical.body);
    if(task.done||!command||w.physical.paused||!requested_turn_task_v1::in_domain(w)||!initial.finite||initial.inset_radius_squared>=1){info.failure_code=101;return close_step(w,info,false);}
    if(task.kind==Turn&&task.controls>=request_control&&!task.request_latched){info.failure_code=102;return close_step(w,info,false);}
    for(int j=0;j<4;++j)if(!p::finite_number(command[j])||fabsf(command[j])>1){info.failure_code=103;return close_step(w,info,false);}
    auto& state=w.physical;state.telemetry={};state.mean_squared_torque=0;
    for(int j=0;j<4;++j)state.raw_action[j]=command[j];
    for(int i=0;i<40;++i){state.actual_torques[i]=0;state.requested_torques[i]=0;}
    // Mechanical operations/order are copied from the sealed joint wrapper.
    for(int k=0;k<10;++k){
        info.before=info.candidate=state.body;const float h=.005f;model::Reference reference;
        if(!joint::filter_targets(state.held_targets,w.reference_rate,command,reference)){w.failure_code=10;state.invalid=true;info.failure_code=10;return close_step(w,info,false);}
        model::Telemetry telemetry;++w.attempted_substeps;++info.substeps;info.attempted_substeps=info.substeps;
        const bool advanced=model::substep(state.body,reference,h,state.fluid,telemetry,true,w.arena,&info.candidate);info.telemetry=telemetry;
        if(!advanced){w.failure_code=telemetry.failure_code;state.invalid=true;info.failure_code=w.failure_code;return close_step(w,info,false);}
        w.signed_work+=telemetry.signed_work;w.positive_work+=telemetry.positive_work;w.fluid_loss+=telemetry.fluid_loss;
        w.numerical_loss+=telemetry.numerical_loss;w.external_work+=telemetry.external_work;
        w.last_energy_error=model::kinetic(state.body)-w.signed_work+w.fluid_loss+w.numerical_loss-w.external_work;
        if(fabs(w.last_energy_error)>.02*(w.positive_work+w.fluid_loss)+2e-10){w.failure_code=9;state.invalid=true;info.failure_code=9;return close_step(w,info,false);}
        ++state.substeps;++state.telemetry.executed_substeps;++info.accepted_substeps;
        state.time_seconds=static_cast<double>(state.substeps)*h;
        state.telemetry.minimum_floor_gap=fminf(state.telemetry.minimum_floor_gap,telemetry.minimum_gap);
        state.telemetry.minimum_other_gap=fminf(state.telemetry.minimum_other_gap,telemetry.minimum_gap);
        for(int j=0;j<4;++j){state.held_targets[j]=reference.angle[j];const float normalized=telemetry.torque[j]/p::torque;
            state.last_actual_torques[j]=normalized;state.actual_torques[k*4+j]=normalized;state.requested_torques[k*4+j]=telemetry.requested[j];
            state.mean_squared_torque+=normalized*normalized;}
        info.seconds=info.dt=info.accepted_substeps*h;
        const auto boundary=bank::bounds(state.body);
        const float depth=fabsf((cudarl::motor::motion(state.body).com_model.y-task.forward.start_com_model.y)*p::scale);
        const float tilt=acosf(p::clamp(state.body.orientation.rotate({0,1,0}).y,-1,1))*180/model::pi;
        info.maximum_depth_m=fmaxf(info.maximum_depth_m,depth);info.maximum_tilt_deg=fmaxf(info.maximum_tilt_deg,tilt);
        info.minimum_clearance_m=fminf(info.minimum_clearance_m,boundary.clearance_m);
        if(saved_substeps)saved_substeps[k]=state.body;
        if(!boundary.finite||boundary.active_radius_squared>1){w.failure_code=104;state.unsafe=true;info.failure_code=104;return close_step(w,info,false);}
        info.terminal_reason=behavior_reason(depth,tilt,boundary);
        if(info.terminal_reason!=Running){info.boundary_terminal=info.terminal_reason==Boundary;info.behavior_terminal=true;break;}
    }
    state.mean_squared_torque/=float(info.accepted_substeps*4);++state.control_steps;
    const bool valid=joint::in_domain(w);if(!valid)info.failure_code=105;
    return close_step(w,info,valid);
}
CUDARL_HD inline double quality(const Task& task,const World& world,double positive_power) {
    if(!finite_double(positive_power)||positive_power<0)return -1;
    const auto z=old::measure(task.forward,world);
    const double depth=z.signed_depth_m/.001f,roll=z.roll_deg/10.f,pitch=z.pitch_deg/10.f;
    if(task.holding){const double speed=p::norm(z.velocity_m_s)/.0005f,anchor=p::norm(z.hold_error_m)/.002f;
        return 1/(1+speed*speed+anchor*anchor+depth*depth+roll*roll+pitch*pitch+6250*positive_power);}
    const auto error=z.velocity_m_s-task.forward.request.terminal_velocity;
    const double vector_error=p::dot(error,error),width=velocity_width_m_s;
    const double along=p::clamp(p::dot(z.velocity_m_s,task.forward.request.heading)/reference_speed_m_s,0,1);
    const double lane=task.heading_only?0:z.lateral_m/.002f;
    return along*exp(-.5*vector_error/(width*width))/(1+.25*lane*lane+.05*depth*depth+.05*roll*roll+.01*pitch*pitch+6250*positive_power);
}
CUDARL_HD inline Transition record(Task& task,const World& world,const StepInfo& info) {
    Transition out;
    if(task.done||!info.valid||info.hard_fault||info.accepted_substeps<1||info.accepted_substeps>10||info.substeps!=info.accepted_substeps||
       world.physical.control_steps!=task.last_world_control+1||!requested_turn_task_v1::in_domain(world))return out;
    if(!info.behavior_terminal&&info.accepted_substeps!=10)return out;
    if(info.behavior_terminal&&info.terminal_reason!=Boundary&&info.terminal_reason!=Depth&&info.terminal_reason!=Tilt)return out;
    if(info.boundary_terminal!=(info.terminal_reason==Boundary)||info.seconds!=info.accepted_substeps*.005f||info.dt!=info.seconds)return out;
    if(!finite_double(info.positive_work_delta)||info.positive_work_delta<0||info.positive_work_after!=world.positive_work||
       info.positive_work_delta!=info.positive_work_after-info.positive_work_before)return out;
    Task next=task;const auto com=cudarl::motor::motion(world.physical.body).com_model;const auto delta=com-task.last_com_model;
    const float arc=p::norm({delta.x,0,delta.z})*p::scale;
    next.motion_window_arc[next.motion_window_next]=arc;next.motion_window_dt[next.motion_window_next]=info.seconds;
    next.motion_window_next=(next.motion_window_next+1)%motion_window_controls;
    if(next.motion_window_count<motion_window_controls)++next.motion_window_count;
    double arc_sum=0,dt_sum=0;for(int k=0;k<next.motion_window_count;++k){arc_sum+=next.motion_window_arc[k];dt_sum+=next.motion_window_dt[k];}
    if(!(dt_sum>0))return out;next.rolling_horizontal_speed_m_s=static_cast<float>(arc_sum/dt_sum);
    const bool stalled=!task.holding&&!info.behavior_terminal&&task.controls+1>=stall_first_control&&
        next.motion_window_count==motion_window_controls&&next.rolling_horizontal_speed_m_s<.002f;
    const bool behavior=info.behavior_terminal||stalled;
    out.seconds=out.dt=info.seconds;out.substeps=info.substeps;out.metrics=old::measure(task.forward,world);
    out.positive_power=info.positive_work_delta/info.seconds;out.quality=quality(task,world,out.positive_power);
    if(!finite_double(out.quality)||out.quality<0||out.quality>1)return out;
    next.elapsed_seconds+=info.seconds;
    const double cost=behavior?remaining_seconds(next)+4:0;
    out.reward=static_cast<float>((info.seconds*out.quality-cost)/return_normalizer_seconds);
    if(!p::finite_number(out.reward))return out;
    ++next.controls;next.last_world_control=world.physical.control_steps;next.forward.controls=int(next.controls);next.forward.last_world_control=next.last_world_control;
    next.maximum_depth_m=fmaxf(next.maximum_depth_m,info.maximum_depth_m);next.maximum_tilt_deg=fmaxf(next.maximum_tilt_deg,info.maximum_tilt_deg);
    next.minimum_clearance_m=fminf(next.minimum_clearance_m,info.minimum_clearance_m);
    next.last_com_model=com;next.path_length_m+=arc;next.reward_sum+=out.reward;next.quality_seconds+=info.seconds*out.quality;
    next.boundary_terminal=info.boundary_terminal;next.behavior_terminal=behavior;
    next.done=behavior||next.controls>=uint64_t(next.maximum_controls);
    next.reason=info.behavior_terminal?info.terminal_reason:stalled?Stalled:next.done?Deadline:Running;
    task=next;out.accepted=true;out.done=task.done;out.behavior_terminal=behavior;out.boundary_terminal=task.boundary_terminal;out.reason=task.reason;return out;
}
}
#endif
