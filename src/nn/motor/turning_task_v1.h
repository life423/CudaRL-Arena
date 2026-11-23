#ifndef CUDARL_TURNING_TASK_V1_H
#define CUDARL_TURNING_TASK_V1_H
#include "motor/straight_task.h"

namespace cudarl::turning_task_v1 {
namespace p=physics;namespace old=straight_motor;namespace joint=joint_motor;namespace model=movement;
using World=joint::World;
constexpr unsigned task_version=1,observation_version=4,sensor_version=1;
constexpr int observations=41,sensors=8,actions=4,controls=320,training_cases=16,withheld_cases=8;
constexpr float radius_x_m=.20f,radius_z_m=.12f,inset_x_m=.195f,inset_z_m=.115f;
constexpr float flat_x_m=.22f,flat_z_m=.14f,reference_speed_m_s=.008f;
enum Reason {Running=0,Boundary=1,Deadline=2,HardFault=3};
struct Case {p::V com_m{};float yaw=0;int pose=0,id=0;bool withheld=false;};
struct Corners {p::V point[24]{};};
struct Bounds {float active_radius_squared=0,inset_radius_squared=0,clearance_m=0;bool finite=true;};
struct Task {
    uint64_t controls=0,last_world_control=0;
    int maximum_controls=320,reason=Running;
    float fixed_depth_model=-2,maximum_depth_m=0,maximum_tilt_deg=0,minimum_clearance_m=1e6f;
    p::V last_com_model{};
    double elapsed_seconds=0,reward_sum=0,path_length_m=0;
    bool done=false,boundary_terminal=false;
};
struct StepInfo {
    int substeps=0,attempted_substeps=0,accepted_substeps=0,failure_code=0;
    float seconds=0,dt=0;
    double positive_work_delta=0;
    bool boundary_terminal=false,hard_fault=false,valid=false;
    p::BodyState before{},candidate{};
    model::Telemetry telemetry{};
};
struct Metrics {
    float forward_m_s=0,side_m_s=0,depth_error_m=0,roll_deg=0,pitch_deg=0,tilt_deg=0,clearance_m=0;
};
struct Transition {
    Metrics metrics{};
    float reward=0,seconds=0,dt=0;
    int substeps=0,reason=Running;
    bool accepted=false,done=false,truncated=false,boundary_terminal=false;
};

CUDARL_HD inline Corners corners(const p::BodyState& body) {
    const auto shape=model::shape(body);Corners out;int index=0;
    for(int b=0;b<3;++b) {const auto g=model::body_geometry(b);const auto& link=shape.link[b];
        for(int x=-1;x<=1;x+=2)for(int y=-1;y<=1;y+=2)for(int z=-1;z<=1;z+=2)
            out.point[index++]=link.center+link.q.rotate({x*g.length*.5f,y*g.half_height,z*g.half_width});
    }return out;
}
CUDARL_HD inline Bounds bounds(const p::BodyState& body) {
    Bounds out;if(!p::finite_body(body)){out.finite=false;return out;}
    const auto box=corners(body);
    for(const auto v:box.point) {
        if(!p::finite_vector(v)){out.finite=false;return out;}
        const float x=v.x/radius_x_m,z=v.z/radius_z_m,ix=v.x/inset_x_m,iz=v.z/inset_z_m;
        out.active_radius_squared=fmaxf(out.active_radius_squared,x*x+z*z);
        out.inset_radius_squared=fmaxf(out.inset_radius_squared,ix*ix+iz*iz);
    }
    // Certified lower bound, not a claim of exact Euclidean distance. Task
    // termination uses the separately registered inset ellipse above.
    out.clearance_m=radius_z_m*(1-sqrtf(out.active_radius_squared));return out;
}
CUDARL_HD inline bool in_domain(const World& world) {
    const auto b=bounds(world.physical.body);
    return joint::in_domain(world)&&b.finite&&b.active_radius_squared<=1;
}
CUDARL_HD inline p::V forward(const p::BodyState& body) {
    auto v=body.orientation.rotate({1,0,0});v.y=0;const float length=p::norm(v);
    // Azimuth is undefined for a vertical head. This finite sensor convention
    // never rotates the physical body; such tilt cannot pass qualification.
    return length>1e-8f?v/length:p::V{1,0,0};
}
CUDARL_HD inline p::V side(const p::V& f) {return {-f.z,0,f.x};}
CUDARL_HD inline float directional_gap(const Corners& box,p::V direction) {
    double gap=1e6;
    const double dx=double(direction.x)/radius_x_m,dz=double(direction.z)/radius_z_m;
    const double a=dx*dx+dz*dz;
    for(const auto v:box.point) {
        const double x=double(v.x)/radius_x_m,z=double(v.z)/radius_z_m;
        const double b=2*(x*dx+z*dz),c=x*x+z*z-1;
        const double discriminant=b*b-4*a*c;
        if(c>0||discriminant<0||a<=0)return 0;
        const double value=(-b+sqrt(discriminant))/(2*a);
        if(value<gap)gap=value;
    }return static_cast<float>(gap);
}
CUDARL_HD inline bool raw_sensors(const World& world,float* output) {
    if(!output||!turning_task_v1::in_domain(world))return false;
    const auto& body=world.physical.body;const auto f=forward(body),s=side(f);const auto box=corners(body);
    const auto motion=motor::motion(body);const auto com=motion.com_model*p::scale;
    auto normal=p::V{com.x/(radius_x_m*radius_x_m),0,com.z/(radius_z_m*radius_z_m)};
    const float length=p::norm(normal);
    // At the geometric centre, an outward direction is undefined. A one-micron
    // sensor convention avoids normalizing nanometre float COM cancellation.
    normal=com.x*com.x+com.z*com.z>1e-12f&&length>1e-8f?normal/length:p::V{};
    output[0]=p::clamp(directional_gap(box,f)/.072f,0,4);
    output[1]=p::clamp(directional_gap(box,s)/.072f,0,4);
    output[2]=p::clamp(directional_gap(box,s*-1)/.072f,0,4);
    output[3]=p::dot(normal,f);output[4]=p::dot(normal,s);
    output[5]=body.velocity[4]/4.f;
    output[6]=p::dot(motion.velocity_m_s,f)/reference_speed_m_s;
    output[7]=p::dot(motion.velocity_m_s,s)/reference_speed_m_s;
    for(int i=0;i<sensors;++i)if(!p::finite_number(output[i]))return false;return true;
}
CUDARL_HD inline Case case_for(int id,bool withheld=false) {
    Case out;out.id=id;out.withheld=withheld;out.pose=id%2?3:0;
    float angle=0,offset=0;
    if(id<4)out.yaw=(withheld?model::pi*.25f:0)+id*model::pi*.5f;
    else {
        const int k=withheld?id-4:(id-4)/3,g=withheld?0:(id-4)%3;
        angle=(withheld?model::pi*.25f:0)+k*model::pi*.5f;
        out.com_m.x=radius_x_m*.55f*cosf(angle);out.com_m.z=radius_z_m*.55f*sinf(angle);
        const auto normal=p::V{out.com_m.x/(radius_x_m*radius_x_m),0,out.com_m.z/(radius_z_m*radius_z_m)};
        offset=withheld?(k%2?1.f:-1.f)*model::pi*.25f:(g-1)*model::pi/3.f;
        out.yaw=atan2f(-normal.z,normal.x)+offset;
    }
    out.com_m.y=-.08f;return out;
}
CUDARL_HD inline bool initialize(int id,World& world,Task& task,int maximum_controls=controls,bool withheld=false) {
    if(id<0||id>=(withheld?withheld_cases:training_cases)||maximum_controls<0)return false;
    const auto fixture=case_for(id,withheld);auto next=old::initial_world(fixture.pose);next.arena=old::long_swim_arena();
    // Initialization only. Every later turn must come from internal motor torques.
    next.physical.body.orientation=p::axis_rotation({0,1,0},fixture.yaw);
    const auto initial= motor::motion(next.physical.body).com_model;
    next.physical.body.position+=fixture.com_m/p::scale-initial;
    const auto b=bounds(next.physical.body);
    if(!turning_task_v1::in_domain(next)||!b.finite||b.inset_radius_squared>=1)return false;
    Task t;t.maximum_controls=maximum_controls;t.fixed_depth_model=motor::motion(next.physical.body).com_model.y;
    t.last_world_control=next.physical.control_steps;t.last_com_model=motor::motion(next.physical.body).com_model;
    t.minimum_clearance_m=b.clearance_m;world=next;task=t;return true;
}
CUDARL_HD inline bool observe(const Task& task,const World& world,float* observation,float* raw) {
    if(task.done||!observation||!raw||!turning_task_v1::in_domain(world))return false;
    auto config=old::config_for_speed(reference_speed_m_s);config.observation_schema=4;config.maximum_controls=controls;
    old::ForwardTask view;if(!old::begin_forward(world,config,view))return false;
    const auto motion=motor::motion(world.physical.body);const auto f=forward(world.physical.body);
    // A read-only local-forward motor request: no horizontal destination/lane.
    // Original world-space depth and uninterrupted physical/task clocks remain.
    view.start_com_model={motion.com_model.x,task.fixed_depth_model,motion.com_model.z};
    view.request.heading=f;view.request.terminal_velocity=f*reference_speed_m_s;
    view.request.position=view.start_com_model+f*(.072f/p::scale);
    view.controls=task.controls>1000000?1000000:static_cast<int>(task.controls);
    return old::observe_appended(view,world,config,observation)&&raw_sensors(world,raw);
}

CUDARL_HD inline bool step_control(World& w,const float* command,StepInfo& info) {
    info={};info.before=info.candidate=w.physical.body;
    const auto initial=bounds(w.physical.body);
    if(!command||w.physical.paused||!turning_task_v1::in_domain(w)||!initial.finite||initial.inset_radius_squared>=1) {
        info.hard_fault=true;info.failure_code=101;return false;
    }
    for(int j=0;j<4;++j)if(!p::finite_number(command[j])||fabsf(command[j])>1){info.hard_fault=true;info.failure_code=102;return false;}
    auto& state=w.physical;state.telemetry={};state.mean_squared_torque=0;
    const float work_before=state.body.work;
    for(int j=0;j<4;++j)state.raw_action[j]=command[j];
    for(int i=0;i<40;++i){state.actual_torques[i]=0;state.requested_torques[i]=0;}
    for(int k=0;k<10;++k) {
        info.before=info.candidate=state.body;const float h=.005f;model::Reference reference;
        if(!joint::filter_targets(state.held_targets,w.reference_rate,command,reference)) {w.failure_code=10;state.invalid=true;info.hard_fault=true;info.failure_code=10;return false;}
        model::Telemetry telemetry;++w.attempted_substeps;++info.substeps;info.attempted_substeps=info.substeps;
        const bool advanced=model::substep(state.body,reference,h,state.fluid,telemetry,true,w.arena,&info.candidate);info.telemetry=telemetry;
        if(!advanced){w.failure_code=telemetry.failure_code;state.invalid=true;info.hard_fault=true;info.failure_code=w.failure_code;return false;}
        w.signed_work+=telemetry.signed_work;w.positive_work+=telemetry.positive_work;w.fluid_loss+=telemetry.fluid_loss;
        w.numerical_loss+=telemetry.numerical_loss;w.external_work+=telemetry.external_work;
        w.last_energy_error=model::kinetic(state.body)-w.signed_work+w.fluid_loss+w.numerical_loss-w.external_work;
        if(fabs(w.last_energy_error)>.02*(w.positive_work+w.fluid_loss)+2e-10){w.failure_code=9;state.invalid=true;info.hard_fault=true;info.failure_code=9;return false;}
        ++state.substeps;++state.telemetry.executed_substeps;++info.accepted_substeps;
        state.time_seconds=static_cast<double>(state.substeps)*h;
        state.telemetry.minimum_floor_gap=fminf(state.telemetry.minimum_floor_gap,telemetry.minimum_gap);
        state.telemetry.minimum_other_gap=fminf(state.telemetry.minimum_other_gap,telemetry.minimum_gap);
        for(int j=0;j<4;++j){state.held_targets[j]=reference.angle[j];const float normalized=telemetry.torque[j]/p::torque;
            state.last_actual_torques[j]=normalized;state.actual_torques[k*4+j]=normalized;state.requested_torques[k*4+j]=telemetry.requested[j];
            state.mean_squared_torque+=normalized*normalized;}
        info.seconds=info.dt=info.accepted_substeps*h;
        // Reward measurement uses the recorded float PRE/POST work values.
        // The independent physical energy audit still uses w.positive_work.
        info.positive_work_delta=double(state.body.work)-double(work_before);
        const auto boundary=bounds(state.body);
        if(!boundary.finite||boundary.active_radius_squared>1){w.failure_code=103;state.unsafe=true;info.hard_fault=true;info.failure_code=103;return false;}
        if(boundary.inset_radius_squared>=1){info.boundary_terminal=true;break;}
    }
    state.mean_squared_torque/=float(info.accepted_substeps*4);++state.control_steps;
    info.valid=joint::in_domain(w);if(!info.valid){info.hard_fault=true;info.failure_code=104;}return info.valid;
}
CUDARL_HD inline Metrics measure(const Task& task,const World& world) {
    Metrics out;const auto& body=world.physical.body;const auto motion=motor::motion(body);const auto f=forward(body),s=side(f);
    out.forward_m_s=p::dot(motion.velocity_m_s,f);out.side_m_s=p::dot(motion.velocity_m_s,s);
    out.depth_error_m=(motion.com_model.y-task.fixed_depth_model)*p::scale;
    const auto up=body.orientation.inverse({0,1,0});out.roll_deg=atan2f(-up.z,up.y)*180/model::pi;
    out.pitch_deg=atan2f(up.x,sqrtf(up.y*up.y+up.z*up.z))*180/model::pi;
    out.tilt_deg=acosf(p::clamp(body.orientation.rotate({0,1,0}).y,-1,1))*180/model::pi;
    out.clearance_m=bounds(body).clearance_m;return out;
}
CUDARL_HD inline Transition record(Task& task,const World& world,const StepInfo& info) {
    Transition out;
    if(task.done||!info.valid||info.hard_fault||info.accepted_substeps<1||info.accepted_substeps>10||
       info.substeps!=info.accepted_substeps||world.physical.control_steps!=task.last_world_control+1||!turning_task_v1::in_domain(world))return out;
    if(!info.boundary_terminal&&info.accepted_substeps!=10)return out;
    out.metrics=measure(task,world);out.seconds=out.dt=info.seconds;out.substeps=info.substeps;
    const auto& z=out.metrics;const double vf=(z.forward_m_s-.008f)/.008f,vs=z.side_m_s/.008f,
        depth=z.depth_error_m/.001f,roll=z.roll_deg/10.f,pitch=z.pitch_deg/10.f;
    out.reward=static_cast<float>(info.seconds*(1-vf*vf-.25*vs*vs-.05*depth*depth-.05*roll*roll-.01*pitch*pitch)-6250*info.positive_work_delta-(info.boundary_terminal?4.:0.));
    if(!p::finite_number(out.reward))return out;
    ++task.controls;task.last_world_control=world.physical.control_steps;task.elapsed_seconds+=info.seconds;task.reward_sum+=out.reward;
    task.maximum_depth_m=fmaxf(task.maximum_depth_m,fabsf(z.depth_error_m));task.maximum_tilt_deg=fmaxf(task.maximum_tilt_deg,z.tilt_deg);
    task.minimum_clearance_m=fminf(task.minimum_clearance_m,z.clearance_m);
    const auto com=motor::motion(world.physical.body).com_model;const auto delta=com-task.last_com_model;
    task.path_length_m+=p::norm({delta.x,0,delta.z})*p::scale;task.last_com_model=com;
    task.boundary_terminal=info.boundary_terminal;
    task.done=info.boundary_terminal||(task.maximum_controls>0&&task.controls>=static_cast<uint64_t>(task.maximum_controls));
    task.reason=info.boundary_terminal?Boundary:task.done?Deadline:Running;
    out.accepted=true;out.done=task.done;out.boundary_terminal=task.boundary_terminal;out.reason=task.reason;return out;
}
}
#endif
