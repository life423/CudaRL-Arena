#ifndef CUDARL_TOOLS_MOTOR_BENCHMARK_H
#define CUDARL_TOOLS_MOTOR_BENCHMARK_H
#include "envs/swimmer/command_motor.h"
#include "envs/swimmer/shelter_motor.h"
#include "nn/mlp_cpu.h"
#include "rl/swimmer_spec.h"
#include <array>
#include <deque>
#include <vector>

namespace cudarl::motor_benchmark {
using State=command_motor::State;
using Request=command_motor::Request;
using Command=Request;
using Four=std::array<float,4>;
constexpr uint32_t protocol=1,reference_seed=0x4D072600u,development_seed=0x4D072603u,final_seed=0x4D072604u;
constexpr int surface_count=216,hold_ticks=200,total_budget_ticks=600;
constexpr float length=.072f,position_band=.0072f,heading_band=.1745329252f;
enum class Kind:int {Move,Brake,Turn,Hold,Recover};
enum class Suite:int {Reference,Development,Final};
enum class TargetMode:int {Original55,Calibrated59,Candidate59};
inline const char* name(Kind k) {return k==Kind::Move?"move":(k==Kind::Brake?"brake":(k==Kind::Turn?"turn":(k==Kind::Hold?"hold":"recover")));}
struct Fixture {
    State initial{};
    Kind kind=Kind::Move;
    uint32_t seed=reference_seed;
    int id=0,warmup_ticks=0,evaluation_ticks=total_budget_ticks;
    int pulse_start=-1,pulse_ticks=0;
    deep::V pulse_velocity{};
    float requested_speed=0,actual_entry_speed=0;
    bool needs_warmup=false,qualified=true;
};
struct SurfacePoint {deep::V position{},velocity{};}; // SI metres and m/s.
struct Sample {
    deep::V head_position{},com_position{},tail_position{}; // SI metres.
    float head_speed=0,com_speed=0,tail_speed=0,point_speed=0,point_rms_speed=0;
    float angular_speed=0,joint_rate=0,goal_error=0,heading_error=0,clearance=1e6f;
    float joint[4]{};
    bool finite=true;
};
CUDARL_HD inline float heading_error(const deep::BodyState& b,deep::V requested) {
    return acosf(deep::clamp(deep::dot(b.orientation.rotate({1,0,0}),requested),-1,1));
}
CUDARL_HD inline Sample measure(const State& state,SurfacePoint* points=nullptr) {
    Sample out;const auto& b=state.body;out.finite=command_motor::finite(state);
    out.head_position=b.position*deep::scale;out.head_speed=deep::norm({b.velocity[0],b.velocity[1],b.velocity[2]});
    out.angular_speed=deep::norm({b.velocity[3],b.velocity[4],b.velocity[5]});
    out.goal_error=deep::norm((b.position-state.request.position)*deep::scale);out.heading_error=heading_error(b,state.request.heading);
    for(int j=0;j<4;++j) {out.joint[j]=b.angle[j];out.joint_rate=fmaxf(out.joint_rate,fabsf(b.velocity[6+j]));}
    const auto shape=deep::shape(b);deep::V com{},com_velocity{};float mass=0,squared_speed=0;int index=0;
    const deep::V normals[6]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    for(int link=0;link<3;++link) {
        const auto& l=shape.link[link];const auto center_velocity=deep::multiply(l.linear,b.velocity);
        com+=l.center*l.mass;com_velocity+=center_velocity*l.mass;mass+=l.mass;
        // Exact continuous ellipsoid support prevents a head grazing a plane
        // between the nine recorded axial slices from passing the safety gate.
        if(link==0) {
            const float plane[6]={l.center.x+4.5f*deep::scale,4.5f*deep::scale-l.center.x,l.center.y+4.f*deep::scale,
                -l.center.y,l.center.z+3.5f*deep::scale,3.5f*deep::scale-l.center.z};
            for(int face=0;face<6;++face) {
                const auto n=l.q.inverse(normals[face]);const float x=n.x*l.length*.5f,y=n.y*.0064f,z=n.z*.008f;
                out.clearance=fminf(out.clearance,plane[face]-sqrtf(x*x+y*y+z*z));
            }
        }
        for(int axial=0;axial<9;++axial) {
            const float f=(axial-4)*.25f,profile=link==0?sqrtf(fmaxf(0.f,1.f-f*f)):1.f;
            const auto center=l.center+l.q.rotate({f*l.length*.5f,0,0});
            const float plane[6]={center.x+4.5f*deep::scale,4.5f*deep::scale-center.x,center.y+4.f*deep::scale,
                -center.y,center.z+3.5f*deep::scale,3.5f*deep::scale-center.z};
            for(int face=0;face<6;++face) out.clearance=fminf(out.clearance,plane[face]-deep::arch_slice_support(l,link,f,normals[face]));
            const float ry=(link==0?.0064f:(link==2?.006f:.0015f))*profile,rz=(link==0?.008f:.0015f)*profile;
            for(int rim=0;rim<8;++rim) {
                const float angle=rim*.7853981633974483f;
                const auto offset=l.q.rotate({f*l.length*.5f,ry*cosf(angle),rz*sinf(angle)});
                SurfacePoint p{l.center+offset,center_velocity+deep::cross(l.omega,offset)};
                const float speed=deep::norm(p.velocity);squared_speed+=speed*speed;out.point_speed=fmaxf(out.point_speed,speed);
                if(link==2) out.tail_speed=fmaxf(out.tail_speed,speed);
                if(points) points[index]=p;++index;
                out.finite=out.finite&&command_motor::finite_vector(p.position)&&command_motor::finite_vector(p.velocity);
            }
        }
    }
    out.com_position=com/mass;out.com_speed=deep::norm(com_velocity/mass);out.point_rms_speed=sqrtf(squared_speed/surface_count);
    const auto& tail=shape.link[2];out.tail_position=tail.center+tail.q.rotate({-tail.length*.5f,0,0});
    out.finite=out.finite&&command_motor::finite_number(out.clearance)&&command_motor::finite_number(out.heading_error);return out;
}
CUDARL_HD inline deep::Physics physics_for(const Fixture& f,int step) {
    auto p=command_motor::physics();
    if(f.pulse_start>=0 && step>=f.pulse_start && step<f.pulse_start+f.pulse_ticks) p.water_velocity=f.pulse_velocity;
    return p;
}
CUDARL_HD inline command_motor::Result advance(State& state,const float* action,const Fixture& f) {
    return command_motor::advance(state,action,physics_for(f,state.steps));
}
inline int suite_count(Suite suite) {return suite==Suite::Final?64:16;}
inline Fixture make_fixture(uint32_t seed,Suite suite,int id) {
    if(id<0 || id>=suite_count(suite)) throw std::invalid_argument("Invalid motor benchmark fixture index");
    Fixture f;f.id=id;f.seed=seed;auto& s=f.initial;s.seed=seed;s.environment=static_cast<uint32_t>(id);
    int local=id;
    if(suite==Suite::Final) {
        if(id<24) f.kind=Kind::Move;
        else if(id<32) {f.kind=Kind::Turn;local=id-24;}
        else if(id<40) {f.kind=Kind::Brake;local=id-32;}
        else if(id<56) {f.kind=Kind::Hold;local=id-40;}
        else {f.kind=Kind::Recover;local=id-56;}
    } else {
        if(id<4) f.kind=Kind::Move;
        else if(id<12) {f.kind=Kind::Brake;local=id-4;}
        else if(id<14) {f.kind=Kind::Turn;local=id-12;}
        else {f.kind=Kind::Recover;local=id-14;}
    }
    const auto draw=[&](uint32_t channel) {return rng::uniform(seed,static_cast<uint32_t>(id),channel);};
    float yaw=(f.kind==Kind::Brake?(local/2)%4:local%4)*1.57079632679f;
    if(suite!=Suite::Reference) yaw+=(draw(1)*2-1)*.2f;
    const float pitch=f.kind==Kind::Move?((local%4==1?.25f:(local%4==3?-.25f:0.f))):0.f;
    s.body.position={(draw(2)*2-1)*.15f,-2.f+(draw(3)*2-1)*.15f,(draw(4)*2-1)*.1f};
    s.body.orientation=deep::axis_rotation({0,1,0},yaw)*deep::axis_rotation({0,0,1},pitch);
    s.request.position=s.body.position;s.request.heading=deep::axis_rotation({0,1,0},yaw).rotate({1,0,0});
    if(f.kind==Kind::Move) {
        const float distance=(.5f+.5f*draw(5))*length/deep::scale;
        // Start/approach can pitch and change depth; the quiet terminal
        // heading stays level because the existing head buoyancy offset
        // supplies a restoring torque at static pitch.
        s.request.position+=s.body.orientation.rotate({1,0,0})*distance;
    } else if(f.kind==Kind::Brake) {
        f.needs_warmup=true;f.qualified=false;f.requested_speed=local%2==0?.01f:.03f;
        // Warmup target is non-food metadata. It is not consumed on arrival.
        s.request.position+=s.request.heading*(2.f*length/deep::scale);
    } else if(f.kind==Kind::Turn) {
        s.request.heading=(deep::axis_rotation({0,1,0},local%2==0?1.57079632679f:-1.57079632679f)*s.body.orientation).rotate({1,0,0});
    } else if(f.kind==Kind::Hold) {
        for(int j=0;j<4;++j) s.body.angle[j]=(draw(10u+static_cast<uint32_t>(j))*2-1)*.12f;
    } else {
        f.pulse_start=20;f.pulse_ticks=10;
        f.pulse_velocity=local%2==0?s.body.orientation.rotate({0,0,.005f}):deep::V{0,.003f,0};
    }
    return f;
}

// Explicit frozen adapters. Original55 has a virtual target but no desired
// heading/stop semantics. Calibrated59 receives the prior cover command with
// its identity cover means; Candidate59 receives the new persistent contract.
// No adapter changes the body's position, applies a rescue, or consumes food.
class FrozenMotor {
    TargetMode mode_;nn::CpuMlp<float> model_;
public:
    FrozenMotor(const std::vector<float>& weights,TargetMode mode):mode_(mode),model_(rl::swimmer_specification(4,mode==TargetMode::Original55?55:59)) {
        if(weights.size()!=model_.params().size()) throw std::invalid_argument("Frozen motor adapter parameter dimensions");model_.params()=weights;
    }
    FrozenMotor(const std::vector<float>& weights,bool legacy59,shelter_motor::ActuatorScale={}):FrozenMotor(weights,legacy59?TargetMode::Calibrated59:TargetMode::Original55) {}
    Four action(const State& state) {
        float observation[59]{};
        if(mode_==TargetMode::Candidate59) command_motor::observe(state,observation);
        else {
            arch::State old;old.habitat.arch=false;old.pond.body=state.body;old.pond.target=0;old.pond.decision=false;
            old.pond.pellets[0].active=true;old.pond.pellets[0].position=state.request.position;old.pond.pellets[0].velocity=state.request.velocity;
            for(int j=0;j<4;++j) old.pond.last_action[j]=state.previous_action[j];
            if(mode_==TargetMode::Original55) arch::observe(old,observation,{});
            else {shelter::Goal goal;goal.position=state.request.position;goal.velocity=state.request.velocity;goal.cover=0;goal.valid=true;
                shelter::motor_observe59(old,goal,state.request.heading,observation,{});}
        }
        model_.forward(observation,1);Four result{};for(int j=0;j<4;++j) result[static_cast<size_t>(j)]=model_.output()[j];return result;
    }
};
struct WarmupFrame {State state;Four action{};};
inline bool prepare_braking(Fixture& f,FrozenMotor& original,std::vector<WarmupFrame>* trace=nullptr) {
    if(!f.needs_warmup) return f.qualified;
    auto state=f.initial;
    for(int tick=0;tick<200;++tick) {
        const auto action=original.action(state);if(trace) trace->push_back({state,action});
        const auto result=command_motor::advance(state,action.data());f.warmup_ticks=tick+1;
        const auto sample=measure(state);
        if(result.invalid || !sample.finite) throw std::runtime_error("Nonfinite original55 braking warmup; no further physical trials permitted");
        if(sample.clearance<0) return false;
        const auto velocity=deep::V{state.body.velocity[0],state.body.velocity[1],state.body.velocity[2]};
        const float speed=deep::norm(velocity),low=f.requested_speed<.02f?.01f:.028f,high=f.requested_speed<.02f?.012f:.03f;
        if(speed>=low && speed<=high && deep::dot(velocity,state.body.orientation.rotate({1,0,0}))>=.7f*speed) {
            f.actual_entry_speed=speed;f.initial=state;f.initial.steps=0;
            f.initial.request.position=state.body.position;f.initial.request.velocity={};
            auto level=state.body.orientation.rotate({1,0,0});level.y=0;
            f.initial.request.heading=deep::norm(level)>1e-6f?level/deep::norm(level):deep::V{1,0,0};
            f.evaluation_ticks=total_budget_ticks-f.warmup_ticks;f.qualified=true;return true;
        }
    }
    return false;
}

// One fixed, deliberately conventional reference: a two-link travelling-wave
// joint PD controller with goal/heading steering, reverse-wave braking and
// joint damping near a settled goal. It is not a learned policy or fallback.
// Constants are fixed before outcomes; no parameter search is performed.
class ConventionalReference {
public:
    Four action(const State& s) const {
        const auto& b=s.body;const auto q=b.orientation;
        const auto position=q.inverse((s.request.position-b.position)*deep::scale);
        const auto local_velocity=q.inverse(deep::V{b.velocity[0],b.velocity[1],b.velocity[2]}-s.request.velocity);
        const auto heading=q.inverse(s.request.heading);const float distance=deep::norm(position);
        const float desired_speed=fminf(.03f,sqrtf(fmaxf(0.f,2.f*.015f*distance)));
        const bool braking=local_velocity.x>desired_speed+.002f;
        const float face=atan2f(heading.z,heading.x),pitch=atan2f(heading.y,sqrtf(heading.x*heading.x+heading.z*heading.z));
        const float turn_yaw=distance>position_band?atan2f(position.z,position.x):face;
        const float turn_pitch=distance>position_band?atan2f(position.y,sqrtf(position.x*position.x+position.z*position.z)):pitch;
        const bool quiet=distance<=position_band && deep::norm(local_velocity)<=.002f && fabsf(face)<=heading_band && fabsf(pitch)<=heading_band;
        const float amplitude=quiet?0.f:(braking?.45f:(distance>position_band?.65f:.25f));
        const float phase=6.28318530718f*2.f*s.steps*deep::dt,lag=braking?1.1f:-1.1f;
        const float bias_yaw=deep::clamp(.35f*turn_yaw,-.3f,.3f),bias_pitch=deep::clamp(-.35f*turn_pitch,-.3f,.3f);
        const float target[4]={bias_yaw+amplitude*sinf(phase),bias_pitch,bias_yaw+amplitude*sinf(phase+lag),bias_pitch};
        const float target_rate[4]={amplitude*12.5663706144f*cosf(phase),0,amplitude*12.5663706144f*cosf(phase+lag),0};
        Four out{};for(int j=0;j<4;++j) out[static_cast<size_t>(j)]=deep::clamp(1.5f*(deep::clamp(target[j],-.95f,.95f)-b.angle[j])+.25f*(target_rate[j]-b.velocity[6+j]),-1,1);
        return out;
    }
};

inline bool settled(const Sample& s) {
    return s.finite&&s.clearance>=0&&s.goal_error<=position_band&&s.heading_error<=heading_band&&
        s.head_speed<=.0005f&&s.com_speed<=.0005f&&s.point_speed<=.001f&&s.angular_speed<=.02f&&s.joint_rate<=.05f;
}
struct Progress {
    std::deque<Sample> window;
    deep::V initial_head{},last_head{},initial_direction{1,0,0};
    float initial_error=0,final_error=0,last_heading_error=0,path=0,minimum_clearance=1e6f,max_speed=0,max_heading_error=0,max_overshoot=0,max_cross_track=0,max_translation=0;
    float half_time=-1,reach_time=-1,heading_time=-1,settle_time=-1,stop_time=-1,stop_distance=-1,hold_seconds=0;
    float completed_hold_start=-1;
    int stop_streak=0,samples=0,first_failure_step=-1,reversals=0,heading_reversals=0,last_direction=0,last_heading_direction=0;
    bool finite=true,penetrated=false;
};
inline void assess(Progress& p,const State& state,const Sample& s,const Fixture& f) {
    const float time=state.steps*deep::dt;
    if(p.samples==0) {
        p.initial_head=p.last_head=s.head_position;p.initial_error=s.goal_error;
        auto v=deep::V{state.body.velocity[0],state.body.velocity[1],state.body.velocity[2]};
        p.initial_direction=deep::norm(v)>1e-6f?v/deep::norm(v):state.request.heading;
    } else {
        p.path+=deep::norm(s.head_position-p.last_head);
        const float slope=s.goal_error-p.final_error,heading_slope=s.heading_error-p.last_heading_error;
        const int direction=slope>1e-5f?1:(slope< -1e-5f?-1:0),heading_direction=heading_slope>1e-4f?1:(heading_slope< -1e-4f?-1:0);
        if(direction) {if(p.last_direction&&direction!=p.last_direction) ++p.reversals;p.last_direction=direction;}
        if(heading_direction) {if(p.last_heading_direction&&heading_direction!=p.last_heading_direction) ++p.heading_reversals;p.last_heading_direction=heading_direction;}
    }
    p.final_error=s.goal_error;p.last_heading_error=s.heading_error;p.max_translation=fmaxf(p.max_translation,deep::norm(s.head_position-p.initial_head));
    p.last_head=s.head_position;++p.samples;p.finite=p.finite&&s.finite;
    p.minimum_clearance=fminf(p.minimum_clearance,s.clearance);p.penetrated=p.penetrated||s.clearance<0;
    if((!s.finite || s.clearance<0) && p.first_failure_step<0) p.first_failure_step=state.steps;
    p.max_speed=fmaxf(p.max_speed,s.point_speed);p.max_heading_error=fmaxf(p.max_heading_error,s.heading_error);
    const auto from_goal=s.head_position-state.request.position*deep::scale;const float along=deep::dot(from_goal,p.initial_direction);
    p.max_overshoot=fmaxf(p.max_overshoot,along);p.max_cross_track=fmaxf(p.max_cross_track,sqrtf(fmaxf(0.f,deep::dot(from_goal,from_goal)-along*along)));
    if(p.half_time<0 && s.goal_error<=.5f*p.initial_error) p.half_time=time;
    if(p.reach_time<0 && s.goal_error<=position_band) p.reach_time=time;
    if(p.heading_time<0 && s.heading_error<=heading_band) p.heading_time=time;
    if(state.steps>0) p.stop_streak=s.head_speed<=.002f&&s.com_speed<=.002f?p.stop_streak+1:0;
    if(p.stop_time<0 && p.stop_streak>=10) {p.stop_time=time;p.stop_distance=p.path;}
    const int recovery_end=f.pulse_start>=0?f.pulse_start+f.pulse_ticks:0;
    if(settled(s) && state.steps>=recovery_end) {
        if(p.settle_time<0) p.settle_time=time;
        p.window.push_back(s);if(p.window.size()>201) p.window.pop_front();
        float low[4],high[4];for(int j=0;j<4;++j) low[j]=high[j]=p.window.front().joint[j];bool stable=true;
        for(const auto& point:p.window) {
            if(deep::norm(point.head_position-p.window.front().head_position)>.0036f) stable=false;
            for(int j=0;j<4;++j) {low[j]=fminf(low[j],point.joint[j]);high[j]=fmaxf(high[j],point.joint[j]);}
        }
        for(int j=0;j<4;++j) if(high[j]-low[j]>.05f) stable=false;
        if(stable) {
            p.hold_seconds=fmaxf(p.hold_seconds,static_cast<float>(p.window.size()-1)*deep::dt);
            if(p.window.size()==201 && p.completed_hold_start<0) p.completed_hold_start=time-hold_ticks*deep::dt;
        }
    } else p.window.clear();
}
struct Outcome {
    bool qualified=false,safety=false,motion=false,settle=false,hold=false,passed=false;
    float half_time=-1,reach_time=-1,heading_time=-1,stop_time=-1,stop_distance=-1,settle_time=-1,hold_seconds=0;
};
inline Outcome finish(const Progress& p,const Fixture& f) {
    Outcome out;out.qualified=f.qualified;out.safety=p.finite&&!p.penetrated&&p.minimum_clearance>=0;
    out.half_time=p.half_time;out.reach_time=p.reach_time;out.heading_time=p.heading_time;out.stop_time=p.stop_time;out.stop_distance=p.stop_distance;out.settle_time=p.completed_hold_start;out.hold_seconds=p.hold_seconds;
    out.motion=true;
    if(f.kind==Kind::Move) out.motion=p.half_time>=0&&p.half_time<=4&&p.reach_time>=0&&p.reach_time<=12;
    if(f.kind==Kind::Brake) out.motion=f.actual_entry_speed>=.01f&&f.actual_entry_speed<=.03f&&p.stop_time>=0&&p.stop_time<=3&&p.stop_distance<=length;
    if(f.kind==Kind::Turn) out.motion=p.heading_time>=0&&p.heading_time<=5;
    const float deadline=f.kind==Kind::Recover?(f.pulse_start+f.pulse_ticks)*deep::dt+5.f:15.f;
    out.settle=p.completed_hold_start>=0&&p.completed_hold_start<=deadline;out.hold=p.hold_seconds>=10.f;
    out.passed=out.qualified&&out.safety&&out.motion&&out.settle&&out.hold;return out;
}
inline float path_efficiency(const Progress& p) {return p.path>0?fmaxf(0.f,p.initial_error-p.final_error)/p.path:0.f;}
// Semantic version and every fixed numerical fixture/gate/reference constant.
// The actual prepared fixture bytes have a separate checksum because braking
// starts are captured from the supplied original motor, not invented velocities.
inline uint64_t rules_fingerprint() {
    const float values[]={static_cast<float>(protocol),16,16,64,216,600,.05f,.072f,.0072f,.1745329252f,
        .0036f,.05f,.0005f,.001f,.02f,200,4,12,3,5,15,10,.002f,.01f,.012f,.028f,.03f,
        20,10,.005f,.003f,.15f,.1f,.2f,.25f,.12f,.5f,1.f,200,.7f,2.f,.015f,
        2.f,.65f,.45f,.25f,1.1f,.35f,.3f,.95f,1.5f,.25f,1e-5f,1e-4f,1.f}; // continuous head-plane support v1
    uint64_t hash=14695981039346656037ull;
    const auto* bytes=reinterpret_cast<const uint8_t*>(values);
    for(size_t i=0;i<sizeof(values);++i) {hash^=bytes[i];hash*=1099511628211ull;}
    const uint32_t seeds[]={reference_seed,development_seed,final_seed};
    bytes=reinterpret_cast<const uint8_t*>(seeds);
    for(size_t i=0;i<sizeof(seeds);++i) {hash^=bytes[i];hash*=1099511628211ull;}
    return hash;
}
}
#endif
