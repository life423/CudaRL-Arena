#ifndef CUDARL_TOOLS_MOTOR_V2_BENCHMARK_H
#define CUDARL_TOOLS_MOTOR_V2_BENCHMARK_H
#include "envs/swimmer/motor_v2.h"
#include "motor_benchmark.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>

// Frozen evaluation only: no goal chooser, learning, hidden braking or rescue.
// Four action coordinates operate two universal joints with two axes each.
namespace cudarl::motor_v2_benchmark {
namespace mv=motor_v2;
namespace mc=motor_contact_v2;
namespace old=motor_benchmark;
constexpr uint32_t protocol=2,final_seed=0x4D082604u;
constexpr int count=64,episode_steps=600,hold_intervals=200;
constexpr float position_band=.0072f,heading_band=.1745329252f;
constexpr float residual_speed=.002f,residual_angular=.05f,unsupported_gap=.002f;
constexpr float current_speed=.005f;
constexpr float rest_touch_band=.000005f,minimum_support_force=1e-8f;
enum class Kind:int {Move,Turn,Brake,Arrival,CalmHold,CurrentHold,BottomRest};
inline const char* name(Kind value) {
    switch(value) {
    case Kind::Move:return "move";case Kind::Turn:return "turn";case Kind::Brake:return "brake";
    case Kind::Arrival:return "slow_arrival";case Kind::CalmHold:return "unsupported_calm_hold";
    case Kind::CurrentHold:return "unsupported_current_hold";default:return "bottom_rest";
    }
}
struct Fixture {
    mv::State initial{};
    Kind kind=Kind::Move;
    uint32_t seed=final_seed;
    int id=0,steps=episode_steps,source_capture=-1;
    float actual_entry_speed=0;
    bool qualified=true;
};
inline std::array<old::Fixture,16> read_archived_captures(const std::vector<uint8_t>& bytes) {
    checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x434D4649u||in.value<uint32_t>()!=1)
        throw std::runtime_error("Motor-v2 requires immutable reference-v1 complete captures");
    (void)in.value<uint64_t>();
    if(in.value<uint64_t>()!=2073117933376950176ull)
        throw std::runtime_error("Motor-v2 archived capture source differs");
    const auto values=in.array<old::Fixture>(16);in.finish();
    if(values.size()!=16) throw std::runtime_error("Motor-v2 archived capture count differs");
    std::array<old::Fixture,16> result{};std::copy(values.begin(),values.end(),result.begin());
    for(int i=4;i<12;++i) {
        const auto& capture=result[static_cast<size_t>(i)];
        const auto speed=deep::norm(deep::V{capture.initial.body.velocity[0],capture.initial.body.velocity[1],capture.initial.body.velocity[2]});
        if(capture.id!=i||capture.kind!=old::Kind::Brake||!capture.qualified||capture.initial.steps!=0||
           !command_motor::finite(capture.initial)||speed<.01f||speed>.03f)
            throw std::runtime_error("Motor-v2 actual-swimming capture failed its fixed identity/speed gate");
    }
    return result;
}
inline std::array<Fixture,count> make_fixtures(uint32_t seed,const std::vector<uint8_t>& archived) {
    const auto captures=read_archived_captures(archived);
    std::array<Fixture,count> result{};
    for(int id=0;id<count;++id) {
        auto& f=result[static_cast<size_t>(id)];f.id=id;f.seed=seed;
        int local=id;
        if(id<8) f.kind=Kind::Move;
        else if(id<16) {f.kind=Kind::Turn;local=id-8;}
        else if(id<24) {f.kind=Kind::Brake;local=id-16;}
        else if(id<32) {f.kind=Kind::Arrival;local=id-24;}
        else if(id<44) {f.kind=Kind::CalmHold;local=id-32;}
        else if(id<56) {f.kind=Kind::CurrentHold;local=id-44;}
        else {f.kind=Kind::BottomRest;local=id-56;}
        const auto draw=[&](uint32_t channel) {return rng::uniform(seed,static_cast<uint32_t>(id),channel);};
        auto& state=f.initial;auto& task=state.task;
        task.seed=seed;task.environment=static_cast<uint32_t>(id);task.episode=0;task.steps=0;
        state.lesson=static_cast<int>(f.kind);
        const float yaw=local*1.57079632679f+(draw(1)*2-1)*.2f;
        const float pitch=f.kind==Kind::Move?(local%4==1?.20f:(local%4==3?-.20f:0.f)):0.f;
        task.body.position={(draw(2)*2-1)*.2f,-2.f+(draw(3)*2-1)*.15f,(draw(4)*2-1)*.15f};
        task.body.orientation=deep::axis_rotation({0,1,0},yaw)*deep::axis_rotation({0,0,1},pitch);
        task.request.position=task.body.position;task.request.velocity={};
        task.request.heading=deep::axis_rotation({0,1,0},yaw).rotate({1,0,0});
        if(f.kind==Kind::Move||f.kind==Kind::Arrival) {
            const float distance=f.kind==Kind::Move?(.5f+.5f*draw(5))*.072f:(.25f+.25f*draw(5))*.072f;
            task.request.position+=task.body.orientation.rotate({1,0,0})*(distance/deep::scale);
        } else if(f.kind==Kind::Turn) {
            task.request.heading=deep::axis_rotation({0,1,0},yaw+(local%2==0?1.f:-1.f)*1.57079632679f).rotate({1,0,0});
        } else if(f.kind==Kind::Brake) {
            const auto& captured=captures[static_cast<size_t>(4+local)];
            // Reuse real swimming momentum/joint state, never invent or scale
            // velocity. A fixed translation centers each complete initial body
            // away from walls; it is not an in-trial teleport or recovery.
            task=captured.initial;task.seed=seed;task.environment=static_cast<uint32_t>(id);task.episode=0;task.steps=0;
            task.body.position={0,-2,0};task.request.position=task.body.position;task.request.velocity={};
            auto level=task.body.orientation.rotate({1,0,0});level.y=0;
            task.request.heading=level/fmaxf(deep::norm(level),1e-6f);
            f.source_capture=4+local;
            f.actual_entry_speed=deep::norm(deep::V{task.body.velocity[0],task.body.velocity[1],task.body.velocity[2]});
        } else if(f.kind==Kind::CurrentHold) {
            // Four body headings crossed with three fixed world flow vectors;
            // do not accidentally keep flow aligned with each body's heading.
            const float direction=(local/4)*1.57079632679f;
            state.water_velocity={current_speed*cosf(direction),0,current_speed*sinf(direction)};
        } else if(f.kind==Kind::BottomRest) {
            // Exact geometry-only placement at contact. No gravity, density,
            // body dimension or physical velocity is changed to create support.
            const auto gap=mc::inspect(task.body).floor_gap;
            task.body.position.y-=gap/deep::scale;task.request.position=task.body.position;
            task.body.velocity[1]=-.001f; // a declared gentle1mm/s approach to contact
        }
        if(f.kind==Kind::CalmHold) for(int axis=0;axis<4;++axis)
            task.body.angle[axis]=(draw(10u+static_cast<uint32_t>(axis))*2-1)*.08f;
        if(f.kind==Kind::CurrentHold) for(int axis=0;axis<4;++axis)
            task.body.angle[axis]=(draw(10u+static_cast<uint32_t>(axis))*2-1)*.04f;
        for(int axis=0;axis<4;++axis) state.held_target[axis]=deep::clamp(task.body.angle[axis],-deep::limit,deep::limit);
        const auto check=mc::inspect(task.body);
        if(check.invalid||check.excessive||check.escape||
           (f.kind!=Kind::BottomRest&&check.floor_gap<unsupported_gap))
            throw std::runtime_error("Motor-v2 registered initial fixture is invalid or physically unsafe");
    }
    return result;
}
struct Sample:old::Sample {
    float floor_gap=0,other_gap=0,torque_rms_uNm=0,positive_work_J=0;
    bool floor_contact=false,excessive=false,escape=false;
};
inline Sample measure(const mv::State& state) {
    Sample result;static_cast<old::Sample&>(result)=old::measure(state.task);
    const auto contact=mc::inspect(state.task.body);result.floor_gap=contact.floor_gap;result.other_gap=contact.min_other_gap;
    result.clearance=contact.min_gap;result.floor_contact=contact.floor_contact;result.excessive=contact.excessive;result.escape=contact.escape;
    result.finite=result.finite&&!contact.invalid;result.torque_rms_uNm=sqrtf(fmaxf(0,state.mean_squared_torque))*10.f;
    result.positive_work_J=state.task.body.work;return result;
}
inline bool endpoint_holding(const Sample& sample) {
    return sample.finite&&!sample.excessive&&!sample.escape&&sample.goal_error<=position_band&&
        sample.heading_error<=heading_band&&sample.head_speed<=residual_speed&&sample.com_speed<=residual_speed&&sample.angular_speed<=residual_angular;
}
inline bool interval_holding(const mv::State& state,const Sample& sample,const Fixture& fixture) {
    if(!endpoint_holding(sample)||state.substep_max_goal_error>position_band||state.substep_max_heading_error>heading_band||
       state.substep_max_head_speed>residual_speed||state.substep_max_com_speed>residual_speed||state.substep_max_omega>residual_angular) return false;
    if(fixture.kind==Kind::BottomRest)
        return state.substep_max_floor_gap<=rest_touch_band;
    return state.substep_min_floor_gap>=unsupported_gap&&state.substep_max_floor_force==0;
}
struct Progress {
    deep::V initial_head{},last_head{};
    float initial_error=0,final_error=0,path=0,minimum_floor_gap=1e6f,minimum_other_gap=1e6f;
    float maximum_position_error=0,maximum_heading_error=0,maximum_head_speed=0,maximum_com_speed=0,maximum_angular_speed=0;
    float maximum_point_speed=0,maximum_joint_rate=0,maximum_torque_rms_uNm=0,torque_squared_integral=0;
    float initial_work=0,positive_work_J=0,half_time=-1,reach_time=-1,heading_time=-1,slow_arrival_time=-1;
    float stop_time=-1,stop_distance=-1,hold_seconds=0,geometric_hold_seconds=0,completed_hold_start=-1,initial_entry_speed=0;
    float completed_hold_support_force_fraction=0;
    std::deque<uint8_t> support_force_window;
    int support_force_window_count=0;
    int samples=0,stop_streak=0,hold_streak=0,first_failure_step=-1,floor_contact_intervals=0,unsupported_contact_intervals=0;
    bool finite=true,unsafe=false;
};
inline void assess(Progress& progress,const mv::State& state,const Sample& sample,const Fixture& fixture) {
    const float time=state.task.steps*deep::dt;
    if(progress.samples==0) {
        progress.initial_head=progress.last_head=sample.head_position;progress.initial_error=sample.goal_error;
        progress.initial_work=sample.positive_work_J;progress.initial_entry_speed=sample.head_speed;
    } else {
        progress.path+=deep::norm(sample.head_position-progress.last_head);
        progress.torque_squared_integral+=deep::dt*state.mean_squared_torque;
        progress.floor_contact_intervals+=state.floor_contact_samples>0?1:0;
        progress.unsupported_contact_intervals+=state.floor_contact_samples>0&&fixture.kind!=Kind::BottomRest?1:0;
    }
    progress.last_head=sample.head_position;progress.final_error=sample.goal_error;++progress.samples;
    progress.finite=progress.finite&&sample.finite;
    progress.unsafe=progress.unsafe||sample.excessive||sample.escape||state.first_unsafe_substep>=0;
    if((!progress.finite||progress.unsafe)&&progress.first_failure_step<0) progress.first_failure_step=state.task.steps;
    progress.minimum_floor_gap=fminf(progress.minimum_floor_gap,sample.floor_gap);
    progress.minimum_other_gap=fminf(progress.minimum_other_gap,sample.other_gap);
    if(state.task.steps>0) {
        progress.minimum_floor_gap=fminf(progress.minimum_floor_gap,state.substep_min_floor_gap);
        progress.minimum_other_gap=fminf(progress.minimum_other_gap,state.substep_min_other_gap);
    }
    progress.maximum_position_error=fmaxf(progress.maximum_position_error,sample.goal_error);
    progress.maximum_heading_error=fmaxf(progress.maximum_heading_error,sample.heading_error);
    progress.maximum_head_speed=fmaxf(progress.maximum_head_speed,sample.head_speed);
    progress.maximum_com_speed=fmaxf(progress.maximum_com_speed,sample.com_speed);
    progress.maximum_angular_speed=fmaxf(progress.maximum_angular_speed,sample.angular_speed);
    progress.maximum_point_speed=fmaxf(progress.maximum_point_speed,sample.point_speed);
    progress.maximum_joint_rate=fmaxf(progress.maximum_joint_rate,sample.joint_rate);
    progress.maximum_torque_rms_uNm=fmaxf(progress.maximum_torque_rms_uNm,sample.torque_rms_uNm);
    progress.positive_work_J=fmaxf(progress.positive_work_J,sample.positive_work_J-progress.initial_work);
    if(progress.half_time<0&&sample.goal_error<=.5f*progress.initial_error) progress.half_time=time;
    if(progress.reach_time<0&&sample.goal_error<=position_band) progress.reach_time=time;
    if(progress.heading_time<0&&sample.heading_error<=heading_band) progress.heading_time=time;
    if(progress.slow_arrival_time<0&&sample.goal_error<=position_band&&sample.head_speed<=residual_speed&&sample.com_speed<=residual_speed)
        progress.slow_arrival_time=time;
    if(state.task.steps>0) {
        const bool slowed=state.substep_max_head_speed<=residual_speed&&state.substep_max_com_speed<=residual_speed;
        progress.stop_streak=slowed?progress.stop_streak+1:0;
        if(progress.stop_time<0&&progress.stop_streak>=10) {progress.stop_time=time;progress.stop_distance=progress.path;}
        const bool good_interval=interval_holding(state,sample,fixture)&&!progress.unsafe;
        progress.hold_streak=good_interval?progress.hold_streak+1:0;
        if(good_interval) {
            const uint8_t loaded=state.substep_max_floor_force>minimum_support_force?1:0;
            progress.support_force_window.push_back(loaded);progress.support_force_window_count+=loaded;
            if(progress.support_force_window.size()>hold_intervals) {
                progress.support_force_window_count-=progress.support_force_window.front();progress.support_force_window.pop_front();
            }
        } else {progress.support_force_window.clear();progress.support_force_window_count=0;}
        const bool actual_support=fixture.kind!=Kind::BottomRest||progress.support_force_window_count>0;
        progress.geometric_hold_seconds=fmaxf(progress.geometric_hold_seconds,progress.hold_streak*deep::dt);
        if(actual_support) progress.hold_seconds=fmaxf(progress.hold_seconds,progress.hold_streak*deep::dt);
        if(progress.hold_streak>=hold_intervals&&actual_support&&progress.completed_hold_start<0) {
            progress.completed_hold_start=time-hold_intervals*deep::dt;
            progress.completed_hold_support_force_fraction=static_cast<float>(progress.support_force_window_count)/hold_intervals;
        }
    }
}
struct Outcome {
    bool qualified=false,safety=false,motion=false,settle=false,hold=false,passed=false;
    float half_time=-1,reach_time=-1,heading_time=-1,stop_time=-1,stop_distance=-1,settle_time=-1,hold_seconds=0,slow_arrival_time=-1,support_force_fraction=0;
};
inline Outcome finish(const Progress& progress,const Fixture& fixture) {
    Outcome out;out.qualified=fixture.qualified;
    out.safety=progress.finite&&!progress.unsafe&&progress.minimum_floor_gap>=-mc::kOverlapTolerance&&progress.minimum_other_gap>=-mc::kOverlapTolerance;
    out.half_time=progress.half_time;out.reach_time=progress.reach_time;out.heading_time=progress.heading_time;
    out.stop_time=progress.stop_time;out.stop_distance=progress.stop_distance;out.settle_time=progress.completed_hold_start;
    out.hold_seconds=progress.hold_seconds;out.slow_arrival_time=progress.slow_arrival_time;out.motion=true;
    out.support_force_fraction=progress.completed_hold_support_force_fraction;
    if(fixture.kind==Kind::Move) out.motion=progress.half_time>=0&&progress.half_time<=4&&progress.reach_time>=0&&progress.reach_time<=12;
    if(fixture.kind==Kind::Turn) out.motion=progress.heading_time>=0&&progress.heading_time<=5;
    if(fixture.kind==Kind::Brake) out.motion=fixture.actual_entry_speed>=.01f&&fixture.actual_entry_speed<=.03f&&progress.stop_time>=0&&progress.stop_time<=3&&progress.stop_distance<=.072f;
    if(fixture.kind==Kind::Arrival) out.motion=progress.slow_arrival_time>=0&&progress.slow_arrival_time<=15;
    out.settle=progress.completed_hold_start>=0&&progress.completed_hold_start<=15;
    out.hold=progress.hold_seconds>=10&&progress.completed_hold_start>=0;out.passed=out.qualified&&out.safety&&out.motion&&out.settle&&out.hold;return out;
}
inline float path_efficiency(const Progress& progress) {
    return progress.path>0?fmaxf(0.f,progress.initial_error-progress.final_error)/progress.path:0.f;
}
inline uint64_t rules_fingerprint() {
    checkpoint::Writer out;out.value(protocol);out.value(final_seed);
    out.array(std::vector<float>{64,600,.05f,200,.072f,position_band,heading_band,residual_speed,residual_angular,unsupported_gap,
        current_speed,mc::kOverlapTolerance,rest_touch_band,minimum_support_force,4,12,5,3,.5f,15,10,8,8,8,8,12,12,8,.01f,.03f,.2f,.15f,.08f,.04f,.001f});
    uint64_t hash=14695981039346656037ull;for(uint8_t byte:out.data) {hash^=byte;hash*=1099511628211ull;}return hash;
}
}
#endif
