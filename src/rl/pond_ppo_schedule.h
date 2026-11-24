#ifndef CUDARL_RL_POND_PPO_SCHEDULE_H
#define CUDARL_RL_POND_PPO_SCHEDULE_H
#include "rl/pond_ppo_task.h"
// Training request timing only: never integrates, resets, clamps or steers a body.
namespace cudarl::rl::pond_ppo_schedule {
namespace pt=pond_ppo_task;
namespace p=physics;
inline constexpr uint32_t version=1;
enum Event : uint32_t { Onset=1, Success=2, Timeout=4, Interrupted=8, EarlyTerminal=16, LongHoldEnd=32 };
struct State {
    uint32_t serial=0,onset_control=0,age=0,streak=0,events=0;
    bool active=false,succeeded=false,long_hold=false,valid_course=false;
    float last_course_x=1,last_course_z=0,angle=0;
    uint64_t onsets=0,successes=0,timeouts=0,interrupted=0,early_terminal=0,long_holds=0;
};
struct SavedTask {pt::Task task{};State schedule{};};
struct Sample {float vx=0,vz=0,depth=0,tilt=0;};
CUDARL_HD inline void reset_episode(State& s) {
    const auto on=s.onsets,su=s.successes,to=s.timeouts,in=s.interrupted,early=s.early_terminal,longs=s.long_holds;
    s=State{};s.onsets=on;s.successes=su;s.timeouts=to;s.interrupted=in;s.early_terminal=early;s.long_holds=longs;
}
CUDARL_HD inline float angle_draw(uint32_t env,uint32_t episode,uint32_t serial,uint32_t seed) {
    const auto draw=rng::bits(seed,rng::hash_u32(env^0x976a2d31u^rng::hash_u32(episode)),serial)%10u;
    const auto magnitude=draw%5u;
    const float degrees=magnitude==0?5.f:(magnitude==1?15.f:(magnitude==2?30.f:(magnitude==3?60.f:90.f)));
    return (draw<5?-degrees:degrees)*pt::obs::pi/180.f;
}
CUDARL_HD inline void course(State& s,const Sample& m) {
    const float speed=sqrtf(m.vx*m.vx+m.vz*m.vz);
    if(p::finite_number(speed)&&speed>=.003f){s.last_course_x=m.vx/speed;s.last_course_z=m.vz/speed;s.valid_course=true;}
}
// Invoked before a control only. Thus a final episode tick cannot issue an
// unexercised replacement, and a rollout/checkpoint edge has no timing effect.
CUDARL_HD inline void before(State& s,pt::Task& task,uint32_t env,uint32_t seed) {
    s.events=0;
    if(task.kind!=pt::Turn||s.active||task.episode_controls<pt::turn_control||task.episode_controls>=pt::episode_limit)return;
    const bool first=s.serial==0;
    s.angle=first?task.turn_radians:angle_draw(env,task.episode_index,s.serial,seed);
    p::V heading=task.request.request.heading;
    if(!first&&s.valid_course)heading={s.last_course_x,0,s.last_course_z};
    const auto rotation=p::axis_rotation({0,1,0},s.angle);
    const p::V target=rotation.rotate(heading);
    auto& request=task.request.request;
    // First request is exactly the original v3 request transformation.
    if(first){request.heading=rotation.rotate(request.heading);request.terminal_velocity=rotation.rotate(request.terminal_velocity);
        request.position=task.request.start_com_model+rotation.rotate(request.position-task.request.start_com_model);}
    else {request.heading=target;request.terminal_velocity=target*pt::target_speed_m_s;}
    task.request_changed=true;task.turn_radians=s.angle;
    ++s.serial;++s.onsets;s.onset_control=task.episode_controls+1;s.age=0;s.streak=0;s.active=true;s.succeeded=false;
    s.long_hold=rng::bits(seed,rng::hash_u32(env^0x35c7b219u^rng::hash_u32(task.episode_index)),s.serial)%10u==0;
    if(s.long_hold)++s.long_holds;s.events|=Onset;
}
CUDARL_HD inline void after(State& s,const pt::Task& task,const Sample& m,bool terminal,bool truncated) {
    course(s,m);
    if(task.kind!=pt::Turn||!s.active)return;
    ++s.age;
    const auto h=task.request.request.heading;
    const float speed=sqrtf(m.vx*m.vx+m.vz*m.vz),projected=m.vx*h.x+m.vz*h.z;
    const bool qualified=p::finite_number(speed)&&speed>0&&projected/speed>=.9961946980917455f&&
        projected>=.006f&&m.depth<=.001f&&m.tilt<=10.f;
    if(!s.succeeded){
        s.streak=s.age>100&&qualified?s.streak+1:0;
        if(s.streak>=20){s.succeeded=true;++s.successes;s.events|=Success;if(!s.long_hold)s.active=false;}
    }
    if(s.age>=400&&s.active){if(!s.succeeded){++s.timeouts;s.events|=Timeout;}else s.events|=LongHoldEnd;s.active=false;}
    if(terminal&&s.active){
        if(!s.succeeded){if(truncated){++s.interrupted;s.events|=Interrupted;}else{++s.early_terminal;s.events|=EarlyTerminal;}}
        s.active=false;
    }
}
}
#endif
