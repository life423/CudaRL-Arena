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
// x/z are mass-weighted COM positions in meters at control boundaries.
// Ten stored endpoints make a half-second displacement at the next endpoint.
struct CourseHistory {
    static constexpr uint32_t intervals = 10;
    float x[intervals]{}, z[intervals]{};
    uint32_t next = 0, count = 0;
    CUDARL_HD bool append(float x_m, float z_m, p::V& velocity) {
        const bool ready = count == intervals;
        if (ready) {
            constexpr float duration = intervals * pt::control_seconds;
            velocity = {(x_m - x[next]) / duration, 0, (z_m - z[next]) / duration};
        } else {
            ++count;
        }
        x[next] = x_m;
        z[next] = z_m;
        next = (next + 1) % intervals;
        return ready;
    }
};
struct HalfSecondTask {pt::Task task{};State schedule{};CourseHistory history{};};
struct HorizontalCourse {float x=0,z=0;};
struct SignedCourseTask {pt::Task task{};State schedule{};CourseHistory history{};HorizontalCourse movement{};};
// Signed yaw in half-turns. Positive yaw turns +X towards -Z in the pond.
// No estimate during settling, other tasks or zero displacement.
CUDARL_HD inline float signed_course_input(const pt::Task& task, HorizontalCourse movement) {
    if (task.kind != pt::Turn || !task.request_changed || (movement.x == 0 && movement.z == 0)) return 0;
    const auto heading = task.request.request.heading;
    return atan2f(movement.z * heading.x - movement.x * heading.z,
                  movement.x * heading.x + movement.z * heading.z) / pt::obs::pi;
}
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
// Balance actual Turn pairs rather than all physical worlds, which also
// contain rest and straight tasks. This runs only when a request is born.
CUDARL_HD inline float medium_request_angle(uint32_t env,uint32_t episode,uint32_t serial,uint32_t seed){
    uint32_t turn_pair_rank=0;
    for(uint32_t pair=0;pair<env/2;++pair)
        if(pt::sample_request(2*pair,episode,seed,pt::signed_half_second_v1).kind==pt::Turn)++turn_pair_rank;
    const uint32_t rotation=cudarl::rng::bits(seed,0x29b1cc7du,episode);
    const float degrees=15.f+5.f*float((turn_pair_rank+rotation+serial)%4u);
    return ((env^serial)&1u?degrees:-degrees)*pt::obs::pi/180.f;
}
// Invoked before a control only. Thus a final episode tick cannot issue an
// unexercised replacement, and a rollout/checkpoint edge has no timing effect.
CUDARL_HD inline void before(State& s,pt::Task& task,uint32_t env,uint32_t seed,uint32_t program=pt::repeated_headings_v1,bool medium_turn=false) {
    s.events=0;
    if(task.kind!=pt::Turn||s.active||task.episode_controls<pt::turn_control||task.episode_controls>=pt::episode_limit)return;
    const bool first=s.serial==0;
    if(medium_turn)s.angle=medium_request_angle(env,task.episode_index,s.serial,seed);
    else s.angle=first?task.turn_radians:(pt::half_second_program(program)?
        pt::paired_turn_angle(env,task.episode_index,s.serial,seed):angle_draw(env,task.episode_index,s.serial,seed));
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
CUDARL_HD inline void after(State& s,const pt::Task& task,const Sample& m,bool terminal,bool truncated,
                            const p::V* measured_course_velocity=nullptr) {
    const Sample movement = measured_course_velocity ?
        Sample{measured_course_velocity->x,measured_course_velocity->z,m.depth,m.tilt} : m;
    course(s,movement);
    if(task.kind!=pt::Turn||!s.active)return;
    ++s.age;
    const auto h=task.request.request.heading;
    const float course_speed=sqrtf(movement.vx*movement.vx+movement.vz*movement.vz);
    const float course_projection=movement.vx*h.x+movement.vz*h.z;
    const bool qualified=p::finite_number(course_speed)&&course_speed>0&&course_projection/course_speed>=.9961946980917455f&&
        m.vx*h.x+m.vz*h.z>=.006f&&m.depth<=.001f&&m.tilt<=10.f;
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
