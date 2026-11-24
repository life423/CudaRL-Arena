#pragma once
#include "rl/pond_ppo_schedule.h"

// A residual study changes the requested angle only, at actual request birth.
// It shares all success, hold, timeout, truncation and history rules with v6.
namespace cudarl::rl::residual_turn_schedule {
namespace original=pond_ppo_schedule;
namespace pt=pond_ppo_task;
namespace p=physics;
using namespace pond_ppo_schedule;

CUDARL_HD inline float requested_angle(uint32_t env,uint64_t issued){
    const float degrees=issued==0?15.f:(issued==1?30.f:(issued==2?60.f:90.f));
    return ((env^issued)&1u?degrees:-degrees)*pt::obs::pi/180.f;
}

CUDARL_HD inline void before(original::State& s,pt::Task& task,uint32_t env,uint32_t seed,uint32_t program=pt::repeated_headings_v1){
    const bool new_request=task.kind==pt::Turn&&!s.active&&task.episode_controls>=pt::turn_control&&task.episode_controls<pt::episode_limit;
    if(!new_request){original::before(s,task,env,seed,program);return;}

    // onsets counts real requests across prior episodes and checkpoint chunks.
    const float angle=requested_angle(env,s.onsets);
    const bool first_in_episode=s.serial==0;
    const auto base=s.valid_course?p::V{s.last_course_x,0,s.last_course_z}:task.request.request.heading;
    if(first_in_episode)task.turn_radians=angle;
    original::before(s,task,env,seed,program);
    if(!first_in_episode){
        // The replacement starts from the same pre-issue course, never from
        // the heading already rotated by the shared scheduler.
        const auto target=p::axis_rotation({0,1,0},angle).rotate(base);
        s.angle=angle;task.turn_radians=angle;
        task.request.request.heading=target;
        task.request.request.terminal_velocity=target*pt::target_speed_m_s;
    }
}
} // namespace cudarl::rl::residual_turn_schedule
