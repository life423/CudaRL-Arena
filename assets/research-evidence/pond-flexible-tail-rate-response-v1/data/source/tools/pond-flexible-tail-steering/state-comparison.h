#pragma once

#include "rl/flexible_tail_navigation.h"
#include <cstddef>
#include <cstring>

namespace navigation_state {
namespace tail = cudarl::flexible_tail;
namespace physics = cudarl::physics;
inline constexpr int comparison_schema = 1;

template<class T> inline bool bits(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}
inline bool vector(const physics::V& a, const physics::V& b) {
    return bits(a.x,b.x) && bits(a.y,b.y) && bits(a.z,b.z);
}

// Every declared state field is compared bit for bit. Padding between fields
// has no C++ value and is intentionally excluded; there is no float tolerance,
// normalization, omitted physical coordinate, or mutation of either input.
inline bool exact(const tail::NavigationWorld& a, const tail::NavigationWorld& b) {
    const auto& x=a.motion;
    const auto& y=b.motion;
    if (!vector(x.body.position,y.body.position) ||
        !bits(x.body.orientation.x,y.body.orientation.x) || !bits(x.body.orientation.y,y.body.orientation.y) ||
        !bits(x.body.orientation.z,y.body.orientation.z) || !bits(x.body.orientation.w,y.body.orientation.w)) return false;
    for(int i=0;i<4;++i)if(!bits(x.body.angle[i],y.body.angle[i]))return false;
    for(int i=0;i<physics::dofs;++i)if(!bits(x.body.velocity[i],y.body.velocity[i]))return false;
    if(!bits(x.body.work,y.body.work)||!bits(x.body.travel,y.body.travel))return false;
    if(!bits(x.request.choice,y.request.choice)||!vector(x.request.origin_m,y.request.origin_m)||
        !vector(x.request.heading,y.request.heading)||!vector(x.request.terminal_velocity_m_s,y.request.terminal_velocity_m_s)||
        !bits(x.request.initialized,y.request.initialized))return false;
    for(int i=0;i<tail::turn_history_controls;++i)if(!vector(x.history[i],y.history[i]))return false;
    if(!bits(x.history_next,y.history_next)||!bits(x.controls,y.controls))return false;
    for(int i=0;i<tail::motor_outputs;++i)if(!bits(x.last_targets[i],y.last_targets[i]))return false;
    if(!bits(x.turn_radians,y.turn_radians)||!bits(x.turn_controls,y.turn_controls)||
        !bits(x.turn_issued,y.turn_issued)||!vector(x.entry_course,y.entry_course))return false;
    for(int i=0;i<tail::navigation_max_legs;++i){
        const auto& p=a.legs[i];const auto& q=b.legs[i];
        if(!vector(p.goal_m,q.goal_m)||!vector(p.heading,q.heading)||
            !bits(p.end_control,q.end_control)||!bits(p.choice,q.choice))return false;
    }
    return bits(a.leg_count,b.leg_count)&&bits(a.active_leg,b.active_leg);
}

struct Check { bool pass=true, padding_differs=false; int real_mutations_rejected=0; };
inline Check self_test() {
    Check result;
    tail::NavigationWorld a{},b=a;
    auto* request=reinterpret_cast<unsigned char*>(&b.motion.request);
    const auto request_begin=offsetof(tail::MovementRequest,initialized)+sizeof(bool);
    for(size_t i=request_begin;i<sizeof(tail::MovementRequest);++i)request[i]^=0x5a;
    auto* motion=reinterpret_cast<unsigned char*>(&b.motion);
    const auto turn_begin=offsetof(tail::TurnWorld,turn_issued)+sizeof(bool);
    for(size_t i=turn_begin;i<offsetof(tail::TurnWorld,entry_course);++i)motion[i]^=0xa5;
    result.padding_differs=std::memcmp(&a,&b,sizeof a)!=0;
    result.pass=result.padding_differs&&exact(a,b);
    const auto reject=[&](const tail::NavigationWorld& changed){
        const bool rejected=!exact(a,changed);result.pass=result.pass&&rejected;
        result.real_mutations_rejected+=rejected;
    };
    b=a;b.motion.body.position.x=1e-30f;reject(b);
    b=a;b.motion.body.orientation.x=1e-30f;reject(b);
    for(int i=0;i<4;++i){b=a;b.motion.body.angle[i]=1e-30f;reject(b);}
    for(int i=0;i<physics::dofs;++i){b=a;b.motion.body.velocity[i]=1e-30f;reject(b);}
    b=a;b.motion.body.work=1e-30f;reject(b);b=a;b.motion.body.travel=1e-30f;reject(b);
    b=a;b.motion.request.choice=tail::Choice::Hold;reject(b);
    b=a;b.motion.request.origin_m.z=1e-30f;reject(b);b=a;b.motion.request.heading.z=1e-30f;reject(b);
    b=a;b.motion.request.terminal_velocity_m_s.z=1e-30f;reject(b);
    b=a;b.motion.request.initialized=!b.motion.request.initialized;reject(b);
    b=a;b.motion.history[tail::turn_history_controls-1].z=1e-30f;reject(b);
    b=a;++b.motion.history_next;reject(b);b=a;++b.motion.controls;reject(b);
    for(int i=0;i<tail::motor_outputs;++i){b=a;b.motion.last_targets[i]=1e-30f;reject(b);}
    b=a;b.motion.turn_radians=1e-30f;reject(b);b=a;++b.motion.turn_controls;reject(b);
    b=a;b.motion.turn_issued=!b.motion.turn_issued;reject(b);b=a;b.motion.entry_course.z=1e-30f;reject(b);
    b=a;b.legs[tail::navigation_max_legs-1].goal_m.z=1e-30f;reject(b);
    b=a;b.legs[tail::navigation_max_legs-1].heading.z=1e-30f;reject(b);
    b=a;++b.legs[tail::navigation_max_legs-1].end_control;reject(b);
    b=a;b.legs[tail::navigation_max_legs-1].choice=tail::Choice::Hold;reject(b);
    b=a;++b.leg_count;reject(b);b=a;++b.active_leg;reject(b);
    return result;
}
} // namespace navigation_state
