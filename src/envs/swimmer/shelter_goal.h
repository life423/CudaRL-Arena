#ifndef CUDARL_SHELTER_GOAL_H
#define CUDARL_SHELTER_GOAL_H
#include "arch.h"

namespace cudarl::shelter {
// A navigation request is metadata. It never becomes an edible physical pellet.
struct Goal {
    deep::V position{},velocity{};
    int food=-1,cover=-1;
    bool valid=false;
};
CUDARL_HD inline void motor_observe(const arch::State& state,const Goal& goal,float* out,deep::Basin basin={2}) {
    arch::observe(state,out,basin);
    if(!goal.valid || goal.cover<0) return;
    const auto relative=state.pond.body.orientation.inverse(goal.position-state.pond.body.position)/2.f;
    const deep::V velocity{state.pond.body.velocity[0],state.pond.body.velocity[1],state.pond.body.velocity[2]};
    const auto drift=state.pond.body.orientation.inverse(goal.velocity-velocity)/.04f;
    out[0]=relative.x;out[1]=relative.y;out[2]=relative.z;
    out[33]=drift.x;out[34]=drift.y;out[35]=drift.z;out[36]=1.f;
}
CUDARL_HD inline void motor_observe59(const arch::State& state,const Goal& goal,deep::V heading,float* out,deep::Basin basin={2}) {
    motor_observe(state,goal,out,basin);
    for(int i=55;i<59;++i) out[i]=0;
    if(goal.valid && goal.cover>=0) {
        const auto facing=state.pond.body.orientation.inverse(heading);
        out[55]=1;out[56]=facing.x;out[57]=facing.y;out[58]=facing.z;
    }
}
}
#endif
