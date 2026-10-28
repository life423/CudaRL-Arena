#ifndef CUDARL_FIXED_TARGET_DIAGNOSTIC_H
#define CUDARL_FIXED_TARGET_DIAGNOSTIC_H
#include "demos/pond_target.h"
// Experiment-only wrapper: delegates all movement and collection to the shared
// pond implementation. Nothing here is used by the deployed controller.
namespace cudarl::fixed_target {
namespace d=cudarl::deep;
namespace p=cudarl::deep_pond;
struct Trial {
    p::State state;
    d::V goal,current;
    int target=0,passive=0;
};
struct Metrics {
    int ticks=0,arrival=0,incidental=0,invalid=0;
    int low_run=0,max_low=0,no_progress=0,max_no_progress=0;
    float initial_distance=0,min_distance=0,progress_anchor=0,remaining=0;
    double ground=0,relative=0,toward=0;
};
CUDARL_HD inline d::V flow(d::V body,d::V goal,int condition) {
    if(condition==0) return {};
    const d::V delta{goal.x-body.x,0,goal.z-body.z};const float length=d::norm(delta);
    const d::V direction=length>1e-8f?delta/length:d::V{1,0,0};
    const float speed=condition<=4?.005f:.01f;
    const int kind=(condition-1)%4;
    if(kind==0) return direction*(-speed);
    if(kind==1) return direction*speed;
    const d::V left{-direction.z,0,direction.x};return left*(kind==2?speed:-speed);
}
CUDARL_HD inline Trial initial(uint32_t seed,int environment,int folded,int condition,int passive) {
    Trial trial;trial.state=p::initial(seed,static_cast<uint32_t>(environment));
    trial.target=environment%3;trial.passive=passive;
    if(folded) for(int j=0;j<4;++j) trial.state.body.angle[j]=(environment&(1<<j))?.98f:-.98f;
    trial.goal=trial.state.food[trial.target];trial.current=flow(trial.state.body.position,trial.goal,condition);return trial;
}
CUDARL_HD inline Metrics metrics(const Trial& trial) {
    Metrics out;out.initial_distance=out.min_distance=out.progress_anchor=out.remaining=d::norm(trial.goal-trial.state.body.position);return out;
}
CUDARL_HD inline void observe(const Trial& trial,float* out) {
    p::observe(trial.state,out);demos::primary_pond_food(out,trial.target);
}
CUDARL_HD inline void step(Trial& trial,Metrics& m,const float* action) {
    if(m.arrival || m.invalid) return;
    const float zero[4]{};const auto result=p::advance(trial.state,trial.passive?zero:action,trial.current);++m.ticks;
    const auto& b=trial.state.body;const d::V velocity{b.velocity[0],b.velocity[1],b.velocity[2]};
    m.remaining=d::norm(trial.goal-b.position);m.min_distance=fminf(m.min_distance,m.remaining);
    m.ground+=d::norm(velocity);m.relative+=d::norm(velocity-trial.current);
    const auto offset=trial.goal-b.position;const float distance=d::norm(offset);
    if(distance>1e-8f) m.toward+=d::dot(velocity,offset/distance);
    m.low_run=d::norm(velocity)<.005f?m.low_run+1:0;if(m.low_run>m.max_low) m.max_low=m.low_run;
    ++m.no_progress;
    if(m.remaining<m.progress_anchor-.05f) {m.progress_anchor=m.remaining;m.no_progress=0;}
    if(m.no_progress>m.max_no_progress) m.max_no_progress=m.no_progress;
    if(result.eaten_slot==trial.target) m.arrival=m.ticks;
    else if(result.eaten_slot>=0) ++m.incidental;
    bool finite=fabsf(b.position.x)<1e10f && fabsf(b.position.y)<1e10f && fabsf(b.position.z)<1e10f &&
        fabsf(b.orientation.x)<1e10f && fabsf(b.orientation.y)<1e10f && fabsf(b.orientation.z)<1e10f && fabsf(b.orientation.w)<1e10f;
    for(int j=0;j<d::dofs;++j) finite=finite && fabsf(b.velocity[j])<1e10f;
    for(int j=0;j<4;++j) finite=finite && fabsf(b.angle[j])<1e10f;
    if(!finite) ++m.invalid;
}
}
#endif
