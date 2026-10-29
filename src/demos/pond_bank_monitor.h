#ifndef CUDARL_POND_BANK_MONITOR_H
#define CUDARL_POND_BANK_MONITOR_H
#include "pond_playback.h"
#include <vector>
namespace cudarl::demos {
// Same sampled near-bank proxy as tools/pond_bank_study.cu; observation only.
inline bool pond_near_bank(const deep_pond::State& state) {
    if(state.body.position.y>=-.2f) return false;
    const auto geometry=deep::shape(state.body);
    for(int b=0;b<3;++b) for(int sample=0;sample<3;++sample) {
        const auto& link=geometry.link[b];
        const auto point=link.center+link.q.rotate({(sample-1)*link.length/3.f,0,0});
        const float x=point.x/deep::scale,z=point.z/deep::scale,r2=x*x/(4.6f*4.6f)+z*z/(3.5f*3.5f);
        const float gx=r2<1?27.f*x*r2*r2/(4.6f*4.6f):0,gz=r2<1?27.f*z*r2*r2/(3.5f*3.5f):0;
        const float gap=(point.y-deep::pond_floor(x,z)*deep::scale)/std::sqrt(1+gx*gx+gz*gz)-(b==0?.0064f:.002f);
        if(r2>.35f && gap<.001f) return true;
    }
    return false;
}
struct PondBankVisit {
    int start=0,clear=-1,last=0,contact_steps=0,slow_steps=0,first_food=-1,entry_foods=0;
};
class PondBankMonitor {
    int active_=-1,clear_steps_=0,last_tick_=-1,last_foods_=0;
public:
    int contact_steps=0;
    std::vector<PondBankVisit> visits;
    bool observe(const deep_pond::State& state) {
        const bool near=pond_near_bank(state);
        const int elapsed=last_tick_<0?0:state.steps-last_tick_;
        if(last_tick_<0) last_foods_=state.eaten;
        if(near && active_<0) {
            visits.push_back({state.steps,-1,state.steps,0,0,-1,last_foods_});
            active_=static_cast<int>(visits.size())-1;clear_steps_=0;
        }
        if(active_>=0) {
            auto& visit=visits[static_cast<size_t>(active_)];visit.last=state.steps;
            if(near) {
                const int delta=elapsed>0?elapsed:0;
                visit.contact_steps+=delta;contact_steps+=delta;clear_steps_=0;
                if(deep::norm({state.body.velocity[0],state.body.velocity[1],state.body.velocity[2]})<.005f) visit.slow_steps+=delta;
            } else {
                clear_steps_+=elapsed>0?elapsed:0;
                if(clear_steps_>=20) {visit.clear=state.steps;active_=-1;clear_steps_=0;}
            }
        }
        if(state.eaten>last_foods_) for(auto& visit:visits)
            if(visit.first_food<0 && state.eaten>visit.entry_foods) visit.first_food=state.steps;
        last_tick_=state.steps;last_foods_=state.eaten;return near;
    }
};
}
#endif
