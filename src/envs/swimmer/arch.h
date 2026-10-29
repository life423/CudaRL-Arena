#ifndef CUDARL_ARCH_H
#define CUDARL_ARCH_H
#include "feeding.h"

namespace cudarl::arch {
using deep::V;
constexpr int kObs=55,kAct=4,kEpisodeSteps=1200;
constexpr float sensing_range=3.f;
struct State {
    feeding::State pond;
    deep::Habitat habitat;
    int contacts=0,passages=0,clean_passages=0,pending_side=0,last_full_side=0;
    bool contacted_in_passage=false,crossed_opening=false,passage_valid=true;
};
CUDARL_HD inline deep::Habitat geometry() {
    deep::Habitat h;h.scale=1.5f;h.arch=true;return h;
}
CUDARL_HD inline float radial(V p,const deep::Habitat& h) {
    const float rx=4.6f*h.scale,rz=3.5f*h.scale;
    return p.x*p.x/(rx*rx)+p.z*p.z/(rz*rz);
}
CUDARL_HD inline bool water(V p,const deep::Habitat& h,float clearance=.3f) {
    return radial(p,h)<.86f*.86f && p.y<-.35f &&
        p.y>deep::habitat_floor(p.x,p.z,h)+clearance && deep::arch_distance(p,h)>clearance;
}
// All 18 rays are expressed in the physical head frame, not the animated mesh.
CUDARL_HD inline V ray_direction(int ray) {
    if(ray<12) {const float angle=ray*6.28318530718f/12.f;return {cosf(angle),0,sinf(angle)};}
    constexpr float d=.70710678118f;
    if(ray==12) return {d,d,0};if(ray==13) return {d,-d,0};
    if(ray==14) return {-d,d,0};if(ray==15) return {-d,-d,0};
    return ray==16?V{0,1,0}:V{0,-1,0};
}
CUDARL_HD inline float proximity(V origin,V direction,const deep::Habitat& h) {
    if(!h.arch) return 0;
    float travel=0;
    for(int step=0;step<32 && travel<sensing_range;++step) {
        const float gap=deep::arch_distance(origin+direction*travel,h);
        if(gap<.025f) return 1.f-travel/sensing_range;
        travel+=fmaxf(.01f,gap*.9f);
    }
    return 0;
}
CUDARL_HD inline void observe(const State& s,float* out) {
    feeding::motor_observe(s.pond,out);
    for(int ray=0;ray<18;++ray)
        out[feeding::kObs+ray]=proximity(s.pond.body.position,s.pond.body.orientation.rotate(ray_direction(ray)),s.habitat);
}
CUDARL_HD inline void place(State& s,int slot,int side) {
    auto& p=s.pond.pellets[slot];const uint32_t key=s.pond.next_id*193u+s.pond.episode*7919u;
    const auto body=deep::shape(s.pond.body);V position;
    for(uint32_t trial=0;trial<64;++trial) {
        const float x=(rng::uniform(s.pond.seed,s.pond.environment,key+trial*3u)*2-1)*.8f;
        const float y=(rng::uniform(s.pond.seed,s.pond.environment,key+trial*3u+1u)*2-1)*.28f;
        const float z=side*(2.4f+.75f*rng::uniform(s.pond.seed,s.pond.environment,key+trial*3u+2u));
        position=deep::arch_to_world({x,y,z},s.habitat);
        bool clear=deep::norm(position-s.pond.body.position)>=.8f && water(position,s.habitat);
        for(const auto& link:body.link) {
            const V axis=link.q.rotate({1,0,0}),relative=position-link.center/deep::scale;
            const float axial=deep::clamp(deep::dot(relative,axis),-.5f*link.length/deep::scale,.5f*link.length/deep::scale);
            if(deep::norm(relative-axis*axial)<.4f) clear=false;
        }
        if(clear) break;
        if(trial==63) {
            const float body_x=deep::arch_to_local(s.pond.body.position,s.habitat).x;
            position=deep::arch_to_world({body_x>=0?-1.3f:1.3f,0,side*2.8f},s.habitat);
        }
    }
    // The chosen envelope fits the wider basin and stays well away from rock.
    p={};p.active=true;p.settling=false;p.position=position;p.id=s.pond.next_id++;
    p.distance=deep::norm(position-s.pond.body.position);s.pond.decision=true;
}
CUDARL_HD inline void replenish(State& s) {
    if(s.pond.retention) {feeding::replenish(s.pond,{});return;}
    for(int slot=0;slot<2;++slot) if(!s.pond.pellets[slot].active) {
        const float previous=deep::arch_to_local(s.pond.pellets[slot].position,s.habitat).z;
        place(s,slot,previous>=0?-1:1);
    }
}
CUDARL_HD inline State initial(uint32_t seed,uint32_t environment,uint32_t episode,bool retention=false) {
    State s;s.pond=feeding::initial(seed,environment,episode,{},true);s.pond.retention=retention;
    if(retention) return s;
    s.habitat=geometry();
    // Visible world zero has fixed geometry. Practice worlds vary its pose;
    // the same analytic geometry drives both perception and physical contact.
    if(environment>0) {
        const uint32_t key=episode*43u;
        s.habitat.center.x=(rng::uniform(seed,environment,key+5u)*2-1)*.4f;
        s.habitat.center.z=(rng::uniform(seed,environment,key+6u)*2-1)*.2f;
        s.habitat.yaw=(rng::uniform(seed,environment,key+7u)*2-1)*.35f;
    }
    for(auto& pellet:s.pond.pellets) pellet={};
    const float side=rng::bits(seed,environment,episode)%2u?1.f:-1.f;
    const float x=(rng::uniform(seed,environment,episode*31u+3u)*2-1)*.65f;
    s.pond.body.position=deep::arch_to_world({x,0,side*2.35f},s.habitat);
    s.pond.body.orientation=deep::axis_rotation({0,1,0},rng::uniform(seed,environment,episode*31u+4u)*6.2831853f);
    s.pond.anchor=s.pond.body.position;s.pond.target=-1;s.pond.desired_foods=2;
    place(s,0,-1);place(s,1,1);return s;
}
// A read-only event detector: it never changes the physical pose. A passage
// requires the entire articulated body to change sides through the aperture.
CUDARL_HD inline void record_passage(State& s,bool contact) {
    const auto shape=deep::shape(s.pond.body);float minimum=1e6f,maximum=-1e6f;
    bool intersects=false,inside=true;
    for(int b=0;b<3;++b) {
        const auto& link=shape.link[b];const V axis=link.q.rotate({1,0,0});
        const V a=deep::arch_to_local((link.center-axis*(link.length*.5f))/deep::scale,s.habitat);
        const V z=deep::arch_to_local((link.center+axis*(link.length*.5f))/deep::scale,s.habitat);
        const float envelope=b==0?.2f:(b==2?.15f:.04f);
        minimum=fminf(minimum,fminf(a.z,z.z)-envelope);maximum=fmaxf(maximum,fmaxf(a.z,z.z)+envelope);
        if(a.z*z.z<=0 && fabsf(z.z-a.z)>1e-7f) {
            const V cross=a+(z-a)*(-a.z/(z.z-a.z));intersects=true;
            if(sqrtf(cross.x*cross.x+cross.y*cross.y)+envelope>s.habitat.major-s.habitat.minor) inside=false;
        }
    }
    const float margin=s.habitat.minor+.03f;
    const int full=minimum>margin?1:(maximum<-margin?-1:0);
    if(s.pending_side==0 && s.last_full_side!=0 && full==0) {
        s.pending_side=s.last_full_side;s.contacted_in_passage=false;s.crossed_opening=false;s.passage_valid=true;
    }
    if(s.pending_side!=0) {
        s.contacted_in_passage=s.contacted_in_passage || contact;
        if(intersects) {s.crossed_opening=true;s.passage_valid=s.passage_valid && inside;}
        if(full!=0) {
            if(full==-s.pending_side && s.crossed_opening && s.passage_valid) {
                ++s.passages;if(!s.contacted_in_passage) ++s.clean_passages;
            }
            s.pending_side=0;
        }
    }
    if(full!=0) s.last_full_side=full;
}
CUDARL_HD inline float advance(State& s,const float* action) {
    if(s.pond.retention) return feeding::advance(s.pond,action,{});
    auto& p=s.pond;const bool targeted=p.target>=0 && p.pellets[p.target].active;
    const float before=targeted?deep::norm(p.pellets[p.target].position-p.body.position):0;
    deep::Physics physics;physics.pond=true;physics.habitat=s.habitat;
    bool contact=false;deep::advance_physics(p.body,action,physics,&contact);++p.steps;++p.since_food;
    float reward=-.005f;
    if(!(deep::norm(p.body.position)<100.f)) p.invalid=true;
    if(deep::norm(p.body.position-p.anchor)>.1f || !targeted) {p.anchor=p.body.position;p.still_steps=0;}
    else {++p.still_steps;p.max_still=p.still_steps>p.max_still?p.still_steps:p.max_still;}
    if(targeted) reward+=3.f*(before-deep::norm(p.pellets[p.target].position-p.body.position));
    const float gap=deep::arch_body_clearance(p.body,s.habitat);
    if(contact) {++s.contacts;reward-=.005f+.015f*deep::clamp(-gap/.1f,0,1);}
    record_passage(s,contact);
    for(int j=0;j<4;++j) {const float a=deep::clamp(action[j],-1,1),d=a-p.last_action[j];reward-=.003f*d*d;p.last_action[j]=a;}
    for(auto& pellet:p.pellets) if(pellet.active && deep::norm(pellet.position-p.body.position)<.24f) {
        pellet.active=false;++p.eaten;p.since_food=0;reward+=5;p.decision=true;
    }
    if(p.steps%20==0) {
        for(auto& pellet:p.pellets) if(pellet.active) {
            const float d=deep::norm(pellet.position-p.body.position);pellet.progress=deep::clamp(pellet.distance-d,-2,2);pellet.distance=d;
        }
        p.decision=true;
    }
    return reward;
}
}
#endif
