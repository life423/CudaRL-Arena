#ifndef CUDARL_ARCH_H
#define CUDARL_ARCH_H
#include "feeding.h"

namespace cudarl::arch {
using deep::V;
constexpr int kObs=55,kAct=4,kEpisodeSteps=1200;
constexpr float sensing_range=3.f;
struct Context {V current{};int version=0;deep::Basin basin{};};
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
CUDARL_HD inline bool water(V p,const deep::Habitat& h,float clearance=.3f,deep::Basin basin={}) {
    return (basin.layout!=0 || radial(p,h)<.86f*.86f) && p.y<-.35f &&
        p.y>deep::habitat_floor(p.x,p.z,h,basin)+clearance && deep::arch_distance(p,h)>clearance;
}
// Passive, neutrally buoyant food/tracers use the same imposed local velocity
// as articulated drag. Only particles receive geometric solid contact; the
// swimmer remains fully force-driven. Coordinates are model units.
CUDARL_HD inline V particle_contact(V p,const deep::Habitat& h,deep::Basin basin={}) {
    constexpr float r=feeding::radius/deep::scale;
    for(int iteration=0;iteration<4;++iteration) {
        V normal;const float rock=deep::arch_distance(p,h,&normal);
        if(rock<r) p+=normal*(r-rock+1e-5f);
        const V gradient=deep::habitat_gradient(p.x,p.z,h,basin);
        const float length=sqrtf(1+gradient.x*gradient.x+gradient.z*gradient.z);
        const V bed_normal=V{-gradient.x,1,-gradient.z}/length;
        const float gap=(p.y-deep::habitat_floor(p.x,p.z,h,basin))/length-r;
        if(gap<0) p+=bed_normal*(-gap+1e-5f);
        if(p.y>-r) p.y=-r;
        if(basin.layout==0) {
            const float edge=sqrtf(radial(p,h));
            if(edge>.94f) {p.x*=.94f/edge;p.z*=.94f/edge;}
        }
    }
    return p;
}
CUDARL_HD inline V advect(V position,V current,const deep::Habitat& h,float seconds,deep::Basin basin={}) {
    if(!(seconds>0) || (current.x==0 && current.y==0 && current.z==0)) return position;
    // The normal caller advances one 50 ms physical step. Small midpoint
    // substeps also make visual tracers agree under ordinary render intervals.
    const int ticks=static_cast<int>(ceilf(seconds/.005f));const float dt=seconds/ticks;
    for(int tick=0;tick<ticks;++tick) {
        const V first=deep::habitat_flow(position,h,current,basin);
        const V midpoint=particle_contact(position+first*(.5f*dt/deep::scale),h,basin);
        const V velocity=deep::habitat_flow(midpoint,h,current,basin);
        position=particle_contact(position+velocity*(dt/deep::scale),h,basin);
    }
    return position;
}
CUDARL_HD inline V practice_current(const State& s,const Context& context) {
    const float maximum=deep::norm(context.current);
    if(s.pond.retention || s.pond.environment%4u==1u || !(maximum>0)) return {};
    const V direction=context.current/maximum;const auto& pond=s.pond;
    if(pond.environment%4u==2u) {
        const uint32_t key=feeding::practice_key(pond,0,0xA2C457EDu);
        return direction*(maximum*rng::uniform(pond.seed,pond.environment,key));
    }
    const uint32_t period=80u+feeding::practice_key(pond,0,0xA2C4C10Cu)%121u;
    const uint32_t tick=static_cast<uint32_t>(pond.steps),frame=tick/period;
    const float u=static_cast<float>(tick%period)/static_cast<float>(period),blend=u*u*(3.f-2.f*u);
    const uint32_t a=feeding::practice_key(pond,frame,0xA2C4C011u),b=feeding::practice_key(pond,frame+1u,0xA2C4C011u);
    const float low=a%4u<2u?0.f:rng::uniform(pond.seed,pond.environment,a);
    const float high=b%4u<2u?0.f:rng::uniform(pond.seed,pond.environment,b);
    return direction*(maximum*(low+(high-low)*blend));
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
CUDARL_HD inline bool cove_food_clear(const State& s,int slot,V position,deep::Basin basin) {
    if(!water(position,s.habitat,.4f,basin) || deep::norm(position-s.pond.body.position)<.8f) return false;
    const auto body=deep::shape(s.pond.body);
    for(const auto& link:body.link) {
        const V axis=link.q.rotate({1,0,0}),relative=position-link.center/deep::scale;
        const float axial=deep::clamp(deep::dot(relative,axis),-.5f*link.length/deep::scale,.5f*link.length/deep::scale);
        if(deep::norm(relative-axis*axial)<.4f) return false;
    }
    for(int i=0;i<feeding::capacity;++i) if(i!=slot && s.pond.pellets[i].active &&
        deep::norm(position-s.pond.pellets[i].position)<1.2f) return false;
    return true;
}
CUDARL_HD inline void place(State& s,int slot,int side,deep::Basin basin={}) {
    if(basin.layout!=0) {
        const uint32_t key=s.pond.next_id*193u+s.pond.episode*7919u;
        // Successive replacements exercise both original sides and the new
        // cove. A blocked placement can use another reachable region; food is
        // never placed inside the animal to enforce a particular visit.
        for(uint32_t trial=0;trial<192;++trial) {
            const int region=static_cast<int>((s.pond.next_id-1u+trial/64u)%3u);
            const float x=rng::uniform(s.pond.seed,s.pond.environment,key+trial*3u)*2-1;
            const float y=rng::uniform(s.pond.seed,s.pond.environment,key+trial*3u+1u)*2-1;
            const float z=rng::uniform(s.pond.seed,s.pond.environment,key+trial*3u+2u)*2-1;
            V position;
            if(region==2) {
                const V center=deep::cove_center(s.habitat),radii=deep::cove_radii(s.habitat);
                position={center.x+radii.x*(.45f+.15f*x),-1.65f+.35f*y,center.z+radii.z*.23f*z};
            } else position=deep::arch_to_world({x*.8f,y*.28f,(region==0?-1.f:1.f)*(2.65f+.35f*z)},s.habitat);
            if(!cove_food_clear(s,slot,position,basin)) continue;
            auto& p=s.pond.pellets[slot];p={};p.active=true;p.settling=false;p.position=position;p.id=s.pond.next_id++;
            p.distance=deep::norm(position-s.pond.body.position);s.pond.decision=true;return;
        }
        return;
    }
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
CUDARL_HD inline void replenish(State& s,deep::Basin basin={}) {
    if(s.pond.retention) {feeding::replenish(s.pond,{});return;}
    for(int slot=0;slot<2;++slot) if(!s.pond.pellets[slot].active) {
        const float previous=deep::arch_to_local(s.pond.pellets[slot].position,s.habitat).z;
        place(s,slot,previous>=0?-1:1,basin);
    }
}
CUDARL_HD inline State initial(uint32_t seed,uint32_t environment,uint32_t episode,bool retention=false,deep::Basin basin={}) {
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
    place(s,0,-1,basin);place(s,1,1,basin);return s;
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
CUDARL_HD inline float advance(State& s,const float* action,V current={},deep::Basin basin={}) {
    if(s.pond.retention) return feeding::advance(s.pond,action,{});
    auto& p=s.pond;const bool targeted=p.target>=0 && p.pellets[p.target].active;
    for(auto& food:p.pellets) if(food.active) {
        const V previous=food.position;food.position=advect(previous,current,s.habitat,deep::dt,basin);
        food.velocity=(food.position-previous)*(deep::scale/deep::dt);
    }
    const float before=targeted?deep::norm(p.pellets[p.target].position-p.body.position):0;
    deep::Physics physics;physics.pond=true;physics.habitat=s.habitat;physics.water_velocity=current;physics.basin=basin;
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
