#ifndef CUDARL_FEEDING_H
#define CUDARL_FEEDING_H
#include "progress_selector.h"
namespace cudarl::feeding {
using deep::V;
constexpr int capacity=128,kObs=37,kAct=4,kEpisodeSteps=1200;
constexpr float radius=.0008f,density=1002.f;
struct Pellet {
    V position{},velocity{};float age=0,distance=0,progress=0;uint32_t id=0;bool active=false,settling=true;
};
struct Context {V current{};V drops[32]{};int count=0,version=0;};
struct State {
    deep::BodyState body;Pellet pellets[capacity]{};float last_action[4]{};
    uint32_t seed=7,environment=0,episode=0,next_id=1;
    int steps=0,eaten=0,since_food=0,target=-1,desired_foods=3,still_steps=0,max_still=0;
    V anchor{};bool decision=true,retention=false,invalid=false;
};
CUDARL_HD inline float radial(V p) {return p.x*p.x/(4.6f*4.6f)+p.z*p.z/(3.5f*3.5f);}
CUDARL_HD inline bool valid_drop(V p) {
    return radial(p)<.9f*.9f && deep::pond_floor(p.x,p.z)<-.65f;
}
CUDARL_HD inline int count(const State& s) {int n=0;for(const auto& p:s.pellets) if(p.active) ++n;return n;}
CUDARL_HD inline int drop(State& s,V p,bool settling=true) {
    if(!valid_drop(p)) return -1;
    for(int i=0;i<capacity;++i) if(!s.pellets[i].active) {
        auto& food=s.pellets[i];food={};food.position={p.x,-.2f,p.z};food.active=true;food.settling=settling;
        food.id=s.next_id++;food.distance=deep::norm(food.position-s.body.position);s.decision=true;return i;
    }
    return -1;
}
// Small slightly denser-than-water spheres; SI velocity and forces, model positions.
// Linear viscous + quadratic form drag, ten semi-implicit 5-ms substeps.
CUDARL_HD inline void move(Pellet& p,V current) {
    if(!p.active || !p.settling) return;
    constexpr float pi=3.14159265359f,volume=4.f*pi*radius*radius*radius/3.f,mass=density*volume;
    constexpr float linear=6.f*pi*.001f*radius,quadratic=.5f*1000.f*.8f*pi*radius*radius,h=deep::dt/10;
    for(int tick=0;tick<10;++tick) {
        const V relative=p.velocity-current;
        const V force=relative*(-linear-quadratic*deep::norm(relative))+V{0,-(density-1000.f)*volume*9.81f,0};
        p.velocity+=force*(h/mass);p.position+=p.velocity*(h/deep::scale);
        const float r2=radial(p.position),gx=r2<1?27.f*p.position.x*r2*r2/(4.6f*4.6f):0,gz=r2<1?27.f*p.position.z*r2*r2/(3.5f*3.5f):0;
        V normal{-gx,1,-gz};const float length=deep::norm(normal);normal=normal/length;
        const float gap=(p.position.y-deep::pond_floor(p.position.x,p.position.z))*deep::scale/length-radius;
        if(gap<0) {
            p.position+=normal*(-gap/deep::scale);
            const float inward=deep::dot(p.velocity,normal);if(inward<0) p.velocity=p.velocity-normal*inward;
            const float speed=deep::norm(p.velocity),friction=.6f*(density-1000.f)/density*9.81f*h/length;
            if(speed>0) p.velocity=p.velocity*fmaxf(0,1-friction/speed);
        }
        // No-flux outer shore; food is never replaced to hide a bad placement.
        const float r=sqrtf(radial(p.position));
        if(r>.94f) {p.position.x*=.94f/r;p.position.z*=.94f/r;V n{p.position.x/(4.6f*4.6f),0,p.position.z/(3.5f*3.5f)};n=n/deep::norm(n);p.velocity=p.velocity-n*fmaxf(0,deep::dot(p.velocity,n));}
    }
    p.age+=deep::dt;
}
CUDARL_HD inline deep_pond::State view(const State& s,int a,int b,int c) {
    deep_pond::State out;out.body=s.body;out.steps=s.steps;out.since_food=s.since_food;out.eaten=s.eaten;out.seed=s.seed;out.environment=s.environment;out.episode=s.episode;
    const int slots[3]={a,b,c};
    for(int i=0;i<3;++i) {out.food[i]=slots[i]>=0?s.pellets[slots[i]].position:s.body.position;out.placements[i]=slots[i]>=0?s.pellets[slots[i]].id:0;}
    for(int j=0;j<4;++j) out.last_action[j]=s.last_action[j];return out;
}
CUDARL_HD inline void features(const State& s,int food,float* out) {
    const auto sample=view(s,food,food,food);float observation[36];pond_selector::observe(sample,s.target==food?0:-1,observation);
    for(int j=0;j<3;++j) out[j]=observation[j];for(int j=0;j<24;++j) out[3+j]=observation[9+j];
    out[27]=s.target==food?1.f:0.f;out[28]=deep::norm(s.pellets[food].position-s.body.position)/2.f;
    float minimum=1e10f,maximum=0,nearest=1e10f;int others=0;
    for(int i=0;i<capacity;++i) if(i!=food && s.pellets[i].active) {
        const float gap=deep::norm(s.pellets[food].position-s.pellets[i].position)/2.f;
        minimum=fminf(minimum,gap);maximum=fmaxf(maximum,gap);nearest=fminf(nearest,deep::norm(s.pellets[i].position-s.body.position)/2.f);++others;
    }
    out[29]=others?minimum:0;out[30]=maximum;out[31]=others?nearest:out[28];
    out[32]=deep::clamp(s.since_food*deep::dt/20.f,0,1);out[33]=s.pellets[food].progress;
}
CUDARL_HD inline void motor_observe(const State& s,float* out) {
    int target=s.target;if(target<0 || !s.pellets[target].active) target=-1;
    int b=-1,c=-1;float db=1e10f,dc=1e10f;
    for(int i=0;i<capacity;++i) if(i!=target && s.pellets[i].active) {
        const float d=deep::norm(s.pellets[i].position-s.body.position);
        if(d<db) {c=b;dc=db;b=i;db=d;} else if(d<dc) {c=i;dc=d;}
    }
    if(b<0) b=target;if(c<0) c=b;
    deep_pond::observe(view(s,target,b,c),out);
    V relative{};if(target>=0) relative=s.body.orientation.inverse(s.pellets[target].velocity-V{s.body.velocity[0],s.body.velocity[1],s.body.velocity[2]})/.04f;
    out[33]=relative.x;out[34]=relative.y;out[35]=relative.z;out[36]=target>=0?1.f:0.f;
}
// Practice events depend on seed and simulation time, never on collection/action
// history. Matched actors therefore receive the same arrival/current schedule.
// Deriving phase from the existing state also preserves full-save binary layout.
CUDARL_HD inline uint32_t practice_key(const State& s,uint32_t event,uint32_t domain) {
    return rng::bits(s.seed^domain,s.environment,s.episode*4099u+event);
}
CUDARL_HD inline V sample_practice_drop(const State& s,const Context& context,uint32_t event) {
    const uint32_t key=practice_key(s,event,0xFEED431u);
    if(context.count>0 && (key&1u)) return context.drops[(key>>1)%static_cast<uint32_t>(context.count)];
    const float angle=6.2831853f*rng::uniform(s.seed^0xBA5E431u,s.environment,key);
    const float r=.86f*sqrtf(rng::uniform(s.seed^0xBA5E431u,s.environment,key+1u));
    return {4.6f*r*cosf(angle),0,3.5f*r*sinf(angle)};
}
CUDARL_HD inline float practice_speed(const State& s,uint32_t frame) {
    const uint32_t key=practice_key(s,frame,0xC011E17u);
    // Consecutive zero keyframes supply genuinely calm intervals as well as lulls.
    return key%4u<2u?0.f:.0075f*rng::uniform(s.seed,s.environment,key);
}
CUDARL_HD inline V practice_current(const State& s,const Context& context) {
    if(s.retention || s.environment%4u<2u) return {};
    const float magnitude=deep::norm(context.current);
    const V direction=magnitude>1e-8f?context.current/magnitude:V{1,0,0};
    if(s.environment%4u==2u) {
        const uint32_t key=practice_key(s,0,0x57EAD1u);
        const float speed=(key&1u) && magnitude>0?fminf(magnitude,.0075f):.0075f*rng::uniform(s.seed,s.environment,key);
        return direction*speed;
    }
    const uint32_t period=80u+practice_key(s,0,0xC10C431u)%121u; // 4–10 simulated seconds.
    const uint32_t tick=static_cast<uint32_t>(s.steps),frame=tick/period;
    const float u=static_cast<float>(tick%period)/static_cast<float>(period),blend=u*u*(3.f-2.f*u);
    const float a=practice_speed(s,frame),b=practice_speed(s,frame+1u);
    return direction*(a+(b-a)*blend);
}
CUDARL_HD inline State initial(uint32_t seed,uint32_t environment,uint32_t episode,const Context& context,bool retention=false) {
    State s;s.seed=seed;s.environment=environment;s.episode=episode;s.retention=retention;
    const auto old=deep_pond::initial(seed,environment,episode);s.body=old.body;s.anchor=s.body.position;
    if(retention) {for(int i=0;i<3;++i) {auto& p=s.pellets[i];p.active=true;p.settling=false;p.id=s.next_id++;p.position=old.food[i];p.distance=deep::norm(p.position-s.body.position);}}
    else {
        const int number=1+static_cast<int>(rng::bits(seed,environment,episode)%8u);
        s.desired_foods=number;
        for(int i=0;i<number;++i) {
            const uint32_t key=rng::bits(seed^0xFEED123u,environment,episode*31u+static_cast<uint32_t>(i));
            const V position=sample_practice_drop(s,context,0x80000000u+static_cast<uint32_t>(i));
            const int slot=drop(s,position);
            if(slot>=0 && i%2) {auto& p=s.pellets[slot];const float bed=deep::pond_floor(p.position.x,p.position.z)+radius/deep::scale+.03f;p.position.y=bed+(-.2f-bed)*rng::uniform(seed,environment,key+2);p.velocity.y=-.0018f;}
        }
    }
    return s;
}
CUDARL_HD inline void replenish(State& s,const Context& context) {
    if(s.retention) {
        auto old=view(s,0,1,2);
        for(int j=0;j<3;++j) if(!s.pellets[j].active && deep_pond::place_food(old,j)) {
            auto& p=s.pellets[j];p={};p.active=true;p.settling=false;p.id=s.next_id++;p.position=old.food[j];p.distance=deep::norm(p.position-s.body.position);s.decision=true;
        }
    } else {
        // One seeded arrival per six-second window, offset within the window;
        // arrivals are 2–10 seconds apart, with occasional pairs. Eating does not
        // trigger replacement. A practice-only occupancy bound prevents buildup.
        const uint32_t tick=static_cast<uint32_t>(s.steps),window=tick/120u;
        const uint32_t key=practice_key(s,window,0xA221431u),offset=20u+key%81u;
        if(tick%120u!=offset || count(s)>=16) return;
        const int arrivals=1+static_cast<int>((key>>16)%4u==0u);
        for(int j=0;j<arrivals && count(s)<16;++j) drop(s,sample_practice_drop(s,context,window*2u+static_cast<uint32_t>(j)));
    }
}
CUDARL_HD inline float advance(State& s,const float* action,V current) {
    for(auto& p:s.pellets) move(p,s.retention?V{}:current);
    const bool targeted=s.target>=0 && s.pellets[s.target].active;
    const float before=targeted?deep::norm(s.pellets[s.target].position-s.body.position):0;
    deep::Physics physics;physics.pond=true;physics.water_velocity=s.retention?V{}:current;deep::advance_physics(s.body,action,physics);
    ++s.steps;++s.since_food;float reward=-.005f;
    if(!(deep::norm(s.body.position)<100.f)) s.invalid=true;
    if(deep::norm(s.body.position-s.anchor)>.1f || !targeted) {s.anchor=s.body.position;s.still_steps=0;}
    else {++s.still_steps;s.max_still=s.still_steps>s.max_still?s.still_steps:s.max_still;}
    if(targeted) reward+=3.f*(before-deep::norm(s.pellets[s.target].position-s.body.position));
    for(int j=0;j<4;++j) {const float a=deep::clamp(action[j],-1,1),difference=a-s.last_action[j];reward-=.003f*difference*difference;s.last_action[j]=a;}
    for(auto& p:s.pellets) if(p.active && deep::norm(p.position-s.body.position)<.24f) {p.active=false;++s.eaten;s.since_food=0;reward+=5;s.decision=true;}
    if(s.steps%20==0) {
        for(auto& p:s.pellets) if(p.active) {const float d=deep::norm(p.position-s.body.position);p.progress=deep::clamp((p.distance-d),-2.f,2.f);p.distance=d;}
        s.decision=true;
    }
    return reward;
}
}
#endif
