#ifndef CUDARL_DEEP_SWIMMER_H
#define CUDARL_DEEP_SWIMMER_H
#include "deep_physics.h"
#include "swimmer.h"
#include "core/checkpoint.h"
namespace cudarl::deep {
constexpr int kObs=27,kAct=4,kEpisodeSteps=1200;
struct State {
    BodyState body;
    V food;
    float last_action[kAct]{};
    uint32_t seed=1,environment=0,episode=0,placements=0;
    int steps=0,eaten=0,since_food=0;
};
CUDARL_HD inline void place_food(State& s) {
    const uint32_t base=s.episode*7919u+s.placements++*2053u;
    for(uint32_t i=0;i<1024;++i) {
        const uint32_t t=base+i*3;
        V p{s.body.position.x+(rng::uniform(s.seed,s.environment,t+101u)*2-1)*2.3f,
            -3.2f+rng::uniform(s.seed,s.environment,t+102u)*2.4f,
            s.body.position.z+(rng::uniform(s.seed,s.environment,t+103u)*2-1)*2.3f};
        const float distance=norm(p-s.body.position);
        if(fabsf(p.x)>3.3f || fabsf(p.z)>2.3f || distance<1.2f || distance>2.6f ||
           fabsf(p.y-s.body.position.y)<.45f || (s.placements>1 && norm(p-s.food)<1.3f)) continue;
        s.food=p;return;
    }
    // Deterministic reachable fallback, never places food on the creature.
    s.food={0,s.body.position.y< -2.f?-.9f:-3.1f,0};
}
CUDARL_HD inline State initial(uint32_t seed,uint32_t environment,uint32_t episode=0,swimmer::Body unused={}) {
    (void)unused;State s;s.seed=seed;s.environment=environment;s.episode=episode;
    s.body.position={(rng::uniform(seed,environment,episode*13+201)*2-1)*1.5f,
                     -2.8f+rng::uniform(seed,environment,episode*13+202)*1.6f,
                     (rng::uniform(seed,environment,episode*13+203)*2-1)};
    s.body.orientation=axis_rotation({0,1,0},rng::uniform(seed,environment,episode*13+204)*6.2831853f);
    for(int i=0;i<4;++i) s.body.angle[i]=(rng::uniform(seed,environment,episode*13+205+i)*2-1)*.25f;
    place_food(s);return s;
}
CUDARL_HD inline void observe(const State& s,float* o) {
    const auto q=s.body.orientation;
    const V f=q.inverse(s.food-s.body.position),v=q.inverse({s.body.velocity[0],s.body.velocity[1],s.body.velocity[2]}),
            w=q.inverse({s.body.velocity[3],s.body.velocity[4],s.body.velocity[5]}),up=q.inverse({0,1,0});
    o[0]=f.x/2;o[1]=f.y/2;o[2]=f.z/2;
    o[3]=v.x/.04f;o[4]=v.y/.04f;o[5]=v.z/.04f;
    o[6]=w.x/4;o[7]=w.y/4;o[8]=w.z/4;o[9]=up.x;o[10]=up.y;o[11]=up.z;
    for(int j=0;j<4;++j) {o[12+j]=s.body.angle[j]/limit;o[16+j]=s.body.velocity[6+j]/12.f;o[23+j]=s.last_action[j];}
    o[20]=s.body.position.x/4.5f;o[21]=(s.body.position.y+2)/2;o[22]=s.body.position.z/3.5f;
}
struct Result {float reward=0;bool ate=false,timeout=false;};
CUDARL_HD inline Result advance(State& s,const float* action) {
    const float before=norm(s.food-s.body.position);advance_physics(s.body,action);++s.steps;++s.since_food;
    Result r;r.reward=3.f*(before-norm(s.food-s.body.position))-.005f;
    for(int j=0;j<kAct;++j) {
        const float a=clamp(action[j],-1,1),difference=a-s.last_action[j];
        r.reward-=.003f*difference*difference;
        const float near=clamp((fabsf(s.body.angle[j])-.8f*limit)/(.2f*limit),0,1);
        r.reward-=.03f*near*fmaxf(0,a*(s.body.angle[j]>=0?1.f:-1.f));s.last_action[j]=a;
    }
    if(norm(s.food-s.body.position)<.24f) {r.reward+=5;++s.eaten;s.since_food=0;r.ate=true;place_food(s);}
    r.timeout=s.steps>=kEpisodeSteps;return r;
}
class Batch {
public:
    Batch(int n,uint32_t seed,swimmer::Body unused={});
    void step(const float* actions,swimmer::Outputs out,cudaStream_t stream,bool autoreset=true);
    const float* obs() const {return obs_.get();}
    const State* states() const {return states_.get();}
    void save(checkpoint::Writer& out) const;
    void load(checkpoint::Reader& in);
private:
    int n_;uint32_t seed_;
    device_ptr<State> states_;device_ptr<uint32_t> episodes_;device_ptr<float> obs_;
};
}
#endif
