#ifndef CUDARL_DEEP_POND_H
#define CUDARL_DEEP_POND_H
#include "deep.h"
namespace cudarl::deep_pond {
using deep::V;
constexpr int kFoods=3,kObs=33,kAct=4,kEpisodeSteps=1200;
constexpr float separation=2.f,vertical_gap=.5f,relocation=2.5f;
struct State {
    deep::BodyState body;
    V food[kFoods]{{99,99,99},{99,99,99},{99,99,99}};
    float last_action[kAct]{};
    uint32_t seed=7,environment=0,episode=0,placements[kFoods]{};
    int steps=0,eaten=0,since_food=0,last_eaten=-1;
};
CUDARL_HD inline bool water(V p) {
    return p.y>=-3.35f && p.y<=-.65f && p.y>deep::pond_floor(p.x,p.z)+.4f &&
           p.x*p.x/(4.6f*4.6f)+p.z*p.z/(3.5f*3.5f)<.85f*.85f;
}
CUDARL_HD inline bool valid_food(const State& s,int slot,V p) {
    if(!water(p) || deep::norm(p-s.body.position)<.8f) return false;
    const auto geometry=deep::shape(s.body);
    for(const auto& l:geometry.link) {
        const V direction=l.q.rotate({1,0,0}),offset=p-l.center/deep::scale;
        const float t=deep::clamp(deep::dot(offset,direction),-.5f*l.length/deep::scale,.5f*l.length/deep::scale);
        if(deep::norm(offset-direction*t)<.35f) return false;
    }
    for(int j=0;j<kFoods;++j) if(j!=slot && water(s.food[j]))
        if(deep::norm(p-s.food[j])<separation || fabsf(p.y-s.food[j].y)<vertical_gap) return false;
    return !water(s.food[slot]) || deep::norm(p-s.food[slot])>=relocation;
}
CUDARL_HD inline bool place_food(State& s,int slot) {
    const uint32_t base=rng::bits(s.seed,s.episode,s.placements[slot]++);
    for(uint32_t i=0;i<4096;++i) {
        const uint32_t t=base+i*3u,stream=s.environment*7u+static_cast<uint32_t>(slot)+451u;
        const float angle=rng::uniform(s.seed,stream,t)*6.2831853f;
        const float r=sqrtf(rng::uniform(s.seed,stream,t+1))* .85f;
        const V p{4.6f*r*cosf(angle),-3.35f+2.7f*rng::uniform(s.seed,stream,t+2),3.5f*r*sinf(angle)};
        if(valid_food(s,slot,p)) {s.food[slot]=p;return true;}
    }
    // Deterministic coverage fallback; still honors every placement constraint.
    for(int y=0;y<19;++y) for(int z=0;z<19;++z) for(int x=0;x<23;++x) {
        const V p{-3.8f+x*(7.6f/22.f),-3.35f+y*.15f,-2.8f+z*(5.6f/18.f)};
        if(valid_food(s,slot,p)) {s.food[slot]=p;return true;}
    }
    return false;
}
CUDARL_HD inline State initial(uint32_t seed,uint32_t environment,uint32_t episode=0,swimmer::Body unused={}) {
    State s;s.seed=seed;s.environment=environment;s.episode=episode;
    const auto origin=deep::initial(seed,environment,episode,unused);s.body=origin.body;
    // Central starts keep the complete tail clear of the sloped bank.
    s.body.position.x*=.35f;s.body.position.z*=.35f;
    for(int i=0;i<kFoods;++i) place_food(s,i);
    return s;
}
CUDARL_HD inline void observe(const State& s,float* o) {
    deep::State original;original.body=s.body;original.food=s.food[0];
    for(int j=0;j<4;++j) original.last_action[j]=s.last_action[j];
    float old[deep::kObs];deep::observe(original,old);
    for(int i=0;i<3;++i) o[i]=old[i];
    for(int f=1;f<3;++f) {
        const auto v=s.body.orientation.inverse(s.food[f]-s.body.position)/2.f;
        o[f*3]=v.x;o[f*3+1]=v.y;o[f*3+2]=v.z;
    }
    for(int i=3;i<deep::kObs;++i) o[i+6]=old[i];
}
CUDARL_HD inline float potential(const State& s) {
    float minimum=1e10f,distances[3];
    for(int i=0;i<3;++i) {distances[i]=deep::norm(s.food[i]-s.body.position);minimum=fminf(minimum,distances[i]);}
    float sum=0;for(float d:distances) sum+=expf(-4.f*(d-minimum));return -minimum+logf(sum)/4.f;
}
struct Result {float reward=0;int eaten_slot=-1;bool timeout=false;};
CUDARL_HD inline Result advance(State& s,const float* action) {
    const float before=potential(s);deep::Physics physics;physics.pond=true;
    deep::advance_physics(s.body,action,physics);++s.steps;++s.since_food;s.last_eaten=-1;
    Result result;result.reward=3.f*(potential(s)-before)-.005f;
    for(int j=0;j<4;++j) {
        const float a=deep::clamp(action[j],-1,1),difference=a-s.last_action[j];
        result.reward-=.003f*difference*difference;
        const float near=deep::clamp((fabsf(s.body.angle[j])-.8f*deep::limit)/(.2f*deep::limit),0,1);
        result.reward-=.03f*near*fmaxf(0,a*(s.body.angle[j]>=0?1.f:-1.f));s.last_action[j]=a;
    }
    for(int i=0;i<3;++i) if(deep::norm(s.food[i]-s.body.position)<.24f) {
        result.reward+=5;++s.eaten;s.since_food=0;s.last_eaten=i;result.eaten_slot=i;place_food(s,i);break;
    }
    result.timeout=s.steps>=kEpisodeSteps;return result;
}
CUDARL_HD inline void permute(float* o,int order) {
    const int orders[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    float previous[9];for(int i=0;i<9;++i) previous[i]=o[i];
    for(int f=0;f<3;++f) for(int k=0;k<3;++k) o[f*3+k]=previous[orders[order][f]*3+k];
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
void shuffle_food_inputs(float* obs,int rows,int environments,uint32_t seed,uint32_t first_tick,cudaStream_t stream);
}
#endif
