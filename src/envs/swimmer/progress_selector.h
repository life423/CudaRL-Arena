#ifndef CUDARL_PROGRESS_SELECTOR_H
#define CUDARL_PROGRESS_SELECTOR_H
#include "pond_selector.h"
namespace cudarl::progress_selector {
constexpr int kObs=40,kFeatures=34;
struct History {float distance[3]{},progress[3]{};uint32_t placements[3]{};};
CUDARL_HD inline History initial_history(const deep_pond::State& s) {
    History h;for(int f=0;f<3;++f) {h.distance[f]=deep::norm(s.food[f]-s.body.position);h.placements[f]=s.placements[f];}return h;
}
CUDARL_HD inline void update_history(const deep_pond::State& s,History& h,int period) {
    const float seconds=period*deep::dt;
    for(int f=0;f<3;++f) {
        const float distance=deep::norm(s.food[f]-s.body.position);
        h.progress[f]=s.placements[f]==h.placements[f]?deep::clamp((h.distance[f]-distance)/seconds,-2.f,2.f):0.f;
        h.distance[f]=distance;h.placements[f]=s.placements[f];
    }
}
CUDARL_HD inline void observe(const deep_pond::State& s,int target,const History& h,float* out) {
    pond_selector::observe(s,target,out);out[36]=deep::clamp(s.since_food*.05f/20.f,0,1);
    for(int f=0;f<3;++f) out[37+f]=h.progress[f];
}
CUDARL_HD inline void features(const float* obs,int food,float* out) {
    pond_selector::features(obs,food,out);out[32]=obs[36];out[33]=obs[37+food];
}
CUDARL_HD inline deep::V sample_current(uint32_t seed,uint32_t environment,uint32_t episode,float maximum) {
    if(rng::uniform(seed^0xF10A83C1u,environment,episode*3)<.25f) return {};
    const float speed=maximum*rng::uniform(seed^0xF10A83C1u,environment,episode*3+1);
    const float direction=6.2831853f*rng::uniform(seed^0xF10A83C1u,environment,episode*3+2);
    return {speed*cosf(direction),0,speed*sinf(direction)};
}
#if CUDARL_WITH_CUDA
void make_features(const float* obs,float* out,int n,cudaStream_t stream);
class Batch {
public:
    Batch(int n,uint32_t seed,int period=20,float max_current=.01f,bool training=true,deep::V fixed_current={});
    const float* obs() const {return observations_.get();}
    const deep_pond::State* states() const {return states_.get();}
    const float* motor_obs(cudaStream_t stream);
    void begin(const int* targets,swimmer::Outputs out,cudaStream_t stream);
    void step(const float* torques,swimmer::Outputs out,cudaStream_t stream,bool autoreset=true);
    void save(checkpoint::Writer& out) const;
    void load(checkpoint::Reader& in);
private:
    int n_,period_;uint32_t seed_;float maximum_;bool training_;deep::V fixed_;
    device_ptr<deep_pond::State> states_;device_ptr<uint32_t> episodes_;device_ptr<int> targets_;
    device_ptr<History> history_;device_ptr<deep::V> currents_;
    device_ptr<float> observations_,motor_inputs_;
};
#endif
}
#endif
