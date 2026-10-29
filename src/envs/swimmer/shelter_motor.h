#ifndef CUDARL_SHELTER_MOTOR_H
#define CUDARL_SHELTER_MOTOR_H
#include "shelter_goal.h"
#include "predator.h"
#include "rl/progress_selector_policy.h"
#if CUDARL_WITH_CUDA
#include "nn/mlp_gpu.h"
#endif

namespace cudarl::shelter_motor {
constexpr int kObs=59,kAct=4,kCoverEpisode=400;
struct ActuatorScale {float value[kAct]{1,1,1,1};};
CUDARL_HD inline float action_mean(float raw,bool cover,int joint,const ActuatorScale& scale) {
    return cover?raw/scale.value[joint]:raw;
}
struct Request {
    shelter::Goal goal{};
    deep::V heading{0,0,1},eye{-8,4,-1.25f};
    bool cover=false;
};
struct State {
    arch::State state{};
    Request request{};
    float exposure=1;
    int held_ticks=0,peak_hold=0;
};
CUDARL_HD inline void observe(const State& world,float* out,deep::Basin basin={2}) {
    shelter::Goal goal=world.request.goal;
    if(!world.request.cover) goal.cover=-1;
    shelter::motor_observe59(world.state,goal,world.request.heading,out,basin);
}
#if CUDARL_WITH_CUDA
void policy_means(const float* raw,const float* observations,float* means,int count,ActuatorScale scale,cudaStream_t stream);
void policy_mean_grad(float* gradient,const float* observations,int count,ActuatorScale scale,cudaStream_t stream);
class Batch {
    int n_;uint32_t seed_;arch::Context context_{{},0,{2}};
    device_ptr<State> states_;
    device_ptr<float> observations_,features_;
    nn::GpuMlp selector_;
    ActuatorScale action_scales_{};
    void prepare(cudaStream_t stream);
public:
    Batch(int n,uint32_t seed,swimmer::Body unused={});
    void configure(const std::vector<float>& shared,const arch::Context& context,cudaStream_t stream,bool reset=false);
    void configure(const std::vector<float>& shared,cudaStream_t stream,bool reset=false) {configure(shared,context_,stream,reset);}
    arch::Context context() const {return context_;}
    ActuatorScale action_scales() const {return action_scales_;}
    void set_action_scales(ActuatorScale scale);
    const float* obs() const {return observations_.get();}
    const State* states() const {return states_.get();}
    void step(const float* torques,swimmer::Outputs out,cudaStream_t stream,bool autoreset=true,bool practice=true);
    State state(cudaStream_t stream) const;
    std::vector<State> host_states(cudaStream_t stream) const;
    void set_states(const std::vector<State>&,cudaStream_t stream);
    void set_state(const State&,cudaStream_t stream);
    void set_request(const Request&,cudaStream_t stream);
    void save(checkpoint::Writer&) const;
    void load(checkpoint::Reader&);
};
#endif
}
#endif
