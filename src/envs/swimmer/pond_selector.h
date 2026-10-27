#ifndef CUDARL_ENV_POND_SELECTOR_H
#define CUDARL_ENV_POND_SELECTOR_H
#include "demos/pond_target.h"
namespace cudarl::pond_selector {
constexpr int kObs=36,kActions=3,kFeatures=32;
// One shared score per food. No slot ID or slot-specific output weights.
// Symmetric summaries of other foods retain route context without order bias.
CUDARL_HD inline void features(const float* o,int food,float* x) {
    for(int j=0;j<3;++j) x[j]=o[food*3+j];
    for(int j=0;j<24;++j) x[3+j]=o[9+j];
    x[27]=o[33+food];float distance=0;for(int j=0;j<3;++j) distance+=x[j]*x[j];x[28]=sqrtf(distance);
    float minimum=1e10f,maximum=0,other_nearest=1e10f;
    for(int other=0;other<3;++other) if(other!=food) {
        float gap=0,origin=0;for(int j=0;j<3;++j) {const float p=o[other*3+j],v=x[j]-p;gap+=v*v;origin+=p*p;}
        minimum=fminf(minimum,gap);maximum=fmaxf(maximum,gap);other_nearest=fminf(other_nearest,origin);
    }
    x[29]=sqrtf(minimum);x[30]=sqrtf(maximum);x[31]=sqrtf(other_nearest);
}
void make_features(const float* observations,float* output,int n,cudaStream_t stream);
CUDARL_HD inline void observe(const deep_pond::State& state,int target,float* output) {
    deep_pond::observe(state,output);
    for(int j=0;j<3;++j) output[33+j]=j==target?1.f:0.f;
}
class Batch {
public:
    Batch(int n,uint32_t seed,int decision_steps,bool recovery=false);
    const float* obs() const {return observations_.get();}
    const float* motor_obs(cudaStream_t stream);
    const deep_pond::State* states() const {return states_.get();}
    void begin(const int* targets,swimmer::Outputs out,cudaStream_t stream);
    void step(const float* torques,swimmer::Outputs out,cudaStream_t stream,bool autoreset=true);
    void save(checkpoint::Writer& out) const;
    void load(checkpoint::Reader& in);
private:
    int n_,decision_steps_,episode_steps_;uint32_t seed_;bool recovery_;
    device_ptr<deep_pond::State> states_;device_ptr<uint32_t> episodes_;
    device_ptr<int> targets_;device_ptr<float> observations_,motor_inputs_;
};
}
#endif
