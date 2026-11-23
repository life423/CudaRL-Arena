#ifndef CUDARL_GAIT_POLICY_V1_H
#define CUDARL_GAIT_POLICY_V1_H
#include "rl/gaussian_math.h"
namespace cudarl::gait_policy_v1 {
constexpr unsigned version=1;
constexpr int parameters=8,actions=4;
constexpr float fixed_sigma=.35f,bias_limit=.75f;
template<class T> CUDARL_HD inline T bias(T latent) {
    return T(bias_limit)*tanh(latent);
}
template<class T> CUDARL_HD inline T gain(T latent) {
    return exp(log(T(2))*tanh(latent));
}
template<class T> CUDARL_HD inline T log_probability(const T* means,const T* latent) {
    T logs[parameters];for(int j=0;j<parameters;++j)logs[j]=log(T(fixed_sigma));
    return rl::gaussian_logp(means,logs,latent,parameters);
}
// A Gaussian row is one persistent choice for a complete physical episode.
// The scored variable is the original eight-dimensional u, not either bounded
// coefficient map, tanh joint commands, or repeated copies at every control.
template<class T> CUDARL_HD inline rl::PpoRowTerms<T> loss(const T* means,const T* latent,
        T old_logp,T advantage,T scale,T* gradient) {
    T logs[parameters],unused_logs[parameters],unused_value;
    for(int j=0;j<parameters;++j)logs[j]=log(T(fixed_sigma));
    return rl::ppo_gaussian_row(means,logs,parameters,latent,old_logp,advantage,T(0),T(0),
                               {.2f,0,0},scale,gradient,unused_logs,&unused_value);
}
template<class T> CUDARL_HD inline void swim_latents(const T* latent,const T* backbone,T* output) {
    bool identity=true;for(int j=0;j<parameters;++j)identity=identity&&latent[j]==T(0);
    if(identity){for(int j=0;j<actions;++j)output[j]=backbone[j];return;}
    for(int j=0;j<actions;++j)output[j]=gain(latent[4+j])*backbone[j]+bias(latent[j]);
}
// The saved holding policy is an exact, untrained branch of this same motor.
// The study never changes its two learned parameters, saved gain, or backbone.
template<class T> CUDARL_HD inline void motor_latents(bool stationary_request,const T* latent,
        const T* backbone,T saved_hold_gain,T* output) {
    if(stationary_request){for(int j=0;j<actions;++j)output[j]=saved_hold_gain*backbone[j];return;}
    swim_latents(latent,backbone,output);
}
}
#endif
