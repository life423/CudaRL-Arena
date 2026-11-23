#ifndef CUDARL_CURRENT_SWIM_MATH_H
#define CUDARL_CURRENT_SWIM_MATH_H
#include "rl/gaussian_math.h"
namespace cudarl::current_swim_policy_v1 {
constexpr int parameters=16;
constexpr float fixed_sigma=.5f,coefficient_limit=.5f,flow_scale_m_s=.002f;
template<class T> CUDARL_HD inline T coefficient(T u) {return T(coefficient_limit)*tanh(u);}
template<class T> CUDARL_HD inline T log_probability(const T* means,const T* latent) {
    T logs[parameters];for(int j=0;j<parameters;++j)logs[j]=log(T(fixed_sigma));
    return rl::gaussian_logp(means,logs,latent,parameters);
}
// This is one persistent policy choice for a whole physical episode. Its score
// is the Gaussian latent u, never theta, tanh joint commands or per-control rows.
template<class T> CUDARL_HD inline rl::PpoRowTerms<T> loss(const T* means,const T* latent,
        T old_logp,T advantage,T scale,T* gradient) {
    T logs[parameters],unused_logs[parameters],unused_value;
    for(int j=0;j<parameters;++j)logs[j]=log(T(fixed_sigma));
    return rl::ppo_gaussian_row(means,logs,parameters,latent,old_logp,advantage,T(0),T(0),
                               {.2f,0,0},scale,gradient,unused_logs,&unused_value);
}
// One frozen body network still supplies all four joint latents. The learned
// flow coordinates adjust both trim and its existing phase-dependent amplitude.
// Exact zero-flow passthrough is deliberate: no floating-point zero additions.
template<class T> CUDARL_HD inline void joint_latents(const T* episode_latent,
        const T* backbone,const T* body_flow,T* output) {
    if(body_flow[0]==T(0)&&body_flow[1]==T(0)&&body_flow[2]==T(0)) {
        for(int j=0;j<4;++j)output[j]=backbone[j];return;
    }
    for(int j=0;j<4;++j) {
        const T forward=body_flow[0],side=body_flow[2],base=backbone[j];
        const T trim=coefficient(episode_latent[j*4])*forward+
                     coefficient(episode_latent[j*4+1])*side;
        const T amplitude=coefficient(episode_latent[j*4+2])*forward*base+
                          coefficient(episode_latent[j*4+3])*side*base;
        output[j]=base+trim+amplitude;
    }
}
}
#endif
