#ifndef CUDARL_TURNING_POLICY_V1_H
#define CUDARL_TURNING_POLICY_V1_H
#include "rl/gaussian_math.h"

namespace cudarl::turning_policy_v1 {
constexpr unsigned version=1;
constexpr int parameters=16,actions=4,raw_features=8,features=8;
constexpr float fixed_sigma=.20f,steering_limit=.75f;

// Raw geometry/motion sensors, not turn commands: three directional full-body
// gaps in body lengths; outward ellipse normal projected on forward/side;
// world-Y yaw rate /4; COM forward/side speed /.008 m/s.
template<class T> CUDARL_HD inline void derived_features(const T* raw,T* output) {
    output[0]=T(1);
    output[1]=T(1)/(T(1)+raw[0]);
    output[2]=T(1)/(T(1)+raw[1])-T(1)/(T(1)+raw[2]);
    output[3]=raw[3];output[4]=raw[4];
    output[5]=tanh(T(8)*raw[5]);
    output[6]=tanh(T(2)*(T(1)-raw[6]));
    output[7]=tanh(raw[7]);
}

template<class T> CUDARL_HD inline T log_probability(const T* means,const T* latent) {
    T logs[parameters];for(int j=0;j<parameters;++j)logs[j]=log(T(fixed_sigma));
    return rl::gaussian_logp(means,logs,latent,parameters);
}

// One scored Gaussian choice lasts a complete episode. The original latent u
// is scored once; bounded coefficients and actuator commands are not rescored.
template<class T> CUDARL_HD inline rl::PpoRowTerms<T> loss(const T* means,const T* latent,
        T old_logp,T advantage,T scale,T* gradient) {
    T logs[parameters],unused_logs[parameters],unused_value;
    for(int j=0;j<parameters;++j)logs[j]=log(T(fixed_sigma));
    return rl::ppo_gaussian_row(means,logs,parameters,latent,old_logp,advantage,T(0),T(0),
                               {.2f,0,0},scale,gradient,unused_logs,&unused_value);
}

template<class T> CUDARL_HD inline void autonomous_latents(const T* latent,const T* backbone,
        const T* raw,T* output) {
    bool identity=true;for(int j=0;j<parameters;++j)identity=identity&&latent[j]==T(0);
    if(identity){for(int j=0;j<actions;++j)output[j]=backbone[j];return;}
    T f[features];derived_features(raw,f);
    for(int j=0;j<actions;++j)output[j]=backbone[j];
    for(int joint=0;joint<2;++joint){T sum=0;
        for(int k=0;k<features;++k)sum+=tanh(latent[joint*features+k])*f[k];
        output[joint*2]=backbone[joint*2]+T(steering_limit)*tanh(sum);
    }
}

// Explicit requested swimming and holding retain their existing meanings and
// saved weights in this same four-output motor. Only the new autonomous mode
// uses the learned boundary-conditioned yaw correction.
template<class T> CUDARL_HD inline void motor_latents(bool autonomous,bool stationary,
        const T* latent,const T* backbone,T saved_hold_gain,const T* raw,T* output) {
    if(stationary){for(int j=0;j<actions;++j)output[j]=saved_hold_gain*backbone[j];return;}
    if(!autonomous){for(int j=0;j<actions;++j)output[j]=backbone[j];return;}
    autonomous_latents(latent,backbone,raw,output);
}
}
#endif
