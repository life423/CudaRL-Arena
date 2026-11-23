#ifndef CUDARL_TURNING_POLICY_V2_H
#define CUDARL_TURNING_POLICY_V2_H
#include "rl/gaussian_math.h"

namespace cudarl::turning_policy_v2 {
constexpr unsigned version=2;
constexpr int parameters=80,actions=4,raw_features=20,features=20;
constexpr float fixed_sigma=.15f,steering_limit=.75f;

// Raw0..7 and derived0..7 retain the version-one sensor meanings exactly.
// Raw8..19 are normalized depth, COM vertical speed, roll, pitch, four
// joint angles and four joint rates. They are bounded only in this feature
// map; the physical state and the inherited motor observation are unchanged.
template<class T> CUDARL_HD inline void derived_features(const T* raw,T* output) {
    output[0]=T(1);
    output[1]=T(1)/(T(1)+raw[0]);
    output[2]=T(1)/(T(1)+raw[1])-T(1)/(T(1)+raw[2]);
    output[3]=raw[3];output[4]=raw[4];
    output[5]=tanh(T(8)*raw[5]);
    output[6]=tanh(T(2)*(T(1)-raw[6]));
    output[7]=tanh(raw[7]);
    for(int k=8;k<features;++k)output[k]=tanh(raw[k]);
}

template<class T> CUDARL_HD inline T log_probability(const T* means,const T* latent) {
    T logs[parameters];for(int j=0;j<parameters;++j)logs[j]=log(T(fixed_sigma));
    return rl::gaussian_logp(means,logs,latent,parameters);
}

// One Gaussian coefficient vector is scored once and persists for an episode.
// Its state-conditioned motor commands are recomputed at every control tick.
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
    for(int joint=0;joint<actions;++joint){T sum=0;
        for(int k=0;k<features;++k)sum+=tanh(latent[joint*features+k])*f[k];
        output[joint]=backbone[joint]+T(steering_limit)*tanh(sum);
    }
}

// The four-output motor retains the exact saved explicit Swim and Rest paths.
// Only autonomous mode enables the version-two all-axis correction.
template<class T> CUDARL_HD inline void motor_latents(bool autonomous,bool stationary,
        const T* latent,const T* backbone,T saved_hold_gain,const T* raw,T* output) {
    if(stationary){for(int j=0;j<actions;++j)output[j]=saved_hold_gain*backbone[j];return;}
    if(!autonomous){for(int j=0;j<actions;++j)output[j]=backbone[j];return;}
    autonomous_latents(latent,backbone,raw,output);
}
}
#endif
