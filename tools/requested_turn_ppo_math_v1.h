#ifndef CUDARL_REQUESTED_TURN_PPO_MATH_V1_H
#define CUDARL_REQUESTED_TURN_PPO_MATH_V1_H
#include "rl/ar_gaussian_math.h"
namespace cudarl::requested_turn_ppo_math_v1 {
constexpr float rho=.8f,log_sigma=-1.f,clip=.2f,value_clip=.2f,value_coefficient=.5f;
// Clipped value loss: max of the original and old-value-clipped squared error.
// The clipped branch has zero derivative outside the clip interval. At ties
// choose the original branch, matching the ordinary differentiable objective.
template<class T> CUDARL_HD inline T value_loss(T current,T old,T target,T scale,T* gradient){
 const T difference=current-old;const T bounded=difference<T(-value_clip)?T(-value_clip):difference>T(value_clip)?T(value_clip):difference;
 const T error=current-target,clipped_error=old+bounded-target;
 const bool original=error*error>=clipped_error*clipped_error;
 *gradient=scale*T(value_coefficient)*(original?error:((difference>=T(-value_clip)&&difference<=T(value_clip))?clipped_error:T(0)));
 return T(.5)*(original?error*error:clipped_error*clipped_error);
}
template<class T> CUDARL_HD inline cudarl::rl::PpoRowTerms<T> loss(
 const T* current,const T* previous,const T* previous_raw,bool history,const T* raw,T old_logp,T advantage,
 T value,T old_value,T target,T scale,T* dcurrent,T* dprevious,T* dvalue){
 const T sigma[4]={T(log_sigma),T(log_sigma),T(log_sigma),T(log_sigma)};
 auto terms=cudarl::rl::ppo_ar_row(current,previous,previous_raw,sigma,4,T(rho),history,raw,old_logp,advantage,
     value,target,{clip,0,0},scale,dcurrent,dprevious,dvalue);
 terms.v_loss=value_loss(value,old_value,target,scale,dvalue);return terms;
}
}
#endif
