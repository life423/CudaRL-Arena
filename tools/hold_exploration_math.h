#ifndef CUDARL_HOLD_EXPLORATION_MATH_H
#define CUDARL_HOLD_EXPLORATION_MATH_H
#include "rl/gaussian_math.h"
#include "rl/ar_gaussian_math.h"
namespace cudarl::hold_exploration {
template<class T>
CUDARL_HD inline void conditional(const T* base, const T* previous_base, const T* previous_action,
                                  const T* gains, const T* log_sigma, bool holding, T amplitude,
                                  T rho, bool history, T* mean, T* sigma) {
    if (!holding) { rl::ar_parameters(base,previous_base,previous_action,log_sigma,4,rho,history,mean,sigma); return; }
    const T r = history ? rho : T(0), g = holding ? amplitude : T(1);
    const T correction = history ? T(.5)*log(T(1)-rho*rho) : T(0);
    for (int j=0;j<4;++j) {
        const T a = holding ? gains[j] : T(1);
        mean[j] = g*a*(base[j]-r*previous_base[j])+r*previous_action[j];
        sigma[j] = log_sigma[j]+log(g)+correction;
    }
}
template<class T>
CUDARL_HD inline rl::PpoRowTerms<T> loss(const T* base, const T* previous_base, const T* previous_action,
                                        const T* gains, const T* log_sigma, bool holding, T amplitude,
                                        T rho, bool history, const T* action, T old_logp, T advantage,
                                        T scale, T* dgain) {
    T mean[4],sigma[4],dmean[4],unused[4],dvalue;
    conditional(base,previous_base,previous_action,gains,log_sigma,holding,amplitude,rho,history,mean,sigma);
    const auto terms=rl::ppo_gaussian_row(mean,sigma,4,action,old_logp,advantage,T(0),T(0),
                                         {.2f,0,0},scale,dmean,unused,&dvalue);
    const T r=history?rho:T(0);
    for(int j=0;j<4;++j)dgain[j]=holding?dmean[j]*amplitude*(base[j]-r*previous_base[j]):T(0);
    return terms;
}
} // namespace cudarl::hold_exploration
#endif
