#ifndef CUDARL_HOLD_LEARNED_AMPLITUDE_MATH_H
#define CUDARL_HOLD_LEARNED_AMPLITUDE_MATH_H
#include "rl/gaussian_math.h"
namespace cudarl::learned_amplitude {
template<class T> CUDARL_HD inline T sigmoid(T x) {
    if(x>=T(0))return T(1)/(T(1)+exp(-x));
    const T e=exp(x);return e/(T(1)+e);
}
template<class T> CUDARL_HD inline T standard_deviation(T variance_coordinate) {
    return T(.05)+T(7.95)*sigmoid(variance_coordinate);
}
template<class T> CUDARL_HD inline T log_std_derivative(T variance_coordinate) {
    const T s=sigmoid(variance_coordinate);
    return T(7.95)*s*(T(1)-s)/(T(.05)+T(7.95)*s);
}
template<class T> CUDARL_HD inline T log_probability(const T* parameters,T latent) {
    const T mean[1]={parameters[0]},logs[1]={log(standard_deviation(parameters[1]))};
    return rl::gaussian_logp(mean,logs,&latent,1);
}
// One PPO row is ONE complete physical episode, never one motor control.
template<class T>
CUDARL_HD inline rl::PpoRowTerms<T> loss(const T* parameters,T latent,T old_logp,T advantage,T scale,T* gradient) {
    const T mean[1]={parameters[0]},logs[1]={log(standard_deviation(parameters[1]))};
    T dmean[1],dlogs[1],unused;
    const auto terms=rl::ppo_gaussian_row(mean,logs,1,&latent,old_logp,advantage,T(0),T(0),
                                         {.2f,0,0},scale,dmean,dlogs,&unused);
    gradient[0]=dmean[0];gradient[1]=dlogs[0]*log_std_derivative(parameters[1]);return terms;
}
} // namespace cudarl::learned_amplitude
#endif
