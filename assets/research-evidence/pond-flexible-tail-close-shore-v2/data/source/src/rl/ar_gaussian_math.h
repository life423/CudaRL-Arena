#ifndef CUDARL_RL_AR_GAUSSIAN_MATH_H
#define CUDARL_RL_AR_GAUSSIAN_MATH_H
#include "gaussian_math.h"
namespace cudarl::rl {
constexpr int kArMaxDimensions = 16;
// State is (current observation, previous observation, previous raw latent).
// Both means belong to CURRENT theta, including the previous-state mean.
// Fixed sigma: z_t=mu_t+rho*(z_prev-mu_prev)+sigma*sqrt(1-rho^2)*eta.
// A true reset draws N(mu_t,sigma^2), making the noise stationary immediately.
template <typename T>
CUDARL_HD inline void ar_parameters(const T *current, const T *previous, const T *previous_action,
                                    const T *log_std, int dim, T rho, bool history, T *mean,
                                    T *conditional_log_std) {
    const T coefficient = history ? rho : T(0);
    const T correction = history ? T(.5) * log(T(1) - rho * rho) : T(0);
    for (int j = 0; j < dim; ++j) {
        mean[j] =
            current[j] + (coefficient != T(0) ? coefficient * (previous_action[j] - previous[j]) : T(0));
        conditional_log_std[j] = log_std[j] + correction;
    }
}
template <typename T>
CUDARL_HD inline T ar_logp(const T *current, const T *previous, const T *previous_action, const T *log_std,
                           int dim, T rho, bool history, const T *action) {
    T mean[kArMaxDimensions], sigma[kArMaxDimensions];
    ar_parameters(current, previous, previous_action, log_std, dim, rho, history, mean, sigma);
    return gaussian_logp(mean, sigma, action, dim);
}
template <typename T> CUDARL_HD inline T tanh_log_jacobian(const T *latent, int dim) {
    T result = 0;
    for (int j = 0; j < dim; ++j) {
        const T x = fabs(latent[j]);
        result += T(2) * (log(T(2)) - x - log(T(1) + exp(-T(2) * x)));
    }
    return result;
}
template <typename T>
CUDARL_HD inline PpoRowTerms<T> ppo_ar_row(const T *current, const T *previous, const T *previous_action,
                                           const T *log_std, int dim, T rho, bool history, const T *action,
                                           T old_logp, T advantage, T value, T target, PpoCoefficients coeff,
                                           T scale, T *dcurrent, T *dprevious, T *dvalue) {
    T mean[kArMaxDimensions], sigma[kArMaxDimensions], unused[kArMaxDimensions];
    ar_parameters(current, previous, previous_action, log_std, dim, rho, history, mean, sigma);
    const auto terms = ppo_gaussian_row(mean, sigma, dim, action, old_logp, advantage, value, target, coeff,
                                        scale, dcurrent, unused, dvalue);
    for (int j = 0; j < dim; ++j)
        dprevious[j] = history ? -rho * dcurrent[j] : T(0);
    return terms;
}
} // namespace cudarl::rl
#endif
