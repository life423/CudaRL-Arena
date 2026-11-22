#ifndef CUDARL_RL_PPO_DIAGNOSTICS_H
#define CUDARL_RL_PPO_DIAGNOSTICS_H
#include "ppo_math.h"
#include <algorithm>
#include <stdexcept>
#include <vector>
namespace cudarl::rl {
struct DiagnosticMoments {
    double mean = 0, variance = 0, minimum = 0, maximum = 0;
    double stddev() const { return std::sqrt(variance); }
};
inline DiagnosticMoments diagnostic_moments(const std::vector<float> &values) {
    if (values.empty())
        throw std::invalid_argument("Empty diagnostic sample");
    DiagnosticMoments out;
    out.minimum = out.maximum = values[0];
    double m2 = 0;
    size_t n = 0;
    for (float x : values) {
        if (!std::isfinite(x))
            throw std::runtime_error("Nonfinite diagnostic data");
        const double delta = x - out.mean;
        out.mean += delta / ++n;
        m2 += delta * (x - out.mean);
        out.minimum = std::min(out.minimum, double(x));
        out.maximum = std::max(out.maximum, double(x));
    }
    out.variance = m2 / n;
    return out;
}
struct DiagnosticValue {
    double half_mse = 0, explained_variance = 0;
    bool variance_defined = false;
};
inline DiagnosticValue diagnostic_value(const std::vector<float> &values, const std::vector<float> &returns) {
    if (values.size() != returns.size())
        throw std::invalid_argument("Diagnostic value shape mismatch");
    std::vector<float> error(values.size());
    DiagnosticValue out;
    for (size_t i = 0; i < values.size(); ++i) {
        error[i] = values[i] - returns[i];
        out.half_mse += .5 * double(error[i]) * error[i];
    }
    out.half_mse /= values.size();
    const auto target = diagnostic_moments(returns), residual = diagnostic_moments(error);
    out.variance_defined = target.variance > 1e-20;
    if (out.variance_defined)
        out.explained_variance = 1 - residual.variance / target.variance;
    return out;
}
struct DiagnosticRatio {
    double k1 = 0, k3 = 0, clipped_fraction = 0;
};
inline DiagnosticRatio diagnostic_ratios(const std::vector<float> &old_logp,
                                         const std::vector<float> &new_logp, double clip) {
    if (old_logp.empty() || old_logp.size() != new_logp.size())
        throw std::invalid_argument("Diagnostic likelihood shape mismatch");
    DiagnosticRatio out;
    for (size_t i = 0; i < old_logp.size(); ++i) {
        const double d = double(new_logp[i]) - old_logp[i];
        out.k1 -= d;
        out.k3 += std::expm1(d) - d;
        out.clipped_fraction += std::fabs(std::expm1(d)) > clip ? 1 : 0;
    }
    const double n = static_cast<double>(old_logp.size());
    out.k1 /= n;
    out.k3 /= n;
    out.clipped_fraction /= n;
    return out;
}
inline std::vector<float> diagnostic_original_order(const std::vector<float> &permuted,
                                                    const Permutation &permutation) {
    if (permuted.size() != permutation.n)
        throw std::invalid_argument("Diagnostic permutation shape mismatch");
    std::vector<float> ordered(permuted.size());
    for (uint32_t i = 0; i < permutation.n; ++i)
        ordered[permute(permutation, i)] = permuted[i];
    return ordered;
}
} // namespace cudarl::rl
#endif
