#ifndef CUDARL_NN_MLP_SPEC_H
#define CUDARL_NN_MLP_SPEC_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/hd.h"
#include "core/rng.h"

namespace cudarl::nn {

enum class Activation : int { Identity = 0, Tanh = 1, ReLU = 2 };

template <typename T>
CUDARL_HD inline T activate(Activation a, T z) {
    switch (a) {
        case Activation::Tanh: return tanh(z);
        case Activation::ReLU: return z > T(0) ? z : T(0);
        default:               return z;
    }
}

// Derivative expressed in terms of the activation's *output* y, so the
// backward pass only needs the stored outputs, not the pre-activations.
template <typename T>
CUDARL_HD inline T activate_grad_from_output(Activation a, T y) {
    switch (a) {
        case Activation::Tanh: return T(1) - y * y;
        case Activation::ReLU: return y > T(0) ? T(1) : T(0);
        default:               return T(1);
    }
}

// Fully connected network: sizes = {in, hidden..., out}, one activation per layer.
// Parameters live in one flat buffer; layer l is W_l [out x in] (row-major)
// followed by b_l [out].
struct MlpSpec {
    std::vector<int>        sizes;
    std::vector<Activation> activations;

    MlpSpec(std::vector<int> layer_sizes, std::vector<Activation> layer_activations)
        : sizes(std::move(layer_sizes)), activations(std::move(layer_activations)) {
        if (sizes.size() < 2 || activations.size() != sizes.size() - 1 ||
            activations.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("MlpSpec: need >= 2 sizes and one activation per layer");
        }
        (void)num_params(); // Validate every layer before any parameter allocation.
    }

    int num_layers() const { return static_cast<int>(activations.size()); }
    int in(int l) const { return sizes[l]; }
    int out(int l) const { return sizes[l + 1]; }
    int input_size() const { return sizes.front(); }
    int output_size() const { return sizes.back(); }

    int max_width() const {
        int w = 0;
        for (int s : sizes) w = s > w ? s : w;
        return w;
    }

    // Offset of W_l in the flat parameter buffer; b_l follows at weight_offset(l) + out*in.
    int weight_offset(int l) const {
        if (l < 0 || l > num_layers()) throw std::out_of_range("MlpSpec: layer offset");
        int off = 0;
        for (int i = 0; i < l; ++i) off = checked_layer_end(off, in(i), out(i));
        return off;
    }
    int bias_offset(int l) const {
        if (l < 0 || l >= num_layers()) throw std::out_of_range("MlpSpec: bias layer");
        return checked_layer_end(weight_offset(l), in(l), out(l)) - out(l);
    }
    int num_params() const { return weight_offset(num_layers()); }

private:
    static int checked_layer_end(int start, int inputs, int outputs) {
        if (inputs <= 0 || outputs <= 0)
            throw std::invalid_argument("MlpSpec: layer sizes must be positive");
        const int64_t end = int64_t(start) + (int64_t(inputs) + 1) * int64_t(outputs);
        if (end > std::numeric_limits<int>::max())
            throw std::invalid_argument("MlpSpec: parameter layout exceeds int index range");
        return static_cast<int>(end);
    }
};

// Weights ~ N(0, 1/fan_in) (LeCun normal), biases zero. Deterministic in seed.
inline std::vector<float> init_params(const MlpSpec& spec, uint32_t seed) {
    std::vector<float> p(static_cast<size_t>(spec.num_params()), 0.f);
    for (int l = 0; l < spec.num_layers(); ++l) {
        const float scale = 1.f / std::sqrt(static_cast<float>(spec.in(l)));
        const int   w0    = spec.weight_offset(l);
        for (int i = 0; i < spec.out(l) * spec.in(l); ++i) {
            p[w0 + i] = scale * rng::normal(seed, static_cast<uint32_t>(l), static_cast<uint32_t>(i));
        }
    }
    return p;
}

// Orthogonal initialization (Saxe et al. 2014), the usual choice for PPO: each
// weight matrix is a random orthonormal matrix scaled by gains[l] (orthonormal
// columns when out >= in, orthonormal rows otherwise); biases zero. Uses
// modified Gram-Schmidt with a second pass for numerical safety.
inline std::vector<float> orthogonal_params(const MlpSpec& spec, uint32_t seed, const std::vector<float>& gains) {
    if (static_cast<int>(gains.size()) != spec.num_layers()) {
        throw std::invalid_argument("orthogonal_params: one gain per layer");
    }
    std::vector<float> p(static_cast<size_t>(spec.num_params()), 0.f);
    for (int l = 0; l < spec.num_layers(); ++l) {
        const int rows = spec.out(l), cols = spec.in(l);
        const bool by_columns = rows >= cols;
        const int count = by_columns ? cols : rows;   // vectors to orthonormalize
        const int dim   = by_columns ? rows : cols;   // their length
        std::vector<std::vector<double>> q(count, std::vector<double>(dim));
        for (int v = 0; v < count; ++v)
            for (int d = 0; d < dim; ++d) q[v][d] = rng::normal(seed, 1000u + l, static_cast<uint32_t>(v * dim + d));
        for (int v = 0; v < count; ++v) {
            for (int pass = 0; pass < 2; ++pass) {
                for (int u = 0; u < v; ++u) {
                    double dot = 0.0;
                    for (int d = 0; d < dim; ++d) dot += q[u][d] * q[v][d];
                    for (int d = 0; d < dim; ++d) q[v][d] -= dot * q[u][d];
                }
            }
            double norm = 0.0;
            for (int d = 0; d < dim; ++d) norm += q[v][d] * q[v][d];
            norm = std::sqrt(norm);
            for (int d = 0; d < dim; ++d) q[v][d] /= norm;
        }
        float* W = &p[spec.weight_offset(l)];
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                W[r * cols + c] = gains[l] * static_cast<float>(by_columns ? q[c][r] : q[r][c]);
            }
    }
    return p;
}

} // namespace cudarl::nn

#endif // CUDARL_NN_MLP_SPEC_H
