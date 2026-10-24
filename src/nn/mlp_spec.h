#ifndef CUDARL_NN_MLP_SPEC_H
#define CUDARL_NN_MLP_SPEC_H

#include <cmath>
#include <cstdint>
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
        if (sizes.size() < 2 || activations.size() != sizes.size() - 1) {
            throw std::invalid_argument("MlpSpec: need >= 2 sizes and one activation per layer");
        }
        for (int s : sizes) {
            if (s <= 0) throw std::invalid_argument("MlpSpec: layer sizes must be positive");
        }
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
        int off = 0;
        for (int i = 0; i < l; ++i) off += out(i) * in(i) + out(i);
        return off;
    }
    int bias_offset(int l) const { return weight_offset(l) + out(l) * in(l); }
    int num_params() const { return weight_offset(num_layers()); }
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

} // namespace cudarl::nn

#endif // CUDARL_NN_MLP_SPEC_H
