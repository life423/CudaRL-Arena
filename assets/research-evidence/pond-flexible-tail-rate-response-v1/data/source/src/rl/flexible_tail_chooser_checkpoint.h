#pragma once

#include "rl/flexible_tail_chooser.h"

namespace cudarl::flexible_tail::chooser {

inline nn::MlpSpec actor_spec() {
    return {{inputs, 32, actions}, {nn::Activation::Tanh, nn::Activation::Identity}};
}

inline void validate_weights(const std::vector<float>& weights) {
    if (weights.size() != static_cast<size_t>(actor_spec().num_params()))
        throw std::runtime_error("Invalid chooser actor shape");
    for (float weight : weights) {
        if (!physics::finite_number(weight)) throw std::runtime_error("Nonfinite chooser actor weight");
    }
}

inline void write_weights(const std::string& path, const std::vector<float>& weights) {
    validate_weights(weights);
    checkpoint::Writer out;
    out.value<uint32_t>(0x48435446); // FTCH
    out.value<uint32_t>(1);
    out.value<uint32_t>(inputs);
    out.value<uint32_t>(actions);
    out.array(weights);
    checkpoint::write_atomic(path, out.data);
}

inline std::vector<float> read_weights(const std::string& path) {
    const auto bytes = checkpoint::read(path);
    checkpoint::Reader in{bytes};
    if (in.value<uint32_t>() != 0x48435446 || in.value<uint32_t>() != 1 ||
        in.value<uint32_t>() != inputs || in.value<uint32_t>() != actions)
        throw std::runtime_error("Unsupported chooser actor checkpoint");
    auto weights = in.array<float>(actor_spec().num_params());
    in.finish();
    validate_weights(weights);
    return weights;
}

} // namespace cudarl::flexible_tail::chooser
