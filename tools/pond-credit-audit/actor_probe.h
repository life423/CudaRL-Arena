#pragma once
// Frozen actor inference only. Compare paired inputs at exactly the same gait state.
#include "physics/dynamics.h"
#include "rl/append_hold_checkpoint.h"
namespace p = cudarl::physics;
struct ProbeMoments {
    int n = 0;
    double sum = 0, sumsq = 0;
    void add(double value) {
        ++n;
        sum += value;
        sumsq += value * value;
    }
};
std::vector<float> original_weights(const fs::path &package) {
    const auto bytes = cp::read(package.string());
    check(cp::checksum(bytes) == 4517507952728352494ull, "Wrong motor 336 package");
    cp::Reader in{bytes};
    for (uint32_t expected : {0x31564c48u, 2u, 6u, 6u, 3u, 4u, 41u, 4u, 336u, 1u, 7u, 1u})
        check(in.value<uint32_t>() == expected, "Wrong original actor schema");
    in.value<uint64_t>();
    array<uint8_t>(in, 64);
    const auto hashes = in.value<uint32_t>();
    for (uint32_t j = 0; j < hashes; ++j)
        array<uint8_t>(in, 64);
    return array<float>(in, pp::actor_weights);
}
struct ActorProbe {
    nn::CpuMlp<float> model{{{pp::observations, 32, 32, 4},
                             {nn::Activation::Tanh, nn::Activation::Tanh, nn::Activation::Identity}}};
    ProbeMoments yaw[3][2];
    ActorProbe(const fs::path &package, const fs::path &historical) {
        model.params() = original_weights(package);
        const auto payload = cp::read(historical.string());
        check(cp::checksum(payload) == 2039539270511780368ull, "Wrong historical 336");
        const auto original = cudarl::rl::read_ppo_actor(payload, 14874979637633386686ull, 1247037243u, 2u);
        check(original.update == 336 &&
                  cudarl::rl::append_zero_input_column(original.parameters) == model.params(),
              "Original 40-input weights differ from packaged actor after zero-column migration");
    }
    void observe(const float *source) {
        std::array<float, pp::observations> obs;
        auto vector = [&](int k) { return p::V{source[k], source[k + 1], source[k + 2]}; };
        const auto up = vector(9), heading = vector(6);
        auto left = p::cross(up, heading);
        left = left / p::norm(left);
        for (int experiment = 0; experiment < 3; ++experiment) {
            float outputs[2][2]{};
            for (int sign : {-1, 1}) {
                std::copy_n(source, obs.size(), obs.begin());
                auto set = [&](int k, p::V v) {
                    obs[k] = v.x;
                    obs[k + 1] = v.y;
                    obs[k + 2] = v.z;
                };
                if (experiment == 0) {
                    const auto rotation = p::axis_rotation(up, sign * float(15 * 3.141592653589793 / 180));
                    set(6, rotation.rotate(heading));
                    set(3, rotation.rotate(vector(3)));
                } else {
                    // Add an equal offset to either the lane-error feature (1 mm) or
                    // actual COM velocity (1 mm/s), along pond-left in body coordinates.
                    const int k = experiment == 1 ? 0 : 12;
                    set(k, vector(k) + left * (sign * (experiment == 1 ? .5f : .05f)));
                }
                model.forward(obs.data(), 1);
                for (int j = 0; j < 2; ++j)
                    outputs[sign > 0][j] = std::tanh(model.output()[2 * j]);
            }
            for (int j = 0; j < 2; ++j)
                yaw[experiment][j].add(outputs[1][j] - outputs[0][j]);
        }
    }
    void print() const {
        std::cout
            << ",\n\"motor336_paired_frozen_actor_sensitivity\":{\"raw_336_equals_packaged_actor_with_zero_"
               "appended_column\":true,\"sampling\":\"Actual straight-swim "
               "observations, 108 worlds at controls 41 through 60 of v5 rollout 1. Change one physical "
               "input type per pair. Heading changes rotate both requested heading and requested terminal "
               "velocity consistently. Sign is pond-left minus pond-right. Differences are normalized "
               "deterministic joint-target commands after tanh, prior to filtering and physics. Positive "
               "lane perturbation means correction toward pond-left.\",\"pairs\":"
            << yaw[0][0].n;
        const char *names[] = {"heading_plus_minus_15_degrees", "lane_error_plus_minus_1_mm",
                               "COM_lateral_velocity_plus_minus_1_mm_per_s"};
        for (int i = 0; i < 3; ++i) {
            std::cout << ",\"" << names[i] << "\":{";
            for (int j = 0; j < 2; ++j) {
                auto m = yaw[i][j];
                std::cout << (j ? "," : "") << "\"" << (j ? "rear" : "front")
                          << "_yaw\":{\"mean_delta\":" << m.sum / m.n
                          << ",\"rms_delta\":" << std::sqrt(m.sumsq / m.n) << "}";
            }
            std::cout << "}";
        }
        std::cout << "}";
    }
};
