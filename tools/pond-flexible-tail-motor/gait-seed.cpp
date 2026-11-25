#include "rl/flexible_tail_gait.h"
#include "habitat_config.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
namespace tail = cudarl::flexible_tail;
namespace physics = cudarl::physics;
namespace movement = cudarl::movement;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr int features = 3;
constexpr int reference_controls = 300;
constexpr std::array<int, 8> starts{0, 1, 10, 64, 65, 130, 192, 193};

struct Sample {
    std::array<double, features> input;
    double target;
};

std::array<double, features> feedback_basis(const tail::NavigationWorld& world) {
    float input[tail::navigation_inputs];
    tail::observe_gait(world, input);
    std::array<double, features> hidden{};
    for (int i = 0; i < features; ++i) {
        hidden[i] = tanhf(tanhf(tail::gait_seed_hidden_gain * input[tail::motor_inputs + i]));
    }
    return hidden;
}

void collect(const std::vector<tail::NavigationWorld>& worlds, std::vector<Sample>& samples,
             Clock::time_point deadline, int& attempted, int& accepted) {
    for (const int index : starts) {
        auto world = worlds.at(index);
        tail::prepare_navigation(world);
        const auto start = tail::snapshot(world.motion.body);
        if (physics::norm(start.com_velocity_m_s) > 1e-6f) {
            throw std::runtime_error("Gait seed requires stationary reference starts");
        }
        float targets[tail::motor_outputs]{};
        for (int control = 0; control < reference_controls; ++control) {
            if ((control & 15) == 0 && Clock::now() >= deadline) {
                throw std::runtime_error("Gait seed capture deadline");
            }
            if (control % tail::launch_action_controls == 0) {
                // The reference supplies training labels. The learned motor
                // never receives this phase or clock.
                const float seconds = control * tail::motor_control_s;
                targets[0] = .55f * sinf(tail::gait_nominal_rate * seconds + tail::pi / 4.f);
                const auto basis = feedback_basis(world);
                if (control == 0 || control >= 100) {
                    const double latent = std::atanh(targets[0]);
                    samples.push_back({basis, latent});
                    if (control >= 100) {
                        samples.push_back({{-basis[0], -basis[1], basis[2]}, -latent});
                    }
                }
            }
            const auto result = tail::step_launch(world, targets, larger_pond);
            attempted += result.attempted_substeps;
            accepted += result.accepted_substeps;
            if (result.fault) throw std::runtime_error("Gait seed reference left valid dynamics or clear water");
        }
    }
}

std::array<float, features> learn_readout(const std::vector<Sample>& samples) {
    double normal[10][10]{}, target[10]{}, fit[10]{};
    for (const auto& row : samples) {
        for (int i = 0; i < features; ++i) {
            target[i] += row.input[i] * row.target;
            for (int j = 0; j < features; ++j) {
                normal[i][j] += row.input[i] * row.input[j];
            }
        }
    }
    for (int i = 0; i < features; ++i) normal[i][i] += .001;
    if (!movement::solve(normal, target, fit, features)) {
        throw std::runtime_error("Could not fit yaw feedback to reference samples");
    }
    return {static_cast<float>(fit[0]), static_cast<float>(fit[1]), static_cast<float>(fit[2])};
}

std::vector<float> initialize_motor(const std::vector<float>& original,
                                    const std::array<float, features>& coefficients) {
    auto weights = tail::initialize_launch_weights(original);
    const auto spec = tail::navigation_motor_spec();
    // Three initialized hidden paths carry the yaw features. The surrounding
    // units retain the fitted motor's body observations for PPO to recruit.
    for (int layer = 0; layer < 2; ++layer) {
        for (int axis = 0; axis < features; ++axis) {
            const int row = spec.out(layer) - features + axis;
            const int column = spec.in(layer) - features + axis;
            auto first = weights.begin() + spec.weight_offset(layer) + row * spec.in(layer);
            std::fill_n(first, spec.in(layer), 0.f);
            first[column] = layer == 0 ? tail::gait_seed_hidden_gain : 1.f;
            weights[spec.bias_offset(layer) + row] = 0.f;
        }
    }
    const int output = spec.num_layers() - 1;
    auto yaw_weights = weights.begin() + spec.weight_offset(output);
    std::fill_n(yaw_weights, spec.in(output), 0.f);
    std::copy(coefficients.begin(), coefficients.end(), yaw_weights + spec.in(output) - features);
    weights[spec.bias_offset(output)] = 0.f;
    return weights;
}
} // namespace

int main(int argc, char** argv) {
    int attempted = 0, accepted = 0;
    try {
        const auto start = Clock::now();
        if (argc != 4) throw std::runtime_error("Usage: gait-seed OLD_ACTOR TRAINING_WORLDS NEW_DIRECTORY");
        const auto output = fs::u8path(argv[3]);
        if (!fs::create_directory(output)) throw std::runtime_error("Gait seed directory must be new");
        const auto original = tail::read_motor_weights(argv[1]);
        const auto worlds = tail::read_navigation_worlds(argv[2]);
        std::vector<Sample> samples;
        collect(worlds, samples, start + std::chrono::milliseconds(3500), attempted, accepted);
        const auto coefficients = learn_readout(samples);
        const auto weights = initialize_motor(original, coefficients);
        tail::write_navigation_weights((output / "actor.bin").u8string(), tail::NavigationMode::Gait, weights);
        double error = 0.;
        for (const auto& sample : samples) {
            double prediction = 0.;
            for (int i = 0; i < features; ++i) prediction += sample.input[i] * coefficients[i];
            const double difference = std::tanh(prediction) - std::tanh(sample.target);
            error += difference * difference;
        }
        std::ofstream evidence(output / "fit.csv");
        evidence.exceptions(std::ios::failbit | std::ios::badbit);
        evidence << std::setprecision(10) << "samples,attempted_substeps,accepted_substeps,yaw_phase_coefficient,"
            "rate_phase_coefficient,startup_coefficient,target_mse_rad2\n"
            << samples.size() << ',' << attempted << ',' << accepted << ',' << coefficients[0] << ','
            << coefficients[1] << ',' << coefficients[2] << ',' << error / samples.size() << '\n';
        std::cout << "fit_samples=" << samples.size() << " attempted=" << attempted << " accepted=" << accepted
            << " yaw_target_mse=" << error / samples.size() << " elapsed_s="
            << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << " attempted=" << attempted << " accepted=" << accepted << '\n';
        return 2;
    }
}
