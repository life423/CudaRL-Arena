#include "turn-kernels.h"
#include "nn/adam.h"
#include "nn/mlp_gpu.h"
#include "rl/flexible_tail_motor.h"
#include "rl/flexible_tail_transition.h"
#include "rl/gaussian_kernels.h"
#include "rl/gaussian_math.h"
#include "rl/ppo_kernels.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
namespace tail = cudarl::flexible_tail;
namespace kernels = tail::turn_cuda;
namespace nn = cudarl::nn;
namespace rl = cudarl::rl;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr int worlds = 256;
constexpr int controls = 64;
constexpr int rows = worlds * controls;
constexpr int minibatch = 4096;
constexpr int updates = 8;
constexpr int epochs = 2;
constexpr int inputs = tail::motor_inputs;
constexpr int actions = tail::motor_outputs;
constexpr int threads = 256;
constexpr uint32_t seed = 20261124;
constexpr float log_std = -3.2f;
constexpr float gamma = .998f;
constexpr float lambda = .9797f;
constexpr float maximum_kl = .05f;
constexpr rl::PpoCoefficients coefficients{.2f, 0.f, .5f};

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

struct Rollout {
    device_ptr<tail::TurnWorld> world{worlds};
    device_ptr<uint32_t> attempted{worlds}, accepted{worlds};
    device_ptr<int> fault{2};
    device_ptr<float> observations{rows * inputs}, last_observation{worlds * inputs};
    device_ptr<float> means{rows * actions}, latent{rows * actions}, logp{rows};
    device_ptr<float> values{rows}, rewards{rows}, last_values{worlds};
    device_ptr<float> terminal_zeros{rows}, advantages{rows}, returns{rows};
    device_ptr<float> sigma{actions}, advantage_stats{2}, means_stats{2}, kl{rows};
};

struct Batch {
    device_ptr<float> observation{minibatch * inputs}, old_mean{minibatch * actions};
    device_ptr<float> actions_sampled{minibatch * actions}, old_logp{minibatch};
    device_ptr<float> advantage{minibatch}, target_return{minibatch};
    device_ptr<float> mean_gradient{minibatch * actions}, sigma_gradient{minibatch * actions};
    device_ptr<float> value_gradient{minibatch};
    device_ptr<float> row_stats{minibatch * rl::kGaussianStats}, stats{rl::kGaussianStats};
};

struct Accounting {
    int started = 0, completed = 0, actor_steps = 0, critic_steps = 0;
    uint64_t attempted = 0, accepted = 0;
    int fault = INT_MAX;
    const char* stage = "initialize";
    double elapsed_seconds = 0;
};

void zero(void* ptr, size_t bytes, cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaMemsetAsync(ptr, 0, bytes, stream));
}

void copy_device(float* dst, const float* src, int size, cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaMemcpyAsync(dst, src, sizeof(float) * size, cudaMemcpyDeviceToDevice, stream));
}

template <class T, size_t count>
void download(std::array<T, count>& dst, const T* src, cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaMemcpyAsync(dst.data(), src, sizeof(T) * count, cudaMemcpyDeviceToHost, stream));
}

void guard(Rollout& r, Accounting& a, cudaStream_t stream) {
    std::array<int, 1> fault{};
    download(fault, r.fault.get(), stream);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    a.fault = fault[0];
    if (a.fault != INT_MAX) throw std::runtime_error("Device fault before publishing candidate");
}

void count_steps(Rollout& r, Accounting& a, cudaStream_t stream) {
    std::array<uint32_t, worlds> attempted{}, accepted{};
    download(attempted, r.attempted.get(), stream);
    download(accepted, r.accepted.get(), stream);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    a.attempted = std::accumulate(attempted.begin(), attempted.end(), uint64_t{0});
    a.accepted = std::accumulate(accepted.begin(), accepted.end(), uint64_t{0});
}

__global__ void audit_samples(const float* old_mean, const float* sigma, const float* latent,
                              const float* old_logp, int count, int* fault) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    const float probability = rl::gaussian_logp(old_mean + row * actions, sigma,
        latent + row * actions, actions);
    if (!isfinite(probability) || !isfinite(old_logp[row]) ||
        fabsf(probability - old_logp[row]) > .0002f) atomicMin(fault, row);
}

__global__ void exact_kl(const float* old_mean, const float* next_mean, const float* sigma,
                         float* result, int count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    float value = 0;
    for (int axis = 0; axis < actions; ++axis) {
        const int i = row * actions + axis;
        const float error = (old_mean[i] - next_mean[i]) * expf(-sigma[axis]);
        value += .5f * error * error;
    }
    result[row] = value;
}

float finite_mean(Rollout& r, const float* values, int count, cudaStream_t stream) {
    rl::mean_std(values, count, r.means_stats.get(), stream);
    std::array<float, 2> values_stats{};
    download(values_stats, r.means_stats.get(), stream);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    for (const auto value : values_stats)
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite reduction");
    return values_stats[0];
}

float collect(Rollout& r, nn::GpuMlp& actor, nn::GpuMlp& critic, Accounting& a,
              cudaStream_t stream, int update) {
    a.stage = "collection";
    a.started = update;
    std::cout << "collection_start=" << update << std::endl;
    constexpr int no_fault = INT_MAX;
    CUDA_CHECK_THROW(cudaMemcpyAsync(r.fault.get(), &no_fault, sizeof(no_fault), cudaMemcpyHostToDevice, stream));
    for (int t = 0; t < controls; ++t) {
        const int first = t * worlds;
        auto* observation = r.observations.get() + first * inputs;
        kernels::observe(r.world.get(), worlds, observation, r.fault.get(), stream);
        actor.forward(observation, worlds, stream);
        critic.forward(observation, worlds, stream);
        kernels::audit(actor.output(), worlds * actions, r.fault.get(), stream);
        kernels::audit(critic.output(), worlds, r.fault.get(), stream);
        auto* old_mean = r.means.get() + first * actions;
        auto* sample = r.latent.get() + first * actions;
        copy_device(old_mean, actor.output(), worlds * actions, stream);
        copy_device(r.values.get() + first, critic.output(), worlds, stream);
        rl::sample_gaussian(old_mean, r.sigma.get(), worlds, actions, seed,
            static_cast<uint32_t>((update - 1) * controls + t), sample, r.logp.get() + first, stream);
        kernels::transition(r.world.get(), worlds, sample, r.rewards.get() + first,
            r.attempted.get(), r.accepted.get(), r.fault.get(), stream);
    }
    count_steps(r, a, stream);
    guard(r, a, stream);
    const uint64_t expected = uint64_t(update) * rows * 16;
    if (a.attempted != expected || a.accepted != expected) {
        throw std::runtime_error("Unexpected collection substeps");
    }
    a.stage = "rollout_audit";
    kernels::observe(r.world.get(), worlds, r.last_observation.get(), r.fault.get(), stream);
    critic.forward(r.last_observation.get(), worlds, stream);
    copy_device(r.last_values.get(), critic.output(), worlds, stream);
    kernels::audit(r.last_values.get(), worlds, r.fault.get(), stream);
    kernels::audit(r.rewards.get(), rows, r.fault.get(), stream);
    audit_samples<<<rows / threads, threads, 0, stream>>>(r.means.get(), r.sigma.get(),
        r.latent.get(), r.logp.get(), rows, r.fault.get());
    CUDA_CHECK_THROW(cudaGetLastError());
    guard(r, a, stream);
    a.stage = "advantages";
    rl::compute_gae(controls, worlds, r.rewards.get(), r.values.get(), r.terminal_zeros.get(),
        r.terminal_zeros.get(), r.terminal_zeros.get(), r.last_values.get(), gamma, lambda,
        r.advantages.get(), r.returns.get(), stream);
    kernels::audit(r.advantages.get(), rows, r.fault.get(), stream);
    kernels::audit(r.returns.get(), rows, r.fault.get(), stream);
    rl::mean_std(r.advantages.get(), rows, r.advantage_stats.get(), stream);
    kernels::audit(r.advantage_stats.get(), 2, r.fault.get(), stream);
    guard(r, a, stream);
    return finite_mean(r, r.rewards.get(), rows, stream);
}

float optimize(Rollout& r, Batch& b, nn::GpuMlp& actor, nn::GpuMlp& critic,
               nn::GpuAdam& actor_adam, nn::GpuAdam& critic_adam, Accounting& a,
               cudaStream_t stream, int update) {
    a.stage = "optimizer";
    for (int epoch = 0; epoch < epochs; ++epoch) {
        const auto permutation = rl::make_permutation(rows, seed + updates * update + epoch);
        for (int first = 0; first < rows; first += minibatch) {
            rl::gather_rows(permutation, first, minibatch, inputs, r.observations.get(), b.observation.get(), stream);
            rl::gather_rows(permutation, first, minibatch, actions, r.means.get(), b.old_mean.get(), stream);
            rl::gather_rows(permutation, first, minibatch, actions, r.latent.get(), b.actions_sampled.get(), stream);
            rl::gather_rows(permutation, first, minibatch, 1, r.logp.get(), b.old_logp.get(), stream);
            rl::gather_rows(permutation, first, minibatch, 1, r.advantages.get(), b.advantage.get(), stream);
            rl::gather_rows(permutation, first, minibatch, 1, r.returns.get(), b.target_return.get(), stream);

            actor.forward(b.observation.get(), minibatch, stream);
            critic.forward(b.observation.get(), minibatch, stream);
            exact_kl<<<minibatch / threads, threads, 0, stream>>>(b.old_mean.get(), actor.output(),
                r.sigma.get(), r.kl.get(), minibatch);
            CUDA_CHECK_THROW(cudaGetLastError());
            if (finite_mean(r, r.kl.get(), minibatch, stream) > maximum_kl) {
                throw std::runtime_error("Minibatch KL cap exceeded");
            }
            zero(b.stats.get(), b.stats.size_bytes(), stream);
            rl::gaussian_ppo_loss_grad(actor.output(), r.sigma.get(), critic.output(), minibatch,
                actions, b.actions_sampled.get(), b.old_logp.get(), b.advantage.get(),
                b.target_return.get(), r.advantage_stats.get(), coefficients, b.mean_gradient.get(),
                b.sigma_gradient.get(), b.value_gradient.get(), b.row_stats.get(), b.stats.get(), stream);
            kernels::audit(b.stats.get(), rl::kGaussianStats, r.fault.get(), stream);
            actor.zero_grad(stream);
            critic.zero_grad(stream);
            actor.backward(b.mean_gradient.get(), minibatch, nullptr, stream);
            critic.backward(b.value_gradient.get(), minibatch, nullptr, stream);
            kernels::audit(actor.grads(), actor.num_params(), r.fault.get(), stream);
            kernels::audit(critic.grads(), critic.num_params(), r.fault.get(), stream);
            guard(r, a, stream);
            actor_adam.step(actor.params(), actor.grads(), stream);
            ++a.actor_steps;
            critic_adam.step(critic.params(), critic.grads(), stream);
            ++a.critic_steps;
            kernels::audit(actor.params(), actor.num_params(), r.fault.get(), stream);
            kernels::audit(critic.params(), critic.num_params(), r.fault.get(), stream);
            guard(r, a, stream);
        }
    }
    a.stage = "final_kl";
    for (int first = 0; first < rows; first += minibatch) {
        actor.forward(r.observations.get() + first * inputs, minibatch, stream);
        exact_kl<<<minibatch / threads, threads, 0, stream>>>(r.means.get() + first * actions,
            actor.output(), r.sigma.get(), r.kl.get() + first, minibatch);
        CUDA_CHECK_THROW(cudaGetLastError());
    }
    const float final_kl = finite_mean(r, r.kl.get(), rows, stream);
    if (final_kl > maximum_kl) throw std::runtime_error("Final whole-rollout KL cap exceeded");
    return final_kl;
}

void write_result(const fs::path& output, const Accounting& a) {
    std::ofstream result(output / "result.json");
    result << std::setprecision(9) << "{\"planned_updates\":" << updates
        << ",\"updates_started\":" << a.started << ",\"updates_completed\":" << a.completed
        << ",\"actor_adam_calls\":" << a.actor_steps << ",\"critic_adam_calls\":" << a.critic_steps
        << ",\"substeps_attempted\":" << a.attempted << ",\"substeps_accepted\":" << a.accepted
        << ",\"stage\":\"" << a.stage << "\",\"first_fault\":"
        << (a.fault == INT_MAX ? -1 : a.fault) << ",\"wall_s\":" << a.elapsed_seconds << "}\n";
}

void train(const std::string& base, const std::string& initial, const fs::path& output) {
    const auto start = Clock::now();
    if (fs::exists(output) || !fs::create_directory(output))
        throw std::runtime_error("PPO output must be a new directory");
    Accounting a;
    try {
        const auto weights = tail::read_motor_weights(base);
        const auto initial_worlds = tail::read_turn_worlds(initial);
        if (initial_worlds.size() != worlds) throw std::runtime_error("Expected 256 capture worlds");
        size_t existing_stack = 0;
        CUDA_CHECK_THROW(cudaDeviceGetLimit(&existing_stack, cudaLimitStackSize));
        CUDA_CHECK_THROW(cudaDeviceSetLimit(cudaLimitStackSize, (std::max)(size_t{8192}, existing_stack)));
        cudaDeviceProp properties{};
        int device = 0;
        CUDA_CHECK_THROW(cudaGetDevice(&device));
        CUDA_CHECK_THROW(cudaGetDeviceProperties(&properties, device));
        std::cout << "device=" << properties.name << " worlds=" << worlds << " controls_per_update="
            << controls << " max_updates=" << updates << std::endl;

        Stream stream;
        Rollout r;
        Batch b;
        constexpr std::array<float, actions> dispersion{log_std, log_std};
        CUDA_CHECK_THROW(cudaMemcpyAsync(r.world.get(), initial_worlds.data(), r.world.size_bytes(),
            cudaMemcpyHostToDevice, stream.value));
        CUDA_CHECK_THROW(cudaMemcpyAsync(r.sigma.get(), dispersion.data(), r.sigma.size_bytes(),
            cudaMemcpyHostToDevice, stream.value));
        zero(r.terminal_zeros.get(), r.terminal_zeros.size_bytes(), stream.value);
        zero(r.attempted.get(), r.attempted.size_bytes(), stream.value);
        zero(r.accepted.get(), r.accepted.size_bytes(), stream.value);
        auto latent_spec = tail::motor_spec();
        latent_spec.activations.back() = nn::Activation::Identity;
        nn::GpuMlp actor(latent_spec, minibatch, seed);
        nn::GpuMlp critic({{inputs, 32, 32, 1},
            {nn::Activation::Tanh, nn::Activation::Tanh, nn::Activation::Identity}}, minibatch, seed + 1);
        actor.set_params(weights);
        nn::AdamConfig actor_config, critic_config;
        actor_config.lr = 1e-4f;
        critic_config.lr = 3e-4f;
        nn::GpuAdam actor_adam(actor.num_params(), actor_config);
        nn::GpuAdam critic_adam(critic.num_params(), critic_config);
        std::ofstream progress(output / "updates.csv");
        progress.exceptions(std::ios::failbit | std::ios::badbit);
        progress << "update,reward_mean,exact_kl,substeps_attempted,substeps_accepted,actor_adam,"
            "critic_adam,collection_ms,ppo_ms,wall_s\n" << std::flush;
        for (int update = 1; update <= updates; ++update) {
            const auto started = Clock::now();
            const auto reward = collect(r, actor, critic, a, stream.value, update);
            const auto collected = Clock::now();
            const auto kl = optimize(r, b, actor, critic, actor_adam, critic_adam, a, stream.value, update);
            const auto optimized = Clock::now();
            a.stage = "checkpoint";
            tail::write_motor_weights((output / ("actor-" + std::to_string(update) + ".bin")).u8string(),
                actor.get_params());
            a.completed = update;
            a.elapsed_seconds = std::chrono::duration<double>(Clock::now() - start).count();
            progress << std::setprecision(9) << update << ',' << reward << ',' << kl << ','
                << a.attempted << ',' << a.accepted << ',' << a.actor_steps << ',' << a.critic_steps << ','
                << std::chrono::duration<double, std::milli>(collected - started).count() << ','
                << std::chrono::duration<double, std::milli>(optimized - collected).count() << ','
                << a.elapsed_seconds << '\n' << std::flush;
            std::cout << "completed_update=" << update << " reward_mean=" << reward
                << " exact_kl=" << kl << " substeps=" << a.accepted << " wall_s=" << a.elapsed_seconds << std::endl;
        }
        a.stage = "completed";
        write_result(output, a);
    } catch (...) {
        a.elapsed_seconds = std::chrono::duration<double>(Clock::now() - start).count();
        write_result(output, a);
        throw;
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: pond_flexible_tail_motor_turn_ppo BASE_ACTOR TRAIN_WORLDS NEW_OUTPUT\n";
        return 1;
    }
    try {
        train(argv[1], argv[2], fs::u8path(argv[3]));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
