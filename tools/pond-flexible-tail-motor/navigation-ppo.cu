#include "navigation-kernels.h"
#include "core/rng.h"
#include "nn/adam.h"
#include "nn/mlp_gpu.h"
#include "rl/gaussian_kernels.h"
#include "rl/gaussian_math.h"
#include "rl/flexible_tail_yaw_offset.h"
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
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
namespace tail = cudarl::flexible_tail;
namespace kernels = tail::navigation_cuda;
namespace nn = cudarl::nn;
namespace rl = cudarl::rl;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr int worlds = 256;
constexpr int controls = 64;
constexpr int rows = worlds * controls;
constexpr int minibatch = 4096;
constexpr int epochs = 2;
constexpr int inputs = tail::navigation_inputs;
constexpr int motor_axes = tail::motor_outputs;
constexpr int threads = 256;
constexpr uint32_t seed = 20261124;
constexpr float lambda = .9797f;
constexpr float maximum_kl = .05f;
constexpr double internal_deadline_s = 3.5;
constexpr rl::PpoCoefficients coefficients{.2f, 0.f, .5f};
static_assert(controls % tail::launch_action_controls == 0);
static_assert(minibatch % tail::launch_action_controls == 0);
static_assert(motor_axes == 2);

struct Schedule {
    int updates;
    int action_controls;
    float actor_lr;
    std::array<float, motor_axes> log_std;
    int action_dimensions = motor_axes;
    bool zero_output_bias = false;

    int decisions() const { return controls / action_controls; }
    int samples() const { return rows / action_controls; }
    int batch_size() const { return minibatch / action_controls; }
};

Schedule make_schedule(tail::NavigationMode mode, bool yaw_only) {
    if (mode == tail::NavigationMode::YawOffset) {
        return {8, tail::launch_action_controls, .003f, {-.8f, 0.f}, 1, true};
    }
    if (mode == tail::NavigationMode::Launch) {
        return {12, tail::launch_action_controls, .00035f, {-1.8f, -2.8f}};
    }
    if (mode == tail::NavigationMode::Gait) {
        return {8, tail::launch_action_controls, .0001f, {-2.6f, -3.8f}, yaw_only ? 1 : motor_axes};
    }
    return {8, 1, .0001f, {-3.2f, -3.2f}};
}

float compound(float discount, int controls_per_action) {
    float result = discount;
    for (int i = 1; i < controls_per_action; ++i) result *= discount;
    return result;
}

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

struct Rollout {
    device_ptr<tail::NavigationWorld> world{worlds};
    device_ptr<uint32_t> attempted{worlds}, accepted{worlds};
    device_ptr<int> fault{2}, physics_faults{worlds};
    device_ptr<float> observations{rows * inputs}, last_observation{worlds * inputs};
    device_ptr<float> means{rows * motor_axes}, latent{rows * motor_axes}, logp{rows};
    device_ptr<float> values{rows}, rewards{rows}, last_values{worlds};
    device_ptr<float> terminal_zeros{rows}, advantages{rows}, returns{rows};
    device_ptr<float> sigma{motor_axes}, advantage_stats{2}, means_stats{2}, kl{rows};
    device_ptr<float> offsets, frozen_observations;
};

struct Batch {
    device_ptr<float> observation{minibatch * inputs}, old_mean{minibatch * motor_axes};
    device_ptr<float> actions_sampled{minibatch * motor_axes}, old_logp{minibatch};
    device_ptr<float> advantage{minibatch}, target_return{minibatch};
    device_ptr<float> mean_gradient{minibatch * motor_axes}, sigma_gradient{minibatch * motor_axes};
    device_ptr<float> value_gradient{minibatch};
    device_ptr<float> row_stats{minibatch * rl::kGaussianStats}, stats{rl::kGaussianStats};
};

struct Accounting {
    int started = 0, completed = 0, actor_steps = 0, critic_steps = 0;
    uint64_t attempted = 0, accepted = 0;
    int fault = INT_MAX, physical_fault = 0;
    bool frozen_motor_verified = false;
    double gpu_rollout_ms = 0;
    const char* stage = "initialize";
    double elapsed_seconds = 0;
};

struct Events {
    cudaEvent_t begin = nullptr, end = nullptr;
    Events() {
        CUDA_CHECK_THROW(cudaEventCreate(&begin));
        const auto status = cudaEventCreate(&end);
        if (status != cudaSuccess) {
            cudaEventDestroy(begin);
            CUDA_CHECK_THROW(status);
        }
    }
    ~Events() { cudaEventDestroy(end); cudaEventDestroy(begin); }
    Events(const Events&) = delete;
    Events& operator=(const Events&) = delete;
};

void check_deadline(Clock::time_point start) {
    if (std::chrono::duration<double>(Clock::now() - start).count() >= internal_deadline_s) {
        throw std::runtime_error("Internal navigation PPO deadline exceeded");
    }
}

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
    if (a.fault == INT_MAX) return;
    if (a.stage == std::string_view{"collection"} && a.fault >= 0 && a.fault < worlds) {
        std::array<int, 1> reason{};
        download(reason, r.physics_faults.get() + a.fault, stream);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        a.physical_fault = reason[0];
    }
    throw std::runtime_error("Device fault before publishing candidate");
}

void count_steps(Rollout& r, Accounting& a, cudaStream_t stream) {
    std::array<uint32_t, worlds> attempted{}, accepted{};
    download(attempted, r.attempted.get(), stream);
    download(accepted, r.accepted.get(), stream);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    a.attempted = std::accumulate(attempted.begin(), attempted.end(), uint64_t{0});
    a.accepted = std::accumulate(accepted.begin(), accepted.end(), uint64_t{0});
}

__global__ void audit_values(const float* values, int count, int* fault) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < count && !isfinite(values[row])) atomicMin(fault, row);
}

void audit(const float* values, int count, int* fault, cudaStream_t stream) {
    audit_values<<<(count + threads - 1) / threads, threads, 0, stream>>>(values, count, fault);
    CUDA_CHECK_THROW(cudaGetLastError());
}

__global__ void sample_yaw(const float* mean, const float* sigma, uint32_t draw,
                           float* latent, float* logp) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= worlds) return;
    // Match the yaw draw to the same world and decision in the two-axis run.
    const auto index = static_cast<uint32_t>(row * motor_axes);
    latent[row] = mean[row] + expf(sigma[0]) * cudarl::rng::normal(seed, draw, index);
    logp[row] = rl::gaussian_logp(mean + row, sigma, latent + row, 1);
}

__global__ void audit_samples(const float* old_mean, const float* sigma, const float* latent,
                              const float* old_logp, int count, int actions, int* fault) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    const float probability = rl::gaussian_logp(old_mean + row * actions, sigma,
        latent + row * actions, actions);
    if (!isfinite(probability) || !isfinite(old_logp[row]) ||
        fabsf(probability - old_logp[row]) > .0002f) atomicMin(fault, row);
}

__global__ void exact_kl(const float* old_mean, const float* next_mean, const float* sigma,
                         float* result, int count, int actions) {
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
              cudaStream_t stream, Events& events, tail::NavigationMode mode,
              const Schedule& schedule, int update, nn::GpuMlp* frozen) {
    const int actions = schedule.action_dimensions;
    a.stage = "collection";
    a.started = update;
    std::cout << "collection_start=" << update << std::endl;
    constexpr int no_fault = INT_MAX;
    CUDA_CHECK_THROW(cudaMemcpyAsync(r.fault.get(), &no_fault, sizeof(no_fault), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_THROW(cudaEventRecord(events.begin, stream));
    for (int t = 0; t < schedule.decisions(); ++t) {
        const int first = t * worlds;
        auto* observation = r.observations.get() + first * inputs;
        kernels::observe(r.world.get(), worlds, mode, observation, r.fault.get(), stream,
            r.offsets.get(), r.frozen_observations.get());
        if (frozen) {
            frozen->forward(r.frozen_observations.get(), worlds, stream);
            audit(frozen->output(), worlds * motor_axes, r.fault.get(), stream);
        }
        actor.forward(observation, worlds, stream);
        critic.forward(observation, worlds, stream);
        audit(actor.output(), worlds * actions, r.fault.get(), stream);
        audit(critic.output(), worlds, r.fault.get(), stream);
        auto* old_mean = r.means.get() + first * actions;
        auto* sample = r.latent.get() + first * actions;
        copy_device(old_mean, actor.output(), worlds * actions, stream);
        copy_device(r.values.get() + first, critic.output(), worlds, stream);
        const auto draw = static_cast<uint32_t>((update - 1) * schedule.decisions() + t);
        if (actions == 1) {
            sample_yaw<<<(worlds + threads - 1) / threads, threads, 0, stream>>>(
                old_mean, r.sigma.get(), draw, sample, r.logp.get() + first);
            CUDA_CHECK_THROW(cudaGetLastError());
        } else {
            rl::sample_gaussian(old_mean, r.sigma.get(), worlds, actions, seed, draw,
                sample, r.logp.get() + first, stream);
        }
        kernels::transition(r.world.get(), worlds, mode, actions, sample, r.rewards.get() + first,
            r.attempted.get(), r.accepted.get(), r.fault.get(), r.physics_faults.get(), stream,
            r.offsets.get(), frozen ? frozen->output() : nullptr);
    }
    CUDA_CHECK_THROW(cudaEventRecord(events.end, stream));
    count_steps(r, a, stream);
    float gpu_rollout_ms = 0;
    CUDA_CHECK_THROW(cudaEventElapsedTime(&gpu_rollout_ms, events.begin, events.end));
    a.gpu_rollout_ms = gpu_rollout_ms;
    guard(r, a, stream);
    const uint64_t expected = uint64_t(update) * rows * tail::turn_substeps_per_control;
    if (a.attempted != expected || a.accepted != expected) {
        throw std::runtime_error("Unexpected collection substeps");
    }
    a.stage = "rollout_audit";
    kernels::observe(r.world.get(), worlds, mode, r.last_observation.get(), r.fault.get(), stream,
        r.offsets.get());
    critic.forward(r.last_observation.get(), worlds, stream);
    copy_device(r.last_values.get(), critic.output(), worlds, stream);
    audit(r.last_values.get(), worlds, r.fault.get(), stream);
    const int samples = schedule.samples();
    audit(r.rewards.get(), samples, r.fault.get(), stream);
    audit_samples<<<samples / threads, threads, 0, stream>>>(r.means.get(), r.sigma.get(),
        r.latent.get(), r.logp.get(), samples, actions, r.fault.get());
    CUDA_CHECK_THROW(cudaGetLastError());
    guard(r, a, stream);
    a.stage = "advantages";
    rl::compute_gae(schedule.decisions(), worlds, r.rewards.get(), r.values.get(), r.terminal_zeros.get(),
        r.terminal_zeros.get(), r.terminal_zeros.get(), r.last_values.get(),
        compound(kernels::control_gamma, schedule.action_controls),
        compound(lambda, schedule.action_controls),
        r.advantages.get(), r.returns.get(), stream);
    audit(r.advantages.get(), samples, r.fault.get(), stream);
    audit(r.returns.get(), samples, r.fault.get(), stream);
    rl::mean_std(r.advantages.get(), samples, r.advantage_stats.get(), stream);
    audit(r.advantage_stats.get(), 2, r.fault.get(), stream);
    guard(r, a, stream);
    return finite_mean(r, r.rewards.get(), samples, stream);
}

float optimize(Rollout& r, Batch& b, nn::GpuMlp& actor, nn::GpuMlp& critic,
               nn::GpuAdam& actor_adam, nn::GpuAdam& critic_adam, Accounting& a,
               cudaStream_t stream, Clock::time_point start, const Schedule& schedule, int update) {
    const int samples = schedule.samples();
    const int batch_size = schedule.batch_size();
    const int actions = schedule.action_dimensions;
    a.stage = "optimizer";
    for (int epoch = 0; epoch < epochs; ++epoch) {
        const auto permutation = rl::make_permutation(samples, seed + schedule.updates * update + epoch);
        for (int first = 0; first < samples; first += batch_size) {
            check_deadline(start);
            rl::gather_rows(permutation, first, batch_size, inputs, r.observations.get(), b.observation.get(), stream);
            rl::gather_rows(permutation, first, batch_size, actions, r.means.get(), b.old_mean.get(), stream);
            rl::gather_rows(permutation, first, batch_size, actions, r.latent.get(), b.actions_sampled.get(), stream);
            rl::gather_rows(permutation, first, batch_size, 1, r.logp.get(), b.old_logp.get(), stream);
            rl::gather_rows(permutation, first, batch_size, 1, r.advantages.get(), b.advantage.get(), stream);
            rl::gather_rows(permutation, first, batch_size, 1, r.returns.get(), b.target_return.get(), stream);

            actor.forward(b.observation.get(), batch_size, stream);
            critic.forward(b.observation.get(), batch_size, stream);
            exact_kl<<<batch_size / threads, threads, 0, stream>>>(b.old_mean.get(), actor.output(),
                r.sigma.get(), r.kl.get(), batch_size, actions);
            CUDA_CHECK_THROW(cudaGetLastError());
            if (finite_mean(r, r.kl.get(), batch_size, stream) > maximum_kl) {
                throw std::runtime_error("Minibatch KL cap exceeded");
            }
            zero(b.stats.get(), b.stats.size_bytes(), stream);
            rl::gaussian_ppo_loss_grad(actor.output(), r.sigma.get(), critic.output(), batch_size,
                actions, b.actions_sampled.get(), b.old_logp.get(), b.advantage.get(),
                b.target_return.get(), r.advantage_stats.get(), coefficients, b.mean_gradient.get(),
                b.sigma_gradient.get(), b.value_gradient.get(), b.row_stats.get(), b.stats.get(), stream);
            audit(b.stats.get(), rl::kGaussianStats, r.fault.get(), stream);
            actor.zero_grad(stream);
            critic.zero_grad(stream);
            actor.backward(b.mean_gradient.get(), batch_size, nullptr, stream);
            critic.backward(b.value_gradient.get(), batch_size, nullptr, stream);
            audit(actor.grads(), actor.num_params(), r.fault.get(), stream);
            audit(critic.grads(), critic.num_params(), r.fault.get(), stream);
            guard(r, a, stream);
            if (schedule.zero_output_bias) zero(actor.grads() + actor.spec().bias_offset(0), sizeof(float), stream);
            actor_adam.step(actor.params(), actor.grads(), stream);
            ++a.actor_steps;
            critic_adam.step(critic.params(), critic.grads(), stream);
            ++a.critic_steps;
            audit(actor.params(), actor.num_params(), r.fault.get(), stream);
            audit(critic.params(), critic.num_params(), r.fault.get(), stream);
            guard(r, a, stream);
        }
    }
    a.stage = "final_kl";
    for (int first = 0; first < samples; first += batch_size) {
        actor.forward(r.observations.get() + first * inputs, batch_size, stream);
        exact_kl<<<batch_size / threads, threads, 0, stream>>>(r.means.get() + first * actions,
            actor.output(), r.sigma.get(), r.kl.get() + first, batch_size, actions);
        CUDA_CHECK_THROW(cudaGetLastError());
    }
    const float final_kl = finite_mean(r, r.kl.get(), samples, stream);
    if (final_kl > maximum_kl) throw std::runtime_error("Final whole-rollout KL cap exceeded");
    return final_kl;
}

const char* mode_name(tail::NavigationMode mode) {
    switch (mode) {
    case tail::NavigationMode::Heading: return "heading";
    case tail::NavigationMode::Destination: return "destination";
    case tail::NavigationMode::Launch: return "launch";
    case tail::NavigationMode::Gait: return "gait";
    case tail::NavigationMode::YawOffset: return "yaw-offset";
    }
    throw std::runtime_error("Invalid navigation mode");
}

tail::NavigationMode parse_mode(std::string_view argument) {
    if (argument == "heading") return tail::NavigationMode::Heading;
    if (argument == "destination") return tail::NavigationMode::Destination;
    if (argument == "launch") return tail::NavigationMode::Launch;
    if (argument == "gait" || argument == "gait-yaw") return tail::NavigationMode::Gait;
    if (argument == "yaw-offset" || argument == "yaw-offset-resume") return tail::NavigationMode::YawOffset;
    throw std::runtime_error("Unknown navigation mode");
}

std::vector<float> initial_actor_weights(tail::NavigationMode mode, const std::string& base,
                                         const char* prior_offset) {
    if (mode == tail::NavigationMode::YawOffset) {
        if (prior_offset) return tail::read_yaw_offset_weights(prior_offset);
        return std::vector<float>(tail::yaw_offset_spec().num_params(), 0.f);
    }
    if (mode == tail::NavigationMode::Gait) return tail::read_navigation_weights(base, mode);
    const auto original = tail::read_motor_weights(base);
    if (mode == tail::NavigationMode::Launch) return tail::initialize_launch_weights(original);
    return tail::expand_navigation_weights(original);
}

// The two-output file layout stores the yaw row, pitch row, then their biases.
// PPO's one-dimensional policy stores the same earlier layers and just yaw.
std::vector<float> extract_yaw_weights(const nn::MlpSpec& full_spec, const std::vector<float>& full) {
    const int last = full_spec.num_layers() - 1;
    const int pitch_begin = full_spec.weight_offset(last) + full_spec.in(last);
    const int bias = full_spec.bias_offset(last);
    if (full[bias + 1] != 0.f ||
        std::any_of(full.begin() + pitch_begin, full.begin() + bias, [](float weight) { return weight != 0.f; })) {
        throw std::runtime_error("Yaw-only PPO requires an exactly neutral fitted pitch row and bias");
    }
    std::vector<float> yaw(full.begin(), full.begin() + pitch_begin);
    yaw.push_back(full[bias]);
    return yaw;
}

std::vector<float> restore_pitch_row(const nn::MlpSpec& full_spec, std::vector<float> yaw) {
    const int last = full_spec.num_layers() - 1;
    const int pitch_begin = full_spec.weight_offset(last) + full_spec.in(last);
    if (yaw.size() != static_cast<size_t>(pitch_begin + 1)) {
        throw std::runtime_error("Unexpected yaw-only motor checkpoint size");
    }
    yaw.insert(yaw.begin() + pitch_begin, full_spec.in(last), 0.f);
    yaw.push_back(0.f);
    return yaw;
}

void write_result(const fs::path& output, const char* run_mode, const Schedule& schedule,
                  const Accounting& a) {
    std::ofstream result(output / "result.json");
    result.exceptions(std::ios::failbit | std::ios::badbit);
    result << std::setprecision(9) << "{\"mode\":\""
        << run_mode << "\",\"internal_deadline_s\":"
        << internal_deadline_s << ",\"planned_updates\":" << schedule.updates
        << ",\"updates_started\":" << a.started << ",\"updates_completed\":" << a.completed
        << ",\"worlds\":" << worlds << ",\"controls_per_world_per_update\":" << controls
        << ",\"controls_per_policy_action\":" << schedule.action_controls
        << ",\"stochastic_action_dimensions\":" << schedule.action_dimensions
        << ",\"policy_samples_per_update\":" << schedule.samples()
        << ",\"policy_samples_in_completed_updates\":" << a.completed * schedule.samples()
        << ",\"minibatch\":" << schedule.batch_size()
        << ",\"actor_adam_calls\":" << a.actor_steps << ",\"critic_adam_calls\":" << a.critic_steps
        << ",\"substeps_attempted\":" << a.attempted << ",\"substeps_accepted\":" << a.accepted
        << ",\"stage\":\"" << a.stage << "\",\"first_fault\":"
        << (a.fault == INT_MAX ? -1 : a.fault) << ",\"physical_fault\":" << a.physical_fault
        << ",\"frozen_motor_verified\":" << (a.frozen_motor_verified ? "true" : "false")
        << ",\"wall_s\":" << a.elapsed_seconds << "}\n";
}

void save_actor(const fs::path& path, tail::NavigationMode mode, const std::vector<float>& weights) {
    if (mode == tail::NavigationMode::YawOffset) {
        tail::write_yaw_offset_weights(path.u8string(), weights);
    } else {
        tail::write_navigation_weights(path.u8string(), mode, weights);
    }
}

void train(tail::NavigationMode mode, bool yaw_only, const std::string& base, const std::string& initial,
           const fs::path& output, const char* prior_offset) {
    const auto start = Clock::now();
    const auto schedule = make_schedule(mode, yaw_only);
    const char* run_mode = prior_offset ? "yaw-offset-resume" : yaw_only ? "gait-yaw" : mode_name(mode);
    if (fs::exists(output) || !fs::create_directory(output))
        throw std::runtime_error("PPO output must be a new directory");
    Accounting a;
    try {
        const auto weights = initial_actor_weights(mode, base, prior_offset);
        const auto full_spec = tail::navigation_motor_spec();
        const bool learn_offset = mode == tail::NavigationMode::YawOffset;
        const auto fitted_weights = learn_offset
            ? tail::read_navigation_weights(base, tail::NavigationMode::Gait) : std::vector<float>{};
        if (learn_offset) (void)extract_yaw_weights(full_spec, fitted_weights);
        const auto training_weights = yaw_only ? extract_yaw_weights(full_spec, weights) : weights;
        if (yaw_only && restore_pitch_row(full_spec, training_weights) != weights) {
            throw std::runtime_error("Yaw-only actor projection changed the fitted motor");
        }
        const auto initial_worlds = tail::read_navigation_worlds(initial);
        if (initial_worlds.size() != worlds) throw std::runtime_error("Expected 256 capture worlds");
        if (learn_offset) {
            for (const auto& world : initial_worlds) {
                if (world.legs[world.leg_count - 1].end_control < schedule.updates * controls) {
                    throw std::runtime_error("Yaw offset worlds must span the entire study");
                }
                for (int leg = 0; leg < world.leg_count; ++leg) {
                    if (world.legs[leg].end_control % schedule.action_controls != 0) {
                        throw std::runtime_error("Yaw offset cue must align with a policy decision");
                    }
                }
            }
        }
        if (mode == tail::NavigationMode::Launch || mode == tail::NavigationMode::Gait) {
            for (const auto& world : initial_worlds) {
                if (world.leg_count != 1 || world.legs[0].choice != tail::Choice::Swim ||
                    world.legs[0].end_control < schedule.updates * controls) {
                    throw std::runtime_error("Launch and gait worlds must use one straight course for the entire pilot");
                }
            }
        }
        size_t existing_stack = 0;
        CUDA_CHECK_THROW(cudaDeviceGetLimit(&existing_stack, cudaLimitStackSize));
        CUDA_CHECK_THROW(cudaDeviceSetLimit(cudaLimitStackSize, (std::max)(size_t{8192}, existing_stack)));
        cudaDeviceProp properties{};
        int device = 0;
        CUDA_CHECK_THROW(cudaGetDevice(&device));
        CUDA_CHECK_THROW(cudaGetDeviceProperties(&properties, device));
        std::cout << "mode=" << run_mode << " device=" << properties.name
            << " worlds=" << worlds << " controls_per_update="
            << controls << " controls_per_action=" << schedule.action_controls
            << " stochastic_action_dimensions=" << schedule.action_dimensions
            << " policy_samples_per_update=" << schedule.samples()
            << " max_updates=" << schedule.updates << std::endl;

        Stream stream;
        Events events;
        Rollout r;
        Batch b;
        CUDA_CHECK_THROW(cudaMemcpyAsync(r.world.get(), initial_worlds.data(), r.world.size_bytes(),
            cudaMemcpyHostToDevice, stream.value));
        CUDA_CHECK_THROW(cudaMemcpyAsync(r.sigma.get(), schedule.log_std.data(), r.sigma.size_bytes(),
            cudaMemcpyHostToDevice, stream.value));
        zero(r.terminal_zeros.get(), r.terminal_zeros.size_bytes(), stream.value);
        zero(r.attempted.get(), r.attempted.size_bytes(), stream.value);
        zero(r.accepted.get(), r.accepted.size_bytes(), stream.value);
        zero(r.physics_faults.get(), r.physics_faults.size_bytes(), stream.value);
        std::unique_ptr<nn::GpuMlp> frozen;
        if (learn_offset) {
            r.offsets.allocate(worlds);
            r.frozen_observations.allocate(worlds * inputs);
            zero(r.offsets.get(), r.offsets.size_bytes(), stream.value);
            auto frozen_spec = full_spec;
            frozen_spec.activations.back() = nn::Activation::Identity;
            frozen = std::make_unique<nn::GpuMlp>(
                nn::GpuMlp::for_inference(frozen_spec, worlds, fitted_weights, stream.value));
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
            if (frozen->get_params() != fitted_weights) throw std::runtime_error("Frozen motor upload differs");
        }
        auto latent_spec = full_spec;
        latent_spec.sizes.back() = schedule.action_dimensions;
        latent_spec.activations.back() = nn::Activation::Identity;
        if (learn_offset) latent_spec = tail::yaw_offset_spec();
        nn::GpuMlp actor(latent_spec, minibatch, seed);
        const nn::MlpSpec critic_spec{{inputs, 32, 32, 1},
            {nn::Activation::Tanh, nn::Activation::Tanh, nn::Activation::Identity}};
        nn::GpuMlp critic(critic_spec, minibatch, seed + 1);
        auto critic_weights = nn::init_params(critic_spec, seed + 1);
        for (int row = 0; row < critic_spec.out(0); ++row) {
            auto first = critic_weights.begin() + row * inputs;
            std::fill(first + tail::motor_inputs, first + inputs, 0.f);
        }
        critic.set_params(critic_weights);
        actor.set_params(training_weights);
        save_actor(output / "actor-0.bin", mode, weights);
        nn::AdamConfig actor_config, critic_config;
        actor_config.lr = schedule.actor_lr;
        critic_config.lr = 3e-4f;
        nn::GpuAdam actor_adam(actor.num_params(), actor_config);
        nn::GpuAdam critic_adam(critic.num_params(), critic_config);
        std::ofstream progress(output / "updates.csv");
        progress.exceptions(std::ios::failbit | std::ios::badbit);
        progress << "update,reward_mean,exact_kl,substeps_attempted,substeps_accepted,actor_adam,"
            "critic_adam,collection_ms,gpu_rollout_ms,ppo_ms,checkpoint_ms,wall_s\n" << std::flush;
        for (int update = 1; update <= schedule.updates; ++update) {
            check_deadline(start);
            const auto started = Clock::now();
            const auto reward = collect(r, actor, critic, a, stream.value, events, mode, schedule, update,
                frozen.get());
            const auto collected = Clock::now();
            const auto kl = optimize(r, b, actor, critic, actor_adam, critic_adam, a, stream.value, start,
                schedule, update);
            const auto optimized = Clock::now();
            a.stage = "checkpoint";
            check_deadline(start);
            auto checkpoint = actor.get_params();
            if (yaw_only) checkpoint = restore_pitch_row(full_spec, std::move(checkpoint));
            save_actor(output / ("actor-" + std::to_string(update) + ".bin"), mode, checkpoint);
            const auto saved = Clock::now();
            a.completed = update;
            a.elapsed_seconds = std::chrono::duration<double>(saved - start).count();
            progress << std::setprecision(9) << update << ',' << reward << ',' << kl << ','
                << a.attempted << ',' << a.accepted << ',' << a.actor_steps << ',' << a.critic_steps << ','
                << std::chrono::duration<double, std::milli>(collected - started).count() << ','
                << a.gpu_rollout_ms << ','
                << std::chrono::duration<double, std::milli>(optimized - collected).count() << ','
                << std::chrono::duration<double, std::milli>(saved - optimized).count() << ','
                << a.elapsed_seconds << '\n' << std::flush;
            std::cout << "completed_update=" << update << " reward_mean=" << reward
                << " exact_kl=" << kl << " substeps=" << a.accepted << " wall_s=" << a.elapsed_seconds << std::endl;
        }
        if (frozen) {
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
            if (frozen->get_params() != fitted_weights ||
                tail::read_navigation_weights(base, tail::NavigationMode::Gait) != fitted_weights) {
                throw std::runtime_error("Frozen motor weights changed");
            }
            a.frozen_motor_verified = true;
        }
        a.stage = "completed";
        write_result(output, run_mode, schedule, a);
    } catch (...) {
        a.elapsed_seconds = std::chrono::duration<double>(Clock::now() - start).count();
        write_result(output, run_mode, schedule, a);
        throw;
    }
}
} // namespace

int main(int argc, char** argv) {
    const std::string_view argument = argc > 1 ? argv[1] : "";
    const bool warm_start = argument == "yaw-offset-resume";
    if (argc != (warm_start ? 6 : 5)) {
        std::cerr << "Usage: pond_flexible_tail_motor_navigation_ppo "
            "heading|destination|launch|gait|gait-yaw|yaw-offset BASE_ACTOR TRAIN_WORLDS NEW_OUTPUT\n"
            "       pond_flexible_tail_motor_navigation_ppo "
            "yaw-offset-resume FITTED_GAIT PRIOR_OFFSET TRAIN_WORLDS NEW_OUTPUT\n";
        return 1;
    }
    try {
        const int world_argument = warm_start ? 4 : 3;
        train(parse_mode(argument), argument == "gait-yaw", argv[2], argv[world_argument],
            fs::u8path(argv[world_argument + 1]), warm_start ? argv[3] : nullptr);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
