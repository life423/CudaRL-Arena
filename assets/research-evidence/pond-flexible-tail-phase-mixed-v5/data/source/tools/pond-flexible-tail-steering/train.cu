#include "rl/flexible_tail_fast_steering.h"
#include "rl/gaussian_kernels.h"
#include "rl/gaussian_math.h"
#include "rl/ppo_kernels.h"
#include "nn/adam.h"
#include "nn/mlp_gpu.h"
#include "habitat_config.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {
namespace tail = cudarl::flexible_tail;
namespace fast = tail::fast;
namespace nn = cudarl::nn;
namespace rl = cudarl::rl;
namespace cp = cudarl::checkpoint;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr int worlds = 1024, decisions = 64, rows = worlds * decisions;
constexpr int batch_size = 4096, epochs = 4, action_controls = 4, max_updates = 128;
constexpr int channels = tail::navigation_inputs, critic_channels = fast::critic_inputs;
constexpr int lane_threads = 8, math_threads = 128;
constexpr uint32_t seed = 20261124;
constexpr float gamma = .995f, lambda = .97f, target_kl = .015f, hard_kl = .05f;
constexpr float actor_lr = .0003f, critic_lr = .001f, log_std = -1.2f;
constexpr double max_session_wall_s = 1200.;
constexpr rl::PpoCoefficients coefficients{.2f, .01f, .5f};
static_assert(rows % batch_size == 0 && worlds % lane_threads == 0);

fast::FeedbackMode host_feedback_mode = fast::FeedbackMode::HalfSecond;
enum class Practice : uint32_t { Original = 0, Mixed15And30 = 1 };
Practice host_practice = Practice::Original;
int host_sampling_update_origin = 0;

const char* practice_name(Practice value) {
    return value == Practice::Mixed15And30 ? "mixed-15-30" : "original";
}

Practice read_practice(const std::string& value) {
    if (value == "original") return Practice::Original;
    if (value == "mixed-15-30") return Practice::Mixed15And30;
    throw std::runtime_error("Practice must be original or mixed-15-30");
}

const char* feedback_name(fast::FeedbackMode mode) {
    if (mode == fast::FeedbackMode::PhaseAware) return "phase-aware";
    if (mode == fast::FeedbackMode::WholeBeat90) return "whole-beat-90";
    return mode == fast::FeedbackMode::WholeBeat ? "whole-beat" : "half-second";
}

fast::FeedbackMode read_feedback(const std::string& value) {
    if (value == "half-second") return fast::FeedbackMode::HalfSecond;
    if (value == "whole-beat") return fast::FeedbackMode::WholeBeat;
    if (value == "whole-beat-90") return fast::FeedbackMode::WholeBeat90;
    if (value == "phase-aware") return fast::FeedbackMode::PhaseAware;
    throw std::runtime_error("Feedback mode must be half-second, whole-beat, whole-beat-90 or phase-aware");
}

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};

struct EventPair {
    cudaEvent_t begin = nullptr, end = nullptr;
    EventPair() {
        CUDA_CHECK_THROW(cudaEventCreate(&begin));
        CUDA_CHECK_THROW(cudaEventCreate(&end));
    }
    ~EventPair() { cudaEventDestroy(end); cudaEventDestroy(begin); }
    void start(cudaStream_t stream) { CUDA_CHECK_THROW(cudaEventRecord(begin, stream)); }
    void stop(cudaStream_t stream) { CUDA_CHECK_THROW(cudaEventRecord(end, stream)); }
    double milliseconds() const {
        float value = 0;
        CUDA_CHECK_THROW(cudaEventSynchronize(end));
        CUDA_CHECK_THROW(cudaEventElapsedTime(&value, begin, end));
        return value;
    }
};

struct Lane {
    fast::World state{};
    uint32_t episode = 0;
    float angle_rad = 0;
    bool needs_start = true;
};
struct Counts {
    uint64_t attempted = 0, accepted = 0, actions = 0;
    uint64_t reliable_course_controls = 0, low_course_controls = 0;
    uint32_t starts = 0, horizons = 0, clearance_stops = 0, posture_stops = 0;
    uint32_t left = 0, right = 0;
    uint32_t measured_beat_periods = 0, rejected_beat_periods = 0;
};
struct Ledger {
    int completed = 0, started = 0, actor_steps = 0, critic_steps = 0;
    int first_fault_lane = INT_MAX, first_fault_code = 0;
    Counts counts{};
    // Invocation work includes repeated verification updates and interrupted
    // collection. It is independent of the resumable logical counters.
    Counts executed_counts{};
    int executed_actor_steps = 0, executed_critic_steps = 0;
    const char* stage = "setup";
    double physics_ms = 0, network_ms = 0, optimization_ms = 0, transfers_ms = 0, evidence_ms = 0;
    double previous_wall_s = 0;
    float policy_loss = 0, value_loss = 0, entropy = 0, clip_fraction = 0, explained_variance = 0;
    int minibatches = 0, epochs_completed = 0;
};
struct Rollout {
    device_ptr<Lane> lanes{worlds};
    device_ptr<Counts> counts{worlds};
    device_ptr<int> first_fault{1}, fault_codes{worlds}, nonfinite{1};
    device_ptr<float> observation{rows * channels}, post_observation{worlds * channels};
    device_ptr<float> critic_observation{rows * critic_channels}, post_critic_observation{worlds * critic_channels};
    device_ptr<float> means{rows}, latent{rows}, logp{rows}, values{rows}, rewards{rows};
    device_ptr<float> done{rows}, truncated{rows}, final_values{rows}, last_values{worlds};
    device_ptr<int> physical_controls{rows};
    device_ptr<float> advantage{rows}, returns{rows}, advantage_stats{2}, summary{2}, kl{rows};
    device_ptr<float> sigma{1}, residual{rows};
    device_ptr<float> frozen_actor_prefix{tail::steering::base_parameters};
    std::array<EventPair, decisions> observation_timers, movement_timers, network_timers, final_network_timers;
};
struct Batch {
    device_ptr<float> observation{batch_size * channels}, actions{batch_size}, old_logp{batch_size};
    device_ptr<float> critic_observation{batch_size * critic_channels};
    device_ptr<float> advantage{batch_size}, returns{batch_size};
    device_ptr<float> mean_gradient{batch_size}, sigma_gradient{batch_size}, value_gradient{batch_size};
    device_ptr<float> row_stats{batch_size * rl::kGaussianStats}, stats{rl::kGaussianStats};
};

struct FeedbackParity {
    double actor_max_difference = 0, critic_max_difference = 0;
    int actor_lane = -1, actor_feature = -1, critic_lane = -1, critic_feature = -1;
    float actor_cpu = 0, actor_cuda = 0, critic_cpu = 0, critic_cuda = 0;
    int nonfinite_values = 0;
    bool pass = false;
};

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
void check_launch() { CUDA_CHECK_THROW(cudaGetLastError()); }
void zero(void* destination, size_t bytes, cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaMemsetAsync(destination, 0, bytes, stream));
}
void copy_device(float* destination, const float* source, int count, cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaMemcpyAsync(destination, source, sizeof(float) * count, cudaMemcpyDeviceToDevice, stream));
}

__device__ void record_fault(int lane, int code, int* first, int* codes) {
    codes[lane] = code;
    atomicMin(first, lane);
}

__global__ void initialize_lanes(Lane* lanes) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane < worlds) lanes[lane].needs_start = true;
}

__global__ void start_needed(Lane* lanes, Counts* counts, const tail::NavigationWorld* starts,
                              int count, fast::FeedbackMode feedback, Practice practice,
                              int* first_fault, int* fault_codes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= worlds || !lanes[index].needs_start) return;
    auto& lane = lanes[index];
    // Every pair has the same physical start and opposite angle. Episode
    // position, not wall time or a host RNG, determines all subsequent draws.
    const auto paired_seed = cudarl::rng::hash_u32(uint32_t(index / 2) ^
        cudarl::rng::hash_u32(lane.episode));
    const int source = int(cudarl::rng::bits(seed, paired_seed, 1) % uint32_t(count));
    constexpr float magnitudes[] = {15.f, 30.f, 45.f, 60.f, 90.f};
    const uint32_t magnitude_count = feedback == fast::FeedbackMode::WholeBeat90 ? 5u : 4u;
    const int angle = int(cudarl::rng::bits(seed, paired_seed, 2) % magnitude_count);
    const float sign = index % 2 == 0 ? 1.f : -1.f;
    // Mixed practice changes a request only at an existing scored episode
    // boundary. It does not interrupt a live turn or reset physical state.
    const float magnitude = practice == Practice::Mixed15And30
        ? (((uint32_t(index / 2) + lane.episode) & 1u) ? 30.f : 15.f)
        : feedback == fast::FeedbackMode::PhaseAware ? 30.f : magnitudes[angle];
    lane.angle_rad = sign * magnitude
        * tail::pi / 180.f;
    fast::initialize(lane.state, starts[source], lane.angle_rad, feedback);
    lane.needs_start = false;
    ++lane.episode;
    ++counts[index].starts;
    if (sign > 0) ++counts[index].left;
    else ++counts[index].right;
    if (fast::is_global_fault(lane.state.fault))
        record_fault(index, lane.state.fault, first_fault, fault_codes);
}

__global__ void observe_lanes(const Lane* lanes, float* observations, float* critic_observations, int* nonfinite) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= worlds) return;
    float values[channels];
    fast::observe(lanes[lane].state, values);
    for (int i = 0; i < channels; ++i) {
        observations[lane * channels + i] = values[i];
        if (!isfinite(values[i])) atomicExch(nonfinite, 1);
    }
    float critic_values[critic_channels];
    fast::observe_critic(lanes[lane].state, critic_values);
    for (int i = 0; i < critic_channels; ++i) {
        critic_observations[lane * critic_channels + i] = critic_values[i];
        if (!isfinite(critic_values[i])) atomicExch(nonfinite, 1);
    }
}

FeedbackParity feedback_parity(const device_ptr<Lane>& lanes, cudaStream_t stream) {
    // Inference on identical saved world bytes; this diagnostic advances no
    // physical body, samples no actions and performs no optimizer updates.
    device_ptr<float> actor_observations(worlds * channels);
    device_ptr<float> critic_observations(worlds * critic_channels);
    device_ptr<int> nonfinite(1);
    zero(nonfinite.get(), nonfinite.size_bytes(), stream);
    observe_lanes<<<worlds / lane_threads, lane_threads, 0, stream>>>(lanes.get(),
        actor_observations.get(), critic_observations.get(), nonfinite.get());
    check_launch();
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    std::vector<Lane> saved(worlds);
    std::vector<float> actor_values(worlds * channels), critic_values(worlds * critic_channels);
    lanes.copy_to_host(saved.data(), saved.size());
    actor_observations.copy_to_host(actor_values.data(), actor_values.size());
    critic_observations.copy_to_host(critic_values.data(), critic_values.size());
    FeedbackParity result;
    int device_nonfinite = 0;
    nonfinite.copy_to_host(&device_nonfinite, 1);
    result.nonfinite_values = device_nonfinite;
    for (int lane = 0; lane < worlds; ++lane) {
        float expected_actor[channels], expected_critic[critic_channels];
        fast::observe(saved[lane].state, expected_actor);
        fast::observe_critic(saved[lane].state, expected_critic);
        for (int i = 0; i < channels; ++i) {
            const float actual = actor_values[lane * channels + i];
            if (!std::isfinite(actual) || !std::isfinite(expected_actor[i])) ++result.nonfinite_values;
            else {
                const double difference = std::abs(double(actual) - expected_actor[i]);
                if (difference > result.actor_max_difference) {
                    result.actor_max_difference = difference;
                    result.actor_lane = lane; result.actor_feature = i;
                    result.actor_cpu = expected_actor[i]; result.actor_cuda = actual;
                }
            }
        }
        for (int i = 0; i < critic_channels; ++i) {
            const float actual = critic_values[lane * critic_channels + i];
            if (!std::isfinite(actual) || !std::isfinite(expected_critic[i])) ++result.nonfinite_values;
            else {
                const double difference = std::abs(double(actual) - expected_critic[i]);
                if (difference > result.critic_max_difference) {
                    result.critic_max_difference = difference;
                    result.critic_lane = lane; result.critic_feature = i;
                    result.critic_cpu = expected_critic[i]; result.critic_cuda = actual;
                }
            }
        }
    }
    result.pass = result.nonfinite_values == 0 && result.actor_max_difference <= 1.e-6 &&
        result.critic_max_difference <= 1.e-6;
    return result;
}

FeedbackParity initial_feedback_parity(const device_ptr<tail::NavigationWorld>& starts,
                                      cudaStream_t stream) {
    device_ptr<Lane> lanes(worlds);
    device_ptr<Counts> counts(worlds);
    device_ptr<int> first(1), codes(worlds);
    zero(lanes.get(), lanes.size_bytes(), stream);
    zero(counts.get(), counts.size_bytes(), stream);
    zero(codes.get(), codes.size_bytes(), stream);
    constexpr int no_fault = INT_MAX;
    CUDA_CHECK_THROW(cudaMemcpyAsync(first.get(), &no_fault, sizeof(no_fault), cudaMemcpyHostToDevice, stream));
    initialize_lanes<<<worlds / lane_threads, lane_threads, 0, stream>>>(lanes.get());
    check_launch();
    start_needed<<<worlds / lane_threads, lane_threads, 0, stream>>>(lanes.get(), counts.get(),
        starts.get(), int(starts.size()), host_feedback_mode, host_practice, first.get(), codes.get());
    check_launch();
    const auto result = feedback_parity(lanes, stream);
    int fault = INT_MAX;
    first.copy_to_host(&fault, 1);
    if (fault != INT_MAX) throw std::runtime_error("Invalid initial archive state in feedback parity");
    return result;
}

__global__ void move_lanes(Lane* lanes, Counts* counts, const float* sampled,
    const float* gait, cudarl::pond_geometry_v1::Parameters pond,
    float* rewards, float* done, float* truncated, int* physical_controls, int* first_fault, int* fault_codes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= worlds) return;
    auto& lane = lanes[index];
    if (fast::is_global_fault(lane.state.fault)) {
        rewards[index] = 0;
        done[index] = 1;
        truncated[index] = 0;
        physical_controls[index] = 0;
        return;
    }
    const auto reliable_before = lane.state.reliable_course_controls;
    const auto low_before = lane.state.low_course_controls;
    const auto measured_before = lane.state.beat_feedback.complete_periods;
    const auto rejected_before = lane.state.beat_feedback.rejected_periods;
    const auto result = fast::advance(lane.state, sampled[index], gait, pond, action_controls);
    rewards[index] = result.reward;
    done[index] = result.done ? 1.f : 0.f;
    truncated[index] = result.truncated ? 1.f : 0.f;
    physical_controls[index] = result.controls;
    auto& count = counts[index];
    count.attempted += result.attempted_substeps;
    count.accepted += result.accepted_substeps;
    count.reliable_course_controls += lane.state.reliable_course_controls - reliable_before;
    count.low_course_controls += lane.state.low_course_controls - low_before;
    count.measured_beat_periods += lane.state.beat_feedback.complete_periods - measured_before;
    count.rejected_beat_periods += lane.state.beat_feedback.rejected_periods - rejected_before;
    ++count.actions;
    if (fast::is_global_fault(result.fault)) {
        record_fault(index, result.fault, first_fault, fault_codes);
    } else if (result.done) {
        lane.needs_start = true;
        if (!result.fault) ++count.horizons;
        else if (result.fault == 2) ++count.clearance_stops;
        else if (result.fault == 5) ++count.posture_stops;
    }
}

__global__ void audit_values(const float* values, int count, int* nonfinite) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && !isfinite(values[i])) atomicExch(nonfinite, 1);
}
void audit(const float* values, int count, Rollout& roll, cudaStream_t stream) {
    audit_values<<<(count + math_threads - 1) / math_threads, math_threads, 0, stream>>>(
        values, count, roll.nonfinite.get());
    check_launch();
}

__global__ void audit_probabilities(const float* means, const float* sigma, const float* samples,
                                     const float* logp, int* nonfinite) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    const float measured = rl::gaussian_logp(means + i, sigma, samples + i, 1);
    if (!isfinite(measured) || !isfinite(logp[i]) || fabsf(measured - logp[i]) > .0002f)
        atomicExch(nonfinite, 1);
}

__global__ void mask_unused_actor_gradients(float* gradient, int count, tail::steering::Kind kind) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && !tail::steering::trainable_parameter(i, kind)) gradient[i] = 0.f;
}
__global__ void check_actor_inactive_weights(const float* weights, int count,
    tail::steering::Kind kind, const float* frozen_base, int* nonfinite) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && !tail::steering::valid_parameter(i, weights[i], kind, frozen_base))
        atomicExch(nonfinite, 1);
}
__global__ void exact_kl_rows(const float* old_mean, const float* next_mean,
                             float sigma, float* result, int count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < count) {
        const float difference = (old_mean[row] - next_mean[row]) * expf(-sigma);
        result[row] = .5f * difference * difference;
    }
}
__global__ void value_residual(const float* target, const float* predicted, float* residual) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) residual[i] = target[i] - predicted[i];
}

__global__ void duration_gae(const float* rewards, const float* values, const float* done,
    const float* truncated, const float* final_values, const float* last_values,
    const int* physical_controls, float* advantages, float* returns) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= worlds) return;
    float advantage = 0;
    for (int t = decisions - 1; t >= 0; --t) {
        const int i = t * worlds + lane;
        const bool terminal = done[i] != 0.f;
        const float next_value = terminal ? (truncated[i] ? final_values[i] : 0.f)
            : (t == decisions - 1 ? last_values[lane] : values[i + worlds]);
        // The750-control horizon ends with two20ms controls rather than
        // inventing two extra physical controls or discounting it as80ms.
        const float action_fraction = float(physical_controls[i]) / float(action_controls);
        const float discount = powf(gamma, action_fraction);
        const float trace_discount = powf(lambda, action_fraction);
        const float delta = rewards[i] + discount * next_value - values[i];
        advantage = delta + (terminal ? 0.f : discount * trace_discount * advantage);
        advantages[i] = advantage;
        returns[i] = advantage + values[i];
    }
}

void guard(Rollout& roll, Ledger& ledger, cudaStream_t stream) {
    const auto start = Clock::now();
    int first = INT_MAX, nonfinite = 0;
    CUDA_CHECK_THROW(cudaMemcpyAsync(&first, roll.first_fault.get(), sizeof(first), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaMemcpyAsync(&nonfinite, roll.nonfinite.get(), sizeof(nonfinite), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    ledger.transfers_ms += elapsed(start) * 1000.;
    if (first != INT_MAX) {
        ledger.first_fault_lane = first;
        CUDA_CHECK_THROW(cudaMemcpy(&ledger.first_fault_code, roll.fault_codes.get() + first,
            sizeof(int), cudaMemcpyDeviceToHost));
        throw std::runtime_error("Native physics or task fault before PPO update");
    }
    if (nonfinite) throw std::runtime_error("Nonfinite values or invalid frozen actor columns");
}

std::array<float, 2> statistics(Rollout& roll, const float* values, int count, cudaStream_t stream) {
    rl::mean_std(values, count, roll.summary.get(), stream);
    std::array<float, 2> result{};
    CUDA_CHECK_THROW(cudaMemcpyAsync(result.data(), roll.summary.get(), sizeof(result), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    for (const float value : result) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite reduction");
    return result;
}

void count_work(Rollout& roll, Ledger& ledger, cudaStream_t stream) {
    const auto start = Clock::now();
    std::vector<Counts> counts(worlds);
    CUDA_CHECK_THROW(cudaMemcpyAsync(counts.data(), roll.counts.get(), roll.counts.size_bytes(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    ledger.counts = {};
    for (const auto& count : counts) {
        ledger.counts.attempted += count.attempted;
        ledger.counts.accepted += count.accepted;
        ledger.counts.actions += count.actions;
        ledger.counts.reliable_course_controls += count.reliable_course_controls;
        ledger.counts.low_course_controls += count.low_course_controls;
        ledger.counts.starts += count.starts;
        ledger.counts.horizons += count.horizons;
        ledger.counts.clearance_stops += count.clearance_stops;
        ledger.counts.posture_stops += count.posture_stops;
        ledger.counts.left += count.left;
        ledger.counts.right += count.right;
        ledger.counts.measured_beat_periods += count.measured_beat_periods;
        ledger.counts.rejected_beat_periods += count.rejected_beat_periods;
    }
    ledger.transfers_ms += elapsed(start) * 1000.;
}

void record_executed_work(Ledger& ledger, const Counts& before) {
    ledger.executed_counts.attempted += ledger.counts.attempted - before.attempted;
    ledger.executed_counts.accepted += ledger.counts.accepted - before.accepted;
    ledger.executed_counts.actions += ledger.counts.actions - before.actions;
}

float collect(Rollout& roll, nn::GpuMlp& actor, nn::GpuMlp& critic,
    const device_ptr<tail::NavigationWorld>& starts, const device_ptr<float>& gait,
    Ledger& ledger, int update, cudaStream_t stream) {
    ledger.stage = "collection";
    ledger.started = update;
    for (int t = 0; t < decisions; ++t) {
        const int first = t * worlds;
        roll.observation_timers[t].start(stream);
        start_needed<<<worlds / lane_threads, lane_threads, 0, stream>>>(roll.lanes.get(), roll.counts.get(),
            starts.get(), int(starts.size()), host_feedback_mode, host_practice,
            roll.first_fault.get(), roll.fault_codes.get());
        check_launch();
        observe_lanes<<<worlds / lane_threads, lane_threads, 0, stream>>>(roll.lanes.get(),
            roll.observation.get() + first * channels, roll.critic_observation.get() + first * critic_channels,
            roll.nonfinite.get());
        check_launch();
        roll.observation_timers[t].stop(stream);
        roll.network_timers[t].start(stream);
        actor.forward(roll.observation.get() + first * channels, worlds, stream);
        critic.forward(roll.critic_observation.get() + first * critic_channels, worlds, stream);
        copy_device(roll.means.get() + first, actor.output(), worlds, stream);
        copy_device(roll.values.get() + first, critic.output(), worlds, stream);
        rl::sample_gaussian(actor.output(), roll.sigma.get(), worlds, 1, seed,
            uint32_t((host_sampling_update_origin + update - 1) * decisions + t),
            roll.latent.get() + first, roll.logp.get() + first, stream);
        roll.network_timers[t].stop(stream);

        // A second pair measures native simulation alone; the first pair's
        // initialization/observation cost remains visible in physics timing.
        roll.movement_timers[t].start(stream);
        move_lanes<<<worlds / lane_threads, lane_threads, 0, stream>>>(roll.lanes.get(), roll.counts.get(),
            roll.latent.get() + first, gait.get(), larger_pond, roll.rewards.get() + first,
            roll.done.get() + first, roll.truncated.get() + first, roll.physical_controls.get() + first,
            roll.first_fault.get(), roll.fault_codes.get());
        check_launch();
        observe_lanes<<<worlds / lane_threads, lane_threads, 0, stream>>>(roll.lanes.get(),
            roll.post_observation.get(), roll.post_critic_observation.get(), roll.nonfinite.get());
        check_launch();
        roll.movement_timers[t].stop(stream);
        // Before any reset, preserve V(final state) for real time-limit
        // truncations. A true terminal has bootstrap zero in compute_gae.
        roll.final_network_timers[t].start(stream);
        critic.forward(roll.post_critic_observation.get(), worlds, stream);
        copy_device(roll.final_values.get() + first, critic.output(), worlds, stream);
        if (t == decisions - 1) copy_device(roll.last_values.get(), critic.output(), worlds, stream);
        roll.final_network_timers[t].stop(stream);
        if ((t + 1) % 8 == 0) guard(roll, ledger, stream);
    }
    for (int t = 0; t < decisions; ++t) {
        ledger.physics_ms += roll.observation_timers[t].milliseconds() + roll.movement_timers[t].milliseconds();
        ledger.network_ms += roll.network_timers[t].milliseconds() + roll.final_network_timers[t].milliseconds();
    }
    audit(roll.means.get(), rows, roll, stream);
    audit(roll.values.get(), rows, roll, stream);
    audit(roll.rewards.get(), rows, roll, stream);
    audit(roll.final_values.get(), rows, roll, stream);
    audit_probabilities<<<rows / math_threads, math_threads, 0, stream>>>(roll.means.get(),
        roll.sigma.get(), roll.latent.get(), roll.logp.get(), roll.nonfinite.get());
    check_launch();
    guard(roll, ledger, stream);
    ledger.stage = "advantages";
    duration_gae<<<worlds / math_threads, math_threads, 0, stream>>>(roll.rewards.get(), roll.values.get(),
        roll.done.get(), roll.truncated.get(), roll.final_values.get(), roll.last_values.get(),
        roll.physical_controls.get(), roll.advantage.get(), roll.returns.get());
    check_launch();
    audit(roll.advantage.get(), rows, roll, stream);
    audit(roll.returns.get(), rows, roll, stream);
    rl::mean_std(roll.advantage.get(), rows, roll.advantage_stats.get(), stream);
    audit(roll.advantage_stats.get(), 2, roll, stream);
    guard(roll, ledger, stream);
    count_work(roll, ledger, stream);
    return statistics(roll, roll.rewards.get(), rows, stream)[0];
}

float rollout_kl(Rollout& roll, nn::GpuMlp& actor, cudaStream_t stream) {
    for (int first = 0; first < rows; first += batch_size) {
        actor.forward(roll.observation.get() + first * channels, batch_size, stream);
        exact_kl_rows<<<batch_size / math_threads, math_threads, 0, stream>>>(
            roll.means.get() + first, actor.output(), log_std, roll.kl.get() + first, batch_size);
        check_launch();
    }
    return statistics(roll, roll.kl.get(), rows, stream)[0];
}

float optimize(Rollout& roll, Batch& batch, nn::GpuMlp& actor, nn::GpuMlp& critic,
    nn::GpuAdam& actor_adam, nn::GpuAdam& critic_adam, Ledger& ledger, int update, cudaStream_t stream) {
    ledger.stage = "optimizer";
    ledger.minibatches = ledger.epochs_completed = 0;
    zero(batch.stats.get(), batch.stats.size_bytes(), stream);
    float final_kl = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        const auto permutation = rl::make_permutation(rows,
            seed + uint32_t((host_sampling_update_origin + update) * epochs + epoch));
        for (int first = 0; first < rows; first += batch_size) {
            rl::gather_rows(permutation, first, batch_size, channels, roll.observation.get(), batch.observation.get(), stream);
            rl::gather_rows(permutation, first, batch_size, critic_channels, roll.critic_observation.get(),
                batch.critic_observation.get(), stream);
            rl::gather_rows(permutation, first, batch_size, 1, roll.latent.get(), batch.actions.get(), stream);
            rl::gather_rows(permutation, first, batch_size, 1, roll.logp.get(), batch.old_logp.get(), stream);
            rl::gather_rows(permutation, first, batch_size, 1, roll.advantage.get(), batch.advantage.get(), stream);
            rl::gather_rows(permutation, first, batch_size, 1, roll.returns.get(), batch.returns.get(), stream);
            actor.forward(batch.observation.get(), batch_size, stream);
            critic.forward(batch.critic_observation.get(), batch_size, stream);
            rl::gaussian_ppo_loss_grad(actor.output(), roll.sigma.get(), critic.output(), batch_size, 1,
                batch.actions.get(), batch.old_logp.get(), batch.advantage.get(), batch.returns.get(),
                roll.advantage_stats.get(), coefficients, batch.mean_gradient.get(), batch.sigma_gradient.get(),
                batch.value_gradient.get(), batch.row_stats.get(), batch.stats.get(), stream);
            actor.zero_grad(stream);
            critic.zero_grad(stream);
            actor.backward(batch.mean_gradient.get(), batch_size, nullptr, stream);
            critic.backward(batch.value_gradient.get(), batch_size, nullptr, stream);
            mask_unused_actor_gradients<<<(actor.num_params() + math_threads - 1) / math_threads,
                math_threads, 0, stream>>>(actor.grads(), actor.num_params(), fast::policy_kind(host_feedback_mode));
            check_launch();
            audit(actor.grads(), actor.num_params(), roll, stream);
            audit(critic.grads(), critic.num_params(), roll, stream);
            guard(roll, ledger, stream);
            actor_adam.step(actor.params(), actor.grads(), stream);
            critic_adam.step(critic.params(), critic.grads(), stream);
            ++ledger.actor_steps;
            ++ledger.critic_steps;
            ++ledger.executed_actor_steps;
            ++ledger.executed_critic_steps;
            ++ledger.minibatches;
            check_actor_inactive_weights<<<(actor.num_params() + math_threads - 1) / math_threads,
                math_threads, 0, stream>>>(actor.params(), actor.num_params(), fast::policy_kind(host_feedback_mode),
                    roll.frozen_actor_prefix.get(), roll.nonfinite.get());
            check_launch();
            audit(critic.params(), critic.num_params(), roll, stream);
        }
        ++ledger.epochs_completed;
        guard(roll, ledger, stream);
        final_kl = rollout_kl(roll, actor, stream);
        if (final_kl > hard_kl) throw std::runtime_error("Hard whole-rollout KL limit exceeded; candidate rejected");
        if (final_kl > target_kl) break;
    }
    std::array<float, rl::kGaussianStats> totals{};
    CUDA_CHECK_THROW(cudaMemcpyAsync(totals.data(), batch.stats.get(), sizeof(totals), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    for (const auto value : totals) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite PPO loss statistics");
    const float normalization = float(ledger.minibatches * batch_size);
    ledger.policy_loss = totals[0] / normalization;
    ledger.value_loss = totals[1] / normalization;
    ledger.entropy = totals[2] / normalization;
    ledger.clip_fraction = totals[4] / normalization;
    for (int first = 0; first < rows; first += batch_size) {
        critic.forward(roll.critic_observation.get() + first * critic_channels, batch_size, stream);
        copy_device(roll.kl.get() + first, critic.output(), batch_size, stream);
    }
    value_residual<<<rows / math_threads, math_threads, 0, stream>>>(roll.returns.get(), roll.kl.get(), roll.residual.get());
    check_launch();
    const auto target = statistics(roll, roll.returns.get(), rows, stream);
    const auto residual = statistics(roll, roll.residual.get(), rows, stream);
    ledger.explained_variance = target[1] > 1.e-8f
        ? 1.f - (residual[1] * residual[1]) / (target[1] * target[1]) : 0.f;
    return final_kl;
}

#include "session.cuh"
} // namespace

int main(int argc, char** argv) {
    try {
        std::string mode = argc > 1 ? argv[1] : "";
        if (mode.rfind("--", 0) == 0) mode = mode.substr(2);
        if ((mode == "fork" || mode == "resume") && argc == 12) {
            run_session(argv[2], argv[3], argv[4], fs::u8path(argv[5]),
                std::stoi(argv[6]), std::stod(argv[7]), mode == "resume", false,
                read_feedback(argv[8]), read_practice(argv[9]), argv[10], argv[11]);
        } else if (mode == "verify-fork" && argc == 11) {
            run_session(argv[2], argv[3], argv[4], fs::u8path(argv[5]), 2,
                std::stod(argv[6]), false, true, read_feedback(argv[7]),
                read_practice(argv[8]), argv[9], argv[10]);
        } else if ((mode == "new" || mode == "resume") && (argc == 8 || argc == 9)) {
            run_session(argv[2], argv[3], argv[4], fs::u8path(argv[5]),
                std::stoi(argv[6]), std::stod(argv[7]), mode == "resume", false,
                argc == 9 ? read_feedback(argv[8]) : fast::FeedbackMode::HalfSecond);
        } else if (mode == "verify-resume" && (argc == 7 || argc == 8)) {
            run_session(argv[2], argv[3], argv[4], fs::u8path(argv[5]), 2,
                std::stod(argv[6]), false, true,
                argc == 8 ? read_feedback(argv[7]) : fast::FeedbackMode::HalfSecond);
        } else throw std::runtime_error("Usage: steering new FITTED OFFSET TRAIN_STARTS OUT MAX_UPDATES WALL_SECONDS | "
            "steering resume FITTED OFFSET TRAIN_STARTS OUT MAX_UPDATES WALL_SECONDS | "
            "steering verify-resume FITTED OFFSET TRAIN_STARTS NEW_OUT WALL_SECONDS "
            "[half-second|whole-beat|whole-beat-90|phase-aware] | "
            "steering fork|resume FITTED OFFSET TRAIN_STARTS OUT NEW_UPDATES WALL_SECONDS "
            "phase-aware mixed-15-30 PARENT_CHECKPOINT PARENT_TASK_ID | "
            "steering verify-fork FITTED OFFSET TRAIN_STARTS NEW_OUT WALL_SECONDS "
            "phase-aware mixed-15-30 PARENT_CHECKPOINT PARENT_TASK_ID");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
