#if defined(CUDARL_RATE_CHOOSER)
#include "rl/flexible_tail_rate_chooser.h"
#else
#include "rl/flexible_tail_chooser_checkpoint.h"
#endif
#include "bank_training_starts.h"
#if defined(CUDARL_REGION_CREDIT)
#include "region_credit.h"
#endif
#include "core/rng.h"
#include "habitat_config.h"
#include "nn/adam.h"
#include "nn/mlp_cpu.h"
#include "nn/mlp_gpu.h"
#include "rl/ppo_kernels.h"
#if defined(CUDARL_FRONTIER_REWARD)
#include "rl/flexible_tail_frontier_reward.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32) && defined(CUDARL_SUSTAINED_CHOOSER)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {
#if defined(CUDARL_RATE_CHOOSER)
namespace chooser = cudarl::flexible_tail::rate_chooser;
#else
namespace chooser = cudarl::flexible_tail::chooser;
#endif
namespace bank_training = cudarl::flexible_tail::bank_training;
namespace tail = cudarl::flexible_tail;
namespace geometry = cudarl::pond_geometry_v1;
namespace nn = cudarl::nn;
namespace rl = cudarl::rl;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

#if defined(CUDARL_SUSTAINED_CHOOSER)
constexpr int worlds = 1024, decisions = 8, rows = worlds * decisions;
constexpr int batch_size = 1024, epochs = 4;
#if defined(CUDARL_RELIABLE_CHOOSER)
constexpr int max_updates = 128;
#else
constexpr int max_updates = 64;
#endif
#if defined(CUDARL_RATE_CHOOSER)
constexpr int macro_controls = 40;
#else
constexpr int macro_controls = 300;
#endif
#else
constexpr int worlds = 96, decisions = 4, rows = worlds * decisions;
constexpr int batch_size = 128, epochs = 4, max_updates = 16, macro_controls = 300;
#endif
constexpr int channels = chooser::inputs, actions = chooser::actions;
constexpr int lane_threads = 8, math_threads = 64;
constexpr uint64_t max_substeps = uint64_t(worlds) * 2 * max_updates * decisions *
    macro_controls * tail::turn_substeps_per_control;
constexpr uint32_t seed = 20261124;
constexpr double deadline_s = 17.5;
#if defined(CUDARL_RATE_CHOOSER)
constexpr float gamma = .994f, lambda = .98f, max_kl = .05f;
#else
constexpr float gamma = .97f, lambda = .90f, max_kl = .05f;
#endif
constexpr rl::PpoCoefficients loss{.20f, .01f, .50f};
static_assert(rows % batch_size == 0);
static_assert(actions == 3 || actions == 9 || actions == 15);

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

struct Events {
    cudaEvent_t start = nullptr, end = nullptr;
    cudaEvent_t after_start = nullptr, after_network = nullptr;
    cudaEvent_t after_bootstrap = nullptr;
    Events() {
        CUDA_CHECK_THROW(cudaEventCreate(&start));
        const auto status = cudaEventCreate(&end);
        if (status != cudaSuccess) {
            cudaEventDestroy(start);
            CUDA_CHECK_THROW(status);
        }
        CUDA_CHECK_THROW(cudaEventCreate(&after_start));
        CUDA_CHECK_THROW(cudaEventCreate(&after_network));
        CUDA_CHECK_THROW(cudaEventCreate(&after_bootstrap));
    }
    ~Events() { cudaEventDestroy(after_bootstrap); cudaEventDestroy(after_network); cudaEventDestroy(after_start); cudaEventDestroy(end); cudaEventDestroy(start); }
    Events(const Events&) = delete;
    Events& operator=(const Events&) = delete;
};

struct Lane {
    chooser::World state;
    uint32_t episode = 0;
    bool needs_start = true;
#if defined(CUDARL_RELIABLE_CHOOSER)
    int curriculum_level = 0, curriculum_failures = 0;
    int curriculum_mark_control = 0, curriculum_mark_visits = 0;
    float curriculum_mark_transport = 0;
#endif
};

#if defined(CUDARL_RATE_CHOOSER)
__device__ tail::steering::Kind session_motor_kind;
__device__ tail::steering::Config session_motor_config;
#elif defined(CUDARL_SUSTAINED_CHOOSER)
// Configured once before collection. The value is embedded in every session
// checkpoint and must match before a restored world can advance.
__device__ chooser::SteeringMode session_steering_mode;
chooser::SteeringMode host_steering_mode = chooser::SteeringMode::MeasuredCourse;
__device__ chooser::MotorFeedbackMode session_motor_feedback_mode;
__device__ chooser::HorizonMode session_horizon_mode;
chooser::MotorFeedbackMode host_motor_feedback_mode = chooser::MotorFeedbackMode::HalfSecond;
chooser::HorizonMode host_horizon_mode = chooser::HorizonMode::Original;
#endif

struct Counts {
    uint64_t attempted = 0, accepted = 0, warmup_attempted = 0, warmup_accepted = 0;
    uint32_t samples[actions]{}, stops = 0, cue_timeouts = 0, posture_stops = 0, starts = 0;
    uint32_t bank_left_starts = 0, bank_right_starts = 0;
#if defined(CUDARL_RELIABLE_CHOOSER)
    uint32_t curriculum_starts[3]{}, promotions = 0, demotions = 0;
#endif
};

struct Ledger {
    Counts counts;
#if defined(CUDARL_THREE_DEPTHS)
    std::array<uint64_t,3> samples_at_depth{};
#if defined(CUDARL_MOVING_STARTS)
    std::array<std::array<uint64_t,5>,3> samples_by_practice{};
#endif
#endif
    bool bank_curriculum = false;
    int started = 0, completed = 0, actor_steps = 0, critic_steps = 0;
    int first_fault_lane = -1, first_fault_code = 0;
    double wall_s = 0, gpu_dynamics_ms = 0, max_wave_s = 0;
    float preflight_max_difference_rad = 0;
    int preflight_checks = 0;
    const char* stage = "setup";
    double gpu_physics_ms = 0, gpu_inference_ms = 0, gpu_optimization_ms = 0;
    double transfers_ms = 0, evidence_ms = 0;
    float policy_loss = 0, value_loss = 0, entropy = 0, clip_fraction = 0, explained_variance = 0;
    int minibatches = 0;
    // Physical work from repeated resume-verification updates remains distinct
    // from the resumable logical counters.
    uint64_t executed_attempted = 0, executed_accepted = 0, executed_decisions = 0;
    int executed_actor_steps = 0, executed_critic_steps = 0;
    uint64_t execution_physical_limit = max_substeps;
    uint64_t time_limits=0,near_decisions=0,episode_decisions=0;
};

#if defined(CUDARL_EPISODIC_BANK)
#if defined(CUDARL_OPEN_WATER)
constexpr int training_episode_controls=1200; // Serialized base interval shared with the parent.
enum class Practice { Moving, Tangent, Interior };
CUDARL_HD Practice practice_for(int lane) {
    const int pair_group = (lane / 2) % 4;
    return pair_group < 2 ? Practice::Moving : pair_group == 2 ? Practice::Tangent : Practice::Interior;
}
CUDARL_HD int practice_limit(int lane) {
    const auto group = practice_for(lane);
    return group == Practice::Moving ? 2*training_episode_controls
        : group == Practice::Tangent ? training_episode_controls : 0;
}
#elif defined(CUDARL_NORTH_BANK)
constexpr int training_episode_controls=1200; // 24 s near the bank; other lanes remain continuous.
CUDARL_HD bool north_practice_lane(int lane){return (lane/2)%2==0;}
#else
constexpr int training_episode_controls=3000; // 60 seconds, training only.
#endif
struct EpisodeInfo {int first_decision_control=0;uint64_t time_limits=0,near_decisions=0,decisions=0;};
#endif
struct Rollout {
#if defined(CUDARL_REGION_CREDIT)
    device_ptr<int> region_hits{rows};
#endif
    device_ptr<Lane> lanes{worlds};
    device_ptr<Counts> counts{worlds};
    device_ptr<int> first_fault{1}, fault_code{worlds};
    device_ptr<float> obs{rows * channels}, final_obs{worlds * channels};
    device_ptr<int> action{rows};
    device_ptr<float> logits{rows * actions}, old_logp{rows}, value{rows}, reward{rows}, done{rows};
    device_ptr<float> zero{rows}, final_value{worlds}, advantage{rows}, target_return{rows};
    device_ptr<float> summary{2}, advantage_stats{2}, kl{rows};
    device_ptr<float> residual{rows};
#if defined(CUDARL_EPISODIC_BANK)
    device_ptr<EpisodeInfo> episodes{worlds};
    device_ptr<float> truncated{rows},terminal_value{rows};
#endif
};

struct Batch {
    device_ptr<float> obs{batch_size * channels}, old_logp{batch_size};
    device_ptr<int> action{batch_size};
    device_ptr<float> advantage{batch_size}, target_return{batch_size};
    device_ptr<float> logit_grad{batch_size * actions}, value_grad{batch_size};
    device_ptr<float> row_stats{batch_size * rl::kPpoStats}, stats{rl::kPpoStats};
};

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void check_launch() { CUDA_CHECK_THROW(cudaGetLastError()); }

void zero(void* pointer, size_t bytes, cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaMemsetAsync(pointer, 0, bytes, stream));
}

void copy_device(float* out, const float* in, int count, cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaMemcpyAsync(out, in, sizeof(float) * count, cudaMemcpyDeviceToDevice, stream));
}

template<class T, size_t N>
void download(std::array<T, N>& out, const T* in, cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaMemcpyAsync(out.data(), in, sizeof(T) * N, cudaMemcpyDeviceToHost, stream));
}

__device__ void fail(int lane, int reason, int* first_fault, int* fault_code) {
    fault_code[lane] = reason;
    atomicMin(first_fault, lane);
}

__global__ void initialize_lanes(Lane* lanes) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane < worlds) lanes[lane].needs_start = true;
}

CUDARL_HD uint32_t start_episode_seed(uint32_t lane, uint32_t episode) {
    return cudarl::rng::hash_u32(lane ^ cudarl::rng::hash_u32(episode));
}

CUDARL_HD void place_interior(uint32_t episode_seed, geometry::Parameters pond, cudarl::physics::V& com,float& yaw) {
    const float radius=.40f*sqrtf(cudarl::rng::uniform(seed,episode_seed,0));
    const float angle=2.f*tail::pi*cudarl::rng::uniform(seed,episode_seed,1);
    yaw=2.f*tail::pi*cudarl::rng::uniform(seed,episode_seed,2);
    com={float(pond.water_radius_x_m)*radius*cosf(angle),0.f,float(pond.water_radius_z_m)*radius*sinf(angle)};
}

CUDARL_HD uint32_t start_source_row(uint32_t episode_seed, int count) {
#if defined(CUDARL_OPEN_WATER)
    return 12u + cudarl::rng::bits(seed,episode_seed,3) % uint32_t(count-12);
#elif defined(CUDARL_CLOSE_SHORE) || defined(CUDARL_NORTH_BANK)
    const auto row=cudarl::rng::bits(seed,episode_seed,3)%uint32_t(count-2);
    return row>=8?row+2:row;
#else
    return cudarl::rng::bits(seed, episode_seed, 3) % uint32_t(count);
#endif
}
#if defined(CUDARL_CLOSE_SHORE)
struct ClosePlacement {cudarl::physics::V position{};float yaw=0;int valid=0;};
__global__ void close_placement_probe(const tail::NavigationWorld* starts,int count,int episode,geometry::Parameters pond,ClosePlacement* output){
    const int lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=worlds)return;
    const auto row=start_source_row(start_episode_seed(lane,episode),count);auto& out=output[lane];
    out.valid=bank_training::close_start(lane,episode,starts[row],pond,out.position,out.yaw);
}
#endif

#if defined(CUDARL_THREE_DEPTHS)
static_assert(tail::distant_map::depth_count==3);
CUDARL_HD int practice_depth_index(int lane) {
    return (lane/2)%tail::distant_map::depth_count;
}
CUDARL_HD uint32_t depth_practice_row(int lane, uint32_t draw, int total) {
    const int poses=total/tail::distant_map::depth_count;
    return practice_depth_index(lane)*poses+start_source_row(draw,poses);
}
#endif

__global__ void start_needed_lanes(Lane* lanes, Counts* counts, const tail::NavigationWorld* starts, int count,
                                   const float* gait, const float* offset, geometry::Parameters pond,
                                   bool bank_curriculum, int* first_fault, int* fault_code
#if defined(CUDARL_EPISODIC_BANK)
                                   ,EpisodeInfo* episodes
#endif
                                   ) {
    const int lane_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane_index >= worlds) return;
    auto& lane = lanes[lane_index];
    if (!lane.needs_start) return;

    const auto episode_seed = start_episode_seed(uint32_t(lane_index), lane.episode);
#if defined(CUDARL_THREE_DEPTHS)
    const auto row = depth_practice_row(lane_index, episode_seed, count);
#else
    const auto row = start_source_row(episode_seed, count);
#endif
    cudarl::physics::V com;
    float yaw;
    bank_training::ApproachEvidence approach{};
    bool bank_start = bank_curriculum;
#if defined(CUDARL_SUSTAINED_CHOOSER)
    bank_start = ((uint32_t(lane_index / 2) + lane.episode) % 2) == 0;
#endif
#if defined(CUDARL_OPEN_WATER)
    const auto practice=practice_for(lane_index);
    bank_start=practice!=Practice::Interior;
    if (practice==Practice::Moving) {
        if (!bank_training::moving_start(lane_index,lane.episode,starts[row],pond,com,yaw,approach)) {
            fail(lane_index,41,first_fault,fault_code);return;
        }
    } else if (practice==Practice::Tangent) {
        approach.inward_side=lane_index%2 ? -1:1;
        const bool north=(lane_index/8)%2==0;
        if (!bank_training::close_start(lane_index,lane.episode,starts[row],pond,com,yaw,north)) {
            fail(lane_index,42,first_fault,fault_code);return;
        }
    } else
#elif defined(CUDARL_NORTH_BANK)
    if(north_practice_lane(lane_index)){
        bank_start=true;
        approach.inward_side=lane_index%2 ? -1:1;
        if(!bank_training::close_start(lane_index,lane.episode,starts[row],pond,com,yaw,true)){
            fail(lane_index,42,first_fault,fault_code);return;
        }
    }else
#endif
#if defined(CUDARL_CLOSE_SHORE)
    const auto start_group=(uint32_t(lane_index/2)+lane.episode)%4;
    bank_start=start_group==2;
    if(start_group<2){
        if(!bank_training::close_start(uint32_t(lane_index),lane.episode,starts[row],pond,com,yaw)){
            fail(lane_index,42,first_fault,fault_code);return;}
    }else
#endif
#if !defined(CUDARL_OPEN_WATER)
    if (bank_start) {
        int difficulty = 2;
#if defined(CUDARL_RELIABLE_CHOOSER)
        difficulty = lane.curriculum_level;
#endif
        if (!bank_training::approach_start(uint32_t(lane_index), lane.episode, starts[row], pond, com, yaw, &approach, difficulty)) {
            fail(lane_index, 41, first_fault, fault_code);
            return;
        }
    } else
#endif
    {
        place_interior(episode_seed,pond,com,yaw);
    }
#if defined(CUDARL_RATE_CHOOSER)
    chooser::initialize(lane.state, starts[row], com, yaw, session_motor_kind, session_motor_config);
#elif defined(CUDARL_SUSTAINED_CHOOSER)
    chooser::initialize(lane.state, starts[row], com, yaw, session_steering_mode,
        session_motor_feedback_mode, session_horizon_mode);
#else
    chooser::initialize(lane.state, starts[row], com, yaw);
#endif
    const auto result = chooser::advance(lane.state, gait, offset, pond, macro_controls);
    auto& counter = counts[lane_index];
    counter.warmup_attempted += result.attempted_substeps;
    counter.warmup_accepted += result.accepted_substeps;
    if (result.fault || !chooser::ready(lane.state)) {
        fail(lane_index, 20 + result.fault, first_fault, fault_code);
        return;
    }
    ++counter.starts;
#if defined(CUDARL_EPISODIC_BANK)
    episodes[lane_index].first_decision_control=lane.state.motor.navigation.motion.controls;
#endif
#if defined(CUDARL_RELIABLE_CHOOSER)
    ++counter.curriculum_starts[lane.curriculum_level];
    lane.curriculum_mark_control = lane.state.motor.navigation.motion.controls;
    lane.curriculum_mark_visits = lane.state.visits;
    lane.curriculum_mark_transport = lane.state.transport_m;
#endif
    if (bank_start) {
        if (approach.inward_side > 0) ++counter.bank_left_starts;
        else ++counter.bank_right_starts;
    }
    ++lane.episode;
    lane.needs_start = false;
}

#if defined(CUDARL_RATE_CHOOSER)
constexpr int collector_distant_default=chooser::default_observation_mode;
#else
constexpr int collector_distant_default=0;
#endif
__global__ void observe_lanes(const Lane* lanes, geometry::Parameters pond, float* rows,
                              int* first_fault, int* fault_code,int observation_mode=collector_distant_default) {
    const int lane_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane_index >= worlds) return;
    float obs[channels]{};
    const auto& lane = lanes[lane_index];
    if (!lane.needs_start) {
        if (!chooser::ready(lane.state)) fail(lane_index, 30, first_fault, fault_code);
        else {
#if defined(CUDARL_RATE_CHOOSER)
            chooser::observe(lane.state, pond, obs,observation_mode);
#else
            chooser::observe(lane.state, pond, obs);
#endif
        }
    }
    for (int i = 0; i < channels; ++i) {
        if (!isfinite(obs[i])) {
            fail(lane_index, 31, first_fault, fault_code);
            obs[i] = 0.f;
        }
        rows[lane_index * channels + i] = obs[i];
    }
}

__global__ void move_lanes(Lane* lanes, Counts* counts, const int* action, const float* old_logp,
                           const float* value, const float* gait, const float* offset,
                           geometry::Parameters pond, float* reward, float* done,
                           int* first_fault, int* fault_code
#if defined(CUDARL_REGION_CREDIT)
                           ,int* region_hits
#endif
#if defined(CUDARL_EPISODIC_BANK)
                           ,EpisodeInfo* episodes,float* truncated
#endif
                           ) {
    const int lane_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane_index >= worlds) return;
#if defined(CUDARL_REGION_CREDIT)
    region_hits[lane_index] = 0;
    const int previous_visits = lanes[lane_index].state.visits;
#endif
    reward[lane_index] = 0.f;
    done[lane_index] = 1.f;
#if defined(CUDARL_EPISODIC_BANK)
    truncated[lane_index]=0;
#endif
    if (*first_fault != INT_MAX) return;
    auto& lane = lanes[lane_index];
#if defined(CUDARL_EPISODIC_BANK)
    auto& episode=episodes[lane_index];++episode.decisions;
    const auto initial_gap=tail::submerged_gap_m(pond,lane.state.motor.navigation.motion.body,lane.state.pose);
    if(initial_gap>0&&initial_gap<=.020)++episode.near_decisions;
#endif
    const int proposed = action[lane_index];
#if defined(CUDARL_THREE_DEPTHS)
    const float request_y = lane.state.motor.navigation.motion.request.origin_m.y;
    const float depth_y = tail::distant_map::depth_at(practice_depth_index(lane_index));
    if (!isfinite(request_y) || fabsf(request_y-depth_y)>1.e-5f) {
        fail(lane_index,43,first_fault,fault_code);
        return;
    }
#endif
    if (proposed < 0 || proposed >= actions || !isfinite(old_logp[lane_index]) ||
        !isfinite(value[lane_index]) || !chooser::accept(lane.state, proposed)) {
        fail(lane_index, 32, first_fault, fault_code);
        return;
    }

#if defined(CUDARL_FRONTIER_REWARD)
    const float potential_before=tail::frontier::potential(lane.state,lane.state.pose.com_m,
        lane.state.motor.navigation.motion.request.origin_m.y);
#endif
    const auto result = chooser::advance(lane.state, gait, offset, pond, macro_controls);
    auto& counter = counts[lane_index];
    counter.attempted += result.attempted_substeps;
    counter.accepted += result.accepted_substeps;
    ++counter.samples[proposed];
    if (!isfinite(result.reward) || (result.controls <= 0 && result.fault == 0)) {
        fail(lane_index, 33, first_fault, fault_code);
        return;
    }
    reward[lane_index] = result.reward;
#if defined(CUDARL_REGION_CREDIT)
    // Each credit needs one second of residence in a cell. A 0.8 s decision
    // can therefore contain at most one newly credited cell.
    static_assert(macro_controls * tail::motor_control_s < 1.f);
    if (!result.fault) {
        const int hits = tail::regional::new_regions(lane.state, previous_visits);
        region_hits[lane_index] = hits;
        reward[lane_index] += tail::regional::reward_per_region * hits;
    }
#endif
#if defined(CUDARL_CURVED_BANK)
    if(!result.fault){
    const auto curved=tail::curved_bank::observe(lane.state.motor,pond);
    const auto movement=tail::fast::feedback_sample(lane.state.motor);
    const float gap=float(tail::submerged_gap_m(pond,lane.state.motor.navigation.motion.body,lane.state.pose));
    reward[lane_index]+=tail::curved_bank::reward(curved,gap,cudarl::physics::norm(movement.velocity_m_s),result.controls*tail::motor_control_s);
    if(!isfinite(reward[lane_index])){fail(lane_index,33,first_fault,fault_code);return;}
    }
#endif
#if defined(CUDARL_FRONTIER_REWARD)
    const float potential_after=tail::frontier::potential(lane.state,lane.state.pose.com_m,
        lane.state.motor.navigation.motion.request.origin_m.y);
    reward[lane_index]+=tail::frontier::shaping(potential_before,potential_after,result.fault!=0,gamma);
    if(!isfinite(reward[lane_index])){fail(lane_index,33,first_fault,fault_code);return;}
#endif
    switch (result.fault) {
    case 0:
        if (!chooser::ready(lane.state)) fail(lane_index, 34, first_fault, fault_code);
#if defined(CUDARL_RELIABLE_CHOOSER)
        // Promotion records demonstrated safe transport. It changes only a
        // later training spawn, never the live body, command or episode signal.
        if (lane.curriculum_level < 2 &&
            lane.state.motor.navigation.motion.controls - lane.curriculum_mark_control >= 3000 &&
            lane.state.visits - lane.curriculum_mark_visits >= 8 &&
            lane.state.transport_m - lane.curriculum_mark_transport >= .30f && lane.state.max_stall_s <= 2.f) {
            ++lane.curriculum_level; ++counter.promotions; lane.curriculum_failures = 0;
            lane.curriculum_mark_control = lane.state.motor.navigation.motion.controls;
            lane.curriculum_mark_visits = lane.state.visits;
            lane.curriculum_mark_transport = lane.state.transport_m;
        }
#endif
        done[lane_index] = 0.f;
#if defined(CUDARL_EPISODIC_BANK)
        const int limit=
#if defined(CUDARL_OPEN_WATER)
            practice_limit(lane_index);
#elif defined(CUDARL_NORTH_BANK)
            north_practice_lane(lane_index) ? training_episode_controls : 0;
#else
            training_episode_controls;
#endif
        if(limit && lane.state.motor.navigation.motion.controls-episode.first_decision_control>=limit){
            truncated[lane_index]=done[lane_index]=1;lane.needs_start=true;++episode.time_limits;
        }
#endif
        return;
    case 2: ++counter.stops; break;
    case 4: ++counter.cue_timeouts; break;
    case 5: ++counter.posture_stops; break;
    default:
        fail(lane_index, result.fault, first_fault, fault_code);
        return;
    }
    // Training respawns only after a scored terminal. Rollout boundaries
    // otherwise carry the complete body, course history, trim and visit map.
    lane.needs_start = true;
#if defined(CUDARL_RELIABLE_CHOOSER)
    if (++lane.curriculum_failures >= 2 && lane.curriculum_level > 0) {
        --lane.curriculum_level; ++counter.demotions; lane.curriculum_failures = 0;
    }
#endif
}

#if defined(CUDARL_EPISODIC_BANK)
__global__ void observe_time_limits(const Lane* lanes,const float* truncated,geometry::Parameters pond,float* rows){
    const int lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=worlds)return;float observation[channels]{};
    // Use the real pre-reset healthy final state, even though needs_start is set.
    if(truncated[lane])chooser::observe(lanes[lane].state,pond,observation,chooser::default_observation_mode);
    for(int i=0;i<channels;++i)rows[lane*channels+i]=observation[i];
}
#endif

#if defined(CUDARL_FRONTIER_REWARD)
__global__ void frontier_probe(const Lane* lanes,float* values) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<worlds)values[i]=lanes[i].needs_start?0.f:tail::frontier::potential(lanes[i].state,
        lanes[i].state.pose.com_m,lanes[i].state.motor.navigation.motion.request.origin_m.y);
}
#endif
__global__ void audit_values(const float* values, int count, int* first_fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && !isfinite(values[i])) atomicMin(first_fault, i);
}

__global__ void exact_kl(const float* old_logits, const float* new_logits, float* kl, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float old_logp[actions], new_logp[actions];
    rl::log_softmax(old_logits + i * actions, actions, old_logp);
    rl::log_softmax(new_logits + i * actions, actions, new_logp);
    float div = 0.f;
    for (int j = 0; j < actions; ++j) div += expf(old_logp[j]) * (old_logp[j] - new_logp[j]);
    kl[i] = div;
}

__global__ void value_residual(const float* predicted, const float* target, float* residual, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) residual[i] = target[i] - predicted[i];
}

void audit(const float* values, int count, Rollout& roll, cudaStream_t stream) {
    audit_values<<<(count + math_threads - 1) / math_threads, math_threads, 0, stream>>>(
        values, count, roll.first_fault.get());
    check_launch();
}

void require_clear(Rollout& roll, Ledger& ledger, cudaStream_t stream, bool lane_fault) {
    std::array<int, 1> fault{};
    download(fault, roll.first_fault.get(), stream);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    if (fault[0] == INT_MAX) return;
    ledger.first_fault_lane = fault[0];
    if (lane_fault) {
        download(fault, roll.fault_code.get() + fault[0], stream);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        ledger.first_fault_code = fault[0];
    }
    throw std::runtime_error("CUDA chooser field or world failed validation");
}

void count_work(Rollout& roll, Ledger& ledger, cudaStream_t stream) {
    std::array<Counts, worlds> local{};
    download(local, roll.counts.get(), stream);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    ledger.counts = {};
#if defined(CUDARL_THREE_DEPTHS)
    ledger.samples_at_depth.fill(0);
#if defined(CUDARL_MOVING_STARTS)
    ledger.samples_by_practice = {};
#endif
    for (int lane=0; lane<worlds; ++lane)
        for (auto samples:local[lane].samples) {
            ledger.samples_at_depth[practice_depth_index(lane)]+=samples;
#if defined(CUDARL_MOVING_STARTS)
            const auto practice=practice_for(lane);
            const int group=practice==Practice::Moving ? bank_training::moving_runway_band(lane) :
                practice==Practice::Tangent ? 3 : 4;
            ledger.samples_by_practice[practice_depth_index(lane)][group]+=samples;
#endif
        }
#endif
    for (const auto& item : local) {
        ledger.counts.attempted += item.attempted;
        ledger.counts.accepted += item.accepted;
        ledger.counts.warmup_attempted += item.warmup_attempted;
        ledger.counts.warmup_accepted += item.warmup_accepted;
        ledger.counts.stops += item.stops;
        ledger.counts.cue_timeouts += item.cue_timeouts;
        ledger.counts.posture_stops += item.posture_stops;
        ledger.counts.starts += item.starts;
        ledger.counts.bank_left_starts += item.bank_left_starts;
        ledger.counts.bank_right_starts += item.bank_right_starts;
#if defined(CUDARL_RELIABLE_CHOOSER)
        for (int i = 0; i < 3; ++i) ledger.counts.curriculum_starts[i] += item.curriculum_starts[i];
        ledger.counts.promotions += item.promotions; ledger.counts.demotions += item.demotions;
#endif
        for (int i = 0; i < actions; ++i) ledger.counts.samples[i] += item.samples[i];
    }
    if (ledger.counts.attempted + ledger.counts.warmup_attempted > max_substeps)
        throw std::runtime_error("Chooser exceeded the physical work allowance");
#if defined(CUDARL_EPISODIC_BANK)
    std::array<EpisodeInfo,worlds> episodes{};download(episodes,roll.episodes.get(),stream);CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    ledger.time_limits=ledger.near_decisions=ledger.episode_decisions=0;
    for(const auto& e:episodes){ledger.time_limits+=e.time_limits;ledger.near_decisions+=e.near_decisions;ledger.episode_decisions+=e.decisions;}
#endif
}

float finite_mean(Rollout& roll, const float* in, int count, cudaStream_t stream) {
    rl::mean_std(in, count, roll.summary.get(), stream);
    std::array<float, 2> result{};
    download(result, roll.summary.get(), stream);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    if (!std::isfinite(result[0]) || !std::isfinite(result[1])) throw std::runtime_error("Nonfinite chooser reduction");
    return result[0];
}

bool collect(Rollout& roll, nn::GpuMlp& actor, nn::GpuMlp& critic, const device_ptr<tail::NavigationWorld>& starts,
             const device_ptr<float>& gait, const device_ptr<float>& offset, Ledger& ledger, int update,
             cudaStream_t stream, Events& events, Clock::time_point launch) {
    ledger.stage = "collection";
    ledger.started = update;
    constexpr int blocks = (worlds + lane_threads - 1) / lane_threads;
    for (int t = 0; t < decisions; ++t) {
        const auto counters_before = ledger.counts;
#if !defined(CUDARL_SUSTAINED_CHOOSER)
        if (elapsed(launch) + ledger.max_wave_s >= deadline_s) {
            ledger.stage = "deadline";
            return false;
        }
#endif
        const auto wave_start = Clock::now();
        const int first = t * worlds;
        auto* observation = roll.obs.get() + first * channels;
        auto* stored_action = roll.action.get() + first;
        auto* old_logp = roll.old_logp.get() + first;
        auto* value = roll.value.get() + first;
        CUDA_CHECK_THROW(cudaEventRecord(events.start, stream));
        start_needed_lanes<<<blocks, lane_threads, 0, stream>>>(roll.lanes.get(), roll.counts.get(), starts.get(),
            int(starts.size()), gait.get(), offset.get(), larger_pond, ledger.bank_curriculum,
            roll.first_fault.get(), roll.fault_code.get()
#if defined(CUDARL_EPISODIC_BANK)
            ,roll.episodes.get()
#endif
            );
        check_launch();
        CUDA_CHECK_THROW(cudaEventRecord(events.after_start, stream));
        observe_lanes<<<blocks, lane_threads, 0, stream>>>(roll.lanes.get(), larger_pond, observation,
            roll.first_fault.get(), roll.fault_code.get());
        check_launch();
        actor.forward(observation, worlds, stream);
        critic.forward(observation, worlds, stream);
        copy_device(roll.logits.get() + first * actions, actor.output(), worlds * actions, stream);
        copy_device(value, critic.output(), worlds, stream);
        const uint32_t draw = uint32_t((update - 1) * decisions + t);
        rl::sample_actions(actor.output(), worlds, actions, seed, draw, false, stored_action, old_logp, stream);
        CUDA_CHECK_THROW(cudaEventRecord(events.after_network, stream));
        move_lanes<<<blocks, lane_threads, 0, stream>>>(roll.lanes.get(), roll.counts.get(), stored_action, old_logp,
            value, gait.get(), offset.get(), larger_pond, roll.reward.get() + first, roll.done.get() + first,
            roll.first_fault.get(), roll.fault_code.get()
#if defined(CUDARL_REGION_CREDIT)
            ,roll.region_hits.get()+first
#endif
#if defined(CUDARL_EPISODIC_BANK)
            ,roll.episodes.get(),roll.truncated.get()+first
#endif
            );
        check_launch();
        CUDA_CHECK_THROW(cudaEventRecord(events.end, stream));
#if defined(CUDARL_EPISODIC_BANK)
        observe_time_limits<<<blocks,lane_threads,0,stream>>>(roll.lanes.get(),roll.truncated.get()+first,larger_pond,roll.final_obs.get());check_launch();
        critic.forward(roll.final_obs.get(),worlds,stream);
        copy_device(roll.terminal_value.get()+first,critic.output(),worlds,stream);
        CUDA_CHECK_THROW(cudaEventRecord(events.after_bootstrap,stream));
#endif
        const auto transfer_start = Clock::now();
        count_work(roll, ledger, stream);
        ledger.executed_attempted += ledger.counts.attempted + ledger.counts.warmup_attempted
            - counters_before.attempted - counters_before.warmup_attempted;
        ledger.executed_accepted += ledger.counts.accepted + ledger.counts.warmup_accepted
            - counters_before.accepted - counters_before.warmup_accepted;
        for (int i = 0; i < actions; ++i)
            ledger.executed_decisions += ledger.counts.samples[i] - counters_before.samples[i];
        if (ledger.executed_attempted > ledger.execution_physical_limit)
            throw std::runtime_error("Chooser invocation exceeded its physical execution allocation");
        require_clear(roll, ledger, stream, true);
        ledger.transfers_ms += elapsed(transfer_start) * 1000.;
        float ms = 0.f;
        CUDA_CHECK_THROW(cudaEventElapsedTime(&ms, events.start, events.end));
        ledger.gpu_dynamics_ms += ms;
        CUDA_CHECK_THROW(cudaEventElapsedTime(&ms, events.start, events.after_start));
        ledger.gpu_physics_ms += ms;
        CUDA_CHECK_THROW(cudaEventElapsedTime(&ms, events.after_network, events.end));
        ledger.gpu_physics_ms += ms;
        CUDA_CHECK_THROW(cudaEventElapsedTime(&ms, events.after_start, events.after_network));
        ledger.gpu_inference_ms += ms;
#if defined(CUDARL_EPISODIC_BANK)
        CUDA_CHECK_THROW(cudaEventElapsedTime(&ms,events.end,events.after_bootstrap));ledger.gpu_inference_ms+=ms;
#endif
        ledger.max_wave_s = std::max(ledger.max_wave_s, elapsed(wave_start));
    }
    observe_lanes<<<blocks, lane_threads, 0, stream>>>(roll.lanes.get(), larger_pond, roll.final_obs.get(),
        roll.first_fault.get(), roll.fault_code.get());
    check_launch();
    critic.forward(roll.final_obs.get(), worlds, stream);
    copy_device(roll.final_value.get(), critic.output(), worlds, stream);
    require_clear(roll, ledger, stream, true);
    return true;
}

float optimize(Rollout& roll, Batch& batch, nn::GpuMlp& actor, nn::GpuMlp& critic,
               nn::GpuAdam& actor_adam, nn::GpuAdam& critic_adam, Ledger& ledger, int update, cudaStream_t stream) {
    ledger.stage = "ppo";
    rl::compute_gae(decisions, worlds, roll.reward.get(), roll.value.get(), roll.done.get(),
#if defined(CUDARL_EPISODIC_BANK)
        roll.truncated.get(),roll.terminal_value.get(),roll.final_value.get(),gamma,lambda,roll.advantage.get(),
#else
        roll.zero.get(), roll.zero.get(), roll.final_value.get(), gamma, lambda, roll.advantage.get(),
#endif
        roll.target_return.get(), stream);
    audit(roll.advantage.get(), rows, roll, stream);
    audit(roll.target_return.get(), rows, roll, stream);
    require_clear(roll, ledger, stream, false);
    rl::mean_std(roll.advantage.get(), rows, roll.advantage_stats.get(), stream);
    ledger.policy_loss = ledger.value_loss = ledger.entropy = ledger.clip_fraction = 0.f;
    ledger.minibatches = 0;
    bool target_kl_reached = false;
    for (int epoch = 0; epoch < epochs && !target_kl_reached; ++epoch) {
        const auto perm = rl::make_permutation(rows, cudarl::rng::bits(seed, uint32_t(update), uint32_t(epoch)));
        for (int first = 0; first < rows; first += batch_size) {
            rl::gather_minibatch(perm, first, batch_size, channels, roll.obs.get(), roll.action.get(),
                roll.old_logp.get(), roll.advantage.get(), roll.target_return.get(), batch.obs.get(),
                batch.action.get(), batch.old_logp.get(), batch.advantage.get(), batch.target_return.get(), stream);
            actor.forward(batch.obs.get(), batch_size, stream);
            critic.forward(batch.obs.get(), batch_size, stream);
            zero(batch.stats.get(), batch.stats.size_bytes(), stream);
            rl::ppo_loss_grad(actor.output(), critic.output(), batch_size, actions, batch.action.get(),
                batch.old_logp.get(), batch.advantage.get(), batch.target_return.get(), roll.advantage_stats.get(),
                loss, batch.logit_grad.get(), batch.value_grad.get(), batch.row_stats.get(), batch.stats.get(), stream);
#if defined(CUDARL_SUSTAINED_CHOOSER)
            std::array<float, rl::kPpoStats> statistics{};
            download(statistics, batch.stats.get(), stream);
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
            for (const float statistic : statistics) {
                if (!std::isfinite(statistic)) throw std::runtime_error("Nonfinite PPO minibatch diagnostic");
            }
            if (statistics[3] / batch_size > .015f) {
                target_kl_reached = true;
                break;
            }
            ledger.policy_loss += statistics[0] / batch_size;
            ledger.value_loss += statistics[1] / batch_size;
            ledger.entropy += statistics[2] / batch_size;
            ledger.clip_fraction += statistics[4] / batch_size;
#endif
            actor.zero_grad(stream);
            critic.zero_grad(stream);
            actor.backward(batch.logit_grad.get(), batch_size, nullptr, stream);
            critic.backward(batch.value_grad.get(), batch_size, nullptr, stream);
            audit(batch.stats.get(), rl::kPpoStats, roll, stream);
            audit(actor.grads(), actor.num_params(), roll, stream);
            audit(critic.grads(), critic.num_params(), roll, stream);
            require_clear(roll, ledger, stream, false);
            actor_adam.step(actor.params(), actor.grads(), stream);
            ++ledger.actor_steps;
            ++ledger.executed_actor_steps;
            critic_adam.step(critic.params(), critic.grads(), stream);
            ++ledger.critic_steps;
            ++ledger.executed_critic_steps;
            ++ledger.minibatches;
            audit(actor.params(), actor.num_params(), roll, stream);
            audit(critic.params(), critic.num_params(), roll, stream);
            require_clear(roll, ledger, stream, false);
        }
    }
    if (ledger.minibatches > 0) {
        ledger.policy_loss /= ledger.minibatches;
        ledger.value_loss /= ledger.minibatches;
        ledger.entropy /= ledger.minibatches;
        ledger.clip_fraction /= ledger.minibatches;
    }
    for (int first = 0; first < rows; first += batch_size) {
        actor.forward(roll.obs.get() + first * channels, batch_size, stream);
        exact_kl<<<(batch_size + math_threads - 1) / math_threads, math_threads, 0, stream>>>(
            roll.logits.get() + first * actions, actor.output(), roll.kl.get() + first, batch_size);
        check_launch();
        critic.forward(roll.obs.get() + first * channels, batch_size, stream);
        value_residual<<<(batch_size + math_threads - 1) / math_threads, math_threads, 0, stream>>>(
            critic.output(), roll.target_return.get() + first, roll.residual.get() + first, batch_size);
        check_launch();
    }
#if defined(CUDARL_SUSTAINED_CHOOSER)
    std::array<float, 2> return_statistics{}, residual_statistics{};
    rl::mean_std(roll.target_return.get(), rows, roll.summary.get(), stream);
    download(return_statistics, roll.summary.get(), stream);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    rl::mean_std(roll.residual.get(), rows, roll.summary.get(), stream);
    download(residual_statistics, roll.summary.get(), stream);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    const float return_variance = return_statistics[1] * return_statistics[1];
    ledger.explained_variance = return_variance > 1e-12f
        ? 1.f - residual_statistics[1] * residual_statistics[1] / return_variance : 0.f;
#endif
    const float kl = finite_mean(roll, roll.kl.get(), rows, stream);
    if (kl > max_kl) throw std::runtime_error("Chooser whole-rollout KL cap exceeded");
    return kl;
}

void write_result(const fs::path& output, const Ledger& l) {
    std::ofstream result(output / "result.json");
    result.exceptions(std::ios::failbit | std::ios::badbit);
    const auto& c = l.counts;
    result << std::setprecision(9) << "{\"training_mode\":\""
           << (l.bank_curriculum ? "bank-curriculum" : "original") << "\",\"worlds\":" << worlds
           << ",\"decisions_per_update\":" << decisions
           << ",\"planned_updates\":" << max_updates << ",\"updates_started\":" << l.started
           << ",\"updates_completed\":" << l.completed << ",\"actor_adam_calls\":" << l.actor_steps
           << ",\"critic_adam_calls\":" << l.critic_steps << ",\"sampled_continue\":" << c.samples[0]
           << ",\"sampled_left\":" << c.samples[1] << ",\"sampled_right\":" << c.samples[2]
           << ",\"training_starts\":" << c.starts << ",\"clearance_stops\":" << c.stops
           << ",\"bank_left_starts\":" << c.bank_left_starts << ",\"bank_right_starts\":" << c.bank_right_starts
           << ",\"cue_timeouts\":" << c.cue_timeouts << ",\"posture_stops\":" << c.posture_stops
           << ",\"warmup_attempted_substeps\":" << c.warmup_attempted
           << ",\"warmup_accepted_substeps\":" << c.warmup_accepted
           << ",\"substeps_attempted\":" << c.attempted + c.warmup_attempted
           << ",\"substeps_accepted\":" << c.accepted + c.warmup_accepted
           << ",\"substep_cap\":" << max_substeps
           << ",\"frozen_motor_preflight_checks\":" << l.preflight_checks
           << ",\"frozen_motor_preflight_max_difference_rad\":" << l.preflight_max_difference_rad
           << ",\"gpu_collection_ms\":" << l.gpu_dynamics_ms << ",\"first_fault_lane\":" << l.first_fault_lane
           << ",\"first_fault_code\":" << l.first_fault_code << ",\"stage\":\"" << l.stage
           << "\",\"wall_s\":" << l.wall_s << "}\n";
}

#if !defined(CUDARL_RATE_CHOOSER)
void verify_frozen_motor(const std::vector<tail::NavigationWorld>& starts, const std::vector<float>& gait_weights,
                         const std::vector<float>& offset_weights, Ledger& ledger) {
    auto gait_spec = tail::navigation_motor_spec();
    gait_spec.activations.back() = nn::Activation::Identity;
    nn::CpuMlp<float> gait(gait_spec), offset(tail::yaw_offset_spec());
    gait.params() = gait_weights;
    offset.params() = offset_weights;
    for (const auto& source : starts) {
        chooser::World world{};
        world.navigation = source;
        auto& motion = world.navigation.motion;
        tail::prepare_navigation(world.navigation);
        world.pose = tail::snapshot(motion.body);
        if (cudarl::physics::norm(world.pose.com_velocity_m_s) < .003f)
            throw std::runtime_error("A training start was not captured while swimming");
        for (int test = 0; test < 2; ++test) {
            motion.request.choice = test == 0 ? tail::Choice::Swim :
                ledger.preflight_checks % 4 == 1 ? tail::Choice::TurnLeft : tail::Choice::TurnRight;
            world.applied_offset = test == 0 ? 0.f : .01f;
            float expected_offset = world.applied_offset, expected[tail::motor_outputs]{};
            float gait_obs[tail::navigation_inputs], offset_obs[tail::navigation_inputs];
            tail::observe_gait(world.navigation, gait_obs);
            tail::observe_yaw_offset(world.navigation, expected_offset, offset_obs);
            gait.forward(gait_obs, 1);
            offset.forward(offset_obs, 1);
            tail::decode_yaw_offset(world.navigation, gait.output(), offset.output()[0], expected_offset, expected);
            if (!chooser::frozen_targets(world, gait_weights.data(), offset_weights.data()))
                throw std::runtime_error("Nonfinite frozen motor preflight");
            for (int axis = 0; axis < tail::motor_outputs; ++axis) {
                ledger.preflight_max_difference_rad = std::max(ledger.preflight_max_difference_rad,
                    std::abs(expected[axis] - world.targets[axis]));
            }
            ++ledger.preflight_checks;
        }
    }
    if (ledger.preflight_max_difference_rad > 2e-7f)
        throw std::runtime_error("Frozen motor sparse inference differs from the original dense inference");
}

void record_bank_starts(const std::vector<tail::NavigationWorld>& starts, const fs::path& output) {
    std::ofstream evidence(output / "initial-approaches.csv");
    evidence.exceptions(std::ios::failbit | std::ios::badbit);
    evidence << "lane,archived_row,com_x_m,com_z_m,relative_yaw_rad,slant_deg,straight_runway_m,warmup_gap_m,inward_side\n"
             << std::setprecision(9);
    for (uint32_t lane = 0; lane < worlds; ++lane) {
        const auto row = start_source_row(start_episode_seed(lane, 0), int(starts.size()));
        cudarl::physics::V com;
        float yaw;
        bank_training::ApproachEvidence approach{};
        if (!bank_training::approach_start(lane, 0, starts[row], larger_pond, com, yaw, &approach))
            throw std::runtime_error("Invalid bank approach geometry for lane " + std::to_string(lane));
        evidence << lane << ',' << row << ',' << com.x << ',' << com.z << ',' << yaw << ',' << approach.slant_deg
                 << ',' << approach.straight_runway_m << ',' << approach.warmup_gap_m << ',' << approach.inward_side << '\n';
    }
}

void train(const std::string& gait_path, const std::string& offset_path, const std::string& starts_path,
           const fs::path& output, Clock::time_point launch, const char* prior_actor_path = nullptr) {
    if (fs::exists(output) || !fs::create_directory(output))
        throw std::runtime_error("Chooser output must start as a new directory");
    Ledger ledger;
    ledger.bank_curriculum = prior_actor_path != nullptr;
    try {
        const auto prior_actor = ledger.bank_curriculum ? chooser::read_weights(prior_actor_path) : std::vector<float>{};
        const auto gait_weights = tail::read_navigation_weights(gait_path, tail::NavigationMode::Gait);
        const auto offset_weights = tail::read_yaw_offset_weights(offset_path);
        chooser::validate_frozen_layout(gait_weights, offset_weights);
        const auto initial = tail::read_navigation_worlds(starts_path);
        if (initial.size() != 256) throw std::runtime_error("Expected the 256 training-only fitted starts");
        verify_frozen_motor(initial, gait_weights, offset_weights, ledger);
        std::cout << "frozen_motor_preflight_checks=" << ledger.preflight_checks
                  << " maximum_difference_rad=" << ledger.preflight_max_difference_rad << std::endl;
        if (ledger.bank_curriculum) record_bank_starts(initial, output);

        size_t existing_stack = 0;
        CUDA_CHECK_THROW(cudaDeviceGetLimit(&existing_stack, cudaLimitStackSize));
        CUDA_CHECK_THROW(cudaDeviceSetLimit(cudaLimitStackSize, (std::max)(existing_stack, size_t{16384})));
        cudaDeviceProp properties{};
        int device = 0;
        CUDA_CHECK_THROW(cudaGetDevice(&device));
        CUDA_CHECK_THROW(cudaGetDeviceProperties(&properties, device));
        std::cout << "gpu=" << properties.name << " worlds=" << worlds << " decisions=" << decisions
                  << " max_updates=" << max_updates << std::endl;

        Stream stream;
        Events events;
        Rollout roll;
        Batch batch;
        device_ptr<tail::NavigationWorld> starts(initial.size());
        device_ptr<float> gait(gait_weights.size()), offset(offset_weights.size());
        starts.copy_from_host(initial.data(), initial.size());
        gait.copy_from_host(gait_weights.data(), gait_weights.size());
        offset.copy_from_host(offset_weights.data(), offset_weights.size());
        nn::GpuMlp actor(chooser::actor_spec(), batch_size, seed);
        nn::GpuMlp critic({{channels, 32, 1}, {nn::Activation::Tanh, nn::Activation::Identity}},
            batch_size, seed + 1);
        auto checkpoint = actor.get_params();
        if (ledger.bank_curriculum) checkpoint = prior_actor;
        else std::fill(checkpoint.begin() + actor.spec().weight_offset(1), checkpoint.end(), 0.f);
        actor.set_params(checkpoint);
        nn::AdamConfig policy_config, value_config;
        policy_config.lr = .0015f;
        value_config.lr = .003f;
        policy_config.max_grad_norm = value_config.max_grad_norm = 1.f;
        nn::GpuAdam actor_adam(actor.num_params(), policy_config);
        nn::GpuAdam critic_adam(critic.num_params(), value_config);

        constexpr int no_fault = INT_MAX;
        CUDA_CHECK_THROW(cudaMemcpyAsync(roll.first_fault.get(), &no_fault, sizeof(no_fault),
            cudaMemcpyHostToDevice, stream.value));
        zero(roll.lanes.get(), roll.lanes.size_bytes(), stream.value);
        zero(roll.counts.get(), roll.counts.size_bytes(), stream.value);
        zero(roll.zero.get(), roll.zero.size_bytes(), stream.value);
        initialize_lanes<<<(worlds + lane_threads - 1) / lane_threads, lane_threads, 0, stream.value>>>(roll.lanes.get());
        check_launch();

        const auto save = [&](const std::string& name) {
            chooser::write_weights((output / name).u8string(), checkpoint);
        };
        save("actor-0.bin");
        std::ofstream progress(output / "updates.csv");
        progress.exceptions(std::ios::failbit | std::ios::badbit);
        progress << "update,reward_mean,exact_kl,substeps_attempted,substeps_accepted,"
            "sampled_continue,sampled_left,sampled_right,clearance_stops,cue_timeouts,posture_stops,"
            "collection_ms,ppo_ms,wall_s\n" << std::flush;
        for (int update = 1; update <= max_updates; ++update) {
            if (elapsed(launch) + decisions * ledger.max_wave_s >= deadline_s) {
                ledger.stage = "deadline";
                break;
            }
            const double gpu_ms_before = ledger.gpu_dynamics_ms;
            if (!collect(roll, actor, critic, starts, gait, offset, ledger, update, stream.value, events, launch)) break;
            const float reward = finite_mean(roll, roll.reward.get(), rows, stream.value);
            const auto collected = Clock::now();
            const float kl = optimize(roll, batch, actor, critic, actor_adam, critic_adam, ledger, update, stream.value);
            const auto optimized = Clock::now();
            ledger.stage = "checkpoint";
            checkpoint = actor.get_params();
            if (update % 4 == 0) save("actor-" + std::to_string(update) + ".bin");
            ledger.completed = update;
            ledger.wall_s = elapsed(launch);
            const auto& c = ledger.counts;
            progress << std::setprecision(9) << update << ',' << reward << ',' << kl << ','
                << c.attempted + c.warmup_attempted << ',' << c.accepted + c.warmup_accepted << ','
                << c.samples[0] << ',' << c.samples[1] << ',' << c.samples[2] << ','
                << c.stops << ',' << c.cue_timeouts << ',' << c.posture_stops << ','
                << ledger.gpu_dynamics_ms - gpu_ms_before << ','
                << std::chrono::duration<double, std::milli>(optimized - collected).count() << ','
                << ledger.wall_s << '\n' << std::flush;
            std::cout << "completed_update=" << update << " reward=" << reward << " exact_kl=" << kl
                      << " attempted_substeps=" << c.attempted + c.warmup_attempted << " elapsed_s="
                      << ledger.wall_s << std::endl;
        }
        if (ledger.completed != 0) save("actor-final.bin");
        if (ledger.completed == max_updates) ledger.stage = "complete";
        ledger.wall_s = elapsed(launch);
        write_result(output, ledger);
    } catch (...) {
        ledger.wall_s = elapsed(launch);
        write_result(output, ledger);
        throw;
    }
}
#endif
#if defined(CUDARL_RATE_CHOOSER)
#include "rate_session.cuh"
#elif defined(CUDARL_SUSTAINED_CHOOSER)
#include "sustained_session.cuh"
#endif
} // namespace

int main(int argc, char** argv) {
#if defined(CUDARL_RATE_CHOOSER)
    return rate_main(argc, argv);
#elif defined(CUDARL_SUSTAINED_CHOOSER)
    return sustained_main(argc, argv);
#else
    const bool bank_curriculum = argc > 1 && std::string(argv[1]) == "bank-curriculum";
    if (argc != (bank_curriculum ? 7 : 5)) {
        std::cerr << "Usage: pond_flexible_tail_chooser_ppo FITTED_GAIT OFFSET TRAIN_WORLDS NEW_OUTPUT\n"
            << "       pond_flexible_tail_chooser_ppo verify-frozen FITTED_GAIT OFFSET TRAIN_WORLDS\n"
            << "       pond_flexible_tail_chooser_ppo bank-curriculum FITTED_GAIT OFFSET TRAIN_WORLDS PRIOR_ACTOR NEW_OUTPUT\n";
        return 1;
    }
    try {
        if (bank_curriculum) {
            train(argv[2], argv[3], argv[4], fs::u8path(argv[6]), Clock::now(), argv[5]);
            return 0;
        }
        if (std::string(argv[1]) == "verify-frozen") {
            const auto gait = tail::read_navigation_weights(argv[2], tail::NavigationMode::Gait);
            const auto offset = tail::read_yaw_offset_weights(argv[3]);
            chooser::validate_frozen_layout(gait, offset);
            Ledger ledger;
            verify_frozen_motor(tail::read_navigation_worlds(argv[4]), gait, offset, ledger);
            std::cout << "frozen_motor_preflight_checks=" << ledger.preflight_checks
                      << " maximum_difference_rad=" << ledger.preflight_max_difference_rad << '\n';
            return 0;
        }
        train(argv[1], argv[2], argv[3], fs::u8path(argv[4]), Clock::now());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
#endif
}

