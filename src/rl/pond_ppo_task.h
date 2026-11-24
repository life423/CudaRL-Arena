#ifndef CUDARL_RL_POND_PPO_TASK_H
#define CUDARL_RL_POND_PPO_TASK_H

#include "core/rng.h"
#include "motor/pond_motor_observation.h"
#include "simulation/pond_runtime.h"

// Shared task bookkeeping and rewards only. No action selection, bank policy,
// neural network, optimizer, pose correction or physical integration lives here.
// CUDA callers include their fixed math profile before including this header.
namespace cudarl::rl::pond_ppo_task {
namespace runtime = pond_runtime;
namespace obs = pond_motor_observation;
namespace p = physics;

enum TaskKind : int { Hold = 0, Straight = 1, Turn = 2 };
inline constexpr uint32_t version = 4;
inline constexpr uint32_t curriculum_off = 0, paired_gentle_moderate_v1 = 1, course_direction_v1 = 2, repeated_headings_v1 = 3;
inline constexpr uint32_t turn_control = 40, episode_limit = 1200;
inline constexpr float target_speed_m_s = .008f, control_seconds = .05f;

struct Task {
    obs::RequestState request{};
    uint32_t episode_index = 0, episode_controls = 0;
    int kind = Hold;
    float turn_radians = 0;
    bool request_changed = false;
    uint64_t task_failures = 0, truncations = 0, completed_episodes = 0;
    uint32_t occupied_grid[8]{}, occupied_cells = 0;
    float bank_near_seconds = 0, contact_seconds = 0, postwarmup_stall_seconds = 0;
    double last_spring_work = 0, last_damping_loss = 0;
};

struct Transition {
    float reward = 0;
    bool terminal = false, truncated = false;
    int reason = 0; // 0 clean/running, 1 depth task failure, 2 tilt task failure
};

struct RequestSample {
    int kind = Hold;
    uint32_t start = 0, stage = 0;
    float turn_radians = 0;
};

// Request sampling only. Off preserves the exact v2 RNG/formulas. Opt-in pairs
// share kind/start/magnitude at a matching episode ordinal, with opposite signs.
// Staging happens at a genuine episode birth, never at a rollout boundary.
CUDARL_HD inline RequestSample sample_request(uint32_t env,uint32_t episode,uint32_t seed,
                                             uint32_t curriculum=curriculum_off) {
    RequestSample sample;
    if(curriculum>repeated_headings_v1){sample.kind=-1;return sample;}
    const uint32_t key=curriculum==paired_gentle_moderate_v1?env/2:env;
    const uint32_t mode=rng::bits(seed,rng::hash_u32(key^0x6c9a24d1u),episode)%4u;
    sample.kind=mode==0?Hold:(mode==1?Straight:Turn);
    sample.start=rng::bits(seed,rng::hash_u32(key^0xa4175b39u),episode)%(sample.kind==Hold?6u:4u);
    if(sample.kind==Turn) {
        const uint32_t draw=rng::bits(seed,rng::hash_u32(key^0xe31f680bu),episode);
        if(curriculum!=paired_gentle_moderate_v1) {
            const uint32_t turn=draw%10u,magnitude=turn%5u;
            const float degrees=magnitude==0?5.f:(magnitude==1?15.f:(magnitude==2?30.f:(magnitude==3?60.f:90.f)));
            sample.turn_radians=(turn<5?-degrees:degrees)*obs::pi/180.f;
        } else {
            sample.stage=episode==0?0u:1u;
            const uint32_t magnitude=draw%3u;
            const float degrees=sample.stage==0?(magnitude==0?7.f:(magnitude==1?12.f:18.f)):
                (magnitude==0?18.f:(magnitude==1?24.f:33.f));
            sample.turn_radians=((env&1u)?degrees:-degrees)*obs::pi/180.f;
        }
    }
    return sample;
}

CUDARL_HD inline bool initialize(runtime::World& world, Task& task, uint32_t env,
                                 uint32_t episode, uint32_t seed,uint32_t curriculum=curriculum_off) {
    const uint64_t failures = task.task_failures, truncations = task.truncations;
    const uint64_t completed = task.completed_episodes;
    task = Task{};
    task.task_failures = failures; task.truncations = truncations; task.completed_episodes = completed;
    task.episode_index = episode;
    const auto sample=sample_request(env,episode,seed,curriculum);
    if(sample.kind<Hold||sample.kind>Turn)return false;
    task.kind=sample.kind;
    const bool holding = task.kind == Hold;
    task.turn_radians=sample.turn_radians;
    world = runtime::World{};
    auto& physical = world.motor.physical;
    if (!obs::initialize(static_cast<int>(sample.start), holding, physical.body, physical.held_targets,
                         world.motor.reference_rate, physical.last_actual_torques, task.request)) return false;
    physical.paused = false;
    // prepare() certifies the initial geometry and energy bookkeeping; it does
    // not advance the body, time, filters, physical controls or substeps.
    return runtime::prepare(world);
}

CUDARL_HD inline bool observe(Task& task, runtime::World& world, float* output, bool automatic_first_request = true) {
    if (task.kind < Hold || task.kind > Turn) return false;
    // Rotate the ORIGINAL persistent world request exactly once after forty
    // successful controls. Preserve its depth, body, filter and torque history.
    if (automatic_first_request && task.kind == Turn && task.episode_controls == turn_control && !task.request_changed) {
        const auto rotation = p::axis_rotation({0, 1, 0}, task.turn_radians);
        auto& request = task.request.request;
        request.heading = rotation.rotate(request.heading);
        request.terminal_velocity = rotation.rotate(request.terminal_velocity);
        request.position = task.request.start_com_model +
            rotation.rotate(request.position - task.request.start_com_model);
        task.request_changed = true;
    }
    const auto& physical = world.motor.physical;
    if (!obs::observe(task.request, physical.body, physical.held_targets, world.motor.reference_rate,
                      physical.last_actual_torques, output)) return false;
    // Straight uses the saved motor's full cross-track and depth feedback.
    // A heading-only turn carries depth error through the same body-frame transform.
    if (task.kind == Turn) {
        const auto current = obs::motion(physical.body);
        const p::V vertical_error{0, (task.request.start_com_model.y - current.com_model.y) * p::scale, 0};
        const auto body_error = physical.body.orientation.inverse(vertical_error) / obs::path_scale_m;
        if (!p::finite_vector(body_error)) return false;
        obs::vector_features(output, 0, body_error);
    }
    return true;
}

CUDARL_HD inline float squared(float value) { return value * value; }

CUDARL_HD inline float direction_factor(p::V horizontal_velocity,p::V requested_heading) {
    const float speed=p::norm(horizontal_velocity);
    const p::V horizontal_heading{requested_heading.x,0,requested_heading.z};
    const float heading_length=p::norm(horizontal_heading);
    if(!(speed>0)||!(heading_length>0))return 1.f;
    const float cosine=p::clamp(p::dot(horizontal_velocity,horizontal_heading)/(speed*heading_length),-1.f,1.f);
    constexpr float one_minus_cos15=1.f-.9659258262890683f;
    return 1.f/(1.f+(1.f-cosine)/one_minus_cos15);
}
CUDARL_HD inline float moving_reward_terms(float speed_quality,float positive_requested_progress,
        p::V horizontal_velocity,p::V requested_heading,int kind,bool request_changed,uint32_t program) {
    float rewarded_progress=positive_requested_progress;
    if((program==course_direction_v1||program==repeated_headings_v1)&&kind==Turn&&request_changed)
        rewarded_progress*=direction_factor(horizontal_velocity,requested_heading);
    return .35f*speed_quality+.65f*rewarded_progress;
}

// Called ONLY after one complete successful runtime control. A physical stop is
// rejected by the collector before this call, not relabeled as an episode end.
// A 64-control rollout segment is not a task episode boundary.
CUDARL_HD inline Transition record(Task& task, runtime::World& world,uint32_t program=curriculum_off) {
    Transition transition;
    if (world.failure != 0 || world.motor.physical.invalid || world.motor.physical.unsafe) return transition;
    ++task.episode_controls; ++task.request.controls;
    const auto& physical = world.motor.physical;
    const auto current = obs::motion(physical.body);
    const auto metrics = obs::measure(task.request, physical.body);
    // Raw per-episode world-grid diagnostics, never a reward or steering rule.
    const float x=current.com_model.x*p::scale,z=current.com_model.z*p::scale;
    const double rx=world.terrain.water_radius_x_m,rz=world.terrain.water_radius_z_m;
    if(x>=-rx&&x<rx&&z>=-rz&&z<rz) {
        const auto column=static_cast<uint32_t>(floor((double(x)/rx+1)*8));
        const auto row=static_cast<uint32_t>(floor((double(z)/rz+1)*8));
        if(column<16&&row<16){const uint32_t cell=row*16+column,bit=1u<<(cell%32);
            if((task.occupied_grid[cell/32]&bit)==0){task.occupied_grid[cell/32]|=bit;++task.occupied_cells;}}
    }
    if(world.minimum_control_gap_m<=.01)task.bank_near_seconds+=control_seconds;
    if(world.contact_potential_J>0||world.contact_spring_work!=task.last_spring_work||world.contact_damping_loss!=task.last_damping_loss)
        task.contact_seconds+=control_seconds;
    task.last_spring_work=world.contact_spring_work;task.last_damping_loss=world.contact_damping_loss;
    const float diagnostic_speed=p::norm({current.velocity_m_s.x,0,current.velocity_m_s.z});
    if(task.episode_controls>turn_control&&diagnostic_speed<.003f)task.postwarmup_stall_seconds+=control_seconds;
    if (metrics.depth_m > .005f || metrics.tilt_deg > 30.f) {
        transition.reward = -1.f; transition.terminal = true;
        transition.reason = metrics.depth_m > .005f ? 1 : 2;
        ++task.task_failures; ++task.completed_episodes;
        return transition;
    }
    const float tilt_quality = expf(-squared(metrics.tilt_deg / 15.f));
    if (task.kind == Hold) {
        const float anchor_error = p::norm(metrics.hold_error_m);
        const float speed = p::norm(current.velocity_m_s);
        transition.reward = control_seconds * expf(-squared(anchor_error / .001f) - squared(speed / .001f)) * tilt_quality;
    } else {
        const float horizontal_speed = p::norm({current.velocity_m_s.x, 0, current.velocity_m_s.z});
        const float speed_quality = expf(-squared((horizontal_speed - target_speed_m_s) / .006f));
        const p::V horizontal_velocity{current.velocity_m_s.x, 0, current.velocity_m_s.z};
        const float positive_requested_progress = p::clamp(p::dot(horizontal_velocity, task.request.request.heading) /
                                                           target_speed_m_s, 0, 1);
        const float motion_gate = p::clamp(horizontal_speed / target_speed_m_s, 0, 1);
        const float depth_quality = expf(-squared(metrics.depth_m / .003f));
        const float effort_quality = 1.f - .1f * p::clamp(physical.mean_squared_torque, 0, 1);
        transition.reward = control_seconds * motion_gate * moving_reward_terms(speed_quality,positive_requested_progress,
            horizontal_velocity,task.request.request.heading,task.kind,task.request_changed,program) *
            depth_quality * tilt_quality * effort_quality;
    }
    // All clamping above bounds reward factors only, never physical state.
    if (task.episode_controls >= episode_limit) {
        transition.terminal = true; transition.truncated = true;
        ++task.truncations; ++task.completed_episodes;
    }
    return transition;
}

} // namespace cudarl::rl::pond_ppo_task
#endif
