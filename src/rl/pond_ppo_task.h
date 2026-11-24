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
inline constexpr uint32_t version = 2;
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

CUDARL_HD inline bool initialize(runtime::World& world, Task& task, uint32_t env,
                                 uint32_t episode, uint32_t seed) {
    const uint64_t failures = task.task_failures, truncations = task.truncations;
    const uint64_t completed = task.completed_episodes;
    task = Task{};
    task.task_failures = failures; task.truncations = truncations; task.completed_episodes = completed;
    task.episode_index = episode;
    const uint32_t mode = rng::bits(seed, rng::hash_u32(env ^ 0x6c9a24d1u), episode) % 4u;
    task.kind = mode == 0 ? Hold : (mode == 1 ? Straight : Turn);
    const bool holding = task.kind == Hold;
    const uint32_t start = rng::bits(seed, rng::hash_u32(env ^ 0xa4175b39u), episode) % (holding ? 6u : 4u);
    if (task.kind == Turn) {
        const uint32_t turn = rng::bits(seed, rng::hash_u32(env ^ 0xe31f680bu), episode) % 10u;
        const uint32_t magnitude = turn % 5u;
        const float degrees = magnitude == 0 ? 5.f : (magnitude == 1 ? 15.f :
            (magnitude == 2 ? 30.f : (magnitude == 3 ? 60.f : 90.f)));
        task.turn_radians = (turn < 5 ? -degrees : degrees) * obs::pi / 180.f;
    }
    world = runtime::World{};
    auto& physical = world.motor.physical;
    if (!obs::initialize(static_cast<int>(start), holding, physical.body, physical.held_targets,
                         world.motor.reference_rate, physical.last_actual_torques, task.request)) return false;
    physical.paused = false;
    // prepare() certifies the initial geometry and energy bookkeeping; it does
    // not advance the body, time, filters, physical controls or substeps.
    return runtime::prepare(world);
}

CUDARL_HD inline bool observe(Task& task, runtime::World& world, float* output) {
    if (task.kind < Hold || task.kind > Turn) return false;
    // Rotate the ORIGINAL persistent world request exactly once after forty
    // successful controls. Preserve its depth, body, filter and torque history.
    if (task.kind == Turn && task.episode_controls == turn_control && !task.request_changed) {
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

// Called ONLY after one complete successful runtime control. A physical stop is
// rejected by the collector before this call, not relabeled as an episode end.
// A 64-control rollout segment is not a task episode boundary.
CUDARL_HD inline Transition record(Task& task, runtime::World& world) {
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
        transition.reward = control_seconds * motion_gate * (.35f * speed_quality + .65f * positive_requested_progress) *
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
