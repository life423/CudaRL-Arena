// Synthetic endpoint bookkeeping only: never integrates, samples, or optimizes.
#include "godot/untimed_swim_preview.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace sm = cudarl::straight_motor;
namespace live = cudarl::untimed_swim_preview;
namespace p = cudarl::physics;
namespace {
int assertions = 0;
void check(bool passed, const char* message) {
    ++assertions;
    if (!passed) throw std::runtime_error(message);
}
bool same(p::V a, p::V b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
bool task_fields(const sm::ForwardTask& a, const sm::ForwardTask& b) {
    const double x[] = {a.start_work_J,a.velocity_error_sq,a.head_yaw_sq,
        a.steady_speed_m_s,a.speed_rmse,a.path_angle_deg,a.maximum_lateral,a.maximum_depth,
        a.maximum_tilt,a.maximum_head_yaw};
    const double y[] = {b.start_work_J,b.velocity_error_sq,b.head_yaw_sq,
        b.steady_speed_m_s,b.speed_rmse,b.path_angle_deg,b.maximum_lateral,b.maximum_depth,
        b.maximum_tilt,b.maximum_head_yaw};
    return std::equal(std::begin(x),std::end(x),std::begin(y)) &&
        a.controls == b.controls && a.steady_samples == b.steady_samples &&
        a.last_world_control == b.last_world_control && a.hold_anchor_set == b.hold_anchor_set &&
        same(a.request.position,b.request.position) && same(a.request.heading,b.request.heading) &&
        same(a.request.terminal_velocity,b.request.terminal_velocity) && same(a.hold_anchor_model,b.hold_anchor_model) &&
        same(a.start_com_model,b.start_com_model) && same(a.steady_start_com_model,b.steady_start_com_model);
}
bool transition_fields(const sm::Transition& a, const sm::Transition& b) {
    const float x[] = {a.reward,a.metrics.forward_m,a.metrics.lateral_m,a.metrics.depth_m,a.metrics.speed_error_m_s,
        a.metrics.head_yaw_deg,a.metrics.tilt_deg,a.metrics.signed_depth_m,a.metrics.vertical_velocity_m_s,
        a.metrics.horizontal_speed_error_m_s,a.metrics.roll_deg,a.metrics.pitch_deg,
        a.costs.speed,a.costs.lane,a.costs.depth,a.costs.posture,a.costs.effort,
        a.costs.horizontal,a.costs.recovery,a.costs.roll,a.costs.pitch};
    const float y[] = {b.reward,b.metrics.forward_m,b.metrics.lateral_m,b.metrics.depth_m,b.metrics.speed_error_m_s,
        b.metrics.head_yaw_deg,b.metrics.tilt_deg,b.metrics.signed_depth_m,b.metrics.vertical_velocity_m_s,
        b.metrics.horizontal_speed_error_m_s,b.metrics.roll_deg,b.metrics.pitch_deg,
        b.costs.speed,b.costs.lane,b.costs.depth,b.costs.posture,b.costs.effort,
        b.costs.horizontal,b.costs.recovery,b.costs.roll,b.costs.pitch};
    return std::equal(std::begin(x),std::end(x),std::begin(y)) &&
        same(a.metrics.hold_error_m,b.metrics.hold_error_m) && same(a.metrics.velocity_m_s,b.metrics.velocity_m_s) &&
        a.accepted == b.accepted && a.truncated == b.truncated;
}
struct Fixture {
    sm::World world = sm::initial_world(0);
    sm::ForwardConfig config = sm::config_for_speed(.008f);
    sm::ForwardTask task{};
    Fixture() {
        world.arena = sm::long_swim_arena();
        config.observation_schema = 4;
        config.maximum_controls = 160;
        if (!sm::begin_forward(world,config,task)) throw std::runtime_error("Synthetic initial fixture invalid");
    }
};
void run() {
    Fixture f;
    auto finite_task = f.task, untimed_task = f.task;
    const auto initial_body = f.world.physical.body;
    const auto origin = untimed_task.start_com_model;
    const auto request = untimed_task.request;
    for (int tick = 1; tick <= 159; ++tick) {
        // Explicit synthetic endpoint clock assignment, not a physical step.
        f.world.physical.control_steps = static_cast<uint64_t>(tick);
        auto finite = sm::record_control(finite_task,f.world,f.config);
        auto untimed = live::record(untimed_task,f.world,f.config);
        check(task_fields(finite_task,untimed_task),"Original pre-deadline task bookkeeping changed");
        check(transition_fields(finite,untimed) && finite.done == untimed.done && finite.outcome == untimed.outcome,
            "Original pre-deadline metrics/reward/outcome changed");
    }
    f.world.physical.control_steps = 160;
    const auto old_final = sm::record_control(finite_task,f.world,f.config);
    const auto live_160 = live::record(untimed_task,f.world,f.config);
    check(old_final.done && finite_task.outcome != sm::Outcome::Running,"Original finite deadline was lost");
    check(live_160.accepted && !live_160.done && untimed_task.outcome == sm::Outcome::Running,"Live swim still terminates at160");
    check(task_fields(finite_task,untimed_task) && transition_fields(old_final,live_160),"Horizon changed nonterminal bookkeeping");
    float observed[41]{};
    check(!sm::observe_appended(finite_task,f.world,f.config,observed),"Original terminal task was revived");
    check(sm::observe_appended(untimed_task,f.world,f.config,observed),"Live observation rejected after160");
    for (int tick = 161; tick <= 162; ++tick) {
        f.world.physical.control_steps = static_cast<uint64_t>(tick);
        const auto out = live::record(untimed_task,f.world,f.config);
        check(out.accepted && !out.done && untimed_task.controls == tick && untimed_task.last_world_control == static_cast<uint64_t>(tick),
            "Live continuation lost clock/order after160");
    }
    check(same(untimed_task.start_com_model,origin) && same(untimed_task.request.position,request.position) &&
        same(untimed_task.request.heading,request.heading) && same(untimed_task.request.terminal_velocity,request.terminal_velocity),
        "Untimed bookkeeping rebased original course or origin");
    check(std::memcmp(&initial_body,&f.world.physical.body,sizeof(initial_body)) == 0 && f.world.physical.substeps == 0 &&
        f.world.attempted_substeps == 0 && f.world.physical.time_seconds == 0,"Zero-time bookkeeping mutated physical state");

    Fixture order;
    order.world.physical.control_steps = 2;
    const auto out_of_order = live::record(order.task,order.world,order.config);
    check(!out_of_order.accepted && out_of_order.done && order.task.outcome == sm::Outcome::Rejected && order.task.controls == 0,
        "Out-of-order control was accepted");
    Fixture completed;
    completed.task.outcome = sm::Outcome::Success;
    completed.world.physical.control_steps = 1;
    const auto terminal = live::record(completed.task,completed.world,completed.config);
    check(!terminal.accepted && terminal.done && completed.task.outcome == sm::Outcome::Success && completed.task.controls == 0,
        "Completed task was revived");
    Fixture bad_reward;
    // Keep the physical world valid and exercise the separate reward-finite
    // guard after warmup, when horizontal velocity error is scored.
    bad_reward.task.controls = 40;
    bad_reward.task.last_world_control = 40;
    bad_reward.world.physical.control_steps = 41;
    bad_reward.task.request.terminal_velocity.x = std::numeric_limits<float>::infinity();
    check(sm::in_domain(bad_reward.world,bad_reward.config),"Finite-reward probe bypassed before reward check");
    const auto nonfinite = live::record(bad_reward.task,bad_reward.world,bad_reward.config);
    check(!nonfinite.accepted && nonfinite.done && bad_reward.task.outcome == sm::Outcome::Rejected && bad_reward.task.controls == 40,
        "Nonfinite reward escaped rejection");
    Fixture invalid_world;
    invalid_world.world.physical.control_steps = 1;
    invalid_world.world.physical.body.position.x = std::numeric_limits<float>::infinity();
    const auto invalid = live::record(invalid_world.task,invalid_world.world,invalid_world.config);
    check(!invalid.accepted && invalid.done && invalid_world.task.outcome == sm::Outcome::Rejected,"Physical domain rejection lost");
    Fixture overflow;
    overflow.task.controls = std::numeric_limits<int>::max();
    overflow.task.last_world_control = 10;
    overflow.world.physical.control_steps = 11;
    const auto maximum = live::record(overflow.task,overflow.world,overflow.config);
    check(!maximum.accepted && maximum.done && overflow.task.controls == std::numeric_limits<int>::max() &&
        overflow.task.outcome == sm::Outcome::Rejected,"Signed task counter overflow was not rejected");
}
}
int main() {
    try {
        run();
        std::cout << "{\"passed\":true,\"assertions\":" << assertions
            << ",\"synthetic_endpoint_records\":327,\"physical_controls\":0,\"physical_substeps\":0,\"CUDA_kernels\":0,\"optimizer_steps\":0}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Bookkeeping check failed: " << e.what() << '\n';
        return 3;
    }
}
