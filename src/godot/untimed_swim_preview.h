#ifndef CUDARL_GODOT_UNTIMED_SWIM_PREVIEW_H
#define CUDARL_GODOT_UNTIMED_SWIM_PREVIEW_H
#include "motor/straight_task.h"
#include <limits>

namespace cudarl::untimed_swim_preview {
namespace sm = straight_motor;
namespace p = physics;

// Interactive endpoint bookkeeping, using the original metrics and ordering.
// The research recorder in straight_task.h keeps its finite episode/scorer.
// This path has no episode deadline and never revives a completed task.
inline sm::Transition record(sm::ForwardTask& t, const sm::World& w, const sm::ForwardConfig& c) {
    sm::Transition out;
    if (t.outcome != sm::Outcome::Running) {
        out.done = true;
        out.outcome = t.outcome;
        return out;
    }
    if (t.controls == std::numeric_limits<int>::max() || !sm::in_domain(w,c) ||
        w.physical.control_steps != t.last_world_control + 1) {
        t.outcome = out.outcome = sm::Outcome::Rejected;
        out.done = true;
        return out;
    }
    out.metrics = sm::measure(t,w);
    out.costs = sm::reward_costs(out.metrics,w,c,t.controls >= c.measure_start);
    out.reward = -p::dt * out.costs.total();
    if (!p::finite_number(out.reward)) {
        t.outcome = out.outcome = sm::Outcome::Rejected;
        out.done = true;
        return out;
    }
    ++t.controls;
    t.last_world_control = w.physical.control_steps;
    t.maximum_lateral = fmaxf(t.maximum_lateral,out.metrics.lateral_m);
    t.maximum_depth = fmaxf(t.maximum_depth,out.metrics.depth_m);
    t.maximum_tilt = fmaxf(t.maximum_tilt,out.metrics.tilt_deg);
    t.maximum_head_yaw = fmaxf(t.maximum_head_yaw,out.metrics.head_yaw_deg);
    t.head_yaw_sq += out.metrics.head_yaw_deg * out.metrics.head_yaw_deg;
    const auto motion = motor::motion(w.physical.body);
    if (t.controls == c.measure_start) t.steady_start_com_model = motion.com_model;
    if (t.controls > c.measure_start) {
        ++t.steady_samples;
        t.velocity_error_sq += out.metrics.speed_error_m_s * out.metrics.speed_error_m_s;
        const auto delta = (motion.com_model-t.steady_start_com_model)*p::scale;
        const float forward = p::dot(delta,t.request.heading);
        const auto perpendicular = delta-t.request.heading*forward;
        t.steady_speed_m_s = forward/(float(t.steady_samples)*p::dt);
        t.speed_rmse = sqrtf(static_cast<float>(t.velocity_error_sq/t.steady_samples));
        t.path_angle_deg = atan2f(p::norm({perpendicular.x,0,perpendicular.z}),forward)*180/movement::pi;
    }
    out.accepted = true;
    out.outcome = t.outcome;
    return out;
}
}
#endif
