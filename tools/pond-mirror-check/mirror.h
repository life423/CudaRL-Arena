#pragma once
#include "rl/pond_ppo_task.h"
#include <array>

namespace mirror {
namespace p = cudarl::physics;
namespace runtime = cudarl::pond_runtime;
namespace obs = cudarl::pond_motor_observation;
namespace task = cudarl::rl::pond_ppo_task;
using Command = std::array<float, 4>;
constexpr int num_cases = 16, controls = 360, gpu_controls = 20;
constexpr double tolerance = 2e-6;
CUDARL_HD inline int joint_sign(int axis) { return axis % 2 ? 1 : -1; }
CUDARL_HD inline int velocity_sign(int axis) {
    if (axis < 3) return axis == 2 ? -1 : 1;
    if (axis < 6) return axis == 5 ? 1 : -1;
    return joint_sign(axis - 6);
}
CUDARL_HD inline p::V polar(p::V v) { return {v.x,v.y,-v.z}; }
CUDARL_HD inline p::BodyState reflected(p::BodyState b) {
    b.position = polar(b.position);
    b.orientation.x *= -1;
    b.orientation.y *= -1;
    for (int j = 0; j < 4; ++j) b.angle[j] *= joint_sign(j);
    for (int i = 0; i < 10; ++i) b.velocity[i] *= velocity_sign(i);
    return b;
}
enum Field { Position, Quaternion, Angle, Velocity, FilterTarget, FilterRate, Applied,
             Requested, PreviousTorque, Work, Travel, ComVelocity, Fields };
inline const char* names[Fields] = {"position_model","quaternion","angle_rad","velocity",
    "filter_target_rad","filter_rate_rad_s","applied_normalized","requested_Nm",
    "previous_torque_normalized","work_J","travel_model","COM_velocity_m_s"};
struct Stats {
    double max_error[Fields]{}, min_gap_m = 1e30;
    unsigned long long steps = 0;
    int first_field = -1, first_step = -1, failure = 0;
    CUDARL_HD void compare(Field field, double a, double b) {
        const double diff = fabs(a-b);
        const bool finite = cudarl::pond_geometry_v1::finite(diff);
        max_error[field] = fmax(max_error[field], diff);
        if ((!finite || diff > tolerance) && first_field < 0) {
            first_field = field;
            first_step = static_cast<int>(steps);
        }
    }
    CUDARL_HD void vector(Field f, p::V a, p::V b) {
        compare(f,a.x,b.x); compare(f,a.y,b.y); compare(f,a.z,b.z);
    }
    CUDARL_HD void compare_step(const runtime::SubstepTrace& a, const runtime::SubstepTrace& b) {
        ++steps;
        if (!a.accepted || !b.accepted) { failure = 1; return; }
        const auto expected = reflected(a.after);
        const auto& actual = b.after;
        vector(Position,expected.position,actual.position);
        compare(Quaternion,expected.orientation.x,actual.orientation.x);
        compare(Quaternion,expected.orientation.y,actual.orientation.y);
        compare(Quaternion,expected.orientation.z,actual.orientation.z);
        compare(Quaternion,expected.orientation.w,actual.orientation.w);
        compare(Work,expected.work,actual.work);
        compare(Travel,expected.travel,actual.travel);
        vector(ComVelocity,polar(obs::motion(a.after).velocity_m_s),obs::motion(b.after).velocity_m_s);
        for (int i=0; i<10; ++i) compare(Velocity,expected.velocity[i],actual.velocity[i]);
        for (int j=0; j<4; ++j) {
            const int s = joint_sign(j);
            compare(Angle,expected.angle[j],actual.angle[j]);
            compare(FilterTarget,s*a.held_targets[j],b.held_targets[j]);
            compare(FilterRate,s*a.reference_rate[j],b.reference_rate[j]);
            compare(Applied,s*a.applied_torque_normalized[j],b.applied_torque_normalized[j]);
            compare(Requested,s*a.requested_torque_Nm[j],b.requested_torque_Nm[j]);
            compare(PreviousTorque,s*a.last_actual_torques[j],b.last_actual_torques[j]);
        }
        min_gap_m = fmin(min_gap_m, fmin(a.gap_lower_m,b.gap_lower_m));
        if (min_gap_m <= 0 || a.step_spring_work != 0 || b.step_spring_work != 0 ||
            a.step_damping_loss != 0 || b.step_damping_loss != 0 ||
            a.step_potential_after != 0 || b.step_potential_after != 0)
            failure = 2;
    }
};
struct GpuPair {
    runtime::World worlds[2];
    float commands[gpu_controls][4]{};
    Stats stats{};
};
void replay_cuda(GpuPair* pairs);
}
