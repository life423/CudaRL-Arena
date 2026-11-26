#pragma once
#include "rl/flexible_tail_fast_steering.h"

namespace cudarl::flexible_tail::curved_bank {
inline constexpr uint64_t identity=0x4355525645323401ULL;
inline constexpr float horizon_s=24.f, interval_s=.5f;
struct Forecast {float loss_s=horizon_s,min_gap_m=FLT_MAX,rate_rad_s=0;bool valid=false;};

// Kinematic sensory estimate from actual whole-body travel, not a future force,
// prescribed action or guarantee of future gait/contact. The rigid current
// envelope is swept along a constant-curvature path. Half-interval inflation
// bounds motion between samples for this kinematic model only.
CUDARL_HD inline Forecast project(const physics::BodyState& body,const Snapshot& pose,
    const pond_geometry_v1::Parameters& pond,physics::V velocity,float omega) {
    Forecast out;out.rate_rad_s=omega;velocity.y=0;
    const float speed=physics::norm(velocity);
    if(speed<.003f||!physics::finite_number(omega)||!physics::finite_body(body))return out;
    out.valid=true;
    const auto origin=pose.com_m;
    float extent=physics::norm(body.position*physics::scale-origin);
    for(const auto& point:pose.spine_m)extent=fmaxf(extent,physics::norm(point-origin));
    const float margin=(speed+fabsf(omega)*extent)*interval_s*.5f;
    for(int step=0;step<int(horizon_s/interval_s);++step){
        const float time=(step+.5f)*interval_s,angle=omega*time;
        const auto rotation=physics::axis_rotation({0,1,0},angle);
        const auto sideways=physics::cross({0,1,0},velocity);
        const auto displacement=fabsf(omega)>1.e-5f?
            velocity*(sinf(angle)/omega)+sideways*((1-cosf(angle))/omega):velocity*time;
        auto projected_body=body;auto projected_pose=pose;
        projected_body.position=(origin+displacement+rotation.rotate(body.position*physics::scale-origin))/physics::scale;
        projected_body.orientation=rotation*body.orientation;
        projected_pose.com_m=origin+displacement;
        for(auto& point:projected_pose.spine_m)point=origin+displacement+rotation.rotate(point-origin);
        const float gap=float(submerged_gap_m(pond,projected_body,projected_pose,margin));
        out.min_gap_m=fminf(out.min_gap_m,gap);
        if(gap<=0){out.loss_s=step*interval_s;break;}
    }
    return out;
}
CUDARL_HD inline Forecast observe(const fast::World& world,const pond_geometry_v1::Parameters& pond){
    const auto now=fast::feedback_sample(world),before=fast::feedback_sample(world,25);
    if(!now.reliable||!before.reliable)return {};
    const float rate=yaw_offset_error(before.velocity_m_s,now.velocity_m_s)/.5f;
    return project(world.navigation.motion.body,world.pose,pond,now.velocity_m_s,rate);
}
CUDARL_HD inline float reward(const Forecast& prediction,float physical_gap_m,float speed_m_s,float seconds){
    if(!prediction.valid)return 0;
    const float risk=physics::clamp((horizon_s-prediction.loss_s)/horizon_s,0.f,1.f);
    const float close=physical_gap_m>0&&physical_gap_m<=.020f&&speed_m_s>=.006f&&prediction.loss_s>=horizon_s?.02f:0.f;
    return seconds*(close-.10f*risk);
}
}
