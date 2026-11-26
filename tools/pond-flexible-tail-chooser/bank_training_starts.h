#pragma once

#include "core/rng.h"
#include "rl/flexible_tail_chooser.h"

namespace cudarl::flexible_tail::bank_training {

inline constexpr float min_slant_deg = 34.f;
inline constexpr float max_slant_deg = 44.f;
inline constexpr float min_straight_runway_m = .170f;
inline constexpr float max_straight_runway_m = .205f;
inline constexpr float warmup_travel_m = .072f;
inline constexpr float min_warmup_gap_m = .020f;
inline constexpr uint32_t placement_seed = 26112402u;

struct ApproachEvidence {
    float slant_deg = 0;
    float straight_runway_m = 0;
    float warmup_gap_m = 0;
    int inward_side = 0;
};

// This is a spawn geometry query. It leaves the saved fixture unchanged and
// neither advances the body nor prescribes any chooser action.
CUDARL_HD inline bool approach_start(uint32_t lane, uint32_t episode, const NavigationWorld& source,
                                    const pond_geometry_v1::Parameters& pond, physics::V& position_m,
                                    float& relative_yaw_rad, ApproachEvidence* evidence = nullptr, int level = 2) {
    if (!pond_geometry_v1::valid_parameters(pond) || !physics::finite_body(source.motion.body) ||
        source.leg_count < 1) return false;

    const uint32_t pair_seed = rng::hash_u32(lane / 2) ^ rng::hash_u32(episode + 1);
    const float azimuth = 2.f * pi * rng::uniform(placement_seed, pair_seed, 0);
    const float lower_slant = level == 0 ? 25.f : level == 1 ? 30.f : min_slant_deg;
    const float upper_slant = level == 0 ? 30.f : level == 1 ? 36.f : max_slant_deg;
    const float lower_runway = level == 0 ? .24f : level == 1 ? .205f : min_straight_runway_m;
    const float upper_runway = level == 0 ? .27f : level == 1 ? .24f : max_straight_runway_m;
    const float slant_deg = lower_slant + (upper_slant - lower_slant) *
        rng::uniform(placement_seed, pair_seed, 1);
    const float runway = lower_runway + (upper_runway - lower_runway) *
        rng::uniform(placement_seed, pair_seed, 2);
    const int inward_side = (lane % 2) == 0 ? 1 : -1;
    const float ax = static_cast<float>(pond.water_radius_x_m);
    const float az = static_cast<float>(pond.water_radius_z_m);
    const float cx = cosf(azimuth);
    const float sz = sinf(azimuth);
    const physics::V radial{ax * cx, 0.f, az * sz};
    const physics::V normal{cx / ax, 0.f, sz / az};
    const auto outward = normal / physics::norm(normal);
    const auto tangent = physics::cross({0.f, 1.f, 0.f}, outward) * static_cast<float>(inward_side);
    const float slant = slant_deg * (pi / 180.f);
    const auto course = tangent * cosf(slant) + outward * sinf(slant);

    const auto& first = source.legs[0];
    const auto archived_course = chooser::horizontal(yaw_offset_turn(first.choice)
        ? source.motion.entry_course : first.heading);
    if (physics::norm(archived_course) < .5f) return false;
    relative_yaw_rad = atan2f(physics::cross(archived_course, course).y, physics::dot(archived_course, course));
    const auto rotation = physics::axis_rotation({0.f, 1.f, 0.f}, relative_yaw_rad);
    auto body = source.motion.body;
    auto pose = snapshot(body);
    const auto origin = chooser::horizontal(pose.com_m);
    const auto head_m = rotation.rotate(body.position * physics::scale - origin);
    for (auto& station : pose.spine_m) station = rotation.rotate(station - origin);
    const auto fixed_pose_gap = [&](physics::V com_horizontal_m) {
        body.position = (head_m + com_horizontal_m) / physics::scale;
        auto shifted = pose;
        for (auto& station : shifted.spine_m) station += com_horizontal_m;
        return submerged_gap_m(pond, body, shifted);
    };

    // Each pose component has a convex elliptical radius along a line.
    // Their maximum also has convex sublevel sets. Starting from clearance
    // at the origin, the first root along this radial line is unique.
    if (!(fixed_pose_gap({}) > 0.) || !(fixed_pose_gap(radial) <= 0.)) return false;
    float clear_radius = 0.f;
    float blocked_radius = 1.f;
    for (int i = 0; i < 18; ++i) {
        const float middle = (clear_radius + blocked_radius) * .5f;
        if (fixed_pose_gap(radial * middle) > 0.) clear_radius = middle;
        else blocked_radius = middle;
    }
    const auto end = radial * blocked_radius;
    position_m = end - course * runway;

    // A translated pose has the same depth at both endpoints. Convex
    // radii bound the static clearance along the entire warm-up segment.
    // The actual warm-up is still integrated and checked by the trainer.
    float warmup_gap = static_cast<float>(fmin(fixed_pose_gap(position_m),
        fixed_pose_gap(position_m + course * warmup_travel_m)));
    // A long tangential runway may leave a curved bank's clear region. Move
    // only the INITIAL training placement inward until both warmup endpoints
    // satisfy the unchanged certificate. No live world or request is modified.
    if (level < 2) for (int i = 0; i < 200 && warmup_gap <= min_warmup_gap_m; ++i) {
        position_m = position_m - outward * .001f;
        warmup_gap = static_cast<float>(fmin(fixed_pose_gap(position_m),
            fixed_pose_gap(position_m + course * warmup_travel_m)));
    }
    if (evidence) *evidence = {slant_deg, runway, warmup_gap, inward_side};
    return warmup_gap > min_warmup_gap_m;
}

// Close-tangent TRAINING placement, never an action or live correction. Bank
// azimuth, gap and gait phase vary independently of the fixed held-out starts.
CUDARL_HD inline bool close_start(uint32_t lane,uint32_t episode,const NavigationWorld& source,
    const pond_geometry_v1::Parameters& pond,physics::V& position_m,float& yaw){
    const uint32_t draw=rng::hash_u32(lane/2)^rng::hash_u32(episode+1);
    const float azimuth=2*pi*rng::uniform(26112503u,draw,0);
    const float wanted=.015f+.015f*rng::uniform(26112503u,draw,1);
    const float x=cosf(azimuth),z=sinf(azimuth);
    const physics::V radial{float(pond.water_radius_x_m)*x,0,float(pond.water_radius_z_m)*z};
    auto normal=physics::V{x/float(pond.water_radius_x_m),0,z/float(pond.water_radius_z_m)};
    normal=normal/physics::norm(normal);
    const auto tangent=physics::cross({0,1,0},normal)*(lane%2? -1.f:1.f);
    const auto& first=source.legs[0];const auto course=chooser::horizontal(yaw_offset_turn(first.choice)?source.motion.entry_course:first.heading);
    yaw=atan2f(physics::cross(course,tangent).y,physics::dot(course,tangent));
    const auto rotation=physics::axis_rotation({0,1,0},yaw);auto body=source.motion.body;auto pose=snapshot(body);
    const auto origin=chooser::horizontal(pose.com_m),head=rotation.rotate(body.position*physics::scale-origin);
    for(auto& point:pose.spine_m)point=rotation.rotate(point-origin);
    const auto gap=[&](physics::V p){body.position=(head+p)/physics::scale;auto moved=pose;for(auto& point:moved.spine_m)point+=p;return submerged_gap_m(pond,body,moved);};
    if(gap({})<wanted||gap(radial)>wanted)return false;
    float low=0,high=1;for(int k=0;k<22;++k){const float mid=.5f*(low+high);if(gap(radial*mid)>=wanted)low=mid;else high=mid;}
    position_m=radial*low;
    return gap(position_m)>=wanted&&gap(position_m+tangent*.016f)>.002;
}
} // namespace cudarl::flexible_tail::bank_training

