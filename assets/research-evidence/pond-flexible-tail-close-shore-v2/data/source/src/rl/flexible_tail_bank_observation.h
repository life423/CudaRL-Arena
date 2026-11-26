#pragma once

#include "environment/flexible_tail_clearance.h"

namespace cudarl::flexible_tail::bank {

inline constexpr uint32_t schema = 1;
inline constexpr int inputs = 18;
inline constexpr float near_m = .08f, far_m = .24f, maximum_range_m = .5f;
inline constexpr float speed_scale_m_s = .008f, time_scale_s = 30.f;

enum Feature : int {
    CertifiedGap = 0, NormalForward = 1, NormalUp = 2, NormalRight = 3,
    HorizontalSpeed = 4, CourseValid = 5,
    AheadNear = 6, AheadFar = 7, LeftNear = 8, LeftFar = 9,
    RightNear = 10, RightFar = 11, AheadTransportRange = 12,
    ConstantTransportTime = 13, TransportTimeValid = 14, TerrainValid = 15,
    LocalCourseForward = 16, LocalCourseRight = 17
};

// Translate the present whole-body envelope without rotating or integrating it.
// This is sensory information only; it selects no action and changes no state.
CUDARL_HD inline float translated_gap(const physics::BodyState& source,
    const Snapshot& original, const pond_geometry_v1::Parameters& pond, physics::V translation) {
    auto body = source;
    auto pose = original;
    body.position += translation / physics::scale;
    for (auto& point : pose.spine_m) point += translation;
    return static_cast<float>(submerged_gap_m(pond, body, pose));
}

struct TransportRange {
    float distance_m = 0;
    bool bounded = false;
};

// A first loss of the conservative separation certificate along a straight
// translation. The body's current shape and orientation are held fixed.
// It is neither a collision/contact measurement nor a guaranteed future TTC.
CUDARL_HD inline TransportRange transport_range(const physics::BodyState& body,
    const Snapshot& pose, const pond_geometry_v1::Parameters& pond, physics::V direction) {
    if (!(submerged_gap_m(pond, body, pose) > 0)) return {};
    if (translated_gap(body, pose, pond, direction * maximum_range_m) > 0)
        return {maximum_range_m, false};
    float low = 0, high = maximum_range_m;
    for (int iteration = 0; iteration < 10; ++iteration) {
        const float mid = .5f * (low + high);
        if (translated_gap(body, pose, pond, direction * mid) > 0) low = mid;
        else high = mid;
    }
    return {low, true};
}

CUDARL_HD inline void observe(const physics::BodyState& body, const Snapshot& pose,
    const pond_geometry_v1::Parameters& pond, physics::V measured_com_velocity_m_s,
    bool measured_course_reliable, float (&out)[inputs]) {
    for (auto& feature : out) feature = 0.f;
    if (!pond_geometry_v1::valid_parameters(pond) || !physics::finite_body(body) ||
        !physics::finite_vector(pose.com_m) || !physics::finite_vector(measured_com_velocity_m_s)) return;
    const auto query = pond_geometry_v1::sample(pond, pose.com_m.x, pose.com_m.z);
    if (query.valid) {
        const auto normal = body.orientation.inverse({static_cast<float>(query.normal.x),
            static_cast<float>(query.normal.y), static_cast<float>(query.normal.z)});
        out[NormalForward] = normal.x; out[NormalUp] = normal.y; out[NormalRight] = normal.z;
        out[TerrainValid] = 1.f;
    }
    const float gap = static_cast<float>(submerged_gap_m(pond, body, pose));
    out[CertifiedGap] = physics::clamp(gap / .05f, -1.f, 1.f);
    measured_com_velocity_m_s.y = 0;
    const float speed = physics::norm(measured_com_velocity_m_s);
    const bool course_valid = measured_course_reliable && speed >= .003f;
    auto direction = course_valid ? measured_com_velocity_m_s / speed : body.orientation.rotate({1, 0, 0});
    direction.y = 0;
    const float length = physics::norm(direction);
    if (length < .0001f) return;
    direction = direction / length;
    const auto local_course = body.orientation.inverse(direction);
    out[HorizontalSpeed] = speed / speed_scale_m_s;
    out[CourseValid] = course_valid ? 1.f : 0.f;
    out[LocalCourseForward] = local_course.x; out[LocalCourseRight] = local_course.z;
    for (int ray = 0; ray < 3; ++ray) {
        const float angle = ray == 1 ? pi / 4.f : ray == 2 ? -pi / 4.f : 0.f;
        const auto bearing = physics::axis_rotation({0, 1, 0}, angle).rotate(direction);
        out[AheadNear + 2 * ray] = physics::clamp(translated_gap(body, pose, pond, bearing * near_m) / .05f, -1.f, 1.f);
        out[AheadFar + 2 * ray] = physics::clamp(translated_gap(body, pose, pond, bearing * far_m) / .05f, -1.f, 1.f);
    }
    const auto range = transport_range(body, pose, pond, direction);
    out[AheadTransportRange] = range.distance_m / maximum_range_m;
    if (course_valid && range.bounded && gap > 0) {
        out[ConstantTransportTime] = physics::clamp(range.distance_m / (speed * time_scale_s), 0.f, 1.f);
        out[TransportTimeValid] = 1.f;
    }
}

} // namespace cudarl::flexible_tail::bank
