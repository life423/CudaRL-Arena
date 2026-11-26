#pragma once

#include "environment/pond_geometry_v1.h"
#include "physics/flexible_tail.h"

namespace cudarl::flexible_tail {

// A certified vertical gap for the fully submerged, contact-free model.
// Each segment's capsule is enclosed by the convex hull of its end balls.
// The head ellipsoid fits inside an 11 mm ball, and the tapered tail's
// cross sections fit inside 12 mm balls. No force or state change is applied.
CUDARL_HD inline double submerged_gap_m(const pond_geometry_v1::Parameters& pond,
                                        const physics::BodyState& body,
                                        const Snapshot& pose) {
    if (!pond_geometry_v1::valid_parameters(pond)) return -1;
    constexpr double min_shore_radius = 1 - .022 - .014 - .009;
    const double min_flat_radius = fmin(pond.flat_radius_x_m / pond.water_radius_x_m,
                                        pond.flat_radius_z_m / pond.water_radius_z_m);
    const double min_axis = fmin(pond.water_radius_x_m, pond.water_radius_z_m);
    double far_radius = 0, lowest_y = 1, highest_y = -1;
    const auto include_ball = [&](physics::V center, double radius) {
        const double x = center.x / pond.water_radius_x_m;
        const double z = center.z / pond.water_radius_z_m;
        far_radius = fmax(far_radius, sqrt(x*x + z*z) + radius/min_axis);
        lowest_y = fmin(lowest_y, center.y - radius);
        highest_y = fmax(highest_y, center.y + radius);
    };
    include_ball(body.position * physics::scale, .011);
    for (const auto& point : pose.spine_m) include_ball(point, .012);
    if (far_radius >= min_shore_radius) return -1;
    // Within the shore the slope coordinate increases with radial distance,
    // and decreases with both the flat-bed radius and shoreline radius.
    // Therefore these minimum radii bound bed height above the whole body.
    const double t = fmax(0., (far_radius - min_flat_radius) / (min_shore_radius - min_flat_radius));
    const double rise = t*t*t*(t*(t*6 - 15) + 10);
    const double bed_upper = pond.floor_y_m * (1 - rise);
    return fmin(lowest_y - bed_upper, -highest_y) - 1e-6;
}

} // namespace cudarl::flexible_tail
