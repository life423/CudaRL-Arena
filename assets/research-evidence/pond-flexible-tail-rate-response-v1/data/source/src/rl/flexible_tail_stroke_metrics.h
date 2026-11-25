#pragma once

#include "physics/flexible_tail.h"
#include <cstdint>

namespace cudarl::flexible_tail::stroke {

inline constexpr uint32_t metric_schema = 1;
inline constexpr float prominence_rad = .1f;
inline constexpr float passive_reference_rad = .2f;
inline constexpr float passive_reference_rate_rad_s = 4.f * pi;

struct Point {
    double time_s = 0;
    float angle_rad = 0;
    physics::V com_m{};
};

struct Cycle {
    double begin_s = 0, end_s = 0;
    float swing_rad = 0;
    physics::V begin_com{}, end_com{};
};

// Directional-change extrema: a reversal is confirmed only after the measured
// angle moves at least 0.1rad away from the extremum. Tiny flexible-tail ripples
// do not become principal strokes merely by crossing zero angular velocity.
// This is a distinct versioned metric; it never replaces historical scores.
struct MainStrokeTracker {
    bool initialized = false, have_minimum = false;
    int direction = 0;
    Point low{}, high{}, previous_minimum{};
    float cycle_high_rad = -1e30f;

    CUDARL_HD bool add(double time_s, float angle_rad, physics::V com_m, Cycle& out) {
        const Point point{time_s, angle_rad, com_m};
        if (!initialized) {
            initialized = true;
            low = high = point;
            return false;
        }
        if (direction == 0) {
            if (angle_rad < low.angle_rad) low = point;
            if (angle_rad > high.angle_rad) high = point;
            if (angle_rad - low.angle_rad >= prominence_rad) {
                direction = 1;
                high = point;
            } else if (high.angle_rad - angle_rad >= prominence_rad) {
                direction = -1;
                low = point;
            }
            return false;
        }
        if (direction > 0) {
            if (angle_rad > high.angle_rad) high = point;
            if (high.angle_rad - angle_rad < prominence_rad) return false;
            cycle_high_rad = fmaxf(cycle_high_rad, high.angle_rad);
            direction = -1;
            low = point;
            return false;
        }
        if (angle_rad < low.angle_rad) low = point;
        if (angle_rad - low.angle_rad < prominence_rad) return false;
        bool completed = false;
        if (have_minimum && low.time_s > previous_minimum.time_s) {
            out = {previous_minimum.time_s, low.time_s,
                cycle_high_rad - fminf(previous_minimum.angle_rad, low.angle_rad),
                previous_minimum.com_m, low.com_m};
            completed = out.swing_rad >= prominence_rad;
        }
        previous_minimum = low;
        have_minimum = true;
        cycle_high_rad = -1e30f;
        direction = 1;
        high = point;
        return completed;
    }
};

// Causal, bounded physical feedback. No future window, reversal count, reset,
// reference trajectory, or artificial force enters the reward feature.
CUDARL_HD inline float passive_quality(const physics::BodyState& body) {
    const float q = body.angle[2];
    const float rate = body.velocity[8] / passive_reference_rate_rad_s;
    return physics::clamp((q * q + rate * rate) /
        (passive_reference_rad * passive_reference_rad), 0.f, 1.f);
}

} // namespace cudarl::flexible_tail::stroke
