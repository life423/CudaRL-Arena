#pragma once

#include "environment/flexible_tail_clearance.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>

namespace bank_replay {

namespace physics = cudarl::physics;
namespace tail = cudarl::flexible_tail;
namespace geometry = cudarl::pond_geometry_v1;
using physics::V;

struct Sample {
    int sample = 0;
    V com_m, head_m;
    std::array<V, tail::stations + 1> spine_m{};
    float root_yaw_rad = 0, passive_yaw_rad = 0;
};

namespace detail {

constexpr int samples_per_second = 800;
constexpr int turn_samples = 12 * samples_per_second;
constexpr int onward_samples = 14 * samples_per_second;
constexpr int trace_samples = 16 * samples_per_second;

struct Clearance {
    size_t limiting_index = 0;
    double gap_m = std::numeric_limits<double>::infinity();
};

inline void point(std::ostream& out, V position) {
    out << '[' << position.x << ',' << position.y << ',' << position.z << ']';
}

inline void write_gap(std::ostream& out, double gap_m) {
    if (std::isfinite(gap_m) && gap_m != -1.) out << 1000 * gap_m;
    else out << "null";
}

inline bool valid_trace(const std::vector<Sample>& samples) {
    if (samples.size() != trace_samples + 1) return false;
    const int initial_sample = samples.front().sample;
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        if (sample.sample != initial_sample + static_cast<int>(i) ||
            !physics::finite_vector(sample.com_m) || !physics::finite_vector(sample.head_m) ||
            !physics::finite_number(sample.root_yaw_rad) || !physics::finite_number(sample.passive_yaw_rad) ||
            !std::all_of(sample.spine_m.begin(), sample.spine_m.end(), physics::finite_vector)) return false;
    }
    return true;
}

} // namespace detail

inline void write_estimate(const std::filesystem::path& output, const geometry::Parameters& pond,
                           const std::vector<Sample>& samples, V approach_heading, float turn_degrees) {
    std::ofstream out(output / "east-bank.json");
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::setprecision(10)
        << "{\n  \"method\": \"offline native whole-body clearance envelope\","
        << "\n  \"dynamics_steps\": 0,"
        << "\n  \"samples\": " << samples.size();

    const auto east = geometry::outline(pond, 0.);
    approach_heading.y = 0;
    if (!east.valid || !detail::valid_trace(samples) || !physics::finite_vector(approach_heading) ||
        physics::norm(approach_heading) < 1e-6f || std::abs(turn_degrees) != 15.f) {
        out << ",\n  \"status\": \"invalid_trace_or_placement\",\n  \"qualifying_setback_mm\": null\n}\n";
        return;
    }

    const V shore{float(east.point_m.x), 0.f, 0.f};
    V tangent{float(pond.water_radius_x_m * geometry::edge_derivative(0.)), 0.f,
              float(pond.water_radius_z_m * east.edge_factor)};
    tangent = tangent / physics::norm(tangent);
    const V outward{tangent.z, 0.f, -tangent.x};
    if (turn_degrees > 0.f) tangent = tangent * -1.f;
    constexpr float angle_rad = float(7.5 * geometry::pi / 180.);
    const V incoming = tangent * cosf(angle_rad) + outward * sinf(angle_rad);
    const V expected_onward = physics::axis_rotation({0.f, 1.f, 0.f},
        turn_degrees * float(geometry::pi / 180.)).rotate(incoming);
    const float rotation_rad = atan2f(physics::cross(approach_heading, incoming).y,
                                     physics::dot(approach_heading, incoming));
    const auto rotation = physics::axis_rotation({0.f, 1.f, 0.f}, rotation_rad);
    const V origin{samples.front().com_m.x, 0.f, samples.front().com_m.z};
    const auto place_at_edge = [&](V point) { return shore + rotation.rotate(point - origin); };

    const V onward = rotation.rotate(samples[detail::onward_samples].com_m - samples[detail::turn_samples].com_m);
    const double inward_displacement_mm = -1000 * double(physics::dot(onward, outward));
    const bool moves_inward = inward_displacement_mm > 0.;

    // The certificate bounds radial extent with normalized ellipse radius plus
    // a constant ball radius. Each such radius decreases along the inward
    // line until its nearest point to the pond centre. Limit the search to
    // the earliest such point over every head and tail sample.
    const double metric_x = 1. / (pond.water_radius_x_m * pond.water_radius_x_m);
    const double metric_z = 1. / (pond.water_radius_z_m * pond.water_radius_z_m);
    const double nx = outward.x, nz = outward.z;
    const double metric_norm = nx * nx * metric_x + nz * nz * metric_z;
    double monotone_bound_m = std::numeric_limits<double>::infinity();
    const auto bound = [&](V point) {
        const double closest_m = (double(point.x) * nx * metric_x + double(point.z) * nz * metric_z) /
            metric_norm;
        monotone_bound_m = std::min(monotone_bound_m, closest_m);
    };
    std::vector<Sample> placed;
    placed.reserve(samples.size());
    for (const auto& source : samples) {
        Sample sample = source;
        sample.com_m = place_at_edge(source.com_m);
        sample.head_m = place_at_edge(source.head_m);
        bound(sample.head_m);
        for (size_t i = 0; i < sample.spine_m.size(); ++i) {
            sample.spine_m[i] = place_at_edge(source.spine_m[i]);
            bound(sample.spine_m[i]);
        }
        placed.push_back(sample);
    }
    const int upper_mm = static_cast<int>(std::max(0., std::floor(1000 * monotone_bound_m)));
    const auto clearance = [&](int setback_mm) {
        detail::Clearance result;
        const V translation = outward * (-float(setback_mm) / 1000.f);
        for (size_t i = 0; i < placed.size(); ++i) {
            const auto& sample = placed[i];
            physics::BodyState body;
            body.position = (sample.head_m + translation) / physics::scale;
            tail::Snapshot pose;
            for (size_t j = 0; j < sample.spine_m.size(); ++j) pose.spine_m[j] = sample.spine_m[j] + translation;
            const double gap_m = tail::submerged_gap_m(pond, body, pose);
            if (gap_m < result.gap_m) result = {i, gap_m};
        }
        return result;
    };

    const auto edge_clearance = clearance(0);
    const auto upper_clearance = clearance(upper_mm);
    int failed_mm = 0, clear_mm = upper_mm;
    auto failed = edge_clearance;
    auto clear = upper_clearance;
    const bool bracketed = upper_mm > 0 && failed.gap_m <= 0. && clear.gap_m > 0.;
    if (bracketed) {
        while (clear_mm - failed_mm > 1) {
            const int mid_mm = failed_mm + (clear_mm - failed_mm) / 2;
            const auto mid = clearance(mid_mm);
            if (mid.gap_m > 0.) {
                clear_mm = mid_mm;
                clear = mid;
            } else {
                failed_mm = mid_mm;
                failed = mid;
            }
        }
    }
    const bool qualified = bracketed && moves_inward;
    out << ",\n  \"status\": \"" << (qualified ? "qualified_geometry" : bracketed
        ? "clearance_without_inward_travel" : "no_certified_1mm_bracket") << "\","
        << "\n  \"turn_degrees\": " << turn_degrees << ",\n  \"shore_point_m\": ";
    detail::point(out, shore);
    out << ",\n  \"outward_normal\": ";
    detail::point(out, outward);
    out << ",\n  \"incoming_course\": ";
    detail::point(out, incoming);
    out << ",\n  \"requested_onward_course\": ";
    detail::point(out, expected_onward);
    out << ",\n  \"cue_com_depth_mm\": " << -1000 * double(samples.front().com_m.y)
        << ",\n  \"onward_inward_com_mm\": " << inward_displacement_mm
        << ",\n  \"search_upper_mm\": " << upper_mm
        << ",\n  \"search_upper_gap_mm\": ";
    detail::write_gap(out, upper_clearance.gap_m);
    out << ",\n  \"edge_gap_mm\": ";
    detail::write_gap(out, edge_clearance.gap_m);
    out << ",\n  \"qualifying_setback_mm\": ";
    if (qualified) out << clear_mm;
    else out << "null";
    out << ",\n  \"clearance_bracket\": ";
    if (!bracketed) {
        out << "null\n}\n";
        return;
    }
    out << "{\n    \"failing_setback_mm\": " << failed_mm
        << ",\n    \"failing_min_gap_mm\": ";
    detail::write_gap(out, failed.gap_m);
    out << ",\n    \"certified_setback_mm\": " << clear_mm
        << ",\n    \"minimum_certified_gap_mm\": " << 1000 * clear.gap_m
        << "\n  },\n  \"limiting_pose\": {";
    const auto& limit = placed[clear.limiting_index];
    const V translation = outward * (-float(clear_mm) / 1000.f);
    out << "\n    \"sample\": " << limit.sample
        << ",\n    \"seconds_from_cue\": " << double(clear.limiting_index) / detail::samples_per_second
        << ",\n    \"root_yaw_rad\": " << limit.root_yaw_rad
        << ",\n    \"passive_yaw_rad\": " << limit.passive_yaw_rad
        << ",\n    \"com_m\": ";
    detail::point(out, limit.com_m + translation);
    out << ",\n    \"head_m\": ";
    detail::point(out, limit.head_m + translation);
    out << ",\n    \"spine_m\": [";
    for (size_t i = 0; i < limit.spine_m.size(); ++i) {
        if (i) out << ',';
        detail::point(out, limit.spine_m[i] + translation);
    }
    out << "]\n  }\n}\n";
}

} // namespace bank_replay
