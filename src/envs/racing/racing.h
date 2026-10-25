#ifndef CUDARL_ENVS_RACING_H
#define CUDARL_ENVS_RACING_H

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/hd.h"
#include "core/rng.h"
#include "cuda/cuda_memory.h"

// A car on a closed track, steered and throttled continuously. The dynamics are
// a kinematic bicycle model, integrated with Euler steps. The car senses the
// walls with a fan of raycasts. Reward is progress around the track, minus a
// small penalty per steering input and a larger one for crashing. The track is
// one fixed circuit, built on the host and shared by the GPU batch, the CPU
// showcase and the Godot view.
namespace cudarl::racing {

constexpr float kPi         = 3.14159265358979f;
constexpr float kTwoPi      = 2.f * kPi;
constexpr int   kRays       = 9;              // raycasts over the front 180 degrees
constexpr int   kObsDim     = kRays + 1;      // ray distances, then speed
constexpr int   kActionDim  = 2;              // steer, throttle; raw actions are clamped to [-1, 1]
constexpr int   kWallPoints = 64;             // points per wall; each wall is a closed loop
constexpr int   kMaxSteps   = 1000;           // time limit: 50 s of driving
constexpr float kDt         = 0.05f;
constexpr float kWheelbase  = 2.5f;
constexpr float kMaxSteer   = 0.5f;           // rad, at full steer
constexpr float kMaxAccel   = 6.f;            // m/s^2, at full throttle
constexpr float kDrag       = 0.4f;           // 1/s; top speed is kMaxAccel / kDrag
constexpr float kMaxSpeed   = kMaxAccel / kDrag;
constexpr float kCarRadius  = 1.f;            // a centre closer than this to a wall is a crash
constexpr float kStartSpeed = 5.f;
constexpr float kRayRange   = 40.f;
constexpr float kMeanRadius = 30.f;           // the centreline is about this far from the origin
constexpr float kHalfWidth  = 4.f;            // the walls sit this far either side of the centreline
constexpr float kProgressReward = 3.f;        // per radian around the origin: 0.1 per metre
constexpr float kSteerPenalty   = 0.02f;
constexpr float kCrashPenalty   = 5.f;

CUDARL_HD inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// --- Track -----------------------------------------------------------------------

struct Vec2 {
    float x, y;
};

// Each wall is a closed polyline: segment k joins point k and point (k + 1) % kWallPoints.
struct Track {
    Vec2 outer[kWallPoints];
    Vec2 inner[kWallPoints];
};

// The centreline is r = R(theta) in polar coordinates about the origin. Because
// it is star-shaped, the car's polar angle measures its progress around the
// track, and the walls can be offset radially without ever crossing.
CUDARL_HD inline float centre_radius(float theta) {
    return kMeanRadius + 6.f * sinf(2.f * theta) + 2.5f * cosf(3.f * theta);
}
CUDARL_HD inline float centre_radius_slope(float theta) { return 12.f * cosf(2.f * theta) - 7.5f * sinf(3.f * theta); }

inline Track make_track() {
    Track t{};
    for (int k = 0; k < kWallPoints; ++k) {
        const float theta = kTwoPi * static_cast<float>(k) / static_cast<float>(kWallPoints);
        const float r     = centre_radius(theta);
        t.outer[k] = Vec2{(r + kHalfWidth) * cosf(theta), (r + kHalfWidth) * sinf(theta)};
        t.inner[k] = Vec2{(r - kHalfWidth) * cosf(theta), (r - kHalfWidth) * sinf(theta)};
    }
    return t;
}

// --- Geometry ------------------------------------------------------------------------

CUDARL_HD inline float distance_to_segment(Vec2 p, Vec2 a, Vec2 b) {
    const float ex = b.x - a.x, ey = b.y - a.y;
    const float len2 = ex * ex + ey * ey;
    const float u    = clampf(((p.x - a.x) * ex + (p.y - a.y) * ey) / len2, 0.f, 1.f);
    const float dx = a.x + u * ex - p.x, dy = a.y + u * ey - p.y;
    return sqrtf(dx * dx + dy * dy);
}

// Distance along the ray from p in unit direction d to segment ab; `range` if it misses.
CUDARL_HD inline float ray_segment(Vec2 p, Vec2 d, Vec2 a, Vec2 b, float range) {
    const float ex = b.x - a.x, ey = b.y - a.y;
    const float denom = d.x * ey - d.y * ex;   // cross(d, e); zero when parallel
    if (fabsf(denom) < 1e-9f) return range;
    const float wx = a.x - p.x, wy = a.y - p.y;
    const float t = (wx * ey - wy * ex) / denom;   // distance along the ray
    const float s = (wx * d.y - wy * d.x) / denom; // position along the segment
    if (t < 0.f || t > range || s < 0.f || s > 1.f) return range;
    return t;
}

CUDARL_HD inline float distance_to_walls(const Track& track, Vec2 p) {
    float best = 1e30f;
    for (int k = 0; k < kWallPoints; ++k) {
        const int next = (k + 1) % kWallPoints;
        const float d_outer = distance_to_segment(p, track.outer[k], track.outer[next]);
        const float d_inner = distance_to_segment(p, track.inner[k], track.inner[next]);
        best = d_outer < best ? d_outer : best;
        best = d_inner < best ? d_inner : best;
    }
    return best;
}

// --- Car -----------------------------------------------------------------------------

struct State {
    float x, y;        // position, m
    float heading;     // rad
    float speed;       // m/s, in [0, kMaxSpeed]
    float last_angle;  // polar angle about the origin at the previous step
    float progress;    // rad travelled around the origin since the episode started
    int   steps;       // steps taken in the current episode
};

// Angle of ray r relative to the heading: the fan spans -90 to +90 degrees.
CUDARL_HD inline float ray_offset(int r) {
    return (static_cast<float>(r) - 0.5f * static_cast<float>(kRays - 1)) * (kPi / static_cast<float>(kRays - 1));
}

// Raycast distances (each divided by kRayRange), then speed (divided by kMaxSpeed).
CUDARL_HD inline void write_obs(const State& s, const Track& track, float* obs) {
    const Vec2 p{s.x, s.y};
    for (int r = 0; r < kRays; ++r) {
        const float a = s.heading + ray_offset(r);
        const Vec2  d{cosf(a), sinf(a)};
        float dist = kRayRange;
        for (int k = 0; k < kWallPoints; ++k) {
            const int next = (k + 1) % kWallPoints;
            const float d_outer = ray_segment(p, d, track.outer[k], track.outer[next], kRayRange);
            const float d_inner = ray_segment(p, d, track.inner[k], track.inner[next], kRayRange);
            dist = d_outer < dist ? d_outer : dist;
            dist = d_inner < dist ? d_inner : dist;
        }
        obs[r] = dist / kRayRange;
    }
    obs[kRays] = s.speed / kMaxSpeed;
}

CUDARL_HD inline float wrap_angle(float a) {
    if (a > kPi) return a - kTwoPi;
    if (a < -kPi) return a + kTwoPi;
    return a;
}

// Starts near the beginning of the centreline (polar angle 0), with a random
// offset, heading and speed. A pure function of (seed, env, episode).
CUDARL_HD inline State initial_state(uint32_t seed, uint32_t env, uint32_t episode) {
    auto u = [&](uint32_t i) { return rng::uniform(seed, env, episode * 3u + i); };
    const float r0 = centre_radius(0.f);
    // The centreline's heading at angle theta is theta + atan2(r, r'); here theta = 0.
    const float tangent = atan2f(r0, centre_radius_slope(0.f));
    State s{};
    s.x          = r0 + 2.f * (u(0) - 0.5f);
    s.y          = 2.f * (u(1) - 0.5f);
    s.heading    = tangent + 0.2f * (u(2) - 0.5f);
    s.speed      = kStartSpeed;
    s.last_angle = atan2f(s.y, s.x);
    s.progress   = 0.f;
    s.steps      = 0;
    return s;
}

// One Euler step of the bicycle model. The new speed sets how far the car moves.
CUDARL_HD inline void integrate(State& s, float steer, float throttle) {
    const float accel = kMaxAccel * throttle - kDrag * s.speed;
    s.speed           = clampf(s.speed + kDt * accel, 0.f, kMaxSpeed);
    s.heading += kDt * s.speed / kWheelbase * tanf(kMaxSteer * steer);
    s.x += kDt * s.speed * cosf(s.heading);
    s.y += kDt * s.speed * sinf(s.heading);
}

struct StepResult {
    float reward;
    bool  terminated;   // the car hit a wall
    bool  truncated;    // the time limit was reached
    float laps;         // laps covered when the episode ends (meaningful only then)
};

// Advances one car. When the episode ends, the final observation goes to
// final_obs and the car restarts as episode + 1.
CUDARL_HD inline StepResult step(State& s, const float* action, const Track& track, uint32_t seed, uint32_t env,
                                 uint32_t& episode, float* final_obs) {
    const float steer    = clampf(action[0], -1.f, 1.f);
    const float throttle = clampf(action[1], -1.f, 1.f);
    integrate(s, steer, throttle);

    const float angle = atan2f(s.y, s.x);
    const float delta = wrap_angle(angle - s.last_angle);
    s.last_angle = angle;
    s.progress += delta;
    s.steps += 1;

    StepResult r{};
    r.terminated = distance_to_walls(track, Vec2{s.x, s.y}) < kCarRadius;
    r.truncated  = !r.terminated && s.steps >= kMaxSteps;
    r.reward     = kProgressReward * delta - kSteerPenalty * steer * steer - (r.terminated ? kCrashPenalty : 0.f);
    r.laps       = s.progress / kTwoPi;
    if (r.terminated || r.truncated) {
        write_obs(s, track, final_obs);
        episode += 1;
        s = initial_state(seed, env, episode);
    }
    return r;
}

// --- Batch on the GPU ------------------------------------------------------------------

// Per-step outputs of a batch: reward, done, truncated and finished_* are [num_envs];
// final_obs is [num_envs x kObsDim].
struct StepOutputs {
    float* reward;
    float* done;
    float* truncated;
    float* final_obs;
    float* finished_length;
    float* finished_laps;
};

// num_envs cars stepped in parallel, one GPU thread each, on the shared track.
class RacingBatch {
public:
    RacingBatch(int num_envs, uint32_t seed);

    int num_envs() const { return n_; }
    void reset(cudaStream_t stream = nullptr);
    // d_actions: raw [num_envs x kActionDim] actions, clamped inside step().
    void step(const float* d_actions, const StepOutputs& out, cudaStream_t stream = nullptr);
    const float* obs() const { return obs_.get(); }   // device [num_envs x kObsDim]

private:
    int                  n_;
    uint32_t             seed_;
    device_ptr<State>    states_;
    device_ptr<uint32_t> episodes_;
    device_ptr<float>    obs_;
};

} // namespace cudarl::racing

#endif // CUDARL_ENVS_RACING_H
