#ifndef CUDARL_RACING_RACE_H
#define CUDARL_RACING_RACE_H

#include "racing.h"

namespace cudarl::racing {

// Track distance expressed in radians for compatibility with the PPO lap metric.
CUDARL_HD inline float track_position(const Track &t, Vec2 p) {
    float best = 1e30f, arc = 0.f;
    for (int k = 0; k < kWallPoints; ++k) {
        const Vec2 a = t.centre[k], b = t.centre[(k + 1) % kWallPoints];
        const float dx = b.x - a.x, dy = b.y - a.y, len2 = dx * dx + dy * dy;
        const float u = clampf(((p.x - a.x) * dx + (p.y - a.y) * dy) / len2, 0.f, 1.f);
        const float ex = a.x + u * dx - p.x, ey = a.y + u * dy - p.y;
        if (ex * ex + ey * ey < best) {
            best = ex * ex + ey * ey;
            arc = t.arc[k] + u * (t.arc[k + 1] - t.arc[k]);
        }
    }
    return kTwoPi * arc / t.arc[kWallPoints];
}

CUDARL_HD inline State grid_state(const Track &t, int slot, uint32_t seed, uint32_t race, uint32_t episode) {
    // Even arc-length grid, randomized rotation per race/episode, never random
    // jitter that can put a spawn into a neighbour or a wall.
    const float offset = rng::uniform(seed, race, episode) * t.arc[kWallPoints];
    float arc = offset + static_cast<float>(slot) * t.arc[kWallPoints] / 64.f;
    if (arc >= t.arc[kWallPoints])
        arc -= t.arc[kWallPoints];
    int k = 0;
    while (k < kWallPoints - 1 && t.arc[k + 1] < arc)
        ++k;
    const Vec2 a = t.centre[k], b = t.centre[(k + 1) % kWallPoints];
    const float u = (arc - t.arc[k]) / (t.arc[k + 1] - t.arc[k]);
    State s{};
    s.x = a.x + u * (b.x - a.x);
    s.y = a.y + u * (b.y - a.y);
    s.heading = atan2f(b.y - a.y, b.x - a.x);
    s.speed = kStartSpeed;
    s.last_angle = track_position(t, {s.x, s.y});
    return s;
}

CUDARL_HD inline float ray_car(Vec2 p, Vec2 d, const State &other) {
    const float x = other.x - p.x, y = other.y - p.y;
    const float along = x * d.x + y * d.y;
    const float disc = kTrafficRadius * kTrafficRadius - (x * x + y * y - along * along);
    if (disc < 0.f || along < 0.f)
        return kRayRange;
    return clampf(along - sqrtf(disc), 0.f, kRayRange);
}

CUDARL_HD inline void observe_race(const State *cars, int i, const Track &t, float *obs) {
    write_obs(cars[i], t, obs);
    const State &s = cars[i];
    for (int r = 0; r < kRays; ++r) {
        const float a = s.heading + ray_offset(r);
        const Vec2 d{cosf(a), sinf(a)};
        float dist = kRayRange, closing = 0.f;
        for (int j = 0; j < kRaceCars; ++j)
            if (j != i) {
                const float hit = ray_car({s.x, s.y}, d, cars[j]);
                // Walls occlude cars. Zero relative speed is reserved for no visible car.
                if (hit < dist && hit < obs[r] * kRayRange) {
                    dist = hit;
                    closing = ((s.speed * cosf(s.heading) - cars[j].speed * cosf(cars[j].heading)) * d.x +
                               (s.speed * sinf(s.heading) - cars[j].speed * sinf(cars[j].heading)) * d.y) /
                              kMaxSpeed;
                }
            }
        obs[kRays + 1 + r] = dist / kRayRange;
        obs[2 * kRays + 1 + r] = closing;
    }
    float best = 1e30f, heading = s.heading;
    for (int k = 0; k < kWallPoints; ++k) {
        const Vec2 a = t.centre[k], b = t.centre[(k + 1) % kWallPoints];
        const float d = distance_to_segment({s.x, s.y}, a, b);
        if (d < best) {
            best = d;
            heading = atan2f(b.y - a.y, b.x - a.x);
        }
    }
    obs[kObsDim - 2] = sinf(heading - s.heading);
    obs[kObsDim - 1] = cosf(heading - s.heading);
}

CUDARL_HD inline bool swept_contact(const State &a, const State &an, const State &b, const State &bn) {
    const float x = a.x - b.x, y = a.y - b.y;
    const float dx = (an.x - a.x) - (bn.x - b.x), dy = (an.y - a.y) - (bn.y - b.y);
    const float v2 = dx * dx + dy * dy;
    const float u = v2 > 1e-12f ? clampf(-(x * dx + y * dy) / v2, 0.f, 1.f) : 0.f;
    return (x + u * dx) * (x + u * dx) + (y + u * dy) * (y + u * dy) <= 4.f * kTrafficRadius * kTrafficRadius;
}

CUDARL_HD inline StepResult advance_race(State &s, const float *action, const Track &t) {
    const float steer = clampf(action[0], -1.f, 1.f), throttle = clampf(action[1], -1.f, 1.f);
    const float old_progress = s.progress;
    // Maximum 6 m/s^2 lateral acceleration: corner entry now requires braking.
    const float next_speed = clampf(s.speed + kDt * (kMaxAccel * throttle - kDrag * s.speed), 0.f, kMaxSpeed);
    const float limit = atanf(6.f * kWheelbase / (next_speed * next_speed + 1e-6f)) / kMaxSteer;
    integrate(s, clampf(steer, -limit, limit), throttle);
    const float pos = track_position(t, {s.x, s.y});
    const float delta = wrap_angle(pos - s.last_angle);
    s.last_angle = pos;
    s.progress += delta;
    ++s.steps;
    // Wall collision approximates a capsule with front, centre and rear samples.
    bool wall = false;
    for (int n = -1; n <= 1; ++n) {
        const Vec2 p{s.x + n * cosf(s.heading), s.y + n * sinf(s.heading)};
        wall = wall || distance_to_walls(t, p) < 1.05f;
    }
    StepResult result{};
    result.terminated = wall;
    result.truncated = !wall && s.steps >= kMaxSteps;
    // Pay for signed route progress; an idle step costs reward. Parking before
    // a bend must not become a better strategy than learning to negotiate it.
    result.reward = 12.f * delta - 0.002f * steer * steer - 0.01f - (wall ? kCrashPenalty : 0.f);
    if (s.progress >= kTwoPi && old_progress < kTwoPi)
        result.reward += 2.f;
    result.laps = s.progress / kTwoPi;
    result.clean_lap = !wall && static_cast<int>(s.progress / kTwoPi) > static_cast<int>(old_progress / kTwoPi);
    if (result.clean_lap) {
        result.lap_seconds = (s.steps - s.lap_step) * kDt;
        s.lap_step = s.steps;
    }
    return result;
}

CUDARL_HD inline void respawn_race(State *cars, const StepResult *results, uint32_t *episodes,
                                  const Track &track, uint32_t seed, uint32_t race) {
    // Remove all ended cars before reinserting any of them; every respawn sees
    // the final survivor positions and previously inserted respawns.
    bool ended[kRaceCars];
    for (int i = 0; i < kRaceCars; ++i) {
        ended[i] = results[i].terminated || results[i].truncated;
        if (ended[i]) {
            cars[i].x = 10000.f;
            cars[i].y = 10000.f;
        }
    }
    for (int i = 0; i < kRaceCars; ++i)
        if (ended[i]) {
            ++episodes[i];
            for (int slot = 0; slot < 64; ++slot) {
                const State candidate = grid_state(track, slot, seed, race, episodes[i]);
                bool clear = true;
                for (int j = 0; j < kRaceCars; ++j)
                    if (i != j) {
                        const float dx = candidate.x - cars[j].x, dy = candidate.y - cars[j].y;
                        clear =
                            clear && dx * dx + dy * dy > (2.f * kTrafficRadius + 0.8f) * (2.f * kTrafficRadius + 0.8f);
                    }
                if (clear && distance_to_walls(track, {candidate.x, candidate.y}) > 1.5f) {
                    cars[i] = candidate;
                    break;
                }
            }
        }
}

// Deterministic race resolution shared by CUDA and the CPU showcase. Each car's
// own terminal flag/penalty is independent; safe cars keep driving.
CUDARL_HD inline void resolve_race(State *cars, const State *old, StepResult *results, uint32_t *episodes,
                                   const Track &track, uint32_t seed, uint32_t race, float *final_obs) {
    bool contacts[kRaceCars] = {};
    for (int i = 0; i < kRaceCars; ++i)
        for (int j = i + 1; j < kRaceCars; ++j) {
            if (swept_contact(old[i], cars[i], old[j], cars[j]))
                contacts[i] = contacts[j] = true;
        }
    for (int i = 0; i < kRaceCars; ++i) {
        if (contacts[i]) {
            if (!results[i].terminated)
                results[i].reward -= kCrashPenalty;
            results[i].terminated = true;
            results[i].truncated = false;
            results[i].contact = true;
            results[i].clean_lap = false;
            results[i].lap_seconds = 0.f;
        }
        if (results[i].terminated || results[i].truncated)
            observe_race(cars, i, track, final_obs + i * kObsDim);
    }
    respawn_race(cars, results, episodes, track, seed, race);
}

CUDARL_HD inline void reset_race(State *cars, uint32_t *episodes, const Track &t, uint32_t seed, uint32_t race) {
    for (int i = 0; i < kRaceCars; ++i) {
        cars[i] = grid_state(t, i * 8, seed, race, 0);
        episodes[i] = 0;
    }
}

} // namespace cudarl::racing
#endif
