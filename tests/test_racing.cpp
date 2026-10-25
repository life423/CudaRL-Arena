// The racing car against hand-computed and analytic references: wall geometry,
// raycasts on a circle, crashes and the time limit, the bicycle model's first
// step, and the GPU batch against the host step on the same random actions.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "envs/racing/racing.h"
#include "gpu_util.h"

using namespace cudarl;
using namespace cudarl::racing;
using cudarl::test::to_host;

namespace {

// A track whose outer wall is a circle of radius R about the origin. Its inner
// wall is a tiny loop far away, so only the circle matters to the rays.
Track circle_track(float R) {
    Track t{};
    for (int k = 0; k < kWallPoints; ++k) {
        const float theta = kTwoPi * static_cast<float>(k) / kWallPoints;
        t.outer[k] = Vec2{R * cosf(theta), R * sinf(theta)};
        t.inner[k] = Vec2{100.f + 0.5f * cosf(theta), 0.5f * sinf(theta)};
    }
    return t;
}

State car_at(float x, float y, float heading, float speed) {
    State s{};
    s.x = x;
    s.y = y;
    s.heading = heading;
    s.speed = speed;
    s.last_angle = atan2f(y, x);
    return s;
}

} // namespace

TEST_CASE("racing: the track is a band around the centreline") {
    const Track track = make_track();
    const float r0 = centre_radius(0.f);
    // At angle 0 the centreline is at (r0, 0); the nearest wall is about kHalfWidth away.
    const float d = distance_to_walls(track, Vec2{r0, 0.f});
    CHECK(d > kHalfWidth - 0.5f && d < kHalfWidth + 0.5f);
    // Every wall point lies on its radial line at the expected offset.
    for (int k = 0; k < kWallPoints; ++k) {
        const float theta = kTwoPi * static_cast<float>(k) / kWallPoints;
        const float ro = std::hypot(track.outer[k].x, track.outer[k].y);
        const float ri = std::hypot(track.inner[k].x, track.inner[k].y);
        CHECK(std::fabs(ro - (centre_radius(theta) + kHalfWidth)) < 1e-4f);
        CHECK(std::fabs(ri - (centre_radius(theta) - kHalfWidth)) < 1e-4f);
    }
    // The track is star-shaped, so the centreline's radius stays positive and the inner wall stays outside the origin.
    for (int k = 0; k < 360; ++k) {
        CHECK(centre_radius(kTwoPi * k / 360.f) - kHalfWidth > 10.f);
    }
}

TEST_CASE("racing: raycasts on a circle read its radius in every direction") {
    const Track track = circle_track(10.f);
    State s = car_at(0.f, 0.f, 0.3f, 0.f);
    float obs[kObsDim];
    write_obs(s, track, obs);
    // The circle's edges are within 0.02 of the true radius; allow for the polygon.
    for (int r = 0; r < kRays; ++r) CHECK(std::fabs(obs[r] * kRayRange - 10.f) < 0.05f);
    CHECK(obs[kRays] == 0.f);   // speed zero
}

TEST_CASE("racing: an off-centre car sees the nearer wall shorter") {
    const Track track = circle_track(10.f);
    State s = car_at(3.f, 0.f, 0.f, 5.f);
    float obs[kObsDim];
    write_obs(s, track, obs);
    // Straight ahead (ray 4) the wall is 10 - 3 = 7 m away.
    CHECK(std::fabs(obs[4] * kRayRange - 7.f) < 0.05f);
    // Straight behind (ray 0 is -90 degrees, ray 8 is +90), the wall is sqrt(100 - 9) away.
    CHECK(std::fabs(obs[0] * kRayRange - std::sqrt(91.f)) < 0.05f);
    CHECK(std::fabs(obs[kRays] - 5.f / kMaxSpeed) < 1e-6f);
}

TEST_CASE("racing: one bicycle step matches the hand computation") {
    // Straight ahead at 5 m/s, no throttle: drag slows the car by 0.05 * 0.4 * 5 = 0.1 m/s.
    State s = car_at(0.f, 0.f, 0.f, 5.f);
    integrate(s, 0.f, 0.f);
    CHECK(std::fabs(s.speed - 4.9f) < 1e-5f);
    CHECK(std::fabs(s.x - 0.05f * 4.9f) < 1e-5f);
    CHECK(std::fabs(s.y) < 1e-6f);
    CHECK(std::fabs(s.heading) < 1e-6f);

    // Full steer turns the heading by dt * v / L * tan(max steer).
    State t = car_at(0.f, 0.f, 0.f, 5.f);
    integrate(t, 1.f, 1.f);
    const float v = t.speed;
    CHECK(std::fabs(t.heading - 0.05f * v / kWheelbase * std::tan(kMaxSteer)) < 1e-5f);
    // Speed is clamped to [0, kMaxSpeed].
    State fast = car_at(0.f, 0.f, 0.f, kMaxSpeed);
    integrate(fast, 0.f, 1.f);
    CHECK(fast.speed <= kMaxSpeed + 1e-5f);
    State slow = car_at(0.f, 0.f, 0.f, 0.f);
    integrate(slow, 0.f, -1.f);
    CHECK(slow.speed == 0.f);
}

TEST_CASE("racing: crashing ends the episode with the crash penalty and restarts the car") {
    const Track track = circle_track(10.f);
    State s = car_at(9.5f, 0.f, 0.f, 0.f);   // half a metre from the wall, closer than the car radius
    float action[kActionDim] = {0.f, 0.f};
    float final_obs[kObsDim] = {};
    uint32_t episode = 0;
    const StepResult r = step(s, action, track, 7, 3, episode, final_obs);
    CHECK(r.terminated);
    CHECK(!r.truncated);
    CHECK(std::fabs(r.reward + kCrashPenalty) < 1e-5f);   // no progress, no steering
    CHECK(episode == 1);
    CHECK(s.steps == 0);                                  // a fresh episode
    // The final observation is the crash position: the wall dead ahead (ray 4) is 0.5 m away.
    CHECK(std::fabs(final_obs[4] * kRayRange - 0.5f) < 0.05f);
}

TEST_CASE("racing: the time limit truncates an episode without a crash") {
    const Track track = circle_track(10.f);
    State s = car_at(0.f, 0.f, 0.f, 0.f);   // parked in the middle: never crashes
    float action[kActionDim] = {0.f, 0.f};
    float final_obs[kObsDim] = {};
    uint32_t episode = 0;
    for (int t = 0; t < kMaxSteps - 1; ++t) {
        const StepResult r = step(s, action, track, 7, 0, episode, final_obs);
        CHECK(!r.terminated && !r.truncated);
    }
    const StepResult last = step(s, action, track, 7, 0, episode, final_obs);
    CHECK(last.truncated);
    CHECK(!last.terminated);
    CHECK(episode == 1);
}

TEST_CASE("racing: progress is the polar angle travelled, and reward pays for it") {
    const Track track = make_track();
    State s = initial_state(5, 0, 0);
    const float start_angle = s.last_angle;
    float action[kActionDim] = {0.1f, 0.5f};
    float final_obs[kObsDim] = {};
    uint32_t episode = 0;
    const StepResult r = step(s, action, track, 5, 0, episode, final_obs);
    CHECK(!r.terminated);
    const float travelled = atan2f(s.y, s.x) - start_angle;
    CHECK(std::fabs(s.progress - travelled) < 1e-5f);
    CHECK(std::fabs(r.reward - (kProgressReward * travelled - kSteerPenalty * 0.1f * 0.1f)) < 1e-4f);
}

TEST_CASE("racing: a car steered into the wall crashes on the real track") {
    const Track track = make_track();
    State s = initial_state(9, 0, 0);
    float action[kActionDim] = {1.f, 1.f};   // full left lock at full throttle: it leaves the track
    float final_obs[kObsDim] = {};
    uint32_t episode = 0;
    bool crashed = false;
    for (int t = 0; t < 200 && !crashed; ++t) {
        const StepResult r = step(s, action, track, 9, 0, episode, final_obs);
        crashed = r.terminated;
    }
    CHECK(crashed);
}

GPU_TEST_CASE("racing: GPU batch matches the host step on the same actions") {
    const int n = 64, steps = 100;
    const uint32_t seed = 11;
    RacingBatch batch(n, seed);
    const Track track = make_track();

    // Host copies of every car, started exactly as the GPU starts them.
    std::vector<State> host(n);
    std::vector<uint32_t> episodes(n, 0);
    for (int i = 0; i < n; ++i) host[i] = initial_state(seed, static_cast<uint32_t>(i), 0);

    device_ptr<float> d_actions(static_cast<size_t>(n) * kActionDim), d_reward(n), d_done(n), d_trunc(n),
        d_final(static_cast<size_t>(n) * kObsDim), d_len(n), d_laps(n);
    StepOutputs out{d_reward.get(), d_done.get(), d_trunc.get(), d_final.get(), d_len.get(), d_laps.get()};

    int disagreements = 0;
    for (int t = 0; t < steps; ++t) {
        // Raw actions outside [-1, 1] on purpose: both sides clamp them.
        std::vector<float> actions(static_cast<size_t>(n) * kActionDim);
        for (size_t k = 0; k < actions.size(); ++k) {
            actions[k] = 3.f * rng::uniform(21, static_cast<uint32_t>(t), static_cast<uint32_t>(k)) - 1.5f;
        }
        d_actions.copy_from_host(actions.data(), actions.size());
        batch.step(d_actions.get(), out);
        const auto reward = to_host(d_reward, n);
        const auto done   = to_host(d_done, n);

        for (int i = 0; i < n; ++i) {
            float final_obs[kObsDim];
            const StepResult r = step(host[i], &actions[static_cast<size_t>(i) * kActionDim], track, seed,
                                      static_cast<uint32_t>(i), episodes[i], final_obs);
            const bool host_done = r.terminated || r.truncated;
            if (host_done != (done[i] > 0.5f)) ++disagreements;
            CHECK_MSG(std::fabs(reward[i] - r.reward) < 1e-3f, "reward at env " + std::to_string(i) + " step " +
                                                                  std::to_string(t));
        }
    }
    // A single differing crash decision would desynchronize that car forever, so require none.
    CHECK_MSG(disagreements == 0, "done flags differ from the host in " + std::to_string(disagreements) + " steps");

    // After the last step, the observations match too.
    std::vector<float> host_obs(static_cast<size_t>(n) * kObsDim);
    for (int i = 0; i < n; ++i) write_obs(host[i], track, &host_obs[static_cast<size_t>(i) * kObsDim]);
    CHECK_CLOSE(to_host(batch.obs(), host_obs.size()), host_obs, 1e-3);
}
