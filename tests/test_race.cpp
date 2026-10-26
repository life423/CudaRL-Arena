#include <cmath>
#include <vector>
#include "check.h"
#include "gpu_util.h"
#include "envs/racing/race.h"
#include "rl/racing_ppo.h"
#include "demos/racing_showcase.h"

using namespace cudarl;
using namespace cudarl::racing;

namespace {
void check_separation(const State *s) {
    for (int i = 0; i < kRaceCars; ++i) {
        CHECK(std::isfinite(s[i].x));
        CHECK(std::fabs(s[i].x) < 100.f);
        for (int j = i + 1; j < kRaceCars; ++j) {
            const float dx = s[i].x - s[j].x, dy = s[i].y - s[j].y;
            CHECK_MSG(dx * dx + dy * dy >= 4.f * kTrafficRadius * kTrafficRadius - 1e-3f, "cars overlap");
        }
    }
}
} // namespace

TEST_CASE("race: staggered grid is on track and separated for many seeds") {
    const Track t = make_track(true);
    CHECK(t.arc[kWallPoints] > 250.f);
    for (uint32_t seed = 1; seed <= 64; ++seed) {
        State s[kRaceCars];
        uint32_t ep[kRaceCars];
        reset_race(s, ep, t, seed, seed * 7);
        check_separation(s);
        for (const auto &c : s)
            CHECK(distance_to_walls(t, {c.x, c.y}) > 1.5f);
    }
}

TEST_CASE("race: swept contact catches crossing paths even when endpoints are clear") {
    State a{}, b{}, an{}, bn{};
    a.x = -10.f;
    b.x = 10.f;
    an.x = 10.f;
    bn.x = -10.f;
    CHECK(swept_contact(a, an, b, bn));
    b.y = bn.y = 5.f;
    CHECK(!swept_contact(a, an, b, bn));
}

TEST_CASE("race: the hard circuit walls never cross themselves or each other") {
    const Track t = make_track(true);
    for (int w = 0; w < 2; ++w)
        for (int v = w; v < 2; ++v) {
            const Vec2 *a = w ? t.inner : t.outer;
            const Vec2 *b = v ? t.inner : t.outer;
            for (int i = 0; i < kWallPoints; ++i)
                for (int j = 0; j < kWallPoints; ++j) {
                    if (w == v && (j == i || j == (i + 1) % kWallPoints || i == (j + 1) % kWallPoints))
                        continue;
                    const Vec2 p = a[i], q = a[(i + 1) % kWallPoints];
                    const float len = hypotf(q.x - p.x, q.y - p.y);
                    const float hit =
                        ray_segment(p, {(q.x - p.x) / len, (q.y - p.y) / len}, b[j], b[(j + 1) % kWallPoints], len);
                    CHECK_MSG(!(hit > 1e-4f && hit < len - 1e-4f), "track walls intersect");
                }
        }
}

TEST_CASE("race: car rays see a slower leader and respect wall occlusion") {
    const Track t = make_track(true);
    State s[kRaceCars];
    uint32_t ep[kRaceCars];
    reset_race(s, ep, t, 1, 0);
    s[0] = {};
    s[0].x = 42.f;
    s[0].y = -16.f;
    s[0].heading = kPi / 2.f;
    s[0].speed = 10.f;
    s[1] = s[0];
    s[1].y = -6.f;
    s[1].speed = 3.f;
    for (int j = 2; j < kRaceCars; ++j) {
        s[j].x = -40.f;
        s[j].y = -20.f + 6.f * j;
    }
    float obs[kObsDim];
    observe_race(s, 0, t, obs);
    CHECK(std::fabs(obs[kRays + 1 + 4] * kRayRange - (10.f - kTrafficRadius)) < 1e-4f);
    CHECK(std::fabs(obs[2 * kRays + 1 + 4] - 7.f / kMaxSpeed) < 1e-4f);
    s[1].x = 60.f;
    s[1].y = -16.f;
    observe_race(s, 0, t, obs);
    CHECK(obs[kRays + 1] == 1.f); // right-hand ray has a wall before the car
}

TEST_CASE("race: cornering at high speed is limited and braking increases steering authority") {
    const Track t = make_track(true);
    State fast{};
    fast.x = 42.f;
    fast.y = -16.f;
    fast.heading = kPi / 2.f;
    fast.speed = 15.f;
    fast.last_angle = track_position(t, {fast.x, fast.y});
    State slow = fast;
    slow.speed = 5.f;
    const float act[2] = {1.f, 0.f};
    const float h = fast.heading;
    advance_race(fast, act, t);
    advance_race(slow, act, t);
    CHECK(slow.heading - h > fast.heading - h);
    CHECK((fast.heading - h) * fast.speed / kDt < 6.01f);
}

TEST_CASE("race: contact terminates involved cars and respawns with clear space") {
    const Track t = make_track(true);
    State old[kRaceCars], s[kRaceCars];
    uint32_t ep[kRaceCars];
    reset_race(old, ep, t, 1, 0);
    old[0].x = 42.f;
    old[0].y = -16.f;
    old[1] = old[0];
    old[1].y = -10.f;
    for (int i = 0; i < kRaceCars; ++i)
        s[i] = old[i];
    s[0].y = -12.f;
    s[1].y = -14.f;
    StepResult r[kRaceCars] = {};
    float final[kRaceCars * kObsDim] = {};
    resolve_race(s, old, r, ep, t, 1, 0, final);
    CHECK(r[0].contact && r[1].contact);
    CHECK(r[0].terminated && !r[0].truncated);
    CHECK(r[0].reward == -kCrashPenalty);
    CHECK(ep[0] == 1 && ep[1] == 1);
    check_separation(s);
}

TEST_CASE("race: time limit preserves final traffic observation and clean laps record their duration") {
    const Track t = make_track(true);
    State s[kRaceCars], old[kRaceCars];
    uint32_t ep[kRaceCars];
    reset_race(s, ep, t, 31, 0);
    StepResult result[kRaceCars];
    float final[kRaceCars * kObsDim] = {};
    const float action[2] = {0.f, 0.f};
    for (int i = 0; i < kRaceCars; ++i) {
        s[i].speed = 0.f;
        s[i].steps = kMaxSteps - 1;
        old[i] = s[i];
        result[i] = advance_race(s[i], action, t);
        CHECK(result[i].truncated && !result[i].terminated);
    }
    float expected[kObsDim];
    observe_race(s, 0, t, expected);
    resolve_race(s, old, result, ep, t, 31, 0, final);
    for (int k = 0; k < kObsDim; ++k)
        CHECK(final[k] == expected[k]);
    CHECK(s[0].steps == 0 && ep[0] == 1);
    check_separation(s);

    State lap = grid_state(t, 0, 31, 0, 0);
    lap.progress = kTwoPi - 1e-4f;
    lap.steps = 500;
    lap.lap_step = 100;
    const auto finish = advance_race(lap, action, t);
    CHECK(finish.clean_lap);
    CHECK(std::fabs(finish.lap_seconds - 401.f * kDt) < 1e-4f);
}

GPU_TEST_CASE("race: grouped CUDA steps match CPU including collisions and safe respawns") {
    constexpr int n = 16;
    const Track t = make_track(true);
    RacingBatch batch(n, 123, true);
    State host[n];
    uint32_t ep[n];
    for (int group = 0; group < n / kRaceCars; ++group)
        reset_race(host + group * kRaceCars, ep + group * kRaceCars, t, 123, group);
    device_ptr<float> actions(n * 2), rewards(n), done(n), trunc(n), final(n * kObsDim), len(n), laps(n),
        metrics(n * 4);
    StepOutputs out{rewards.get(), done.get(), trunc.get(), final.get(), len.get(), laps.get(), metrics.get()};
    for (int step_i = 0; step_i < 100; ++step_i) {
        const auto a = test::random_floats(n * 2, static_cast<uint32_t>(step_i + 17), 0.6f);
        actions.copy_from_host(a.data(), a.size());
        batch.step(actions.get(), out);
        const auto dr = test::to_host(rewards, n), dd = test::to_host(done, n);
        for (int g = 0; g < n / kRaceCars; ++g) {
            State prev[kRaceCars];
            StepResult r[kRaceCars];
            float fo[kRaceCars * kObsDim];
            for (int j = 0; j < kRaceCars; ++j) {
                const int i = g * kRaceCars + j;
                prev[j] = host[i];
                r[j] = advance_race(host[i], a.data() + i * 2, t);
            }
            resolve_race(host + g * kRaceCars, prev, r, ep + g * kRaceCars, t, 123, static_cast<uint32_t>(g), fo);
            check_separation(host + g * kRaceCars);
            for (int j = 0; j < kRaceCars; ++j) {
                const int i = g * kRaceCars + j;
                CHECK(dd[i] == ((r[j].terminated || r[j].truncated) ? 1.f : 0.f));
                CHECK(std::fabs(dr[i] - r[j].reward) < 1e-3f);
            }
        }
    }
    std::vector<float> expected(n * kObsDim);
    for (int i = 0; i < n; ++i)
        observe_race(host + (i / kRaceCars) * kRaceCars, i % kRaceCars, t, expected.data() + i * kObsDim);
    CHECK_CLOSE(test::to_host(batch.obs(), expected.size()), expected, 1e-3);
    std::vector<State> gpu(n);
    CUDA_CHECK_THROW(cudaMemcpy(gpu.data(), batch.states(), n * sizeof(State), cudaMemcpyDeviceToHost));
    for (int g = 0; g < n / kRaceCars; ++g)
        check_separation(gpu.data() + g * kRaceCars);
}

GPU_TEST_CASE("race learning: traffic policy improves and completes clean laps") {
    rl::RacingPpoConfig cfg;
    cfg.total_updates = 1200;
    rl::RacingPpoTrainer trainer(cfg);
    float first = 0.f, last = 0.f;
    int clean = 0;
    for (int u = 0; u < cfg.total_updates; ++u) {
        const auto s = trainer.update();
        if (u == 0)
            first = s.mean_laps;
        last = s.mean_laps;
        clean += s.clean_laps;
        if (u % 100 == 0 || u + 1 == cfg.total_updates) {
            std::printf("    traffic update %d: laps %.3f, contacts %d, walls %d, clean %d, %.2f M steps/s\n", u + 1,
                        s.mean_laps, s.contact_cars, s.wall_crashes, s.clean_laps, s.env_steps_per_second / 1e6);
            std::fflush(stdout);
        }
    }
    CHECK(last > first + 0.3f);
    CHECK(clean > 0);
    demos::RacingShowcase demo(kRaceCars, cfg.hidden, 55);
    demo.set_policy(trainer.actor_params());
    for (int step_i = 0; step_i < 1200; ++step_i) {
        demo.step();
        State s[kRaceCars];
        for (int i = 0; i < kRaceCars; ++i)
            s[i] = demo.state(i);
        check_separation(s);
    }
    std::printf("    showcase: %d clean laps, %d contact cars, %d walls, best %.2f s\n", demo.clean_laps(),
                demo.contacts(), demo.wall_crashes(), demo.best_lap());
    CHECK(demo.clean_laps() > 0);
}

GPU_TEST_CASE("race comparison: trained policy visibly outperforms its own frozen initial policy") {
    rl::RacingPpoConfig cfg;
    cfg.total_updates=1200;
    rl::RacingPpoTrainer trainer(cfg);
    const auto initial=trainer.actor_params();
    demos::RacingShowcase untrained(kRaceCars,cfg.hidden,55);
    untrained.set_policy(initial);
    for(int step_i=0;step_i<1200;++step_i)untrained.step();
    for(int u=0;u<200;++u)trainer.update();
    demos::RacingShowcase learned(kRaceCars,cfg.hidden,55);
    learned.set_policy(trainer.actor_params());
    for(int step_i=0;step_i<1200;++step_i)learned.step();
    std::printf("    Same grid/seed, 60 s: initial %d clean laps / %d wall hits; update 200 %d clean laps / %d wall hits\n",
        untrained.clean_laps(),untrained.wall_crashes(),learned.clean_laps(),learned.wall_crashes());
    CHECK(learned.clean_laps()>untrained.clean_laps());
    CHECK(learned.clean_laps()>0);
    CHECK(trainer.actor_params()!=initial);
}

TEST_CASE("race showcase: safe cars continue past the training episode time limit") {
    demos::RacingShowcase demo(kRaceCars,64,55);
    nn::MlpSpec spec({kObsDim,64,64,kActionDim},
        {nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity});
    std::vector<float> parked_policy(static_cast<size_t>(spec.num_params()),0.f);
    parked_policy.back()=-1.f; // full braking parks safely; isolate the episode clock
    demo.set_policy(parked_policy);
    int continued=0;
    for(int tick=0;tick<kMaxSteps+100;++tick) {
        State before[kRaceCars];
        for(int i=0;i<kRaceCars;++i)before[i]=demo.state(i);
        demo.step();
        for(int i=0;i<kRaceCars;++i) {
            const State& after=demo.state(i);
            if(before[i].steps>=kMaxSteps-1 && !demo.just_crashed(i)) {
                CHECK(after.steps==before[i].steps+1);
                CHECK(hypotf(after.x-before[i].x,after.y-before[i].y)<=kMaxSpeed*kDt+1e-4f);
                ++continued;
            }
        }
    }
    CHECK(continued>0);
}

TEST_CASE("race showcase: starting final evaluation clears metrics without moving cars") {
    demos::RacingShowcase demo(kRaceCars,64,55);
    for(int tick=0;tick<100;++tick)demo.step();
    State before[kRaceCars];
    for(int i=0;i<kRaceCars;++i)before[i]=demo.state(i);
    demo.begin_evaluation();
    CHECK(demo.contacts()==0 && demo.wall_crashes()==0 && demo.clean_laps()==0 && demo.best_lap()==0.f);
    for(int i=0;i<kRaceCars;++i) {
        const State& after=demo.state(i);
        CHECK(after.x==before[i].x && after.y==before[i].y);
        CHECK(after.heading==before[i].heading && after.speed==before[i].speed);
        CHECK(after.steps==before[i].steps && after.last_angle==before[i].last_angle);
        CHECK(after.progress==0.f && after.lap_step==after.steps);
    }
}
