// CartPole physics, termination, truncation and reset, plus the GPU batch vs
// the shared host code. The physics is checked against a separately written
// fp64 transcription of the equations, so a typo in the shared code can't be
// mirrored by the reference.

#include <cmath>
#include <vector>

#include "check.h"
#include "core/rng.h"
#include "envs/cartpole/cartpole.h"
#include "gpu_util.h"

using namespace cudarl::cartpole;

namespace {

// Gym's CartPole equations of motion, written out independently in double.
struct RefState { double x, x_dot, th, th_dot; };

RefState reference_step(RefState s, int action) {
    const double g = 9.8, mc = 1.0, mp = 0.1, l = 0.5, f = action == 1 ? 10.0 : -10.0, dt = 0.02;
    const double total = mc + mp;
    const double c = std::cos(s.th), sn = std::sin(s.th);
    const double temp = (f + mp * l * s.th_dot * s.th_dot * sn) / total;
    const double th_acc = (g * sn - c * temp) / (l * (4.0 / 3.0 - mp * c * c / total));
    const double x_acc = temp - mp * l * th_acc * c / total;
    return {s.x + dt * s.x_dot, s.x_dot + dt * x_acc, s.th + dt * s.th_dot, s.th_dot + dt * th_acc};
}

} // namespace

TEST_CASE("cartpole: dynamics match the reference equations") {
    State s{0.01f, -0.02f, 0.03f, 0.04f, 0};
    RefState r{0.01, -0.02, 0.03, 0.04};
    double worst = 0.0;
    for (int t = 0; t < 40; ++t) {
        const int a = static_cast<int>(cudarl::rng::bits(3, 0, static_cast<uint32_t>(t)) & 1u);
        integrate(s, a);
        r = reference_step(r, a);
        worst = std::fmax(worst, std::fabs(s.x - r.x) + std::fabs(s.x_dot - r.x_dot) + std::fabs(s.theta - r.th) +
                                     std::fabs(s.theta_dot - r.th_dot));
    }
    CHECK_MSG(worst < 1e-4, "max state error " + std::to_string(worst));
}

TEST_CASE("cartpole: termination thresholds") {
    CHECK(!terminated(State{0, 0, 0.2094f, 0, 0}));
    CHECK(terminated(State{0, 0, 0.2095f, 0, 0}));
    CHECK(terminated(State{0, 0, -0.2095f, 0, 0}));
    CHECK(!terminated(State{2.39f, 0, 0, 0, 0}));
    CHECK(terminated(State{2.41f, 0, 0, 0, 0}));
    CHECK(terminated(State{-2.41f, 0, 0, 0, 0}));
}

TEST_CASE("cartpole: truncation at 500 steps resets the episode") {
    State s{0, 0, 0, 0, kMaxSteps - 1};
    uint32_t episode = 7;
    float final_obs[4] = {};
    const StepResult r = step(s, 1, 42, 3, episode, final_obs);
    CHECK(r.truncated && !r.terminated && r.reward == 1.0f);
    CHECK(episode == 8);
    CHECK(s.steps == 0);
    CHECK(final_obs[1] != 0.f);   // the pushed cart was moving when the episode ended
}

TEST_CASE("cartpole: termination wins over truncation on the last step") {
    State s{0, 0, 0.2093f, 2.0f, kMaxSteps - 1};   // falls past 12 degrees on step 500
    uint32_t episode = 0;
    float final_obs[4] = {};
    const StepResult r = step(s, 0, 1, 0, episode, final_obs);
    CHECK(r.terminated && !r.truncated);
}

TEST_CASE("cartpole: initial states are small, deterministic, and vary") {
    for (uint32_t ep = 0; ep < 200; ++ep) {
        const State s = initial_state(9, 5, ep);
        for (float v : {s.x, s.x_dot, s.theta, s.theta_dot}) CHECK(std::fabs(v) <= 0.05f);
        const State again = initial_state(9, 5, ep);
        CHECK(s.x == again.x && s.theta_dot == again.theta_dot);
    }
    CHECK(initial_state(9, 5, 0).x != initial_state(9, 5, 1).x);
    CHECK(initial_state(9, 5, 0).x != initial_state(9, 6, 0).x);
}

GPU_TEST_CASE("cartpole: GPU batch matches the host step, including resets") {
    const int n = 2000, steps = 120;
    const uint32_t seed = 21;
    CartPoleBatch batch(n, seed);
    device_ptr<int> d_actions(n);
    device_ptr<float> reward(n), done(n), truncated(n), final_obs(size_t(n) * 4), length(n);
    const StepOutputs out{reward.get(), done.get(), truncated.get(), final_obs.get(), length.get()};

    std::vector<State> host(n);
    std::vector<uint32_t> episodes(n, 0);
    for (int e = 0; e < n; ++e) host[e] = initial_state(seed, e, 0);

    int mismatched_done = 0, total_done = 0;
    double worst_obs = 0.0;
    std::vector<int> actions(n);
    for (int t = 0; t < steps; ++t) {
        // Mostly-constant pushes, so many episodes end and reset within the run.
        for (int e = 0; e < n; ++e) actions[e] = (cudarl::rng::bits(77, t / 8, e) & 1u);
        d_actions.copy_from_host(actions.data(), n);
        batch.step(d_actions.get(), out);
        const auto gpu_done = cudarl::test::to_host(done, n);
        const auto gpu_obs  = cudarl::test::to_host(batch.obs(), size_t(n) * 4);
        for (int e = 0; e < n; ++e) {
            float fo[4];
            const StepResult r = step(host[e], actions[e], seed, e, episodes[e], fo);
            const bool d = r.terminated || r.truncated;
            total_done += d;
            if (d != (gpu_done[e] == 1.f)) ++mismatched_done;
            float obs[4];
            write_obs(host[e], obs);
            for (int i = 0; i < 4; ++i) worst_obs = std::fmax(worst_obs, std::fabs(obs[i] - gpu_obs[e * 4 + i]));
        }
    }
    CHECK(total_done > n);              // the test really exercised resets
    CHECK(mismatched_done == 0);
    CHECK_MSG(worst_obs < 1e-4, "max observation difference " + std::to_string(worst_obs));
}
