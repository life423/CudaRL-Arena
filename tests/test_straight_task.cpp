#include "check.h"
#include "motor/straight_task.h"
#include "rl/ppo_math.h"
namespace st = cudarl::straight_motor;
namespace p = cudarl::physics;
TEST_CASE("path_scale: production observations change only the three rotated path channels") {
    for (int pose = 0; pose < 4; ++pose)
        for (int sign : {-1, 1}) {
            st::ForwardConfig a, b;
            b.observation_schema = 2;
            auto w = st::initial_world(pose);
            st::ForwardTask t;
            CHECK(st::begin_forward(w, a, t));
            w.physical.body.position.y += sign * .0005f / p::scale;
            w.physical.body.position.z += .001f / p::scale;
            float x[40], y[40];
            CHECK(st::observe(t, w, a, x) && st::observe(t, w, b, y));
            for (int j = 0; j < 40; ++j)
                CHECK(fabsf(y[j] - (j < 3 ? 36.f * x[j] : x[j])) < 1e-6f);
            const float decoded = -.002f * (y[0] * y[9] + y[1] * y[10] + y[2] * y[11]);
            CHECK(fabsf(decoded - sign * .0005f) < 1e-8f);
            CHECK(w.attempted_substeps == 0);
            b.observation_schema = 99;
            CHECK(!st::observe(t, w, b, y));
        }
    st::TrainingEpisode episode;
    episode.observation_schema = 2;
    CHECK(episode.task_config().observation_schema == 2);
}
TEST_CASE("straight_depth_mix: balanced phase cycle preserves original poses and fixed references (zero "
          "integration)") {
    st::TrainingEpisode episode;
    episode.controls = 640;
    episode.fixed_depth_offsets = true;
    int count[3]{};
    for (int phase = 0; phase < 3; ++phase) {
        episode.offset_phase = phase;
        for (int i = 0; i < 32; ++i) {
            const int group = (i / 2 + phase) % 3;
            ++count[group];
            const float offset = episode.initial_vertical_offset(i);
            st::World world;
            st::ForwardTask task;
            CHECK(st::begin_training_episode(i, false, episode.task_config(), world, task, episode.arena(),
                                             offset));
            const auto baseline = st::initial_world(i % 2 ? 3 : 0);
            const auto target = cudarl::motor::motion(baseline.physical.body).com_model;
            const auto actual = cudarl::motor::motion(world.physical.body).com_model;
            CHECK(p::norm(task.start_com_model - target) == 0 && task.request.position.y == target.y);
            CHECK(fabsf((actual.y - target.y) * p::scale - offset) < 1e-8f);
            CHECK(world.physical.body.orientation.x == baseline.physical.body.orientation.x &&
                  world.physical.body.orientation.y == baseline.physical.body.orientation.y);
            CHECK(world.attempted_substeps == 0);
        }
    }
    CHECK(count[0] == 32 && count[1] == 32 && count[2] == 32);
}
TEST_CASE(
    "straight_fixed_target: vertical starts retain reference observations and reward (zero integration)") {
    for (int environment : {0, 1}) {
        st::ForwardConfig config;
        st::World centered;
        st::ForwardTask reference;
        CHECK(st::begin_training_episode(environment, false, config, centered, reference));
        for (float offset : {0.f, -.0005f, .0005f}) {
            st::World world;
            st::ForwardTask task;
            CHECK(st::begin_training_episode(environment, false, config, world, task, {}, offset));
            CHECK(p::norm(task.start_com_model - reference.start_com_model) == 0);
            CHECK(p::norm(task.request.position - reference.request.position) == 0);
            CHECK(p::norm(task.request.heading - reference.request.heading) == 0);
            const auto com = cudarl::motor::motion(world.physical.body).com_model;
            CHECK(fabsf((com.y - reference.start_com_model.y) * p::scale - offset) < 1e-8f);
            CHECK(p::norm(task.steady_start_com_model - com) == 0);
            world.physical.body.velocity[1] = -.0006f;
            float obs[st::kObservations];
            CHECK(st::observe(task, world, config, obs));
            const float e = -.072f * (obs[0] * obs[9] + obs[1] * obs[10] + obs[2] * obs[11]);
            const float vy = .02f * (obs[12] * obs[9] + obs[13] * obs[10] + obs[14] * obs[11]);
            CHECK(fabsf(e - offset) < 1e-8f && fabsf(vy + .0006f) < 1e-9f);
            CHECK(fabsf(st::measure(task, world).depth_m - fabsf(offset)) < 1e-8f);
            world.physical.control_steps = 1;
            const auto first = st::record_control(task, world, config);
            CHECK(first.accepted && !first.done && first.costs.horizontal == 0);
            const float expected = .05f * (offset / .001f) * (offset / .001f);
            CHECK(fabsf(first.costs.depth - expected) < 1e-6f);
            CHECK(first.costs.recovery > 0);
            CHECK(fabsf(first.reward + .05f * (expected + first.costs.recovery)) < 1e-7f);
            CHECK(p::norm(task.start_com_model - reference.start_com_model) == 0);
            CHECK(world.attempted_substeps == 0);
            const auto unchanged_y = world.physical.body.position.y;
            CHECK(!st::offset_episode_start(world, task, config, .0005f));
            CHECK(world.physical.body.position.y == unchanged_y);
        }
    }
    st::World world;
    st::ForwardTask task;
    st::ForwardConfig config;
    CHECK(st::begin_forward(world, config, task));
    const auto y = world.physical.body.position.y;
    CHECK(!st::offset_episode_start(world, task, config, .2f));
    CHECK(world.physical.body.position.y == y && task.start_com_model.y == -2);
}
TEST_CASE("straight_reward5: symmetric recovery and near-target braking apply from first step (synthetic)") {
    st::World world;
    st::ForwardConfig config;
    for (int previous : {0, 39, 40, 159}) {
        auto score = [&](float depth, float vy) {
            st::World sample = world;
            st::ForwardTask task;
            CHECK(st::begin_forward(sample, config, task));
            sample.physical.body.position.y += depth / p::scale;
            sample.physical.body.velocity[0] = .008f;
            sample.physical.body.velocity[1] = vy;
            task.controls = previous;
            task.last_world_control = previous;
            sample.physical.control_steps = previous + 1;
            const auto result = st::record_control(task, sample, config);
            CHECK(result.accepted && sample.attempted_substeps == 0);
            return result;
        };
        for (float sign : {-1.f, 1.f}) {
            const auto departing = score(sign * .0005f, sign * .00025f);
            const auto holding = score(sign * .0005f, 0);
            const auto recovering = score(sign * .0005f, -sign * .00025f);
            CHECK(recovering.reward > holding.reward && holding.reward > departing.reward);
            CHECK(fabsf(recovering.reward + .000625f) < 1e-7f);
            CHECK(fabsf(holding.reward + .000673828125f) < 1e-7f);
            CHECK(fabsf(departing.reward + .0008203125f) < 1e-7f);
            const auto too_fast = score(sign * .0001f, -sign * .001f);
            const auto settling = score(sign * .0001f, -sign * .00005f);
            CHECK(settling.reward > too_fast.reward);
            CHECK(recovering.done == (previous == 159));
            CHECK(score(0, 0).reward == 0 && score(0, sign * .001f).reward < 0);
            if (previous == 40 && sign < 0)
                std::printf("    below0.5mm per50ms: depart %.12f, hold %.12f, recover %.12f; near0.1mm fast "
                            "%.12f, settle %.12f\n",
                            departing.reward, holding.reward, recovering.reward, too_fast.reward,
                            settling.reward);
        }
    }
}
TEST_CASE("straight_training_arena:32s uses existing long arena without changing initial pose or target") {
    st::TrainingEpisode episode;
    CHECK(episode.arena().half_width_m == cudarl::contact::Arena{}.half_width_m);
    CHECK(episode.arena().half_depth_m == cudarl::contact::Arena{}.half_depth_m);
    episode.controls = 640;
    CHECK(episode.arena().half_width_m == .6f && episode.arena().half_depth_m == .4f);
    for (int i = 0; i < 32; ++i) {
        auto baseline = st::initial_world(i % 2 ? 3 : 0);
        st::ForwardTask reference;
        CHECK(st::begin_forward(baseline, st::ForwardConfig{}, reference));
        st::World world;
        st::ForwardTask task;
        CHECK(st::begin_training_episode(i, false, episode.task_config(), world, task, episode.arena()));
        CHECK(p::norm(world.physical.body.position - baseline.physical.body.position) == 0);
        CHECK(p::norm(task.start_com_model - reference.start_com_model) == 0);
        CHECK(p::norm(task.request.position - reference.request.position) == 0);
        CHECK(p::norm(task.request.heading - reference.request.heading) == 0);
        CHECK(p::norm(task.request.terminal_velocity - reference.request.terminal_velocity) == 0);
        CHECK(world.physical.fluid.density == baseline.physical.fluid.density);
        CHECK(world.physical.fluid.viscosity == baseline.physical.fluid.viscosity);
        CHECK(world.physical.fluid.gravity == baseline.physical.fluid.gravity);
        for (int j = 0; j < 4; ++j) {
            CHECK(world.physical.body.angle[j] == baseline.physical.body.angle[j]);
            CHECK(world.physical.held_targets[j] == baseline.physical.held_targets[j]);
        }
        for (float v : world.physical.body.velocity)
            CHECK(v == 0);
        CHECK(world.attempted_substeps == 0);
    }
}
TEST_CASE("straight_reward5: head roll and pitch separate while combined tilt acceptance stays10degrees") {
    st::World world;
    st::ForwardConfig config;
    st::ForwardTask task;
    CHECK(st::begin_forward(world, config, task));
    for (float yaw : {0.f, .7f}) {
        for (float roll : {0.f, 5.f}) {
            for (float pitch : {-5.f, 0.f, 5.f}) {
                world.physical.body.orientation =
                    p::axis_rotation({0, 1, 0}, yaw) *
                    p::axis_rotation({0, 0, 1}, pitch * cudarl::movement::pi / 180) *
                    p::axis_rotation({1, 0, 0}, roll * cudarl::movement::pi / 180);
                const auto metrics = st::measure(task, world);
                CHECK(fabsf(metrics.pitch_deg - pitch) < 1e-4f && fabsf(metrics.roll_deg - roll) < 1e-4f);
                const auto cost = st::reward_costs(metrics, world, config, false);
                CHECK(fabsf(cost.roll - .05f * (roll / 10) * (roll / 10)) < 1e-6f);
                CHECK(fabsf(cost.pitch - .01f * (pitch / 10) * (pitch / 10)) < 1e-6f);
                CHECK(cost.posture == cost.roll + cost.pitch);
            }
        }
    }
    CHECK(config.tilt_limit_deg == 10 && config.depth_limit_m == .001f && config.speed_rmse_limit == .0025f);
    CHECK(world.attempted_substeps == 0);
    st::Metrics hold, useful, pointing_only;
    hold.signed_depth_m = useful.signed_depth_m = pointing_only.signed_depth_m = -.0005f;
    hold.depth_m = useful.depth_m = pointing_only.depth_m = .0005f;
    useful.vertical_velocity_m_s = .00025f;
    useful.pitch_deg = pointing_only.pitch_deg = 1.8f;
    const auto h = st::reward_costs(hold, world, config, true);
    const auto u = st::reward_costs(useful, world, config, true);
    const auto p = st::reward_costs(pointing_only, world, config, true);
    CHECK(u.total() < h.total() && p.total() > h.total());
    std::printf("    synthetic useful1.8deg pitch reward %.12f vs level hold %.12f, pointing-only %.12f\n",
                -.05f * u.total(), -.05f * h.total(), -.05f * p.total());
}
TEST_CASE(
    "straight_reward5: settled optimum beats bounded recovery and oscillation (synthetic trajectories)") {
    st::World world;
    st::ForwardConfig config;
    double holding = 0, recovering = 0, oscillating = 0;
    for (int i = 0; i < 160; ++i) {
        const float time = (i + 1) * .05f;
        for (int kind = 0; kind < 3; ++kind) {
            st::Metrics m;
            if (kind == 0)
                m.signed_depth_m = -.0005f;
            if (kind == 1) {
                m.signed_depth_m = -.0005f * expf(-time / 2);
                m.vertical_velocity_m_s = -m.signed_depth_m / 2;
            }
            if (kind == 2) {
                m.signed_depth_m = .0005f * sinf(time * cudarl::movement::pi);
                m.vertical_velocity_m_s = .0005f * cudarl::movement::pi * cosf(time * cudarl::movement::pi);
            }
            m.depth_m = fabsf(m.signed_depth_m);
            const auto cost = st::reward_costs(m, world, config, true);
            CHECK(std::isfinite(cost.total()) && cost.total() >= 0);
            if (kind == 0)
                holding -= .05 * cost.total();
            if (kind == 1)
                recovering -= .05 * cost.total();
            if (kind == 2)
                oscillating -= .05 * cost.total();
        }
    }
    CHECK(recovering > holding && recovering < 0 && oscillating < 0);
    CHECK(st::reward_costs({}, world, config, true).total() == 0);
    CHECK(st::recovery_velocity(-.01f) == .001f && st::recovery_velocity(.01f) == -.001f);
    CHECK(st::recovery_velocity(0) == 0);
    CHECK(st::recovery_velocity(-.0001f) > 0 && st::recovery_velocity(.0001f) < 0);
    for (float angle : {-180.f, -90.f, 90.f, 180.f}) {
        auto posed = world;
        posed.physical.body.orientation = p::axis_rotation({0, 0, 1}, angle * cudarl::movement::pi / 180);
        st::ForwardTask task;
        CHECK(st::begin_forward(world, config, task));
        const auto cost = st::reward_costs(st::measure(task, posed), posed, config, true);
        CHECK(std::isfinite(cost.total()) && cost.total() >= 0);
    }
    std::printf("    synthetic8s reward: hold %.9f, recover %.9f, oscillate %.9f, settled0\n", holding,
                recovering, oscillating);
}
TEST_CASE("straight_episode32: task continues past8s and includes late rewards until32s terminal") {
    st::TrainingEpisode episode;
    episode.controls = 640;
    CHECK(episode.valid());
    const auto config = episode.task_config();
    CHECK(config.maximum_controls == 640 && config.measure_start == 40);
    st::World world;
    st::ForwardTask task;
    CHECK(st::begin_training_episode(0, false, config, world, task));
    float rewards[640]{}, values[640]{}, dones[640]{}, zeros[640]{}, last[1]{}, advantages[640]{},
        returns[640]{};
    double late_reward = 0;
    for (int i = 0; i < 640; ++i) {
        float obs[40]{};
        CHECK(st::observe(task, world, config, obs));
        world.physical.control_steps = i + 1; // Bookkeeping fixture; no physics.
        const auto transition = st::record_control(task, world, config);
        CHECK(transition.accepted && transition.done == (i == 639));
        rewards[i] = transition.reward;
        dones[i] = transition.done ? 1.f : 0.f;
        if (i < 40)
            CHECK(rewards[i] == 0);
        else
            CHECK(fabsf(rewards[i] + .05f) < 1e-7f);
        if (i >= 160)
            late_reward += rewards[i];
    }
    CHECK(fabs(late_reward + 24.) < 1e-5 && task.controls == 640);
    cudarl::rl::gae_one_env(640, 1, 0, rewards, values, dones, zeros, zeros, last, .99f, 1.f, advantages,
                            returns);
    CHECK(returns[160] < -4.f && returns[639] == rewards[639]);
    CHECK(world.attempted_substeps == 0);
}
TEST_CASE("straight_initial_tilts: rigid pitch mixture preserves COM depth, joints and horizontal request") {
    int level_count = 0, up_count = 0, down_count = 0;
    for (int i = 0; i < 32; ++i) {
        const auto level = st::initial_world(i % 2 ? 3 : 0);
        const auto original_com = cudarl::motor::motion(level.physical.body).com_model;
        st::World world;
        st::ForwardTask task;
        st::ForwardConfig config;
        CHECK(st::begin_training_episode(i, true, config, world, task));
        const auto com = cudarl::motor::motion(world.physical.body).com_model;
        CHECK(fabsf(com.y - original_com.y) < 3e-7f);
        CHECK(fabsf(task.request.heading.y) < 1e-7f && fabsf(task.request.terminal_velocity.y) < 1e-9f);
        const int group = (i / 2) % 4;
        const float degrees = group == 1 ? 3.f : (group == 2 ? -3.f : 0.f);
        const auto forward = world.physical.body.orientation.rotate({1, 0, 0});
        CHECK(fabsf(forward.y - sinf(degrees * cudarl::movement::pi / 180)) < 1e-6f);
        if (degrees == 0)
            ++level_count;
        else if (degrees > 0)
            ++up_count;
        else
            ++down_count;
        for (int j = 0; j < 4; ++j) {
            CHECK(world.physical.body.angle[j] == level.physical.body.angle[j]);
            CHECK(world.physical.held_targets[j] == level.physical.held_targets[j]);
        }
        for (float velocity : world.physical.body.velocity)
            CHECK(velocity == 0);
        CHECK(world.attempted_substeps == 0 && world.physical.body.work == 0);
    }
    CHECK(level_count == 16 && up_count == 8 && down_count == 8);
}
TEST_CASE(
    "straight_reward5: lane depth and head tilt cost apply from first control through startup boundary") {
    for (int previous : {0, 19, 39, 40}) {
        st::World original;
        st::ForwardConfig config;
        st::ForwardTask initial;
        CHECK(st::begin_forward(original, config, initial));
        initial.controls = previous;
        initial.last_world_control = previous;
        auto baseline = original;
        baseline.physical.control_steps = previous + 1;
        auto baseline_task = initial;
        const auto clean = st::record_control(baseline_task, baseline, config);
        CHECK(clean.accepted && clean.costs.lane == 0 && clean.costs.depth == 0 && clean.costs.posture == 0);
        for (int axis = 0; axis < 3; ++axis) {
            auto changed = baseline;
            auto task = initial;
            if (axis == 0)
                changed.physical.body.position.z += config.lateral_limit_m / p::scale;
            if (axis == 1)
                changed.physical.body.position.y += config.depth_limit_m / p::scale;
            if (axis == 2) {
                changed.physical.body.orientation =
                    p::axis_rotation({1, 0, 0}, config.tilt_limit_deg * cudarl::movement::pi / 180);
                changed.physical.body.position +=
                    initial.start_com_model - cudarl::motor::motion(changed.physical.body).com_model;
            }
            const auto q = changed.physical.body.orientation;
            const auto position = changed.physical.body.position;
            const auto result = st::record_control(task, changed, config);
            CHECK(result.accepted);
            const float extra_recovery = axis == 1 ? .00390625f : 0.f;
            CHECK(std::fabs((result.reward - clean.reward) + p::dt * (.05f + extra_recovery)) < 1e-6f);
            CHECK(std::fabs((axis == 0   ? result.costs.lane
                             : axis == 1 ? result.costs.depth
                                         : result.costs.posture) -
                            .05f) < 2e-5f);
            CHECK(changed.physical.body.orientation.x == q.x && changed.physical.body.orientation.w == q.w);
            CHECK(changed.physical.body.position.x == position.x &&
                  changed.physical.body.position.y == position.y);
            CHECK(changed.attempted_substeps == 0);
        }
    }
}
TEST_CASE("straight_reward5: posture uses head up yaw invariant and does not punish tail bending as tilt") {
    st::World world;
    st::ForwardTask task;
    st::ForwardConfig config;
    CHECK(st::begin_forward(world, config, task));
    for (float angle : {0.f, .2f, .5f}) {
        for (int j = 0; j < 4; ++j)
            world.physical.body.angle[j] = (j % 2 ? 1.f : -1.f) * angle;
        CHECK(st::measure(task, world).tilt_deg == 0);
    }
    for (p::V axis : {p::V{1, 0, 0}, p::V{0, 0, 1}}) {
        world.physical.body.orientation =
            p::axis_rotation({0, 1, 0}, .7f) * p::axis_rotation(axis, cudarl::movement::pi / 6);
        CHECK(std::fabs(st::measure(task, world).tilt_deg - 30) < 1e-4f);
    }
    world.physical.body.orientation = p::axis_rotation({0, 1, 0}, 1.2f);
    CHECK(st::measure(task, world).tilt_deg == 0);
    CHECK(world.attempted_substeps == 0);
}
TEST_CASE("straight_reward5: meaningful progress outweighs all near-limit soft state costs") {
    st::World world;
    st::ForwardConfig config;
    st::Metrics rest;
    rest.horizontal_speed_error_m_s = config.target_speed_m_s;
    const auto stationary = st::reward_costs(rest, world, config, true);
    st::Metrics moving;
    moving.horizontal_speed_error_m_s = .5f * config.target_speed_m_s;
    moving.lateral_m = config.lateral_limit_m;
    moving.depth_m = config.depth_limit_m;
    moving.signed_depth_m = moving.depth_m;
    moving.roll_deg = config.tilt_limit_deg;
    world.physical.mean_squared_torque = 1;
    const auto progress = st::reward_costs(moving, world, config, true);
    CHECK(std::fabs(stationary.total() - 1) < 1e-6f);
    CHECK(std::fabs(progress.total() - .40590625f) < 1e-6f);
    CHECK(progress.total() < stationary.total());
    moving.roll_deg = 90;
    const auto extreme = st::reward_costs(moving, world, config, false);
    CHECK(std::fabs(extreme.posture - 4.05f) < 1e-5f);
    CHECK(world.attempted_substeps == 0);
}
TEST_CASE("straight_recovery: copied prefix preserves original deadline observations safety and work (zero "
          "integration)") {
    st::World world;
    auto config = st::config_for_speed(.008f);
    st::ForwardTask task;
    CHECK(st::begin_forward(world, config, task));
    world.physical.body.velocity[0] = .008f;
    for (int n = 1; n <= 20; ++n) {
        world.physical.control_steps = n;
        CHECK(st::record_control(task, world, config).accepted);
    }
    auto fork = world;
    auto recovery = task;
    fork.physical.body.work = .0001f;
    recovery.maximum_lateral = .003f;
    float obs[st::kObservations]{};
    CHECK(st::observe(recovery, fork, config, obs));
    CHECK(std::fabs(obs[34] - .5f) < 1e-7f);
    for (int n = 21; n <= config.maximum_controls; ++n) {
        fork.physical.control_steps = n;
        fork.physical.body.position.x = .008f * p::dt * float(n) / p::scale;
        CHECK(st::record_control(recovery, fork, config).accepted);
    }
    CHECK(recovery.controls == config.maximum_controls &&
          recovery.steady_samples == config.maximum_controls - config.measure_start &&
          recovery.outcome == st::Outcome::Timeout);
    CHECK(recovery.start_work_J == 0 && recovery.maximum_lateral == .003f && task.controls == 20);
    CHECK(world.attempted_substeps == 0 && fork.attempted_substeps == 0);
}
TEST_CASE(
    "straight_reward5: interval40 is warmup and41 is scored, effort always charged (zero integration)") {
    st::World world;
    st::ForwardConfig config;
    st::ForwardTask task;
    CHECK(st::begin_forward(world, config, task));
    task.controls = 39;
    task.last_world_control = 39;
    world.physical.control_steps = 40;
    world.physical.mean_squared_torque = .5f;
    const auto warmup = st::record_control(task, world, config);
    CHECK(warmup.accepted && task.steady_samples == 0);
    CHECK(std::fabs(warmup.reward + .00005f) < 1e-8f);
    world.physical.control_steps = 41;
    const auto scored = st::record_control(task, world, config);
    CHECK(scored.accepted && task.steady_samples == 1);
    CHECK(std::fabs(scored.reward - warmup.reward + .05f) < 1e-7f);
    CHECK(world.attempted_substeps == 0);
}
TEST_CASE("straight_reward5: warmup safety and work are not discarded (zero integration)") {
    st::World bad;
    st::ForwardConfig config;
    st::ForwardTask rejected;
    CHECK(st::begin_forward(bad, config, rejected));
    bad.physical.invalid = true;
    bad.physical.control_steps = 1;
    CHECK(!st::record_control(rejected, bad, config).accepted);
    st::World world;
    st::ForwardTask task;
    CHECK(st::begin_forward(world, config, task));
    world.physical.body.velocity[0] = config.target_speed_m_s;
    world.physical.body.work = .0001f; // already spent in warmup; must count at acceptance
    for (int n = 1; n <= config.maximum_controls; ++n) {
        world.physical.body.position.x = config.target_speed_m_s * p::dt * float(n) / p::scale;
        world.physical.control_steps = n;
        CHECK(st::record_control(task, world, config).accepted);
    }
    CHECK(task.outcome == st::Outcome::Timeout);
    CHECK(world.attempted_substeps == 0 && bad.attempted_substeps == 0);
}
TEST_CASE("straight_reward5: later scored costs retain full-return credit to warmup (zero integration)") {
    float rewards[80]{}, values[80]{}, done[80]{}, zero[80]{}, advantage[80]{}, returns[80]{};
    const float last[1]{99};
    for (int t = 40; t < 80; ++t)
        rewards[t] = -.01f;
    done[79] = 1;
    cudarl::rl::gae_one_env(80, 1, 0, rewards, values, done, zero, zero, last, .99f, 1.f, advantage, returns);
    CHECK(rewards[0] == 0 && advantage[0] < -.1f);
    CHECK(returns[39] < 0 && std::fabs(returns[39] - .99f * returns[40]) < 1e-6f);
}
TEST_CASE(
    "straight_learning: full-episode lambda1 returns ignore bad terminal bootstrap (zero integration)") {
    float rewards[80]{}, values[80]{}, dones[80]{}, zeros[80]{}, advantage[80]{}, returns[80]{};
    const float last[1]{999};
    for (int t = 0; t < 80; ++t) {
        rewards[t] = t < 20 ? -.05f : -.002f;
        values[t] = .3f * std::cos(float(t));
    }
    dones[79] = 1;
    cudarl::rl::gae_one_env(80, 1, 0, rewards, values, dones, zeros, zeros, last, .99f, 1.f, advantage,
                            returns);
    double reference = 0;
    for (int t = 79; t >= 0; --t) {
        reference = rewards[t] + double(.99f) * reference;
        CHECK(std::fabs(returns[t] - reference) < 2e-6);
        CHECK(std::fabs(advantage[t] + values[t] - reference) < 2e-6);
    }
}
TEST_CASE("straight_task6: forward travel beyond old destination has no arrival penalty (zero integration)") {
    st::World world;
    st::ForwardConfig config;
    st::ForwardTask task;
    CHECK(st::begin_forward(world, config, task));
    world.physical.body.velocity[0] = config.target_speed_m_s;
    auto near = world, far = world;
    auto near_task = task, far_task = task;
    near.physical.body.position.x += .020f / p::scale;
    far.physical.body.position.x += .040f / p::scale;
    near.physical.control_steps = far.physical.control_steps = 1;
    const auto a = st::record_control(near_task, near, config), b = st::record_control(far_task, far, config);
    CHECK(a.accepted && b.accepted && !a.done && !b.done);
    CHECK(std::fabs(a.reward - b.reward) < 1e-7f);
    CHECK(world.attempted_substeps == 0);
}
TEST_CASE("straight_task6: fixed window accepts straight speed and reports head yaw independently (zero "
          "integration)") {
    st::World world;
    st::ForwardConfig config;
    st::ForwardTask task;
    CHECK(st::begin_forward(world, config, task));
    const auto start = task.start_com_model;
    world.physical.body.orientation = p::axis_rotation({0, 1, 0}, .4f);
    world.physical.body.velocity[0] = config.target_speed_m_s;
    st::Transition transition;
    for (int n = 1; n <= config.maximum_controls; ++n) {
        const auto now = cudarl::motor::motion(world.physical.body).com_model;
        const auto desired = start + p::V{config.target_speed_m_s * p::dt * float(n) / p::scale, 0, 0};
        world.physical.body.position += desired - now;
        world.physical.control_steps = n;
        transition = st::record_control(task, world, config);
        CHECK(transition.accepted);
    }
    CHECK(transition.done && transition.outcome == st::Outcome::Success);
    CHECK(task.maximum_head_yaw > 20);
    CHECK(std::fabs(task.steady_speed_m_s - .008f) < 1e-6f);
    CHECK(world.attempted_substeps == 0);
}
TEST_CASE("straight_task6: cross-track drift fails despite correct forward speed (zero integration)") {
    st::World world;
    st::ForwardConfig config;
    st::ForwardTask task;
    CHECK(st::begin_forward(world, config, task));
    world.physical.body.velocity[0] = config.target_speed_m_s;
    for (int n = 1; n <= config.maximum_controls; ++n) {
        world.physical.body.position.x = config.target_speed_m_s * p::dt * float(n) / p::scale;
        world.physical.body.position.z = .003f / p::scale;
        world.physical.control_steps = n;
        const auto result = st::record_control(task, world, config);
        CHECK(result.accepted);
    }
    CHECK(task.outcome == st::Outcome::Timeout && task.maximum_lateral > .002f);
    CHECK(world.attempted_substeps == 0);
}
