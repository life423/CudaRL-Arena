#include "check.h"
#include "motor/straight_task.h"
#include "rl/ppo_math.h"
namespace st = cudarl::straight_motor;
namespace p = cudarl::physics;
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
    "straight_reward4: lane depth and head tilt cost apply from first control through startup boundary") {
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
            CHECK(std::fabs((result.reward - clean.reward) + p::dt * .05f) < 1e-6f);
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
TEST_CASE("straight_reward4: posture uses head up yaw invariant and does not punish tail bending as tilt") {
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
TEST_CASE("straight_reward4: meaningful progress outweighs all near-limit soft state costs") {
    st::World world;
    st::ForwardConfig config;
    st::Metrics rest;
    rest.speed_error_m_s = config.target_speed_m_s;
    const auto stationary = st::reward_costs(rest, world, config, true);
    st::Metrics moving;
    moving.speed_error_m_s = .5f * config.target_speed_m_s;
    moving.lateral_m = config.lateral_limit_m;
    moving.depth_m = config.depth_limit_m;
    moving.tilt_deg = config.tilt_limit_deg;
    world.physical.mean_squared_torque = 1;
    const auto progress = st::reward_costs(moving, world, config, true);
    CHECK(std::fabs(stationary.total() - 1) < 1e-6f);
    CHECK(std::fabs(progress.total() - .402f) < 1e-6f);
    CHECK(progress.total() < stationary.total());
    moving.tilt_deg = 90;
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
    "straight_reward4: interval40 is warmup and41 is scored, effort always charged (zero integration)") {
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
TEST_CASE("straight_reward4: warmup safety and work are not discarded (zero integration)") {
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
TEST_CASE("straight_reward4: later scored costs retain full-return credit to warmup (zero integration)") {
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
