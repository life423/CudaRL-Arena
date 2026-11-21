#include "check.h"
#include "core/rng.h"
#include "motor/straight_task.h"
#include <limits>
namespace jm = cudarl::joint_motor;
namespace sm = cudarl::straight_motor;
TEST_CASE("joint_motor: each latent selects its own yaw pitch yaw pitch angle target without integration") {
    for (int axis = 0; axis < 4; ++axis)
        for (float sign : {-1.f, 1.f}) {
            float z[4]{}, angle[4]{}, rate[4]{};
            z[axis] = atanhf(sign * .1f);
            jm::Action action;
            CHECK(jm::action_from_latent(z, action));
            cudarl::movement::Reference reference;
            for (int tick = 0; tick < 64; ++tick)
                CHECK(jm::filter_targets(angle, rate, action.command, reference));
            for (int j = 0; j < 4; ++j) {
                if (j == axis)
                    CHECK(sign * angle[j] > .01f);
                else
                    CHECK(angle[j] == 0 && rate[j] == 0);
            }
        }
}
TEST_CASE("joint_motor: bounded adversarial target sequence respects angle rate and acceleration limits") {
    float angle[4]{}, rate[4]{};
    cudarl::movement::Reference reference;
    for (int tick = 0; tick < 20000; ++tick) {
        float target[4];
        for (int j = 0; j < 4; ++j)
            target[j] = tick % 1000 < 500 ? 1.f : -1.f;
        for (int j = 0; j < 4; ++j)
            if (tick % 7)
                target[j] = 2 * cudarl::rng::uniform(1247037233, j, tick) - 1;
        const float old_angle = angle[2], old_rate = rate[2];
        CHECK(jm::filter_targets(angle, rate, target, reference));
        for (int j = 0; j < 4; ++j)
            CHECK(fabsf(angle[j]) <= 1 && fabsf(rate[j]) <= 12 && fabsf(reference.acceleration[j]) <= 200);
        CHECK(fabsf((angle[2] - old_angle) - .005f * rate[2]) < 1e-7f);
        CHECK(fabsf((rate[2] - old_rate) - .005f * reference.acceleration[2]) < 1e-6f);
    }
}
TEST_CASE("joint_motor: invalid commands reject atomically before physics or filter mutation") {
    jm::World world;
    float z[4]{}, command[4]{};
    jm::Action action;
    z[3] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!jm::action_from_latent(z, action));
    command[1] = 1.01f;
    CHECK(!jm::step_state(world, command));
    float angle[4]{}, rate[4]{};
    cudarl::movement::Reference reference;
    CHECK(!jm::filter_targets(angle, rate, command, reference));
    for (int j = 0; j < 4; ++j)
        CHECK(angle[j] == 0 && rate[j] == 0);
    CHECK(world.attempted_substeps == 0);
}
TEST_CASE("joint_motor: all target filter states are observed and corrupt rates reject without integration") {
    auto world = sm::initial_world(0);
    sm::ForwardConfig config;
    sm::ForwardTask task;
    CHECK(sm::begin_forward(world, config, task));
    float obs[sm::kObservations]{};
    for (int j = 0; j < 4; ++j) {
        world.physical.held_targets[j] = .1f * (j + 1);
        world.reference_rate[j] = .2f * (j + 1);
    }
    CHECK(sm::observe(task, world, config, obs));
    for (int j = 0; j < 4; ++j) {
        CHECK(obs[30 + j] == world.physical.held_targets[j] / cudarl::physics::limit);
        CHECK(obs[36 + j] == world.reference_rate[j] / 12);
    }
    world.reference_rate[3] = std::numeric_limits<float>::infinity();
    CHECK(!sm::observe(task, world, config, obs));
    CHECK(world.attempted_substeps == 0);
    static_assert(sm::kObservations == 40 && sm::kActions == 4 && jm::kActionVersion == 3,
                  "Wrong independent joint interface");
}
