#include "check.h"
#include "rl/pond_ppo_task.h"
#include <array>
#include <cmath>

TEST_CASE("pond motor: straight requests retain signed sideways and depth feedback") {
    namespace task = cudarl::rl::pond_ppo_task;
    namespace obs = cudarl::pond_motor_observation;
    for (float side_m : {-.001f, .001f}) {
        task::runtime::World world{};
        task::Task request{};
        auto& motor = world.motor;
        auto& physical = motor.physical;
        auto& body = physical.body;
        CHECK(obs::initialize(0, false, body, physical.held_targets, motor.reference_rate,
                              physical.last_actual_torques, request.request));
        const auto side = body.orientation.rotate({0, 0, 1});
        body.position = body.position + side * (side_m / task::p::scale);
        body.position.y += .0005f / task::p::scale;

        std::array<float, 41> original{}, straight{}, turn{};
        CHECK(obs::observe(request.request, body, physical.held_targets, motor.reference_rate,
                           physical.last_actual_torques, original.data()));
        request.kind = task::Straight;
        CHECK(task::observe(request, world, straight.data()));
        CHECK(straight == original);
        CHECK(std::abs(straight[2] + side_m / obs::path_scale_m) < 1e-4f);

        request.kind = task::Turn;
        CHECK(task::observe(request, world, turn.data()));
        CHECK(std::abs(turn[2]) < 1e-6f);
        CHECK(std::abs(turn[1] - straight[1]) < 1e-5f);
        CHECK(std::abs(turn[1]) > .2f);
        for (std::size_t i = 3; i < turn.size(); ++i) CHECK(turn[i] == straight[i]);
    }
}
