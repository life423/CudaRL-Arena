#include "check.h"
#include "motor/straight_task.h"
#include <limits>
#include <filesystem>
#include "motor/frozen_controller.h"
namespace speed=cudarl::straight_motor;
TEST_CASE("speed_request: target is observed and each speed has fixed relative accuracy (zero integration)") {
    for(float target:{.004f,.005f,.006f,.007f,.008f}) {
        const auto config=speed::config_for_speed(target);CHECK(speed::valid(config));
        auto world=speed::initial_world(0);speed::ForwardTask task;CHECK(speed::begin_forward(world,config,task));
        float observation[speed::kObservations]{};CHECK(speed::observe(task,world,config,observation));
        CHECK(std::fabs(observation[3]-target/.02f)<1e-6f);
        CHECK(std::fabs(config.speed_rmse_limit-.3125f*target)<1e-8f);
        CHECK(world.attempted_substeps==0);
    }
}
TEST_CASE("speed_request: switching preserves world, targets, heading and lifetime work (zero integration)") {
    auto world=speed::initial_world(0);auto config=speed::config_for_speed(.006f);speed::ForwardTask task;
    CHECK(speed::begin_forward(world,config,task));const auto heading=task.request.heading;
    world.physical.held_targets[0]=.55f;world.reference_rate[0]=.2f;world.signed_work=2e-6;
    world.physical.body.work=3e-6f;world.physical.control_steps=80;world.physical.time_seconds=4;
    CHECK(speed::request_speed(world,.004f,config,task));
    CHECK(world.physical.held_targets[0]==.55f&&world.reference_rate[0]==.2f);
    CHECK(world.physical.body.work==3e-6f&&world.signed_work==2e-6);
    CHECK(task.start_work_J==3e-6f&&task.controls==0&&task.last_world_control==80);
    CHECK(task.request.heading.x==heading.x&&task.request.heading.z==heading.z);
    CHECK(world.physical.time_seconds==4&&world.attempted_substeps==0);
    const auto saved=task;CHECK(!speed::request_speed(world,0,config,task));
    CHECK(!speed::request_speed(world,std::numeric_limits<float>::quiet_NaN(),config,task));
    CHECK(task.controls==saved.controls&&config.target_speed_m_s==.004f);
}
TEST_CASE("speed_request: segment effort excludes previous segments without erasing physical accounting") {
    auto world=speed::initial_world(0);auto config=speed::config_for_speed(.008f);speed::ForwardTask task;
    world.physical.body.work=.001f;CHECK(speed::begin_forward(world,config,task));
    const auto start=task.start_com_model;world.physical.body.velocity[0]=config.target_speed_m_s;
    const auto heading=task.request.heading;world.physical.body.velocity[0]=heading.x*config.target_speed_m_s;world.physical.body.velocity[2]=heading.z*config.target_speed_m_s;
    for(int tick=1;tick<=config.maximum_controls;++tick){const auto now=cudarl::motor::motion(world.physical.body).com_model;
        world.physical.body.position+=start+heading*(config.target_speed_m_s*cudarl::physics::dt*float(tick)/cudarl::physics::scale)-now;
        world.physical.control_steps=tick;CHECK(speed::record_control(task,world,config).accepted);}
    CHECK(task.outcome==speed::Outcome::Success&&world.physical.body.work==.001f);
    CHECK(world.attempted_substeps==0);
}

TEST_CASE("speed_request: absent or invalid policy never starts or advances physics (zero integration)") {
    const auto root=std::filesystem::path(__FILE__).parent_path().parent_path();cudarl::motor::FrozenController controller;
    CHECK(!controller.load((root/"live/assets/frozen-controller.bin").string()));CHECK(!controller.loaded()&&controller.paused());
    CHECK(!controller.step()&&controller.world().attempted_substeps==0);
    CHECK(!controller.start()&&!controller.set_speed(.008f));
    CHECK(!controller.load((root/"CMakeLists.txt").string()));CHECK(!controller.start()&&!controller.step());
    CHECK(controller.world().attempted_substeps==0);
}
