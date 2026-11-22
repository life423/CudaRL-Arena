#include "check.h"
#include "motor/straight_task.h"
#include <limits>
#include <chrono>
#include "core/checkpoint.h"
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

namespace {
namespace frozen_cp = cudarl::checkpoint;
using LiveController = cudarl::motor::FrozenController;

std::vector<uint8_t> frozen_fixture(uint32_t version, uint32_t inputs, uint32_t observation_schema,
                                   bool wrong_contract = false) {
    frozen_cp::Writer out;
    out.value(uint32_t{0x36434D46});
    out.value(version);
    if (version == 4) return out.data; // old header lacks a typed observation contract
    out.value(uint32_t{cudarl::movement::kVersion});
    out.value(uint32_t{speed::kTaskVersion});
    out.value(uint32_t{speed::kRewardVersion});
    out.value(inputs);
    out.value(uint32_t{cudarl::joint_motor::kActions});
    out.value(uint32_t{336}); // fixture label, not the real training checkpoint
    out.value(cudarl::joint_motor::kActionVersion);
    out.value(observation_schema);
    out.value(.004f);
    out.value(.008f);
    out.value(uint64_t{123});
    const std::vector<float> parameters(2500, 0.f);
    frozen_cp::Writer raw;
    for (float weight : parameters) raw.value(weight);
    out.value(frozen_cp::checksum(raw.data));
    const auto contracts = LiveController::source_contract_hashes();
    out.value(static_cast<uint32_t>(contracts.size()));
    for (const auto hash : contracts) {
        auto bytes = std::vector<uint8_t>(hash.begin(), hash.end());
        if (wrong_contract && !bytes.empty()) bytes[0] ^= 1;
        out.array(bytes);
    }
    out.array(parameters);
    return out.data;
}

class TemporaryFrozenFixture {
    std::filesystem::path directory_;
public:
    TemporaryFrozenFixture() {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = std::filesystem::temp_directory_path() / ("cudarl-frozen-fixture-" + std::to_string(nonce));
        if (!std::filesystem::create_directory(directory_)) throw std::runtime_error("Fixture directory exists");
    }
    ~TemporaryFrozenFixture() {
        std::error_code ignored;
        std::filesystem::remove(directory_ / "actor.bin", ignored);
        std::filesystem::remove(directory_ / "actor.bin.lock", ignored);
        std::filesystem::remove(directory_ / "actor.bin.tmp", ignored);
        std::filesystem::remove(directory_, ignored);
    }
    std::string path() const { return (directory_ / "actor.bin").string(); }
    void write(uint32_t version, uint32_t inputs, uint32_t observation_schema, bool wrong_contract = false) const {
        frozen_cp::write_atomic(path(), frozen_fixture(version, inputs, observation_schema, wrong_contract));
    }
};
}

TEST_CASE("frozen_viewer: explicit v5 observation scale survives speed changes and resets (zero integration)") {
    TemporaryFrozenFixture fixture;
    LiveController viewer;
    for (uint32_t schema : {cudarl::motor_observation::legacy, cudarl::motor_observation::task_scale}) {
        fixture.write(5, 40, schema);
        CHECK(viewer.load(fixture.path()));
        CHECK(viewer.loaded() && viewer.paused());
        CHECK(viewer.training_update() == 336 && viewer.source_protocol_checksum() == 123);
        CHECK(viewer.observation_schema() == schema);
        CHECK(viewer.set_speed(.006f) && viewer.set_speed(.004f));
        CHECK(viewer.observation_schema() == schema);
        CHECK(viewer.reset(3));
        CHECK(viewer.observation_schema() == schema);
        CHECK(viewer.target_speed() == .004f);
        CHECK(viewer.benchmark_controls() == speed::config_for_speed(.004f).maximum_controls);
        CHECK(viewer.benchmark_controls() * viewer.substeps_per_control() == 1600);
        CHECK(viewer.world().attempted_substeps == 0 && viewer.viewing_substeps() == 0);
    }
}

TEST_CASE("frozen_viewer: legacy ambiguous, hold, and nonmatching source contracts fail closed (zero integration)") {
    TemporaryFrozenFixture fixture;
    LiveController viewer;
    fixture.write(5, 40, cudarl::motor_observation::task_scale);
    CHECK(viewer.load(fixture.path()));
    const auto refused = [&](uint32_t version, uint32_t inputs, uint32_t schema, bool wrong_contract,
                             const std::string& expected) {
        fixture.write(version, inputs, schema, wrong_contract);
        CHECK(!viewer.load(fixture.path()));
        CHECK(viewer.error().find(expected) != std::string::npos);
        CHECK(!viewer.loaded() && viewer.paused() && !viewer.start() && !viewer.step());
        CHECK(viewer.training_update() == 0 && viewer.source_protocol_checksum() == 0);
        CHECK(viewer.world().attempted_substeps == 0 && viewer.viewing_substeps() == 0);
    };
    refused(4, 40, 0, false, "omits observation schema");
    refused(5, 41, cudarl::motor_observation::appended_hold, false, "40-input forward actors");
    refused(5, 40, cudarl::motor_observation::position_hold, false, "observation schema");
    refused(5, 40, cudarl::motor_observation::task_scale, true, "compiled source contract differ");
}
