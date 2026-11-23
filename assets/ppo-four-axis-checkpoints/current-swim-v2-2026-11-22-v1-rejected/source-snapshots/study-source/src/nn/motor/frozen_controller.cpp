#include "frozen_controller.h"
#include "core/checkpoint.h"
#include "frozen_contract.h"
#include <cmath>
#include <iterator>
#include <stdexcept>
namespace cudarl::motor {
namespace sm = straight_motor;
namespace cp = checkpoint;
namespace p = physics;
static_assert(movement::kVersion == 6 && sm::kTaskVersion == 6 && sm::kRewardVersion == 5,
              "Frozen actor contract changed");
FrozenController::FrozenController()
    : actor_({{40, 32, 32, 4}, {nn::Activation::Tanh, nn::Activation::Tanh, nn::Activation::Identity}}) {}
std::vector<std::string_view> FrozenController::source_contract_hashes() {
    return {std::begin(kFrozenContracts), std::end(kFrozenContracts)};
}
bool FrozenController::load(const std::string &path) {
    loaded_ = false;
    paused_ = true;
    error_.clear();
    training_update_ = 0;
    source_protocol_checksum_ = 0;
    try {
        const auto bytes = cp::read(path);
        cp::Reader r{bytes};
        if (r.value<uint32_t>() != 0x36434D46)
            throw std::runtime_error("Wrong frozen controller magic");
        // Package v5 separates wire format version from observation schema.
        // Header: magic/version/model/task/reward/input/output/update/mapping/
        // observation schema, then min/max speed, source protocol checksum,
        // actor checksum, source-contract hashes and actor parameters.
        const auto schema = r.value<uint32_t>();
        if (schema == 4)
            throw std::runtime_error("Legacy frozen controller omits observation schema; an explicit v5 package is required");
        if (schema != 5)
            throw std::runtime_error("Unsupported frozen controller package version");
        if (r.value<uint32_t>() != movement::kVersion || r.value<uint32_t>() != sm::kTaskVersion ||
            r.value<uint32_t>() != sm::kRewardVersion || r.value<uint32_t>() != sm::kObservations ||
            r.value<uint32_t>() != joint_motor::kActions)
            throw std::runtime_error("Wrong frozen controller model/task/reward/shape; viewer supports 40-input forward actors only");
        const auto training_update = r.value<uint32_t>();
        const auto mapping = r.value<uint32_t>();
        if (mapping != joint_motor::kActionVersion)
            throw std::runtime_error("Unsupported frozen action mapping");
        const auto observations = r.value<uint32_t>();
        if (observations != motor_observation::legacy && observations != motor_observation::task_scale)
            throw std::runtime_error("Unsupported frozen forward actor observation schema");
        const float low = r.value<float>(), high = r.value<float>();
        if (!sm::supported_speed(low) || !sm::supported_speed(high) || low > high)
            throw std::runtime_error("Invalid validated speed range");
        const auto source_protocol_checksum = r.value<uint64_t>();
        const auto actor_checksum = r.value<uint64_t>();
        if (r.value<uint32_t>() != kFrozenContractCount)
            throw std::runtime_error("Source contract count mismatch");
        for (const char *expected : kFrozenContracts) {
            const auto hash = r.array<uint8_t>(64);
            if (std::string(hash.begin(), hash.end()) != expected)
                throw std::runtime_error("Frozen actor and compiled source contract differ");
        }
        auto params = r.array<float>(2500);
        r.finish();
        if (params.size() != 2500)
            throw std::runtime_error("Wrong parameter count");
        cp::Writer actor_bytes;
        for (float value : params) {
            if (!std::isfinite(value))
                throw std::runtime_error("Nonfinite actor parameter");
            actor_bytes.value(value);
        }
        if (cp::checksum(actor_bytes.data) != actor_checksum)
            throw std::runtime_error("Frozen actor checksum differs from selected checkpoint");
        actor_.params() = std::move(params);
        minimum_speed_ = low;
        maximum_speed_ = high;
        config_ = sm::config_for_speed(high);
        config_.observation_schema = observations;
        loaded_ = true;
        if (!reset(0)) {
            loaded_ = false;
            return false;
        }
        training_update_ = training_update;
        source_protocol_checksum_ = source_protocol_checksum;
        return true;
    } catch (const std::exception &e) {
        error_ = e.what();
        return false;
    }
}
bool FrozenController::reset(int id) {
    paused_ = true;
    error_.clear();
    if (!loaded_) {
        error_ = "Load the validated frozen actor first";
        return false;
    }
    if (id < 0 || id > 3) {
        error_ = "Case must be0..3";
        return false;
    }
    viewing_controls_ = viewing_substeps_ = loops_ = 0;
    case_ = id;
    world_ = sm::initial_world(id);
    world_.arena = arena_;
    task_ = {};
    if (!sm::begin_forward(world_, config_, task_)) {
        error_ = "Invalid evaluation reset";
        return false;
    }
    return true;
}
bool FrozenController::set_speed(float speed) {
    if (!loaded_ || !std::isfinite(speed) || speed < minimum_speed_ || speed > maximum_speed_)
        return false;
    if (speed == target_speed())
        return true;
    auto next_config = config_;
    auto next_task = task_;
    if (!sm::request_speed(world_, speed, next_config, next_task))
        return false;
    next_config.observation_schema = config_.observation_schema;
    config_ = next_config;
    task_ = next_task;
    return true;
}
bool FrozenController::start() {
    if (!loaded_ || finished() || !error_.empty())
        return false;
    paused_ = false;
    return true;
}
bool FrozenController::step() {
    if (!loaded_ || paused_ || finished() || !error_.empty())
        return false;
    // An explicit viewing loop restarts from rest before the finite pond edge.
    // Invalid states still stop; a reset must never conceal a failed safety check.
    if (!sm::in_domain(world_, config_)) {
        error_ = "Viewing safety/domain stop";
        paused_ = true;
        return false;
    }
    if (continuous_ && movement::clearance(world_.physical.body, true, world_.arena) < .02f) {
        world_ = sm::initial_world(case_);
        world_.arena = arena_;
        task_ = {};
        if (!sm::begin_forward(world_, config_, task_)) {
            error_ = "Viewing loop reset failed";
            paused_ = true;
            return false;
        }
        ++loops_;
    }
    // Freeze the eight-second task score. Beyond it the trained time feature
    // stays at zero; no new benchmark, reward, optimizer or physical rescue.
    auto observation_task = task_;
    observation_task.outcome = sm::Outcome::Running;
    float observation[sm::kObservations]{};
    if (!sm::observe(observation_task, world_, config_, observation)) {
        error_ = "Invalid observation";
        paused_ = true;
        return false;
    }
    actor_.forward(observation, 1);
    sm::Action action;
    const auto before = world_.attempted_substeps;
    const bool advanced =
        sm::action_from_latent(actor_.output(), action) && joint_motor::step_state(world_, action.command);
    viewing_substeps_ += world_.attempted_substeps - before;
    if (!advanced) {
        error_ = "Physical/controller failure code " + std::to_string(world_.failure_code);
        paused_ = true;
        return false;
    }
    ++viewing_controls_;
    if (!benchmark_complete()) {
        const auto transition = sm::record_control(task_, world_, config_);
        if (!transition.accepted) {
            error_ = "Task rejected native interval";
            paused_ = true;
            return false;
        }
        if (transition.done && !continuous_)
            paused_ = true;
    }
    return true;
}
} // namespace cudarl::motor
