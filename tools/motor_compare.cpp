// CPU-only diagnostic. The historical reference is never passed to a learner.
#include "core/rng.h"
#include "motor/straight_task.h"
#include "nn/mlp_cpu.h"
#include "rl/gaussian_math.h"
#include "rl/ppo_checkpoint.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
namespace {
namespace fs = std::filesystem;
namespace m = cudarl::straight_motor;
namespace p = cudarl::physics;
namespace cp = cudarl::checkpoint;
constexpr uint32_t policy_seed = 1247037233, noise_seed = 0x44494147;
constexpr int controls_per_arm = 160, control_cap = 640, substep_cap = 6400;
auto started = std::chrono::steady_clock::now();
int controls = 0, substeps = 0;
int active_control_cap = control_cap, active_substep_cap = substep_cap;
double active_wall_cap = 5;
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
double elapsed() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); }
// Equations/axis order from d231474:src/nn/motor/stroke_motor.h::step_reference.
// Fixed A=.6rad,f=1Hz also appears in that commit's movement_model.h::prescribed.
// Sample the historical generator at the FIRST5ms sample of each50ms interval.
// Only angle targets enter the NEW target filter/servo; no old rate feedforward.
struct HistoricalReference {
    float parameter[2]{0, 1}, rate[2]{}, phase = 0;
    float desired[2]{.6f, 1.f}, rear_lag = cudarl::movement::pi / 2;
    void targets(float *command) {
        constexpr float h = .005f, omega = 2 * cudarl::movement::pi;
        for (int k = 0; k < 10; ++k) {
            for (int j = 0; j < 2; ++j) {
                const float a = omega * omega * (desired[j] - parameter[j]) - 2 * omega * rate[j];
                rate[j] += h * a;
                parameter[j] += h * rate[j];
            }
            if (k == 0) {
                for (int j = 0; j < 4; ++j)
                    command[j] = 0;
                for (int j = 0; j < 4; j += 2)
                    command[j] = parameter[0] * sinf(phase - (j == 2 ? rear_lag : 0)) / p::limit;
            }
            phase += h * 2 * cudarl::movement::pi * parameter[1];
            if (phase > 2 * cudarl::movement::pi)
                phase -= 2 * cudarl::movement::pi;
            if (phase < 0)
                phase += 2 * cudarl::movement::pi;
        }
    }
};
std::vector<uint8_t> protocol(const std::vector<uint8_t> &actor, const std::vector<uint8_t> &source) {
    cp::Writer out;
    out.value(uint32_t{0x504d434d});
    out.value(uint32_t{1});
    out.value(cp::checksum(actor));
    out.value(cp::checksum(source));
    for (int v : {6, m::kTaskVersion, m::kRewardVersion, 3, 0, controls_per_arm, control_cap, substep_cap})
        out.value(v);
    out.value(noise_seed);
    for (float v : {.008f, .6f, 1.f, .05f, .005f, .99f, 5.f})
        out.value(v);
    return out.data;
}
cudarl::rl::FrozenPpoActor load(const fs::path &root) {
    auto actor = cudarl::rl::read_ppo_actor(
        cp::read((root / "checkpoint.bin").string()),
        cp::checksum(cp::read((root / "source-ppo-protocol.bin").string())), policy_seed);
    require(actor.update == 128, "Wrong frozen update");
    return actor;
}
int prepare(const fs::path &root, const fs::path &source) {
    require(!fs::exists(root), "Comparison identity exists");
    fs::create_directories(root);
    const auto actor = cp::read(source.string()),
               source_protocol = cp::read((source.parent_path() / "protocol.bin").string());
    cp::write_atomic((root / "checkpoint.bin").string(), actor);
    cp::write_atomic((root / "source-ppo-protocol.bin").string(), source_protocol);
    (void)load(root);
    cp::write_atomic((root / "protocol.bin").string(), protocol(actor, source_protocol));
    std::ofstream out(root / "task-contract.json");
    out << "{\"diagnostic_only\":true,\"training\":false,\"model\":6,\"task\":6,\"reward_version\":4,"
           "\"decision_ms\":50,\"physics_ms\":5,\"pose\":0,\"seconds_per_arm\":8,\"mandatory_arms\":["
           "\"neutral\",\"historical_reference\",\"frozen_PPO_mean\"],\"historical_source_commit\":"
           "\"d2314740d9688a5147f53f1bef71ef272c650e86\",\"reference_amplitude_rad\":0.6,\"reference_cadence_"
           "Hz\":1,\"reference_tail_lag_rad\":1.57079632679,\"reference_sampling\":\"first5ms sample "
           "each50ms; new target filter and servo; old reference-rate feedforward NOT "
           "used\",\"optional_sampled_arm\":\"only if reference exceeds neutral forward distance by1um AND "
           "discounted return by1e-8\",\"noise_seed\":"
        << noise_seed
        << ",\"noise_log_std\":-1,\"reward\":\"current reward4 every50ms; posture/lane/depth/effort from "
           "first step; speed scored2-8s; report undiscounted and gamma0.99 discounted "
           "returns\",\"max_controls\":640,\"max_substeps\":6400,\"native_seconds\":5,\"process_seconds\":10,"
           "\"retries\":0,\"runtime_adoption\":false}\n";
    std::ofstream ready(root / "readiness.json");
    ready << "{\"ready\":true,\"diagnostic_only\":true,\"physical_substeps\":0,\"optimizer_steps\":0,"
             "\"controls_cap\":640,\"substeps_cap\":6400}\n";
    return 0;
}
struct Stroke {
    float amplitude = .6f, frequency = 1, lag = cudarl::movement::pi / 2;
    int filter_hz = 1;
};
struct Arm {
    std::string name;
    std::string error;
    Stroke stroke;
    uint32_t sample_seed = 0;
    bool passed = false;
    double reward = 0, discounted_return = 0, cost[5]{};
    float forward = 0, lane = 0, depth = 0, tilt = 0, speed = 0, rmse = 0, work = 0, maximum_torque = 0,
          path = 0;
    double angle_error[4]{}, filtered_error[4]{}, rate_error[4]{};
    double harmonic_sin[4][4]{}, harmonic_cos[4][4]{};
    int saturations = 0;
};
Arm execute(const std::string &name, int kind, const cudarl::rl::FrozenPpoActor &checkpoint,
            const fs::path &root, uint32_t sample_seed = noise_seed, Stroke stroke = {}) {
    Arm result;
    result.name = name;
    result.stroke = stroke;
    result.sample_seed = kind == 3 ? sample_seed : 0;
    auto world = m::initial_world(0);
    auto config = m::config_for_speed(.008f);
    m::ForwardTask task;
    require(m::begin_forward(world, config, task), "Invalid comparison reset");
    std::unique_ptr<cudarl::nn::CpuMlp<float>> actor;
    if (kind >= 2) {
        actor = std::make_unique<cudarl::nn::CpuMlp<float>>(cudarl::nn::MlpSpec{
            {40, 32, 32, 4},
            {cudarl::nn::Activation::Tanh, cudarl::nn::Activation::Tanh, cudarl::nn::Activation::Identity}});
        actor->params() = checkpoint.parameters;
    }
    HistoricalReference reference;
    reference.desired[0] = stroke.amplitude;
    reference.desired[1] = stroke.frequency;
    reference.rear_lag = stroke.lag;
    std::ofstream trace(root / (std::string(name) + "-controls.csv"));
    trace << "control,time_s,forward_mm,lane_mm,depth_mm,head_tilt_deg,reward,speed_cost,lane_cost,depth_"
             "cost,posture_cost,effort_cost,work_J,sampled_logp,COM_forward_mm_s,COM_side_mm_s,COM_vertical_"
             "mm_s,raw_vector_error_sq_mm2_s2,COM_x_m,COM_y_m,COM_z_m";
    for (int j = 0; j < 4; ++j)
        trace << ",command_angle" << j << ",requested_angle" << j << ",latent" << j << ",command_rate_fd" << j
              << ",filtered_angle" << j << ",filtered_rate" << j << ",actual_angle" << j << ",actual_rate"
              << j << ",last_torque_Nm" << j;
    trace << '\n';
    float previous[4]{};
    double discount = 1;
    for (int tick = 0; tick < controls_per_arm; ++tick) {
        require(elapsed() < active_wall_cap, "Native wall cap reached");
        require(controls < active_control_cap && substeps + 10 <= active_substep_cap,
                "Comparison physical budget exhausted");
        float command[4]{}, logp = 0;
        float requested[4]{}, latent[4]{};
        if (kind == 1) {
            reference.targets(command);
            for (int j = 0; j < 4; ++j) {
                requested[j] = command[j] * p::limit;
                require(fabsf(command[j]) < 1, "Reference cannot encode within action range");
                latent[j] = atanhf(command[j]);
            }
            m::Action decoded;
            require(m::action_from_latent(latent, decoded), "Reference decoder rejected");
            for (int j = 0; j < 4; ++j)
                command[j] = decoded.command[j];
        }
        if (kind >= 2) {
            float obs[40]{};
            require(m::observe(task, world, config, obs), "Invalid frozen observation");
            actor->forward(obs, 1);
            for (int j = 0; j < 4; ++j)
                latent[j] = actor->output()[j] +
                            (kind == 3 ? expf(checkpoint.log_sigma[j]) *
                                             cudarl::rng::normal(sample_seed, static_cast<uint32_t>(tick),
                                                                 static_cast<uint32_t>(j))
                                       : 0.f);
            m::Action action;
            require(m::action_from_latent(latent, action), "Invalid frozen action");
            for (int j = 0; j < 4; ++j)
                command[j] = action.command[j];
            for (int j = 0; j < 4; ++j)
                requested[j] = command[j] * p::limit;
            if (kind == 3) {
                logp = cudarl::rl::gaussian_logp(actor->output(), checkpoint.log_sigma.data(), latent, 4);
                require(std::isfinite(logp), "Invalid sampled likelihood");
            }
        }
        const auto before = world.attempted_substeps;
        ++controls;
        const bool ok = stroke.filter_hz == 3 ? cudarl::joint_motor::step_state<3>(world, command)
                                              : m::step_state(world, command);
        substeps += static_cast<int>(world.attempted_substeps - before);
        if (!ok || world.attempted_substeps - before != 10)
            throw std::runtime_error("Physical safety stop code" + std::to_string(world.failure_code));
        const auto interval = m::record_control(task, world, config);
        require(interval.accepted, "Task rejected comparison control");
        result.reward += interval.reward;
        result.discounted_return += discount * interval.reward;
        discount *= .99;
        const float cost[5]{interval.costs.speed, interval.costs.lane, interval.costs.depth,
                            interval.costs.posture, interval.costs.effort};
        for (int j = 0; j < 5; ++j)
            result.cost[j] += .05 * cost[j];
        for (float torque : world.physical.actual_torques) {
            result.maximum_torque = fmaxf(result.maximum_torque, fabsf(torque) * p::torque);
            if (fabsf(torque) >= .999999f)
                ++result.saturations;
        }
        trace << tick + 1 << ',' << (tick + 1) * .05f << ',' << interval.metrics.forward_m * 1000 << ','
              << interval.metrics.lateral_m * 1000 << ',' << interval.metrics.depth_m * 1000 << ','
              << interval.metrics.tilt_deg << ',' << interval.reward;
        for (float value : cost)
            trace << ',' << value;
        trace << ',' << world.physical.body.work << ',' << logp;
        const auto motion = cudarl::motor::motion(world.physical.body);
        const auto sideways = p::cross(task.request.heading, {0, 1, 0});
        const auto com = motion.com_model * p::scale;
        trace << ',' << p::dot(motion.velocity_m_s, task.request.heading) * 1000 << ','
              << p::dot(motion.velocity_m_s, sideways) * 1000 << ',' << motion.velocity_m_s.y * 1000 << ','
              << interval.metrics.speed_error_m_s * interval.metrics.speed_error_m_s * 1e6f << ',' << com.x
              << ',' << com.y << ',' << com.z;
        for (int j = 0; j < 4; ++j) {
            const float rate = (command[j] * p::limit - previous[j]) / .05f;
            const float error = command[j] * p::limit - world.physical.body.angle[j],
                        filtered = world.physical.held_targets[j] - world.physical.body.angle[j],
                        rate_error = world.reference_rate[j] - world.physical.body.velocity[6 + j];
            result.angle_error[j] += error * error;
            result.filtered_error[j] += filtered * filtered;
            result.rate_error[j] += rate_error * rate_error;
            if (tick >= 40 && kind == 1) {
                const double phase = 2 * cudarl::movement::pi * stroke.frequency * (tick + 1) * .05;
                const float values[4]{requested[j], command[j] * p::limit, world.physical.held_targets[j],
                                      world.physical.body.angle[j]};
                for (int s = 0; s < 4; ++s) {
                    result.harmonic_sin[s][j] += values[s] * sin(phase);
                    result.harmonic_cos[s][j] += values[s] * cos(phase);
                }
            }
            trace << ',' << command[j] * p::limit << ',' << requested[j] << ',' << latent[j] << ',' << rate
                  << ',' << world.physical.held_targets[j] << ',' << world.reference_rate[j] << ','
                  << world.physical.body.angle[j] << ',' << world.physical.body.velocity[6 + j] << ','
                  << world.physical.last_actual_torques[j] * p::torque;
            previous[j] = command[j] * p::limit;
        }
        trace << '\n';
    }
    result.forward = m::measure(task, world).forward_m;
    result.lane = task.maximum_lateral;
    result.depth = task.maximum_depth;
    result.tilt = task.maximum_tilt;
    result.speed = task.steady_speed_m_s;
    result.rmse = task.speed_rmse;
    result.work = world.physical.body.work;
    result.passed = task.outcome == m::Outcome::Success;
    result.path = task.path_angle_deg;
    return result;
}
void write_results(const fs::path &root, const std::vector<Arm> &arms, const std::string &fault, bool sampled,
                   bool checkpoint_unchanged, bool weights_used = true) {
    std::ofstream out(root / "result.json");
    out << std::setprecision(9)
        << "{\"diagnostic_only\":true,\"training\":false,\"optimizer_updates\":0,\"completed_updates\":0,"
           "\"candidate_accepted\":false,\"runtime_adopted\":false,\"fault\":"
        << std::quoted(fault) << ",\"controls\":" << controls << ",\"physical_substeps\":" << substeps
        << ",\"sampled_arm_triggered\":" << (sampled ? "true" : "false")
        << ",\"native_seconds\":" << elapsed() << ",\"arms\":[";
    for (size_t i = 0; i < arms.size(); ++i) {
        const auto &a = arms[i];
        if (i)
            out << ',';
        out << "{\"name\":" << std::quoted(a.name) << ",\"noise_seed\":" << a.sample_seed
            << ",\"error\":" << std::quoted(a.error) << ",\"filter_Hz\":" << a.stroke.filter_hz
            << ",\"amplitude_rad\":" << a.stroke.amplitude << ",\"frequency_Hz\":" << a.stroke.frequency
            << ",\"rear_lag_deg\":" << a.stroke.lag * 180 / cudarl::movement::pi
            << ",\"passed_original_gates\":" << (a.passed ? "true" : "false")
            << ",\"forward_mm\":" << a.forward * 1000 << ",\"lane_max_mm\":" << a.lane * 1000
            << ",\"full8s_average_mm_s\":" << a.forward * 1000 / 8 << ",\"depth_max_mm\":" << a.depth * 1000
            << ",\"head_tilt_max_deg\":" << a.tilt << ",\"steady_path_angle_deg\":" << a.path
            << ",\"steady_speed_mm_s\":" << a.speed * 1000 << ",\"RMSE_mm_s\":" << a.rmse * 1000
            << ",\"work_J\":" << a.work << ",\"maximum_actual_torque_Nm\":" << a.maximum_torque
            << ",\"saturated_axis_substeps\":" << a.saturations << ",\"reward\":" << a.reward
            << ",\"discounted_return_gamma_0_99\":" << a.discounted_return << ",\"cost_integrals\":[";
        for (int j = 0; j < 5; ++j) {
            if (j)
                out << ',';
            out << a.cost[j];
        }
        out << "],\"command_angle_RMSE_rad\":[";
        for (int j = 0; j < 4; ++j) {
            if (j)
                out << ',';
            out << sqrt(a.angle_error[j] / 160);
        }
        out << "],\"filtered_angle_RMSE_rad\":[";
        for (int j = 0; j < 4; ++j) {
            if (j)
                out << ',';
            out << sqrt(a.filtered_error[j] / 160);
        }
        out << "],\"filtered_rate_RMSE_rad_s\":[";
        for (int j = 0; j < 4; ++j) {
            if (j)
                out << ',';
            out << sqrt(a.rate_error[j] / 160);
        }
        out << ']';
        const char *labels[4]{"requested_amplitude_rad", "decoded_amplitude_rad", "filtered_amplitude_rad",
                              "actual_amplitude_rad"};
        for (int stage = 0; stage < 4; ++stage) {
            out << ',' << std::quoted(labels[stage]) << ":[";
            for (int j = 0; j < 4; ++j) {
                if (j)
                    out << ',';
                out << 2 *
                           sqrt(a.harmonic_sin[stage][j] * a.harmonic_sin[stage][j] +
                                a.harmonic_cos[stage][j] * a.harmonic_cos[stage][j]) /
                           120;
            }
            out << ']';
        }
        out << '}';
    }
    out << "],\"policy_weights_used\":" << (weights_used ? "true" : "false")
        << ",\"checkpoint_unchanged\":" << (weights_used ? (checkpoint_unchanged ? "true" : "false") : "null")
        << ",\"retries\":0,\"closed\":true}\n";
    out.close();
    std::ifstream immediate(root / "result.json");
    std::cout << immediate.rdbuf() << std::flush;
}
std::vector<uint32_t> seeds() {
    std::vector<uint32_t> result{noise_seed};
    for (uint32_t i = 1; i < 20; ++i) {
        const auto value = cudarl::rng::bits(noise_seed, 0x53454544u, i);
        for (uint32_t previous : result)
            require(previous != value, "Duplicate seed; no run launched");
        result.push_back(value);
    }
    return result;
}
std::vector<uint8_t> seed_protocol(const std::vector<uint8_t> &actor, const std::vector<uint8_t> &source) {
    cp::Writer out;
    out.value(uint32_t{0x534d434d});
    out.value(uint32_t{1});
    out.value(cp::checksum(protocol(actor, source)));
    out.array(seeds());
    out.value(uint32_t{3040});
    out.value(uint32_t{30400});
    return out.data;
}
int prepare_seeds(const fs::path &root, const fs::path &reused) {
    require(!fs::exists(root), "Seed identity exists");
    const auto actor = cp::read((reused / "checkpoint.bin").string()),
               source = cp::read((reused / "source-ppo-protocol.bin").string());
    require(cp::read((reused / "protocol.bin").string()) == protocol(actor, source),
            "Reused comparison protocol differs");
    require(fs::exists(reused / "run-started") && fs::exists(reused / "result.json"),
            "Missing completed comparison evidence");
    fs::create_directories(root);
    cp::write_atomic((root / "checkpoint.bin").string(), actor);
    cp::write_atomic((root / "source-ppo-protocol.bin").string(), source);
    (void)load(root);
    cp::write_atomic((root / "protocol.bin").string(), seed_protocol(actor, source));
    fs::copy_file(reused / "result.json", root / "reused-result.json");
    fs::copy_file(reused / "frozen_PPO_sampled-controls.csv", root / "reused-sampled-controls.csv");
    std::ofstream out(root / "task-contract.json");
    out << "{\"diagnostic_only\":true,\"training\":false,\"pose\":0,\"seconds_per_seed\":8,\"decision_ms\":"
           "50,\"physics_ms\":5,\"model\":6,\"task\":6,\"reward\":4,\"reused_seed_count\":1,\"new_seed_"
           "count\":19,\"seeds\":[";
    const auto list = seeds();
    for (size_t i = 0; i < list.size(); ++i) {
        if (i)
            out << ',';
        out << list[i];
    }
    out << "],\"seed_generation\":\"first prior seed; then core rng::bits(base,0x53454544,index1to19), fixed "
           "before "
           "execution\",\"controls_cap\":3040,\"substeps_cap\":30400,\"native_seconds\":5,\"process_"
           "seconds\":10,\"retries\":0}\n";
    std::ofstream ready(root / "readiness.json");
    ready << "{\"ready\":true,\"seed_batch\":true,\"physical_substeps\":0,\"optimizer_updates\":0}\n";
    return 0;
}
constexpr float grid_amplitudes[3]{.3f, .6f, .9f}, grid_frequencies[3]{.5f, 1.f, 1.5f};
constexpr float grid_lags[3]{cudarl::movement::pi / 4, cudarl::movement::pi / 2,
                             3 * cudarl::movement::pi / 4};
std::vector<uint8_t> grid_protocol() {
    cp::Writer out;
    out.value(uint32_t{0x474d434d});
    out.value(uint32_t{1});
    for (int v : {6, m::kTaskVersion, m::kRewardVersion, 3, 0, 50, 5, 160, 27, 54, 8640, 86400, 1, 3})
        out.value(v);
    for (float value : grid_amplitudes)
        out.value(value);
    for (float value : grid_frequencies)
        out.value(value);
    for (float value : grid_lags)
        out.value(value);
    out.value(float{10});
    out.value(float{15});
    return out.data;
}
int prepare_grid(const fs::path &root) {
    require(!fs::exists(root), "Grid identity exists");
    int checked = 0;
    for (float amplitude : grid_amplitudes)
        for (float frequency : grid_frequencies)
            for (float lag : grid_lags) {
                HistoricalReference ref;
                ref.desired[0] = amplitude;
                ref.desired[1] = frequency;
                ref.rear_lag = lag;
                for (int tick = 0; tick < 160; ++tick) {
                    float command[4], latent[4];
                    ref.targets(command);
                    for (int j = 0; j < 4; ++j) {
                        require(fabsf(command[j]) < 1, "Grid command out of range");
                        latent[j] = atanhf(command[j]);
                    }
                    m::Action decoded;
                    require(m::action_from_latent(latent, decoded), "Grid decoder rejection");
                    for (int j = 0; j < 4; ++j)
                        require(fabsf(decoded.command[j] - command[j]) < 1e-6f,
                                "Grid decoder roundtrip error");
                    ++checked;
                }
            }
    fs::create_directories(root);
    cp::write_atomic((root / "protocol.bin").string(), grid_protocol());
    std::ofstream out(root / "task-contract.json");
    out << "{\"diagnostic_only\":true,\"training\":false,\"weights_used\":false,\"model\":6,\"task\":6,"
           "\"reward\":4,\"pose\":0,\"still_water\":true,\"decision_ms\":50,\"physics_ms\":5,\"seconds_per_"
           "case\":8,\"speed_scored_seconds\":[2,8],\"amplitudes_rad\":[0.3,0.6,0.9],\"frequencies_Hz\":[0.5,"
           "1,1.5],\"rear_lags_deg\":[45,90,135],\"default_filter_Hz\":1,\"conditional_filter_Hz\":3,"
           "\"conditional_rule\":\"run same27 with3Hz ONLY if no current1Hz case passes all original "
           "gates\",\"reference_ramp\":\"historical1Hz parameter filter; amplitude starts0,cadence starts1; "
           "no old derivative feedforward\",\"input_path\":\"reference angle/limit -> atanh latent -> same "
           "action_from_latent tanh decoder -> held normalized command for10physics substeps -> same bounded "
           "reference "
           "filter/"
           "servo\",\"limits\":{\"angle_rad\":1,\"reference_rate_rad_s\":12,\"reference_acceleration_rad_"
           "s2\":200,\"torque_Nm\":0.00001},\"gates\":{\"steady_speed_mm_s\":[7.2,8.8],\"vector_RMSE_mm_s\":"
           "2.5,\"lane_mm\":2,\"depth_mm\":1,\"head_tilt_deg\":10,\"path_deg\":5,\"work_J_per_m\":0.001},"
           "\"safety_rule\":\"stop failed case without retry; continue remaining preregistered cases from "
           "fresh identical rest state; stop campaign on wall/budget "
           "cap\",\"max_cases\":54,\"controls_cap\":8640,\"substeps_cap\":86400,\"native_seconds\":10,"
           "\"process_seconds\":15,\"runtime_default_unchanged\":true,\"retries\":0}\n";
    std::ofstream ready(root / "readiness.json");
    ready << "{\"ready\":true,\"decoder_decision_checks\":" << checked
          << ",\"physical_substeps\":0,\"optimizer_updates\":0}\n";
    std::cout << "READY grid decoder checks=" << checked << " physical_substeps=0 optimizer_updates=0\n";
    return 0;
}
int run_grid(const fs::path &root) {
    require(cp::read((root / "protocol.bin").string()) == grid_protocol(), "Frozen grid identity differs");
    require(fs::create_directory(root / "run-started"), "Grid already attempted; no retry");
    active_control_cap = 8640;
    active_substep_cap = 86400;
    active_wall_cap = 10;
    started = std::chrono::steady_clock::now();
    std::cout << "GRID_STARTED unix_seconds=" << std::setprecision(17)
              << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()
              << std::endl;
    std::vector<Arm> arms;
    std::string fault;
    int current_passes = 0;
    try {
        for (int filter : {1, 3}) {
            if (filter == 3 && current_passes > 0)
                break;
            int index = 0;
            for (float amplitude : grid_amplitudes)
                for (float frequency : grid_frequencies)
                    for (float lag : grid_lags) {
                        const Stroke stroke{amplitude, frequency, lag, filter};
                        const auto name =
                            "filter" + std::to_string(filter) + "_case" + std::to_string(index++);
                        try {
                            auto arm = execute(name, 1, {}, root, noise_seed, stroke);
                            if (filter == 1 && arm.passed)
                                ++current_passes;
                            arms.push_back(std::move(arm));
                        } catch (const std::exception &error) {
                            if (elapsed() >= active_wall_cap || controls >= active_control_cap)
                                throw;
                            Arm failed;
                            failed.name = name;
                            failed.stroke = stroke;
                            failed.error = error.what();
                            arms.push_back(std::move(failed));
                        }
                    }
        }
    } catch (const std::exception &error) {
        fault = error.what();
    }
    write_results(root, arms, fault, false, false, false);
    return fault.empty() ? 0 : 3;
}
constexpr float fine_amplitudes[3]{.88f, .89f, .9f}, fine_frequencies[3]{.95f, 1.f, 1.05f};
constexpr float fine_lags[3]{40 * cudarl::movement::pi / 180, cudarl::movement::pi / 4,
                             50 * cudarl::movement::pi / 180};
std::vector<uint8_t> fine_protocol() {
    cp::Writer out;
    out.value(uint32_t{0x464d434d});
    out.value(uint32_t{1});
    for (int v : {6, m::kTaskVersion, m::kRewardVersion, 3, 0, 50, 5, 160, 27, 4320, 43200, 1})
        out.value(v);
    for (float v : fine_amplitudes)
        out.value(v);
    for (float v : fine_frequencies)
        out.value(v);
    for (float v : fine_lags)
        out.value(v);
    out.value(float{10});
    out.value(float{15});
    return out.data;
}
int prepare_fine(const fs::path &root) {
    require(!fs::exists(root), "Fine grid identity exists");
    int checks = 0;
    for (float a : fine_amplitudes)
        for (float f : fine_frequencies)
            for (float lag : fine_lags) {
                HistoricalReference ref;
                ref.desired[0] = a;
                ref.desired[1] = f;
                ref.rear_lag = lag;
                for (int tick = 0; tick < 160; ++tick) {
                    float u[4], z[4];
                    ref.targets(u);
                    for (int j = 0; j < 4; ++j) {
                        require(fabsf(u[j]) < 1, "Fine action range invalid");
                        z[j] = atanhf(u[j]);
                    }
                    m::Action action;
                    require(m::action_from_latent(z, action), "Fine decoder invalid");
                    for (int j = 0; j < 4; ++j)
                        require(fabsf(action.command[j] - u[j]) < 1e-6, "Fine decoder mismatch");
                    ++checks;
                }
            }
    fs::create_directories(root);
    cp::write_atomic((root / "protocol.bin").string(), fine_protocol());
    std::ofstream out(root / "task-contract.json");
    out << "{\"diagnostic_only\":true,\"training\":false,\"weights_used\":false,\"model\":6,\"task\":6,"
           "\"reward\":4,\"pose\":0,\"still_water\":true,\"filter_Hz\":1,\"decision_ms\":50,\"physics_ms\":5,"
           "\"seconds_per_case\":8,\"speed_scoring_seconds\":[2,8],\"state_costs_from_first_step\":true,"
           "\"amplitudes_rad\":[0.88,0.89,0.90],\"frequencies_Hz\":[0.95,1,1.05],\"rear_lags_deg\":[40,45,50]"
           ",\"baseline\":\"replay0.9rad1Hz45deg FIRST as one of27, to record missing actual COM velocities "
           "and raw startup vector error; skip duplicate grid cell\",\"input_path\":\"atanh reference target "
           "-> same tanh decoder -> 50ms held command -> same1Hz target filter and3Hz torque-limited "
           "physical "
           "servo\",\"all_previous_gates_unchanged\":true,\"controls_cap\":4320,\"substeps_cap\":43200,"
           "\"native_seconds\":10,\"process_seconds\":15,\"retries\":0,\"runtime_adoption\":false}\n";
    std::ofstream ready(root / "readiness.json");
    ready << "{\"ready\":true,\"decoder_decision_checks\":" << checks
          << ",\"physical_substeps\":0,\"optimizer_updates\":0}\n";
    std::cout << "READY fine grid checks=" << checks << " physical_substeps=0\n";
    return 0;
}
int run_fine(const fs::path &root) {
    require(cp::read((root / "protocol.bin").string()) == fine_protocol(), "Fine grid identity differs");
    require(fs::create_directory(root / "run-started"), "Fine grid already attempted");
    active_control_cap = 4320;
    active_substep_cap = 43200;
    active_wall_cap = 10;
    started = std::chrono::steady_clock::now();
    std::cout << "FINE_GRID_STARTED unix_seconds=" << std::setprecision(17)
              << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()
              << std::endl;
    std::vector<Arm> arms;
    std::string fault;
    try {
        arms.push_back(execute("baseline_090_100_045", 1, {}, root, noise_seed,
                               Stroke{.9f, 1.f, cudarl::movement::pi / 4, 1}));
        int index = 0;
        for (float amplitude : fine_amplitudes)
            for (float frequency : fine_frequencies)
                for (float lag : fine_lags) {
                    const auto name = "fine_case" + std::to_string(index++);
                    if (amplitude == .9f && frequency == 1.f && lag == cudarl::movement::pi / 4)
                        continue;
                    const Stroke stroke{amplitude, frequency, lag, 1};
                    try {
                        arms.push_back(execute(name, 1, {}, root, noise_seed, stroke));
                    } catch (const std::exception &error) {
                        if (elapsed() >= active_wall_cap || controls >= active_control_cap)
                            throw;
                        Arm failed;
                        failed.name = name;
                        failed.stroke = stroke;
                        failed.error = error.what();
                        arms.push_back(std::move(failed));
                    }
                }
    } catch (const std::exception &error) {
        fault = error.what();
    }
    write_results(root, arms, fault, false, false, false);
    return fault.empty() ? 0 : 3;
}
constexpr float rest_amplitudes[3]{.088f, .22f, .44f};
std::vector<uint8_t> rest_protocol() {
    cp::Writer out;
    out.value(uint32_t{0x524d434d});
    out.value(uint32_t{1});
    for (int v : {6, m::kTaskVersion, m::kRewardVersion, 3, 0, 50, 5, 160, 3, 480, 4800, 1})
        out.value(v);
    for (float a : rest_amplitudes)
        out.value(a);
    for (float f : {1.05f, 50 * cudarl::movement::pi / 180, 5.f, 10.f})
        out.value(f);
    return out.data;
}
int prepare_rest(const fs::path &root) {
    require(!fs::exists(root), "Near-rest identity exists");
    int checks = 0;
    for (float amplitude : rest_amplitudes) {
        HistoricalReference ref;
        ref.desired[0] = amplitude;
        ref.desired[1] = 1.05f;
        ref.rear_lag = 50 * cudarl::movement::pi / 180;
        for (int tick = 0; tick < 160; ++tick) {
            float u[4], z[4];
            ref.targets(u);
            for (int j = 0; j < 4; ++j) {
                require(fabsf(u[j]) < 1, "Near-rest input out of range");
                z[j] = atanhf(u[j]);
            }
            m::Action action;
            require(m::action_from_latent(z, action), "Near-rest decoder rejected");
            for (int j = 0; j < 4; ++j)
                require(fabsf(u[j] - action.command[j]) < 1e-6, "Near-rest decoder mismatch");
            ++checks;
        }
    }
    fs::create_directories(root);
    cp::write_atomic((root / "protocol.bin").string(), rest_protocol());
    std::ofstream out(root / "task-contract.json");
    out << "{\"diagnostic_only\":true,\"training\":false,\"weights_used\":false,\"model\":6,\"task\":6,"
           "\"reward\":4,\"pose\":0,\"still_water\":true,\"filter_Hz\":1,\"decision_ms\":50,\"physics_ms\":5,"
           "\"seconds_per_case\":8,\"amplitudes_rad\":[0.088,0.22,0.44],\"cadence_Hz\":1.05,\"rear_lag_deg\":"
           "50,\"reused_neutral\":\"joint-target-reward-comparison-v1/"
           "neutral\",\"reused_full_amplitude\":\"joint-target-fine-grid-v1/"
           "fine_case8\",\"speed_scoring_seconds\":[2,8],\"state_and_effort_costs_active_from_first_"
           "interval\":true,\"input_path\":\"same50ms four-latent decoder/hold/filter/servo; reference "
           "fixture "
           "only\",\"all_limits_and_gates_unchanged\":true,\"controls_cap\":480,\"substeps_cap\":4800,"
           "\"native_seconds\":5,\"process_seconds\":10,\"retries\":0}\n";
    std::ofstream ready(root / "readiness.json");
    ready << "{\"ready\":true,\"decoder_decision_checks\":" << checks
          << ",\"physical_substeps\":0,\"optimizer_updates\":0}\n";
    std::cout << "READY near-rest checks=" << checks << " physical_substeps=0\n";
    return 0;
}
int run_rest(const fs::path &root) {
    require(cp::read((root / "protocol.bin").string()) == rest_protocol(), "Near-rest protocol differs");
    require(fs::create_directory(root / "run-started"), "Near-rest already attempted");
    active_control_cap = 480;
    active_substep_cap = 4800;
    active_wall_cap = 5;
    started = std::chrono::steady_clock::now();
    std::cout << "NEAR_REST_STARTED unix_seconds=" << std::setprecision(17)
              << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()
              << std::endl;
    std::vector<Arm> arms;
    std::string fault;
    try {
        int index = 0;
        for (float amplitude : rest_amplitudes)
            arms.push_back(execute("weak_case" + std::to_string(index++), 1, {}, root, noise_seed,
                                   Stroke{amplitude, 1.05f, 50 * cudarl::movement::pi / 180, 1}));
    } catch (const std::exception &error) {
        fault = error.what();
    }
    write_results(root, arms, fault, false, false, false);
    return fault.empty() ? 0 : 3;
}
int run(const fs::path &root, bool seed_batch = false) {
    const auto bytes = cp::read((root / "checkpoint.bin").string()),
               source = cp::read((root / "source-ppo-protocol.bin").string());
    require(cp::read((root / "protocol.bin").string()) ==
                (seed_batch ? seed_protocol(bytes, source) : protocol(bytes, source)),
            "Frozen comparison identity changed");
    const auto actor = load(root);
    require(fs::create_directory(root / "run-started"), "Comparison already attempted");
    if (seed_batch) {
        active_control_cap = 3040;
        active_substep_cap = 30400;
    }
    started = std::chrono::steady_clock::now();
    const double start =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::cout << "COMPARISON_STARTED unix_seconds=" << std::setprecision(17) << start << std::endl;
    std::vector<Arm> arms;
    std::string fault;
    bool sampled = false;
    try {
        if (seed_batch) {
            const auto list = seeds();
            for (size_t i = 1; i < list.size(); ++i)
                arms.push_back(execute("sampled_seed_" + std::to_string(list[i]), 3, actor, root, list[i]));
        } else {
            arms.push_back(execute("neutral", 0, actor, root));
            arms.push_back(execute("historical_reference", 1, actor, root));
            arms.push_back(execute("frozen_PPO_mean", 2, actor, root));
            sampled = arms[1].forward > arms[0].forward + 1e-6f &&
                      arms[1].discounted_return > arms[0].discounted_return + 1e-8;
            if (sampled)
                arms.push_back(execute("frozen_PPO_sampled", 3, actor, root));
        }
        require(cp::read((root / "checkpoint.bin").string()) == bytes,
                "Frozen checkpoint changed during comparison");
    } catch (const std::exception &error) {
        fault = error.what();
    }
    const bool checkpoint_unchanged = cp::read((root / "checkpoint.bin").string()) == bytes;
    if (!checkpoint_unchanged && fault.empty())
        fault = "Frozen checkpoint changed";
    write_results(root, arms, fault, sampled, checkpoint_unchanged);
    return fault.empty() ? 0 : 3;
}
} // namespace
int main(int argc, char **argv) {
    try {
        require(argc >= 3, "Use --prepare DIR CHECKPOINT or --run DIR");
        const auto root = fs::absolute(argv[2]);
        if (std::string(argv[1]) == "--prepare-rest") {
            require(argc == 3, "Unexpected near-rest arguments");
            return prepare_rest(root);
        }
        if (std::string(argv[1]) == "--run-rest") {
            require(argc == 3, "Unexpected near-rest arguments");
            return run_rest(root);
        }
        if (std::string(argv[1]) == "--prepare-fine") {
            require(argc == 3, "Unexpected fine arguments");
            return prepare_fine(root);
        }
        if (std::string(argv[1]) == "--run-fine") {
            require(argc == 3, "Unexpected fine arguments");
            return run_fine(root);
        }
        if (std::string(argv[1]) == "--prepare-grid") {
            require(argc == 3, "Unexpected grid arguments");
            return prepare_grid(root);
        }
        if (std::string(argv[1]) == "--run-grid") {
            require(argc == 3, "Unexpected grid arguments");
            return run_grid(root);
        }
        if (std::string(argv[1]) == "--prepare-seeds") {
            require(argc == 4, "Provide completed comparison directory");
            return prepare_seeds(root, fs::absolute(argv[3]));
        }
        if (std::string(argv[1]) == "--run-seeds") {
            require(argc == 3, "Unexpected seed arguments");
            return run(root, true);
        }
        if (std::string(argv[1]) == "--prepare") {
            require(argc == 4, "Provide frozen checkpoint");
            return prepare(root, fs::absolute(argv[3]));
        }
        require(argc == 3 && std::string(argv[1]) == "--run", "Unknown comparison mode");
        return run(root);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
