// One native PPO runner. Four independent joint targets; fresh initialization.
// Frozen evaluations never update the network. No preset gait or imported weights.
#include "motor/frozen_controller.h"
#include "nn/adam.h"
#include "nn/mlp_gpu.h"
#include "rl/ar_gaussian_math.h"
#include "rl/gaussian_kernels.h"
#include "rl/gaussian_math.h"
#include "rl/motor_rollout.cuh"
#include "rl/ppo_checkpoint.h"
#include "rl/ppo_diagnostics.h"
#include "rl/ppo_kernels.h"
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
namespace {
namespace fs = std::filesystem;
namespace m = cudarl::straight_motor;
namespace p = cudarl::physics;
namespace nn = cudarl::nn;
namespace rl = cudarl::rl;
namespace gpu = rl::motor_rollout;
namespace cp = cudarl::checkpoint;
constexpr int observations = m::kObservations, actions = m::kActions;
constexpr int environments = 32, minibatch = 256, epochs = 2;
m::TrainingEpisode training_episode;
int &horizon = training_episode.controls;
int rows = environments * horizon;
int updates = 128;
int training_cap = updates * rows, control_cap = training_cap + 64 + 1280 + 2400,
    substep_cap = control_cap * 10;
bool &tilt_mixture = training_episode.tilt_mixture;
bool &depth_offsets = training_episode.fixed_depth_offsets;
bool &stopping = training_episode.stopping;
int optimizer_steps_per_update() { return epochs * (rows / minibatch); }
size_t rollout_elements(int components = 1) { return static_cast<size_t>(rows) * components; }
uint32_t seed = 0x4a544731u;
float correlation = 0;
std::optional<rl::PpoTrainingState> resume_state;
std::optional<float> actor_lr_override;
fs::path parent_checkpoint_path, parent_protocol_path;
std::vector<uint8_t> parent_checkpoint_payload, parent_protocol_payload;
double active_wall_cap = 60;
bool verification_mode = false;
constexpr float log_sigma = -1, wall_cap = 60;
auto started = std::chrono::steady_clock::now();
double training_started_unix_seconds = 0;
void require(bool ok, const char *reason) {
    if (!ok)
        throw std::runtime_error(reason);
}
void wall_guard() {
    require(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() <
                active_wall_cap,
            "Native wall cap reached");
}
nn::MlpSpec spec(int output) {
    return {{observations, 32, 32, output},
            {nn::Activation::Tanh, nn::Activation::Tanh, nn::Activation::Identity}};
}
std::vector<float> fresh_actor() {
    auto result = nn::orthogonal_params(spec(actions), seed, {1.414213562f, 1.414213562f, .01f});
    return result;
}
std::vector<float> fresh_critic() {
    return nn::orthogonal_params(spec(1), seed + 1, {1.414213562f, 1.414213562f, 1.f});
}
int starting_update() { return resume_state ? resume_state->actor.update : 0; }
std::vector<float> initial_actor() { return resume_state ? resume_state->actor.parameters : fresh_actor(); }
std::vector<float> initial_critic() { return resume_state ? resume_state->critic : fresh_critic(); }
std::vector<float> policy_sigma() {
    return resume_state ? resume_state->actor.log_sigma : std::vector<float>(actions, log_sigma);
}
nn::AdamConfig initial_optimizer_config(bool actor) {
    nn::AdamConfig config;
    if (resume_state)
        config = actor ? resume_state->actor_adam.config : resume_state->critic_adam.config;
    else {
        config.lr = .001f;
        config.eps = .00001f;
        config.max_grad_norm = .5f;
    }
    if (actor && actor_lr_override)
        config.lr = *actor_lr_override;
    return config;
}
rl::PpoAdamState initial_optimizer(bool actor) {
    if (resume_state)
        return actor ? resume_state->actor_adam : resume_state->critic_adam;
    rl::PpoAdamState state;
    state.config = initial_optimizer_config(actor);
    const auto count = spec(actor ? actions : 1).num_params();
    state.first.assign(count, 0);
    state.second.assign(count, 0);
    return state;
}
void load_resume_source(const fs::path &checkpoint, const fs::path &source_protocol, bool seed_explicit,
                        bool rho_explicit) {
    parent_checkpoint_payload = cp::read(checkpoint.string());
    parent_protocol_payload = cp::read(source_protocol.string());
    auto state =
        rl::read_ppo_training_state(parent_checkpoint_payload, cp::checksum(parent_protocol_payload));
    require(!seed_explicit || seed == state.actor.seed, "Explicit seed differs from PPO parent");
    require(!rho_explicit || correlation == state.actor.correlation, "Explicit rho differs from PPO parent");
    require(state.actor.update <= static_cast<int>(std::numeric_limits<uint32_t>::max() / horizon) - updates,
            "Resume update exceeds RNG counter range");
    training_episode.observation_schema = state.actor.observation_schema;
    seed = state.actor.seed;
    correlation = state.actor.correlation;
    parent_checkpoint_path = fs::absolute(checkpoint);
    parent_protocol_path = fs::absolute(source_protocol);
    resume_state = std::move(state);
}
void restore_optimizer(nn::GpuAdam &optimizer, const rl::PpoAdamState &state) {
    const auto bytes = rl::adam_payload(state);
    cp::Reader reader{bytes};
    optimizer.load(reader);
    reader.finish();
}
void restore_training(const rl::PpoTrainingState &state, nn::GpuMlp &actor, nn::GpuMlp &critic,
                      nn::GpuAdam &actor_opt, nn::GpuAdam &critic_opt) {
    actor.set_params(state.actor.parameters);
    critic.set_params(state.critic);
    restore_optimizer(actor_opt, state.actor_adam);
    restore_optimizer(critic_opt, state.critic_adam);
    if (actor_lr_override)
        actor_opt.set_learning_rate(*actor_lr_override);
    require(actor_opt.steps() == state.actor.actor_steps && critic_opt.steps() == state.actor.critic_steps,
            "Restored Adam counters differ");
}
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() {
        if (value) {
            cudaStreamSynchronize(value);
            cudaStreamDestroy(value);
        }
    }
};
struct Timer {
    cudaEvent_t first{}, last{};
    Timer() {
        CUDA_CHECK_THROW(cudaEventCreate(&first));
        CUDA_CHECK_THROW(cudaEventCreate(&last));
    }
    ~Timer() {
        cudaEventDestroy(first);
        cudaEventDestroy(last);
    }
    void begin(cudaStream_t s) { CUDA_CHECK_THROW(cudaEventRecord(first, s)); }
    double end(cudaStream_t s) {
        CUDA_CHECK_THROW(cudaEventRecord(last, s));
        CUDA_CHECK_THROW(cudaEventSynchronize(last));
        float ms = 0;
        CUDA_CHECK_THROW(cudaEventElapsedTime(&ms, first, last));
        return ms;
    }
};
template <class T> std::vector<T> download(const T *pointer, size_t count, cudaStream_t stream) {
    std::vector<T> values(count);
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(values.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    return values;
}
void finite(const std::vector<float> &values) {
    for (float v : values)
        require(std::isfinite(v), "Nonfinite PPO numerical data");
}
struct Budget {
    int controls = 0, substeps = 0;
    int maximum_controls = control_cap, maximum_substeps = substep_cap;
    double seconds = wall_cap;
    void step(m::World &world, const float *command, cudarl::movement::Telemetry *diagnostic = nullptr,
              p::BodyState *before_state = nullptr, p::BodyState *candidate = nullptr) {
        require(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < seconds,
                "Physical evaluation wall cap reached");
        require(controls < maximum_controls && substeps + 10 <= maximum_substeps,
                "Physical budget exhausted");
        ++controls;
        const auto before = world.attempted_substeps;
        const bool ok = cudarl::joint_motor::step_state(world, command, diagnostic, before_state, candidate);
        substeps += static_cast<int>(world.attempted_substeps - before);
        if (!ok || world.attempted_substeps - before != 10)
            throw std::runtime_error(
                "CPU physical safety stop code=" + std::to_string(world.failure_code) +
                " completed_controls=" + std::to_string(world.physical.control_steps) +
                " attempted_substeps=" + std::to_string(world.attempted_substeps) + " current_clearance_m=" +
                std::to_string(cudarl::movement::clearance(world.physical.body, true, world.arena)));
    }
};
struct Rollout {
    device_ptr<float> cost_trace{rollout_elements(5)};
    device_ptr<m::World> worlds{environments};
    device_ptr<m::ForwardTask> tasks{environments};
    device_ptr<gpu::Counts> counts{environments};
    device_ptr<int> fault{1};
    device_ptr<float> previous_obs{rollout_elements(observations)},
        previous_action{rollout_elements(actions)}, history{rollout_elements()},
        actor_input{2 * environments * observations};
    device_ptr<float> obs{rollout_elements(observations)}, action{rollout_elements(actions)},
        logp{rollout_elements()}, value{rollout_elements()}, reward{rollout_elements()},
        done{rollout_elements()}, zeros{rollout_elements()}, last{environments},
        advantage{rollout_elements()}, returns{rollout_elements()}, adv_stats{2}, sigma{actions};
    Rollout(cudaStream_t s) {
        CUDA_CHECK_THROW(cudaMemsetAsync(counts.get(), 0, counts.size_bytes(), s));
        CUDA_CHECK_THROW(cudaMemsetAsync(fault.get(), 0, fault.size_bytes(), s));
        CUDA_CHECK_THROW(cudaMemsetAsync(zeros.get(), 0, zeros.size_bytes(), s));
        CUDA_CHECK_THROW(cudaMemsetAsync(last.get(), 0, last.size_bytes(), s));
        sigma.copy_from_host(policy_sigma().data(), actions);
    }
};
struct Batches {
    device_ptr<float> obs{2 * minibatch * observations}, previous_action{minibatch * actions},
        history{minibatch}, action{minibatch * actions}, logp{minibatch}, advantage{minibatch},
        returns{minibatch}, mean_gradient{2 * minibatch * actions}, value_gradient{minibatch},
        row_stats{minibatch * 5}, stats{5};
};
std::vector<uint8_t> protocol() {
    cp::Writer out;
    out.value(uint32_t{0x3150504d});
    out.value(uint32_t{13});
    out.value(uint32_t{stopping ? 1u : 0u});
    out.value(m::StopSequence::request_control);
    out.value(training_episode.observation_schema);
    out.value(cudarl::motor_observation::path_scale_m(training_episode.observation_schema));
    out.value(uint32_t{actor_lr_override ? 1u : 0u});
    out.value(actor_lr_override.value_or(0));
    out.value(uint32_t{depth_offsets ? 1u : 0u});
    out.value(.0005f); // Symmetric fixed-target starts; pair-group phase = global update %3.
    out.value(resume_state ? resume_state->actor.reward_version
                           : uint32_t{static_cast<uint32_t>(training_episode.reward_version())});
    for (float v : {m::kDepthSettlingSeconds, m::kRecoverySpeedLimit, m::kRollCost, m::kPitchCost})
        out.value(v);
    out.value(training_episode.arena().half_width_m);
    out.value(training_episode.arena().half_depth_m);
    out.value(uint32_t{1}); // Gate long evaluation on four short passes.
    out.value(uint32_t{tilt_mixture ? 1u : 0u});
    out.value(tilt_mixture ? 3.f : 0.f);
    out.value(seed);
    out.value(correlation);
    out.value(uint32_t{verification_mode ? 1u : 0u});
    out.value(uint32_t{resume_state ? 1u : 0u});
    out.value(starting_update());
    out.value(cp::checksum(parent_checkpoint_payload));
    out.value(cp::checksum(parent_protocol_payload));
    out.array(policy_sigma());
    out.value(initial_optimizer_config(true));
    out.value(initial_optimizer_config(false));
    for (int v : {6, m::kTaskVersion, training_episode.reward_version(), 3, observations, actions,
                  environments, horizon, verification_mode ? 1 : updates, minibatch, epochs,
                  verification_mode ? rows : training_cap, verification_mode ? rows : control_cap,
                  verification_mode ? rows * 10 : substep_cap})
        out.value(v);
    for (float v : {.008f, log_sigma, .001f, .00001f, .5f, .99f, 1.f, .2f, .01f,
                    verification_mode ? 10.f : static_cast<float>(active_wall_cap)})
        out.value(v);
    return out.data;
}
std::vector<uint8_t> initialization() {
    cp::Writer out;
    out.value(uint32_t{0x31494f50});
    out.value(uint32_t{2});
    out.value(seed);
    out.array(initial_actor());
    out.array(initial_critic());
    out.array(policy_sigma());
    out.value(starting_update());
    out.array(rl::adam_payload(initial_optimizer(true)));
    out.array(rl::adam_payload(initial_optimizer(false)));
    return out.data;
}
void status(const fs::path &root, const char *phase, const Budget &budget, int update = 0) {
    std::ofstream out(root / "status.json");
    out << "{\"phase\":" << std::quoted(phase) << ",\"unix_seconds\":" << std::setprecision(17)
        << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()
        << ",\"updates\":" << update << ",\"controls\":" << budget.controls
        << ",\"starting_update\":" << starting_update()
        << ",\"additional_updates\":" << std::max(0, update - starting_update())
        << ",\"training_started_unix_seconds\":" << training_started_unix_seconds
        << ",\"physical_substeps\":" << budget.substeps << "}\n";
    out.flush();
}
void check_initialization() {
    Stream stream;
    Rollout data(stream.value);
    gpu::initialize<<<1, 32, 0, stream.value>>>(data.worlds.get(), data.tasks.get(), environments,
                                                data.fault.get(), training_episode);
    gpu::observe<<<1, 32, 0, stream.value>>>(data.worlds.get(), data.tasks.get(), data.obs.get(),
                                             environments, data.fault.get(), training_episode);
    CUDA_CHECK_THROW(cudaGetLastError());
    const auto actual = download(data.obs.get(), environments * observations, stream.value);
    const auto setup_worlds = download(data.worlds.get(), environments, stream.value);
    finite(actual);
    require(download(data.fault.get(), 1, stream.value)[0] == 0, "Zero-step device initialization failed");
    for (int i = 0; i < environments; ++i) {
        m::World world;
        m::ForwardTask task;
        require(m::begin_training_episode(i, tilt_mixture, training_episode.task_config(), world, task,
                                          training_episode.arena(),
                                          training_episode.initial_vertical_offset(i)),
                "CPU initialization failed");
        require(setup_worlds[i].arena.half_width_m == world.arena.half_width_m &&
                    setup_worlds[i].arena.half_depth_m == world.arena.half_depth_m,
                "CPU/CUDA training arena mismatch");
        require(p::norm(setup_worlds[i].physical.body.position - world.physical.body.position) < 1e-7f,
                "CPU/CUDA training pose mismatch");
        float expected[observations]{};
        require(m::observe(task, world, training_episode.task_config(), expected), "CPU observation failed");
        for (int j = 0; j < observations; ++j)
            require(fabsf(expected[j] - actual[i * observations + j]) <= 1e-6f,
                    "Zero-step CPU/CUDA observation mismatch");
    }
    nn::GpuMlp actor(spec(actions), environments, seed);
    actor.set_params(initial_actor());
    actor.forward(data.obs.get(), environments, stream.value);
    const auto prediction = download(actor.output(), environments * actions, stream.value);
    finite(prediction);
    nn::CpuMlp<float> reference(spec(actions));
    reference.params() = initial_actor();
    reference.forward(actual.data(), environments);
    for (size_t i = 0; i < prediction.size(); ++i)
        require(fabsf(prediction[i] - reference.output()[i]) < 1e-5f, "Fresh actor CPU/CUDA mismatch");
    rl::sample_gaussian(actor.output(), data.sigma.get(), environments, actions, seed, 0, data.action.get(),
                        data.logp.get(), stream.value);
    const auto sampled = download(data.action.get(), environments * actions, stream.value),
               probability = download(data.logp.get(), environments, stream.value);
    finite(sampled);
    finite(probability);
    const auto sigma = policy_sigma();
    for (int i = 0; i < environments; ++i)
        require(fabsf(probability[i] - rl::gaussian_logp(prediction.data() + i * actions, sigma.data(),
                                                         sampled.data() + i * actions, actions)) < 1e-5f,
                "Sampled raw latent/log probability mismatch");
    for (const auto &count : download(data.counts.get(), environments, stream.value))
        require(count.controls == 0 && count.substeps == 0, "Initialization advanced physics");
}
void verify_nominal_arena(const fs::path &root) {
    if (horizon != 1280)
        return;
    const auto links = cudarl::movement::shape(m::initial_world(0).physical.body);
    float center_bound = 0, root_envelope = 0, mass = 0, weighted_center_bound = 0;
    for (int b = 0; b < 3; ++b) {
        const auto g = cudarl::movement::body_geometry(b);
        if (b)
            center_bound += .5f * (links.link[b - 1].length + g.length);
        const float radius = std::max(g.half_height, g.half_width);
        const float support = b ? .5f * g.length + radius : std::max(.5f * g.length, radius);
        root_envelope = std::max(root_envelope, center_bound + support);
        mass += links.link[b].mass;
        weighted_center_bound += links.link[b].mass * center_bound;
    }
    const float com_envelope = root_envelope + weighted_center_bound / mass;
    std::ofstream out(root / "arena-envelope.json");
    out << std::setprecision(12)
        << "{\"seconds\":64,\"target_speed_m_s\":0.008,\"COM_radius_bound_m\":" << com_envelope
        << ",\"assumption\":\"nominal constant target velocity plus2mm cross-track; not a guarantee for "
           "exploratory overspeed\",\"poses\":[";
    int count = 0;
    for (int pose : {0, 3}) {
        auto world = m::initial_world(pose);
        world.arena = training_episode.arena();
        const auto config = training_episode.task_config();
        m::ForwardTask task;
        require(m::begin_forward(world, config, task), "Arena envelope setup");
        const auto initial = cudarl::motor::motion(world.physical.body).com_model * p::scale;
        const auto finish = initial + task.request.heading * (.008f * 64);
        const auto upper = initial + task.request.heading * (.0088f * 64);
        const float mx = world.arena.half_width_m - config.clearance_m - com_envelope -
                         std::max(fabsf(initial.x), fabsf(finish.x)) - config.lateral_limit_m;
        const float mz = world.arena.half_depth_m - config.clearance_m - com_envelope -
                         std::max(fabsf(initial.z), fabsf(finish.z)) - config.lateral_limit_m;
        const float upper_mx = world.arena.half_width_m - config.clearance_m - com_envelope -
                               std::max(fabsf(initial.x), fabsf(upper.x)) - config.lateral_limit_m;
        out << (count++ ? "," : "") << "{\"pose\":" << pose << ",\"nominal_margin_x_m\":" << mx
            << ",\"nominal_margin_z_m\":" << mz << ",\"upper_speed_conservative_margin_x_m\":" << upper_mx
            << '}';
        require(mx > 0 && mz > 0, "64s nominal target trajectory does not fit existing arena");
    }
    out << "],\"physics_steps\":0,\"arena_unchanged\":true}\n";
}
int prepare(const fs::path &root) {
    require(!fs::exists(root), "Run identity already exists");
    fs::create_directories(root);
    if (resume_state) {
        fs::copy_file(parent_checkpoint_path, root / "parent-checkpoint.bin");
        fs::copy_file(parent_protocol_path, root / "parent-protocol.bin");
    }
    check_initialization();
    verify_nominal_arena(root);
    cp::write_atomic((root / "protocol.bin").string(), protocol());
    cp::write_atomic((root / "initialization.bin").string(), initialization());
    cp::Writer actor_identity, critic_identity;
    actor_identity.array(initial_actor());
    critic_identity.array(initial_critic());
    cp::write_atomic((root / "initial-actor.bin").string(), actor_identity.data);
    cp::write_atomic((root / "initial-critic.bin").string(), critic_identity.data);
    std::ofstream out(root / "readiness.json");
    out << std::setprecision(9) << "{\"ready\":true,\"algorithm\":\"PPO\",\"seed\":" << seed
        << ",\"verify_resume\":" << (verification_mode ? "true" : "false") << ",\"rho\":" << correlation
        << ",\"resumed\":" << (resume_state ? "true" : "false")
        << ",\"starting_update\":" << starting_update()
        << ",\"starting_actor_steps\":" << initial_optimizer(true).steps
        << ",\"starting_critic_steps\":" << initial_optimizer(false).steps
        << ",\"actor_shape\":[40,32,32,4],\"critic_shape\":[40,32,32,1],\"checkpoint_schema\":4,\"action_"
           "mapping\":3,\"task\":6,\"reward\":"
        << training_episode.reward_version() << ",\"stopping_sequence\":" << (stopping ? "true" : "false")
        << ",\"model\":6,\"fresh_episode_rollout\":true,\"teacher_"
           "examples\":false,\"preset_oscillator\":false,\"target_mm_s\":8,\"environments\":32,\"rollout_"
           "controls\":"
        << horizon << ",\"episode_seconds\":" << horizon * .05
        << ",\"training_arena_width_m\":" << 2 * training_episode.arena().half_width_m
        << ",\"training_arena_depth_m\":" << 2 * training_episode.arena().half_depth_m
        << ",\"PPO_additional_updates_max\":" << (verification_mode ? 1 : updates)
        << ",\"ending_update_max\":" << starting_update() + (verification_mode ? 1 : updates)
        << ",\"epochs\":2,\"minibatch\":256,\"additional_optimizer_steps_max_each\":"
        << ((verification_mode ? 1 : updates) * optimizer_steps_per_update())
        << ",\"controls_cap\":" << (verification_mode ? rows : control_cap)
        << ",\"substeps_cap\":" << (verification_mode ? rows * 10 : substep_cap)
        << ",\"tilt_mixture\":" << (tilt_mixture ? "true" : "false")
        << ",\"fixed_depth_offsets\":" << (depth_offsets ? "true" : "false")
        << ",\"actor_learning_rate\":" << initial_optimizer_config(true).lr
        << ",\"critic_learning_rate\":" << initial_optimizer_config(false).lr
        << ",\"long_evaluation_requires_short4of4\":true"
        << ",\"native_wall_seconds\":" << (verification_mode ? 10 : active_wall_cap)
        << ",\"process_wall_seconds\":"
        << (depth_offsets
                ? 75
                : (verification_mode ? 15
                                     : (horizon == 640 ? 35 : ((tilt_mixture || updates <= 32) ? 25 : 75))))
        << ",\"physical_steps\":0,\"optimizer_steps_"
           "performed\":0,\"runtime_adoption\":false,\"retries\":0,\"log_std\":[";
    const auto sigma = policy_sigma();
    for (int j = 0; j < actions; ++j)
        out << (j ? "," : "") << sigma[j];
    for (bool actor : {true, false}) {
        const auto cfg = initial_optimizer_config(actor);
        out << "],\"" << (actor ? "actor_optimizer_config" : "critic_optimizer_config") << "\":[" << cfg.lr
            << ',' << cfg.beta1 << ',' << cfg.beta2 << ',' << cfg.eps << ',' << cfg.max_grad_norm;
    }
    out << "],\"resume_semantics\":\"completed PPO update boundary; fresh true episodes, rollout/history "
           "reset; global update-keyed RNG and Adam counters retained\"}\n";
    out.close();
    status(root, "ready", {}, starting_update());
    std::ifstream immediate(root / "readiness.json");
    std::cout << immediate.rdbuf() << std::flush;
    return 0;
}
std::vector<float> fields(const m::World &world) {
    const auto &b = world.physical.body;
    std::vector<float> f{b.position.x,    b.position.y,    b.position.z,    b.orientation.x,
                         b.orientation.y, b.orientation.z, b.orientation.w, b.work};
    for (int j = 0; j < 4; ++j) {
        f.push_back(world.physical.held_targets[j]);
        f.push_back(world.reference_rate[j]);
    }
    for (float v : b.angle)
        f.push_back(v);
    for (float v : b.velocity)
        f.push_back(v);
    return f;
}
void parity(Budget &budget, cudaStream_t stream) {
    constexpr int n = 8;
    std::vector<m::World> cpu(n);
    std::vector<m::ForwardTask> tasks(n);
    std::vector<float> latent(n * actions);
    for (int i = 0; i < n; ++i) {
        cpu[i].physical.paused = false;
        const int axis = i / 2;
        latent[i * actions + axis] = atanhf(i % 2 ? .1f : -.1f);
        require(m::begin_forward(cpu[i], m::config_for_speed(.008f), tasks[i]), "Invalid axis fixture");
    }
    device_ptr<m::World> worlds(n);
    device_ptr<m::ForwardTask> device_tasks(n);
    device_ptr<gpu::Counts> counts(n);
    device_ptr<int> fault(1);
    device_ptr<float> raw(n * actions), reward(n), done(n);
    worlds.copy_from_host(cpu.data(), n);
    device_tasks.copy_from_host(tasks.data(), n);
    raw.copy_from_host(latent.data(), latent.size());
    CUDA_CHECK_THROW(cudaMemsetAsync(counts.get(), 0, counts.size_bytes(), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(fault.get(), 0, fault.size_bytes(), stream));
    for (int tick = 0; tick < 4; ++tick) {
        for (int i = 0; i < n; ++i) {
            m::Action action;
            require(m::action_from_latent(latent.data() + i * actions, action), "Invalid probe action");
            budget.step(cpu[i], action.command);
        }
        gpu::advance<<<1, 32, 0, stream>>>(worlds.get(), device_tasks.get(), raw.get(), reward.get(),
                                           done.get(), counts.get(), n, fault.get());
        CUDA_CHECK_THROW(cudaGetLastError());
    }
    for (const auto &count : download(counts.get(), n, stream)) {
        budget.controls += static_cast<int>(count.controls);
        budget.substeps += static_cast<int>(count.substeps);
    }
    require(download(fault.get(), 1, stream)[0] == 0, "Four-axis CUDA actuator fault");
    const auto result = download(worlds.get(), n, stream);
    for (int i = 0; i < n; ++i) {
        const auto a = fields(cpu[i]), b = fields(result[i]);
        finite(a);
        finite(b);
        for (size_t j = 0; j < a.size(); ++j)
            require(fabsf(a[j] - b[j]) <= 2e-4f + 2e-3f * fabsf(a[j]), "Four-axis CPU/CUDA parity failure");
        const int axis = i / 2;
        const float sign = i % 2 ? 1.f : -1.f;
        require(sign * cpu[i].physical.body.angle[axis] > 1e-6f &&
                    sign * result[i].physical.body.angle[axis] > 1e-6f,
                "Axis did not respond in the commanded direction");
        for (float torque : cpu[i].physical.actual_torques)
            require(fabsf(torque) <= 1, "Applied torque cap violated");
    }
    require(budget.controls == 64 && budget.substeps == 640, "Actuator verification accounting mismatch");
}
struct Evaluation {
    int passed = 0;
    float speed = 0, rmse = 0, work_per_m = 0;
};
Evaluation reused_baseline(const fs::path &root) {
    std::ifstream in(root / "baseline-evaluation.csv");
    std::string line;
    std::getline(in, line);
    Evaluation out;
    int rows_seen = 0;
    while (std::getline(in, line)) {
        std::stringstream row(line);
        std::string field;
        std::vector<float> v;
        while (std::getline(row, field, ','))
            v.push_back(std::stof(field));
        require(v.size() == 11 && v[0] == rows_seen && v[1] == 1, "Invalid preserved336 baseline evaluation");
        ++rows_seen;
        ++out.passed;
        out.speed += v[2] * .001f / 4;
        out.rmse += v[3] * .001f / 4;
        out.work_per_m += v[5] / 4;
    }
    require(rows_seen == 4, "Preserved baseline must contain4poses");
    return out;
}
struct Sustained {
    int passed = 0, windows_passed = 0, windows_total = 0;
    int completed_poses = 0, safety_code = 0;
    std::string safety_fault;
};
void state_json(std::ostream &out, const p::BodyState &body);
Sustained sustained(const std::vector<float> &params, const fs::path &root, Budget &budget) {
    nn::CpuMlp<float> actor(spec(actions));
    actor.params() = params;
    Sustained result;
    std::ofstream trace(root / "sustained-windows.csv");
    trace << "pose,end_s,mean_mm_s,RMSE_mm_s,lateral_mm,depth_mm,tilt_deg,path_deg,work_J_per_m,passed\n";
    std::ofstream controls(root / "sustained-controls.csv");
    controls << std::setprecision(12)
             << "pose,control,time_s,forward_mm,signed_depth_mm,lane_mm,tilt_deg,vy_mm_s,clearance_mm\n";
    for (int pose : {0, 3}) {
        auto world = m::initial_world(pose);
        world.arena = m::long_swim_arena();
        auto config = m::config_for_speed(.008f);
        config.observation_schema = training_episode.observation_schema;
        config.maximum_controls = 1200;
        m::ForwardTask task;
        require(m::begin_forward(world, config, task), "Long evaluation initialization failed");
        auto origin = cudarl::motor::motion(world.physical.body).com_model;
        float work0 = world.physical.body.work;
        double error2 = 0;
        bool all = true;
        for (int tick = 0; tick < 1200; ++tick) {
            float obs[observations]{};
            require(m::observe(task, world, config, obs), "Long observation failed");
            actor.forward(obs, 1);
            m::Action action;
            require(m::action_from_latent(actor.output(), action), "Long action failed");
            const auto before = world.physical.body;
            try {
                budget.step(world, action.command);
            } catch (const std::exception &error) {
                std::ofstream failure(root / "sustained-failure.json");
                failure << std::setprecision(17) << "{\"pose\":" << pose
                        << ",\"attempted_time_s\":" << world.attempted_substeps * .005
                        << ",\"code\":" << world.failure_code << ",\"error\":" << std::quoted(error.what())
                        << ",\"before_control\":";
                state_json(failure, before);
                failure << ",\"last_accepted_substep_state\":";
                state_json(failure, world.physical.body);
                failure << "}\n";
                failure.close();
                if ((world.failure_code == 7 || world.failure_code == 8) &&
                    p::finite_body(world.physical.body)) {
                    result.safety_fault = error.what();
                    result.safety_code = world.failure_code;
                    return result;
                }
                throw;
            }
            const auto interval = m::record_control(task, world, config);
            require(interval.accepted, "Long task rejected");
            const auto motion = cudarl::motor::motion(world.physical.body);
            controls << pose << ',' << tick + 1 << ',' << (tick + 1) * .05 << ','
                     << interval.metrics.forward_m * 1000 << ','
                     << (motion.com_model.y - task.start_com_model.y) * p::scale * 1000 << ','
                     << interval.metrics.lateral_m * 1000 << ',' << interval.metrics.tilt_deg << ','
                     << motion.velocity_m_s.y * 1000 << ','
                     << cudarl::movement::clearance(world.physical.body, true, world.arena) * 1000 << '\n';
            if (tick >= 40)
                error2 += interval.metrics.speed_error_m_s * interval.metrics.speed_error_m_s;
            if (tick == 39) {
                origin = cudarl::motor::motion(world.physical.body).com_model;
                work0 = world.physical.body.work;
            }
            if ((tick + 1) % 80 == 0) {
                const auto com = cudarl::motor::motion(world.physical.body).com_model;
                const auto delta = (com - origin) * p::scale;
                const float forward = p::dot(delta, task.request.heading);
                const auto side = delta - task.request.heading * forward;
                const int samples = tick == 79 ? 40 : 80;
                const float speed = forward / (samples * p::dt),
                            rmse = sqrtf(static_cast<float>(error2 / samples)),
                            path = atan2f(p::norm({side.x, 0, side.z}), forward) * 180 / cudarl::movement::pi,
                            work = (world.physical.body.work - work0) / fmaxf(forward, 1e-9f);
                const bool pass =
                    speed >= config.minimum_speed_m_s && speed <= config.maximum_speed_m_s &&
                    rmse <= config.speed_rmse_limit && task.maximum_lateral <= config.lateral_limit_m &&
                    task.maximum_depth <= config.depth_limit_m &&
                    task.maximum_tilt <= config.tilt_limit_deg && path <= config.path_angle_limit_deg &&
                    work <= config.work_per_distance_limit;
                all = all && pass;
                ++result.windows_total;
                result.windows_passed += pass ? 1 : 0;
                trace << pose << ',' << (tick + 1) * p::dt << ',' << speed * 1000 << ',' << rmse * 1000 << ','
                      << task.maximum_lateral * 1000 << ',' << task.maximum_depth * 1000 << ','
                      << task.maximum_tilt << ',' << path << ',' << work << ',' << pass << '\n';
                trace.flush();
                origin = com;
                work0 = world.physical.body.work;
                error2 = 0;
            }
        }
        result.passed += (all && task.outcome == m::Outcome::Success) ? 1 : 0;
        ++result.completed_poses;
    }
    return result;
}
Evaluation evaluate(const std::vector<float> &params, const fs::path &root, const char *name,
                    Budget &budget) {
    nn::CpuMlp<float> actor(spec(actions));
    actor.params() = params;
    Evaluation out;
    std::ofstream trace(root / (std::string(name) + "-evaluation.csv"));
    trace << "pose,passed,steady_mm_s,RMSE_mm_s,lateral_mm,work_J_per_m,depth_mm,tilt_deg,path_deg,forward_"
             "mm,late_forward_mm\n";
    std::ofstream controls(root / (std::string(name) + "-controls.csv"));
    controls << "pose,control,tilt_deg,lane_mm,depth_mm,work_J";
    for (int j = 0; j < 4; ++j)
        controls << ",command" << j << ",target" << j << ",joint_angle" << j << ",joint_rate" << j;
    controls << ",forward_mm\n";
    for (int pose = 0; pose < 4; ++pose) {
        float forward_at_four = 0;
        auto world = m::initial_world(pose);
        auto config = m::config_for_speed(.008f);
        config.observation_schema = training_episode.observation_schema;
        m::ForwardTask task;
        require(m::begin_forward(world, config, task), "Frozen evaluation initialization failed");
        for (int t = 0; t < config.maximum_controls; ++t) {
            float obs[observations]{};
            require(m::observe(task, world, config, obs), "Invalid frozen observation");
            actor.forward(obs, 1);
            m::Action action;
            require(m::action_from_latent(actor.output(), action), "Invalid frozen action");
            budget.step(world, action.command);
            const auto interval = m::record_control(task, world, config);
            require(interval.accepted, "Frozen physical/task rejection");
            controls << pose << ',' << t + 1 << ',' << interval.metrics.tilt_deg << ','
                     << interval.metrics.lateral_m * 1000 << ',' << interval.metrics.depth_m * 1000 << ','
                     << world.physical.body.work;
            for (int j = 0; j < 4; ++j)
                controls << ',' << action.command[j] << ',' << world.physical.held_targets[j] << ','
                         << world.physical.body.angle[j] << ',' << world.physical.body.velocity[6 + j];
            controls << ',' << m::measure(task, world).forward_m * 1000 << '\n';
            if (t == 79)
                forward_at_four = m::measure(task, world).forward_m;
        }
        const bool passed = task.outcome == m::Outcome::Success;
        const float work = world.physical.body.work / fmaxf(m::measure(task, world).forward_m, 1e-9f);
        out.passed += passed ? 1 : 0;
        out.speed += task.steady_speed_m_s / 4;
        out.rmse += task.speed_rmse / 4;
        out.work_per_m += work / 4;
        trace << pose << ',' << passed << ',' << task.steady_speed_m_s * 1000 << ',' << task.speed_rmse * 1000
              << ',' << task.maximum_lateral * 1000 << ',' << work << ',' << task.maximum_depth * 1000 << ','
              << task.maximum_tilt << ',' << task.path_angle_deg << ','
              << m::measure(task, world).forward_m * 1000 << ','
              << (m::measure(task, world).forward_m - forward_at_four) * 1000 << '\n';
    }
    return out;
}
struct StoppingResult {
    int passed = 0, completed = 0;
};
StoppingResult evaluate_stopping(const std::vector<float> &parameters, const fs::path &root, Budget &budget) {
    require(fs::create_directory(root), "Stopping evaluation identity already exists");
    nn::CpuMlp<float> actor(spec(actions));
    actor.params() = parameters;
    StoppingResult result;
    std::ofstream trace(root / "controls.csv"), summary(root / "summary.csv");
    trace << std::setprecision(12)
          << "pose,control,time_s,requested_mm_s,COM_speed_mm_s,forward_velocity_mm_s,vx_mm_s,vy_mm_s,vz_mm_"
             "s,forward_mm,depth_mm,lane_mm,tilt_deg,head_yaw_deg,reward,speed_cost,lane_cost,depth_cost,"
             "posture_cost,effort_cost\n";
    summary << std::setprecision(12)
            << "pose,passed,pre_speed_mm_s,pre_RMSE_mm_s,permanent_stop_relative_s,first20samples_stop_"
               "relative_s,stopping_distance_mm,post_command_forward_mm,reverse_distance_mm,peak_reverse_mm_"
               "s,speed_at48_mm_s,final_speed_mm_s,max_depth_mm,max_lane_mm,peak_tilt_deg,peak_head_yaw_deg,"
               "hold_max_excursion_mm,hold_net_drift_mm,hold_mean_speed_mm_s,hold_mean_vy_mm_s,final_depth_"
               "mm,stop_positive_work_J,deadline_pass,hold_pass,state_pass,pre_swim_pass\n";
    for (int pose : {0, 3}) {
        auto world = m::initial_world(pose);
        world.arena = m::long_swim_arena();
        m::ForwardTask task;
        const m::StopSequence sequence;
        require(m::begin_forward(world, sequence.config(0, training_episode.observation_schema), task),
                "Stopping setup");
        const auto target = task.start_com_model, heading = task.request.heading;
        float forward[1281]{}, pre_origin = 0, pre_work = 0, command_work = 0, pre_speed = 0, pre_rmse = 0;
        float hold_origin = 0, hold_excursion = 0, hold_net = 0, reverse_distance = 0,
              min_forward_velocity = 0;
        float max_depth = 0, max_lane = 0, max_tilt = 0, max_yaw = 0, speed48 = 0, final_speed = 0;
        double pre_error2 = 0, hold_speed = 0, hold_vy = 0;
        int last_fast = 640, streak = 0, first_stop = -1;
        bool deadline = true, hold = true, pre_pass = false;
        for (int control = 1; control <= 1280; ++control) {
            const auto config = sequence.config(task.controls, training_episode.observation_schema);
            require(task.start_com_model.y == target.y && p::norm(task.request.heading - heading) == 0,
                    "Stopping target reference changed");
            require(m::update_velocity_request(world, config, task), "Stopping request update");
            float observation[observations];
            require(m::observe(task, world, config, observation), "Stopping observation");
            if (control == 641)
                for (int j = 3; j < 6; ++j)
                    require(observation[j] == 0, "Zero request absent from observation");
            actor.forward(observation, 1);
            m::Action action;
            require(m::action_from_latent(actor.output(), action), "Stopping action");
            p::BodyState before{}, candidate{};
            try {
                budget.step(world, action.command, nullptr, &before, &candidate);
            } catch (const std::exception &error) {
                std::ofstream fault(root / "failure.json");
                fault << std::setprecision(17) << "{\"pose\":" << pose
                      << ",\"attempted_time_s\":" << world.attempted_substeps * .005
                      << ",\"code\":" << world.failure_code << ",\"error\":" << std::quoted(error.what())
                      << ",\"before\":";
                state_json(fault, before);
                fault << ",\"candidate_not_committed\":";
                state_json(fault, candidate);
                fault << ",\"last_accepted\":";
                state_json(fault, world.physical.body);
                fault << "}\n";
                throw;
            }
            const auto interval = m::record_control(task, world, config);
            require(interval.accepted, "Stopping scoring");
            const auto motion = cudarl::motor::motion(world.physical.body);
            const auto &metrics = interval.metrics;
            const float speed = p::norm(motion.velocity_m_s), velocity = p::dot(motion.velocity_m_s, heading);
            forward[control] = metrics.forward_m;
            max_depth = std::max(max_depth, metrics.depth_m);
            max_lane = std::max(max_lane, metrics.lateral_m);
            max_tilt = std::max(max_tilt, metrics.tilt_deg);
            max_yaw = std::max(max_yaw, metrics.head_yaw_deg);
            if (control == 40) {
                pre_origin = metrics.forward_m;
                pre_work = world.physical.body.work;
            }
            if (control > 40 && control <= 640)
                pre_error2 += metrics.speed_error_m_s * metrics.speed_error_m_s;
            if (control == 640) {
                pre_speed = (metrics.forward_m - pre_origin) / 30;
                pre_rmse = sqrtf(float(pre_error2 / 600));
                const auto delta = (motion.com_model - task.steady_start_com_model) * p::scale;
                const auto side = delta - heading * p::dot(delta, heading);
                const float path =
                    atan2f(p::norm({side.x, 0, side.z}), p::dot(delta, heading)) * 180 / cudarl::movement::pi;
                const float work =
                    (world.physical.body.work - pre_work) / fmaxf(metrics.forward_m - pre_origin, 1e-9f);
                pre_pass = pre_speed >= .0072f && pre_speed <= .0088f && pre_rmse <= .0025f && path <= 5 &&
                           work <= .001f;
                command_work = world.physical.body.work;
            }
            if (control > 640) {
                if (speed > .0005f) {
                    last_fast = control;
                    streak = 0;
                } else
                    ++streak;
                if (streak == 20 && first_stop < 0)
                    first_stop = control - 19;
                reverse_distance += fmaxf(0, forward[control - 1] - forward[control]);
                min_forward_velocity = std::min(min_forward_velocity, velocity);
                if (control >= 960 && speed > .0005f)
                    deadline = false;
            }
            if (control == 960)
                speed48 = speed;
            if (control == 1120)
                hold_origin = metrics.forward_m;
            if (control > 1120) {
                hold_net = metrics.forward_m - hold_origin;
                hold_excursion = std::max(hold_excursion, fabsf(hold_net));
                hold_speed += speed;
                hold_vy += motion.velocity_m_s.y;
                hold = hold && speed <= .0005f && metrics.depth_m <= .001f && metrics.lateral_m <= .002f &&
                       metrics.tilt_deg <= 10;
            }
            final_speed = speed;
            trace << pose << ',' << control << ',' << control * .05 << ',' << config.target_speed_m_s * 1000
                  << ',' << speed * 1000 << ',' << velocity * 1000 << ',' << motion.velocity_m_s.x * 1000
                  << ',' << motion.velocity_m_s.y * 1000 << ',' << motion.velocity_m_s.z * 1000 << ','
                  << metrics.forward_m * 1000 << ',' << metrics.signed_depth_m * 1000 << ','
                  << metrics.lateral_m * 1000 << ',' << metrics.tilt_deg << ',' << metrics.head_yaw_deg << ','
                  << interval.reward;
            for (float cost : {interval.costs.speed, interval.costs.lane, interval.costs.depth,
                               interval.costs.posture, interval.costs.effort})
                trace << ',' << cost;
            trace << '\n';
        }
        const bool stopped = last_fast < 1280;
        const float permanent = stopped ? (last_fast + 1) * .05f - 32 : -1;
        const float distance = stopped ? (forward[last_fast + 1] - forward[640]) * 1000
                                       : std::numeric_limits<float>::quiet_NaN();
        const bool state = max_depth <= .001f && max_lane <= .002f && max_tilt <= 10;
        hold = hold && hold_excursion <= .002f;
        const bool passed = deadline && stopped && hold && state && pre_pass;
        result.passed += passed ? 1 : 0;
        ++result.completed;
        summary << pose << ',' << passed << ',' << pre_speed * 1000 << ',' << pre_rmse * 1000 << ','
                << permanent << ',' << (first_stop < 0 ? -1 : first_stop * .05f - 32) << ',' << distance
                << ',' << (forward[1280] - forward[640]) * 1000 << ',' << reverse_distance * 1000 << ','
                << min_forward_velocity * 1000 << ',' << speed48 * 1000 << ',' << final_speed * 1000 << ','
                << max_depth * 1000 << ',' << max_lane * 1000 << ',' << max_tilt << ',' << max_yaw << ','
                << hold_excursion * 1000 << ',' << hold_net * 1000 << ',' << hold_speed / 160 * 1000 << ','
                << hold_vy / 160 * 1000 << ',' << m::measure(task, world).signed_depth_m * 1000 << ','
                << world.physical.body.work - command_work << ',' << deadline << ',' << hold << ',' << state
                << ',' << pre_pass << '\n';
        summary.flush();
        trace.flush();
    }
    return result;
}
bool preserved_long_windows(const fs::path &root) {
    std::ifstream in(root / "sustained-windows.csv");
    std::string line;
    std::getline(in, line);
    int rows_seen = 0;
    while (std::getline(in, line)) {
        std::stringstream row(line);
        std::string field;
        std::vector<double> values;
        while (std::getline(row, field, ','))
            values.push_back(std::stod(field));
        require(values.size() == 10, "Long regression row shape");
        ++rows_seen;
        if (values[1] > 4 && values[9] != 1)
            return false;
    }
    return rows_seen == 30;
}
int frozen_sustained(const fs::path &root) {
    const auto original = cp::read((root / "checkpoint.bin").string());
    const auto source_protocol = cp::read((root / "protocol.bin").string());
    const auto policy = rl::read_ppo_actor(original, cp::checksum(source_protocol), seed, 2);
    require(policy.update == 336 && policy.reward_version == 5 && policy.correlation == .8f,
            "Frozen sustained336 identity mismatch");
    require(fs::create_directory(root / "run-started"), "Frozen sustained test already attempted");
    training_episode.observation_schema = policy.observation_schema;
    started = std::chrono::steady_clock::now();
    const auto utc = [] {
        return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    };
    const auto begin = utc();
    std::cout << std::setprecision(17) << "FROZEN_SUSTAINED_STARTED unix_seconds=" << begin << std::endl;
    Budget budget;
    budget.maximum_controls = 2400;
    budget.maximum_substeps = 24000;
    budget.seconds = 10;
    const auto result = sustained(policy.parameters, root, budget);
    require(cp::read((root / "checkpoint.bin").string()) == original &&
                cp::read((root / "protocol.bin").string()) == source_protocol,
            "Frozen sustained inputs changed");
    if (result.safety_fault.empty())
        require(budget.controls == 2400 && budget.substeps == 24000 && result.completed_poses == 2,
                "Frozen sustained accounting mismatch");
    std::ofstream out(root / "result.json");
    out << std::setprecision(17) << "{\"update\":336,\"training_updates\":0,\"optimizer_steps\":0"
        << ",\"start_unix_seconds\":" << begin << ",\"end_unix_seconds\":" << utc()
        << ",\"completed_poses\":" << result.completed_poses << ",\"passed_poses\":" << result.passed
        << ",\"windows_passed\":" << result.windows_passed << ",\"windows_total\":" << result.windows_total
        << ",\"controls\":" << budget.controls << ",\"substeps\":" << budget.substeps
        << ",\"safety_code\":" << result.safety_code << ",\"fault\":" << std::quoted(result.safety_fault)
        << ",\"checkpoint_unchanged\":true,\"closed\":true}\n";
    out.close();
    std::ifstream report(root / "result.json");
    std::cout << report.rdbuf() << std::flush;
    return result.safety_fault.empty() ? 0 : 3;
}
void snapshot(const fs::path &root, int update, nn::GpuMlp &actor, nn::GpuMlp &critic, nn::GpuAdam &actor_opt,
              nn::GpuAdam &critic_opt) {
    cp::Writer out;
    out.value(uint32_t{0x31525050});
    out.value(uint32_t{4});
    out.value(seed);
    out.value(cp::checksum(protocol()));
    out.value(update);
    out.value(uint32_t{cudarl::movement::kVersion});
    out.value(uint32_t{cudarl::joint_motor::kActionVersion});
    out.value(uint32_t{m::kTaskVersion});
    out.value(uint32_t{static_cast<uint32_t>(training_episode.reward_version())});
    out.value(uint32_t{observations});
    out.value(uint32_t{actions});
    out.value(uint32_t{1}); // history-conditioned Gaussian stochastic-policy schema
    out.value(correlation);
    out.value(training_episode.observation_schema);
    out.array(actor.get_params());
    out.array(critic.get_params());
    out.array(policy_sigma());
    actor_opt.save(out);
    critic_opt.save(out);
    cp::write_atomic((root / ("ppo-update-" + std::to_string(update) + ".bin")).string(), out.data);
}
void state_json(std::ostream &out, const p::BodyState &body) {
    out << "{\"position_m\":[" << body.position.x * p::scale << ',' << body.position.y * p::scale << ','
        << body.position.z * p::scale << "],\"quaternion_xyzw\":[" << body.orientation.x << ','
        << body.orientation.y << ',' << body.orientation.z << ',' << body.orientation.w
        << "],\"angles_rad\":[";
    for (int j = 0; j < 4; ++j)
        out << (j ? "," : "") << body.angle[j];
    out << "],\"velocity_SI\":[";
    for (int j = 0; j < 10; ++j)
        out << (j ? "," : "") << body.velocity[j];
    out << "],\"finite\":" << (p::finite_body(body) ? "true" : "false") << '}';
}
// Fixed world-space COM target; translating a start must not translate its lesson.
// Harness only: all four actions still come from the unchanged frozen actor.
int fixed_depth_replay(const fs::path &root, bool setup_only, Budget *shared_budget = nullptr,
                       bool *recovered = nullptr, int expected_update = 288, int duration_controls = 640) {
    require(duration_controls == 640 || duration_controls == 1280, "Unsupported recovery duration");
    const auto original = cp::read((root / "checkpoint.bin").string());
    const auto policy = rl::read_ppo_actor(original, cp::checksum(cp::read((root / "protocol.bin").string())),
                                           seed, training_episode.observation_schema);
    require(policy.update == expected_update && policy.correlation == .8f,
            "Depth diagnostic policy mismatch");
    nn::CpuMlp<float> actor(spec(actions));
    actor.params() = policy.parameters;
    auto base = m::initial_world(0);
    base.arena = m::long_swim_arena();
    auto config = m::config_for_speed(.008f);
    config.observation_schema = training_episode.observation_schema;
    config.maximum_controls = duration_controls;
    m::ForwardTask base_task;
    require(m::begin_forward(base, config, base_task), "Depth base setup invalid");
    const auto target = base_task.start_com_model;
    require(base_task.request.heading.y == 0 && base_task.request.position.y == target.y,
            "Depth target is not the fixed horizontal level");
    const char *names[] = {"centered", "below", "above"};
    const float offsets[] = {0, -.5f, .5f};
    m::World worlds[3];
    m::ForwardTask tasks[3];
    float initial_obs[3][observations]{};
    std::ofstream setup(root / "setup.json");
    setup << std::setprecision(17) << "{\"target_model_y\":" << target.y
          << ",\"target_m_y\":" << target.y * p::scale << ",\"physical_steps\":0,\"cases\":[";
    auto error_mm = [&](const m::World &w) {
        return (cudarl::motor::motion(w.physical.body).com_model.y - target.y) * p::scale * 1000;
    };
    for (int c = 0; c < 3; ++c) {
        worlds[c] = base;
        tasks[c] = base_task;
        require(m::offset_episode_start(worlds[c], tasks[c], config, offsets[c] / 1000),
                "Translated setup invalid");
        require(m::observe(tasks[c], worlds[c], config, initial_obs[c]), "Translated observation invalid");
        const auto *o = initial_obs[c];
        const float decoded_depth =
            (-1000 * cudarl::motor_observation::path_scale_m(config.observation_schema)) *
            (o[0] * o[9] + o[1] * o[10] + o[2] * o[11]);
        const float decoded_vy = 20 * (o[12] * o[9] + o[13] * o[10] + o[14] * o[11]);
        require(fabsf(decoded_depth - error_mm(worlds[c])) < .0001f && decoded_vy == 0,
                "Depth/vertical velocity observation decoding failed");
        for (int j = 3; j < observations; ++j)
            require(o[j] == initial_obs[0][j], "Translation altered a non-position observation");
        setup << (c ? "," : "") << "{\"name\":" << std::quoted(names[c])
              << ",\"translation_mm\":" << offsets[c] << ",\"actual_error_mm\":" << error_mm(worlds[c])
              << ",\"decoded_error_mm\":" << decoded_depth << ",\"decoded_vy_mm_s\":" << decoded_vy
              << ",\"task_target_model_y\":" << tasks[c].start_com_model.y
              << ",\"request_target_model_y\":" << tasks[c].request.position.y << '}';
    }
    setup << "]}\n";
    setup.close();
    if (setup_only) {
        std::cout << "FIXED_DEPTH_SETUP_PASS cases=3 physical_steps=0 target_m_y=" << std::setprecision(12)
                  << target.y * p::scale << std::endl;
        return 0;
    }
    require(fs::create_directory(root / "run-started"), "Depth diagnostic already attempted; no retries");
    if (!shared_budget)
        started = std::chrono::steady_clock::now();
    const auto utc = [] {
        return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    };
    const double begin = utc();
    const auto case_clock = std::chrono::steady_clock::now();
    Budget local_budget;
    if (!shared_budget) {
        local_budget.maximum_controls = 3 * duration_controls;
        local_budget.maximum_substeps = 30 * duration_controls;
        local_budget.seconds = 5;
    }
    Budget &budget = shared_budget ? *shared_budget : local_budget;
    std::ofstream trace(root / "controls.csv");
    trace << std::setprecision(12)
          << "case,control,time_s,error_mm,vy_mm_s,forward_mm,forward_velocity_mm_s,lane_mm,tilt_deg,reward,"
             "speed_cost,lane_cost,depth_cost,posture_cost,effort_cost,horizontal_cost,recovery_cost,roll_"
             "cost,pitch_cost";
    for (const char *field : {"command", "filtered", "angle", "rate"})
        for (int j = 0; j < 4; ++j)
            trace << ',' << field << j;
    trace << '\n';
    std::cout << std::setprecision(17) << "FIXED_DEPTH_STARTED unix_seconds=" << begin << std::endl;
    int completed = 0;
    std::string fault;
    bool recovery_ok = true;
    std::ofstream recovery_summary(root / "recovery-gates.csv");
    recovery_summary << "case,existing" << duration_controls / 20
                     << "s_gates_pass,final8s_within_half_initial_error,passed\n";
    for (int c = 0; c < 3 && fault.empty(); ++c) {
        auto &world = worlds[c];
        auto &task = tasks[c];
        bool final8_closer = true;
        const float half_initial_error = fabsf(error_mm(world)) * .5f;
        auto row = [&](int control, const m::Transition &interval, const float *commands) {
            const auto motion = cudarl::motor::motion(world.physical.body);
            const auto metrics = m::measure(task, world);
            trace << names[c] << ',' << control << ',' << control * .05 << ',' << error_mm(world) << ','
                  << motion.velocity_m_s.y * 1000 << ',' << metrics.forward_m * 1000 << ','
                  << p::dot(motion.velocity_m_s, task.request.heading) * 1000 << ','
                  << metrics.lateral_m * 1000 << ',' << metrics.tilt_deg << ',' << interval.reward;
            for (float cost : {interval.costs.speed, interval.costs.lane, interval.costs.depth,
                               interval.costs.posture, interval.costs.effort})
                trace << ',' << cost;
            for (float detail : {interval.costs.horizontal, interval.costs.recovery, interval.costs.roll,
                                 interval.costs.pitch})
                trace << ',' << detail;
            for (int j = 0; j < 4; ++j)
                trace << ',' << commands[j];
            for (float x : world.physical.held_targets)
                trace << ',' << x;
            for (float x : world.physical.body.angle)
                trace << ',' << x;
            for (int j = 0; j < 4; ++j)
                trace << ',' << world.physical.body.velocity[6 + j];
            trace << '\n';
        };
        const float zeros[4]{};
        row(0, {}, zeros);
        for (int t = 0; t < duration_controls; ++t) {
            p::BodyState before{}, candidate{};
            cudarl::movement::Telemetry telemetry{};
            try {
                require(task.start_com_model.y == target.y && task.request.position.y == target.y,
                        "Fixed depth target drifted");
                float obs[observations];
                require(m::observe(task, world, config, obs), "Depth replay observation failed");
                actor.forward(obs, 1);
                m::Action action;
                require(m::action_from_latent(actor.output(), action), "Depth replay action failed");
                budget.step(world, action.command, &telemetry, &before, &candidate);
                const auto interval = m::record_control(task, world, config);
                require(interval.accepted, "Depth replay task rejected");
                row(t + 1, interval, action.command);
                if (c != 0 && t >= duration_controls - 160 && fabsf(error_mm(world)) >= half_initial_error)
                    final8_closer = false;
            } catch (const std::exception &e) {
                fault = e.what();
                std::ofstream failure(root / "failure.json");
                failure << std::setprecision(17) << "{\"case\":" << std::quoted(names[c])
                        << ",\"control\":" << t + 1
                        << ",\"attempted_physics_time_s\":" << world.attempted_substeps * .005
                        << ",\"code\":" << world.failure_code << ",\"fault\":" << std::quoted(fault)
                        << ",\"before_failed_substep\":";
                state_json(failure, before);
                failure << ",\"candidate_NOT_committed\":";
                state_json(failure, candidate);
                failure << ",\"last_accepted\":";
                state_json(failure, world.physical.body);
                failure << "}\n";
                break;
            }
        }
        trace.flush();
        if (fault.empty()) {
            ++completed;
            const bool passed = task.outcome == m::Outcome::Success && (c == 0 || final8_closer);
            recovery_ok = recovery_ok && passed;
            recovery_summary << names[c] << ',' << (task.outcome == m::Outcome::Success) << ','
                             << (c == 0 || final8_closer) << ',' << passed << '\n';
        }
    }
    require(cp::read((root / "checkpoint.bin").string()) == original, "Depth checkpoint changed");
    if (recovered)
        *recovered = fault.empty() && completed == 3 && recovery_ok;
    std::ofstream result(root / "result.json");
    result << std::setprecision(17) << "{\"start_unix_seconds\":" << begin
           << ",\"end_unix_seconds\":" << utc() << ",\"completed_cases\":" << completed
           << ",\"controls\":" << budget.controls << ",\"substeps\":" << budget.substeps
           << ",\"fault\":" << std::quoted(fault) << ",\"native_seconds\":"
           << std::chrono::duration<double>(std::chrono::steady_clock::now() - case_clock).count()
           << ",\"checkpoint_unchanged\":true,\"training_updates\":0,\"closed\":true}\n";
    result.close();
    std::ifstream immediate(root / "result.json");
    std::cout << immediate.rdbuf() << std::flush;
    return fault.empty() ? 0 : 3;
}
// Zero optimizer updates. Original trajectories only recover production observations;
// converted actor/critic are compared on identical observations, never used as assistance.
int verify_path_scale(const fs::path &root) {
    require(fs::create_directory(root / "run-started"), "Path-scale verification already attempted");
    active_wall_cap = 15;
    Budget budget;
    budget.maximum_controls = 320;
    budget.maximum_substeps = 3200;
    budget.seconds = 15;
    std::ofstream report(root / "equivalence.csv");
    report << std::setprecision(12)
           << "policy,rows,latent_max,angle_max,value_max,logp_max,gpu_latent_max,gpu_value_max,reproduction_"
              "max\n";
    for (int version : {288, 320}) {
        const auto suffix = std::to_string(version);
        const auto original = cp::read((root / ("policy-" + suffix + ".bin")).string());
        const auto old_protocol = cp::read((root / ("protocol-" + suffix + ".bin")).string());
        const auto old = rl::read_ppo_training_state(original, cp::checksum(old_protocol), seed, 1);
        auto converted = rl::rescale_ppo_path_inputs(old);
        cp::Writer provenance;
        provenance.value(uint32_t{0x32534f50});
        provenance.value(uint32_t{1});
        provenance.value(cp::checksum(original));
        provenance.value(cp::checksum(old_protocol));
        provenance.value(uint32_t{1});
        provenance.value(uint32_t{2});
        provenance.value(36.f);
        converted.source_protocol = cp::checksum(provenance.data);
        const auto encoded = rl::write_ppo_training_state(converted);
        const auto back = rl::read_ppo_training_state(encoded, converted.source_protocol, seed, 2);
        require(rl::write_ppo_training_state(back) == encoded, "Scaled checkpoint roundtrip failed");
        bool rejected = false;
        try {
            (void)rl::read_ppo_actor(encoded, converted.source_protocol, seed);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected, "Legacy evaluator accepted scaled observations");
        nn::CpuMlp<float> actor(spec(4)), next_actor(spec(4)), critic(spec(1)), next_critic(spec(1));
        actor.params() = old.actor.parameters;
        next_actor.params() = back.actor.parameters;
        critic.params() = old.critic;
        next_critic.params() = back.critic;
        std::vector<float> inputs, next_inputs, means, next_means, values, next_values;
        float latent_max = 0, angle_max = 0, value_max = 0, logp_max = 0, reproduction_max = 0;
        auto compare = [&](const float *x, const float *y) {
            inputs.insert(inputs.end(), x, x + observations);
            next_inputs.insert(next_inputs.end(), y, y + observations);
            actor.forward(x, 1);
            next_actor.forward(y, 1);
            critic.forward(x, 1);
            next_critic.forward(y, 1);
            for (int j = 0; j < 4; ++j) {
                latent_max = std::max(latent_max, fabsf(actor.output()[j] - next_actor.output()[j]));
                angle_max =
                    std::max(angle_max, fabsf(tanhf(actor.output()[j]) - tanhf(next_actor.output()[j])));
                means.push_back(actor.output()[j]);
                next_means.push_back(next_actor.output()[j]);
            }
            value_max = std::max(value_max, fabsf(critic.output()[0] - next_critic.output()[0]));
            values.push_back(critic.output()[0]);
            next_values.push_back(next_critic.output()[0]);
        };
        auto config = m::config_for_speed(.008f), next_config = config;
        config.maximum_controls = next_config.maximum_controls = 640;
        next_config.observation_schema = 2;
        auto world = m::initial_world(0);
        world.arena = m::long_swim_arena();
        m::ForwardTask task;
        require(m::begin_forward(world, config, task), "Scale reconstruction initialization");
        std::ifstream expected(root / ("expected-" + suffix + ".csv"));
        std::string line;
        std::getline(expected, line);
        for (int control = 1; control <= 160; ++control) {
            wall_guard();
            float base_obs[observations];
            require(m::observe(task, world, config, base_obs), "Scale baseline observation");
            actor.forward(base_obs, 1);
            m::Action action;
            require(m::action_from_latent(actor.output(), action), "Scale baseline action");
            for (int probe = 0; probe < 7; ++probe) {
                auto w = world;
                const auto motion = cudarl::motor::motion(w.physical.body);
                if (probe >= 1 && probe <= 3)
                    w.physical.body.position.y += (float(probe - 2) * .0005f -
                                                   (motion.com_model.y - task.start_com_model.y) * p::scale) /
                                                  p::scale;
                if (probe >= 4)
                    w.physical.body.velocity[1] += float(probe - 5) * .0005f - motion.velocity_m_s.y;
                float x[observations], y[observations];
                require(m::observe(task, w, config, x) && m::observe(task, w, next_config, y),
                        "Scale probe observation");
                compare(x, y);
            }
            budget.step(world, action.command);
            require(m::record_control(task, world, config).accepted, "Scale reconstruction task");
            require(bool(std::getline(expected, line)), "Missing saved scale reconstruction row");
            std::stringstream row(line);
            std::string field;
            std::vector<float> saved;
            while (std::getline(row, field, ','))
                saved.push_back(std::stof(field));
            const auto motion = cudarl::motor::motion(world.physical.body);
            const auto metrics = m::measure(task, world);
            std::vector<float> actual{float(control),
                                      metrics.signed_depth_m * 1000,
                                      motion.velocity_m_s.y * 1000,
                                      metrics.forward_m * 1000,
                                      metrics.tilt_deg,
                                      metrics.lateral_m * 1000};
            for (float v : action.command)
                actual.push_back(v);
            for (float v : world.physical.held_targets)
                actual.push_back(v);
            for (float v : world.physical.body.angle)
                actual.push_back(v);
            for (int j = 0; j < 4; ++j)
                actual.push_back(world.physical.body.velocity[6 + j]);
            require(actual.size() == saved.size(), "Scale reconstruction row shape");
            for (size_t i = 0; i < actual.size(); ++i)
                reproduction_max = std::max(reproduction_max, fabsf(actual[i] - saved[i]));
        }
        // Broader deterministic numerical probes, explicitly synthetic observations.
        for (int row = 0; row < 256; ++row) {
            float x[observations], y[observations];
            for (int j = 0; j < observations; ++j) {
                x[j] = .5f * sinf(float(row * 41 + j));
                if (j < 3)
                    x[j] *= .05f;
                y[j] = j < 3 ? x[j] * 36.f : x[j];
            }
            compare(x, y);
        }
        const int count = static_cast<int>(values.size());
        for (int row = 0; row < count; ++row) {
            const int previous = row ? row - 1 : 0;
            float z[4], zp[4];
            for (int j = 0; j < 4; ++j) {
                z[j] = means[row * 4 + j] + .15f;
                zp[j] = means[previous * 4 + j] - .1f;
            }
            const auto lp = rl::ar_logp(&means[row * 4], &means[previous * 4], zp, old.actor.log_sigma.data(),
                                        4, .8f, row != 0, z);
            const auto nq = rl::ar_logp(&next_means[row * 4], &next_means[previous * 4], zp,
                                        back.actor.log_sigma.data(), 4, .8f, row != 0, z);
            logp_max = std::max(logp_max, fabsf(lp - nq));
        }
        float gpu_latent_max = 0, gpu_value_max = 0;
        device_ptr<float> device_x(inputs.size());
        for (int network : {4, 1}) {
            nn::GpuMlp gpu_old(spec(network), count, seed), gpu_new(spec(network), count, seed);
            gpu_old.set_params(network == 4 ? old.actor.parameters : old.critic);
            gpu_new.set_params(network == 4 ? back.actor.parameters : back.critic);
            device_x.copy_from_host(inputs.data(), inputs.size());
            gpu_old.forward(device_x.get(), count);
            const auto a = download(gpu_old.output(), count * network, nullptr);
            device_x.copy_from_host(next_inputs.data(), next_inputs.size());
            gpu_new.forward(device_x.get(), count);
            const auto b = download(gpu_new.output(), count * network, nullptr);
            for (size_t i = 0; i < a.size(); ++i) {
                auto &maximum = network == 4 ? gpu_latent_max : gpu_value_max;
                maximum = std::max(maximum, fabsf(a[i] - b[i]));
            }
        }
        require(latent_max < 1e-5f && angle_max < 1e-5f && value_max < 1e-5f && logp_max < 1e-4f &&
                    gpu_latent_max < 1e-5f && gpu_value_max < 1e-5f && reproduction_max < 1e-5f,
                "Path-scale equivalence exceeded preregistered numerical limits");
        cp::write_atomic((root / ("scaled-protocol-" + suffix + ".bin")).string(), provenance.data);
        cp::write_atomic((root / ("scaled-policy-" + suffix + ".bin")).string(), encoded);
        require(cp::read((root / ("policy-" + suffix + ".bin")).string()) == original,
                "Source checkpoint changed");
        report << version << ',' << count << ',' << latent_max << ',' << angle_max << ',' << value_max << ','
               << logp_max << ',' << gpu_latent_max << ',' << gpu_value_max << ',' << reproduction_max
               << '\n';
        report.flush();
    }
    std::ofstream(root / "result.json")
        << "{\"passed\":true,\"controls\":" << budget.controls << ",\"substeps\":" << budget.substeps
        << ",\"optimizer_steps\":0,\"GPU_executed\":true}\n";
    return 0;
}
// Bounded frozen diagnostic. World-space perturbations only; no policy assistance.
int depth_sensitivity(const fs::path &root, bool continuation = false) {
    const int versions[2] = {288, 320}, phases[2][4] = {{81, 85, 88, 92}, {78, 82, 85, 89}};
    std::vector<uint8_t> payload[2];
    nn::CpuMlp<float> actors[2] = {nn::CpuMlp<float>(spec(actions)), nn::CpuMlp<float>(spec(actions))};
    std::vector<std::vector<float>> expected[2];
    for (int pidx = 0; pidx < 2; ++pidx) {
        const auto suffix = std::to_string(versions[pidx]);
        payload[pidx] = cp::read((root / ("policy-" + suffix + ".bin")).string());
        const auto policy = rl::read_ppo_actor(
            payload[pidx], cp::checksum(cp::read((root / ("protocol-" + suffix + ".bin")).string())), seed);
        require(policy.update == versions[pidx] && policy.correlation == .8f,
                "Sensitivity policy identity mismatch");
        actors[pidx].params() = policy.parameters;
        std::ifstream input(root / ("expected-" + suffix + ".csv"));
        std::string line;
        std::getline(input, line);
        while (std::getline(input, line)) {
            std::stringstream in(line);
            std::string field;
            std::vector<float> row;
            while (std::getline(in, field, ','))
                row.push_back(std::stof(field));
            require(row.size() == 22, "Saved sensitivity trace has wrong columns");
            expected[pidx].push_back(row);
        }
        require(expected[pidx].size() == 160, "Saved sensitivity trace does not cover8s");
    }
    struct Snapshot {
        m::World world;
        m::ForwardTask task;
    };
    Snapshot snapshots[2][4];
    auto config = m::config_for_speed(.008f);
    config.observation_schema = training_episode.observation_schema;
    config.maximum_controls = 640;
    require(fs::create_directory(root / "run-started"), "Sensitivity diagnostic already attempted");
    started = std::chrono::steady_clock::now();
    const auto utc = []() {
        return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    };
    const double begin = utc();
    Budget budget;
    budget.maximum_controls = 680;
    budget.maximum_substeps = 6800;
    budget.seconds = 5;
    if (continuation) {
        std::ifstream proof(root / "prior-result.json");
        const std::string text((std::istreambuf_iterator<char>(proof)), {});
        require(text.find("\"controls\":320,") != std::string::npos &&
                    text.find("\"completed_replays\":0,") != std::string::npos &&
                    text.find("Sensitivity perturbation altered unrelated observation") != std::string::npos,
                "Sensitivity continuation lacks completed reconstruction evidence");
        budget.controls = 320;
        budget.substeps = 3200;
    }
    std::cout << std::setprecision(17) << "SENSITIVITY_STARTED unix_seconds=" << begin << std::endl;
    std::ofstream reproduction(root / "reproduction.csv");
    reproduction << "policy,controls,max_defined_field_difference,passed\n";
    std::string fault;
    int completed_replays = 0, sensitivity_rows = 0;
    auto observe = [&](const Snapshot &s, float *obs) {
        require(m::observe(s.task, s.world, config, obs), "Sensitivity observation rejected");
        const auto motion = cudarl::motor::motion(s.world.physical.body);
        const float e = (motion.com_model.y - s.task.start_com_model.y) * p::scale * 1000;
        const float decoded = (-1000 * cudarl::motor_observation::path_scale_m(config.observation_schema)) *
                              (obs[0] * obs[9] + obs[1] * obs[10] + obs[2] * obs[11]);
        const float vy = motion.velocity_m_s.y * 1000,
                    decoded_vy = 20 * (obs[12] * obs[9] + obs[13] * obs[10] + obs[14] * obs[11]);
        require(fabsf(e - decoded) < .0001f && fabsf(vy - decoded_vy) < .0001f,
                "Sensitivity encoded depth/velocity mismatch");
    };
    auto move_depth = [&](Snapshot &s, float e_mm) {
        const auto motion = cudarl::motor::motion(s.world.physical.body);
        s.world.physical.body.position.y +=
            (e_mm / 1000 - (motion.com_model.y - s.task.start_com_model.y) * p::scale) / p::scale;
    };
    auto fields = [&](const Snapshot &s, int control, const float *command) {
        const auto motion = cudarl::motor::motion(s.world.physical.body);
        const auto metrics = m::measure(s.task, s.world);
        std::vector<float> out{float(control),
                               metrics.signed_depth_m * 1000,
                               motion.velocity_m_s.y * 1000,
                               metrics.forward_m * 1000,
                               metrics.tilt_deg,
                               metrics.lateral_m * 1000};
        for (int j = 0; j < 4; ++j)
            out.push_back(command[j]);
        for (float v : s.world.physical.held_targets)
            out.push_back(v);
        for (float v : s.world.physical.body.angle)
            out.push_back(v);
        for (int j = 0; j < 4; ++j)
            out.push_back(s.world.physical.body.velocity[6 + j]);
        return out;
    };
    try {
        for (int pidx = 0; pidx < (continuation ? 1 : 2); ++pidx) {
            Snapshot state;
            state.world = m::initial_world(0);
            state.world.arena = m::long_swim_arena();
            require(m::begin_forward(state.world, config, state.task),
                    "Sensitivity reconstruction initial state");
            float maximum = 0;
            const int last_control = continuation ? 92 : 160;
            for (int t = 1; t <= last_control; ++t) {
                float obs[observations];
                observe(state, obs);
                actors[pidx].forward(obs, 1);
                m::Action action;
                require(m::action_from_latent(actors[pidx].output(), action),
                        "Sensitivity reconstruction action");
                budget.step(state.world, action.command);
                require(m::record_control(state.task, state.world, config).accepted,
                        "Sensitivity reconstruction task");
                const auto actual = fields(state, t, action.command);
                require(expected[pidx][t - 1][0] == t, "Sensitivity saved control ordering");
                for (size_t j = 0; j < actual.size(); ++j) {
                    maximum = fmaxf(maximum, fabsf(actual[j] - expected[pidx][t - 1][j]));
                    require(fabsf(actual[j] - expected[pidx][t - 1][j]) < .0001f,
                            "Baseline reconstruction differs from saved swim; no counterfactuals");
                }
                for (int phase = 0; phase < 4; ++phase)
                    if (t == phases[pidx][phase])
                        snapshots[pidx][phase] = state;
            }
            reproduction << versions[pidx] << ',' << last_control << ',' << std::setprecision(12) << maximum
                         << ",1\n";
            reproduction.flush();
        }
        std::cout << "RECONSTRUCTION_VERIFIED both_policies controls=" << budget.controls << std::endl;
        std::ofstream table(root / "sensitivity.csv");
        table << std::setprecision(12)
              << "source_policy,source_control,time_s,tested_policy,variant,error_mm,vy_mm_s,decoded_error_"
                 "mm,decoded_vy_mm_s";
        for (int j : {0, 1, 2, 9, 10, 11, 12, 13, 14})
            table << ",obs" << j;
        for (const char *name : {"latent", "command"})
            for (int j = 0; j < 4; ++j)
                table << ',' << name << j;
        table << '\n';
        const char *variants[] = {"original", "depth_minus", "depth_zero", "depth_plus",
                                  "vy_minus", "vy_zero",     "vy_plus"};
        double maximum_unrelated_roundoff = 0;
        for (int source = 0; source < (continuation ? 1 : 2); ++source)
            for (int phase = 0; phase < 4; ++phase) {
                const auto &baseline = snapshots[source][phase];
                float original_obs[observations];
                observe(baseline, original_obs);
                for (int kind = 0; kind < 7; ++kind) {
                    auto state = baseline;
                    if (kind >= 1 && kind <= 3)
                        move_depth(state, (kind - 2) * .5f);
                    if (kind >= 4) {
                        const auto motion = cudarl::motor::motion(state.world.physical.body);
                        state.world.physical.body.velocity[1] += (kind - 5) * .0005f - motion.velocity_m_s.y;
                    }
                    float obs[observations];
                    observe(state, obs);
                    for (int j = 0; j < observations; ++j) {
                        const bool changed =
                            (kind >= 1 && kind <= 3) ? j < 3 : (kind >= 4 && j >= 12 && j < 15);
                        if (!changed)
                            maximum_unrelated_roundoff =
                                std::max(maximum_unrelated_roundoff, fabs(double(obs[j]) - original_obs[j]));
                    }
                    // Primitive state is copied, then ONLY root Y or root Vy is changed.
                    // COM Jacobians can differ by a few FP32 ULPs after translation;
                    // measure this rather than requiring cross-derived values bitwise equal.
                    for (int j = 0; j < 4; ++j)
                        require(state.world.physical.body.angle[j] == baseline.world.physical.body.angle[j] &&
                                    state.world.physical.held_targets[j] ==
                                        baseline.world.physical.held_targets[j] &&
                                    state.world.reference_rate[j] == baseline.world.reference_rate[j],
                                "Perturbation changed joint/filter state");
                    for (int j = 0; j < 10; ++j)
                        if (!(kind >= 4 && j == 1))
                            require(state.world.physical.body.velocity[j] ==
                                        baseline.world.physical.body.velocity[j],
                                    "Perturbation changed angular/joint velocity");
                    const auto motion = cudarl::motor::motion(state.world.physical.body);
                    const float error = (motion.com_model.y - state.task.start_com_model.y) * p::scale * 1000;
                    for (int subject = 0; subject < 2; ++subject) {
                        actors[subject].forward(obs, 1);
                        m::Action action;
                        require(m::action_from_latent(actors[subject].output(), action),
                                "Sensitivity action");
                        table << versions[source] << ',' << state.task.controls << ','
                              << state.task.controls * .05 << ',' << versions[subject] << ','
                              << variants[kind] << ',' << error << ',' << motion.velocity_m_s.y * 1000 << ','
                              << (-1000 *
                                  cudarl::motor_observation::path_scale_m(config.observation_schema)) *
                                     (obs[0] * obs[9] + obs[1] * obs[10] + obs[2] * obs[11])
                              << ',' << 20 * (obs[12] * obs[9] + obs[13] * obs[10] + obs[14] * obs[11]);
                        for (int j : {0, 1, 2, 9, 10, 11, 12, 13, 14})
                            table << ',' << obs[j];
                        for (float v : action.latent)
                            table << ',' << v;
                        for (float v : action.command)
                            table << ',' << v;
                        table << '\n';
                        ++sensitivity_rows;
                    }
                }
            }
        table.close();
        std::ofstream observation_report(root / "observation-verification.json");
        observation_report
            << std::setprecision(17)
            << "{\"max_unrelated_derived_observation_roundoff\":" << maximum_unrelated_roundoff
            << ",\"joint_filter_and_angular_states_exact\":true,\"signed_decoding_verified\":true}\n";
        std::ofstream trace(root / "paired-controls.csv");
        trace << std::setprecision(12)
              << "policy,start_error_mm,step,relative_s,absolute_s,error_mm,vy_mm_s,forward_mm,forward_"
                 "velocity_mm_s,tilt_deg,lane_mm,reward";
        for (const char *name : {"command", "filtered", "angle", "rate", "torque"})
            for (int j = 0; j < 4; ++j)
                trace << ',' << name << j;
        trace << '\n';
        for (int subject = 0; subject < 2; ++subject)
            for (float offset : {-.5f, 0.f, .5f}) {
                auto state = snapshots[continuation ? 0 : subject][0];
                move_depth(state, offset);
                const auto origin = cudarl::motor::motion(state.world.physical.body).com_model;
                const int control0 = state.task.controls;
                auto row = [&](int n, const m::Transition &interval, const float *command) {
                    const auto motion = cudarl::motor::motion(state.world.physical.body);
                    const auto metrics = m::measure(state.task, state.world);
                    trace << versions[subject] << ',' << offset << ',' << n << ',' << n * .05 << ','
                          << state.task.controls * .05 << ',' << metrics.signed_depth_m * 1000 << ','
                          << motion.velocity_m_s.y * 1000 << ','
                          << p::dot((motion.com_model - origin) * p::scale, state.task.request.heading) * 1000
                          << ',' << p::dot(motion.velocity_m_s, state.task.request.heading) * 1000 << ','
                          << metrics.tilt_deg << ',' << metrics.lateral_m * 1000 << ',' << interval.reward;
                    for (int j = 0; j < 4; ++j)
                        trace << ',' << command[j];
                    for (float v : state.world.physical.held_targets)
                        trace << ',' << v;
                    for (float v : state.world.physical.body.angle)
                        trace << ',' << v;
                    for (int j = 0; j < 4; ++j)
                        trace << ',' << state.world.physical.body.velocity[6 + j];
                    for (int j = 0; j < 4; ++j) {
                        float peak = 0;
                        for (int k = 0; k < 10; ++k)
                            peak = fmaxf(peak, fabsf(state.world.physical.actual_torques[k * 4 + j]));
                        trace << ',' << peak * p::torque;
                    }
                    trace << '\n';
                };
                float obs[observations];
                observe(state, obs);
                actors[subject].forward(obs, 1);
                m::Action action;
                require(m::action_from_latent(actors[subject].output(), action), "Paired initial action");
                row(0, {}, action.command);
                for (int n = 1; n <= (continuation ? 40 : 60); ++n) {
                    observe(state, obs);
                    actors[subject].forward(obs, 1);
                    require(m::action_from_latent(actors[subject].output(), action), "Paired action");
                    p::BodyState before{}, candidate{};
                    cudarl::movement::Telemetry telemetry{};
                    try {
                        budget.step(state.world, action.command, &telemetry, &before, &candidate);
                    } catch (const std::exception &e) {
                        std::ofstream failure(root / "failure.json");
                        failure << std::setprecision(17) << "{\"policy\":" << versions[subject]
                                << ",\"offset_mm\":" << offset << ",\"relative_attempted_s\":"
                                << (state.world.attempted_substeps - control0 * 10) * .005
                                << ",\"code\":" << state.world.failure_code
                                << ",\"fault\":" << std::quoted(e.what()) << ",\"before\":";
                        state_json(failure, before);
                        failure << ",\"candidate\":";
                        state_json(failure, candidate);
                        failure << "}\n";
                        throw;
                    }
                    const auto interval = m::record_control(state.task, state.world, config);
                    require(interval.accepted, "Paired task rejected");
                    row(n, interval, action.command);
                }
                trace.flush();
                ++completed_replays;
            }
        require(budget.controls == (continuation ? 652 : 680) &&
                    budget.substeps == (continuation ? 6520 : 6800) &&
                    sensitivity_rows == (continuation ? 56 : 112) && completed_replays == 6,
                "Sensitivity accounting mismatch");
    } catch (const std::exception &e) {
        fault = e.what();
    }
    for (int i = 0; i < 2; ++i)
        require(cp::read((root / ("policy-" + std::to_string(versions[i]) + ".bin")).string()) == payload[i],
                "Sensitivity checkpoint changed");
    std::ofstream result(root / "result.json");
    result << std::setprecision(17) << "{\"start_unix_seconds\":" << begin
           << ",\"end_unix_seconds\":" << utc() << ",\"controls\":" << budget.controls
           << ",\"substeps\":" << budget.substeps << ",\"sensitivity_rows\":" << sensitivity_rows
           << ",\"completed_replays\":" << completed_replays << ",\"fault\":" << std::quoted(fault)
           << ",\"training_updates\":0,\"checkpoints_unchanged\":true,\"native_seconds\":"
           << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
           << ",\"closed\":true}\n";
    result.close();
    std::ifstream immediate(root / "result.json");
    std::cout << immediate.rdbuf() << std::flush;
    return fault.empty() ? 0 : 3;
}
int diagnose_safety(const fs::path &root) {
    require(seed == 1247037242 || seed == 1247037243, "Only declared safety cases");
    const int cap = seed == 1247037242 ? 630 : 1129;
    const uint64_t expected_attempts = seed == 1247037242 ? 6295 : 11282;
    require(!fs::exists(root / "safety-replay.json"), "Safety replay already attempted; no retry");
    const auto bytes = cp::read((root / "ppo-update-128.bin").string());
    const auto policy =
        rl::read_ppo_actor(bytes, cp::checksum(cp::read((root / "protocol.bin").string())), seed);
    require(policy.update == 128 && policy.correlation == .8f, "Safety policy identity mismatch");
    nn::CpuMlp<float> actor(spec(actions));
    actor.params() = policy.parameters;
    auto world = m::initial_world(0);
    world.arena = m::long_swim_arena();
    auto config = m::config_for_speed(.008f);
    config.observation_schema = training_episode.observation_schema;
    config.maximum_controls = 1200;
    m::ForwardTask task;
    require(m::begin_forward(world, config, task), "Safety replay initial state");
    const double utc =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    const auto begin = std::chrono::steady_clock::now();
    std::ofstream trace(root / "safety-replay-controls.csv");
    trace << std::setprecision(12) << "control,forward_mm,depth_mm,lateral_mm,tilt_deg,clearance_mm\n";
    for (int control = 1; control <= cap; ++control) {
        require(std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() < 10,
                "Safety replay wall cap");
        float obs[observations];
        require(m::observe(task, world, config, obs), "Safety replay observation");
        actor.forward(obs, 1);
        m::Action action;
        require(m::action_from_latent(actor.output(), action), "Safety replay action");
        p::BodyState before, candidate;
        cudarl::movement::Telemetry telemetry;
        if (!cudarl::joint_motor::step_state(world, action.command, &telemetry, &before, &candidate)) {
            float minimum = 1e6f;
            int face_min = -1, body_min = -1;
            const auto shape = cudarl::movement::shape(candidate);
            for (int b = 0; b < 3; ++b)
                for (int face = 0; face < 6; ++face) {
                    const auto &link = shape.link[b];
                    const auto n = cudarl::contact::normal(face), u = link.q.inverse(n);
                    const auto g = cudarl::movement::body_geometry(b);
                    const float support = b == 0 ? sqrtf(g.length * g.length * .25f * u.x * u.x +
                                                         g.half_height * g.half_height * u.y * u.y +
                                                         g.half_width * g.half_width * u.z * u.z)
                                                 : fabsf(u.x) * g.length * .5f +
                                                       sqrtf(g.half_height * g.half_height * u.y * u.y +
                                                             g.half_width * g.half_width * u.z * u.z);
                    const float gap =
                        p::dot(n, link.center) + cudarl::contact::offset(face, world.arena) - support;
                    if (gap < minimum) {
                        minimum = gap;
                        face_min = face;
                        body_min = b;
                    }
                }
            const char *names[6] = {"left_wall",     "right_wall", "floor",
                                    "water_surface", "back_wall",  "front_wall"};
            std::ofstream out(root / "safety-replay.json");
            out << std::setprecision(17) << "{\"seed\":" << seed
                << ",\"pose\":0,\"replay\":true,\"started_unix\":" << utc << ",\"ended_unix\":"
                << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()
                << ",\"controls\":" << control << ",\"substeps\":" << world.attempted_substeps
                << ",\"expected_attempts_match\":"
                << (world.attempted_substeps == expected_attempts ? "true" : "false")
                << ",\"code\":" << world.failure_code
                << ",\"failed_attempt_time_s\":" << double(world.attempted_substeps) * .005
                << ",\"before_attempt_time_s\":" << double(world.attempted_substeps - 1) * .005
                << ",\"telemetry_minimum_gap_m\":" << telemetry.minimum_gap
                << ",\"candidate_minimum_gap_m\":" << minimum << ",\"clearance_limit_m\":0.01,\"boundary\":"
                << std::quoted(face_min >= 0 ? names[face_min] : "unknown") << ",\"link\":" << body_min
                << ",\"prior_clearance_m\":" << cudarl::movement::clearance(before, true, world.arena)
                << ",\"last_accepted_clearance_m\":"
                << cudarl::movement::clearance(world.physical.body, true, world.arena)
                << ",\"last_energy_error_J\":" << world.last_energy_error
                << ",\"energy_error_limit_J\":" << .02 * (world.positive_work + world.fluid_loss) + 2e-10
                << ",\"before\":";
            state_json(out, before);
            out << ",\"candidate_NOT_committed\":";
            state_json(out, candidate);
            out << ",\"after_return\":";
            state_json(out, world.physical.body);
            out << ",\"checkpoint_unchanged\":"
                << (cp::read((root / "ppo-update-128.bin").string()) == bytes ? "true" : "false") << "}\n";
            out.close();
            require(world.attempted_substeps == expected_attempts && control == cap,
                    "Safety replay prefix differs from original; no retry");
            std::cout << "SAFETY_REPLAY code=" << world.failure_code
                      << " boundary=" << (face_min >= 0 ? names[face_min] : "unknown") << " gap_m=" << minimum
                      << " at_s=" << double(world.attempted_substeps) * .005 << std::endl;
            return 0;
        }
        const auto interval = m::record_control(task, world, config);
        require(interval.accepted, "Safety replay task rejected");
        trace << control << ',' << interval.metrics.forward_m * 1000 << ',' << interval.metrics.depth_m * 1000
              << ',' << interval.metrics.lateral_m * 1000 << ',' << interval.metrics.tilt_deg << ','
              << cudarl::movement::clearance(world.physical.body, true, world.arena) * 1000 << '\n';
    }
    throw std::runtime_error("Expected safety stop not reproduced within cap; no extension");
}
int neutral_pitch_replay(const fs::path &root, bool continue_recorded = false) {
    require(seed == 1247037243 && correlation == .8f, "Wrong frozen recovery identity");
    if (continue_recorded) {
        require(fs::exists(root / "run-started") && !fs::exists(root / "controls.csv"),
                "Recovery already started or reference absent");
        require(fs::create_directory(root / "recovery-started"), "Recovery branches already attempted");
    } else
        require(fs::create_directory(root / "run-started"), "Reference already attempted");
    auto read_text = [](const fs::path &path) {
        std::ifstream input(path, std::ios::binary);
        require(bool(input), "Cannot open diagnostic text");
        std::ostringstream content;
        content << input.rdbuf();
        return content.str();
    };
    const auto checkpoint = cp::read((root / "ppo-update-128.bin").string());
    const auto policy =
        rl::read_ppo_actor(checkpoint, cp::checksum(cp::read((root / "protocol.bin").string())), seed);
    require(policy.update == 128 && policy.correlation == .8f, "Wrong policy");
    struct Record {
        float command[4]{}, filtered[4]{}, angle[4]{}, rate[4]{};
        float tilt = 0, lane = 0, depth = 0, work = 0, forward = 0;
    };
    Record original[160];
    std::vector<Record> recorded(277);
    std::ifstream input(root / "recorded-targets.txt");
    for (int i = 0; i < 160; ++i) {
        int control = 0;
        auto &r = original[i];
        input >> control;
        for (float &v : r.command)
            input >> v;
        for (float &v : r.filtered)
            input >> v;
        for (float &v : r.angle)
            input >> v;
        for (float &v : r.rate)
            input >> v;
        input >> r.tilt >> r.lane >> r.depth >> r.work >> r.forward;
        require(bool(input) && control == i + 1, "Missing original target row");
    }
    std::string extra;
    require(!(input >> extra), "Unexpected target row");
    auto describe = [](const m::World &w, const m::ForwardTask &t, const float *command) {
        Record r;
        const auto metrics = m::measure(t, w);
        for (int j = 0; j < 4; ++j) {
            r.command[j] = command[j];
            r.filtered[j] = w.physical.held_targets[j];
            r.angle[j] = w.physical.body.angle[j];
            r.rate[j] = w.physical.body.velocity[6 + j];
        }
        r.tilt = metrics.tilt_deg;
        r.lane = metrics.lateral_m * 1000;
        r.depth = metrics.depth_m * 1000;
        r.forward = metrics.forward_m * 1000;
        r.work = w.physical.body.work;
        return r;
    };
    auto compare = [](const Record &a, const Record &b, float *diff) {
        const float fields[4] = {fabsf(a.forward - b.forward), fabsf(a.lane - b.lane),
                                 fabsf(a.depth - b.depth), fabsf(a.tilt - b.tilt)};
        for (int j = 0; j < 4; ++j) {
            diff[j] = fmaxf(diff[j], fields[j]);
            diff[4] = fmaxf(diff[4], fabsf(a.filtered[j] - b.filtered[j]));
            diff[5] = fmaxf(diff[5], fabsf(a.angle[j] - b.angle[j]));
            diff[6] = fmaxf(diff[6], fabsf(a.rate[j] - b.rate[j]));
            diff[8] = fmaxf(diff[8], fabsf(a.command[j] - b.command[j]));
        }
        diff[7] = fmaxf(diff[7], fabsf(a.work - b.work));
    };
    auto valid_diff = [](const float *d) {
        return d[0] <= .001f && d[1] <= .001f && d[2] <= .001f && d[3] <= .001f && d[4] <= 1e-5f &&
               d[5] <= 1e-5f && d[6] <= 1e-4f && d[7] <= 1e-9f && d[8] <= 2e-6f;
    };
    auto save_diff = [&](const char *file, const float *diff) {
        std::ofstream out(root / file);
        out << std::setprecision(17) << "{\"passed\":" << (valid_diff(diff) ? "true" : "false")
            << ",\"maxdiff_forward_lane_depth_mm_tilt_deg_filter_angle_rad_rate_rad_s_work_J_command_rad\":[";
        for (int j = 0; j < 9; ++j)
            out << (j ? "," : "") << diff[j];
        out << "]}\n";
    };
    nn::CpuMlp<float> actor(spec(actions));
    actor.params() = policy.parameters;
    Budget budget;
    budget.maximum_controls = 597;
    budget.maximum_substeps = 5970;
    budget.seconds = 5;
    started = std::chrono::steady_clock::now();
    auto config = m::config_for_speed(.008f);
    config.observation_schema = training_episode.observation_schema;
    config.maximum_controls = 320;
    auto reference = m::initial_world(0);
    m::ForwardTask reference_task;
    require(m::begin_forward(reference, config, reference_task), "Reference init failed");
    m::World source;
    m::ForwardTask source_task;
    float overlap[9]{};
    if (continue_recorded) {
        // The failed first process completed the reference recording but performed
        // no branch physics. Reconstruct task bookkeeping from saved worlds only.
        budget.controls = 277;
        budget.substeps = 2770;
        budget.seconds = 4.8;
        std::ifstream states(root / "reference-full-world-states.bin", std::ios::binary | std::ios::ate);
        require(states && states.tellg() == std::streamoff(277 * sizeof(m::World)),
                "Invalid saved reference world count/ABI");
        states.seekg(0);
        std::ifstream commands(root / "reference-targets.csv");
        std::string line;
        std::getline(commands, line);
        for (int i = 0; i < 277; ++i) {
            require(bool(std::getline(commands, line)), "Missing saved reference target");
            for (char &c : line)
                if (c == ',')
                    c = ' ';
            std::istringstream row(line);
            int control = 0;
            double time = 0;
            float command[4]{};
            row >> control >> time;
            for (float &v : command)
                row >> v;
            require(row && control == i + 1 && fabs(time - (i + 1) * .05) < 1e-12,
                    "Saved reference order mismatch");
            states.read(reinterpret_cast<char *>(&reference), sizeof(reference));
            require(bool(states) && reference.physical.control_steps == uint64_t(i + 1),
                    "Saved reference world mismatch");
            require(m::record_control(reference_task, reference, config).accepted,
                    "Saved task reconstruction failed");
            recorded[i] = describe(reference, reference_task, command);
            if (i == 116) {
                source = reference;
                source_task = reference_task;
            }
            if (i < 160)
                compare(recorded[i], original[i], overlap);
        }
        require(!std::getline(commands, line), "Extra saved target row");
        save_diff("continued-overlap-verification.json", overlap);
        require(valid_diff(overlap), "Saved reference overlap mismatch; branches not run");
    } else {
        std::ofstream recording(root / "reference-targets.csv");
        recording << std::setprecision(17) << "control,time_s,command0,command1,command2,command3\n";
        std::ofstream state_stream(root / "reference-full-world-states.bin", std::ios::binary);
        for (int i = 0; i < 277; ++i) {
            float obs[observations];
            require(m::observe(reference_task, reference, config, obs), "Reference observation failed");
            actor.forward(obs, 1);
            m::Action action;
            require(m::action_from_latent(actor.output(), action), "Reference action failed");
            budget.step(reference, action.command);
            require(m::record_control(reference_task, reference, config).accepted, "Reference task failed");
            recorded[i] = describe(reference, reference_task, action.command);
            recording << i + 1 << ',' << (i + 1) * .05;
            for (float v : action.command)
                recording << ',' << v;
            recording << '\n';
            state_stream.write(reinterpret_cast<const char *>(&reference), sizeof(reference));
            if (i == 116) {
                source = reference;
                source_task = reference_task;
            }
            if (i < 160)
                compare(recorded[i], original[i], overlap);
            if (i == 159) {
                save_diff("overlap-verification.json", overlap);
                require(valid_diff(overlap), "Original0to8s overlap failed; no recovery branches");
            }
        }
        recording.close();
        state_stream.close();
    }
    const auto frozen_record = read_text(root / "reference-targets.csv");
    const auto initial = m::measure(source_task, source);
    const auto initial_motion = cudarl::motor::motion(source.physical.body);
    require(fabsf((initial_motion.com_model.y - source_task.start_com_model.y) * p::scale * 1000 +
                  .512504518f) < .0001f &&
                fabsf(initial_motion.velocity_m_s.y * 1000 + .642554343f) < .0001f,
            "Sinking state mismatch");
    {
        std::ofstream world_out(root / "start-world.bin", std::ios::binary),
            task_out(root / "start-task.bin", std::ios::binary);
        world_out.write(reinterpret_cast<const char *>(&source), sizeof(source));
        task_out.write(reinterpret_cast<const char *>(&source_task), sizeof(source_task));
    }
    std::ofstream saved(root / "start-state.json");
    saved << std::setprecision(17) << "{\"time_s\":" << source_task.controls * .05
          << ",\"task_controls\":" << source_task.controls
          << ",\"world_controls\":" << source.physical.control_steps
          << ",\"world_substeps\":" << source.physical.substeps
          << ",\"forward_mm\":" << initial.forward_m * 1000 << ",\"signed_depth_error_mm\":"
          << (initial_motion.com_model.y - source_task.start_com_model.y) * p::scale * 1000
          << ",\"vertical_velocity_mm_s\":" << initial_motion.velocity_m_s.y * 1000 << ",\"held_targets\":[";
    for (int j = 0; j < 4; ++j)
        saved << (j ? "," : "") << source.physical.held_targets[j];
    saved << "],\"filter_rates\":[";
    for (int j = 0; j < 4; ++j)
        saved << (j ? "," : "") << source.reference_rate[j];
    saved << "],\"signed_work_J\":" << source.signed_work << ",\"positive_work_J\":" << source.positive_work
          << ",\"fluid_loss_J\":" << source.fluid_loss << ",\"numerical_loss_J\":" << source.numerical_loss
          << ",\"external_work_J\":" << source.external_work << ",\"body\":";
    state_json(saved, source.physical.body);
    saved << "}\n";
    saved.close();
    require(read_text(root / "start-state.json") == read_text(root / "expected-start-state.json"),
            "Complete saved start-state mismatch; no branches");
    struct Result {
        float depth = 0, vy = 0, maxdepth = 0, tilt = 0, forward = 0, steady = 0, mean = 0;
        double reward = 0, cost[5]{};
    } results[2];
    const char *names[2] = {"all_recorded_targets", "same_recorded_yaw_zero_pitch"};
    float reproduction[9]{};
    std::ofstream trace(root / "controls.csv");
    trace << std::setprecision(12)
          << "branch,control,time_s,signed_depth_mm,vertical_velocity_mm_s,forward_mm,forward_velocity_mm_s,"
             "tilt_deg,reward,speed_cost,lane_cost,depth_cost,posture_cost,effort_cost";
    for (int j = 0; j < 4; ++j)
        trace << ",command" << j << ",filtered" << j << ",actual" << j << ",rate" << j;
    trace << '\n';
    for (int branch = 0; branch < 2; ++branch) {
        auto world = source;
        auto task = source_task;
        auto &r = results[branch];
        r.maxdepth = initial.depth_m * 1000;
        r.tilt = initial.tilt_deg;
        float steady_origin = 0;
        for (int i = 0; i < 160; ++i) {
            const auto &expected = recorded[117 + i];
            float command[4];
            for (int j = 0; j < 4; ++j)
                command[j] = expected.command[j];
            if (branch) {
                command[1] = 0;
                command[3] = 0;
            }
            require(command[0] == expected.command[0] && command[2] == expected.command[2],
                    "Recorded yaw mismatch");
            budget.step(world, command);
            const auto interval = m::record_control(task, world, config);
            require(interval.accepted, "Recovery step failed");
            const auto motion = cudarl::motor::motion(world.physical.body);
            const float depth = (motion.com_model.y - task.start_com_model.y) * p::scale * 1000;
            if (!branch)
                compare(describe(world, task, command), expected, reproduction);
            r.maxdepth = fmaxf(r.maxdepth, fabsf(depth));
            r.tilt = fmaxf(r.tilt, interval.metrics.tilt_deg);
            r.reward += interval.reward;
            const float costs[5] = {interval.costs.speed, interval.costs.lane, interval.costs.depth,
                                    interval.costs.posture, interval.costs.effort};
            for (int j = 0; j < 5; ++j)
                r.cost[j] += .05 * costs[j];
            if (i == 39)
                steady_origin = interval.metrics.forward_m;
            trace << names[branch] << ',' << task.controls << ',' << task.controls * .05 << ',' << depth
                  << ',' << motion.velocity_m_s.y * 1000 << ',' << interval.metrics.forward_m * 1000 << ','
                  << p::dot(motion.velocity_m_s, task.request.heading) * 1000 << ','
                  << interval.metrics.tilt_deg << ',' << interval.reward;
            for (float c : costs)
                trace << ',' << c;
            for (int j = 0; j < 4; ++j)
                trace << ',' << command[j] << ',' << world.physical.held_targets[j] << ','
                      << world.physical.body.angle[j] << ',' << world.physical.body.velocity[6 + j];
            trace << '\n';
        }
        const auto motion = cudarl::motor::motion(world.physical.body);
        r.depth = (motion.com_model.y - task.start_com_model.y) * p::scale * 1000;
        r.vy = motion.velocity_m_s.y * 1000;
        r.forward = (m::measure(task, world).forward_m - initial.forward_m) * 1000;
        r.mean = r.forward / 8;
        r.steady = (m::measure(task, world).forward_m - steady_origin) * 1000 / 6;
        if (!branch) {
            save_diff("branch-reproduction.json", reproduction);
            trace.flush();
            require(valid_diff(reproduction),
                    "Recorded fulltarget recovery failed reproduction; neutral not run");
        }
    }
    require(budget.controls == 597 && budget.substeps == 5970, "Recovery accounting mismatch");
    require(cp::read((root / "ppo-update-128.bin").string()) == checkpoint, "Checkpoint changed");
    require(read_text(root / "reference-targets.csv") == frozen_record, "Frozen reference changed");
    std::ofstream out(root / "result.json");
    out << std::setprecision(17) << "{\"training\":0,\"controls\":" << budget.controls
        << ",\"substeps\":" << budget.substeps
        << ",\"checkpoint_unchanged\":true,\"recorded_yaw_identical\":true,\"fullstate_cloned\":true,"
           "\"branches\":[";
    for (int b = 0; b < 2; ++b) {
        const auto &r = results[b];
        out << (b ? "," : "") << "{\"name\":" << std::quoted(names[b])
            << ",\"final_signed_depth_mm\":" << r.depth << ",\"final_vy_mm_s\":" << r.vy
            << ",\"maximum_absolute_depth_mm\":" << r.maxdepth << ",\"peak_tilt_deg\":" << r.tilt
            << ",\"forward_travel_mm\":" << r.forward << ",\"mean_forward_speed_mm_s\":" << r.mean
            << ",\"relative2to8_speed_mm_s\":" << r.steady << ",\"reward\":" << r.reward
            << ",\"integrated_costs_speed_lane_depth_posture_effort\":[";
        for (int j = 0; j < 5; ++j)
            out << (j ? "," : "") << r.cost[j];
        out << "]}";
    }
    out << "],\"native_seconds\":"
        << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << "}\n";
    out.close();
    std::ifstream immediate(root / "result.json");
    std::cout << immediate.rdbuf() << std::flush;
    return 0;
}
// Shared by production training and the save/reload verification: no shadow updater.
void collect_rollout(int update, nn::GpuMlp &actor, nn::GpuMlp &critic, Rollout &data, cudaStream_t stream,
                     const fs::path &diagnostics = {}) {
    training_episode.offset_phase = update % 3;
    gpu::initialize<<<1, 32, 0, stream>>>(data.worlds.get(), data.tasks.get(), environments, data.fault.get(),
                                          training_episode);
    if (depth_offsets) {
        const auto worlds = download(data.worlds.get(), environments, stream);
        const auto tasks = download(data.tasks.get(), environments, stream);
        const bool header = !diagnostics.empty() && !fs::exists(diagnostics / "gpu-reset-starts.csv");
        std::ofstream reset_log;
        if (!diagnostics.empty())
            reset_log.open(diagnostics / "gpu-reset-starts.csv", std::ios::app);
        if (header)
            reset_log << "update,world,pose,offset_mm,target_model_y,request_model_y,actual_error_mm\n";
        reset_log << std::setprecision(12);
        for (int i = 0; i < environments; ++i) {
            int centered = i % 2;
            while (training_episode.initial_vertical_offset(centered) != 0)
                centered += 2;
            const auto target = tasks[centered].start_com_model;
            const auto actual = cudarl::motor::motion(worlds[i].physical.body).com_model;
            require(target.y == -2 && p::norm(tasks[i].start_com_model - target) == 0 &&
                        p::norm(tasks[i].request.position - tasks[centered].request.position) == 0 &&
                        tasks[i].request.position.y == target.y &&
                        fabsf((actual.y - target.y) * p::scale -
                              training_episode.initial_vertical_offset(i)) < 1e-8f,
                    "Device reset rebased fixed depth target");
            if (reset_log)
                reset_log << update << ',' << i << ',' << (i % 2 ? 3 : 0) << ','
                          << training_episode.initial_vertical_offset(i) * 1000 << ',' << target.y << ','
                          << tasks[i].request.position.y << ',' << (actual.y - target.y) * p::scale * 1000
                          << '\n';
        }
    }
    for (int t = 0; t < horizon; ++t) {
        const int offset = t * environments;
        float *obs = data.obs.get() + offset * observations;
        gpu::observe<<<1, 32, 0, stream>>>(data.worlds.get(), data.tasks.get(), obs, environments,
                                           data.fault.get(), training_episode);
        float *previous_obs = data.previous_obs.get() + offset * observations;
        float *previous_action = data.previous_action.get() + offset * actions;
        float *history = data.history.get() + offset;
        if (t == 0) {
            CUDA_CHECK_THROW(
                cudaMemsetAsync(previous_obs, 0, environments * observations * sizeof(float), stream));
            CUDA_CHECK_THROW(
                cudaMemsetAsync(previous_action, 0, environments * actions * sizeof(float), stream));
        } else {
            CUDA_CHECK_THROW(cudaMemcpyAsync(
                previous_obs, data.obs.get() + (offset - environments) * observations,
                environments * observations * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK_THROW(
                cudaMemcpyAsync(previous_action, data.action.get() + (offset - environments) * actions,
                                environments * actions * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
        gpu::set_history<<<1, 32, 0, stream>>>(history, environments, t != 0);
        CUDA_CHECK_THROW(cudaMemcpyAsync(data.actor_input.get(), obs,
                                         environments * observations * sizeof(float),
                                         cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(data.actor_input.get() + environments * observations, previous_obs,
                                         environments * observations * sizeof(float),
                                         cudaMemcpyDeviceToDevice, stream));
        actor.forward(data.actor_input.get(), 2 * environments, stream);
        critic.forward(obs, environments, stream);
        rl::sample_ar_gaussian(actor.output(), data.sigma.get(), previous_action, history, environments,
                               actions, correlation, seed + 2,
                               static_cast<uint32_t>(static_cast<uint64_t>(update) * horizon + t),
                               data.action.get() + offset * actions, data.logp.get() + offset, stream);
        CUDA_CHECK_THROW(cudaMemcpyAsync(data.value.get() + offset, critic.output(),
                                         environments * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        gpu::advance<<<1, 32, 0, stream>>>(
            data.worlds.get(), data.tasks.get(), data.action.get() + offset * actions,
            data.reward.get() + offset, data.done.get() + offset, data.counts.get(), environments,
            data.fault.get(), actor.output(), training_episode, data.cost_trace.get() + offset * 5);
        CUDA_CHECK_THROW(cudaGetLastError());
    }
}
void verify_long_episode_batch(const fs::path &root, int update, Rollout &data, Batches &batch,
                               const rl::Permutation &permutation, cudaStream_t stream) {
    const int late_start = horizon == 1280 ? 640 : 160;
    const auto rewards = download(data.reward.get(), rows, stream),
               values = download(data.value.get(), rows, stream),
               dones = download(data.done.get(), rows, stream),
               advantages = download(data.advantage.get(), rows, stream),
               returns = download(data.returns.get(), rows, stream),
               costs = download(data.cost_trace.get(), rows * 5, stream),
               gathered_adv = download(batch.advantage.get(), minibatch, stream),
               gathered_returns = download(batch.returns.get(), minibatch, stream),
               gradients = download(batch.mean_gradient.get(), 2 * minibatch * actions, stream);
    const auto tasks = download(data.tasks.get(), environments, stream);
    const auto worlds = download(data.worlds.get(), environments, stream);
    require((horizon == 640 || horizon == 1280) && training_episode.task_config().maximum_controls == horizon,
            "Runner/task horizon mismatch");
    double maximum_return_error = 0, maximum_reward_error = 0, absolute_late_rewards = 0, late_reward_sum = 0;
    std::vector<double> independent(rows), without_late(rows);
    for (int e = 0; e < environments; ++e) {
        require(tasks[e].controls == horizon &&
                    worlds[e].physical.control_steps == static_cast<uint64_t>(horizon) &&
                    worlds[e].physical.substeps == static_cast<uint64_t>(horizon * 10) &&
                    fabs(worlds[e].physical.time_seconds - horizon * .05) < 1e-5,
                "Actual simulator episode did not reach declared horizon");
        double target = 0, masked = 0;
        for (int t = horizon - 1; t >= 0; --t) {
            const int row = t * environments + e;
            require(dones[row] == (t == horizon - 1 ? 1.f : 0.f),
                    "Early terminal/reset inside declared episode");
            target = rewards[row] + double(.99f) * target;
            masked = (t < late_start ? rewards[row] : 0) + double(.99f) * masked;
            independent[row] = target;
            without_late[row] = masked;
            const double error = fabs(returns[row] - target);
            maximum_return_error = std::max(maximum_return_error, error);
            require(error <= .002 + .0002 * fabs(target) &&
                        fabs(advantages[row] + values[row] - target) <= .002 + .0002 * fabs(target),
                    "Actual rewards did not propagate through GAE/returns");
            float sum = 0;
            for (int c = 0; c < 5; ++c)
                sum += costs[row * 5 + c];
            const double reward_error = fabs(rewards[row] + p::dt * sum);
            maximum_reward_error = std::max(maximum_reward_error, reward_error);
            require(reward_error <= 1e-6 + 1e-5 * fabs(rewards[row]),
                    "Stored reward does not match actual cost components");
            if (t >= late_start) {
                absolute_late_rewards += fabs(rewards[row]);
                late_reward_sum += rewards[row];
            }
        }
    }
    int late_rows = 0, late_rows_with_gradient = 0;
    for (int i = 0; i < minibatch; ++i) {
        const int row = static_cast<int>(rl::permute(permutation, i));
        require(gathered_adv[i] == advantages[row] && gathered_returns[i] == returns[row],
                "Optimizer minibatch dropped/replaced actual returns");
        if (row / environments >= late_start) {
            ++late_rows;
            float magnitude = 0;
            for (int j = 0; j < actions; ++j)
                magnitude +=
                    fabsf(gradients[i * actions + j]) + fabsf(gradients[(minibatch + i) * actions + j]);
            if (magnitude > 1e-12f)
                ++late_rows_with_gradient;
        }
    }
    require(absolute_late_rewards > 0 && late_rows > 0 && late_rows_with_gradient > 0,
            "No late reward/optimizer signal");
    std::ofstream trace(root / "duration-reward-rows.csv");
    trace << std::setprecision(12)
          << "row,world,control,end_s,reward,done,value,advantage,GPU_return,independent_return,return_"
             "without_post8_rewards,speed_cost,lane_cost,depth_cost,posture_cost,effort_cost\n";
    for (int row = 0; row < rows; ++row) {
        trace << row << ',' << row % environments << ',' << row / environments + 1 << ','
              << (row / environments + 1) * .05 << ',' << rewards[row] << ',' << dones[row] << ','
              << values[row] << ',' << advantages[row] << ',' << returns[row] << ',' << independent[row]
              << ',' << without_late[row];
        for (int c = 0; c < 5; ++c)
            trace << ',' << costs[row * 5 + c];
        trace << '\n';
    }
    trace.close();
    std::ofstream out(root / "duration-reward-verification.json");
    out << std::setprecision(17)
        << "{\"passed\":true,\"before_first_optimizer_step\":true,\"global_update\":" << update
        << ",\"runner_controls\":" << horizon
        << ",\"GPU_task_controls\":" << training_episode.task_config().maximum_controls
        << ",\"actual_worlds\":32,\"actual_seconds_each\":" << horizon * .05
        << ",\"actual_substeps_each\":" << horizon * 10 << ",\"terminal_only_at_horizon\":true"
        << ",\"late_cutoff_s\":" << late_start * .05
        << ",\"late_reward_rows\":" << (horizon - late_start) * environments
        << ",\"sum_late_rewards\":" << late_reward_sum
        << ",\"max_reward_component_error\":" << maximum_reward_error
        << ",\"max_GAE_return_error\":" << maximum_return_error
        << ",\"first_optimizer_batch_late_rows\":" << late_rows
        << ",\"late_batch_rows_with_nonzero_policy_gradient\":" << late_rows_with_gradient
        << ",\"first_late_world0_return\":" << returns[late_start * environments]
        << ",\"same_return_without_late_rewards\":" << without_late[late_start * environments]
        << ",\"Adam_steps_before\":" << initial_optimizer(true).steps << ",\"unix_seconds\":"
        << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()
        << "}\n";
    out.close();
    std::cout << "EPISODE_REWARD_VERIFIED before_optimizer update=" << update << " seconds=" << horizon * .05
              << " late_rows=" << (horizon - late_start) * environments
              << " minibatch_late_rows=" << late_rows << " gradient_rows=" << late_rows_with_gradient
              << std::endl;
}
std::vector<float> score_batch(nn::GpuMlp &actor, Rollout &data, Batches &batch, cudaStream_t stream);
void rollout_diagnostics(const fs::path &root, int update, nn::GpuMlp &actor, nn::GpuMlp &critic,
                         Rollout &data, Batches &batch, cudaStream_t stream, const double *pooled,
                         int optimizer_rows, int actor_steps, int critic_steps);
double optimize_rollout(int update, nn::GpuMlp &actor, nn::GpuMlp &critic, nn::GpuAdam &actor_opt,
                        nn::GpuAdam &critic_opt, Rollout &data, Batches &batch, cudaStream_t stream,
                        const fs::path &diagnostics = {}) {
    rl::compute_gae(horizon, environments, data.reward.get(), data.value.get(), data.done.get(),
                    data.zeros.get(), data.zeros.get(), data.last.get(), .99f, 1.f, data.advantage.get(),
                    data.returns.get(), stream);
    rl::mean_std(data.advantage.get(), rows, data.adv_stats.get(), stream);
    const int actor_before = actor_opt.steps(), critic_before = critic_opt.steps();
    double pooled[5]{};
    int optimizer_rows = 0;
    if (actor_lr_override && update == starting_update()) {
        const auto rescore = score_batch(actor, data, batch, stream);
        const auto saved = download(data.logp.get(), rows, stream);
        float maximum = 0;
        for (int i = 0; i < rows; ++i)
            maximum = fmaxf(maximum, fabsf(rescore[i] - saved[i]));
        require(maximum < 1e-4f, "Old rollout policy rescore mismatch before optimizer");
        std::ofstream proof(diagnostics / "old-policy-rescore.json");
        proof << std::setprecision(12) << "{\"rows\":" << rows << ",\"max_abs_logp_error\":" << maximum
              << ",\"same_recorded_latents_and_AR_history\":true,\"before_first_optimizer\":true}\n";
    }
    double last_kl = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        CUDA_CHECK_THROW(cudaMemsetAsync(batch.stats.get(), 0, batch.stats.size_bytes(), stream));
        const auto permutation =
            rl::make_permutation(rows, seed + static_cast<uint32_t>(update * epochs + epoch));
        for (int first = 0; first < rows; first += minibatch) {
            wall_guard();
            rl::gather_rows(permutation, first, minibatch, observations, data.obs.get(), batch.obs.get(),
                            stream);
            rl::gather_rows(permutation, first, minibatch, observations, data.previous_obs.get(),
                            batch.obs.get() + minibatch * observations, stream);
            rl::gather_rows(permutation, first, minibatch, actions, data.previous_action.get(),
                            batch.previous_action.get(), stream);
            rl::gather_rows(permutation, first, minibatch, 1, data.history.get(), batch.history.get(),
                            stream);
            rl::gather_rows(permutation, first, minibatch, actions, data.action.get(), batch.action.get(),
                            stream);
            rl::gather_rows(permutation, first, minibatch, 1, data.logp.get(), batch.logp.get(), stream);
            rl::gather_rows(permutation, first, minibatch, 1, data.advantage.get(), batch.advantage.get(),
                            stream);
            rl::gather_rows(permutation, first, minibatch, 1, data.returns.get(), batch.returns.get(),
                            stream);
            actor.forward(batch.obs.get(), 2 * minibatch, stream);
            critic.forward(batch.obs.get(), minibatch, stream);
            rl::ar_gaussian_ppo_loss_grad(
                actor.output(), data.sigma.get(), batch.previous_action.get(), batch.history.get(),
                critic.output(), minibatch, actions, correlation, batch.action.get(), batch.logp.get(),
                batch.advantage.get(), batch.returns.get(), data.adv_stats.get(), {.2f, 0, .5f},
                batch.mean_gradient.get(), batch.value_gradient.get(), batch.row_stats.get(),
                batch.stats.get(), stream);
            if (horizon >= 640 && update == starting_update() && epoch == 0 && first == 0 &&
                !diagnostics.empty())
                verify_long_episode_batch(diagnostics, update, data, batch, permutation, stream);
            actor.zero_grad(stream);
            critic.zero_grad(stream);
            actor.backward(batch.mean_gradient.get(), 2 * minibatch, nullptr, stream);
            critic.backward(batch.value_gradient.get(), minibatch, nullptr, stream);
            actor_opt.step(actor.params(), actor.grads(), stream);
            critic_opt.step(critic.params(), critic.grads(), stream);
        }

        const auto metrics = download(batch.stats.get(), 5, stream);
        finite(metrics);
        for (int i = 0; i < 5; ++i)
            pooled[i] += metrics[i];
        optimizer_rows += rows;
        last_kl = metrics[3] / rows;
        if (last_kl > .01)
            break;
    }
    gpu::finite_parameters<<<(actor.num_params() + 255) / 256, 256, 0, stream>>>(
        actor.params(), actor.num_params(), data.fault.get());
    gpu::finite_parameters<<<(critic.num_params() + 255) / 256, 256, 0, stream>>>(
        critic.params(), critic.num_params(), data.fault.get());
    CUDA_CHECK_THROW(cudaGetLastError());
    if (actor_lr_override && !diagnostics.empty())
        rollout_diagnostics(diagnostics, update + 1, actor, critic, data, batch, stream, pooled,
                            optimizer_rows, actor_opt.steps() - actor_before,
                            critic_opt.steps() - critic_before);

    return last_kl;
}
__global__ void score_recorded_rows(const float *means, const float *sigma, const float *previous,
                                    const float *history, const float *action, int n, float rho,
                                    float *scores) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        scores[i] = rl::ar_logp(means + i * actions, means + (n + i) * actions, previous + i * actions, sigma,
                                actions, rho, history[i] != 0, action + i * actions);
}
std::vector<float> score_batch(nn::GpuMlp &actor, Rollout &data, Batches &batch, cudaStream_t stream) {
    device_ptr<float> scores(rows);
    const auto permutation = rl::make_permutation(rows, seed + 73);
    for (int first = 0; first < rows; first += minibatch) {
        rl::gather_rows(permutation, first, minibatch, observations, data.obs.get(), batch.obs.get(), stream);
        rl::gather_rows(permutation, first, minibatch, observations, data.previous_obs.get(),
                        batch.obs.get() + minibatch * observations, stream);
        rl::gather_rows(permutation, first, minibatch, actions, data.previous_action.get(),
                        batch.previous_action.get(), stream);
        rl::gather_rows(permutation, first, minibatch, actions, data.action.get(), batch.action.get(),
                        stream);
        rl::gather_rows(permutation, first, minibatch, 1, data.history.get(), batch.history.get(), stream);
        actor.forward(batch.obs.get(), 2 * minibatch, stream);
        score_recorded_rows<<<1, minibatch, 0, stream>>>(
            actor.output(), data.sigma.get(), batch.previous_action.get(), batch.history.get(),
            batch.action.get(), minibatch, correlation, scores.get() + first);
        CUDA_CHECK_THROW(cudaGetLastError());
    }
    return rl::diagnostic_original_order(download(scores.get(), rows, stream), permutation);
}
void rollout_diagnostics(const fs::path &root, int update, nn::GpuMlp &actor, nn::GpuMlp &critic,
                         Rollout &data, Batches &batch, cudaStream_t stream, const double *pooled,
                         int optimizer_rows, int actor_steps, int critic_steps) {
    const auto oldp = download(data.logp.get(), rows, stream), newp = score_batch(actor, data, batch, stream);
    const auto returns = download(data.returns.get(), rows, stream),
               advantages = download(data.advantage.get(), rows, stream),
               values = download(data.value.get(), rows, stream),
               adv_stats = download(data.adv_stats.get(), 2, stream);
    critic.forward(data.obs.get(), rows, stream);
    const auto after = download(critic.output(), rows, stream);
    const auto before_value = rl::diagnostic_value(values, returns),
               after_value = rl::diagnostic_value(after, returns);
    const auto ratio = rl::diagnostic_ratios(oldp, newp, .2);
    std::vector<float> normalized(rows);
    for (int i = 0; i < rows; ++i)
        normalized[i] = (advantages[i] - adv_stats[0]) / (adv_stats[1] + 1e-8f);
    std::ofstream out(root / "ppo-diagnostics.jsonl", std::ios::app);
    out << std::setprecision(12) << "{\"update\":" << update << ",\"rows\":" << rows
        << ",\"actor_steps\":" << actor_steps << ",\"critic_steps\":" << critic_steps
        << ",\"pooled_during_epoch_KL\":" << pooled[3] / optimizer_rows
        << ",\"pooled_clip_fraction\":" << pooled[4] / optimizer_rows
        << ",\"pooled_value_half_MSE\":" << pooled[1] / optimizer_rows
        << ",\"latent_conditional_entropy_nats\":" << pooled[2] / optimizer_rows
        << ",\"final_same_batch_KL_k1\":" << ratio.k1 << ",\"final_same_batch_KL_k3\":" << ratio.k3
        << ",\"final_same_batch_clip_fraction\":" << ratio.clipped_fraction
        << ",\"critic_before_half_MSE\":" << before_value.half_mse
        << ",\"critic_after_half_MSE\":" << after_value.half_mse
        << ",\"critic_before_EV\":" << before_value.explained_variance
        << ",\"critic_after_EV\":" << after_value.explained_variance;
    auto moments = [&](const char *name, const std::vector<float> &v) {
        const auto m = rl::diagnostic_moments(v);
        out << ",\"" << name << "\":{\"mean\":" << m.mean << ",\"std\":" << m.stddev()
            << ",\"min\":" << m.minimum << ",\"max\":" << m.maximum << '}';
    };
    moments("returns", returns);
    moments("raw_advantages", advantages);
    moments("normalized_advantages", normalized);
    out << ",\"counter_rng_unchanged\":true,\"AR_history_rescored\":true}\n";
}
std::vector<uint8_t> recorded_batch(Rollout &data, cudaStream_t stream) {
    cp::Writer out;
    out.value(uint32_t{1});
    out.value(rows);
    out.value(observations);
    out.value(actions);
    out.array(download(data.obs.get(), rows * observations, stream));
    out.array(download(data.previous_obs.get(), rows * observations, stream));
    out.array(download(data.previous_action.get(), rows * actions, stream));
    out.array(download(data.history.get(), rows, stream));
    out.array(download(data.action.get(), rows * actions, stream));
    out.array(download(data.logp.get(), rows, stream));
    out.array(download(data.value.get(), rows, stream));
    out.array(download(data.reward.get(), rows, stream));
    out.array(download(data.done.get(), rows, stream));
    return out.data;
}
struct Difference {
    double absolute = 0, relative = 0;
    bool bitwise = true;
};
Difference compare_vectors(const std::vector<float> &a, const std::vector<float> &b) {
    require(a.size() == b.size(), "Resume comparison shape mismatch");
    Difference d;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "Nonfinite resume comparison");
        const double error = fabs(double(a[i]) - b[i]);
        d.absolute = std::max(d.absolute, error);
        d.relative = std::max(d.relative, error / std::max({1e-12, fabs(double(a[i])), fabs(double(b[i]))}));
        require(error <= 1e-7 + 1e-6 * std::max(fabs(double(a[i])), fabs(double(b[i]))),
                "Resume equivalence tolerance exceeded");
    }
    d.bitwise = std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    return d;
}
int verify_resume(const fs::path &root) {
    require(verification_mode, "Verification protocol mode required");
    require(resume_state.has_value(), "Verification requires an explicit PPO parent copy and protocol");
    require(fs::create_directory(root / "run-started"), "Resume verification already attempted; no retry");
    active_wall_cap = 10;
    started = std::chrono::steady_clock::now();
    Budget progress;
    status(root, "verification_collecting", progress, starting_update());
    cp::write_atomic((root / "verification-model-protocol.bin").string(), protocol());
    Stream stream;
    Rollout data(stream.value);
    Batches batch;
    nn::GpuMlp actor(spec(actions), rows, seed), critic(spec(1), rows, seed + 1);
    nn::GpuAdam actor_opt(actor.num_params(), initial_optimizer_config(true)),
        critic_opt(critic.num_params(), initial_optimizer_config(false));
    restore_training(*resume_state, actor, critic, actor_opt, critic_opt);
    const auto uninterrupted = root / "uninterrupted", reloaded = root / "reloaded";
    fs::create_directories(uninterrupted);
    fs::create_directories(reloaded);
    const int update = starting_update();
    snapshot(uninterrupted, update, actor, critic, actor_opt, critic_opt);
    const auto saved = cp::read((uninterrupted / ("ppo-update-" + std::to_string(update) + ".bin")).string());
    const auto restored = rl::read_ppo_training_state(saved, cp::checksum(protocol()), seed);
    nn::GpuMlp actor_reload(spec(actions), rows, seed), critic_reload(spec(1), rows, seed + 1);
    nn::GpuAdam actor_opt_reload(actor_reload.num_params(), restored.actor_adam.config),
        critic_opt_reload(critic_reload.num_params(), restored.critic_adam.config);
    restore_training(restored, actor_reload, critic_reload, actor_opt_reload, critic_opt_reload);
    collect_rollout(update, actor, critic, data, stream.value);
    const auto counts = download(data.counts.get(), environments, stream.value);
    unsigned long long controls = 0, substeps = 0;
    for (const auto &count : counts) {
        controls += count.controls;
        substeps += count.substeps;
    }
    progress.controls = static_cast<int>(controls);
    progress.substeps = static_cast<int>(substeps);
    status(root, "verification_optimizing", progress, update);
    require(download(data.fault.get(), 1, stream.value)[0] == 0 && controls == rows && substeps == rows * 10,
            "Verification rollout failed; stop before optimizer");
    const auto batch_bytes = recorded_batch(data, stream.value);
    cp::write_atomic((root / "recorded-batch.bin").string(), batch_bytes);
    const auto before = score_batch(actor, data, batch, stream.value),
               before_reload = score_batch(actor_reload, data, batch, stream.value);
    const auto initial_score_difference = compare_vectors(before, before_reload);
    const double kl =
        optimize_rollout(update, actor, critic, actor_opt, critic_opt, data, batch, stream.value, root);
    require(download(data.fault.get(), 1, stream.value)[0] == 0, "Uninterrupted verification update failed");
    std::cout << "VERIFY_UPDATE_COMPLETED branch=uninterrupted global_update=" << update + 1
              << " actor_steps=" << actor_opt.steps() << std::endl;
    const double reload_kl = optimize_rollout(update, actor_reload, critic_reload, actor_opt_reload,
                                              critic_opt_reload, data, batch, stream.value);
    require(download(data.fault.get(), 1, stream.value)[0] == 0, "Reloaded verification update failed");
    std::cout << "VERIFY_UPDATE_COMPLETED branch=reloaded global_update=" << update + 1
              << " actor_steps=" << actor_opt_reload.steps() << std::endl;
    const auto after = score_batch(actor, data, batch, stream.value),
               after_reload = score_batch(actor_reload, data, batch, stream.value);
    const auto score_difference = compare_vectors(after, after_reload);
    snapshot(uninterrupted, update + 1, actor, critic, actor_opt, critic_opt);
    snapshot(reloaded, update + 1, actor_reload, critic_reload, actor_opt_reload, critic_opt_reload);
    const auto a_bytes =
        cp::read((uninterrupted / ("ppo-update-" + std::to_string(update + 1) + ".bin")).string());
    const auto b_bytes =
        cp::read((reloaded / ("ppo-update-" + std::to_string(update + 1) + ".bin")).string());
    const auto a = rl::read_ppo_training_state(a_bytes, cp::checksum(protocol()), seed),
               b = rl::read_ppo_training_state(b_bytes, cp::checksum(protocol()), seed);
    const std::vector<float> *av[] = {&a.actor.parameters,  &a.critic,
                                      &a.actor_adam.first,  &a.actor_adam.second,
                                      &a.critic_adam.first, &a.critic_adam.second,
                                      &a.actor.log_sigma};
    const std::vector<float> *bv[] = {&b.actor.parameters,  &b.critic,
                                      &b.actor_adam.first,  &b.actor_adam.second,
                                      &b.critic_adam.first, &b.critic_adam.second,
                                      &b.actor.log_sigma};
    const char *names[] = {"actor_weights", "critic_weights", "actor_Adam_m", "actor_Adam_v",
                           "critic_Adam_m", "critic_Adam_v",  "fixed_log_std"};
    require(a.actor.update == update + 1 && b.actor.update == update + 1 &&
                a.actor.actor_steps == b.actor.actor_steps && a.actor.critic_steps == b.actor.critic_steps,
            "Restored update/Adam counters differ");
    require(a.actor.actor_steps <= resume_state->actor.actor_steps + optimizer_steps_per_update() &&
                a.actor.critic_steps <= resume_state->actor.critic_steps + optimizer_steps_per_update(),
            "Verification optimizer budget exceeded");
    require(std::memcmp(&a.actor_adam.config, &b.actor_adam.config, sizeof(nn::AdamConfig)) == 0 &&
                std::memcmp(&a.critic_adam.config, &b.critic_adam.config, sizeof(nn::AdamConfig)) == 0,
            "Restored optimizer configs differ");
    require(recorded_batch(data, stream.value) == batch_bytes, "Training batch changed between branches");
    require(cp::read(parent_checkpoint_path.string()) == parent_checkpoint_payload, "PPO parent changed");
    Difference differences[7];
    for (int j = 0; j < 7; ++j)
        differences[j] = compare_vectors(*av[j], *bv[j]);
    std::ofstream out(root / "result.json");
    out << std::setprecision(17) << "{\"verification_only\":true,\"passed\":true,\"seed\":" << seed
        << ",\"starting_update\":" << update << ",\"ending_update_each\":" << update + 1
        << ",\"controls\":" << controls << ",\"substeps\":" << substeps
        << ",\"actor_steps_before\":" << resume_state->actor.actor_steps
        << ",\"actor_steps_after_each\":" << a.actor.actor_steps
        << ",\"critic_steps_before\":" << resume_state->actor.critic_steps
        << ",\"critic_steps_after_each\":" << a.actor.critic_steps
        << ",\"initial_logp_max_abs\":" << initial_score_difference.absolute
        << ",\"final_logp_max_abs\":" << score_difference.absolute
        << ",\"final_logp_max_relative\":" << score_difference.relative << ",\"KL_uninterrupted\":" << kl
        << ",\"KL_reloaded\":" << reload_kl
        << ",\"snapshots_byte_identical\":" << (a_bytes == b_bytes ? "true" : "false")
        << ",\"batch_unchanged\":true,\"parent_unchanged\":true,\"parameters\":[";
    for (int j = 0; j < 7; ++j) {
        const auto d = differences[j];
        out << (j ? "," : "") << "{\"name\":" << std::quoted(names[j]) << ",\"max_abs\":" << d.absolute
            << ",\"max_relative\":" << d.relative << ",\"bitwise_equal\":" << (d.bitwise ? "true" : "false")
            << '}';
    }
    out << "],\"native_seconds\":"
        << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
        << ",\"runtime_adopted\":false,\"retries\":0}\n";
    out.close();
    std::ifstream immediate(root / "result.json");
    std::cout << immediate.rdbuf() << std::flush;
    status(root, "done", progress, update + 1);
    return 0;
}
int run(const fs::path &root, bool continue_pre_training = false) {
    require(cp::read((root / "protocol.bin").string()) == protocol() &&
                cp::read((root / "initialization.bin").string()) == initialization(),
            "Frozen PPO identity changed");
    if (continue_pre_training) {
        require((depth_offsets || fs::exists(root / "run-started")) && !fs::exists(root / "training.csv"),
                "Training already started or first phase absent");
        if (depth_offsets) {
            std::ifstream evidence_file(root / "baseline-result.json");
            const std::string evidence((std::istreambuf_iterator<char>(evidence_file)), {});
            require(evidence.find("\"additional_updates\":0,") != std::string::npos &&
                        evidence.find("\"controls\":2624,") != std::string::npos &&
                        evidence.find("\"physical_substeps\":26240,") != std::string::npos &&
                        evidence.find("\"fault\":\"Device reset rebased fixed depth target\"") !=
                            std::string::npos &&
                        (actor_lr_override
                             ? cp::read((root / "baseline-parent.bin").string()) == parent_checkpoint_payload
                             : cp::read((root / "baseline-protocol.bin").string()) == protocol()),
                    "Recovery continuation lacks compatible zero-update baseline evidence");
            require(fs::create_directory(root / "run-started"), "Recovery continuation already attempted");
        }
        require(fs::create_directory(root / "training-started"), "Training continuation already attempted");
    } else
        require(fs::create_directory(root / "run-started"), "Run already attempted; no retries");
    started = std::chrono::steady_clock::now();
    Budget budget;
    budget.seconds = active_wall_cap;
    Evaluation before, after;
    bool initial_recovery_pass = false, final_recovery_pass = false;
    const bool extended = horizon == 1280;
    bool final64_recovery_pass = false, qualified_stop = false, long_regression_pass = false;
    StoppingResult initial_stopping, final_stopping;
    int long_attempts = 0;
    auto recovery_evaluation = [&](const fs::path &directory, const fs::path &checkpoint,
                                   const fs::path &source_protocol, int expected_update, bool &passed,
                                   int duration_controls = 640) {
        require(fs::create_directory(directory), "Recovery evaluation identity already exists");
        fs::copy_file(checkpoint, directory / "checkpoint.bin");
        fs::copy_file(source_protocol, directory / "protocol.bin");
        require(fixed_depth_replay(directory, false, &budget, &passed, expected_update, duration_controls) ==
                    0,
                "Recovery evaluation physical/execution stop; no continuation");
    };
    if (continue_pre_training) {
        std::ifstream spent(root / "pre-training-continuation.txt");
        double elapsed = 0;
        spent >> budget.controls >> budget.substeps >> elapsed >> before.passed >> before.speed >>
            before.rmse >> before.work_per_m;
        const int expected_spent = depth_offsets ? 2624 : 224;
        require(spent && budget.controls == expected_spent && budget.substeps == expected_spent * 10 &&
                    elapsed >= 0 && elapsed < 5 && before.passed == 4,
                "Invalid pretraining continuation evidence");
        require(starting_update() == 288 && horizon == 640 && updates == (depth_offsets ? 32 : 16),
                "Wrong continuation protocol");
        started -= std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(elapsed));
    }
    Sustained long_result;
    const int base_update = starting_update();
    const int base_actor_steps = initial_optimizer(true).steps,
              base_critic_steps = initial_optimizer(false).steps;
    int completed = base_update, actor_steps = base_actor_steps, critic_steps = base_critic_steps;
    double rollout_ms = 0, update_ms = 0, training_seconds = 0;
    std::string fault;
    try {
        Stream stream;
        if (!continue_pre_training) {
            parity(budget, stream.value);
            before = stopping ? reused_baseline(root) : evaluate(initial_actor(), root, "initial", budget);
            if (stopping)
                initial_stopping = evaluate_stopping(initial_actor(), root / "initial-stop", budget);
            if (depth_offsets && !stopping)
                recovery_evaluation(root / "initial-recovery", parent_checkpoint_path, parent_protocol_path,
                                    base_update, initial_recovery_pass);
            if (extended && !stopping) {
                bool baseline64 = false;
                recovery_evaluation(root / "initial-recovery64", parent_checkpoint_path, parent_protocol_path,
                                    base_update, baseline64, 1280);
            }
        }
        nn::GpuMlp actor(spec(actions), rows, seed), critic(spec(1), rows, seed + 1);
        actor.set_params(initial_actor());
        critic.set_params(initial_critic());
        nn::GpuAdam actor_opt(actor.num_params(), initial_optimizer_config(true)),
            critic_opt(critic.num_params(), initial_optimizer_config(false));
        if (resume_state)
            restore_training(*resume_state, actor, critic, actor_opt, critic_opt);
        if (actor_lr_override) {
            require(resume_state.has_value(), "Matched rate trial requires restored state");
            auto expected_actor = resume_state->actor_adam;
            expected_actor.config.lr = *actor_lr_override;
            cp::Writer actual_actor, actual_critic;
            actor_opt.save(actual_actor);
            critic_opt.save(actual_critic);
            require(actual_actor.data == rl::adam_payload(expected_actor) &&
                        actual_critic.data == rl::adam_payload(resume_state->critic_adam) &&
                        actor.get_params() == resume_state->actor.parameters &&
                        critic.get_params() == resume_state->critic,
                    "Rate override changed restored parameters or optimizer state beyond actorLR");
            std::ofstream proof(root / "effective-restoration.json");
            proof << std::setprecision(12) << "{\"actor_lr_after_restore\":" << actor_opt.config().lr
                  << ",\"critic_lr_after_restore\":" << critic_opt.config().lr
                  << ",\"actor_steps\":" << actor_opt.steps() << ",\"critic_steps\":" << critic_opt.steps()
                  << ",\"actor_and_critic_parameters_exact\":true,\"both_Adam_moments_exact\":true,"
                     "\"critic_config_exact\":true,\"only_actor_lr_changed\":true}\n";
        }
        require(actor_opt.steps() == base_actor_steps && critic_opt.steps() == base_critic_steps,
                "Initial optimizer counters differ");
        snapshot(root, base_update, actor, critic, actor_opt, critic_opt);
        Rollout data(stream.value);
        Batches batch;
        Timer timer;
        unsigned long long previous_controls = 0, previous_substeps = 0;
        std::ofstream log(root / "training.csv");
        std::ofstream episodes(root / "exploratory-episodes.csv");
        episodes << "update,world,pose,forward_mm,latent_RMS0,latent_RMS1,latent_RMS2,latent_RMS3,decoded_"
                    "noise_RMS0,decoded_noise_RMS1,decoded_noise_RMS2,decoded_noise_RMS3,filtered_RMS0,"
                    "filtered_RMS1,filtered_RMS2,filtered_RMS3\n";
        log << "update,controls,actor_Adam_steps,critic_Adam_steps,mean_KL,rollout_GPU_ms,update_GPU_ms,"
               "cumulative_speed_cost,cumulative_lane_cost,cumulative_depth_cost,cumulative_posture_cost,"
               "cumulative_effort_cost,command_RMS0,command_RMS1,command_RMS2,command_RMS3,joint_min0,joint_"
               "max0,joint_min1,joint_max1,joint_min2,joint_max2,joint_min3,joint_max3,maximum_tilt_deg,"
               "saturated_axis_substeps,cumulative_horizontal_cost,cumulative_recovery_cost,"
               "cumulative_roll_cost,cumulative_pitch_cost\n";
        const auto training_started = std::chrono::steady_clock::now();
        training_started_unix_seconds =
            std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        status(root, "training", budget, base_update);
        std::cout << "TRAINING_STARTED unix_seconds=" << std::setprecision(17)
                  << training_started_unix_seconds
                  << (resume_state ? " PPO restored actor/critic/Adam; fresh episodes"
                                   : " PPO fresh actor/critic/Adam")
                  << " starting_update=" << base_update << "; model6 worlds and rollouts device-resident"
                  << std::endl;
        for (int update = base_update; update < base_update + updates; ++update) {
            wall_guard();
            status(root, "training", budget, completed);
            timer.begin(stream.value);
            collect_rollout(update, actor, critic, data, stream.value, root);
            const double elapsed_rollout = timer.end(stream.value);
            rollout_ms += elapsed_rollout;
            const auto counts = download(data.counts.get(), environments, stream.value);
            unsigned long long controls = 0, substeps = 0;
            for (const auto &c : counts) {
                controls += c.controls;
                substeps += c.substeps;
            }
            budget.controls += static_cast<int>(controls - previous_controls);
            budget.substeps += static_cast<int>(substeps - previous_substeps);
            previous_controls = controls;
            previous_substeps = substeps;
            const int rollout_fault = download(data.fault.get(), 1, stream.value)[0];
            if (rollout_fault) {
                if (completed % 8 != 0)
                    snapshot(root, completed, actor, critic, actor_opt, critic_opt);
                const auto failed_worlds = download(data.worlds.get(), environments, stream.value);
                const auto failed_tasks = download(data.tasks.get(), environments, stream.value);
                const auto partial_rewards = download(data.reward.get(), rows, stream.value);
                const auto partial_dones = download(data.done.get(), rows, stream.value);
                const auto partial_costs = download(data.cost_trace.get(), rows * 5, stream.value);
                std::ofstream partial(root / "gpu-rollout-partial-rewards.csv");
                partial << std::setprecision(12)
                        << "world,control,end_s,reward,done,speed_cost,lane_cost,depth_cost,posture_cost,"
                           "effort_cost\n";
                for (int e = 0; e < environments; ++e)
                    for (int t = 0; t < failed_tasks[e].controls; ++t) {
                        const int row = t * environments + e;
                        partial << e << ',' << t + 1 << ',' << (t + 1) * .05 << ',' << partial_rewards[row]
                                << ',' << partial_dones[row];
                        for (int c = 0; c < 5; ++c)
                            partial << ',' << partial_costs[row * 5 + c];
                        partial << '\n';
                    }
                std::ofstream diagnostic(root / "gpu-rollout-failure.json");
                diagnostic << std::setprecision(17) << "{\"global_update\":" << update
                           << ",\"kernel_fault\":" << rollout_fault << ",\"episode_controls\":" << horizon
                           << ",\"worlds\":[";
                for (int i = 0; i < environments; ++i) {
                    const auto &w = failed_worlds[i];
                    diagnostic << (i ? "," : "") << "{\"world\":" << i
                               << ",\"physical_code\":" << w.failure_code
                               << ",\"accepted_controls\":" << w.physical.control_steps
                               << ",\"task_controls\":" << failed_tasks[i].controls
                               << ",\"attempted_substeps\":" << w.attempted_substeps
                               << ",\"attempted_time_s\":" << w.attempted_substeps * .005
                               << ",\"clearance_m\":"
                               << cudarl::movement::clearance(w.physical.body, true, w.arena)
                               << ",\"state\":";
                    state_json(diagnostic, w.physical.body);
                    diagnostic << ",\"before_failed_substep\":";
                    state_json(diagnostic, counts[i].last_before);
                    diagnostic << ",\"rejected_candidate\":";
                    state_json(diagnostic, counts[i].rejected_candidate);
                    diagnostic << ",\"rejected_minimum_gap_m\":" << counts[i].rejected_gap_m;
                    diagnostic << '}';
                }
                diagnostic << "]}\n";
            }
            require(rollout_fault == 0, "Device rollout physical/observation fault; stop before PPO");
            require(controls == static_cast<unsigned long long>((update - base_update + 1) * rows) &&
                        substeps == controls * 10,
                    "GPU rollout accounting mismatch");
            for (int i = 0; i < environments; ++i) {
                episodes << update + 1 << ',' << i << ',' << (i % 2 ? 3 : 0) << ','
                         << counts[i].episode_forward_mm;
                for (double x : counts[i].latent_square)
                    episodes << ',' << sqrt(x / horizon);
                for (double x : counts[i].decoded_noise_square)
                    episodes << ',' << sqrt(x / horizon);
                for (double x : counts[i].filtered_square)
                    episodes << ',' << sqrt(x / horizon);
                episodes << '\n';
            }
            episodes.flush();
            timer.begin(stream.value);
            const double last_kl = optimize_rollout(update, actor, critic, actor_opt, critic_opt, data, batch,
                                                    stream.value, root);
            actor_steps = actor_opt.steps();
            critic_steps = critic_opt.steps();
            const double elapsed_update = timer.end(stream.value);
            update_ms += elapsed_update;
            require(download(data.fault.get(), 1, stream.value)[0] == 0,
                    "Nonfinite optimized parameters; stop before further physics");
            completed = update + 1;
            if (completed == base_update + 1) {
                const double utc =
                    std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
                        .count();
                std::ofstream event(root / "first-update.json");
                event << std::setprecision(17)
                      << "{\"event\":\"first_update_completed\",\"unix_seconds\":" << utc
                      << ",\"seed\":" << seed << ",\"update\":" << completed
                      << ",\"additional_update\":1,\"actor_optimizer_steps\":" << actor_steps
                      << ",\"critic_optimizer_steps\":" << critic_steps << ",\"controls\":" << budget.controls
                      << ",\"physical_substeps\":" << budget.substeps << "}\n";
                event.flush();
                std::cout << "FIRST_UPDATE_COMPLETED unix_seconds=" << std::setprecision(17) << utc
                          << " controls=" << budget.controls << std::endl;
            }
            require(actor_steps <= base_actor_steps + updates * optimizer_steps_per_update() &&
                        critic_steps <= base_critic_steps + updates * optimizer_steps_per_update(),
                    "Optimizer cap exceeded");
            if (completed % 8 == 0)
                snapshot(root, completed, actor, critic, actor_opt, critic_opt);
            log << completed << ',' << controls << ',' << actor_steps << ',' << critic_steps << ',' << last_kl
                << ',' << elapsed_rollout << ',' << elapsed_update;
            double cost[5]{}, squares[4]{}, details[4]{};
            float low[4]{}, high[4]{}, tilt = 0;
            unsigned long long saturations = 0;
            for (const auto &count : counts) {
                for (int j = 0; j < 5; ++j)
                    cost[j] += count.cost_integral[j];
                for (int j = 0; j < 4; ++j) {
                    details[j] += count.reward_detail_integral[j];
                    squares[j] += count.command_square[j];
                    low[j] = fminf(low[j], count.joint_min[j]);
                    high[j] = fmaxf(high[j], count.joint_max[j]);
                }
                tilt = fmaxf(tilt, count.maximum_tilt_deg);
                saturations += count.saturated_axis_substeps;
            }
            for (double v : cost)
                log << ',' << v;
            for (double v : squares)
                log << ',' << sqrt(v / controls);
            for (int j = 0; j < 4; ++j)
                log << ',' << low[j] << ',' << high[j];
            log << ',' << tilt << ',' << saturations;
            for (double v : details)
                log << ',' << v;
            log << '\n';
            log.flush();
            if (actor_lr_override && completed - base_update != updates &&
                (completed - base_update == 8 || completed - base_update == 16)) {
                const auto directory = root / ("checkpoint-" + std::to_string(completed));
                require(fs::create_directory(directory), "Intermediate evaluation already exists");
                const auto actor_before_eval = actor.get_params(), critic_before_eval = critic.get_params();
                cp::Writer optimizer_before, optimizer_after;
                actor_opt.save(optimizer_before);
                critic_opt.save(optimizer_before);
                const auto short_result = evaluate(actor_before_eval, directory, "frozen", budget);
                bool recovery = false;
                if (!stopping)
                    recovery_evaluation(directory / "recovery",
                                        root / ("ppo-update-" + std::to_string(completed) + ".bin"),
                                        root / "protocol.bin", completed, recovery);
                bool recovery64 = false;
                if (extended)
                    recovery_evaluation(directory / "recovery64",
                                        root / ("ppo-update-" + std::to_string(completed) + ".bin"),
                                        root / "protocol.bin", completed, recovery64, 1280);
                StoppingResult stop_result;
                if (stopping)
                    stop_result = evaluate_stopping(actor_before_eval, directory / "stop", budget);
                actor_opt.save(optimizer_after);
                critic_opt.save(optimizer_after);
                require(actor.get_params() == actor_before_eval &&
                            critic.get_params() == critic_before_eval &&
                            optimizer_before.data == optimizer_after.data,
                        "Frozen evaluation changed training state");
                std::ofstream summary(directory / "summary.json");
                summary
                    << "{\"update\":" << completed << ",\"short_passes\":" << short_result.passed
                    << ",\"recovery_pass\":" << (recovery ? "true" : "false")
                    << ",\"recovery64_pass\":" << (recovery64 ? "true" : "false")
                    << ",\"stopping_passes\":" << stop_result.passed
                    << ",\"parameters_and_optimizers_unchanged\":true,\"sampling_counter_unchanged\":true}\n";
                std::cout << "INTERMEDIATE_EVALUATED update=" << completed
                          << " short_passes=" << short_result.passed << " recovery=" << recovery << std::endl;
                if (extended && short_result.passed == 4 && recovery64 &&
                    (stopping ? stop_result.passed == 2 : recovery)) {
                    require(fs::create_directory(directory / "long"), "Long evaluation identity exists");
                    ++long_attempts;
                    long_result = sustained(actor_before_eval, directory / "long", budget);
                    require(long_result.safety_fault.empty(), "64s campaign long physical safety stop");
                    long_regression_pass =
                        stopping ? preserved_long_windows(directory / "long") : long_result.passed == 2;
                    if (long_regression_pass) {
                        after = short_result;
                        final_recovery_pass = recovery;
                        final64_recovery_pass = recovery64;
                        final_stopping = stop_result;
                        qualified_stop = true;
                        break;
                    }
                }
            }
        }
        training_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - training_started).count();
        status(root, "frozen_evaluation", budget, completed);
        if (!qualified_stop)
            after = evaluate(actor.get_params(), root, "frozen", budget);
        if (depth_offsets && !qualified_stop && !stopping)
            recovery_evaluation(root / "final-recovery",
                                root / ("ppo-update-" + std::to_string(completed) + ".bin"),
                                root / "protocol.bin", completed, final_recovery_pass);
        if (extended && !qualified_stop)
            recovery_evaluation(root / "final-recovery64",
                                root / ("ppo-update-" + std::to_string(completed) + ".bin"),
                                root / "protocol.bin", completed, final64_recovery_pass, 1280);
        if (stopping && !qualified_stop)
            final_stopping = evaluate_stopping(actor.get_params(), root / "final-stop", budget);
        const bool run_long =
            after.passed == 4 &&
            (stopping ? final_stopping.passed == 2 : (!depth_offsets || final_recovery_pass)) &&
            (!extended || final64_recovery_pass);
        if (run_long && !qualified_stop) {
            status(root, "long_evaluation", budget, completed);
            ++long_attempts;
            long_result = sustained(actor.get_params(), root, budget);
            if (extended)
                require(long_result.safety_fault.empty(), "64s campaign long physical safety stop");
            long_regression_pass = stopping ? preserved_long_windows(root) : long_result.passed == 2;
        }
        const int expected_controls =
            stopping
                ? (completed - base_update) * rows + 64 + 2560 + ((completed - base_update >= 8) ? 7040 : 0) +
                      ((completed - base_update >= 16) ? 7040 : 0) + long_attempts * 2400
            : extended
                ? (completed - base_update) * rows + 64 + 640 + 1920 + 3840 +
                      ((completed - base_update >= 8) ? 6400 : 0) +
                      ((completed - base_update >= 16) ? 6400 : 0) + long_attempts * 2400
                : control_cap - (run_long ? 0 : 2400) - (continue_pre_training && !depth_offsets ? 480 : 0);
        require((long_result.safety_fault.empty() && budget.controls == expected_controls &&
                 budget.substeps == expected_controls * 10) ||
                    (!long_result.safety_fault.empty() && budget.controls <= control_cap &&
                     budget.substeps <= substep_cap),
                "Total run accounting mismatch");
    } catch (const std::exception &error) {
        fault = error.what();
    }
    const bool accepted =
        fault.empty() && after.passed == 4 &&
        (stopping ? final_stopping.passed == 2 && final64_recovery_pass && long_regression_pass
                  : after.rmse < before.rmse && long_result.passed == 2);
    std::ofstream out(root / "result.json");
    out << std::setprecision(9)
        << "{\"algorithm\":\"PPO\",\"reward_version\":" << training_episode.reward_version()
        << ",\"parent_reward_version\":"
        << (resume_state ? resume_state->actor.reward_version : training_episode.reward_version())
        << ",\"fresh_initialization\":" << (resume_state ? "false" : "true") << ",\"seed\":" << seed
        << ",\"starting_update\":" << base_update << ",\"additional_updates\":" << completed - base_update
        << ",\"rho\":" << correlation << ",\"fault\":" << std::quoted(fault)
        << ",\"completed_updates\":" << completed << ",\"actor_optimizer_steps\":" << actor_steps
        << ",\"critic_optimizer_steps\":" << critic_steps << ",\"controls\":" << budget.controls
        << ",\"physical_substeps\":" << budget.substeps << ",\"initial_passes\":" << before.passed
        << ",\"frozen_passes\":" << after.passed << ",\"sustained_passes\":" << long_result.passed
        << ",\"initial_recovery_pass\":" << (initial_recovery_pass ? "true" : "false")
        << ",\"final_recovery_pass\":" << (final_recovery_pass ? "true" : "false")
        << ",\"final64_recovery_pass\":" << (final64_recovery_pass ? "true" : "false")
        << ",\"stopping_lesson\":" << (stopping ? "true" : "false")
        << ",\"initial_stopping_passes\":" << initial_stopping.passed
        << ",\"final_stopping_passes\":" << final_stopping.passed
        << ",\"long_regression_preserved\":" << (long_regression_pass ? "true" : "false")
        << ",\"long_evaluation_attempts\":" << long_attempts
        << ",\"qualified_early_stop\":" << (qualified_stop ? "true" : "false")
        << ",\"sustained_windows_passed\":" << long_result.windows_passed
        << ",\"sustained_windows_total\":" << long_result.windows_total
        << ",\"long_evaluation_attempted\":" << (long_attempts ? "true" : "false")
        << ",\"long_completed_poses\":" << long_result.completed_poses
        << ",\"long_safety_code\":" << long_result.safety_code
        << ",\"long_safety_fault\":" << std::quoted(long_result.safety_fault)
        << ",\"initial_speed_mm_s\":" << before.speed * 1000
        << ",\"frozen_speed_mm_s\":" << after.speed * 1000 << ",\"initial_RMSE_mm_s\":" << before.rmse * 1000
        << ",\"frozen_RMSE_mm_s\":" << after.rmse * 1000 << ",\"frozen_work_J_per_m\":" << after.work_per_m
        << ",\"candidate_accepted\":" << (accepted ? "true" : "false") << ",\"rollout_GPU_ms\":" << rollout_ms
        << ",\"update_GPU_ms\":" << update_ms << ",\"training_seconds\":" << training_seconds
        << ",\"training_controls_per_second\":"
        << (training_seconds > 0 ? training_cap / training_seconds : 0)
        << ",\"training_started_unix_seconds\":" << std::setprecision(17) << training_started_unix_seconds
        << std::setprecision(9) << ",\"native_seconds\":"
        << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
        << ",\"runtime_adopted\":false,\"closed\":true,\"retries\":0}\n";
    out.close();
    status(root, fault.empty() ? "done" : "failed", budget, completed);
    std::ifstream immediate(root / "result.json");
    std::cout << immediate.rdbuf() << std::flush;
    return fault.empty() ? 0 : 3;
}

} // namespace
int main(int argc, char **argv) {
    try {
        verification_mode = argc > 1 && (std::string(argv[1]) == "--prepare-verify" ||
                                         std::string(argv[1]) == "--verify-resume");
        require(argc >= 3 && argc <= 21 && argc % 2 == 1,
                "Use --prepare/--run/--verify-resume DIR [--seed UINT32] [--rho 0|0.8] [--resume-from "
                "CHECKPOINT --resume-protocol PROTOCOL]");
        fs::path resume_checkpoint, resume_protocol;
        bool seed_explicit = false, rho_explicit = false;
        for (int option = 3; option < argc; option += 2) {
            if (std::string(argv[option]) == "--stopping") {
                require(std::string(argv[option + 1]) == "0" || std::string(argv[option + 1]) == "1",
                        "Invalid stopping flag");
                stopping = std::string(argv[option + 1]) == "1";
                continue;
            }
            if (std::string(argv[option]) == "--actor-lr") {
                size_t used = 0;
                const float rate = std::stof(argv[option + 1], &used);
                require(used == std::string(argv[option + 1]).size() && rate == .00025f,
                        "Unsupported matched actor rate");
                actor_lr_override = rate;
                continue;
            }
            if (std::string(argv[option]) == "--depth-offsets") {
                require(std::string(argv[option + 1]) == "0" || std::string(argv[option + 1]) == "1",
                        "Invalid fixed-depth start mix");
                depth_offsets = std::string(argv[option + 1]) == "1";
                continue;
            }
            if (std::string(argv[option]) == "--episode-controls") {
                size_t used = 0;
                horizon = std::stoi(argv[option + 1], &used);
                require(used == std::string(argv[option + 1]).size() && training_episode.valid(),
                        "Unsupported episode horizon");
                continue;
            }
            if (std::string(argv[option]) == "--updates") {
                size_t used = 0;
                updates = std::stoi(argv[option + 1], &used);
                require(used == std::string(argv[option + 1]).size() && updates >= 1 && updates <= 128,
                        "Invalid additional update cap");
                continue;
            }
            if (std::string(argv[option]) == "--tilt-mixture") {
                require(std::string(argv[option + 1]) == "0" || std::string(argv[option + 1]) == "1",
                        "Invalid tilt mixture");
                tilt_mixture = std::string(argv[option + 1]) == "1";
                continue;
            }
            if (std::string(argv[option]) == "--rho") {
                rho_explicit = true;
                size_t used = 0;
                correlation = std::stof(argv[option + 1], &used);
                require(used == std::string(argv[option + 1]).size() &&
                            (correlation == 0 || correlation == .8f),
                        "Invalid preregistered rho");
                continue;
            }
            if (std::string(argv[option]) == "--resume-from") {
                resume_checkpoint = argv[option + 1];
                continue;
            }
            if (std::string(argv[option]) == "--resume-protocol") {
                resume_protocol = argv[option + 1];
                continue;
            }
            require(std::string(argv[option]) == "--seed", "Expected --seed or --rho");
            seed_explicit = true;
            size_t consumed = 0;
            const auto value = std::stoull(argv[option + 1], &consumed);
            require(consumed == std::string(argv[option + 1]).size() &&
                        value <= std::numeric_limits<uint32_t>::max(),
                    "Invalid seed");
            seed = static_cast<uint32_t>(value);
        }
        rows = environments * horizon;
        training_cap = updates * rows;
        control_cap = training_cap + 64 + 1280 + 2400;
        if (depth_offsets)
            control_cap += 3840;
        if (actor_lr_override)
            control_cap += 5120; // Two additional short+recovery checkpoint assessments.
        if (horizon == 1280)
            control_cap = training_cap + 64 + 640 + 1920 + 3840 + 2 * 6400 + 2 * 2400;
        if (stopping)
            control_cap = training_cap + 64 + 2560 + 2 * 7040 + 2 * 2400;
        substep_cap = control_cap * 10;
        if (tilt_mixture || updates <= 32)
            active_wall_cap = 20;
        if (horizon >= 640)
            active_wall_cap = 30;
        if (depth_offsets) {
            require(!tilt_mixture &&
                        ((horizon == 640 && updates == 32) || (horizon == 1280 && updates == 16)),
                    "Unsupported fixed-depth recovery horizon/update budget");
            active_wall_cap = 60;
        }
        const auto root = fs::absolute(argv[2]);
        if ((std::string(argv[1]) == "--run" || std::string(argv[1]) == "--verify-resume" ||
             std::string(argv[1]) == "--continue-pre-training") &&
            resume_checkpoint.empty() && fs::exists(root / "parent-checkpoint.bin")) {
            resume_checkpoint = root / "parent-checkpoint.bin";
            resume_protocol = root / "parent-protocol.bin";
        }
        require(resume_checkpoint.empty() == resume_protocol.empty(),
                "Both resume checkpoint and protocol are required");
        if (!resume_checkpoint.empty())
            load_resume_source(resume_checkpoint, resume_protocol, seed_explicit, rho_explicit);
        if (depth_offsets)
            require(resume_state && seed == 1247037243 && correlation == .8f &&
                        ((horizon == 640 && starting_update() == 288) ||
                         (horizon == 1280 && starting_update() == (stopping ? 336 : 320) &&
                          resume_state->actor.observation_schema == 2 &&
                          resume_state->actor.reward_version == 5)),
                    "Recovery lesson parent/schema mismatch");
        if (stopping)
            require(horizon == 1280 && updates == 16 && depth_offsets && actor_lr_override && resume_state &&
                        starting_update() == 336 && resume_state->actor.reward_version == 5,
                    "Stopping lesson requires copied336,16updates,64s,fixed-targetmix and matched rate");
        require(!verification_mode || resume_state.has_value(), "Resume verification requires a PPO parent");
        if (std::string(argv[1]) == "--fixed-depth-prepare" || std::string(argv[1]) == "--fixed-depth-replay")
            return fixed_depth_replay(root, std::string(argv[1]) == "--fixed-depth-prepare");
        if (std::string(argv[1]) == "--sensitivity" || std::string(argv[1]) == "--sensitivity-continue")
            return depth_sensitivity(root, std::string(argv[1]) == "--sensitivity-continue");
        if (std::string(argv[1]) == "--verify-path-scale")
            return verify_path_scale(root);
        if (std::string(argv[1]) == "--frozen-sustained")
            return frozen_sustained(root);
        if (std::string(argv[1]) == "--verify-resume")
            return verify_resume(root);
        if (std::string(argv[1]) == "--neutral-pitch" || std::string(argv[1]) == "--neutral-pitch-continue")
            return neutral_pitch_replay(root, std::string(argv[1]) == "--neutral-pitch-continue");
        if (std::string(argv[1]) == "--diagnose-safety")
            return diagnose_safety(root);
        if (std::string(argv[1]) == "--evaluate") {
            const auto checkpoint_path = root / "ppo-update-128.bin";
            const auto original = cp::read(checkpoint_path.string());
            const auto actor =
                rl::read_ppo_actor(original, cp::checksum(cp::read((root / "protocol.bin").string())), seed);
            require(actor.update == 128 && actor.correlation == correlation,
                    "Frozen evaluation identity mismatch");
            require(!fs::exists(root / "motion-evaluation.csv"),
                    "Motion evaluation already attempted; no repeats");
            started = std::chrono::steady_clock::now();
            Budget budget;
            budget.maximum_controls = 640;
            budget.maximum_substeps = 6400;
            budget.seconds = 10;
            const auto result = evaluate(actor.parameters, root, "motion", budget);
            require(budget.controls == 640 && budget.substeps == 6400 &&
                        cp::read(checkpoint_path.string()) == original,
                    "Frozen evaluation changed checkpoint/accounting");
            std::cout << "FROZEN_MOTION_DONE controls=" << budget.controls << " substeps=" << budget.substeps
                      << " passes=" << result.passed << std::endl;
            return 0;
        }
        if (std::string(argv[1]) == "--prepare" || std::string(argv[1]) == "--prepare-verify")
            return prepare(root);
        require(std::string(argv[1]) == "--run" || std::string(argv[1]) == "--continue-pre-training",
                "Unknown mode");
        return run(root, std::string(argv[1]) == "--continue-pre-training");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
