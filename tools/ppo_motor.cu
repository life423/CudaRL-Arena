// One native PPO runner. Four independent joint targets; fresh initialization.
// Frozen evaluations never update the network. No preset gait or imported weights.
#include "motor/frozen_controller.h"
#include "nn/adam.h"
#include "nn/mlp_gpu.h"
#include "rl/gaussian_kernels.h"
#include "rl/gaussian_math.h"
#include "rl/motor_rollout.cuh"
#include "rl/ppo_checkpoint.h"
#include "rl/ppo_kernels.h"
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
constexpr int environments = 32, horizon = 160, rows = environments * horizon, minibatch = 256, epochs = 2,
              updates = 128;
constexpr int training_cap = updates * rows, control_cap = training_cap + 64 + 1280 + 2400,
              substep_cap = control_cap * 10;
constexpr uint32_t seed = 0x4a544731u;
constexpr float log_sigma = -1, wall_cap = 60;
auto started = std::chrono::steady_clock::now();
void require(bool ok, const char *reason) {
    if (!ok)
        throw std::runtime_error(reason);
}
void wall_guard() {
    require(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < wall_cap,
            "Native60s wall cap reached");
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
    void step(m::World &world, const float *command) {
        require(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < seconds,
                "Physical evaluation wall cap reached");
        require(controls < maximum_controls && substeps + 10 <= maximum_substeps,
                "Physical budget exhausted");
        ++controls;
        const auto before = world.attempted_substeps;
        const bool ok = cudarl::joint_motor::step_state(world, command);
        substeps += static_cast<int>(world.attempted_substeps - before);
        require(ok && world.attempted_substeps - before == 10, "CPU physical safety stop");
    }
};
struct Rollout {
    device_ptr<m::World> worlds{environments};
    device_ptr<m::ForwardTask> tasks{environments};
    device_ptr<gpu::Counts> counts{environments};
    device_ptr<int> fault{1};
    device_ptr<float> obs{rows * observations}, action{rows * actions}, logp{rows}, value{rows}, reward{rows},
        done{rows}, zeros{rows}, last{environments}, advantage{rows}, returns{rows}, adv_stats{2},
        sigma{actions};
    Rollout(cudaStream_t s) {
        CUDA_CHECK_THROW(cudaMemsetAsync(counts.get(), 0, counts.size_bytes(), s));
        CUDA_CHECK_THROW(cudaMemsetAsync(fault.get(), 0, fault.size_bytes(), s));
        CUDA_CHECK_THROW(cudaMemsetAsync(zeros.get(), 0, zeros.size_bytes(), s));
        CUDA_CHECK_THROW(cudaMemsetAsync(last.get(), 0, last.size_bytes(), s));
        sigma.copy_from_host(std::vector<float>(actions, log_sigma).data(), actions);
    }
};
struct Batches {
    device_ptr<float> obs{minibatch * observations}, action{minibatch * actions}, logp{minibatch},
        advantage{minibatch}, returns{minibatch}, mean_gradient{minibatch * actions},
        std_gradient{minibatch * actions}, value_gradient{minibatch}, row_stats{minibatch * 5}, stats{5};
};
std::vector<uint8_t> protocol() {
    cp::Writer out;
    out.value(uint32_t{0x3150504d});
    out.value(uint32_t{2});
    out.value(seed);
    for (int v : {6, 6, 3, 3, observations, actions, environments, horizon, updates, minibatch, epochs,
                  training_cap, control_cap, substep_cap})
        out.value(v);
    for (float v : {.008f, log_sigma, .001f, .00001f, .5f, .99f, 1.f, .2f, .01f, wall_cap})
        out.value(v);
    return out.data;
}
std::vector<uint8_t> initialization() {
    cp::Writer out;
    out.value(uint32_t{0x31494f50});
    out.value(seed);
    out.array(fresh_actor());
    out.array(fresh_critic());
    out.array(std::vector<float>(actions, log_sigma));
    out.value(uint32_t{0});
    return out.data;
}
void status(const fs::path &root, const char *phase, const Budget &budget, int update = 0) {
    std::ofstream out(root / "status.json");
    out << "{\"phase\":" << std::quoted(phase) << ",\"unix_seconds\":" << std::setprecision(17)
        << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()
        << ",\"updates\":" << update << ",\"controls\":" << budget.controls
        << ",\"physical_substeps\":" << budget.substeps << "}\n";
    out.flush();
}
void check_initialization() {
    Stream stream;
    Rollout data(stream.value);
    gpu::initialize<<<1, 32, 0, stream.value>>>(data.worlds.get(), data.tasks.get(), environments,
                                                data.fault.get());
    gpu::observe<<<1, 32, 0, stream.value>>>(data.worlds.get(), data.tasks.get(), data.obs.get(),
                                             environments, data.fault.get());
    CUDA_CHECK_THROW(cudaGetLastError());
    const auto actual = download(data.obs.get(), environments * observations, stream.value);
    finite(actual);
    require(download(data.fault.get(), 1, stream.value)[0] == 0, "Zero-step device initialization failed");
    for (int i = 0; i < environments; ++i) {
        auto world = m::initial_world(i % 2 ? 3 : 0);
        m::ForwardTask task;
        require(m::begin_forward(world, m::config_for_speed(.008f), task), "CPU initialization failed");
        float expected[observations]{};
        require(m::observe(task, world, m::config_for_speed(.008f), expected), "CPU observation failed");
        for (int j = 0; j < observations; ++j)
            require(fabsf(expected[j] - actual[i * observations + j]) <= 1e-6f,
                    "Zero-step CPU/CUDA observation mismatch");
    }
    nn::GpuMlp actor(spec(actions), environments, seed);
    actor.set_params(fresh_actor());
    actor.forward(data.obs.get(), environments, stream.value);
    const auto prediction = download(actor.output(), environments * actions, stream.value);
    finite(prediction);
    nn::CpuMlp<float> reference(spec(actions));
    reference.params() = fresh_actor();
    reference.forward(actual.data(), environments);
    for (size_t i = 0; i < prediction.size(); ++i)
        require(fabsf(prediction[i] - reference.output()[i]) < 1e-5f, "Fresh actor CPU/CUDA mismatch");
    rl::sample_gaussian(actor.output(), data.sigma.get(), environments, actions, seed, 0, data.action.get(),
                        data.logp.get(), stream.value);
    const auto sampled = download(data.action.get(), environments * actions, stream.value),
               probability = download(data.logp.get(), environments, stream.value);
    finite(sampled);
    finite(probability);
    const float sigma[4]{log_sigma, log_sigma, log_sigma, log_sigma};
    for (int i = 0; i < environments; ++i)
        require(fabsf(probability[i] - rl::gaussian_logp(prediction.data() + i * actions, sigma,
                                                         sampled.data() + i * actions, actions)) < 1e-5f,
                "Sampled raw latent/log probability mismatch");
    for (const auto &count : download(data.counts.get(), environments, stream.value))
        require(count.controls == 0 && count.substeps == 0, "Initialization advanced physics");
}
int prepare(const fs::path &root) {
    require(!fs::exists(root), "Run identity already exists");
    fs::create_directories(root);
    check_initialization();
    cp::write_atomic((root / "protocol.bin").string(), protocol());
    cp::write_atomic((root / "initialization.bin").string(), initialization());
    std::ofstream out(root / "readiness.json");
    out << "{\"ready\":true,\"algorithm\":\"PPO\",\"seed\":" << seed
        << ",\"actor_shape\":[40,32,32,4],\"critic_shape\":[40,32,32,1],\"checkpoint_schema\":2,\"action_"
           "mapping\":3,\"task\":6,\"reward\":3,\"independent_joint_targets\":true,\"axes\":[\"first_yaw\","
           "\"first_pitch\",\"second_yaw\",\"second_pitch\"],\"angle_limit_rad\":1,\"reference_rate_limit_"
           "rad_s\":12,\"reference_acceleration_limit_rad_s2\":200,\"torque_limit_Nm\":0.00001,\"fresh_actor_"
           "critic_Adam_rollout\":true,\"imported_weights_or_examples\":false,\"initial_mean_bias\":[0,0,0,0]"
           ",\"preset_oscillator\":false,\"Gaussian_log_std\":-1,\"target_mm_s\":8,\"environments\":32,"
           "\"rollout_controls\":160,\"PPO_updates_max\":128,\"epochs\":2,\"minibatch\":256,\"actor_"
           "optimizer_steps_max\":5120,\"critic_optimizer_steps_max\":5120,\"controls_cap\":"
        << control_cap << ",\"substeps_cap\":" << substep_cap
        << ",\"native_wall_seconds\":60,\"process_wall_seconds\":75,\"physical_steps\":0,\"optimizer_steps\":"
           "0,\"physical_axis_checks_in_run_before_training\":64,\"runtime_adoption\":false,\"retries\":0}\n";
    out.close();
    status(root, "ready", {});
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
struct Sustained {
    int passed = 0, windows_passed = 0, windows_total = 0;
};
Sustained sustained(const std::vector<float> &params, const fs::path &root, Budget &budget) {
    nn::CpuMlp<float> actor(spec(actions));
    actor.params() = params;
    Sustained result;
    std::ofstream trace(root / "sustained-windows.csv");
    trace << "pose,end_s,mean_mm_s,RMSE_mm_s,lateral_mm,depth_mm,tilt_deg,path_deg,work_J_per_m,passed\n";
    for (int pose : {0, 3}) {
        auto world = m::initial_world(pose);
        world.arena = {.6f, .4f};
        auto config = m::config_for_speed(.008f);
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
            budget.step(world, action.command);
            const auto interval = m::record_control(task, world, config);
            require(interval.accepted, "Long task rejected");
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
    }
    return result;
}
Evaluation evaluate(const std::vector<float> &params, const fs::path &root, const char *name,
                    Budget &budget) {
    nn::CpuMlp<float> actor(spec(actions));
    actor.params() = params;
    Evaluation out;
    std::ofstream trace(root / (std::string(name) + "-evaluation.csv"));
    trace << "pose,passed,steady_mm_s,RMSE_mm_s,lateral_mm,work_J_per_m\n";
    for (int pose = 0; pose < 4; ++pose) {
        auto world = m::initial_world(pose);
        auto config = m::config_for_speed(.008f);
        m::ForwardTask task;
        require(m::begin_forward(world, config, task), "Frozen evaluation initialization failed");
        for (int t = 0; t < horizon; ++t) {
            float obs[observations]{};
            require(m::observe(task, world, config, obs), "Invalid frozen observation");
            actor.forward(obs, 1);
            m::Action action;
            require(m::action_from_latent(actor.output(), action), "Invalid frozen action");
            budget.step(world, action.command);
            require(m::record_control(task, world, config).accepted, "Frozen physical/task rejection");
        }
        const bool passed = task.outcome == m::Outcome::Success;
        const float work = world.physical.body.work / fmaxf(m::measure(task, world).forward_m, 1e-9f);
        out.passed += passed ? 1 : 0;
        out.speed += task.steady_speed_m_s / 4;
        out.rmse += task.speed_rmse / 4;
        out.work_per_m += work / 4;
        trace << pose << ',' << passed << ',' << task.steady_speed_m_s * 1000 << ',' << task.speed_rmse * 1000
              << ',' << task.maximum_lateral * 1000 << ',' << work << '\n';
    }
    return out;
}
void snapshot(const fs::path &root, int update, nn::GpuMlp &actor, nn::GpuMlp &critic, nn::GpuAdam &actor_opt,
              nn::GpuAdam &critic_opt) {
    cp::Writer out;
    out.value(uint32_t{0x31525050});
    out.value(uint32_t{2});
    out.value(seed);
    out.value(cp::checksum(protocol()));
    out.value(update);
    out.value(uint32_t{6});
    out.value(uint32_t{3});
    out.value(uint32_t{6});
    out.value(uint32_t{3});
    out.value(uint32_t{40});
    out.value(uint32_t{4});
    out.array(actor.get_params());
    out.array(critic.get_params());
    out.array(std::vector<float>(actions, log_sigma));
    actor_opt.save(out);
    critic_opt.save(out);
    cp::write_atomic((root / ("ppo-update-" + std::to_string(update) + ".bin")).string(), out.data);
}
int run(const fs::path &root) {
    require(cp::read((root / "protocol.bin").string()) == protocol() &&
                cp::read((root / "initialization.bin").string()) == initialization(),
            "Frozen PPO identity changed");
    require(fs::create_directory(root / "run-started"), "Run already attempted; no retries");
    started = std::chrono::steady_clock::now();
    Budget budget;
    Evaluation before, after;
    Sustained long_result;
    int completed = 0, actor_steps = 0, critic_steps = 0;
    double rollout_ms = 0, update_ms = 0, training_seconds = 0;
    std::string fault;
    try {
        Stream stream;
        parity(budget, stream.value);
        before = evaluate(fresh_actor(), root, "initial", budget);
        nn::GpuMlp actor(spec(actions), rows, seed), critic(spec(1), rows, seed + 1);
        actor.set_params(fresh_actor());
        critic.set_params(fresh_critic());
        nn::AdamConfig optimizer;
        optimizer.lr = .001f;
        optimizer.eps = .00001f;
        optimizer.max_grad_norm = .5f;
        nn::GpuAdam actor_opt(actor.num_params(), optimizer), critic_opt(critic.num_params(), optimizer);
        require(actor_opt.steps() == 0 && critic_opt.steps() == 0, "Fresh optimizer counters are nonzero");
        snapshot(root, 0, actor, critic, actor_opt, critic_opt);
        Rollout data(stream.value);
        Batches batch;
        Timer timer;
        unsigned long long previous_controls = 0, previous_substeps = 0;
        std::ofstream log(root / "training.csv");
        log << "update,controls,actor_Adam_steps,critic_Adam_steps,mean_KL,rollout_GPU_ms,update_GPU_ms\n";
        const auto training_started = std::chrono::steady_clock::now();
        std::cout
            << "TRAINING_STARTED PPO fresh actor/critic/Adam; model6 worlds and rollouts device-resident"
            << std::endl;
        for (int update = 0; update < updates; ++update) {
            wall_guard();
            status(root, "training", budget, completed);
            timer.begin(stream.value);
            gpu::initialize<<<1, 32, 0, stream.value>>>(data.worlds.get(), data.tasks.get(), environments,
                                                        data.fault.get());
            for (int t = 0; t < horizon; ++t) {
                const int offset = t * environments;
                float *obs = data.obs.get() + offset * observations;
                gpu::observe<<<1, 32, 0, stream.value>>>(data.worlds.get(), data.tasks.get(), obs,
                                                         environments, data.fault.get());
                actor.forward(obs, environments, stream.value);
                critic.forward(obs, environments, stream.value);
                rl::sample_gaussian(actor.output(), data.sigma.get(), environments, actions, seed + 2,
                                    static_cast<uint32_t>(update * horizon + t),
                                    data.action.get() + offset * actions, data.logp.get() + offset,
                                    stream.value);
                CUDA_CHECK_THROW(cudaMemcpyAsync(data.value.get() + offset, critic.output(),
                                                 environments * sizeof(float), cudaMemcpyDeviceToDevice,
                                                 stream.value));
                gpu::advance<<<1, 32, 0, stream.value>>>(data.worlds.get(), data.tasks.get(),
                                                         data.action.get() + offset * actions,
                                                         data.reward.get() + offset, data.done.get() + offset,
                                                         data.counts.get(), environments, data.fault.get());
                CUDA_CHECK_THROW(cudaGetLastError());
            }
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
            require(download(data.fault.get(), 1, stream.value)[0] == 0,
                    "Device rollout physical/observation fault; stop before PPO");
            require(controls == static_cast<unsigned long long>((update + 1) * rows) &&
                        substeps == controls * 10,
                    "GPU rollout accounting mismatch");
            timer.begin(stream.value);
            rl::compute_gae(horizon, environments, data.reward.get(), data.value.get(), data.done.get(),
                            data.zeros.get(), data.zeros.get(), data.last.get(), .99f, 1.f,
                            data.advantage.get(), data.returns.get(), stream.value);
            rl::mean_std(data.advantage.get(), rows, data.adv_stats.get(), stream.value);
            double last_kl = 0;
            for (int epoch = 0; epoch < epochs; ++epoch) {
                CUDA_CHECK_THROW(
                    cudaMemsetAsync(batch.stats.get(), 0, batch.stats.size_bytes(), stream.value));
                const auto permutation =
                    rl::make_permutation(rows, seed + static_cast<uint32_t>(update * epochs + epoch));
                for (int first = 0; first < rows; first += minibatch) {
                    wall_guard();
                    rl::gather_rows(permutation, first, minibatch, observations, data.obs.get(),
                                    batch.obs.get(), stream.value);
                    rl::gather_rows(permutation, first, minibatch, actions, data.action.get(),
                                    batch.action.get(), stream.value);
                    rl::gather_rows(permutation, first, minibatch, 1, data.logp.get(), batch.logp.get(),
                                    stream.value);
                    rl::gather_rows(permutation, first, minibatch, 1, data.advantage.get(),
                                    batch.advantage.get(), stream.value);
                    rl::gather_rows(permutation, first, minibatch, 1, data.returns.get(), batch.returns.get(),
                                    stream.value);
                    actor.forward(batch.obs.get(), minibatch, stream.value);
                    critic.forward(batch.obs.get(), minibatch, stream.value);
                    rl::gaussian_ppo_loss_grad(actor.output(), data.sigma.get(), critic.output(), minibatch,
                                               actions, batch.action.get(), batch.logp.get(),
                                               batch.advantage.get(), batch.returns.get(),
                                               data.adv_stats.get(), {.2f, 0, .5f}, batch.mean_gradient.get(),
                                               batch.std_gradient.get(), batch.value_gradient.get(),
                                               batch.row_stats.get(), batch.stats.get(), stream.value);
                    actor.zero_grad(stream.value);
                    critic.zero_grad(stream.value);
                    actor.backward(batch.mean_gradient.get(), minibatch, nullptr, stream.value);
                    critic.backward(batch.value_gradient.get(), minibatch, nullptr, stream.value);
                    actor_opt.step(actor.params(), actor.grads(), stream.value);
                    critic_opt.step(critic.params(), critic.grads(), stream.value);
                    actor_steps = actor_opt.steps();
                    critic_steps = critic_opt.steps();
                }
                actor_steps = actor_opt.steps();
                critic_steps = critic_opt.steps();
                const auto metrics = download(batch.stats.get(), 5, stream.value);
                finite(metrics);
                last_kl = metrics[3] / rows;
                if (last_kl > .01)
                    break;
            }
            gpu::finite_parameters<<<(actor.num_params() + 255) / 256, 256, 0, stream.value>>>(
                actor.params(), actor.num_params(), data.fault.get());
            gpu::finite_parameters<<<(critic.num_params() + 255) / 256, 256, 0, stream.value>>>(
                critic.params(), critic.num_params(), data.fault.get());
            CUDA_CHECK_THROW(cudaGetLastError());
            const double elapsed_update = timer.end(stream.value);
            update_ms += elapsed_update;
            require(download(data.fault.get(), 1, stream.value)[0] == 0,
                    "Nonfinite optimized parameters; stop before further physics");
            completed = update + 1;
            require(actor_steps <= 5120 && critic_steps <= 5120, "Optimizer cap exceeded");
            if (completed % 8 == 0)
                snapshot(root, completed, actor, critic, actor_opt, critic_opt);
            log << completed << ',' << controls << ',' << actor_steps << ',' << critic_steps << ',' << last_kl
                << ',' << elapsed_rollout << ',' << elapsed_update << '\n';
            log.flush();
        }
        training_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - training_started).count();
        status(root, "frozen_evaluation", budget, completed);
        after = evaluate(actor.get_params(), root, "frozen", budget);
        long_result = sustained(actor.get_params(), root, budget);
        require(budget.controls == control_cap && budget.substeps == substep_cap,
                "Total run accounting mismatch");
    } catch (const std::exception &error) {
        fault = error.what();
    }
    const bool accepted =
        fault.empty() && after.passed == 4 && after.rmse < before.rmse && long_result.passed == 2;
    std::ofstream out(root / "result.json");
    out << std::setprecision(9) << "{\"algorithm\":\"PPO\",\"fresh_initialization\":true,\"seed\":" << seed
        << ",\"fault\":" << std::quoted(fault) << ",\"completed_updates\":" << completed
        << ",\"actor_optimizer_steps\":" << actor_steps << ",\"critic_optimizer_steps\":" << critic_steps
        << ",\"controls\":" << budget.controls << ",\"physical_substeps\":" << budget.substeps
        << ",\"initial_passes\":" << before.passed << ",\"frozen_passes\":" << after.passed
        << ",\"sustained_passes\":" << long_result.passed
        << ",\"sustained_windows_passed\":" << long_result.windows_passed
        << ",\"sustained_windows_total\":" << long_result.windows_total
        << ",\"initial_speed_mm_s\":" << before.speed * 1000
        << ",\"frozen_speed_mm_s\":" << after.speed * 1000 << ",\"initial_RMSE_mm_s\":" << before.rmse * 1000
        << ",\"frozen_RMSE_mm_s\":" << after.rmse * 1000 << ",\"frozen_work_J_per_m\":" << after.work_per_m
        << ",\"candidate_accepted\":" << (accepted ? "true" : "false") << ",\"rollout_GPU_ms\":" << rollout_ms
        << ",\"update_GPU_ms\":" << update_ms << ",\"training_seconds\":" << training_seconds
        << ",\"training_controls_per_second\":"
        << (training_seconds > 0 ? training_cap / training_seconds : 0) << ",\"native_seconds\":"
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
        require(argc == 3, "Use --prepare/--run DIR");
        const auto root = fs::absolute(argv[2]);
        if (std::string(argv[1]) == "--prepare")
            return prepare(root);
        require(std::string(argv[1]) == "--run", "Unknown mode");
        return run(root);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
