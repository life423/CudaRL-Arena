// One native PPO runner. No imported actor, teacher, data set or runtime fallback.
#include "motor/frozen_controller.h"
#include "nn/adam.h"
#include "nn/mlp_gpu.h"
#include "rl/gaussian_kernels.h"
#include "rl/gaussian_math.h"
#include "rl/motor_rollout.cuh"
#include "rl/ppo_kernels.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace {
namespace fs = std::filesystem;
namespace m = cudarl::straight_motor;
namespace p = cudarl::physics;
namespace nn = cudarl::nn;
namespace rl = cudarl::rl;
namespace gpu = rl::motor_rollout;
namespace cp = cudarl::checkpoint;
constexpr int environments = 32, horizon = 80, rows = environments * horizon, minibatch = 256, epochs = 2,
              updates = 64;
constexpr int training_cap = updates * rows, control_cap = training_cap + 32 + 640,
              substep_cap = control_cap * 10;
constexpr uint32_t seed = 0x50504f31u;
constexpr float log_sigma = -2, wall_cap = 60;
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
    return {{42, 32, 32, output}, {nn::Activation::Tanh, nn::Activation::Tanh, nn::Activation::Identity}};
}
std::vector<float> fresh_actor() {
    auto result = nn::orthogonal_params(spec(2), seed, {1.414213562f, 1.414213562f, .01f});
    result[spec(2).bias_offset(2) + 1] = atanhf(1.f / 1.5f);
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
    void step(m::World &world, const float *command) {
        wall_guard();
        require(controls < control_cap && substeps + 10 <= substep_cap, "Physical budget exhausted");
        ++controls;
        const auto before = world.attempted_substeps;
        const bool ok = cudarl::stroke_motor::step_state(world, command,
                                                         cudarl::stroke_motor::ActionMapping::SignedCadence);
        substeps += static_cast<int>(world.attempted_substeps - before);
        require(ok && world.attempted_substeps - before == 10, "CPU physical safety stop");
    }
};
struct Rollout {
    device_ptr<m::World> worlds{environments};
    device_ptr<m::ForwardTask> tasks{environments};
    device_ptr<gpu::Counts> counts{environments};
    device_ptr<int> fault{1};
    device_ptr<float> obs{rows * 42}, action{rows * 2}, logp{rows}, value{rows}, reward{rows}, done{rows},
        zeros{rows}, last{environments}, advantage{rows}, returns{rows}, adv_stats{2}, sigma{2};
    Rollout(cudaStream_t s) {
        CUDA_CHECK_THROW(cudaMemsetAsync(counts.get(), 0, counts.size_bytes(), s));
        CUDA_CHECK_THROW(cudaMemsetAsync(fault.get(), 0, fault.size_bytes(), s));
        CUDA_CHECK_THROW(cudaMemsetAsync(zeros.get(), 0, zeros.size_bytes(), s));
        CUDA_CHECK_THROW(cudaMemsetAsync(last.get(), 0, last.size_bytes(), s));
        sigma.copy_from_host(std::vector<float>{log_sigma, log_sigma}.data(), 2);
    }
};
struct Batches {
    device_ptr<float> obs{minibatch * 42}, action{minibatch * 2}, logp{minibatch}, advantage{minibatch},
        returns{minibatch}, mean_gradient{minibatch * 2}, std_gradient{minibatch * 2},
        value_gradient{minibatch}, row_stats{minibatch * 5}, stats{5};
};
std::vector<uint8_t> protocol() {
    cp::Writer out;
    out.value(uint32_t{0x3150504d});
    out.value(uint32_t{1});
    out.value(seed);
    for (int v : {6, 5, 2, 2, environments, horizon, updates, minibatch, epochs, training_cap, control_cap,
                  substep_cap})
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
    out.array(std::vector<float>{log_sigma, log_sigma});
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
    const auto actual = download(data.obs.get(), environments * 42, stream.value);
    finite(actual);
    require(download(data.fault.get(), 1, stream.value)[0] == 0, "Zero-step device initialization failed");
    for (int i = 0; i < environments; ++i) {
        auto world = m::initial_world(i % 2 ? 3 : 0);
        m::ForwardTask task;
        require(m::begin_forward(world, m::config_for_speed(.008f), task), "CPU initialization failed");
        float expected[42]{};
        require(m::observe(task, world, m::config_for_speed(.008f), expected), "CPU observation failed");
        for (int j = 0; j < 42; ++j)
            require(fabsf(expected[j] - actual[i * 42 + j]) <= 1e-6f,
                    "Zero-step CPU/CUDA observation mismatch");
    }
    nn::GpuMlp actor(spec(2), environments, seed);
    actor.set_params(fresh_actor());
    actor.forward(data.obs.get(), environments, stream.value);
    const auto prediction = download(actor.output(), environments * 2, stream.value);
    finite(prediction);
    nn::CpuMlp<float> reference(spec(2));
    reference.params() = fresh_actor();
    reference.forward(actual.data(), environments);
    for (size_t i = 0; i < prediction.size(); ++i)
        require(fabsf(prediction[i] - reference.output()[i]) < 1e-5f, "Fresh actor CPU/CUDA mismatch");
    rl::sample_gaussian(actor.output(), data.sigma.get(), environments, 2, seed, 0, data.action.get(),
                        data.logp.get(), stream.value);
    const auto actions = download(data.action.get(), environments * 2, stream.value),
               probability = download(data.logp.get(), environments, stream.value);
    finite(actions);
    finite(probability);
    const float sigma[2]{log_sigma, log_sigma};
    for (int i = 0; i < environments; ++i)
        require(fabsf(probability[i] -
                      rl::gaussian_logp(prediction.data() + i * 2, sigma, actions.data() + i * 2, 2)) < 1e-5f,
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
        << ",\"fresh_actor_critic_Adam_rollout\":true,\"imported_weights_or_examples\":false,\"initial_"
           "nominal_mean_amplitude_rad\":0.35,\"initial_nominal_mean_cadence_Hz\":1,\"Gaussian_log_std\":-2,"
           "\"target_mm_s\":8,\"environments\":32,\"rollout_controls\":80,\"PPO_updates_max\":64,\"epochs\":"
           "2,\"minibatch\":256,\"actor_optimizer_steps_max\":1280,\"critic_optimizer_steps_max\":1280,"
           "\"controls_cap\":"
        << control_cap << ",\"substeps_cap\":" << substep_cap
        << ",\"native_wall_seconds\":60,\"process_wall_seconds\":75,\"physical_steps\":0,\"optimizer_steps\":"
           "0,\"training_worlds_and_rollout_GPU_resident\":true,\"frozen_evaluation\":\"CPU shared model6; "
           "initial/final four poses; unchanged4second8mm/s "
           "gates\",\"runtime_adoption\":false,\"retries\":0}\n";
    out.close();
    status(root, "ready", {});
    std::ifstream immediate(root / "readiness.json");
    std::cout << immediate.rdbuf() << std::flush;
    return 0;
}
std::vector<float> fields(const m::World &world) {
    const auto &b = world.physical.body;
    std::vector<float> f{b.position.x,    b.position.y,       b.position.z,       b.orientation.x,
                         b.orientation.y, b.orientation.z,    b.orientation.w,    b.work,
                         world.phase,     world.parameter[0], world.parameter[1], world.rate[0],
                         world.rate[1]};
    for (float v : b.angle)
        f.push_back(v);
    for (float v : b.velocity)
        f.push_back(v);
    return f;
}
void parity(Budget &budget, cudaStream_t stream) {
    std::vector<m::World> cpu;
    for (int i = 0; i < 4; ++i)
        cpu.push_back(m::initial_world(i));
    device_ptr<m::World> worlds(4);
    device_ptr<m::ForwardTask> tasks(4);
    device_ptr<gpu::Counts> counts(4);
    device_ptr<int> fault(1);
    device_ptr<float> actions(8), rewards(4), done(4);
    std::vector<m::ForwardTask> requests(4);
    for (int i = 0; i < 4; ++i)
        require(m::begin_forward(cpu[i], m::config_for_speed(.008f), requests[i]), "Parity start invalid");
    worlds.copy_from_host(cpu.data(), 4);
    tasks.copy_from_host(requests.data(), 4);
    CUDA_CHECK_THROW(cudaMemsetAsync(counts.get(), 0, counts.size_bytes(), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(fault.get(), 0, fault.size_bytes(), stream));
    const float latent[2]{0, atanhf(1.f / 1.5f)};
    std::vector<float> raw;
    for (int i = 0; i < 4; ++i)
        raw.insert(raw.end(), latent, latent + 2);
    actions.copy_from_host(raw.data(), 8);
    for (int t = 0; t < 4; ++t) {
        m::Action action;
        require(m::action_from_latent(latent, action), "Invalid parity action");
        for (auto &world : cpu)
            budget.step(world, action.command);
        gpu::advance<<<1, 32, 0, stream>>>(worlds.get(), tasks.get(), actions.get(), rewards.get(),
                                           done.get(), counts.get(), 4, fault.get());
        CUDA_CHECK_THROW(cudaGetLastError());
    }
    const auto attempted = download(counts.get(), 4, stream);
    for (const auto &count : attempted) {
        budget.controls += static_cast<int>(count.controls);
        budget.substeps += static_cast<int>(count.substeps);
    }
    require(download(fault.get(), 1, stream)[0] == 0, "GPU parity physical fault");
    const auto actual = download(worlds.get(), 4, stream);
    for (int i = 0; i < 4; ++i) {
        const auto a = fields(cpu[i]), b = fields(actual[i]);
        finite(a);
        finite(b);
        for (size_t j = 0; j < a.size(); ++j)
            require(fabsf(a[j] - b[j]) <= 2e-4f + 2e-3f * fabsf(a[j]), "Model6 CPU/CUDA parity mismatch");
        require(cpu[i].physical.substeps == actual[i].physical.substeps, "Parity count mismatch");
    }
    require(budget.controls == 32 && budget.substeps == 320, "Parity accounting mismatch");
}
struct Evaluation {
    int passed = 0;
    float speed = 0, rmse = 0, work_per_m = 0;
};
Evaluation evaluate(const std::vector<float> &params, const fs::path &root, const char *name,
                    Budget &budget) {
    nn::CpuMlp<float> actor(spec(2));
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
            float obs[42]{};
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
    out.value(uint32_t{1});
    out.value(seed);
    out.value(cp::checksum(protocol()));
    out.value(update);
    out.value(uint32_t{6});
    out.value(uint32_t{2});
    out.array(actor.get_params());
    out.array(critic.get_params());
    out.array(std::vector<float>{log_sigma, log_sigma});
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
    int completed = 0, actor_steps = 0, critic_steps = 0;
    double rollout_ms = 0, update_ms = 0, training_seconds = 0;
    std::string fault;
    try {
        Stream stream;
        parity(budget, stream.value);
        before = evaluate(fresh_actor(), root, "initial", budget);
        nn::GpuMlp actor(spec(2), rows, seed), critic(spec(1), rows, seed + 1);
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
                float *obs = data.obs.get() + offset * 42;
                gpu::observe<<<1, 32, 0, stream.value>>>(data.worlds.get(), data.tasks.get(), obs,
                                                         environments, data.fault.get());
                actor.forward(obs, environments, stream.value);
                critic.forward(obs, environments, stream.value);
                rl::sample_gaussian(actor.output(), data.sigma.get(), environments, 2, seed + 2,
                                    static_cast<uint32_t>(update * horizon + t),
                                    data.action.get() + offset * 2, data.logp.get() + offset, stream.value);
                CUDA_CHECK_THROW(cudaMemcpyAsync(data.value.get() + offset, critic.output(),
                                                 environments * sizeof(float), cudaMemcpyDeviceToDevice,
                                                 stream.value));
                gpu::advance<<<1, 32, 0, stream.value>>>(data.worlds.get(), data.tasks.get(),
                                                         data.action.get() + offset * 2,
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
                    rl::gather_rows(permutation, first, minibatch, 42, data.obs.get(), batch.obs.get(),
                                    stream.value);
                    rl::gather_rows(permutation, first, minibatch, 2, data.action.get(), batch.action.get(),
                                    stream.value);
                    rl::gather_rows(permutation, first, minibatch, 1, data.logp.get(), batch.logp.get(),
                                    stream.value);
                    rl::gather_rows(permutation, first, minibatch, 1, data.advantage.get(),
                                    batch.advantage.get(), stream.value);
                    rl::gather_rows(permutation, first, minibatch, 1, data.returns.get(), batch.returns.get(),
                                    stream.value);
                    actor.forward(batch.obs.get(), minibatch, stream.value);
                    critic.forward(batch.obs.get(), minibatch, stream.value);
                    rl::gaussian_ppo_loss_grad(
                        actor.output(), data.sigma.get(), critic.output(), minibatch, 2, batch.action.get(),
                        batch.logp.get(), batch.advantage.get(), batch.returns.get(), data.adv_stats.get(),
                        {.2f, 0, .5f}, batch.mean_gradient.get(), batch.std_gradient.get(),
                        batch.value_gradient.get(), batch.row_stats.get(), batch.stats.get(), stream.value);
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
            require(actor_steps <= 1280 && critic_steps <= 1280, "Optimizer cap exceeded");
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
        require(budget.controls == control_cap && budget.substeps == substep_cap,
                "Total run accounting mismatch");
    } catch (const std::exception &error) {
        fault = error.what();
    }
    const bool accepted = fault.empty() && after.passed == 4 && after.rmse < before.rmse;
    std::ofstream out(root / "result.json");
    out << std::setprecision(9) << "{\"algorithm\":\"PPO\",\"fresh_initialization\":true,\"seed\":" << seed
        << ",\"fault\":" << std::quoted(fault) << ",\"completed_updates\":" << completed
        << ",\"actor_optimizer_steps\":" << actor_steps << ",\"critic_optimizer_steps\":" << critic_steps
        << ",\"controls\":" << budget.controls << ",\"physical_substeps\":" << budget.substeps
        << ",\"initial_passes\":" << before.passed << ",\"frozen_passes\":" << after.passed
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
