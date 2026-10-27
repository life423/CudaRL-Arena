// Reproducible end-to-end GPU training benchmark. No renderer, throttle or ML framework.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "rl/racing_ppo.h"
#include "demos/racing_showcase.h"
#include "core/rng.h"

namespace {
int option(int argc, char** argv, const char* key, int fallback) {
    const std::string prefix = std::string(key) + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.compare(0, prefix.size(), prefix) == 0) {
            size_t end = 0;
            const std::string value = arg.substr(prefix.size());
            const int parsed = std::stoi(value, &end);
            if (end != value.size())
                throw std::invalid_argument("invalid option: " + arg);
            return parsed;
        }
    }
    return fallback;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double index = fraction * static_cast<double>(values.size() - 1);
    const size_t low = static_cast<size_t>(index);
    const size_t high = std::min(low + 1, values.size() - 1);
    return values[low] + (values[high] - values[low]) * (index - static_cast<double>(low));
}

void simulation_benchmark(const cudarl::rl::RacingPpoConfig& cfg, int warmup, int updates, int trial) {
    const int n = cfg.num_envs, steps = cfg.rollout_steps;
    cudarl::racing::RacingBatch batch(n, cfg.seed, true, cfg.distributed_traffic);
    // The identical fixed action tape drives both implementations, independently
    // of sensor rounding or a learned policy. Setup/copies are outside timing.
    std::vector<float> tape(static_cast<size_t>(steps) * n * cudarl::racing::kActionDim);
    for (size_t i = 0; i < tape.size(); ++i)
        tape[i] = 0.6f * cudarl::rng::normal(cfg.seed, 19, static_cast<uint32_t>(i));
    device_ptr<float> actions(tape.size()), reward(n), done(n), truncated(n),
        final(static_cast<size_t>(n) * cudarl::racing::kObsDim), length(n), laps(n), metrics(n * 4);
    actions.copy_from_host(tape.data(), tape.size());
    CUDA_CHECK_THROW(cudaDeviceSynchronize()); // finish setup before even a zero-warmup measurement
    const cudarl::racing::StepOutputs out{reward.get(), done.get(), truncated.get(), final.get(),
                                          length.get(), laps.get(), metrics.get(),   cfg.distributed_traffic};
    cudaStream_t stream = nullptr;
    CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<double> times;
    double total = 0;
    try {
        for (int u = 0; u < warmup + updates; ++u) {
            const auto start = std::chrono::steady_clock::now();
            for (int t = 0; t < steps; ++t)
                batch.step(actions.get() + static_cast<size_t>(t) * n * cudarl::racing::kActionDim, out,
                           stream);
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (u >= warmup) {
                times.push_back(ms);
                total += ms;
            }
        }
        CUDA_CHECK_THROW(cudaStreamDestroy(stream));
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }
    const double mean = total / updates;
    std::printf("%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n", trial + 1, cfg.distributed_traffic ? 1 : 0, n,
                n / cudarl::racing::kRaceCars, warmup, updates, mean, percentile(times, .1),
                percentile(times, .5), percentile(times, .9), static_cast<double>(n) * steps / (mean * 1000));
    std::fflush(stdout);
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::set<std::string> supported{
            "envs",   "seed",    "total_updates", "profile",    "distributed", "compare",       "evaluate",
            "warmup", "updates", "repeats",       "simulation", "graphs",      "compare_graphs"};
        for (int i = 1; i < argc; ++i) {
            const std::string argument(argv[i]);
            if (argument == "--help") {
                std::puts("racing_bench [envs=2048] [seed=1] [warmup=10] [updates=50] [repeats=3] "
                          "[total_updates=1200] [compare=0|1] [distributed=0|1] [profile=0|1] [evaluate=0|1] "
                          "[simulation=0|1] [graphs=0|1, default 1] [compare_graphs=0|1]\ncompare alternates "
                          "retained-reference/distributed order. "
                          "profile adds diagnostic events. evaluate runs a 60 s CPU showcase after training. "
                          "simulation uses an identical fixed action tape without training.");
                return 0;
            }
            if (!supported.count(argument.substr(0, argument.find('='))) ||
                argument.find('=') == std::string::npos)
                throw std::invalid_argument("unknown option: " + argument);
        }
        cudarl::rl::RacingPpoConfig cfg;
        cfg.num_envs = option(argc, argv, "envs", 2048);
        cfg.seed = static_cast<uint32_t>(option(argc, argv, "seed", 1));
        cfg.total_updates = option(argc, argv, "total_updates", 1200);
        cfg.profile_phases = option(argc, argv, "profile", 0) != 0;
        cfg.distributed_traffic = option(argc, argv, "distributed", 1) != 0;
        cfg.graph_optimization = option(argc, argv, "graphs", 1) != 0;
        const bool compare = option(argc, argv, "compare", 0) != 0;
        const bool compare_graphs = option(argc, argv, "compare_graphs", 0) != 0;
        if (compare && compare_graphs)
            throw std::invalid_argument("choose compare or compare_graphs, not both");
        const bool evaluate = option(argc, argv, "evaluate", 0) != 0;
        const bool simulation = option(argc, argv, "simulation", 0) != 0;
        const int warmup = option(argc, argv, "warmup", 10);
        const int updates = option(argc, argv, "updates", 50);
        const int repeats = option(argc, argv, "repeats", 3);
        if (warmup < 0 || updates <= 0 || repeats <= 0 || warmup > cfg.total_updates ||
            updates > cfg.total_updates - warmup)
            throw std::invalid_argument(
                "require warmup>=0, updates/repeats>0, total_updates>=warmup+updates");
        if (cfg.num_envs <= 0 || cfg.num_envs % cudarl::racing::kRaceCars != 0 ||
            cfg.num_envs > std::numeric_limits<int>::max() / (cfg.rollout_steps * cudarl::racing::kObsDim))
            throw std::invalid_argument(
                "envs must be a positive multiple of eight within supported indexing limits");
        if (simulation &&
            (evaluate || cfg.profile_phases || compare_graphs || option(argc, argv, "graphs", 0) != 0))
            throw std::invalid_argument("simulation cannot be combined with evaluate/profile/graphs");
        if (simulation)
            cfg.graph_optimization = false;
        cudaDeviceProp device{};
        CUDA_CHECK_THROW(cudaGetDeviceProperties(&device, 0));
        int driver = 0, runtime = 0;
        CUDA_CHECK_THROW(cudaDriverGetVersion(&driver));
        CUDA_CHECK_THROW(cudaRuntimeGetVersion(&runtime));
        std::printf("# %s; SMs=%d; CUDA driver/runtime=%d/%d; seed=%u; horizon=%d; epochs=%d; "
                    "minibatches=%d; anneal_total=%d; profile=%d; distributed=%d; simulation=%d; graphs=%d\n",
                    device.name, device.multiProcessorCount, driver, runtime, cfg.seed, cfg.rollout_steps,
                    cfg.epochs, cfg.minibatches, cfg.total_updates, cfg.profile_phases ? 1 : 0,
                    cfg.distributed_traffic ? 1 : 0, simulation ? 1 : 0, cfg.graph_optimization ? 1 : 0);
        if (simulation)
            std::printf("trial,distributed,agents,races,warmup,updates,mean_ms,p10_ms,p50_ms,p90_ms,M_agent_"
                        "steps_s\n");
        else
            std::printf("trial,distributed,agents,races,warmup,updates,mean_ms,p10_ms,p50_ms,p90_ms,M_agent_"
                        "steps_s,rollout_ms,bootstrap_gae_ms,metrics_ms,optimize_ms,readback_ms,advance_ms,"
                        "resolve_ms,observe_ms,last_laps,clean_laps,contacts,walls,eval_clean,eval_contacts,"
                        "eval_walls,eval_best_s,gather_ms,forward_loss_ms,actor_update_ms,critic_update_ms,"
                        "optimization_submit_ms,graphs,graph_setup_ms,graph_nodes\n");
        for (int trial = 0; trial < repeats; ++trial) {
            // Alternate order to reduce the effect of clock/temperature drift.
            const int modes = (compare || compare_graphs) ? 2 : 1;
            for (int mode = 0; mode < modes; ++mode) {
                if (compare)
                    cfg.distributed_traffic = ((mode + trial) % 2) != 0;
                if (compare_graphs)
                    cfg.graph_optimization = ((mode + trial) % 2) != 0;
                if (simulation) {
                    simulation_benchmark(cfg, warmup, updates, trial);
                    continue;
                }
                cudarl::rl::RacingPpoTrainer trainer(cfg);
                for (int u = 0; u < warmup; ++u)
                    trainer.update();
                std::vector<double> times;
                cudarl::rl::RacingStats last{}, sum{};
                double total = 0.0;
                for (int u = 0; u < updates; ++u) {
                    const auto start = std::chrono::steady_clock::now();
                    last = trainer.update();
                    const double ms =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                            .count();
                    times.push_back(ms);
                    total += ms;
                    sum.rollout_ms += last.rollout_ms;
                    sum.bootstrap_gae_ms += last.bootstrap_gae_ms;
                    sum.metrics_ms += last.metrics_ms;
                    sum.optimize_ms += last.optimize_ms;
                    sum.readback_ms += last.readback_ms;
                    sum.advance_ms += last.advance_ms;
                    sum.resolve_ms += last.resolve_ms;
                    sum.observe_ms += last.observe_ms;
                    sum.gather_ms += last.gather_ms;
                    sum.forward_loss_ms += last.forward_loss_ms;
                    sum.actor_update_ms += last.actor_update_ms;
                    sum.critic_update_ms += last.critic_update_ms;
                    sum.optimization_submit_ms += last.optimization_submit_ms;
                    sum.optimization_graph_setup_ms += last.optimization_graph_setup_ms;
                    sum.clean_laps += last.clean_laps;
                    sum.contact_cars += last.contact_cars;
                    sum.wall_crashes += last.wall_crashes;
                }
                const double mean = total / updates;
                int eval_clean = -1, eval_contacts = -1, eval_walls = -1;
                float eval_best = 0.f;
                if (evaluate) {
                    cudarl::demos::RacingShowcase demo(cudarl::racing::kRaceCars, cfg.hidden, 55);
                    demo.set_policy(trainer.actor_params());
                    for (int tick = 0; tick < 1200; ++tick)
                        demo.step();
                    eval_clean = demo.clean_laps();
                    eval_contacts = demo.contacts();
                    eval_walls = demo.wall_crashes();
                    eval_best = demo.best_lap();
                }
                std::printf("%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%."
                            "4f,%.5f,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%.4f,%d\n",
                            trial + 1, cfg.distributed_traffic ? 1 : 0, cfg.num_envs,
                            cfg.num_envs / cudarl::racing::kRaceCars, warmup, updates, mean,
                            percentile(times, .1), percentile(times, .5), percentile(times, .9),
                            static_cast<double>(cfg.num_envs) * cfg.rollout_steps / (mean * 1000.0),
                            sum.rollout_ms / updates, sum.bootstrap_gae_ms / updates,
                            sum.metrics_ms / updates, sum.optimize_ms / updates, sum.readback_ms / updates,
                            sum.advance_ms / updates, sum.resolve_ms / updates, sum.observe_ms / updates,
                            last.mean_laps, sum.clean_laps, sum.contact_cars, sum.wall_crashes, eval_clean,
                            eval_contacts, eval_walls, eval_best, sum.gather_ms / updates,
                            sum.forward_loss_ms / updates, sum.actor_update_ms / updates,
                            sum.critic_update_ms / updates, sum.optimization_submit_ms / updates,
                            cfg.graph_optimization ? 1 : 0, sum.optimization_graph_setup_ms / updates,
                            last.optimization_graph_nodes);
                std::fflush(stdout);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "racing_bench: %s\n", error.what());
        return 1;
    }
}
