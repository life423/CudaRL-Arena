// Benchmarks the hand-written NN kernels against the hardware roofline and
// against cuBLAS. cuBLAS is linked into this tool only, as a yardstick; the
// library never uses it.
//
// Output is markdown, ready to paste into the README.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "attention_baseline.h"
#include "core/bf16.h"
#include "core/rng.h"
#include "cuda/cuda_memory.h"
#include "nn/adam.h"
#include "nn/attention.h"
#include "nn/attention_bf16.h"
#include "nn/gemm.h"
#include "nn/gemm_bf16.h"
#include "nn/loss.h"
#include "nn/mlp_cpu.h"
#include "nn/mlp_gpu.h"
#include "rl/ppo.h"

using namespace cudarl::nn;

namespace {

#define CUBLAS_CHECK(expr)                                                              \
    do {                                                                                \
        const cublasStatus_t s_ = (expr);                                               \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                              \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(s_), __FILE__, __LINE__); \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

struct Hardware {
    std::string name;
    int    sms = 0;
    int    fp32_lanes_per_sm = 0;
    double clock_ghz = 0;
    double peak_gflops = 0;        // fp32 FMA = 2 flops
    double spec_bandwidth_gbs = 0; // from memory clock and bus width
    double copy_bandwidth_gbs = 0; // measured device-to-device copy
};

int fp32_lanes_per_sm(int major, int minor) {
    if (major == 12 || major == 9 || (major == 8 && minor >= 6)) return 128;   // Blackwell consumer, Hopper, GA10x/Ada
    if (major == 8 || major == 7) return 64;
    return 128;
}

// Mean time per call over `reps` calls, taking the best of 5 such trials: GPU
// clocks and thermals drift between runs, and the minimum is the most
// repeatable estimate of what the kernel itself costs.
template <typename F>
double time_ms(F&& fn, int reps) {
    constexpr int kTrials = 5;
    for (int i = 0; i < 3; ++i) fn();   // warm up
    cudaEvent_t start, stop;
    CUDA_CHECK_THROW(cudaEventCreate(&start));
    CUDA_CHECK_THROW(cudaEventCreate(&stop));
    double best = 0.0;
    for (int trial = 0; trial < kTrials; ++trial) {
        CUDA_CHECK_THROW(cudaEventRecord(start));
        for (int i = 0; i < reps; ++i) fn();
        CUDA_CHECK_THROW(cudaEventRecord(stop));
        CUDA_CHECK_THROW(cudaEventSynchronize(stop));
        float ms = 0.f;
        CUDA_CHECK_THROW(cudaEventElapsedTime(&ms, start, stop));
        if (trial == 0 || ms / reps < best) best = ms / reps;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return best;
}

Hardware query_hardware() {
    Hardware hw;
    cudaDeviceProp prop;
    CUDA_CHECK_THROW(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0, mem_clock_khz = 0, bus_bits = 0;
    CUDA_CHECK_THROW(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    CUDA_CHECK_THROW(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK_THROW(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    hw.name               = prop.name;
    hw.sms                = prop.multiProcessorCount;
    hw.fp32_lanes_per_sm  = fp32_lanes_per_sm(prop.major, prop.minor);
    hw.clock_ghz          = clock_khz * 1e-6;
    hw.peak_gflops        = 2.0 * hw.sms * hw.fp32_lanes_per_sm * hw.clock_ghz;
    hw.spec_bandwidth_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8.0) * 1e-9;

    const size_t bytes = size_t(1) << 30;   // 1 GiB, far larger than L2
    device_ptr<char> a(bytes), b(bytes);
    const double ms = time_ms([&] { CUDA_CHECK_THROW(cudaMemcpyAsync(b.get(), a.get(), bytes, cudaMemcpyDeviceToDevice)); }, 10);
    hw.copy_bandwidth_gbs = 2.0 * bytes / (ms * 1e-3) * 1e-9;   // each byte is read and written
    return hw;
}

device_ptr<float> random_device(size_t n, uint32_t seed) {
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = cudarl::rng::normal(seed, 0, static_cast<uint32_t>(i));
    device_ptr<float> d(n);
    d.copy_from_host(h.data(), n);
    return d;
}

std::vector<float> host_copy(const device_ptr<float>& d) {
    std::vector<float> h(d.size());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    d.copy_to_host(h.data(), h.size());
    return h;
}

// Row-major C = op(A) op(B) through column-major cuBLAS: compute C^T = op(B)^T op(A)^T.
void cublas_gemm(cublasHandle_t h, Trans ta, Trans tb, int m, int n, int k, const float* A, int lda,
                 const float* B, int ldb, float* C, int ldc) {
    const float one = 1.f, zero = 0.f;
    CUBLAS_CHECK(cublasSgemm(h, tb == Trans::N ? CUBLAS_OP_N : CUBLAS_OP_T, ta == Trans::N ? CUBLAS_OP_N : CUBLAS_OP_T,
                             n, m, k, &one, B, ldb, A, lda, &zero, C, ldc));
}

struct GemmCase {
    const char* label;
    Trans ta, tb;
    int m, n, k;
};

void bench_gemm(const Hardware& hw, cublasHandle_t handle, const GemmCase& c, int max_splits_override = -1) {
    const int lda = c.ta == Trans::N ? c.k : c.m;
    const int ldb = c.tb == Trans::N ? c.n : c.k;
    auto A = random_device(size_t(c.m) * c.k, 1), B = random_device(size_t(c.k) * c.n, 2);
    device_ptr<float> C(size_t(c.m) * c.n), C_ref(size_t(c.m) * c.n), ws(size_t(kMaxSplitK) * c.m * c.n);

    const double flops     = 2.0 * c.m * c.n * c.k;
    const double min_bytes = 4.0 * (double(c.m) * c.k + double(c.k) * c.n + double(c.m) * c.n);
    const double intensity = flops / min_bytes;
    const double roof      = std::min(hw.peak_gflops, intensity * hw.copy_bandwidth_gbs);
    const int    reps      = std::clamp(static_cast<int>(2e10 / flops), 5, 200);
    const int    splits    = max_splits_override >= 0 ? max_splits_override : split_k_count(c.m, c.n, c.k, hw.sms);

    const double t_naive = time_ms([&] { gemm(GemmKernel::Naive, c.ta, c.tb, c.m, c.n, c.k, A.get(), lda, B.get(), ldb, C.get(), c.n); }, reps);
    const double t_tiled = time_ms([&] {
        gemm_split_k(c.ta, c.tb, c.m, c.n, c.k, A.get(), lda, B.get(), ldb, C.get(), c.n, splits, ws.get());
    }, reps);
    const double t_cublas = time_ms([&] { cublas_gemm(handle, c.ta, c.tb, c.m, c.n, c.k, A.get(), lda, B.get(), ldb, C_ref.get(), c.n); }, reps);

    // Sanity: our result must match cuBLAS before its timing means anything.
    const auto ours = host_copy(C), ref = host_copy(C_ref);
    double max_diff = 0, max_ref = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_diff = std::max(max_diff, double(std::fabs(ours[i] - ref[i])));
        max_ref  = std::max(max_ref, double(std::fabs(ref[i])));
    }
    const bool agree = max_diff <= 1e-4 * std::max(1.0, max_ref);

    auto gf = [&](double ms) { return flops / (ms * 1e-3) * 1e-9; };
    std::printf("| %-26s | %5d x %5d x %5d | %6.1f | %7.0f | %7.0f | %4.1f%% | %4.0f%% | %7.0f | %4.0f%% | %2d | %s |\n",
                c.label, c.m, c.n, c.k, intensity, gf(t_naive), gf(t_tiled), 100 * gf(t_tiled) / hw.peak_gflops,
                100 * gf(t_tiled) / roof, gf(t_cublas), 100 * t_cublas / t_tiled, splits, agree ? "yes" : "NO");
}

void bench_training_step(int batch, const Hardware& hw) {
    const MlpSpec spec({8, 64, 64, 2}, {Activation::Tanh, Activation::Tanh, Activation::Identity});
    GpuMlp net(spec, batch, 1);
    AdamConfig cfg;
    cfg.max_grad_norm = 0.5f;
    GpuAdam opt(net.num_params(), cfg);
    auto x = random_device(size_t(batch) * 8, 3), target = random_device(size_t(batch) * 2, 4);
    device_ptr<float> dy(size_t(batch) * 2);

    const double ms = time_ms([&] {
        net.forward(x.get(), batch);
        mse_loss_grad(net.output(), target.get(), dy.get(), batch * 2);
        net.zero_grad();
        net.backward(dy.get(), batch);
        opt.step(net.params(), net.grads());
    }, 200);

    double macs = 0;
    for (int l = 0; l < spec.num_layers(); ++l) macs += double(spec.in(l)) * spec.out(l);
    const double flops = 3.0 * 2.0 * macs * batch;   // forward + two backward GEMMs per layer
    std::printf("| %6d | %7.3f | %9.2f | %7.0f | %4.1f%% |\n", batch, ms, batch / (ms * 1e-3) * 1e-6,
                flops / (ms * 1e-3) * 1e-9, 100 * flops / (ms * 1e-3) * 1e-9 / hw.peak_gflops);
}

void bench_cpu_reference(int batch) {
    const MlpSpec spec({8, 64, 64, 2}, {Activation::Tanh, Activation::Tanh, Activation::Identity});
    CpuMlp<float> net(spec);
    net.params() = init_params(spec, 1);
    CpuAdam<float> opt(spec.num_params(), AdamConfig{});
    std::vector<float> x(size_t(batch) * 8, 0.5f), dy(size_t(batch) * 2, 0.01f);
    const int steps = 5;
    const auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        net.forward(x.data(), batch);
        net.zero_grad();
        net.backward(dy.data(), batch);
        opt.step(net.params(), net.grads());
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / steps;
    std::printf("| %6d | %7.3f | %9.2f |\n", batch, ms, batch / (ms * 1e-3) * 1e-6);
}

// --- Tensor-core GEMM ---------------------------------------------------------

device_ptr<uint16_t> random_bf16(size_t n, uint32_t seed) {
    std::vector<uint16_t> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = cudarl::bf16::from_float(cudarl::rng::normal(seed, 0, static_cast<uint32_t>(i)));
    device_ptr<uint16_t> d(n);
    d.copy_from_host(h.data(), n);
    return d;
}

// Row-major C = A B^T (bf16 in, fp32 accumulate and out) through column-major cuBLAS.
void cublas_bf16_nt(cublasHandle_t h, int m, int n, int k, const uint16_t* A, const uint16_t* B, float* C) {
    const float one = 1.f, zero = 0.f;
    CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, B, CUDA_R_16BF, k, A, CUDA_R_16BF, k, &zero,
                              C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

void bench_tensor_core(cublasHandle_t handle, double mma_peak_tflops, int m, int n, int k) {
    auto A = random_bf16(size_t(m) * k, 11), B = random_bf16(size_t(n) * k, 12);
    device_ptr<float> C(size_t(m) * n), C_ref(size_t(m) * n);
    const double flops = 2.0 * m * n * k;
    const int    reps  = std::clamp(static_cast<int>(5e11 / flops), 3, 200);
    auto tf = [&](double ms) { return flops / (ms * 1e-3) * 1e-12; };

    const double t_cublas = time_ms([&] { cublas_bf16_nt(handle, m, n, k, A.get(), B.get(), C_ref.get()); }, reps);
    const auto ref = host_copy(C_ref);
    std::printf("| %5d x %5d x %5d | cuBLAS bf16 | %6.1f | %4.0f%% | 100%% | |\n", m, n, k, tf(t_cublas),
                100 * tf(t_cublas) / mma_peak_tflops);
    for (int v = 0; v < kNumTcKernels; ++v) {
        const auto kernel = static_cast<TcKernel>(v);
        const double ms = time_ms([&] { gemm_bf16_nt(kernel, m, n, k, A.get(), k, B.get(), k, C.get(), n); }, reps);
        const auto ours = host_copy(C);
        double max_diff = 0, max_ref = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            max_diff = std::max(max_diff, double(std::fabs(ours[i] - ref[i])));
            max_ref  = std::max(max_ref, double(std::fabs(ref[i])));
        }
        std::printf("| | %-24s | %6.1f | %4.0f%% | %3.0f%% | %s |\n", tc_kernel_name(kernel), tf(ms),
                    100 * tf(ms) / mma_peak_tflops, 100 * t_cublas / ms, max_diff <= 1e-4 * max_ref ? "yes" : "NO");
    }
}

// The v4 tuning sweep: every candidate configuration at several sizes.
void tuning_section(cublasHandle_t handle) {
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    std::printf("### Tensor-core tuning sweep (TFLOP/s, %% of cuBLAS)\n\n| Candidate |");
    const int sizes[] = {2048, 4096, 8192};
    for (int s : sizes) std::printf(" %d^3 |", s);
    std::printf("\n|---|---|---|---|\n");

    struct Size { int s; device_ptr<uint16_t> A, B; device_ptr<float> C, C_ref; double cublas_ms; std::vector<float> ref; };
    std::vector<Size> runs;
    for (int s : sizes) {
        Size r{s, random_bf16(size_t(s) * s, 21), random_bf16(size_t(s) * s, 22), device_ptr<float>(size_t(s) * s),
               device_ptr<float>(size_t(s) * s), 0.0, {}};
        const int reps = std::clamp(static_cast<int>(5e11 / (2.0 * s * s * s)), 3, 50);
        r.cublas_ms = time_ms([&] { cublas_bf16_nt(handle, s, s, s, r.A.get(), r.B.get(), r.C_ref.get()); }, reps);
        r.ref = host_copy(r.C_ref);
        runs.push_back(std::move(r));
    }
    std::printf("| cuBLAS bf16 |");
    for (const auto& r : runs) std::printf(" %.1f |", 2.0 * r.s * r.s * r.s / (r.cublas_ms * 1e-3) * 1e-12);
    std::printf("\n");
    for (int c = 0; c < tuning_candidate_count(); ++c) {
        std::printf("| %s |", tuning_candidate_name(c));
        for (auto& r : runs) {
            const int s = r.s;
            const int reps = std::clamp(static_cast<int>(5e11 / (2.0 * s * s * s)), 3, 50);
            const double ms = time_ms([&] {
                gemm_bf16_nt_candidate(c, s, s, s, r.A.get(), s, r.B.get(), s, r.C.get(), s);
            }, reps);
            const auto ours = host_copy(r.C);
            double max_diff = 0, max_ref = 0;
            for (size_t i = 0; i < r.ref.size(); ++i) {
                max_diff = std::max(max_diff, double(std::fabs(ours[i] - r.ref[i])));
                max_ref  = std::max(max_ref, double(std::fabs(r.ref[i])));
            }
            std::printf(" %.1f (%.0f%%)%s |", 2.0 * s * s * s / (ms * 1e-3) * 1e-12, 100 * r.cublas_ms / ms,
                        max_diff <= 1e-4 * max_ref ? "" : " WRONG");
        }
        std::printf("\n");
    }
    std::printf("\n");
}

// --- Reinforcement learning ---------------------------------------------------

// Trains PPO on CartPole until the mean episode length stays >= 475 (the
// CartPole-v1 "solved" bar) for 3 consecutive updates, printing the curve.
void ppo_curve(const cudarl::rl::PpoConfig& cfg, int max_updates, bool print_every) {
    cudarl::rl::PpoTrainer trainer(cfg);
    const auto start = std::chrono::steady_clock::now();
    int streak = 0;
    for (int u = 0; u < max_updates; ++u) {
        const auto s = trainer.update();
        streak = s.mean_episode_length >= 475.f ? streak + 1 : 0;
        if (print_every && (u % 5 == 0 || streak == 3)) {
            std::printf("| %4d | %8lld | %6.1f | %5.3f | %6.4f | %5.3f | %6.2f | %5.2f M |\n", s.update, s.env_steps,
                        s.mean_episode_length, s.entropy, s.approx_kl, s.clip_fraction, s.update_ms,
                        s.env_steps_per_second * 1e-6);
        }
        if (streak == 3) {
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            // Keep training to check it stays solved (a too-high learning rate can collapse later).
            float worst_after = 500.f;
            for (int extra = 0; extra < 40; ++extra) worst_after = std::min(worst_after, trainer.update().mean_episode_length);
            std::printf("\nseed %u: **solved** after %d updates, %lld env steps, %.2f s; worst mean length over the next 40 updates: %.1f\n",
                        cfg.seed, s.update, s.env_steps, sec, worst_after);
            return;
        }
    }
    std::printf("\nseed %u: not solved within %d updates\n", cfg.seed, max_updates);
}

// key=value arguments override PpoConfig fields, e.g. `nn_bench rl lr=1e-3 minibatches=8`.
cudarl::rl::PpoConfig ppo_config_from_args(int argc, char** argv) {
    cudarl::rl::PpoConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const char* eq = std::strchr(argv[i], '=');
        if (!eq) continue;
        const std::string key(argv[i], static_cast<size_t>(eq - argv[i])), value(eq + 1);
        if (key == "lr") cfg.lr = std::stof(value);
        else if (key == "envs") cfg.num_envs = std::stoi(value);
        else if (key == "steps") cfg.rollout_steps = std::stoi(value);
        else if (key == "epochs") cfg.epochs = std::stoi(value);
        else if (key == "minibatches") cfg.minibatches = std::stoi(value);
        else if (key == "updates") cfg.total_updates = std::stoi(value);
        else if (key == "ent") cfg.ent_coef = std::stof(value);
        else if (key == "anneal") cfg.anneal_lr = value != "0";
        else std::printf("unknown option %s\n", key.c_str());
    }
    return cfg;
}

// Raw environment stepping: the batched CartPole kernel vs the same step
// function in a single-threaded CPU loop (a baseline, not a tuned CPU simulator).
void env_throughput() {
    namespace cp = cudarl::cartpole;
    std::printf("### CartPole environment stepping\n\n| Envs | GPU steps/s | CPU (1 thread) steps/s | Speedup |\n|---|---|---|---|\n");
    for (int n : {1024, 16384, 262144, 1048576}) {
        cp::CartPoleBatch batch(n, 1);
        std::vector<int> h_actions(n);
        for (int i = 0; i < n; ++i) h_actions[i] = static_cast<int>(cudarl::rng::bits(2, 0, i) & 1u);
        device_ptr<int> actions(n);
        actions.copy_from_host(h_actions.data(), n);
        device_ptr<float> r(n), d(n), t(n), fo(size_t(n) * 4), len(n);
        const cp::StepOutputs out{r.get(), d.get(), t.get(), fo.get(), len.get()};
        const double gpu_ms = time_ms([&] { batch.step(actions.get(), out); }, 50);

        std::vector<cp::State> states(n);
        std::vector<uint32_t> episodes(n, 0);
        for (int i = 0; i < n; ++i) states[i] = cp::initial_state(1, i, 0);
        const int cpu_steps = std::max(1, 4'000'000 / n);
        float final_obs[4];
        const auto start = std::chrono::steady_clock::now();
        for (int s = 0; s < cpu_steps; ++s)
            for (int i = 0; i < n; ++i) cp::step(states[i], h_actions[i], 1, i, episodes[i], final_obs);
        const double cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / cpu_steps;
        std::printf("| %7d | %8.2f G | %6.1f M | %5.0fx |\n", n, n / (gpu_ms * 1e-3) * 1e-9, n / (cpu_ms * 1e-3) * 1e-6,
                    cpu_ms / gpu_ms);
    }
    std::printf("\n");
}

void rl_section(cudarl::rl::PpoConfig cfg) {
    env_throughput();
    std::printf("### PPO on CartPole (GPU-resident)\n\n");
    std::printf("%d envs x %d steps per update, %d epochs x %d minibatches, lr %.1e%s over %d updates\n\n",
                cfg.num_envs, cfg.rollout_steps, cfg.epochs, cfg.minibatches, cfg.lr,
                cfg.anneal_lr ? " (annealed)" : "", cfg.total_updates);
    std::printf("| Update | Env steps | Mean length | Entropy | Approx KL | Clip frac | ms/update | Steps/s |\n");
    std::printf("|---|---|---|---|---|---|---|---|\n");
    ppo_curve(cfg, cfg.total_updates, true);
    for (uint32_t seed : {2u, 3u, 4u, 5u}) {
        cfg.seed = seed;
        ppo_curve(cfg, cfg.total_updates, false);
    }
    std::printf("\n");
}

bool wants(int argc, char** argv, const char* section) {
    if (argc < 2) return true;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], section) == 0 || std::strcmp(argv[i], "all") == 0) return true;
    return false;
}

void tensor_core_section(cublasHandle_t handle) {
    const double peak = measure_mma_peak_tflops();
    std::printf("- Tensor cores (bf16 in, fp32 accumulate): **%.1f TFLOP/s measured** (register-only mma.sync loop)\n\n", peak);
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    std::printf("### Tensor-core GEMM, bf16 x bf16 -> fp32 (C = A B^T)\n\n");
    std::printf("| m x n x k | Kernel | TFLOP/s | %% of mma peak | %% of cuBLAS | matches cuBLAS |\n|---|---|---|---|---|---|\n");
    for (int s : {1024, 2048, 4096, 8192}) bench_tensor_core(handle, peak, s, s, s);
    bench_tensor_core(handle, peak, 65536, 64, 64);
    std::printf("\n");
}

// --- Attention ----------------------------------------------------------------

// Row-major C[b] = op(A[b]) op(B[b]) for every batch entry b, through column-major cuBLAS as in cublas_gemm.
void cublas_gemm_batched(cublasHandle_t h, Trans ta, Trans tb, int m, int n, int k, const float* A, int lda,
                         long long strideA, const float* B, int ldb, long long strideB, float* C, int ldc,
                         long long strideC, int batch) {
    const float one = 1.f, zero = 0.f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(h, tb == Trans::N ? CUBLAS_OP_N : CUBLAS_OP_T,
                                           ta == Trans::N ? CUBLAS_OP_N : CUBLAS_OP_T, n, m, k, &one, B, ldb,
                                           strideB, A, lda, strideA, &zero, C, ldc, strideC, batch));
}

// Largest |got - want|, relative to the largest |want|.
double max_rel_error(const std::vector<float>& got, const std::vector<float>& want) {
    double diff = 0.0, ref = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        diff = std::max(diff, std::fabs(double(got[i]) - want[i]));
        ref  = std::max(ref, std::fabs(double(want[i])));
    }
    return ref > 0.0 ? diff / ref : diff;
}

// The fused kernels against the unfused equivalent (tools/attention_baseline.h) at
// 32 heads of head dim 64. Each side is timed on the work it does. The forward is
// 4 N^2 D FLOPs per head for both. The fused backward recomputes the scores and
// does 14 N^2 D; the unfused one keeps them and does 8 N^2 D.
void bench_attention(const Hardware& hw, cublasHandle_t handle, int seq) {
    constexpr int H = 32, D = 64;
    const size_t n = size_t(H) * seq * D, r = size_t(H) * seq, s = size_t(H) * seq * seq;
    const long long nd = static_cast<long long>(seq) * D, ss = static_cast<long long>(seq) * seq;
    const int rows = H * seq;
    const AttentionShape shape{H, seq, D};
    const float scale = 1.f / std::sqrt(float(D));
    const int reps = seq >= 2048 ? 5 : seq >= 512 ? 20 : 100;

    auto Q = random_device(n, 61), K = random_device(n, 62), V = random_device(n, 63), dO = random_device(n, 64);
    device_ptr<float> O(n), lse(r), delta(r), dQ(n), dK(n), dV(n);
    device_ptr<float> S(s), dP(s), uO(n), uDQ(n), uDK(n), uDV(n);

    const auto fused_forward = [&] { attention_forward(shape, Q.get(), K.get(), V.get(), O.get(), lse.get()); };
    const auto fused_backward = [&] {
        attention_backward(shape, Q.get(), K.get(), V.get(), O.get(), lse.get(), dO.get(), dQ.get(), dK.get(),
                           dV.get(), delta.get());
    };
    // S holds the scores, then the probabilities in place, as an unfused framework would.
    const auto unfused_forward = [&] {
        cublas_gemm_batched(handle, Trans::N, Trans::T, seq, seq, D, Q.get(), D, nd, K.get(), D, nd, S.get(), seq, ss, H);
        unfused_softmax_rows(S.get(), rows, seq, scale);
        cublas_gemm_batched(handle, Trans::N, Trans::N, seq, D, seq, S.get(), seq, ss, V.get(), D, nd, uO.get(), D, nd, H);
    };
    // Needs the probabilities in S from unfused_forward().
    const auto unfused_backward = [&] {
        cublas_gemm_batched(handle, Trans::N, Trans::T, seq, seq, D, dO.get(), D, nd, V.get(), D, nd, dP.get(), seq, ss, H);
        unfused_softmax_backward_rows(S.get(), dP.get(), rows, seq, scale);
        cublas_gemm_batched(handle, Trans::T, Trans::N, seq, D, seq, S.get(), seq, ss, dO.get(), D, nd, uDV.get(), D, nd, H);
        cublas_gemm_batched(handle, Trans::T, Trans::N, seq, D, seq, dP.get(), seq, ss, Q.get(), D, nd, uDK.get(), D, nd, H);
        cublas_gemm_batched(handle, Trans::N, Trans::N, seq, D, seq, dP.get(), seq, ss, K.get(), D, nd, uDQ.get(), D, nd, H);
    };

    fused_forward();
    fused_backward();
    unfused_forward();
    unfused_backward();
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    const double err = std::max({max_rel_error(host_copy(O), host_copy(uO)),
                                 max_rel_error(host_copy(dQ), host_copy(uDQ)),
                                 max_rel_error(host_copy(dK), host_copy(uDK)),
                                 max_rel_error(host_copy(dV), host_copy(uDV))});

    const double t_fwd = time_ms(fused_forward, reps), u_fwd = time_ms(unfused_forward, reps);
    const double t_bwd = time_ms(fused_backward, reps), u_bwd = time_ms(unfused_backward, reps);
    const double fwd_flops = 4.0 * H * seq * double(seq) * D;
    const double bwd_flops = 14.0 * H * seq * double(seq) * D;
    const auto tflops = [](double flops, double ms) { return flops / (ms * 1e-3) * 1e-12; };
    std::printf("| %5d | %6.0f | %7.3f | %5.2f | %4.1f%% | %7.3f | %5.2fx | %7.3f | %5.2f | %4.1f%% | %7.3f | %5.2fx | %.1e |\n",
                seq, double(s) * sizeof(float) / (1 << 20), t_fwd, tflops(fwd_flops, t_fwd),
                100 * tflops(fwd_flops, t_fwd) / (hw.peak_gflops * 1e-3), u_fwd, u_fwd / t_fwd, t_bwd,
                tflops(bwd_flops, t_bwd), 100 * tflops(bwd_flops, t_bwd) / (hw.peak_gflops * 1e-3), u_bwd,
                u_bwd / t_bwd, err);
}

void attention_section(const Hardware& hw, cublasHandle_t handle) {
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));   // true fp32 on both sides
    std::printf("### Attention (fp32, 32 heads, head dim 64): fused kernels vs unfused cuBLAS + softmax\n\n");
    std::printf("| Seq | Unfused scores (MiB) | Fused fwd ms | TFLOP/s | %% peak | Unfused fwd ms | Fwd speedup | Fused bwd ms | TFLOP/s | %% peak | Unfused bwd ms | Bwd speedup | Max error vs unfused |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for (int seq : {128, 512, 2048}) bench_attention(hw, handle, seq);
    std::printf("\n");
}

// --- Attention, bf16 tensor cores -----------------------------------------------

// Values rounded to bf16, so the fp32 and bf16 kernels see the same inputs.
std::vector<float> rounded_random(size_t n, uint32_t seed) {
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i)
        h[i] = cudarl::bf16::round_trip(cudarl::rng::normal(seed, 0, static_cast<uint32_t>(i)));
    return h;
}

std::vector<uint16_t> to_bf16_bits(const std::vector<float>& h) {
    std::vector<uint16_t> bits(h.size());
    for (size_t i = 0; i < h.size(); ++i) bits[i] = cudarl::bf16::from_float(h[i]);
    return bits;
}

device_ptr<float> upload(const std::vector<float>& h) {
    device_ptr<float> d(h.size());
    d.copy_from_host(h.data(), h.size());
    return d;
}

device_ptr<uint16_t> upload_bits(const std::vector<uint16_t>& h) {
    device_ptr<uint16_t> d(h.size());
    d.copy_from_host(h.data(), h.size());
    return d;
}

// Row-major C[b] = op(A[b]) op(B[b]) with bf16 inputs and fp32 output, through cuBLAS.
void cublas_bf16_gemm_batched(cublasHandle_t h, Trans ta, Trans tb, int m, int n, int k, const uint16_t* A, int lda,
                              long long strideA, const uint16_t* B, int ldb, long long strideB, float* C, int ldc,
                              long long strideC, int batch) {
    const float one = 1.f, zero = 0.f;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(h, tb == Trans::N ? CUBLAS_OP_N : CUBLAS_OP_T,
                                            ta == Trans::N ? CUBLAS_OP_N : CUBLAS_OP_T, n, m, k, &one, B, CUDA_R_16BF,
                                            ldb, strideB, A, CUDA_R_16BF, lda, strideA, &zero, C, CUDA_R_32F, ldc,
                                            strideC, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// The fused tensor-core forward against the unfused bf16 path: cuBLAS bf16 GEMMs with
// the softmax in between, so the fp32 scores go through global memory. Both are timed
// without the fp32 -> bf16 conversion of Q, K and V, which they share. The unfused
// path is skipped where its scores would not fit (8192 needs 8 GiB for them).
void bench_attention_bf16(cublasHandle_t handle, double peak_tflops, int seq, bool run_unfused) {
    constexpr int H = 32, D = 64;
    const size_t n = size_t(H) * seq * D, r = size_t(H) * seq, s = size_t(H) * seq * seq;
    const long long nd = static_cast<long long>(seq) * D, ss = static_cast<long long>(seq) * seq;
    const int rows = H * seq;
    const AttentionShape shape{H, seq, D};
    const float scale = 1.f / std::sqrt(float(D));
    const int reps = seq >= 8192 ? 3 : seq >= 2048 ? 5 : seq >= 512 ? 20 : 100;

    const auto qh = rounded_random(n, 71), kh = rounded_random(n, 72), vh = rounded_random(n, 73);
    auto Q32 = upload(qh), K32 = upload(kh), V32 = upload(vh);
    auto Qb = upload_bits(to_bf16_bits(qh)), Kb = upload_bits(to_bf16_bits(kh)), Vb = upload_bits(to_bf16_bits(vh));
    device_ptr<float> Of(n), Lf(r), Ob(n), Lb(r);
    device_ptr<float> S(run_unfused ? s : 0), uO(run_unfused ? n : 0);
    device_ptr<uint16_t> P(run_unfused ? s : 0);

    const auto fused_fp32 = [&] { attention_forward(shape, Q32.get(), K32.get(), V32.get(), Of.get(), Lf.get()); };
    const auto fused_bf16 = [&] { attention_forward_bf16(shape, Qb.get(), Kb.get(), Vb.get(), Ob.get(), Lb.get()); };
    const auto unfused_bf16 = [&] {
        cublas_bf16_gemm_batched(handle, Trans::N, Trans::T, seq, seq, D, Qb.get(), D, nd, Kb.get(), D, nd, S.get(),
                                 seq, ss, H);
        unfused_softmax_rows_to_bf16(S.get(), P.get(), rows, seq, scale);
        cublas_bf16_gemm_batched(handle, Trans::N, Trans::N, seq, D, seq, P.get(), seq, ss, Vb.get(), D, nd, uO.get(),
                                 D, nd, H);
    };

    fused_fp32();
    fused_bf16();
    if (run_unfused) unfused_bf16();
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    const double err = max_rel_error(host_copy(Ob), host_copy(Of));
    const double err_unfused = run_unfused ? max_rel_error(host_copy(uO), host_copy(Of)) : 0.0;

    const auto convert_all = [&] {
        convert_to_bf16(Q32.get(), Qb.get(), n);
        convert_to_bf16(K32.get(), Kb.get(), n);
        convert_to_bf16(V32.get(), Vb.get(), n);
    };
    const double t_fused = time_ms(fused_bf16, reps);
    const double t_fp32 = time_ms(fused_fp32, reps);
    const double t_convert = time_ms(convert_all, reps);
    const double flops = 4.0 * H * seq * double(seq) * D;   // QK^T and PV, per head
    const double tflops = flops / (t_fused * 1e-3) * 1e-12;
    if (run_unfused) {
        const double t_unfused = time_ms(unfused_bf16, reps);
        std::printf("| %5d | %7.3f | %6.1f | %4.1f%% | %7.3f | %5.2fx | %7.3f | %7.3f | %.1e | %.1e |\n", seq, t_fused,
                    tflops, 100 * tflops / peak_tflops, t_unfused, t_unfused / t_fused, t_fp32, t_convert, err,
                    err_unfused);
    } else {
        std::printf("| %5d | %7.3f | %6.1f | %4.1f%% | n/a | n/a | %7.3f | %7.3f | %.1e | n/a |\n", seq, t_fused, tflops,
                    100 * tflops / peak_tflops, t_fp32, t_convert, err);
    }
}

void attention_bf16_section(cublasHandle_t handle) {
    const double peak = measure_mma_peak_tflops();
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));   // tensor cores on for the bf16 GEMMs
    std::printf("### Attention, bf16 tensor cores (32 heads, head dim 64): fused vs unfused bf16 cuBLAS + softmax\n\n");
    std::printf("Tensor-core ceiling: %.1f TFLOP/s measured. Attention times exclude the fp32 -> bf16 conversion of Q, K and V, "
                "which is timed separately.\n\n",
                peak);
    std::printf("| Seq | Fused bf16 ms | TFLOP/s | %% of ceiling | Unfused bf16 ms | Speedup | Fused fp32 ms | Convert Q,K,V ms | Error vs fp32 (fused bf16) | Error vs fp32 (unfused bf16) |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|\n");
    for (int seq : {128, 512, 2048}) bench_attention_bf16(handle, peak, seq, true);
    bench_attention_bf16(handle, peak, 8192, false);
    std::printf("\n");
}

void fp32_section(const Hardware& hw, cublasHandle_t handle) {
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));   // true fp32: no TF32 tensor cores
    std::printf("### GEMM (fp32, GFLOP/s)\n\n");
    std::printf("| Case | m x n x k | FLOP/B | Naive | Tiled | %% peak | %% roof | cuBLAS | %% of cuBLAS | splits | matches cuBLAS |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|---|\n");
    const GemmCase cases[] = {
        {"square 512", Trans::N, Trans::N, 512, 512, 512},
        {"square 1024", Trans::N, Trans::N, 1024, 1024, 1024},
        {"square 2048", Trans::N, Trans::N, 2048, 2048, 2048},
        {"square 4096", Trans::N, Trans::N, 4096, 4096, 4096},
        {"MLP fwd  Y = X W^T", Trans::N, Trans::T, 65536, 64, 64},
        {"MLP bwd  dX = dZ W", Trans::N, Trans::N, 65536, 64, 64},
        {"MLP bwd  dW = dZ^T X", Trans::T, Trans::N, 64, 64, 65536},
    };
    for (const auto& c : cases) bench_gemm(hw, handle, c);
    bench_gemm(hw, handle, {"dW, split-K off", Trans::T, Trans::N, 64, 64, 65536}, 1);
    std::printf("\n");
}

void training_section(const Hardware& hw) {
    std::printf("### PPO-sized training step (8 -> 64 -> 64 -> 2, forward + backward + Adam)\n\n");
    std::printf("| Batch | ms/step | M samples/s | GFLOP/s | %% peak |\n|---|---|---|---|---|\n");
    for (int batch : {1024, 4096, 16384, 65536}) bench_training_step(batch, hw);
    std::printf("\nCPU reference (plain loops, single thread; correctness baseline, not tuned):\n\n");
    std::printf("| Batch | ms/step | M samples/s |\n|---|---|---|\n");
    bench_cpu_reference(4096);
}

} // namespace

// Usage: nn_bench [tc] [fp32] [attn] [train] [rl] [tune] [profile]
// No arguments runs tc, fp32, attn, train and rl; `tune` and `profile` only run when named.
int main(int argc, char** argv) {
    const Hardware hw = query_hardware();
    std::printf("## %s\n\n", hw.name.c_str());
    std::printf("- %d SMs x %d fp32 lanes @ %.2f GHz -> **fp32 peak %.1f TFLOP/s**\n", hw.sms, hw.fp32_lanes_per_sm,
                hw.clock_ghz, hw.peak_gflops / 1000);
    std::printf("- Memory bandwidth: %.0f GB/s spec, **%.0f GB/s measured** (device-to-device copy; used for the roofline)\n",
                hw.spec_bandwidth_gbs, hw.copy_bandwidth_gbs);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    if (wants(argc, argv, "tc")) tensor_core_section(handle);
    else std::printf("\n");
    // The sweep is slow, so it only runs when asked for by name. `profile` runs
    // each tensor-core version once at 4096^3, for Nsight Compute.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "tune") == 0) tuning_section(handle);
        if (std::strcmp(argv[i], "profile") == 0) {
            const int s = 4096;
            auto A = random_bf16(size_t(s) * s, 31), B = random_bf16(size_t(s) * s, 32);
            device_ptr<float> C(size_t(s) * s);
            for (int v = 0; v < kNumTcKernels; ++v) {
                gemm_bf16_nt(static_cast<TcKernel>(v), s, s, s, A.get(), s, B.get(), s, C.get(), s);
            }
            CUDA_CHECK_THROW(cudaDeviceSynchronize());
        }
    }
    if (wants(argc, argv, "fp32")) fp32_section(hw, handle);
    if (wants(argc, argv, "attn")) attention_section(hw, handle);
    if (wants(argc, argv, "attn")) attention_bf16_section(handle);
    if (wants(argc, argv, "train")) training_section(hw);
    if (wants(argc, argv, "rl")) rl_section(ppo_config_from_args(argc, argv));
    cublasDestroy(handle);
    return 0;
}
