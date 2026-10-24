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

#include "core/bf16.h"
#include "core/rng.h"
#include "cuda/cuda_memory.h"
#include "nn/adam.h"
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

void rl_section(cudarl::rl::PpoConfig cfg) {
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

// Usage: nn_bench [tc] [fp32] [train] [rl] [tune] [profile]
// No arguments runs tc, fp32, train and rl; `tune` and `profile` only run when named.
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
    if (wants(argc, argv, "train")) training_section(hw);
    if (wants(argc, argv, "rl")) rl_section(ppo_config_from_args(argc, argv));
    cublasDestroy(handle);
    return 0;
}
