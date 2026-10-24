// Benchmarks the hand-written NN kernels against the hardware roofline and
// against cuBLAS. cuBLAS is linked into this tool only, as a yardstick; the
// library never uses it.
//
// Output is markdown, ready to paste into the README.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "core/rng.h"
#include "cuda/cuda_memory.h"
#include "nn/adam.h"
#include "nn/gemm.h"
#include "nn/loss.h"
#include "nn/mlp_cpu.h"
#include "nn/mlp_gpu.h"

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

template <typename F>
double time_ms(F&& fn, int reps) {
    for (int i = 0; i < 3; ++i) fn();   // warm up
    cudaEvent_t start, stop;
    CUDA_CHECK_THROW(cudaEventCreate(&start));
    CUDA_CHECK_THROW(cudaEventCreate(&stop));
    CUDA_CHECK_THROW(cudaEventRecord(start));
    for (int i = 0; i < reps; ++i) fn();
    CUDA_CHECK_THROW(cudaEventRecord(stop));
    CUDA_CHECK_THROW(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK_THROW(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / reps;
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

} // namespace

int main() {
    const Hardware hw = query_hardware();
    std::printf("## %s\n\n", hw.name.c_str());
    std::printf("- %d SMs x %d fp32 lanes @ %.2f GHz -> **fp32 peak %.1f TFLOP/s**\n", hw.sms, hw.fp32_lanes_per_sm,
                hw.clock_ghz, hw.peak_gflops / 1000);
    std::printf("- Memory bandwidth: %.0f GB/s spec, **%.0f GB/s measured** (device-to-device copy; used for the roofline)\n\n",
                hw.spec_bandwidth_gbs, hw.copy_bandwidth_gbs);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
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

    std::printf("\n### PPO-sized training step (8 -> 64 -> 64 -> 2, forward + backward + Adam)\n\n");
    std::printf("| Batch | ms/step | M samples/s | GFLOP/s | %% peak |\n|---|---|---|---|---|\n");
    for (int batch : {1024, 4096, 16384, 65536}) bench_training_step(batch, hw);

    std::printf("\nCPU reference (plain loops, single thread; correctness baseline, not tuned):\n\n");
    std::printf("| Batch | ms/step | M samples/s |\n|---|---|---|\n");
    bench_cpu_reference(4096);

    cublasDestroy(handle);
    return 0;
}
