// Tensor-core GEMM tests. bf16 x bf16 products are exact in fp32 (8-bit x
// 8-bit significands fit in 24 bits), so against an fp64 reference computed on
// the same bf16-rounded inputs the only error left is summation order. That
// allows a tight tolerance, so fragment-layout and indexing bugs can't hide
// behind "bf16 is imprecise".

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "check.h"
#include "core/bf16.h"
#include "gpu_util.h"
#include "nn/gemm_bf16.h"

using namespace cudarl;
using namespace cudarl::nn;
using cudarl::test::random_floats;

namespace {

struct Case { int m, n, k; };

const Case kCases[] = {
    {1, 1, 8}, {7, 13, 8}, {16, 8, 16}, {128, 128, 32}, {129, 130, 40},
    {300, 257, 264}, {64, 1000, 72}, {1000, 64, 136}, {512, 512, 512},
};

std::vector<uint16_t> to_bf16(const std::vector<float>& v) {
    std::vector<uint16_t> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = bf16::from_float(v[i]);
    return out;
}

device_ptr<uint16_t> upload(const std::vector<uint16_t>& h) {
    device_ptr<uint16_t> d(h.size());
    d.copy_from_host(h.data(), h.size());
    return d;
}

// C = A B^T in double from bf16 inputs (or from the original fp32 values).
template <typename GetA, typename GetB>
std::vector<double> reference(const Case& c, int lda, int ldb, GetA a, GetB b) {
    std::vector<double> C(static_cast<size_t>(c.m) * c.n);
    for (int i = 0; i < c.m; ++i)
        for (int j = 0; j < c.n; ++j) {
            double s = 0.0;
            for (int kk = 0; kk < c.k; ++kk) s += a(i * lda + kk) * b(j * ldb + kk);
            C[static_cast<size_t>(i) * c.n + j] = s;
        }
    return C;
}

struct Problem {
    Case c;
    int lda, ldb, ldc;
    std::vector<float> a32, b32;
    std::vector<uint16_t> a16, b16;
    device_ptr<uint16_t> dA, dB;

    Problem(Case shape, uint32_t seed, int pad_a = 0, int pad_b = 0, int pad_c = 0)
        : c(shape), lda(shape.k + pad_a), ldb(shape.k + pad_b), ldc(shape.n + pad_c),
          a32(random_floats(static_cast<size_t>(shape.m) * lda, seed)),
          b32(random_floats(static_cast<size_t>(shape.n) * ldb, seed + 1)),
          a16(to_bf16(a32)), b16(to_bf16(b32)), dA(upload(a16)), dB(upload(b16)) {}

    std::vector<float> run(TcKernel kernel) {
        device_ptr<float> dC(static_cast<size_t>(c.m) * ldc);
        CUDA_CHECK_THROW(cudaMemset(dC.get(), 0xff, dC.size_bytes()));   // NaN fill: unwritten outputs show up
        gemm_bf16_nt(kernel, c.m, c.n, c.k, dA.get(), lda, dB.get(), ldb, dC.get(), ldc);
        const auto full = cudarl::test::to_host(dC, dC.size());
        std::vector<float> packed(static_cast<size_t>(c.m) * c.n);
        for (int i = 0; i < c.m; ++i)
            for (int j = 0; j < c.n; ++j) packed[static_cast<size_t>(i) * c.n + j] = full[static_cast<size_t>(i) * ldc + j];
        return packed;
    }

    std::vector<double> bf16_reference() const {
        return reference(c, lda, ldb, [&](int i) { return double(bf16::to_float(a16[i])); },
                         [&](int i) { return double(bf16::to_float(b16[i])); });
    }
};

void check_kernel_on_all_cases(TcKernel kernel) {
    uint32_t seed = 900;
    for (const Case& c : kCases) {
        Problem p(c, seed += 2);
        const auto got = p.run(kernel);
        const double err = cudarl::test::normalized_max_error(got, p.bf16_reference());
        if (!(err <= 1e-5)) {
            char msg[200];
            std::snprintf(msg, sizeof msg, "%s m=%d n=%d k=%d: error %.3g", tc_kernel_name(kernel), c.m, c.n, c.k, err);
            CHECK_MSG(false, msg);
        }
    }
}

} // namespace

TEST_CASE("bf16: conversion rounds to nearest even") {
    CHECK(bf16::from_float(1.0f) == 0x3f80);
    CHECK(bf16::from_float(-2.0f) == 0xc000);
    CHECK(bf16::from_float(0.0f) == 0x0000);
    CHECK(bf16::from_float(-0.0f) == 0x8000);
    // 1 + 2^-8 is exactly halfway between bf16 1.0 and 1.0078125: ties go to even (1.0).
    CHECK(bf16::from_float(bf16::bits_float(0x3f808000u)) == 0x3f80);
    // 1.0078125 + 2^-8 is halfway again: rounds to even, i.e. up to 0x3f82.
    CHECK(bf16::from_float(bf16::bits_float(0x3f818000u)) == 0x3f82);
    CHECK(bf16::from_float(bf16::bits_float(0x3f808001u)) == 0x3f81);   // just above halfway rounds up
    CHECK(bf16::from_float(std::numeric_limits<float>::infinity()) == 0x7f80);
    CHECK(bf16::from_float(3.4e38f) == 0x7f80);                          // overflows to +inf
    const uint16_t nan = bf16::from_float(std::numeric_limits<float>::quiet_NaN());
    CHECK((nan & 0x7f80) == 0x7f80 && (nan & 0x007f) != 0);
    for (float v : {0.1f, -3.5f, 1e-20f, 65504.f}) {
        CHECK(std::fabs(bf16::round_trip(v) - v) <= std::fabs(v) * (1.f / 256));   // half an ulp at 7 bits
    }
}

GPU_TEST_CASE("bf16: device conversion matches host bit for bit") {
    auto values = random_floats(100000, 5, 100.f);
    values.push_back(std::numeric_limits<float>::infinity());
    values.push_back(-std::numeric_limits<float>::infinity());
    values.push_back(bf16::bits_float(0x3f808000u));
    auto d_src = cudarl::test::to_device(values);
    device_ptr<uint16_t> d_dst(values.size());
    convert_to_bf16(d_src.get(), d_dst.get(), values.size());
    std::vector<uint16_t> got(values.size());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    d_dst.copy_to_host(got.data(), got.size());
    CHECK(got == to_bf16(values));
}

GPU_TEST_CASE("gemm_bf16: v0 WMMA exact vs bf16 reference") { check_kernel_on_all_cases(TcKernel::Wmma); }
GPU_TEST_CASE("gemm_bf16: v1 mma.sync exact vs bf16 reference") { check_kernel_on_all_cases(TcKernel::MmaSync); }
GPU_TEST_CASE("gemm_bf16: v2 pipelined exact vs bf16 reference") { check_kernel_on_all_cases(TcKernel::Pipelined); }
GPU_TEST_CASE("gemm_bf16: v3 swizzled exact vs bf16 reference") { check_kernel_on_all_cases(TcKernel::Swizzled); }
GPU_TEST_CASE("gemm_bf16: v4 tuned exact vs bf16 reference") { check_kernel_on_all_cases(TcKernel::Tuned); }

GPU_TEST_CASE("gemm_bf16: padded leading dimensions, odd ldc") {
    for (int v = 0; v < kNumTcKernels; ++v) {
        Problem p({70, 45, 48}, 77, /*pad_a=*/8, /*pad_b=*/16, /*pad_c=*/3);
        CHECK_CLOSE(p.run(static_cast<TcKernel>(v)), p.bf16_reference(), 1e-5);
    }
}

GPU_TEST_CASE("gemm_bf16: bit-identical run to run") {
    for (int v = 0; v < kNumTcKernels; ++v) {
        Problem p({300, 257, 264}, 88);
        const auto first = p.run(static_cast<TcKernel>(v));
        CHECK(p.run(static_cast<TcKernel>(v)) == first);
    }
}

GPU_TEST_CASE("gemm_bf16: quantization error vs fp32 inputs is bf16-sized") {
    // Not a correctness bound but the trade-off made explicit: rounding inputs
    // to bf16 (7-bit mantissa) costs ~2^-9 relative per value, far more than
    // the fp32 accumulation error measured above.
    Problem p({256, 256, 256}, 99);
    const auto exact = reference(p.c, p.lda, p.ldb, [&](int i) { return double(p.a32[i]); },
                                 [&](int i) { return double(p.b32[i]); });
    const double quant = cudarl::test::normalized_max_error(p.run(TcKernel::Tuned), exact);
    std::printf("    bf16 input quantization error vs fp32 inputs: %.2e (normalized max)\n", quant);
    CHECK(quant > 1e-4 && quant < 2e-2);
}

TEST_CASE("gemm_bf16: rejects k, lda, ldb not multiples of 8") {
    auto throws = [](int k, int lda, int ldb) {
        try {
            gemm_bf16_nt(TcKernel::Tuned, 4, 4, k, nullptr, lda, nullptr, ldb, nullptr, 4);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    CHECK(throws(12, 16, 16));
    CHECK(throws(16, 12, 16));
    CHECK(throws(16, 16, 20));
    CHECK(throws(-8, 16, 16));
}
