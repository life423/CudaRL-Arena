// GEMM kernels vs. a double-precision CPU product, for every transpose
// combination, both kernels, and every epilogue, on sizes that don't divide
// the 64x64x16 tile (including 1x1 and K=257).

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "gpu_util.h"
#include "nn/gemm.h"

using namespace cudarl::nn;
using cudarl::test::random_floats;
using cudarl::test::to_device;
using cudarl::test::to_host;

namespace {

struct Shape { int m, n, k; };

const Shape kShapes[] = {
    {1, 1, 1}, {7, 13, 5}, {64, 64, 64}, {65, 33, 17}, {130, 70, 3}, {1, 300, 257}, {300, 1, 64}, {129, 129, 129},
};

// Reference: op(A) op(B) in double, plus the epilogue.
std::vector<float> reference(Trans ta, Trans tb, const Shape& s, const std::vector<float>& A,
                             const std::vector<float>& B, const std::vector<float>& C0,
                             Epilogue ep, const std::vector<float>& bias) {
    std::vector<float> C(static_cast<size_t>(s.m) * s.n);
    for (int r = 0; r < s.m; ++r) {
        for (int c = 0; c < s.n; ++c) {
            double acc = 0.0;
            for (int kk = 0; kk < s.k; ++kk) {
                const double a = ta == Trans::N ? A[r * s.k + kk] : A[kk * s.m + r];
                const double b = tb == Trans::N ? B[kk * s.n + c] : B[c * s.k + kk];
                acc += a * b;
            }
            const size_t i = static_cast<size_t>(r) * s.n + c;
            switch (ep.kind) {
                case Epilogue::Kind::Accumulate:     C[i] = static_cast<float>(C0[i] + acc); break;
                case Epilogue::Kind::BiasActivation: C[i] = static_cast<float>(activate(ep.activation, acc + bias[c])); break;
                default:                             C[i] = static_cast<float>(acc); break;
            }
        }
    }
    return C;
}

void check_all_shapes(GemmKernel kernel, Trans ta, Trans tb, Epilogue::Kind kind, Activation act) {
    uint32_t seed = 100;
    for (const Shape& s : kShapes) {
        const auto A    = random_floats(static_cast<size_t>(s.m) * s.k, seed++);
        const auto B    = random_floats(static_cast<size_t>(s.k) * s.n, seed++);
        const auto C0   = random_floats(static_cast<size_t>(s.m) * s.n, seed++);
        const auto bias = random_floats(static_cast<size_t>(s.n), seed++);
        auto dA = to_device(A), dB = to_device(B), dC = to_device(C0), dbias = to_device(bias);

        Epilogue ep;
        ep.kind       = kind;
        ep.activation = act;
        ep.bias       = dbias.get();
        const int lda = ta == Trans::N ? s.k : s.m;
        const int ldb = tb == Trans::N ? s.n : s.k;
        gemm(kernel, ta, tb, s.m, s.n, s.k, dA.get(), lda, dB.get(), ldb, dC.get(), s.n, ep);

        const auto got  = to_host(dC, C0.size());
        const auto want = reference(ta, tb, s, A, B, C0, ep, bias);
        const double err = cudarl::test::normalized_max_error(got, want);
        if (!(err <= 1e-5)) {
            char msg[160];
            std::snprintf(msg, sizeof msg, "m=%d n=%d k=%d: error %.3g", s.m, s.n, s.k, err);
            CHECK_MSG(false, msg);
        }
    }
}

void check_all_transposes(GemmKernel kernel, Epilogue::Kind kind, Activation act = Activation::Identity) {
    for (Trans ta : {Trans::N, Trans::T})
        for (Trans tb : {Trans::N, Trans::T}) check_all_shapes(kernel, ta, tb, kind, act);
}

} // namespace

TEST_CASE("gemm: tiled, store, all transposes") {
    check_all_transposes(GemmKernel::Tiled, Epilogue::Kind::Store);
}

TEST_CASE("gemm: tiled, accumulate, all transposes") {
    check_all_transposes(GemmKernel::Tiled, Epilogue::Kind::Accumulate);
}

TEST_CASE("gemm: tiled, fused bias + tanh / relu") {
    check_all_transposes(GemmKernel::Tiled, Epilogue::Kind::BiasActivation, Activation::Tanh);
    check_all_transposes(GemmKernel::Tiled, Epilogue::Kind::BiasActivation, Activation::ReLU);
}

TEST_CASE("gemm: naive, all transposes and epilogues") {
    check_all_transposes(GemmKernel::Naive, Epilogue::Kind::Store);
    check_all_transposes(GemmKernel::Naive, Epilogue::Kind::Accumulate);
    check_all_transposes(GemmKernel::Naive, Epilogue::Kind::BiasActivation, Activation::Tanh);
}

TEST_CASE("gemm: split-K matches reference for any split count") {
    // The dW shape: [64 x 48] = dZ^T [64 x 5000] * X [5000 x 48].
    const Shape s{64, 48, 5000};
    const auto A  = random_floats(static_cast<size_t>(s.k) * s.m, 7);   // stored [k x m] (ta = T)
    const auto B  = random_floats(static_cast<size_t>(s.k) * s.n, 8);   // stored [k x n]
    const auto C0 = random_floats(static_cast<size_t>(s.m) * s.n, 9);
    auto dA = to_device(A), dB = to_device(B);
    device_ptr<float> ws(static_cast<size_t>(kMaxSplitK) * s.m * s.n);

    const auto want = reference(Trans::T, Trans::N, s, A, B, C0, Epilogue::accumulate(), {});
    for (int splits : {1, 2, 3, 7, 16, 64}) {
        auto dC = to_device(C0);
        gemm_split_k(Trans::T, Trans::N, s.m, s.n, s.k, dA.get(), s.m, dB.get(), s.n, dC.get(), s.n,
                     splits, ws.get(), Epilogue::accumulate());
        CHECK_CLOSE(to_host(dC, C0.size()), want, 1e-5);
    }
}

TEST_CASE("gemm: split-K is bit-identical run to run") {
    const Shape s{64, 64, 20000};
    auto dA = to_device(random_floats(static_cast<size_t>(s.k) * s.m, 17));
    auto dB = to_device(random_floats(static_cast<size_t>(s.k) * s.n, 18));
    device_ptr<float> ws(static_cast<size_t>(kMaxSplitK) * s.m * s.n);
    std::vector<float> first;
    for (int run = 0; run < 3; ++run) {
        device_ptr<float> dC(static_cast<size_t>(s.m) * s.n);
        gemm_split_k(Trans::T, Trans::N, s.m, s.n, s.k, dA.get(), s.m, dB.get(), s.n, dC.get(), s.n,
                     32, ws.get());
        const auto got = to_host(dC, static_cast<size_t>(s.m) * s.n);
        if (run == 0) first = got;
        else CHECK(got == first);
    }
}

TEST_CASE("gemm: split_k_count heuristics") {
    CHECK(split_k_count(64, 64, 65536, 48) == 64);        // one tile: split to the cap
    CHECK(split_k_count(64, 64, 100, 48) == 1);           // too short to split
    CHECK(split_k_count(4096, 4096, 4096, 48) == 1);      // plenty of tiles already
}

TEST_CASE("gemm: k = 0 gives zero product") {
    auto dC = to_device(std::vector<float>(6, 5.f));
    gemm(GemmKernel::Tiled, Trans::N, Trans::N, 2, 3, 0, nullptr, 1, nullptr, 3, dC.get(), 3);
    for (float v : to_host(dC, 6)) CHECK(v == 0.f);
}
