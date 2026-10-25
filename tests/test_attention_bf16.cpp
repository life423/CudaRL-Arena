// Tensor-core attention forward vs. the fp64 reference on the bf16 values the kernel
// reads, vs. the fp32 kernel, and the exact one-key case. Also stability for large
// logits, identical keys, bit-identical reruns and invalid shapes.

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "attention_reference.h"
#include "check.h"
#include "core/bf16.h"
#include "gpu_util.h"
#include "nn/attention.h"
#include "nn/attention_bf16.h"

using namespace cudarl::nn;
using namespace cudarl::test::attention;
using cudarl::test::random_floats;
using cudarl::test::to_device;
using cudarl::test::to_host;

namespace {

// One-row problems, a partial last tile, several tiles, and every head size.
const Problem kProblems[] = {
    {1, 1, 16}, {2, 10, 16}, {3, 37, 32}, {2, 65, 64}, {1, 257, 64}, {4, 96, 32},
};

// Tolerance for the bf16 output. P is rounded to bf16 (8 significant bits) before the
// P V product. The worst case measured over the problems above is about 1.5e-3, so
// this leaves a 2x margin.
constexpr double kOutputTol = 3e-3;

std::vector<float> round_bf16(const std::vector<float>& f) {
    std::vector<float> r(f.size());
    for (size_t i = 0; i < f.size(); ++i) r[i] = cudarl::bf16::round_trip(f[i]);
    return r;
}

std::vector<uint16_t> to_bits(const std::vector<float>& f) {
    std::vector<uint16_t> b(f.size());
    for (size_t i = 0; i < f.size(); ++i) b[i] = cudarl::bf16::from_float(f[i]);
    return b;
}

device_ptr<uint16_t> upload_bits(const std::vector<uint16_t>& h) {
    device_ptr<uint16_t> d(h.size());
    d.copy_from_host(h.data(), h.size());
    return d;
}

// Inputs already rounded to bf16, so the fp64 reference sees exactly what the kernel reads.
struct Inputs { std::vector<float> Q, K, V; };
struct Outputs { std::vector<float> O, lse; };

Inputs make_inputs(const Problem& p, uint32_t seed, float scale) {
    const size_t n = elems(p);
    return {round_bf16(random_floats(n, seed, scale)), round_bf16(random_floats(n, seed + 1, scale)),
            round_bf16(random_floats(n, seed + 2, scale))};
}

Tensors as_tensors(const Inputs& in) { return {to_double(in.Q), to_double(in.K), to_double(in.V)}; }

Outputs run_bf16(const Problem& p, const Inputs& in) {
    const AttentionShape shape{p.heads, p.seq, p.dim};
    const size_t n = elems(p), r = rows(p);
    auto Q = upload_bits(to_bits(in.Q)), K = upload_bits(to_bits(in.K)), V = upload_bits(to_bits(in.V));
    device_ptr<float> O(n), lse(r);
    attention_forward_bf16(shape, Q.get(), K.get(), V.get(), O.get(), lse.get());
    return {to_host(O, n), to_host(lse, r)};
}

Outputs run_fp32(const Problem& p, const Inputs& in) {
    const AttentionShape shape{p.heads, p.seq, p.dim};
    const size_t n = elems(p), r = rows(p);
    auto Q = to_device(in.Q), K = to_device(in.K), V = to_device(in.V);
    device_ptr<float> O(n), lse(r);
    attention_forward(shape, Q.get(), K.get(), V.get(), O.get(), lse.get());
    return {to_host(O, n), to_host(lse, r)};
}

} // namespace

GPU_TEST_CASE("attention bf16: matches the fp64 reference on bf16 inputs") {
    uint32_t seed = 100;
    for (const Problem& p : kProblems) {
        const Inputs in = make_inputs(p, seed, 1.f);
        const Outputs got = run_bf16(p, in);
        const Forward f = forward_reference(p, as_tensors(in));
        CHECK_CLOSE(got.O, f.O, kOutputTol);
        CHECK_CLOSE(got.lse, f.lse, 1e-5);   // the normalizer is summed in fp32, so only P V sees the rounding
        seed += 10;
    }
}

GPU_TEST_CASE("attention bf16: agrees with the fp32 kernel") {
    uint32_t seed = 300;
    for (const Problem& p : kProblems) {
        const Inputs in = make_inputs(p, seed, 1.f);
        const Outputs got = run_bf16(p, in);
        const Outputs want = run_fp32(p, in);
        CHECK_CLOSE(got.O, want.O, kOutputTol);
        CHECK_CLOSE(got.lse, want.lse, 1e-5);
        seed += 10;
    }
}

GPU_TEST_CASE("attention bf16: a single key returns its value exactly") {
    // With one key, P = 1 and the sum is 1, so bf16 rounding changes nothing.
    const Problem p{2, 1, 32};
    const Inputs in = make_inputs(p, 800, 1.f);
    const Outputs got = run_bf16(p, in);
    CHECK(got.O == in.V);
}

GPU_TEST_CASE("attention bf16: stable for large logits") {
    const Problem p{2, 65, 64};
    const Inputs in = make_inputs(p, 500, 10.f);
    const Outputs got = run_bf16(p, in);
    const Forward f = forward_reference(p, as_tensors(in));
    CHECK_CLOSE(got.O, f.O, kOutputTol);
    CHECK_CLOSE(got.lse, f.lse, 1e-5);
}

GPU_TEST_CASE("attention bf16: identical keys give the mean of the values") {
    const Problem p{2, 37, 32};
    const size_t N = static_cast<size_t>(p.seq), D = static_cast<size_t>(p.dim);
    Inputs in = make_inputs(p, 900, 1.f);
    for (size_t h = 0; h < static_cast<size_t>(p.heads); ++h)
        for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < D; ++d)
                in.K[(h * N + j) * D + d] = cudarl::bf16::round_trip(static_cast<float>(h + d) * 0.1f);

    const Outputs got = run_bf16(p, in);
    Vec want(elems(p), 0.0);
    for (size_t h = 0; h < static_cast<size_t>(p.heads); ++h)
        for (size_t i = 0; i < N; ++i)
            for (size_t d = 0; d < D; ++d) {
                double sum = 0.0;
                for (size_t j = 0; j < N; ++j) sum += in.V[(h * N + j) * D + d];
                want[(h * N + i) * D + d] = sum / static_cast<double>(N);
            }
    CHECK_CLOSE(got.O, want, kOutputTol);
}

GPU_TEST_CASE("attention bf16: bit-identical run to run") {
    const Problem p{3, 100, 64};
    const Inputs in = make_inputs(p, 700, 1.f);
    const Outputs first = run_bf16(p, in);
    for (int run = 1; run < 3; ++run) {
        const Outputs again = run_bf16(p, in);
        CHECK(again.O == first.O);
        CHECK(again.lse == first.lse);
    }
}

TEST_CASE("attention bf16: rejects invalid shapes and null pointers") {
    auto rejects = [](AttentionShape s) {
        try {
            attention_forward_bf16(s, nullptr, nullptr, nullptr, nullptr, nullptr);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    CHECK(rejects({1, 4, 8}));    // dim is not 16, 32 or 64
    CHECK(rejects({0, 4, 16}));   // no heads
    CHECK(rejects({1, 0, 16}));   // no rows
    CHECK(rejects({1, 4, 16}));   // valid shape, null pointers
}
