// Fused attention vs. a double-precision CPU reference: forward output and
// logsumexp, backward gradients, the reference's own gradients against finite
// differences, large logits, a closed-form case, and bit-identical reruns.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "check.h"
#include "gpu_util.h"
#include "nn/attention.h"

using namespace cudarl::nn;
using cudarl::test::random_floats;
using cudarl::test::to_device;
using cudarl::test::to_host;

namespace {

using Vec = std::vector<double>;

struct Problem { int heads, seq, dim; };

// One-row problems, a partial last tile, several tiles, and every head size.
const Problem kProblems[] = {
    {1, 1, 16}, {2, 10, 16}, {3, 37, 32}, {2, 65, 64}, {1, 257, 64}, {4, 96, 32},
};

size_t elems(const Problem& p) { return static_cast<size_t>(p.heads) * p.seq * p.dim; }
size_t rows(const Problem& p)  { return static_cast<size_t>(p.heads) * p.seq; }

Vec to_double(const std::vector<float>& f) { return Vec(f.begin(), f.end()); }
Vec random_vec(size_t n, uint32_t seed) { return to_double(random_floats(n, seed)); }

// Q, K, V, each [heads x seq x dim].
struct Tensors { Vec Q, K, V; };
struct Forward { Vec O, lse; };
struct Gradients { Vec dQ, dK, dV; };

// Forward in double. Scores are shifted by the row max before exponentiating.
Forward forward_reference(const Problem& p, const Tensors& t) {
    const int N = p.seq, D = p.dim;
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    Forward out{Vec(elems(p), 0.0), Vec(rows(p), 0.0)};
    Vec s(N);
    for (int h = 0; h < p.heads; ++h) {
        for (int i = 0; i < N; ++i) {
            const size_t row = static_cast<size_t>(h) * N + i;
            const double* q = &t.Q[row * D];
            double mx = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < N; ++j) {
                const double* k = &t.K[(static_cast<size_t>(h) * N + j) * D];
                double dot = 0.0;
                for (int d = 0; d < D; ++d) dot += q[d] * k[d];
                s[j] = scale * dot;
                mx = std::max(mx, s[j]);
            }
            double sum = 0.0;
            for (int j = 0; j < N; ++j) {
                s[j] = std::exp(s[j] - mx);
                sum += s[j];
            }
            out.lse[row] = mx + std::log(sum);
            double* o = &out.O[row * D];
            for (int j = 0; j < N; ++j) {
                const double w = s[j] / sum;
                const double* v = &t.V[(static_cast<size_t>(h) * N + j) * D];
                for (int d = 0; d < D; ++d) o[d] += w * v[d];
            }
        }
    }
    return out;
}

// Gradients in double, from the standard attention backward:
//   P = softmax(S),  dP = dO V^T,  dS = P (dP - rowsum(P dP)),
//   dQ = scale dS K,  dK = scale dS^T Q,  dV = P^T dO.
Gradients backward_reference(const Problem& p, const Tensors& t, const Vec& dO) {
    const int N = p.seq, D = p.dim;
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    const Forward f = forward_reference(p, t);
    Gradients g{Vec(elems(p), 0.0), Vec(elems(p), 0.0), Vec(elems(p), 0.0)};
    Vec P(N), dP(N);
    for (int h = 0; h < p.heads; ++h) {
        for (int i = 0; i < N; ++i) {
            const size_t row = static_cast<size_t>(h) * N + i;
            const double* q  = &t.Q[row * D];
            const double* go = &dO[row * D];
            for (int j = 0; j < N; ++j) {
                const size_t kr = static_cast<size_t>(h) * N + j;
                const double* k = &t.K[kr * D];
                const double* v = &t.V[kr * D];
                double dot = 0.0, dpv = 0.0;
                for (int d = 0; d < D; ++d) {
                    dot += q[d] * k[d];
                    dpv += go[d] * v[d];
                }
                P[j] = std::exp(scale * dot - f.lse[row]);
                dP[j] = dpv;
            }
            double delta = 0.0;
            for (int j = 0; j < N; ++j) delta += P[j] * dP[j];
            for (int j = 0; j < N; ++j) {
                const size_t kr = static_cast<size_t>(h) * N + j;
                const double ds = P[j] * (dP[j] - delta);
                for (int d = 0; d < D; ++d) {
                    g.dQ[row * D + d] += scale * ds * t.K[kr * D + d];
                    g.dK[kr * D + d] += scale * ds * t.Q[row * D + d];
                    g.dV[kr * D + d] += P[j] * go[d];
                }
            }
        }
    }
    return g;
}

struct Inputs { std::vector<float> Q, K, V, dO; };
struct Outputs { std::vector<float> O, lse, dQ, dK, dV; };

Inputs make_inputs(const Problem& p, uint32_t seed, float scale) {
    const size_t n = elems(p);
    return {random_floats(n, seed, scale), random_floats(n, seed + 1, scale), random_floats(n, seed + 2, scale),
            random_floats(n, seed + 3)};
}

Outputs run_gpu(const Problem& p, const Inputs& in) {
    const AttentionShape shape{p.heads, p.seq, p.dim};
    const size_t n = elems(p), r = rows(p);
    auto Q = to_device(in.Q), K = to_device(in.K), V = to_device(in.V), dO = to_device(in.dO);
    device_ptr<float> O(n), lse(r), dQ(n), dK(n), dV(n), delta(r);
    attention_forward(shape, Q.get(), K.get(), V.get(), O.get(), lse.get());
    attention_backward(shape, Q.get(), K.get(), V.get(), O.get(), lse.get(), dO.get(), dQ.get(), dK.get(),
                       dV.get(), delta.get());
    return {to_host(O, n), to_host(lse, r), to_host(dQ, n), to_host(dK, n), to_host(dV, n)};
}

void check_against_reference(const Problem& p, uint32_t seed, float scale) {
    const Inputs in = make_inputs(p, seed, scale);
    const Outputs got = run_gpu(p, in);
    const Tensors t{to_double(in.Q), to_double(in.K), to_double(in.V)};
    const Forward f = forward_reference(p, t);
    const Gradients g = backward_reference(p, t, to_double(in.dO));
    CHECK_CLOSE(got.O, f.O, 1e-5);
    CHECK_CLOSE(got.lse, f.lse, 1e-5);
    CHECK_CLOSE(got.dQ, g.dQ, 1e-4);
    CHECK_CLOSE(got.dK, g.dK, 1e-4);
    CHECK_CLOSE(got.dV, g.dV, 1e-4);
}

} // namespace

// The GPU results are judged against the reference, so the reference's gradients
// are checked against finite differences of its own forward pass first.
TEST_CASE("attention: reference gradients match finite differences") {
    const Problem p{2, 5, 4};
    const size_t n = elems(p);
    const Tensors t{random_vec(n, 1), random_vec(n, 2), random_vec(n, 3)};
    const Vec G = random_vec(n, 4);
    const Gradients analytic = backward_reference(p, t, G);

    auto loss = [&](const Tensors& x) {
        const Forward f = forward_reference(p, x);
        double sum = 0.0;
        for (size_t e = 0; e < n; ++e) sum += f.O[e] * G[e];
        return sum;
    };
    const double h = 1e-6;
    Vec numeric;
    for (Vec Tensors::*member : {&Tensors::Q, &Tensors::K, &Tensors::V}) {
        for (size_t e = 0; e < n; ++e) {
            Tensors plus = t, minus = t;
            (plus.*member)[e] += h;
            (minus.*member)[e] -= h;
            numeric.push_back((loss(plus) - loss(minus)) / (2 * h));
        }
    }
    Vec want = analytic.dQ;
    want.insert(want.end(), analytic.dK.begin(), analytic.dK.end());
    want.insert(want.end(), analytic.dV.begin(), analytic.dV.end());
    CHECK_CLOSE(numeric, want, 1e-6);
}

GPU_TEST_CASE("attention: matches fp64 reference for every head size and ragged tile") {
    uint32_t seed = 100;
    for (const Problem& p : kProblems) {
        check_against_reference(p, seed, 1.f);
        seed += 10;
    }
}

GPU_TEST_CASE("attention: stable for large logits") {
    // Scores of order 10^4 would overflow exp() without the running max.
    check_against_reference({2, 65, 64}, 500, 10.f);
}

GPU_TEST_CASE("attention: identical keys give the mean of the values") {
    const Problem p{2, 37, 32};
    const size_t N = static_cast<size_t>(p.seq), D = static_cast<size_t>(p.dim);
    Inputs in = make_inputs(p, 900, 1.f);
    for (size_t h = 0; h < static_cast<size_t>(p.heads); ++h)
        for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < D; ++d) in.K[(h * N + j) * D + d] = static_cast<float>(h + d) * 0.1f;

    const Outputs got = run_gpu(p, in);
    Vec want(elems(p), 0.0);
    for (size_t h = 0; h < static_cast<size_t>(p.heads); ++h)
        for (size_t i = 0; i < N; ++i)
            for (size_t d = 0; d < D; ++d) {
                double sum = 0.0;
                for (size_t j = 0; j < N; ++j) sum += in.V[(h * N + j) * D + d];
                want[(h * N + i) * D + d] = sum / static_cast<double>(N);
            }
    CHECK_CLOSE(got.O, want, 1e-5);
}

GPU_TEST_CASE("attention: forward and backward are bit-identical run to run") {
    const Problem p{3, 100, 64};
    const Inputs in = make_inputs(p, 700, 1.f);
    const Outputs first = run_gpu(p, in);
    for (int run = 1; run < 3; ++run) {
        const Outputs again = run_gpu(p, in);
        CHECK(again.O == first.O);
        CHECK(again.lse == first.lse);
        CHECK(again.dQ == first.dQ);
        CHECK(again.dK == first.dK);
        CHECK(again.dV == first.dV);
    }
}

TEST_CASE("attention: rejects invalid shapes and null pointers") {
    auto rejects = [](AttentionShape s) {
        try {
            attention_forward(s, nullptr, nullptr, nullptr, nullptr, nullptr);
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
