#ifndef CUDARL_TESTS_ATTENTION_REFERENCE_H
#define CUDARL_TESTS_ATTENTION_REFERENCE_H

// Double-precision CPU attention: the reference that the fp32 and bf16 attention
// tests compare the GPU results against.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "gpu_util.h"

namespace cudarl::test::attention {

using Vec = std::vector<double>;

struct Problem { int heads, seq, dim; };

inline size_t elems(const Problem& p) { return static_cast<size_t>(p.heads) * p.seq * p.dim; }
inline size_t rows(const Problem& p)  { return static_cast<size_t>(p.heads) * p.seq; }

inline Vec to_double(const std::vector<float>& f) { return Vec(f.begin(), f.end()); }
inline Vec random_vec(size_t n, uint32_t seed) { return to_double(random_floats(n, seed)); }

// Q, K, V, each [heads x seq x dim].
struct Tensors { Vec Q, K, V; };
struct Forward { Vec O, lse; };
struct Gradients { Vec dQ, dK, dV; };

// Forward in double. Scores are shifted by the row max before exponentiating.
inline Forward forward_reference(const Problem& p, const Tensors& t) {
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
inline Gradients backward_reference(const Problem& p, const Tensors& t, const Vec& dO) {
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

} // namespace cudarl::test::attention

#endif // CUDARL_TESTS_ATTENTION_REFERENCE_H
