#include "nn/attention.h"

#include <cmath>
#include <initializer_list>
#include <stdexcept>

#include "cuda/cuda_memory.h"

namespace cudarl::nn {

namespace {

// A block works on one kTile-row tile of one head: query rows for the forward and
// dQ kernels, key rows for dK and dV. Its kGrid x kGrid threads each own a kMicro x
// kMicro micro-tile of every kTile x kTile score tile, so each operand loaded from
// shared memory feeds several multiply-adds.
constexpr int kTile     = 64;
constexpr int kGrid     = 16;
constexpr int kThreads  = kGrid * kGrid;      // 256
constexpr int kMicro    = kTile / kGrid;      // 4 rows and 4 columns per thread per tile
constexpr int kRowLanes = kThreads / kTile;   // threads that share one row in the softmax
constexpr int kRowCols  = kTile / kRowLanes;  // columns of a row handled by each of them

// Shared memory per block, in bytes. Tiles are padded by one column so that the
// threads reading a column of a tile use different banks.
template <int D>
constexpr size_t kForwardSmem = sizeof(float) * (3 * kTile * (D + 1) + kTile * (kTile + 1) + 3 * kTile);
template <int D>
constexpr size_t kBackwardSmem = sizeof(float) * (4 * kTile * (D + 1) + kTile * (kTile + 1) + 2 * kTile);

// -infinity as a float. The INFINITY macro is a double constant, which nvcc rejects here.
__device__ __forceinline__ float neg_inf() { return __uint_as_float(0xff800000u); }

// Copies rows [row0, row0 + kTile) of an [N x D] matrix into a shared tile, zero-filling
// rows past N.
template <int D>
__device__ __forceinline__ void load_tile(float (*dst)[D + 1], const float* src, int row0, int N) {
    const int tid = threadIdx.x;
    for (int i = tid; i < kTile * D; i += kThreads) {
        const int r = i / D, d = i % D;
        const int row = row0 + r;
        dst[r][d] = row < N ? src[static_cast<size_t>(row) * D + d] : 0.f;
    }
}

__device__ __forceinline__ void load_vec(float* dst, const float* src, int row0, int N) {
    const int tid = threadIdx.x;
    if (tid < kTile) {
        const int row = row0 + tid;
        dst[tid] = row < N ? src[row] : 0.f;
    }
}

// Forward. Each block owns query rows [q0, q0 + kTile) of one head and walks the
// keys one tile at a time:
//   1. S = scale * Q K^T for the tile, into shared memory;
//   2. online softmax: each row keeps a running max m and sum l. The tile's
//      probabilities are exp(S - m_new), and the previous sum and output are rescaled
//      by alpha = exp(m_old - m_new) before the tile's contribution is added;
//   3. O += P V, with O in registers, distributed over the threads.
// At the end O is divided by l and lse = m + log(l).
template <int D>
__global__ void __launch_bounds__(kThreads)
attention_forward_kernel(int N, const float* Q, const float* K, const float* V, float* O, float* lse, float scale) {
    extern __shared__ float smem[];
    float (*Qs)[D + 1] = reinterpret_cast<float (*)[D + 1]>(smem);
    float (*Ks)[D + 1] = Qs + kTile;
    float (*Vs)[D + 1] = Ks + kTile;
    float (*Ss)[kTile + 1] = reinterpret_cast<float (*)[kTile + 1]>(Vs + kTile);   // scores, then probabilities
    float* row_max   = reinterpret_cast<float*>(Ss + kTile);
    float* row_sum   = row_max + kTile;
    float* row_alpha = row_sum + kTile;

    const int tid = threadIdx.x, tx = tid % kGrid, ty = tid / kGrid;
    const int q0 = static_cast<int>(blockIdx.x) * kTile;
    const size_t head_rows = static_cast<size_t>(blockIdx.y) * N;
    Q += head_rows * D;
    K += head_rows * D;
    V += head_rows * D;
    O += head_rows * D;
    lse += head_rows;

    load_tile<D>(Qs, Q, q0, N);
    if (tid < kTile) {
        row_max[tid] = neg_inf();
        row_sum[tid] = 0.f;
    }

    float acc[kMicro][D / kGrid] = {};
    for (int k0 = 0; k0 < N; k0 += kTile) {
        load_tile<D>(Ks, K, k0, N);
        load_tile<D>(Vs, V, k0, N);
        __syncthreads();

        float s[kMicro][kMicro] = {};
        for (int d = 0; d < D; ++d) {
            float a[kMicro], b[kMicro];
            for (int i = 0; i < kMicro; ++i) a[i] = Qs[ty + kGrid * i][d];
            for (int j = 0; j < kMicro; ++j) b[j] = Ks[tx + kGrid * j][d];
            for (int i = 0; i < kMicro; ++i)
                for (int j = 0; j < kMicro; ++j) s[i][j] = fmaf(a[i], b[j], s[i][j]);
        }
        for (int i = 0; i < kMicro; ++i)
            for (int j = 0; j < kMicro; ++j) Ss[ty + kGrid * i][tx + kGrid * j] = scale * s[i][j];
        __syncthreads();

        {
            // kRowLanes consecutive threads share a row, each taking kRowCols of its columns,
            // and combine their partial max and sum with shuffles. Columns past N are masked.
            // Every tile has at least one valid key, so m_new is always finite.
            const int r = tid / kRowLanes, part = tid % kRowLanes;
            float v[kRowCols];
            float tile_max = neg_inf();
            for (int u = 0; u < kRowCols; ++u) {
                const int c = part * kRowCols + u;
                v[u] = k0 + c < N ? Ss[r][c] : neg_inf();
                tile_max = fmaxf(tile_max, v[u]);
            }
            for (int off = kRowLanes / 2; off > 0; off /= 2)
                tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, off));

            const float m_old = row_max[r], l_old = row_sum[r];
            const float m_new = fmaxf(m_old, tile_max);
            const float alpha = expf(m_old - m_new);
            float part_sum = 0.f;
            for (int u = 0; u < kRowCols; ++u) {
                v[u] = expf(v[u] - m_new);
                part_sum += v[u];
                Ss[r][part * kRowCols + u] = v[u];
            }
            for (int off = kRowLanes / 2; off > 0; off /= 2)
                part_sum += __shfl_xor_sync(0xffffffffu, part_sum, off);
            if (part == 0) {
                row_max[r] = m_new;
                row_sum[r] = l_old * alpha + part_sum;
                row_alpha[r] = alpha;
            }
        }
        __syncthreads();

        for (int i = 0; i < kMicro; ++i) {
            const float alpha = row_alpha[ty + kGrid * i];
            for (int j = 0; j < D / kGrid; ++j) acc[i][j] *= alpha;
        }
        for (int c = 0; c < kTile; ++c) {
            float p[kMicro], vv[D / kGrid];
            for (int i = 0; i < kMicro; ++i) p[i] = Ss[ty + kGrid * i][c];
            for (int j = 0; j < D / kGrid; ++j) vv[j] = Vs[c][tx + kGrid * j];
            for (int i = 0; i < kMicro; ++i)
                for (int j = 0; j < D / kGrid; ++j) acc[i][j] = fmaf(p[i], vv[j], acc[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < kMicro; ++i) {
        const int r = ty + kGrid * i;
        const int row = q0 + r;
        if (row >= N) continue;
        const float inv = 1.f / row_sum[r];
        for (int j = 0; j < D / kGrid; ++j) O[static_cast<size_t>(row) * D + tx + kGrid * j] = acc[i][j] * inv;
        if (tx == 0) lse[row] = row_max[r] + logf(row_sum[r]);
    }
}

// delta[i] = dot(dO_i, O_i), the rowwise term of the softmax backward.
__global__ void attention_delta_kernel(int rows, int D, const float* dO, const float* O, float* delta) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= rows) return;
    const float* go = dO + static_cast<size_t>(i) * D;
    const float* o = O + static_cast<size_t>(i) * D;
    float sum = 0.f;
    for (int d = 0; d < D; ++d) sum = fmaf(go[d], o[d], sum);
    delta[i] = sum;
}

// dQ. Each block owns query rows and walks the keys. For each tile it recomputes
//   P = exp(scale Q K^T - lse),  dP = dO V^T,  dS = P (dP - delta),
// then accumulates dQ += scale * dS K in registers. P is never stored.
template <int D>
__global__ void __launch_bounds__(kThreads)
attention_backward_dq_kernel(int N, const float* Q, const float* K, const float* V, const float* dO,
                             const float* lse, const float* delta, float* dQ, float scale) {
    extern __shared__ float smem[];
    float (*Qs)[D + 1] = reinterpret_cast<float (*)[D + 1]>(smem);
    float (*dOs)[D + 1] = Qs + kTile;
    float (*Ks)[D + 1] = dOs + kTile;
    float (*Vs)[D + 1] = Ks + kTile;
    float (*dSs)[kTile + 1] = reinterpret_cast<float (*)[kTile + 1]>(Vs + kTile);
    float* Ls = reinterpret_cast<float*>(dSs + kTile);   // lse and delta of the query tile
    float* Ds = Ls + kTile;

    const int tid = threadIdx.x, tx = tid % kGrid, ty = tid / kGrid;
    const int q0 = static_cast<int>(blockIdx.x) * kTile;
    const size_t head_rows = static_cast<size_t>(blockIdx.y) * N;
    Q += head_rows * D;
    K += head_rows * D;
    V += head_rows * D;
    dO += head_rows * D;
    dQ += head_rows * D;
    lse += head_rows;
    delta += head_rows;

    load_tile<D>(Qs, Q, q0, N);
    load_tile<D>(dOs, dO, q0, N);
    load_vec(Ls, lse, q0, N);
    load_vec(Ds, delta, q0, N);

    float acc[kMicro][D / kGrid] = {};
    for (int k0 = 0; k0 < N; k0 += kTile) {
        load_tile<D>(Ks, K, k0, N);
        load_tile<D>(Vs, V, k0, N);
        __syncthreads();

        float s[kMicro][kMicro] = {}, dp[kMicro][kMicro] = {};
        for (int d = 0; d < D; ++d) {
            float q[kMicro], go[kMicro], k[kMicro], v[kMicro];
            for (int i = 0; i < kMicro; ++i) {
                q[i]  = Qs[ty + kGrid * i][d];
                go[i] = dOs[ty + kGrid * i][d];
            }
            for (int j = 0; j < kMicro; ++j) {
                k[j] = Ks[tx + kGrid * j][d];
                v[j] = Vs[tx + kGrid * j][d];
            }
            for (int i = 0; i < kMicro; ++i)
                for (int j = 0; j < kMicro; ++j) {
                    s[i][j]  = fmaf(q[i], k[j], s[i][j]);
                    dp[i][j] = fmaf(go[i], v[j], dp[i][j]);
                }
        }
        for (int i = 0; i < kMicro; ++i)
            for (int j = 0; j < kMicro; ++j) {
                const int r = ty + kGrid * i, c = tx + kGrid * j;
                const bool valid = q0 + r < N && k0 + c < N;
                const float p = valid ? expf(scale * s[i][j] - Ls[r]) : 0.f;
                dSs[r][c] = p * (dp[i][j] - Ds[r]);
            }
        __syncthreads();

        for (int c = 0; c < kTile; ++c) {
            float ds[kMicro], kv[D / kGrid];
            for (int i = 0; i < kMicro; ++i) ds[i] = dSs[ty + kGrid * i][c];
            for (int j = 0; j < D / kGrid; ++j) kv[j] = Ks[c][tx + kGrid * j];
            for (int i = 0; i < kMicro; ++i)
                for (int j = 0; j < D / kGrid; ++j) acc[i][j] = fmaf(ds[i], kv[j], acc[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < kMicro; ++i) {
        const int row = q0 + ty + kGrid * i;
        if (row >= N) continue;
        for (int j = 0; j < D / kGrid; ++j) dQ[static_cast<size_t>(row) * D + tx + kGrid * j] = scale * acc[i][j];
    }
}

// dK and dV. Each block owns key rows and walks the query tiles. For each tile it
// computes P and dS in registers, then accumulates
//   dV += P^T dO,  dK += scale * dS^T Q
// over the tile's query rows. One shared tile holds P for the dV pass and then dS
// for the dK pass, which keeps the block within the shared memory budget.
template <int D>
__global__ void __launch_bounds__(kThreads)
attention_backward_dkv_kernel(int N, const float* Q, const float* K, const float* V, const float* dO,
                              const float* lse, const float* delta, float* dK, float* dV, float scale) {
    extern __shared__ float smem[];
    float (*Ks)[D + 1] = reinterpret_cast<float (*)[D + 1]>(smem);
    float (*Vs)[D + 1] = Ks + kTile;
    float (*Qs)[D + 1] = Vs + kTile;
    float (*dOs)[D + 1] = Qs + kTile;
    float (*Ps)[kTile + 1] = reinterpret_cast<float (*)[kTile + 1]>(dOs + kTile);   // [query][key]
    float* Ls = reinterpret_cast<float*>(Ps + kTile);
    float* Ds = Ls + kTile;

    const int tid = threadIdx.x, tx = tid % kGrid, ty = tid / kGrid;
    const int k0 = static_cast<int>(blockIdx.x) * kTile;
    const size_t head_rows = static_cast<size_t>(blockIdx.y) * N;
    Q += head_rows * D;
    K += head_rows * D;
    V += head_rows * D;
    dO += head_rows * D;
    dK += head_rows * D;
    dV += head_rows * D;
    lse += head_rows;
    delta += head_rows;

    load_tile<D>(Ks, K, k0, N);
    load_tile<D>(Vs, V, k0, N);

    float dk_acc[kMicro][D / kGrid] = {}, dv_acc[kMicro][D / kGrid] = {};
    for (int q0 = 0; q0 < N; q0 += kTile) {
        load_tile<D>(Qs, Q, q0, N);
        load_tile<D>(dOs, dO, q0, N);
        load_vec(Ls, lse, q0, N);
        load_vec(Ds, delta, q0, N);
        __syncthreads();

        float s[kMicro][kMicro] = {}, dp[kMicro][kMicro] = {};
        for (int d = 0; d < D; ++d) {
            float q[kMicro], go[kMicro], k[kMicro], v[kMicro];
            for (int i = 0; i < kMicro; ++i) {
                q[i]  = Qs[ty + kGrid * i][d];
                go[i] = dOs[ty + kGrid * i][d];
            }
            for (int j = 0; j < kMicro; ++j) {
                k[j] = Ks[tx + kGrid * j][d];
                v[j] = Vs[tx + kGrid * j][d];
            }
            for (int i = 0; i < kMicro; ++i)
                for (int j = 0; j < kMicro; ++j) {
                    s[i][j]  = fmaf(q[i], k[j], s[i][j]);
                    dp[i][j] = fmaf(go[i], v[j], dp[i][j]);
                }
        }

        // Row r is a query, column c a key. P and dS stay in registers until their pass.
        float p[kMicro][kMicro], ds[kMicro][kMicro];
        for (int i = 0; i < kMicro; ++i)
            for (int j = 0; j < kMicro; ++j) {
                const int r = ty + kGrid * i, c = tx + kGrid * j;
                const bool valid = q0 + r < N && k0 + c < N;
                p[i][j] = valid ? expf(scale * s[i][j] - Ls[r]) : 0.f;
                ds[i][j] = p[i][j] * (dp[i][j] - Ds[r]);
            }

        for (int i = 0; i < kMicro; ++i)
            for (int j = 0; j < kMicro; ++j) Ps[ty + kGrid * i][tx + kGrid * j] = p[i][j];
        __syncthreads();
        for (int r = 0; r < kTile; ++r) {
            float pv[kMicro], go[D / kGrid];
            for (int i = 0; i < kMicro; ++i) pv[i] = Ps[r][ty + kGrid * i];
            for (int j = 0; j < D / kGrid; ++j) go[j] = dOs[r][tx + kGrid * j];
            for (int i = 0; i < kMicro; ++i)
                for (int j = 0; j < D / kGrid; ++j) dv_acc[i][j] = fmaf(pv[i], go[j], dv_acc[i][j]);
        }
        __syncthreads();

        for (int i = 0; i < kMicro; ++i)
            for (int j = 0; j < kMicro; ++j) Ps[ty + kGrid * i][tx + kGrid * j] = ds[i][j];
        __syncthreads();
        for (int r = 0; r < kTile; ++r) {
            float dsv[kMicro], q[D / kGrid];
            for (int i = 0; i < kMicro; ++i) dsv[i] = Ps[r][ty + kGrid * i];
            for (int j = 0; j < D / kGrid; ++j) q[j] = Qs[r][tx + kGrid * j];
            for (int i = 0; i < kMicro; ++i)
                for (int j = 0; j < D / kGrid; ++j) dk_acc[i][j] = fmaf(dsv[i], q[j], dk_acc[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < kMicro; ++i) {
        const int key = k0 + ty + kGrid * i;
        if (key >= N) continue;
        for (int j = 0; j < D / kGrid; ++j) {
            const size_t at = static_cast<size_t>(key) * D + tx + kGrid * j;
            dK[at] = scale * dk_acc[i][j];
            dV[at] = dv_acc[i][j];
        }
    }
}

void check_shape(AttentionShape s) {
    if (s.heads <= 0 || s.seq <= 0) throw std::invalid_argument("attention: heads and seq must be positive");
    if (s.dim != 16 && s.dim != 32 && s.dim != 64) throw std::invalid_argument("attention: dim must be 16, 32 or 64");
}

void check_pointers(std::initializer_list<const void*> ptrs) {
    for (const void* p : ptrs)
        if (!p) throw std::invalid_argument("attention: null pointer");
}

template <int D>
void forward_launch(AttentionShape s, const float* Q, const float* K, const float* V, float* O, float* lse,
                    cudaStream_t stream) {
    const dim3 blocks((s.seq + kTile - 1) / kTile, s.heads);
    const float scale = 1.f / std::sqrt(static_cast<float>(D));
    CUDA_CHECK_THROW(cudaFuncSetAttribute(attention_forward_kernel<D>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          static_cast<int>(kForwardSmem<D>)));
    attention_forward_kernel<D><<<blocks, kThreads, kForwardSmem<D>, stream>>>(s.seq, Q, K, V, O, lse, scale);
    CUDA_CHECK_THROW(cudaGetLastError());
}

template <int D>
void backward_launch(AttentionShape s, const float* Q, const float* K, const float* V, const float* O,
                     const float* lse, const float* dO, float* dQ, float* dK, float* dV, float* delta,
                     cudaStream_t stream) {
    const float scale = 1.f / std::sqrt(static_cast<float>(D));
    const int rows = s.heads * s.seq;
    attention_delta_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(rows, D, dO, O, delta);
    const dim3 blocks((s.seq + kTile - 1) / kTile, s.heads);
    CUDA_CHECK_THROW(cudaFuncSetAttribute(attention_backward_dq_kernel<D>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          static_cast<int>(kBackwardSmem<D>)));
    CUDA_CHECK_THROW(cudaFuncSetAttribute(attention_backward_dkv_kernel<D>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          static_cast<int>(kBackwardSmem<D>)));
    attention_backward_dq_kernel<D><<<blocks, kThreads, kBackwardSmem<D>, stream>>>(s.seq, Q, K, V, dO, lse, delta,
                                                                                    dQ, scale);
    attention_backward_dkv_kernel<D><<<blocks, kThreads, kBackwardSmem<D>, stream>>>(s.seq, Q, K, V, dO, lse, delta,
                                                                                     dK, dV, scale);
    CUDA_CHECK_THROW(cudaGetLastError());
}

} // namespace

void attention_forward(AttentionShape shape, const float* Q, const float* K, const float* V,
                       float* O, float* lse, cudaStream_t stream) {
    check_shape(shape);
    check_pointers({Q, K, V, O, lse});
    switch (shape.dim) {
        case 16: forward_launch<16>(shape, Q, K, V, O, lse, stream); break;
        case 32: forward_launch<32>(shape, Q, K, V, O, lse, stream); break;
        default: forward_launch<64>(shape, Q, K, V, O, lse, stream); break;
    }
}

void attention_backward(AttentionShape shape, const float* Q, const float* K, const float* V,
                        const float* O, const float* lse, const float* dO, float* dQ, float* dK,
                        float* dV, float* delta, cudaStream_t stream) {
    check_shape(shape);
    check_pointers({Q, K, V, O, lse, dO, dQ, dK, dV, delta});
    switch (shape.dim) {
        case 16: backward_launch<16>(shape, Q, K, V, O, lse, dO, dQ, dK, dV, delta, stream); break;
        case 32: backward_launch<32>(shape, Q, K, V, O, lse, dO, dQ, dK, dV, delta, stream); break;
        default: backward_launch<64>(shape, Q, K, V, O, lse, dO, dQ, dK, dV, delta, stream); break;
    }
}

} // namespace cudarl::nn
