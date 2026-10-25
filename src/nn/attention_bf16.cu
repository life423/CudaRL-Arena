#include "nn/attention_bf16.h"

#include <cmath>
#include <initializer_list>
#include <stdexcept>

#include "core/bf16.h"
#include "cuda/cuda_memory.h"
#include "nn/mma_bf16.h"

namespace cudarl::nn {

namespace {

using namespace ptx;

// A block is four warps, and each warp owns 16 query rows of one head, so a block
// covers 64 query rows. Key and value tiles are 64 rows, staged through shared memory
// in two buffers with cp.async: one buffer loads while the other is used.
constexpr int kWarps    = 4;
constexpr int kThreads  = 32 * kWarps;
constexpr int kBlockQ   = 16 * kWarps;
constexpr int kKeyTile  = 64;
constexpr int kStages   = 2;
// Shared rows are padded by 8 bf16 (16 bytes). That makes the row stride an odd number
// of 16-byte chunks, so the 8 rows one ldmatrix reads fall in different banks.
constexpr int kPad = 8;

template <int D>
constexpr size_t kSmemBytes = sizeof(uint16_t) * 2 * kStages * kKeyTile * (D + kPad);

// -infinity as a float. The INFINITY macro is a double constant, which nvcc rejects here.
__device__ __forceinline__ float neg_inf() { return __uint_as_float(0xff800000u); }

// Two bf16 values in one register, low half first, as mma reads them.
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    return static_cast<uint32_t>(bf16::from_float(lo)) | (static_cast<uint32_t>(bf16::from_float(hi)) << 16);
}

// Each thread of a warp holds, for the score tile, rows g and g + 8 (g = lane / 4) at
// columns 2t, 2t + 1 of every n8 tile (t = lane % 4). The four row maxima and sums are
// shared by the quad of threads with the same g, so they are reduced with shuffles
// where needed. Each thread keeps its own partial sum, which is rescaled together
// with the output and combined once at the end.
template <int D>
__global__ void __launch_bounds__(kThreads)
attention_bf16_forward_kernel(int N, const uint16_t* Q, const uint16_t* K, const uint16_t* V, float* O, float* lse,
                              float scale) {
    static_assert(D == 16 || D == 32 || D == 64, "head dim");
    constexpr int kLd = D + kPad;       // shared row stride, elements
    constexpr int kNT = kKeyTile / 8;   // n8 tiles across the key tile
    constexpr int kND = D / 8;          // n8 tiles across the head dimension
    constexpr int kKD = D / 16;         // k16 steps over the head dimension (Q K^T)
    constexpr int kKC = kKeyTile / 16;  // k16 steps over the key tile (P V)

    extern __shared__ __align__(16) uint16_t smem[];
    uint16_t* const Ks = smem;                            // [stage][key][kLd]
    uint16_t* const Vs = smem + kStages * kKeyTile * kLd; // [stage][key][kLd]

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int q0 = static_cast<int>(blockIdx.x) * kBlockQ;
    const int wq = q0 + warp * 16;   // first query row of this warp
    const size_t head_rows = static_cast<size_t>(blockIdx.y) * N;
    Q += head_rows * D;
    K += head_rows * D;
    V += head_rows * D;
    O += head_rows * D;
    lse += head_rows;

    // Q fragments for the whole head dimension, read once from global memory. Rows
    // past N are zero. Each 32-bit load is two adjacent bf16 values.
    auto pair_at = [&](int row, int col) -> uint32_t {
        if (row >= N) return 0u;
        return *reinterpret_cast<const uint32_t*>(Q + static_cast<size_t>(row) * D + col);
    };
    uint32_t qf[kKD][4];
    for (int kd = 0; kd < kKD; ++kd) {
        qf[kd][0] = pair_at(wq + g, kd * 16 + 2 * t);
        qf[kd][1] = pair_at(wq + g + 8, kd * 16 + 2 * t);
        qf[kd][2] = pair_at(wq + g, kd * 16 + 8 + 2 * t);
        qf[kd][3] = pair_at(wq + g + 8, kd * 16 + 8 + 2 * t);
    }

    // Copies one key tile and its value tile into a shared buffer. Rows past N are
    // zero-filled by cp.async, and their scores are masked in the softmax.
    auto load_tile = [&](int stage, int kt) {
        uint16_t* ks = Ks + stage * kKeyTile * kLd;
        uint16_t* vs = Vs + stage * kKeyTile * kLd;
        for (int i = tid; i < kKeyTile * (D / 8); i += kThreads) {
            const int r = i / (D / 8), c = i % (D / 8);
            const int row = kt * kKeyTile + r;
            const bool valid = row < N;
            const size_t src = static_cast<size_t>(valid ? row : 0) * D + c * 8;
            cp_async_16(smem_addr(ks + r * kLd + c * 8), K + src, valid);
            cp_async_16(smem_addr(vs + r * kLd + c * 8), V + src, valid);
        }
    };

    float o[kND][4] = {};
    float m[2] = {neg_inf(), neg_inf()};   // running row max, rows g and g + 8
    float l[2] = {0.f, 0.f};               // per-thread partial row sums

    const int n_tiles = (N + kKeyTile - 1) / kKeyTile;
    load_tile(0, 0);
    cp_async_commit();
    for (int kt = 0; kt < n_tiles; ++kt) {
        // Issue the next tile, then wait for this one. Every iteration commits exactly
        // one group, so wait<1> leaves only the group just issued in flight.
        if (kt + 1 < n_tiles) load_tile((kt + 1) % kStages, kt + 1);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();

        const uint16_t* ks = Ks + (kt % kStages) * kKeyTile * kLd;
        const uint16_t* vs = Vs + (kt % kStages) * kKeyTile * kLd;

        // S = Q K^T for this warp's 16 rows against the tile's 64 keys.
        float s[kNT][4] = {};
        for (int kd = 0; kd < kKD; ++kd) {
            for (int nj = 0; nj < kNT; nj += 2) {
                // B is stored [key][dim], so this is the same load as gemm_bf16_nt's.
                const int row = nj * 8 + (lane & 7) + ((lane >> 4) << 3);
                const int chunk = kd * 2 + ((lane >> 3) & 1);
                uint32_t b[4];
                ldmatrix_x4(b, smem_addr(ks + row * kLd + chunk * 8));
                mma_16816(s[nj], qf[kd], b[0], b[1]);
                mma_16816(s[nj + 1], qf[kd], b[2], b[3]);
            }
        }

        // Scale, mask keys past N, then the online softmax.
        float mt[2] = {neg_inf(), neg_inf()};
        for (int nt = 0; nt < kNT; ++nt) {
            for (int e = 0; e < 4; ++e) {
                const int key = kt * kKeyTile + nt * 8 + 2 * t + (e & 1);
                const float v = key < N ? s[nt][e] * scale : neg_inf();
                s[nt][e] = v;
                mt[e >> 1] = fmaxf(mt[e >> 1], v);
            }
        }
        for (int h = 0; h < 2; ++h) {
            mt[h] = fmaxf(mt[h], __shfl_xor_sync(0xffffffffu, mt[h], 1));
            mt[h] = fmaxf(mt[h], __shfl_xor_sync(0xffffffffu, mt[h], 2));
        }
        float alpha[2];
        for (int h = 0; h < 2; ++h) {
            const float m_new = fmaxf(m[h], mt[h]);
            alpha[h] = expf(m[h] - m_new);
            m[h] = m_new;
        }
        float psum[2] = {0.f, 0.f};
        for (int nt = 0; nt < kNT; ++nt) {
            for (int e = 0; e < 4; ++e) {
                const float p = expf(s[nt][e] - m[e >> 1]);
                s[nt][e] = p;
                psum[e >> 1] += p;
            }
        }
        for (int h = 0; h < 2; ++h) l[h] = l[h] * alpha[h] + psum[h];
        for (int nd = 0; nd < kND; ++nd) {
            o[nd][0] *= alpha[0];
            o[nd][1] *= alpha[0];
            o[nd][2] *= alpha[1];
            o[nd][3] *= alpha[1];
        }

        // O += P V. P is taken straight from the score registers: n8 tiles 2c and 2c + 1
        // of S, in the same layout, are exactly the A fragment of k16 step c. V is stored
        // [key][dim], so it is loaded transposed to give the column-major B fragment.
        for (int kc = 0; kc < kKC; ++kc) {
            uint32_t pa[4];
            pa[0] = pack_bf16x2(s[2 * kc][0], s[2 * kc][1]);
            pa[1] = pack_bf16x2(s[2 * kc][2], s[2 * kc][3]);
            pa[2] = pack_bf16x2(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            pa[3] = pack_bf16x2(s[2 * kc + 1][2], s[2 * kc + 1][3]);
            for (int np = 0; np < kND; np += 2) {
                const int row = kc * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                const int chunk = np + (lane >> 4);
                uint32_t b[4];
                ldmatrix_x4_trans(b, smem_addr(vs + row * kLd + chunk * 8));
                mma_16816(o[np], pa, b[0], b[1]);
                mma_16816(o[np + 1], pa, b[2], b[3]);
            }
        }
        __syncthreads();
    }

    // Combine the quad's partial sums, then divide and store. Every lane takes part in
    // the shuffles; only the row bounds check is per lane.
    for (int h = 0; h < 2; ++h) {
        float lsum = l[h];
        lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
        lsum += __shfl_xor_sync(0xffffffffu, lsum, 2);
        const int row = wq + g + 8 * h;
        if (row >= N) continue;
        const float inv = 1.f / lsum;
        for (int nd = 0; nd < kND; ++nd) {
            const float2 out = make_float2(o[nd][2 * h] * inv, o[nd][2 * h + 1] * inv);
            *reinterpret_cast<float2*>(O + static_cast<size_t>(row) * D + nd * 8 + 2 * t) = out;
        }
        if (t == 0) lse[row] = m[h] + logf(lsum);
    }
}

void check_shape(AttentionShape s) {
    if (s.heads <= 0 || s.seq <= 0) throw std::invalid_argument("attention_bf16: heads and seq must be positive");
    if (s.dim != 16 && s.dim != 32 && s.dim != 64) throw std::invalid_argument("attention_bf16: dim must be 16, 32 or 64");
}

void check_pointers(std::initializer_list<const void*> ptrs) {
    for (const void* p : ptrs)
        if (!p) throw std::invalid_argument("attention_bf16: null pointer");
}

template <int D>
void forward_launch(AttentionShape s, const uint16_t* Q, const uint16_t* K, const uint16_t* V, float* O, float* lse,
                    cudaStream_t stream) {
    constexpr size_t smem = kSmemBytes<D>;
    CUDA_CHECK_THROW(cudaFuncSetAttribute(attention_bf16_forward_kernel<D>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          static_cast<int>(smem)));
    const dim3 blocks((s.seq + kBlockQ - 1) / kBlockQ, s.heads);
    const float scale = 1.f / std::sqrt(static_cast<float>(D));
    attention_bf16_forward_kernel<D><<<blocks, kThreads, smem, stream>>>(s.seq, Q, K, V, O, lse, scale);
    CUDA_CHECK_THROW(cudaGetLastError());
}

} // namespace

void attention_forward_bf16(AttentionShape shape, const uint16_t* Q, const uint16_t* K, const uint16_t* V,
                            float* O, float* lse, cudaStream_t stream) {
    check_shape(shape);
    check_pointers({Q, K, V, O, lse});
    switch (shape.dim) {
        case 16: forward_launch<16>(shape, Q, K, V, O, lse, stream); break;
        case 32: forward_launch<32>(shape, Q, K, V, O, lse, stream); break;
        default: forward_launch<64>(shape, Q, K, V, O, lse, stream); break;
    }
}

} // namespace cudarl::nn
