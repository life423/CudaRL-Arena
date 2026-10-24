#include "gemm.h"

#include <algorithm>

#include "cuda/cuda_memory.h"

namespace cudarl::nn {

namespace {

// Tiled kernel shape: each 16x16-thread block computes a 64x64 tile of C,
// stepping through K in slices of 16; each thread accumulates 4x4 results.
constexpr int kBM = 64, kBN = 64, kBK = 16;
constexpr int kThreadsX = 16, kThreadsY = 16;
constexpr int kThreads = kThreadsX * kThreadsY;
constexpr int kTM = kBM / kThreadsY, kTN = kBN / kThreadsX;

struct EpilogueArgs {
    Epilogue::Kind kind;
    const float*   bias;
    Activation     activation;
};

__device__ __forceinline__ void write_result(float* C, int ldc, int row, int col, float acc,
                                             const EpilogueArgs& ep) {
    float* c = C + static_cast<size_t>(row) * ldc + col;
    switch (ep.kind) {
        case Epilogue::Kind::Accumulate:     *c += acc; break;
        case Epilogue::Kind::BiasActivation: *c = activate(ep.activation, acc + ep.bias[col]); break;
        default:                             *c = acc; break;
    }
}

// op(A)[row, kk] and op(B)[kk, col] for either storage order.
template <bool TA>
__device__ __forceinline__ float load_a(const float* A, int lda, int row, int kk) {
    return TA ? A[static_cast<size_t>(kk) * lda + row] : A[static_cast<size_t>(row) * lda + kk];
}
template <bool TB>
__device__ __forceinline__ float load_b(const float* B, int ldb, int kk, int col) {
    return TB ? B[static_cast<size_t>(col) * ldb + kk] : B[static_cast<size_t>(kk) * ldb + col];
}

// Baseline: one thread per output element, every operand read from global memory.
template <bool TA, bool TB>
__global__ void gemm_naive_kernel(int m, int n, int k, const float* __restrict__ A, int lda,
                                  const float* __restrict__ B, int ldb, float* __restrict__ C, int ldc,
                                  EpilogueArgs ep) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;
    float acc = 0.f;
    for (int kk = 0; kk < k; ++kk) acc = fmaf(load_a<TA>(A, lda, row, kk), load_b<TB>(B, ldb, kk, col), acc);
    write_result(C, ldc, row, col, acc, ep);
}

// Shared-memory tiling with register blocking. Thread (tx, ty) owns rows
// ty + 16i and columns tx + 16j of the tile: interleaving (rather than a
// contiguous 4x4 patch) keeps shared-memory reads conflict-free and makes
// each warp's global stores contiguous. Tiles are padded by one column so the
// transposed shared-memory stores don't collide on banks either.
//
// blockIdx.z selects a K slice [z * k_per_split, (z + 1) * k_per_split) and
// shifts C by z * c_split_stride; a plain GEMM is the single-slice case.
template <bool TA, bool TB>
__global__ void __launch_bounds__(kThreads)
gemm_tiled_kernel(int m, int n, int k, const float* __restrict__ A, int lda,
                  const float* __restrict__ B, int ldb, float* __restrict__ C, int ldc,
                  EpilogueArgs ep, int k_per_split, size_t c_split_stride) {
    __shared__ float As[kBK][kBM + 1];   // As[kk][r] = op(A)[row0 + r, k0 + kk]
    __shared__ float Bs[kBK][kBN + 1];   // Bs[kk][c] = op(B)[k0 + kk, col0 + c]

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid  = ty * kThreadsX + tx;
    const int row0 = blockIdx.y * kBM;
    const int col0 = blockIdx.x * kBN;
    const int k_begin = blockIdx.z * k_per_split;
    const int k_end   = min(k, k_begin + k_per_split);
    C += blockIdx.z * c_split_stride;

    float acc[kTM][kTN] = {};

    for (int k0 = k_begin; k0 < k_end; k0 += kBK) {
        // Consecutive threads walk the dimension that is contiguous in memory.
        for (int i = tid; i < kBM * kBK; i += kThreads) {
            const int r  = TA ? i % kBM : i / kBK;
            const int kk = TA ? i / kBM : i % kBK;
            const int gr = row0 + r, gk = k0 + kk;
            As[kk][r] = (gr < m && gk < k_end) ? load_a<TA>(A, lda, gr, gk) : 0.f;
        }
        for (int i = tid; i < kBK * kBN; i += kThreads) {
            const int c  = TB ? i / kBK : i % kBN;
            const int kk = TB ? i % kBK : i / kBN;
            const int gc = col0 + c, gk = k0 + kk;
            Bs[kk][c] = (gc < n && gk < k_end) ? load_b<TB>(B, ldb, gk, gc) : 0.f;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kBK; ++kk) {
            float a[kTM], b[kTN];
#pragma unroll
            for (int i = 0; i < kTM; ++i) a[i] = As[kk][ty + i * kThreadsY];
#pragma unroll
            for (int j = 0; j < kTN; ++j) b[j] = Bs[kk][tx + j * kThreadsX];
#pragma unroll
            for (int i = 0; i < kTM; ++i)
#pragma unroll
                for (int j = 0; j < kTN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kTM; ++i) {
        const int row = row0 + ty + i * kThreadsY;
        if (row >= m) continue;
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
            const int col = col0 + tx + j * kThreadsX;
            if (col < n) write_result(C, ldc, row, col, acc[i][j], ep);
        }
    }
}

// Sums the split-K slices in index order (deterministic), then applies the epilogue.
__global__ void split_k_reduce_kernel(const float* __restrict__ workspace, int splits, int m, int n,
                                      float* __restrict__ C, int ldc, EpilogueArgs ep) {
    const size_t count = static_cast<size_t>(m) * n;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        float acc = 0.f;
        for (int z = 0; z < splits; ++z) acc += workspace[z * count + i];
        write_result(C, ldc, static_cast<int>(i / n), static_cast<int>(i % n), acc, ep);
    }
}

struct Launch {
    GemmKernel kernel;
    int m, n, k;
    const float* A; int lda;
    const float* B; int ldb;
    float* C; int ldc;
    EpilogueArgs ep;
    int splits;          // tiled only; > 1 writes slices of C at stride m*n
    cudaStream_t stream;
};

template <bool TA, bool TB>
void launch(const Launch& p) {
    const dim3 block(kThreadsX, kThreadsY);
    if (p.kernel == GemmKernel::Naive) {
        const dim3 grid((p.n + kThreadsX - 1) / kThreadsX, (p.m + kThreadsY - 1) / kThreadsY);
        gemm_naive_kernel<TA, TB><<<grid, block, 0, p.stream>>>(p.m, p.n, p.k, p.A, p.lda, p.B, p.ldb,
                                                                p.C, p.ldc, p.ep);
    } else {
        const int k_per_split = ((p.k + p.splits - 1) / p.splits + kBK - 1) / kBK * kBK;
        const dim3 grid((p.n + kBN - 1) / kBN, (p.m + kBM - 1) / kBM, p.splits);
        gemm_tiled_kernel<TA, TB><<<grid, block, 0, p.stream>>>(
            p.m, p.n, p.k, p.A, p.lda, p.B, p.ldb, p.C, p.ldc, p.ep, p.splits > 1 ? k_per_split : p.k,
            static_cast<size_t>(p.m) * p.n);
    }
}

void dispatch(Trans ta, Trans tb, const Launch& p) {
    const bool a_t = ta == Trans::T, b_t = tb == Trans::T;
    if (!a_t && !b_t)     launch<false, false>(p);
    else if (!a_t && b_t) launch<false, true>(p);
    else if (a_t && !b_t) launch<true, false>(p);
    else                  launch<true, true>(p);
    CUDA_CHECK_THROW(cudaGetLastError());
}

EpilogueArgs to_args(const Epilogue& e) { return {e.kind, e.bias, e.activation}; }

} // namespace

void gemm(GemmKernel kernel, Trans ta, Trans tb, int m, int n, int k,
          const float* A, int lda, const float* B, int ldb, float* C, int ldc,
          Epilogue epilogue, cudaStream_t stream) {
    if (m <= 0 || n <= 0) return;
    dispatch(ta, tb, {kernel, m, n, k, A, lda, B, ldb, C, ldc, to_args(epilogue), 1, stream});
}

void gemm_split_k(Trans ta, Trans tb, int m, int n, int k,
                  const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                  int splits, float* workspace, Epilogue epilogue, cudaStream_t stream) {
    if (m <= 0 || n <= 0) return;
    if (splits <= 1) {
        gemm(GemmKernel::Tiled, ta, tb, m, n, k, A, lda, B, ldb, C, ldc, epilogue, stream);
        return;
    }
    const EpilogueArgs store{Epilogue::Kind::Store, nullptr, Activation::Identity};
    dispatch(ta, tb, {GemmKernel::Tiled, m, n, k, A, lda, B, ldb, workspace, n, store, splits, stream});

    const size_t count  = static_cast<size_t>(m) * n;
    const int    blocks = static_cast<int>(std::min<size_t>((count + 255) / 256, 4096));
    split_k_reduce_kernel<<<blocks, 256, 0, stream>>>(workspace, splits, m, n, C, ldc, to_args(epilogue));
    CUDA_CHECK_THROW(cudaGetLastError());
}

int split_k_count(int m, int n, int k, int num_sms) {
    const int tiles     = ((m + kBM - 1) / kBM) * ((n + kBN - 1) / kBN);
    const int want      = (2 * num_sms + tiles - 1) / tiles;      // ~2 blocks per SM
    const int max_split = std::max(1, k / (4 * kBK));              // keep >= 4 K-steps per slice
    return std::max(1, std::min({want, max_split, kMaxSplitK}));
}

} // namespace cudarl::nn
