#include "gemm_bf16.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>
#include <mma.h>

#include "core/bf16.h"
#include "cuda/cuda_memory.h"

namespace cudarl::nn {

namespace {

// --- PTX wrappers --------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Four 8x8 b16 matrices from shared memory. Lanes 8i..8i+7 supply the row
// addresses of matrix i; afterwards each lane holds, in r[i], two consecutive
// elements of row (lane / 4) of matrix i, at columns 2 * (lane % 4) and +1.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// d[16x8] += a[16x16] * b[16x8], bf16 inputs, fp32 accumulate. With g = lane / 4
// and t = lane % 4 (PTX ISA, m16n8k16 fragment layouts):
//   a[0]: A[g][2t..2t+1]     a[1]: A[g+8][2t..2t+1]    a[2]: A[g][2t+8..]   a[3]: A[g+8][2t+8..]
//   b[0]: B[2t..2t+1][g]     b[1]: B[2t+8..2t+9][g]
//   d[0..1]: D[g][2t..2t+1]  d[2..3]: D[g+8][2t..2t+1]
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 16-byte global -> shared copy that bypasses the registers; a source size of 0
// zero-fills the destination (used for out-of-bounds chunks).
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// --- v1 to v4: mma.sync kernel ----------------------------------------------

// Shared-memory tiles are [rows][BK] bf16, addressed in 16-byte chunks of 8
// elements. Without swizzling, the 8 rows one ldmatrix reads at the same chunk
// land on only 2 of the 8 possible 16-byte bank groups (a 4-way conflict for
// 64-byte rows). XOR-ing the chunk index with a function of the row spreads
// them over all 8.
template <int BK, bool SWIZZLE>
__device__ __forceinline__ int tile_offset(int row, int chunk) {
    constexpr int kChunksPerRow = BK / 8;
    static_assert(kChunksPerRow == 4 || kChunksPerRow == 8, "BK must be 32 or 64");
    if constexpr (SWIZZLE) {
        chunk ^= kChunksPerRow == 4 ? (row >> 1) & 3 : row & 7;
    }
    return row * BK + chunk * 8;
}

// Maps the linear block index to a (block row, block column). Grouping
// GROUP_M block rows together makes consecutive blocks share B tiles in L2.
template <bool RASTER>
__device__ __forceinline__ void block_coords(int pid, int grid_m, int grid_n, int& bm, int& bn) {
    if constexpr (RASTER) {
        constexpr int kGroupM = 8;
        const int per_group = kGroupM * grid_n;
        const int first_m   = (pid / per_group) * kGroupM;
        const int group_m   = min(grid_m - first_m, kGroupM);
        bm = first_m + (pid % per_group) % group_m;
        bn = (pid % per_group) / group_m;
    } else {
        bm = pid / grid_n;
        bn = pid % grid_n;
    }
}

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES, bool SWIZZLE, bool RASTER>
__global__ void __launch_bounds__(32 * WARPS_M * WARPS_N)
mma_gemm_kernel(int m, int n, int k, const uint16_t* __restrict__ A, int lda, const uint16_t* __restrict__ B,
                int ldb, float* __restrict__ C, int ldc, bool vector_store) {
    constexpr int kThreads = 32 * WARPS_M * WARPS_N;
    constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;   // warp tile
    constexpr int MI = WM / 16, NI = WN / 8;               // mma tiles per warp
    constexpr int kChunksPerRow = BK / 8;
    static_assert(BK % 16 == 0 && WM % 16 == 0 && WN % 16 == 0, "tile shape");

    extern __shared__ __align__(16) uint16_t smem[];
    uint16_t* const As = smem;                             // [STAGES][BM * BK]
    uint16_t* const Bs = smem + STAGES * BM * BK;          // [STAGES][BN * BK]

    const int grid_m = (m + BM - 1) / BM, grid_n = (n + BN - 1) / BN;
    int bm, bn;
    block_coords<RASTER>(blockIdx.x, grid_m, grid_n, bm, bn);
    const int m0 = bm * BM, n0 = bn * BN;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;

    float acc[MI][NI][4] = {};

    auto copy_chunk = [&](uint16_t* dst, const uint16_t* src, bool valid) {
        if constexpr (STAGES > 1) {
            cp_async_16(smem_addr(dst), src, valid);
        } else {
            *reinterpret_cast<uint4*>(dst) = valid ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
        }
    };

    // Stage one BK-wide slice of A and B into shared memory. K is a multiple of
    // 8, so every 16-byte chunk is either fully in bounds or fully out.
    auto load_tile = [&](int stage, int kt) {
        const int k0 = kt * BK;
        uint16_t* as = As + stage * BM * BK;
        uint16_t* bs = Bs + stage * BN * BK;
        for (int i = tid; i < BM * kChunksPerRow; i += kThreads) {
            const int r = i / kChunksPerRow, c = i % kChunksPerRow;
            const int gr = m0 + r, gk = k0 + c * 8;
            const bool valid = gr < m && gk < k;
            copy_chunk(as + tile_offset<BK, SWIZZLE>(r, c), valid ? A + static_cast<size_t>(gr) * lda + gk : A, valid);
        }
        for (int i = tid; i < BN * kChunksPerRow; i += kThreads) {
            const int r = i / kChunksPerRow, c = i % kChunksPerRow;
            const int gr = n0 + r, gk = k0 + c * 8;
            const bool valid = gr < n && gk < k;
            copy_chunk(bs + tile_offset<BK, SWIZZLE>(r, c), valid ? B + static_cast<size_t>(gr) * ldb + gk : B, valid);
        }
    };

    auto compute_tile = [&](int stage) {
        const uint16_t* as = As + stage * BM * BK;
        const uint16_t* bs = Bs + stage * BN * BK;
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            uint32_t a[MI][4], b[NI][2];
            // A: matrices (rows 0-7 | 8-15) x (k 0-7 | 8-15) -> a[0..3] in fragment order.
#pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
                const int row = warp_m * WM + mi * 16 + (lane & 15);
                ldmatrix_x4(a[mi], smem_addr(as + tile_offset<BK, SWIZZLE>(row, ks * 2 + (lane >> 4))));
            }
            // B is stored [n][k], so one x4 load covers two n8 tiles x (k 0-7 | 8-15).
#pragma unroll
            for (int ni = 0; ni < NI; ni += 2) {
                const int row = warp_n * WN + ni * 8 + (lane & 7) + ((lane >> 4) << 3);
                uint32_t t[4];
                ldmatrix_x4(t, smem_addr(bs + tile_offset<BK, SWIZZLE>(row, ks * 2 + ((lane >> 3) & 1))));
                b[ni][0] = t[0];
                b[ni][1] = t[1];
                b[ni + 1][0] = t[2];
                b[ni + 1][1] = t[3];
            }
#pragma unroll
            for (int mi = 0; mi < MI; ++mi)
#pragma unroll
                for (int ni = 0; ni < NI; ++ni) mma_16816(acc[mi][ni], a[mi], b[ni][0], b[ni][1]);
        }
    };

    const int k_tiles = (k + BK - 1) / BK;
    if constexpr (STAGES == 1) {
        for (int kt = 0; kt < k_tiles; ++kt) {
            load_tile(0, kt);
            __syncthreads();
            compute_tile(0);
            __syncthreads();
        }
    } else {
        // Prologue: STAGES - 1 slices in flight. Each iteration waits for its
        // slice, refills the stage freed by the previous iteration, then computes.
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < k_tiles) load_tile(s, s);
            cp_async_commit();
        }
        for (int kt = 0; kt < k_tiles; ++kt) {
            cp_async_wait<STAGES - 2>();
            __syncthreads();
            const int next = kt + STAGES - 1;
            if (next < k_tiles) load_tile(next % STAGES, next);
            cp_async_commit();
            compute_tile(kt % STAGES);
        }
        cp_async_wait<0>();
    }

    // Epilogue: each lane holds rows g and g + 8, columns 2t and 2t + 1 of every mma tile.
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
            const int col = n0 + warp_n * WN + ni * 8 + t * 2;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = m0 + warp_m * WM + mi * 16 + g + half * 8;
                if (row >= m) continue;
                float* dst = C + static_cast<size_t>(row) * ldc + col;
                const float v0 = acc[mi][ni][half * 2], v1 = acc[mi][ni][half * 2 + 1];
                if (vector_store && col + 1 < n) {
                    *reinterpret_cast<float2*>(dst) = make_float2(v0, v1);
                } else {
                    if (col < n) dst[0] = v0;
                    if (col + 1 < n) dst[1] = v1;
                }
            }
        }
    }
}

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES, bool SWIZZLE, bool RASTER>
void launch_mma(int m, int n, int k, const uint16_t* A, int lda, const uint16_t* B, int ldb, float* C, int ldc,
                bool vector_store, cudaStream_t stream) {
    constexpr int    kThreads = 32 * WARPS_M * WARPS_N;
    constexpr size_t kSmem    = static_cast<size_t>(STAGES) * (BM + BN) * BK * sizeof(uint16_t);
    auto kernel = mma_gemm_kernel<BM, BN, BK, WARPS_M, WARPS_N, STAGES, SWIZZLE, RASTER>;
    if constexpr (kSmem > 48 * 1024) {
        static const cudaError_t configured =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem));
        CUDA_CHECK_THROW(configured);
    }
    const unsigned blocks = static_cast<unsigned>(((m + BM - 1) / BM) * ((n + BN - 1) / BN));
    kernel<<<blocks, kThreads, kSmem, stream>>>(m, n, k, A, lda, B, ldb, C, ldc, vector_store);
}

// --- v0: WMMA ---------------------------------------------------------------

// 64x64 output tile per block, 4 warps of 32x32 (2x2 WMMA fragments), K in
// steps of 16 staged through padded shared memory. Results go through shared
// memory so edge tiles can be written with bounds checks.
constexpr int kWmmaTile = 64, kWmmaK = 16, kWmmaPad = 8;

__global__ void __launch_bounds__(128)
wmma_gemm_kernel(int m, int n, int k, const uint16_t* __restrict__ A, int lda, const uint16_t* __restrict__ B, int ldb,
                 float* __restrict__ C, int ldc) {
    using namespace nvcuda;
    constexpr int ldm = kWmmaK + kWmmaPad;
    __shared__ __align__(32) __nv_bfloat16 As[kWmmaTile][ldm];
    __shared__ __align__(32) __nv_bfloat16 Bs[kWmmaTile][ldm];
    __shared__ __align__(32) float Cs[4][32 * 32];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int m0 = blockIdx.y * kWmmaTile, n0 = blockIdx.x * kWmmaTile;
    const int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (auto& row : acc)
        for (auto& f : row) wmma::fill_fragment(f, 0.f);

    const auto* Ab = reinterpret_cast<const __nv_bfloat16*>(A);
    const auto* Bb = reinterpret_cast<const __nv_bfloat16*>(B);
    for (int k0 = 0; k0 < k; k0 += kWmmaK) {
        for (int i = tid; i < kWmmaTile * kWmmaK; i += 128) {
            const int r = i / kWmmaK, kk = i % kWmmaK;
            const bool in_k = k0 + kk < k;
            As[r][kk] = (m0 + r < m && in_k) ? Ab[static_cast<size_t>(m0 + r) * lda + k0 + kk] : __nv_bfloat16(0.f);
            Bs[r][kk] = (n0 + r < n && in_k) ? Bb[static_cast<size_t>(n0 + r) * ldb + k0 + kk] : __nv_bfloat16(0.f);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[2];
        for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &As[wm + i * 16][0], ldm);
        for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Bs[wn + j * 16][0], ldm);   // Bs[n][k] is col-major K x N
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        __syncthreads();
    }

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&Cs[warp][i * 16 * 32 + j * 16], acc[i][j], 32, wmma::mem_row_major);
    __syncwarp();
    for (int e = lane; e < 32 * 32; e += 32) {
        const int row = m0 + wm + e / 32, col = n0 + wn + e % 32;
        if (row < m && col < n) C[static_cast<size_t>(row) * ldc + col] = Cs[warp][e];
    }
}

// --- Utilities ------------------------------------------------------------------

__global__ void convert_kernel(const float* __restrict__ src, uint16_t* __restrict__ dst, size_t count) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        dst[i] = bf16::from_float(src[i]);
    }
}

// Eight independent accumulator chains per warp hide the mma latency; inputs
// come from registers, so this measures the tensor cores alone.
constexpr int kPeakChains = 8;

__global__ void mma_peak_kernel(float* out, int iterations) {
    const uint32_t one = 0x3f803f80u;   // two bf16 1.0s
    const uint32_t a[4] = {one, one, one, one};
    float acc[kPeakChains][4] = {};
    for (int i = 0; i < iterations; ++i) {
#pragma unroll
        for (int c = 0; c < kPeakChains; ++c) mma_16816(acc[c], a, one, one);
    }
    float sum = 0.f;
    for (auto& chain : acc)
        for (float v : chain) sum += v;
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

// Candidate configurations for the v4 tuning sweep (all swizzled). Kept in the
// library so `nn_bench tune` reproduces the numbers in docs/gemm_worklog.md.
using LaunchFn = void (*)(int, int, int, const uint16_t*, int, const uint16_t*, int, float*, int, bool, cudaStream_t);
struct TuningCandidate {
    const char* name;
    LaunchFn    launch;
};
constexpr TuningCandidate kTuningCandidates[] = {
    {"128x128x32  4 warps (64x64)  3 stages",          launch_mma<128, 128, 32, 2, 2, 3, true, false>},
    {"128x128x32  4 warps (64x64)  3 stages  raster",  launch_mma<128, 128, 32, 2, 2, 3, true, true>},
    {"128x128x32  4 warps (64x64)  2 stages",          launch_mma<128, 128, 32, 2, 2, 2, true, false>},
    {"128x128x32  4 warps (64x64)  2 stages  raster",  launch_mma<128, 128, 32, 2, 2, 2, true, true>},
    {"128x128x32  4 warps (64x64)  4 stages  raster",  launch_mma<128, 128, 32, 2, 2, 4, true, true>},
    {"128x128x32  8 warps (64x32)  2 stages  raster",  launch_mma<128, 128, 32, 2, 4, 2, true, true>},
    {"128x128x32  8 warps (64x32)  3 stages  raster",  launch_mma<128, 128, 32, 2, 4, 3, true, true>},
    {"128x256x32  8 warps (64x64)  2 stages  raster",  launch_mma<128, 256, 32, 2, 4, 2, true, true>},
    {"128x256x32  8 warps (64x64)  3 stages  raster",  launch_mma<128, 256, 32, 2, 4, 3, true, true>},
    {"256x128x32  8 warps (64x64)  2 stages  raster",  launch_mma<256, 128, 32, 4, 2, 2, true, true>},
    {"256x128x32  8 warps (64x64)  3 stages  raster",  launch_mma<256, 128, 32, 4, 2, 3, true, true>},
    {"128x128x64  4 warps (64x64)  2 stages  raster",  launch_mma<128, 128, 64, 2, 2, 2, true, true>},
    {"128x128x64  8 warps (64x32)  2 stages  raster",  launch_mma<128, 128, 64, 2, 4, 2, true, true>},
    {"128x64x32   4 warps (64x32)  4 stages  raster",  launch_mma<128, 64, 32, 2, 2, 4, true, true>},
};

void check_alignment(const void* p, const char* what) {
    if (reinterpret_cast<uintptr_t>(p) % 16 != 0) {
        throw std::invalid_argument(std::string("gemm_bf16_nt: ") + what + " must be 16-byte aligned");
    }
}

} // namespace

const char* tc_kernel_name(TcKernel kernel) {
    switch (kernel) {
        case TcKernel::Wmma:      return "v0 WMMA";
        case TcKernel::MmaSync:   return "v1 mma.sync + ldmatrix";
        case TcKernel::Pipelined: return "v2 + cp.async pipeline";
        case TcKernel::Swizzled:  return "v3 + swizzled smem";
        case TcKernel::Tuned:     return "v4 tuned";
    }
    return "?";
}

namespace {

// Shared argument checks; returns whether C allows 8-byte (float2) stores.
bool validate(int m, int n, int k, const uint16_t* A, int lda, const uint16_t* B, int ldb, const float* C, int ldc) {
    if (m < 0 || n < 0 || k < 0) throw std::invalid_argument("gemm_bf16_nt: negative size");
    if (k % 8 != 0 || lda % 8 != 0 || ldb % 8 != 0) {
        throw std::invalid_argument("gemm_bf16_nt: k, lda and ldb must be multiples of 8");
    }
    if (m > 0 && n > 0) {
        check_alignment(A, "A");
        check_alignment(B, "B");
    }
    return ldc % 2 == 0 && reinterpret_cast<uintptr_t>(C) % 8 == 0;
}

} // namespace

int tuning_candidate_count() { return static_cast<int>(std::size(kTuningCandidates)); }

const char* tuning_candidate_name(int index) {
    return index >= 0 && index < tuning_candidate_count() ? kTuningCandidates[index].name : "?";
}

void gemm_bf16_nt_candidate(int index, int m, int n, int k, const uint16_t* A, int lda, const uint16_t* B, int ldb,
                            float* C, int ldc, cudaStream_t stream) {
    if (index < 0 || index >= tuning_candidate_count()) throw std::invalid_argument("gemm_bf16_nt_candidate: index");
    const bool vector_store = validate(m, n, k, A, lda, B, ldb, C, ldc);
    if (m == 0 || n == 0) return;
    kTuningCandidates[index].launch(m, n, k, A, lda, B, ldb, C, ldc, vector_store, stream);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void gemm_bf16_nt(TcKernel kernel, int m, int n, int k, const uint16_t* A, int lda, const uint16_t* B, int ldb,
                  float* C, int ldc, cudaStream_t stream) {
    const bool vector_store = validate(m, n, k, A, lda, B, ldb, C, ldc);
    if (m == 0 || n == 0) return;

    switch (kernel) {
        case TcKernel::Wmma: {
            const dim3 grid((n + kWmmaTile - 1) / kWmmaTile, (m + kWmmaTile - 1) / kWmmaTile);
            wmma_gemm_kernel<<<grid, 128, 0, stream>>>(m, n, k, A, lda, B, ldb, C, ldc);
            break;
        }
        //                                BM   BN  BK  WM WN STAGES SWIZZLE RASTER
        case TcKernel::MmaSync:   launch_mma<128, 128, 32, 2, 2, 1, false, false>(m, n, k, A, lda, B, ldb, C, ldc, vector_store, stream); break;
        case TcKernel::Pipelined: launch_mma<128, 128, 32, 2, 2, 3, false, false>(m, n, k, A, lda, B, ldb, C, ldc, vector_store, stream); break;
        case TcKernel::Swizzled:  launch_mma<128, 128, 32, 2, 2, 3, true,  false>(m, n, k, A, lda, B, ldb, C, ldc, vector_store, stream); break;
        // v4 = the tuning sweep's winner: double buffering keeps shared memory at
        // 32 KB per block, so 3 blocks fit per SM, and occupancy beats pipeline depth.
        case TcKernel::Tuned:     launch_mma<128, 128, 32, 2, 2, 2, true,  true >(m, n, k, A, lda, B, ldb, C, ldc, vector_store, stream); break;
        default: throw std::invalid_argument("gemm_bf16_nt: unknown kernel");
    }
    CUDA_CHECK_THROW(cudaGetLastError());
}

void convert_to_bf16(const float* src, uint16_t* dst, size_t count, cudaStream_t stream) {
    if (count == 0) return;
    const int blocks = static_cast<int>(std::min<size_t>((count + 255) / 256, 8192));
    convert_kernel<<<blocks, 256, 0, stream>>>(src, dst, count);
    CUDA_CHECK_THROW(cudaGetLastError());
}

double measure_mma_peak_tflops() {
    int device = 0, sms = 0;
    CUDA_CHECK_THROW(cudaGetDevice(&device));
    CUDA_CHECK_THROW(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    // 16 blocks x 8 warps per SM, best of several runs (the first ones also warm the clocks up).
    const int blocks = sms * 16, threads = 256, iterations = 4096;
    device_ptr<float> out(static_cast<size_t>(blocks) * threads);
    cudaEvent_t start, stop;
    CUDA_CHECK_THROW(cudaEventCreate(&start));
    CUDA_CHECK_THROW(cudaEventCreate(&stop));
    float best_ms = 0.f;
    for (int run = 0; run < 5; ++run) {
        CUDA_CHECK_THROW(cudaEventRecord(start));
        mma_peak_kernel<<<blocks, threads>>>(out.get(), iterations);
        CUDA_CHECK_THROW(cudaEventRecord(stop));
        CUDA_CHECK_THROW(cudaEventSynchronize(stop));
        CUDA_CHECK_THROW(cudaGetLastError());
        float ms = 0.f;
        CUDA_CHECK_THROW(cudaEventElapsedTime(&ms, start, stop));
        if (run == 0 || ms < best_ms) best_ms = ms;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const double warps = static_cast<double>(blocks) * threads / 32;
    const double flops = warps * iterations * kPeakChains * (2.0 * 16 * 8 * 16);
    return flops / (best_ms * 1e-3) * 1e-12;
}

} // namespace cudarl::nn
