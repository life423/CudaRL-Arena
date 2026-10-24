#ifndef CUDARL_NN_GEMM_BF16_H
#define CUDARL_NN_GEMM_BF16_H

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

// Tensor-core GEMM: bf16 inputs, fp32 accumulation, fp32 output. Each version
// adds one optimization on top of the previous one, and all are kept so the
// benchmark can show what each step buys (see docs/gemm_worklog.md).
namespace cudarl::nn {

enum class TcKernel : int {
    Wmma = 0,       // v0: WMMA API, 64x64 tiles through shared memory
    MmaSync = 1,    // v1: mma.sync m16n8k16 + ldmatrix, 128x128x32 tiles, synchronous loads
    Pipelined = 2,  // v2: + 3-stage cp.async pipeline (loads overlap math)
    Swizzled = 3,   // v3: + XOR-swizzled shared memory (no ldmatrix bank conflicts)
    Tuned = 4,      // v4: tuning-sweep winner: 2-stage pipeline (more blocks per SM) + L2-friendly block order
};
constexpr int kNumTcKernels = 5;
const char* tc_kernel_name(TcKernel kernel);

// C[m x n] = A[m x k] * B[n x k]^T. A and B hold bf16 bit patterns, row-major
// with K contiguous (the layout of Y = X W^T); C is fp32, row-major.
// Requires k, lda, ldb to be multiples of 8 and A, B 16-byte aligned (cp.async
// moves 16-byte chunks); any m and n are fine. Throws std::invalid_argument
// otherwise. Asynchronous on `stream`.
void gemm_bf16_nt(TcKernel kernel, int m, int n, int k, const uint16_t* A, int lda, const uint16_t* B, int ldb,
                  float* C, int ldc, cudaStream_t stream = nullptr);

// The configurations the v4 tuning sweep compares (tile shape, warps, pipeline
// depth, block order). Same contract as gemm_bf16_nt; used by `nn_bench tune`.
int         tuning_candidate_count();
const char* tuning_candidate_name(int index);
void        gemm_bf16_nt_candidate(int index, int m, int n, int k, const uint16_t* A, int lda, const uint16_t* B,
                                   int ldb, float* C, int ldc, cudaStream_t stream = nullptr);

// dst[i] = bf16(src[i]), round to nearest even, on the device.
void convert_to_bf16(const float* src, uint16_t* dst, size_t count, cudaStream_t stream = nullptr);

// Measured tensor-core ceiling: a register-only loop of independent
// mma.sync.m16n8k16 (bf16 in, fp32 accumulate) on every SM. TFLOP/s. Synchronizes.
double measure_mma_peak_tflops();

} // namespace cudarl::nn

#endif // CUDARL_NN_GEMM_BF16_H
