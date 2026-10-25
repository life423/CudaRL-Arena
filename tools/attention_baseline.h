#ifndef CUDARL_TOOLS_ATTENTION_BASELINE_H
#define CUDARL_TOOLS_ATTENTION_BASELINE_H

#include <cstdint>

#include <cuda_runtime.h>

// The unfused attention that the fused kernel is measured against: the score
// matrix goes through global memory, and the softmax runs as its own kernels
// between cuBLAS GEMMs. Linked only into nn_bench, never into the library.

// In place, one warp per row: S[r][:] <- softmax(scale * S[r][:]).
void unfused_softmax_rows(float* S, int rows, int cols, float scale, cudaStream_t stream = nullptr);

// In place, one warp per row: dP[r][:] <- scale * P[r][:] * (dP[r][:] - sum_c P[r][c] dP[r][c]).
void unfused_softmax_backward_rows(const float* P, float* dP, int rows, int cols, float scale,
                                   cudaStream_t stream = nullptr);

// One warp per row: P[r][:] <- bf16(softmax(scale * S[r][:])), reading fp32 scores
// and writing bf16 probabilities, as the bf16 unfused path does before its P V GEMM.
void unfused_softmax_rows_to_bf16(const float* S, uint16_t* P, int rows, int cols, float scale,
                                  cudaStream_t stream = nullptr);

#endif // CUDARL_TOOLS_ATTENTION_BASELINE_H
