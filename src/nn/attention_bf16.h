#ifndef CUDARL_NN_ATTENTION_BF16_H
#define CUDARL_NN_ATTENTION_BF16_H

#include <cstdint>

#include <cuda_runtime.h>

#include "nn/attention.h"

namespace cudarl::nn {

// Forward attention on tensor cores. Q, K and V are bf16 bit patterns, as in
// gemm_bf16_nt; the scores accumulate and the softmax runs in fp32, and P is rounded
// to bf16 for the P V product. O and lse are fp32, as in attention_forward. Same
// shape rules and errors as attention_forward. Q, K and V must be 4-byte aligned,
// which any cudaMalloc pointer is. Forward only: training uses attention_backward,
// which needs fp32 inputs.
void attention_forward_bf16(AttentionShape shape, const uint16_t* Q, const uint16_t* K, const uint16_t* V,
                            float* O, float* lse, cudaStream_t stream = nullptr);

} // namespace cudarl::nn

#endif // CUDARL_NN_ATTENTION_BF16_H
