#ifndef CUDARL_NN_ATTENTION_H
#define CUDARL_NN_ATTENTION_H

#include <cuda_runtime.h>

namespace cudarl::nn {

// Scaled dot-product attention, O = softmax(Q K^T / sqrt(dim)) V, for `heads`
// independent problems. Q, K, V, O are [heads x seq x dim] row-major. There is no
// mask. The seq x seq score matrix is never written to global memory: each block
// streams key tiles through shared memory and keeps a running (online) softmax.
// fp32 throughout; dim must be 16, 32 or 64.
struct AttentionShape {
    int heads;
    int seq;
    int dim;
};

// Writes O and lse [heads x seq], the row log-normalizer log(sum_j exp(S_ij)) that
// the backward pass needs. Throws std::invalid_argument for a bad shape or a null
// pointer, and cuda_error if a launch fails. Asynchronous on `stream`.
void attention_forward(AttentionShape shape, const float* Q, const float* K, const float* V,
                       float* O, float* lse, cudaStream_t stream = nullptr);

// Given the forward pass's O and lse and the upstream gradient dO, writes dQ, dK
// and dV. `delta` is scratch [heads x seq]. dK and dV are owned by key tiles and
// dQ by query tiles, so the gradients are written without atomics and are
// bit-identical run to run. Same error behaviour as attention_forward.
void attention_backward(AttentionShape shape, const float* Q, const float* K, const float* V,
                        const float* O, const float* lse, const float* dO, float* dQ, float* dK,
                        float* dV, float* delta, cudaStream_t stream = nullptr);

} // namespace cudarl::nn

#endif // CUDARL_NN_ATTENTION_H
