#ifndef CUDARL_NN_GEMM_H
#define CUDARL_NN_GEMM_H

#include <cuda_runtime.h>

#include "mlp_spec.h"

namespace cudarl::nn {

enum class Trans { N, T };
enum class GemmKernel { Naive, Tiled };

// What happens to each result c = sum_k op(A)[m,k] * op(B)[k,n] when it is written.
struct Epilogue {
    enum class Kind { Store, Accumulate, BiasActivation };

    Kind         kind       = Kind::Store;
    const float* bias       = nullptr;               // [n], BiasActivation only
    Activation   activation = Activation::Identity;  // BiasActivation only

    static Epilogue store() { return {}; }
    static Epilogue accumulate() { return {Kind::Accumulate, nullptr, Activation::Identity}; }
    static Epilogue bias_activation(const float* bias, Activation a) { return {Kind::BiasActivation, bias, a}; }
};

// C[m x n] = op(A) * op(B), all row-major with explicit leading dimensions.
//   op(A) is [m x k]: A is stored [m x k] when ta == N, or [k x m] when ta == T.
//   op(B) is [k x n]: B is stored [k x n] when tb == N, or [n x k] when tb == T.
// Store:          C = op(A) op(B)
// Accumulate:     C += op(A) op(B)
// BiasActivation: C = act(op(A) op(B) + bias)   (bias broadcast over rows)
// Asynchronous on `stream`; throws cuda_error if the launch fails.
void gemm(GemmKernel kernel, Trans ta, Trans tb, int m, int n, int k,
          const float* A, int lda, const float* B, int ldb, float* C, int ldc,
          Epilogue epilogue = Epilogue::store(), cudaStream_t stream = nullptr);

// Deterministic split-K for long, thin reductions (e.g. weight gradients summed
// over a large batch, where m x n is a handful of tiles): K is cut into `splits`
// slices computed in parallel into `workspace` [splits x m x n], which are then
// summed in a fixed order into C with the epilogue. No atomics, so results are
// bit-identical run to run. Tiled kernel only.
constexpr int kMaxSplitK = 64;
void gemm_split_k(Trans ta, Trans tb, int m, int n, int k,
                  const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                  int splits, float* workspace, Epilogue epilogue = Epilogue::store(),
                  cudaStream_t stream = nullptr);

// A split count that gives roughly two blocks per SM without making slices too short.
int split_k_count(int m, int n, int k, int num_sms);

} // namespace cudarl::nn

#endif // CUDARL_NN_GEMM_H
