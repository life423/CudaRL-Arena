#ifndef CUDARL_NN_LOSS_H
#define CUDARL_NN_LOSS_H

#include <cuda_runtime.h>

namespace cudarl::nn {

// dy = 2 (y - target) / count: the gradient of mean((y - target)^2).
void mse_loss_grad(const float* d_y, const float* d_target, float* d_dy, int count, cudaStream_t stream = nullptr);

} // namespace cudarl::nn

#endif // CUDARL_NN_LOSS_H
