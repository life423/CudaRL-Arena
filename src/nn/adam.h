#ifndef CUDARL_NN_ADAM_H
#define CUDARL_NN_ADAM_H

#include <cuda_runtime.h>

#include "adam_config.h"
#include "cuda/cuda_memory.h"
#include "core/checkpoint.h"

namespace cudarl::nn {

// Adam on the GPU. The global gradient norm is computed on the device and read
// by the update kernel directly, so a step never waits on the host. The norm
// reduction uses a fixed grid, making results bit-identical across runs.
class GpuAdam {
public:
    // Initialization completes before return, including for first use on a nonblocking stream.
    GpuAdam(int num_params, AdamConfig cfg);

    void step(float* d_params, const float* d_grads, cudaStream_t stream = nullptr);

    int steps() const { return t_; }
    const AdamConfig& config() const { return cfg_; }
    void set_learning_rate(float lr) { cfg_.lr = lr; }   // e.g. for annealing schedules
    // Norm of the gradients seen by the last step (before clipping). Synchronizes;
    // only computed when clipping is enabled.
    float last_grad_norm() const;
    // Checkpoint boundaries fence the device, including nonblocking streams.
    // Calls must be serialized by the owner; no fence is added to step().
    void save(checkpoint::Writer& out) const;
    void load(checkpoint::Reader& in);

private:
    AdamConfig        cfg_;
    int               n_;
    int               t_ = 0;
    device_ptr<float> m_, v_;
    device_ptr<float> norm_partial_;
    device_ptr<float> norm_;
};

} // namespace cudarl::nn

#endif // CUDARL_NN_ADAM_H
