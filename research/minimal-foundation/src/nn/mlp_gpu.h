#ifndef CUDARL_NN_MLP_GPU_H
#define CUDARL_NN_MLP_GPU_H

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/cuda_memory.h"
#include "gemm.h"
#include "mlp_spec.h"

namespace cudarl::nn {

// MLP whose parameters, gradients, and activations all live on the GPU.
// Construction completes initialization. Compute calls are asynchronous on the
// given stream; explicit host copies synchronize as needed. Results are
// bit-identical run to run: no atomics anywhere in forward or backward.
class GpuMlp {
public:
    GpuMlp(MlpSpec spec, int max_batch, uint32_t seed, GemmKernel kernel = GemmKernel::Tiled);

    const MlpSpec& spec() const { return spec_; }
    int max_batch() const { return max_batch_; }
    int num_params() const { return spec_.num_params(); }

    // d_x: device [batch x input_size]. It must stay valid until backward() for this batch.
    void forward(const float* d_x, int batch, cudaStream_t stream = nullptr);
    const float* output() const { return acts_.back().get(); }   // device [batch x output_size]

    void zero_grad(cudaStream_t stream = nullptr);
    // d_dy = dL/d(output) for the last forward(). Accumulates into grads(); writes
    // dL/d(input) to d_dx when it is non-null.
    void backward(const float* d_dy, int batch, float* d_dx = nullptr, cudaStream_t stream = nullptr);

    float*       params() { return params_.get(); }
    const float* params() const { return params_.get(); }
    float*       grads() { return grads_.get(); }
    const float* grads() const { return grads_.get(); }

    void               set_params(const std::vector<float>& host);
    std::vector<float> get_params() const;
    std::vector<float> get_grads() const;

private:
    void check_batch(int batch) const;

    MlpSpec    spec_;
    int        max_batch_;
    GemmKernel kernel_;
    int        num_sms_;

    device_ptr<float>              params_;
    device_ptr<float>              grads_;
    std::vector<device_ptr<float>> acts_;        // acts_[l] = output of layer l, [max_batch x out(l)]
    device_ptr<float>              dz_;          // dL/d(pre-activation) of the current layer
    device_ptr<float>              dbuf_[2];     // ping-pong dL/d(layer input)
    device_ptr<float>              split_ws_;    // split-K partial weight gradients
    device_ptr<float>              col_partial_; // per-chunk partial bias gradients

    const float* input_ = nullptr;
    int          batch_ = 0;
};

} // namespace cudarl::nn

#endif // CUDARL_NN_MLP_GPU_H
