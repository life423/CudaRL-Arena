#ifndef CUDARL_NN_ADAM_H
#define CUDARL_NN_ADAM_H

#include <cuda_runtime.h>
#include <stdexcept>

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
    // New input coordinates may have a younger Adam age than inherited weights.
    // The versioned owner stores these three fields separately from legacy save().
    void set_fresh_column(int stride, int rows, int age, int width = 1, int first = -1) {
        if(first<0)first=stride-width;
        if (stride < 1 || rows < 1 || stride * rows > n_ || age < 0 || age > t_ || width < 1 || first<0 || first+width>stride)
            throw std::invalid_argument("Invalid fresh Adam column");
        fresh_stride_ = stride; fresh_rows_ = rows; fresh_t_ = age; fresh_width_ = width;fresh_first_=first;
    }
    void set_secondary_fresh_column(int stride,int rows,int age,int width,int first){
        if(stride<1||rows<1||stride*rows>n_||age<0||age>t_||width<1||first<0||first+width>stride)
            throw std::invalid_argument("Invalid secondary Adam columns");
        second_stride_=stride;second_rows_=rows;second_t_=age;second_width_=width;second_first_=first;
    }
    int fresh_steps(int group=0) const { return group?second_t_:fresh_t_; }
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
    int               fresh_stride_ = 0, fresh_rows_ = 0, fresh_t_ = 0, fresh_width_ = 1;
    int               fresh_first_=0,second_stride_=0,second_rows_=0,second_t_=0,second_width_=0,second_first_=0;
    device_ptr<float> m_, v_;
    device_ptr<float> norm_partial_;
    device_ptr<float> norm_;
};

} // namespace cudarl::nn

#endif // CUDARL_NN_ADAM_H
