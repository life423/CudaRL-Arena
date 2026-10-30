#include "adam.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cudarl::nn {

namespace {

constexpr int kThreads    = 256;
constexpr int kNormBlocks = 256;   // fixed, so the reduction order never depends on the GPU

__device__ float block_sum(float v) {
    __shared__ float s[kThreads];
    s[threadIdx.x] = v;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    return s[0];
}

__global__ void sum_squares_partial_kernel(const float* __restrict__ g, int n, float* __restrict__ partial) {
    float acc = 0.f;
    for (int i = blockIdx.x * kThreads + threadIdx.x; i < n; i += kNormBlocks * kThreads) acc += g[i] * g[i];
    const float total = block_sum(acc);
    if (threadIdx.x == 0) partial[blockIdx.x] = total;
}

__global__ void norm_finish_kernel(const float* __restrict__ partial, float* __restrict__ norm) {
    const float total = block_sum(partial[threadIdx.x]);   // kNormBlocks == kThreads
    if (threadIdx.x == 0) *norm = sqrtf(total);
}

__global__ void adam_kernel(float* __restrict__ p, const float* __restrict__ g, float* __restrict__ m,
                            float* __restrict__ v, int n, float lr, float beta1, float beta2, float eps,
                            float bias_corr1, float bias_corr2, const float* __restrict__ norm,
                            float max_grad_norm) {
    const float scale = max_grad_norm > 0.f ? fminf(1.f, max_grad_norm / (*norm + 1e-6f)) : 1.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float gi = g[i] * scale;
        const float mi = beta1 * m[i] + (1.f - beta1) * gi;
        const float vi = beta2 * v[i] + (1.f - beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        p[i] -= lr * (mi / bias_corr1) / (sqrtf(vi / bias_corr2) + eps);
    }
}

int blocks_for(int n) { return std::max(1, std::min((n + kThreads - 1) / kThreads, 4096)); }

} // namespace

static_assert(kNormBlocks == kThreads, "norm_finish_kernel reduces one partial per thread");

GpuAdam::GpuAdam(int num_params, AdamConfig cfg)
    : cfg_(cfg), n_(num_params), m_(num_params), v_(num_params), norm_partial_(kNormBlocks), norm_(1) {
    if (num_params <= 0) throw std::invalid_argument("GpuAdam: num_params must be positive");
    CUDA_CHECK_THROW(cudaMemset(m_.get(), 0, m_.size_bytes()));
    CUDA_CHECK_THROW(cudaMemset(v_.get(), 0, v_.size_bytes()));
    CUDA_CHECK_THROW(cudaMemset(norm_.get(), 0, norm_.size_bytes()));
    // A first step may use a nonblocking stream unrelated to the default stream.
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
}

void GpuAdam::step(float* d_params, const float* d_grads, cudaStream_t stream) {
    ++t_;
    if (cfg_.max_grad_norm > 0.f) {
        sum_squares_partial_kernel<<<kNormBlocks, kThreads, 0, stream>>>(d_grads, n_, norm_partial_.get());
        norm_finish_kernel<<<1, kThreads, 0, stream>>>(norm_partial_.get(), norm_.get());
    }
    // Bias corrections in double on the host, matching the CPU reference.
    const float bc1 = static_cast<float>(1.0 - std::pow(static_cast<double>(cfg_.beta1), t_));
    const float bc2 = static_cast<float>(1.0 - std::pow(static_cast<double>(cfg_.beta2), t_));
    adam_kernel<<<blocks_for(n_), kThreads, 0, stream>>>(d_params, d_grads, m_.get(), v_.get(), n_, cfg_.lr,
                                                         cfg_.beta1, cfg_.beta2, cfg_.eps, bc1, bc2, norm_.get(),
                                                         cfg_.max_grad_norm);
    CUDA_CHECK_THROW(cudaGetLastError());
}

float GpuAdam::last_grad_norm() const {
    float norm = 0.f;
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    norm_.copy_to_host(&norm, 1);
    return norm;
}
void GpuAdam::save(checkpoint::Writer& out) const {
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    out.value(cfg_);out.value(t_);
    std::vector<float> m(n_),v(n_);m_.copy_to_host(m.data(),m.size());v_.copy_to_host(v.data(),v.size());
    out.array(m);out.array(v);
}
void GpuAdam::load(checkpoint::Reader& in) {
    const auto cfg=in.value<AdamConfig>();const int steps=in.value<int>();
    auto m=in.array<float>(n_),v=in.array<float>(n_);
    if(m.size()!=static_cast<size_t>(n_) || v.size()!=m.size() || steps<0 || !std::isfinite(cfg.lr) || cfg.lr<=0 ||
       cfg.beta1!=cfg_.beta1 || cfg.beta2!=cfg_.beta2 || cfg.eps!=cfg_.eps || cfg.max_grad_norm!=cfg_.max_grad_norm)
        throw std::runtime_error("Incompatible Adam checkpoint");
    for(size_t i=0;i<m.size();++i) if(!std::isfinite(m[i]) || !std::isfinite(v[i]) || v[i]<0) throw std::runtime_error("Invalid Adam moments");
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    cfg_=cfg;t_=steps;m_.copy_from_host(m.data(),m.size());v_.copy_from_host(v.data(),v.size());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
}

} // namespace cudarl::nn
