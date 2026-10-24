#include "mlp_gpu.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace cudarl::nn {

namespace {

constexpr int kElementwiseThreads = 256;

// Column sums are done in two deterministic passes: fixed-size row chunks are
// reduced in a fixed order, then the chunk partials are summed in index order.
constexpr int kColThreadsX   = 32;    // columns per block (contiguous in memory)
constexpr int kColThreadsY   = 8;     // row lanes per block
constexpr int kRowsPerChunk  = 256;

int grid_for(size_t count) {
    return static_cast<int>(std::min<size_t>((count + kElementwiseThreads - 1) / kElementwiseThreads, 8192));
}

__global__ void activation_backward_kernel(const float* __restrict__ dy, const float* __restrict__ y,
                                           float* __restrict__ dz, size_t count, Activation a) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        dz[i] = dy[i] * activate_grad_from_output(a, y[i]);
    }
}

__global__ void column_partial_kernel(const float* __restrict__ x, int rows, int cols,
                                      float* __restrict__ partial) {
    __shared__ float s[kColThreadsY][kColThreadsX + 1];
    const int col   = blockIdx.x * kColThreadsX + threadIdx.x;
    const int row0  = blockIdx.y * kRowsPerChunk;
    const int row1  = min(rows, row0 + kRowsPerChunk);
    float acc = 0.f;
    if (col < cols) {
        for (int r = row0 + threadIdx.y; r < row1; r += kColThreadsY) acc += x[static_cast<size_t>(r) * cols + col];
    }
    s[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.y == 0 && col < cols) {
        float sum = 0.f;
        for (int i = 0; i < kColThreadsY; ++i) sum += s[i][threadIdx.x];
        partial[static_cast<size_t>(blockIdx.y) * cols + col] = sum;
    }
}

__global__ void column_finish_kernel(const float* __restrict__ partial, int chunks, int cols,
                                     float* __restrict__ out) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float sum = 0.f;
    for (int c = 0; c < chunks; ++c) sum += partial[static_cast<size_t>(c) * cols + col];
    out[col] += sum;
}

// out[c] += sum over rows of x[r][c], deterministically.
void column_sum_accumulate(const float* x, int rows, int cols, float* partial, float* out, cudaStream_t s) {
    const int chunks = (rows + kRowsPerChunk - 1) / kRowsPerChunk;
    const dim3 block(kColThreadsX, kColThreadsY);
    const dim3 grid((cols + kColThreadsX - 1) / kColThreadsX, chunks);
    column_partial_kernel<<<grid, block, 0, s>>>(x, rows, cols, partial);
    column_finish_kernel<<<(cols + kElementwiseThreads - 1) / kElementwiseThreads, kElementwiseThreads, 0, s>>>(
        partial, chunks, cols, out);
    CUDA_CHECK_THROW(cudaGetLastError());
}

int device_sm_count() {
    int device = 0, sms = 0;
    CUDA_CHECK_THROW(cudaGetDevice(&device));
    CUDA_CHECK_THROW(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    return sms;
}

} // namespace

GpuMlp::GpuMlp(MlpSpec spec, int max_batch, uint32_t seed, GemmKernel kernel)
    : spec_(std::move(spec)), max_batch_(max_batch), kernel_(kernel), num_sms_(device_sm_count()) {
    if (max_batch <= 0) throw std::invalid_argument("GpuMlp: max_batch must be positive");
    const size_t batch  = static_cast<size_t>(max_batch);
    const size_t widest = static_cast<size_t>(spec_.max_width());

    params_.allocate(spec_.num_params());
    grads_.allocate(spec_.num_params());
    for (int l = 0; l < spec_.num_layers(); ++l) acts_.emplace_back(batch * spec_.out(l));
    dz_.allocate(batch * widest);
    dbuf_[0].allocate(batch * widest);
    dbuf_[1].allocate(batch * widest);

    size_t ws = 1;
    for (int l = 0; l < spec_.num_layers(); ++l) {
        const int splits = split_k_count(spec_.out(l), spec_.in(l), max_batch, num_sms_);
        if (splits > 1) ws = std::max(ws, static_cast<size_t>(splits) * spec_.out(l) * spec_.in(l));
    }
    split_ws_.allocate(ws);
    col_partial_.allocate(((batch + kRowsPerChunk - 1) / kRowsPerChunk) * widest);

    set_params(init_params(spec_, seed));
    zero_grad();
}

void GpuMlp::check_batch(int batch) const {
    if (batch <= 0 || batch > max_batch_) {
        throw std::invalid_argument("GpuMlp: batch " + std::to_string(batch) + " outside [1, " +
                                    std::to_string(max_batch_) + "]");
    }
}

void GpuMlp::forward(const float* d_x, int batch, cudaStream_t stream) {
    check_batch(batch);
    input_ = d_x;
    batch_ = batch;
    const float* in = d_x;
    for (int l = 0; l < spec_.num_layers(); ++l) {
        const int n_in = spec_.in(l), n_out = spec_.out(l);
        // Y[batch x out] = act(X[batch x in] * W^T + b), with W stored [out x in].
        gemm(kernel_, Trans::N, Trans::T, batch, n_out, n_in, in, n_in, params_.get() + spec_.weight_offset(l), n_in,
             acts_[l].get(), n_out,
             Epilogue::bias_activation(params_.get() + spec_.bias_offset(l), spec_.activations[l]), stream);
        in = acts_[l].get();
    }
}

void GpuMlp::zero_grad(cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaMemsetAsync(grads_.get(), 0, grads_.size_bytes(), stream));
}

void GpuMlp::backward(const float* d_dy, int batch, float* d_dx, cudaStream_t stream) {
    if (batch != batch_ || input_ == nullptr) {
        throw std::logic_error("GpuMlp::backward: batch must match the preceding forward()");
    }
    const float* d_out = d_dy;   // dL/d(output of layer l)
    int next_buf = 0;
    for (int l = spec_.num_layers() - 1; l >= 0; --l) {
        const int n_in = spec_.in(l), n_out = spec_.out(l);
        const Activation act = spec_.activations[l];
        const float* x = l == 0 ? input_ : acts_[l - 1].get();
        const float* W = params_.get() + spec_.weight_offset(l);

        // dZ = dY * act'(Y); identity layers pass dY straight through.
        const float* dz = d_out;
        if (act != Activation::Identity) {
            const size_t count = static_cast<size_t>(batch) * n_out;
            activation_backward_kernel<<<grid_for(count), kElementwiseThreads, 0, stream>>>(
                d_out, acts_[l].get(), dz_.get(), count, act);
            CUDA_CHECK_THROW(cudaGetLastError());
            dz = dz_.get();
        }

        // dW[out x in] += dZ^T * X: a reduction over the batch, so split K across SMs.
        const int splits = split_k_count(n_out, n_in, batch, num_sms_);
        gemm_split_k(Trans::T, Trans::N, n_out, n_in, batch, dz, n_out, x, n_in,
                     grads_.get() + spec_.weight_offset(l), n_in, splits, split_ws_.get(),
                     Epilogue::accumulate(), stream);

        // db[out] += column sums of dZ.
        column_sum_accumulate(dz, batch, n_out, col_partial_.get(), grads_.get() + spec_.bias_offset(l), stream);

        // dX[batch x in] = dZ * W, the next layer down's dY.
        if (l == 0 && d_dx == nullptr) break;
        float* dst = l == 0 ? d_dx : dbuf_[next_buf].get();
        gemm(kernel_, Trans::N, Trans::N, batch, n_in, n_out, dz, n_out, W, n_in, dst, n_in, Epilogue::store(),
             stream);
        d_out = dst;
        next_buf ^= 1;
    }
}

void GpuMlp::set_params(const std::vector<float>& host) {
    if (static_cast<int>(host.size()) != spec_.num_params()) throw std::invalid_argument("GpuMlp::set_params: size");
    params_.copy_from_host(host.data(), host.size());
}

std::vector<float> GpuMlp::get_params() const {
    std::vector<float> h(params_.size());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    params_.copy_to_host(h.data(), h.size());
    return h;
}

std::vector<float> GpuMlp::get_grads() const {
    std::vector<float> h(grads_.size());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    grads_.copy_to_host(h.data(), h.size());
    return h;
}

} // namespace cudarl::nn
