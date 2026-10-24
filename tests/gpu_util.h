#ifndef CUDARL_TESTS_GPU_UTIL_H
#define CUDARL_TESTS_GPU_UTIL_H

#include <vector>

#include "core/rng.h"
#include "cuda/cuda_memory.h"

namespace cudarl::test {

inline std::vector<float> random_floats(size_t n, uint32_t seed, float scale = 1.f) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = scale * rng::normal(seed, 0, static_cast<uint32_t>(i));
    return v;
}

inline device_ptr<float> to_device(const std::vector<float>& h) {
    device_ptr<float> d(h.size());
    d.copy_from_host(h.data(), h.size());
    return d;
}

inline std::vector<float> to_host(const device_ptr<float>& d, size_t n) {
    std::vector<float> h(n);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    d.copy_to_host(h.data(), n);
    return h;
}

inline std::vector<float> to_host(const float* d, size_t n) {
    std::vector<float> h(n);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    CUDA_CHECK_THROW(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

} // namespace cudarl::test

#endif // CUDARL_TESTS_GPU_UTIL_H
