#include "device.h"
#include "cuda_memory.h"

#include <exception>

namespace cudarl {

namespace {

constexpr int kProbeValue = 0x5eed;

__global__ void probe_kernel(int* out) { *out = kProbeValue; }

} // namespace

DeviceInfo probe_device() {
    DeviceInfo info;

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        info.error = err != cudaSuccess ? cudaGetErrorString(err) : "no CUDA devices found";
        cudaGetLastError();  // clear the error so later calls aren't poisoned
        return info;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        info.error = cudaGetErrorString(err);
        cudaGetLastError();
        return info;
    }
    info.name                = prop.name;
    info.compute_major       = prop.major;
    info.compute_minor       = prop.minor;
    info.multiprocessors     = prop.multiProcessorCount;
    info.global_memory_bytes = prop.totalGlobalMem;

    try {
        device_ptr<int> d_value(1);
        probe_kernel<<<1, 1>>>(d_value.get());
        CUDA_CHECK_THROW(cudaGetLastError());
        int value = 0;
        d_value.copy_to_host(&value, 1);
        info.available = (value == kProbeValue);
        if (!info.available) info.error = "probe kernel returned an unexpected value";
    } catch (const std::exception& e) {
        info.error = e.what();
    }
    return info;
}

} // namespace cudarl
