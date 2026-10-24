#ifndef CUDARL_CUDA_DEVICE_H
#define CUDARL_CUDA_DEVICE_H

#include <cstddef>
#include <string>

namespace cudarl {

struct DeviceInfo {
    bool        available = false;      // a device exists and our kernels run on it
    std::string name;
    int         compute_major = 0;
    int         compute_minor = 0;
    int         multiprocessors = 0;
    std::size_t global_memory_bytes = 0;
    std::string error;                  // why CUDA is unavailable, when it is
};

// Queries device 0 and launches a tiny kernel to confirm this binary's GPU code
// actually executes there (catches a missing driver or a mismatched architecture).
// Never throws.
DeviceInfo probe_device();

} // namespace cudarl

#endif // CUDARL_CUDA_DEVICE_H
