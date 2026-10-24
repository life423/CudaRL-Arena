#ifndef CUDARL_GODOT_CUDA_DEVICE_H
#define CUDARL_GODOT_CUDA_DEVICE_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>

#include "cuda/device.h"

namespace godot {

// Reports whether CUDA works on this machine. Probes the GPU once, on creation.
class CudaDevice : public RefCounted {
    GDCLASS(CudaDevice, RefCounted);

protected:
    static void _bind_methods();

public:
    CudaDevice();

    bool       is_available() const { return info_.available; }
    Dictionary get_info() const;
    String     describe() const;   // one line, e.g. for logging at startup

private:
    cudarl::DeviceInfo info_;
};

} // namespace godot

#endif // CUDARL_GODOT_CUDA_DEVICE_H
