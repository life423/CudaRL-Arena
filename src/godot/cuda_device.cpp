#include "cuda_device.h"

#include <godot_cpp/core/class_db.hpp>

namespace godot {

void CudaDevice::_bind_methods() {
    ClassDB::bind_method(D_METHOD("is_available"), &CudaDevice::is_available);
    ClassDB::bind_method(D_METHOD("get_info"), &CudaDevice::get_info);
    ClassDB::bind_method(D_METHOD("describe"), &CudaDevice::describe);
}

CudaDevice::CudaDevice() : info_(cudarl::probe_device()) {}

Dictionary CudaDevice::get_info() const {
    Dictionary d;
    d["available"]       = info_.available;
    d["name"]            = String(info_.name.c_str());
    d["compute_capability"] = String::num_int64(info_.compute_major) + "." + String::num_int64(info_.compute_minor);
    d["multiprocessors"] = info_.multiprocessors;
    d["memory_bytes"]    = static_cast<int64_t>(info_.global_memory_bytes);
    d["error"]           = String(info_.error.c_str());
    return d;
}

String CudaDevice::describe() const {
    if (!info_.available) {
        return String("CUDA unavailable: ") + info_.error.c_str();
    }
    const double gib = static_cast<double>(info_.global_memory_bytes) / (1024.0 * 1024.0 * 1024.0);
    return String("CUDA ready: ") + info_.name.c_str() +
           " (sm_" + String::num_int64(info_.compute_major) + String::num_int64(info_.compute_minor) +
           ", " + String::num_int64(info_.multiprocessors) + " SMs, " + String::num(gib, 1) + " GiB)";
}

} // namespace godot
