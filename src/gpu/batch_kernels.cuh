#pragma once

// Stub header for GPU batch operations
// This would contain CUDA kernel declarations in a real implementation

namespace cudarl {

// Forward declarations for CUDA kernels (stubs)
// In a real implementation, these would be actual CUDA kernel declarations

// Stub function declarations
void launch_reset_environments_kernel(int num_envs, int width, int height);
void launch_step_environments_kernel(int num_envs, int* actions);
void launch_update_grids_kernel(int num_envs, float* grids);

// Device buffer template forward declaration
template<typename T>
class DeviceBuffer;

} // namespace cudarl
