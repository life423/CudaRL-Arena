// This file contains only the method implementations that don't require CUDA compilation
// All other implementations are in src/gpu/environment.cu

#include "environment.h"
#include <stdexcept>
#include <string>

namespace cudarl {

// Note: Constructor, destructor, move operations, and other methods are implemented in src/gpu/environment.cu
// This file only contains methods that need to be available without CUDA compilation

void Environment::copyGridToBuffer(float* host_buffer, size_t num_elements) const {
    if (!host_buffer) {
        throw std::invalid_argument("Host buffer cannot be null");
    }
    
    size_t expected_size = static_cast<size_t>(m_state.width) * m_state.height;
    if (num_elements != expected_size) {
        throw std::invalid_argument("Buffer size mismatch: expected " + 
                                  std::to_string(expected_size) + " but got " +
                                  std::to_string(num_elements));
    }
    
    // Copy from device grid to host buffer
    m_deviceGrid.copyToHost(host_buffer, num_elements);
}

} // namespace cudarl
