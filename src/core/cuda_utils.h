#pragma once

#include <memory>
#include <vector>

namespace cudarl {

// Simple DeviceBuffer template for CUDA memory management
// This is a stub implementation - real version would use CUDA APIs
template<typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count = 0) : m_count(count), m_hostData(count) {}
    
    // Move constructor
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : m_count(other.m_count), m_hostData(std::move(other.m_hostData)) {
        other.m_count = 0;
    }
    
    // Move assignment
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            m_count = other.m_count;
            m_hostData = std::move(other.m_hostData);
            other.m_count = 0;
        }
        return *this;
    }
    
    // Disable copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    ~DeviceBuffer() = default;
    
    // Resize buffer
    void resize(size_t new_count) {
        m_count = new_count;
        m_hostData.resize(new_count);
    }
    
    // Get size
    size_t size() const { return m_count; }
    
    // Get raw pointer (stub)
    T* data() { return m_hostData.data(); }
    const T* data() const { return m_hostData.data(); }

private:
    size_t m_count;
    std::vector<T> m_hostData; // Using host memory as stub
};

} // namespace cudarl
