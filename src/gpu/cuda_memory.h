#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>

#define CUDA_CHECK_THROW(expr) do {                                      \
    cudaError_t _err = (expr);                                           \
    if (_err != cudaSuccess) {                                           \
        throw cuda_error(#expr, _err, __FILE__, __LINE__);              \
    }                                                                    \
} while (0)

// Custom exception for CUDA errors
class cuda_error : public std::runtime_error {
public:
    cuda_error(const char* expr, cudaError_t err, const char* file, int line)
        : std::runtime_error(build_error_message(expr, err, file, line)),
          error_code(err) {}
    
    cudaError_t get_error_code() const { return error_code; }
    
private:
    cudaError_t error_code;
    
    static std::string build_error_message(const char* expr, cudaError_t err, 
                                           const char* file, int line) {
        return std::string("CUDA error in ") + file + ":" + std::to_string(line) +
               ": " + expr + " failed with " + cudaGetErrorString(err);
    }
};

// RAII wrapper for device memory
template<typename T>
class device_ptr {
public:
    device_ptr() = default;
    
    explicit device_ptr(size_t count) : count_(count) {
        if (count > 0) {
            CUDA_CHECK_THROW(cudaMalloc(&ptr_, count * sizeof(T)));
        }
    }
    
    ~device_ptr() {
        reset();
    }
    
    // Move constructor
    device_ptr(device_ptr&& other) noexcept 
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    
    // Move assignment
    device_ptr& operator=(device_ptr&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    // Delete copy operations
    device_ptr(const device_ptr&) = delete;
    device_ptr& operator=(const device_ptr&) = delete;
    
    // Reset the pointer
    void reset() {
        if (ptr_) {
            cudaFree(ptr_);  // Ignore error in destructor path
            ptr_ = nullptr;
            count_ = 0;
        }
    }
    
    // Allocate new memory
    void allocate(size_t new_count) {
        reset();
        count_ = new_count;
        if (count_ > 0) {
            CUDA_CHECK_THROW(cudaMalloc(&ptr_, count_ * sizeof(T)));
        }
    }
    
    // Copy from host to device
    void copy_from_host(const T* host_ptr, size_t elements) {
        if (!ptr_ || elements > count_) {
            throw std::runtime_error("Invalid copy_from_host: insufficient device memory");
        }
        CUDA_CHECK_THROW(cudaMemcpy(ptr_, host_ptr, elements * sizeof(T), 
                                    cudaMemcpyHostToDevice));
    }
    
    // Copy from device to host
    void copy_to_host(T* host_ptr, size_t elements) const {
        if (!ptr_ || elements > count_) {
            throw std::runtime_error("Invalid copy_to_host: insufficient device memory");
        }
        CUDA_CHECK_THROW(cudaMemcpy(host_ptr, ptr_, elements * sizeof(T), 
                                    cudaMemcpyDeviceToHost));
    }
    
    // Getters
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return count_; }
    size_t size_bytes() const { return count_ * sizeof(T); }
    
    // Conversion to raw pointer for kernel calls
    operator T*() { return ptr_; }
    operator const T*() const { return ptr_; }
    
private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// Helper function to create device_ptr
template<typename T>
device_ptr<T> make_device_ptr(size_t count) {
    return device_ptr<T>(count);
}

#endif // CUDA_MEMORY_H