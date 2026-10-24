#ifndef CUDA_ARENA_H
#define CUDA_ARENA_H

#include <cstdint>
#include <vector>
#include <string>
#include "cuda_memory.h"

class CudaArena {
public:
    explicit CudaArena(int num_envs = 1000);
    ~CudaArena();
    
    // Rule of Five - prevent accidental copying
    CudaArena(const CudaArena&) = delete;
    CudaArena& operator=(const CudaArena&) = delete;
    CudaArena(CudaArena&&) = delete;
    CudaArena& operator=(CudaArena&&) = delete;
    
    void reset_environments(uint32_t seed = 1234);
    // actions.size() must equal get_num_environments(); throws std::invalid_argument otherwise.
    void step_environments(const std::vector<int>& actions);
    
    std::vector<float> get_observations() const;
    std::vector<float> get_rewards() const;
    std::vector<int> get_dones() const;
    
    int get_num_environments() const { return m_num_envs; }
    
    // Static CUDA utility functions; these never throw and return 0 / "" on failure
    static int get_device_count();
    static std::string get_device_name(int device_id);
    void hello_cuda();  // throws cuda_error on failure
    
private:
    int m_num_envs;
    
    // Device memory with RAII
    device_ptr<float> d_observations;
    device_ptr<float> d_rewards;
    device_ptr<int> d_dones;
    device_ptr<int> d_actions;
    
    // Host memory
    std::vector<float> h_observations;
    std::vector<float> h_rewards;
    std::vector<int> h_dones;
    
    void allocate_memory();
    void free_memory();
};

#endif // CUDA_ARENA_H