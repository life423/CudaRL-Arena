#ifndef CUDARL_DEMOS_POND_CUDA_API_H
#define CUDARL_DEMOS_POND_CUDA_API_H
#include <cstddef>
#include <cstdint>
#include "envs/swimmer/deep_pond.h"
namespace cudarl::demos {
// Versioned POD interface: no CUDA headers, STL allocations or exceptions cross
// the optional module boundary. The caller owns every output/error buffer.
struct PondCudaApi {
    uint32_t version;
    uint32_t api_size;
    uint32_t state_size;
    void* (*create)(const float* motor,int motor_count,const float* selector,int selector_count,int period,
        uint32_t seed,uint32_t environment,char* message,size_t message_size);
    int (*step)(void* handle,deep_pond::State* state,int* target,char* error,size_t error_size);
    void (*destroy)(void* handle);
};
using GetPondCudaApi=const PondCudaApi* (*)();
}
#endif
