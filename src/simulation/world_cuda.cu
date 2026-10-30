#include "world.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cudarl::simulation {
namespace {
void check(cudaError_t error) {if(error!=cudaSuccess) throw std::runtime_error(std::string("Foundation CUDA: ")+cudaGetErrorString(error));}
__global__ void step_kernel(WorldState* states,const float* inputs,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) step_state(states[i],inputs+4*i);
}
}
void cuda_step(std::vector<WorldState>& states,const std::vector<std::array<float,4>>& inputs,void* opaque) {
    if(states.size()!=inputs.size()) throw std::runtime_error("Physical CUDA input count differs");
    if(states.empty()) return;
    if(states.size()>65536) throw std::runtime_error("Physical CUDA diagnostic batch exceeds bound");
    const auto stream=static_cast<cudaStream_t>(opaque);
    WorldState* device_states=nullptr;float* device_inputs=nullptr;
    try {
        check(cudaMalloc(reinterpret_cast<void**>(&device_states),states.size()*sizeof(WorldState)));
        check(cudaMalloc(reinterpret_cast<void**>(&device_inputs),inputs.size()*4*sizeof(float)));
        // Every transfer and consumer uses the same supplied stream.
        check(cudaMemcpyAsync(device_states,states.data(),states.size()*sizeof(WorldState),cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(device_inputs,inputs.data(),inputs.size()*4*sizeof(float),cudaMemcpyHostToDevice,stream));
        step_kernel<<<static_cast<int>((states.size()+31)/32),32,0,stream>>>(device_states,device_inputs,static_cast<int>(states.size()));
        check(cudaGetLastError());
        check(cudaMemcpyAsync(states.data(),device_states,states.size()*sizeof(WorldState),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
    } catch(...) {cudaStreamSynchronize(stream);cudaFree(device_inputs);cudaFree(device_states);throw;}
    cudaFree(device_inputs);cudaFree(device_states);
}
}
