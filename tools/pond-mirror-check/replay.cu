#include "cuda/pond_math_profile.h"
#include "mirror.h"
#include <cuda_runtime.h>
#include <stdexcept>
namespace mirror {
namespace {
void check(cudaError_t value) {
    if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value));
}
__global__ void replay(GpuPair* pairs, runtime::SubstepTrace* traces, int t) {
    const int id = blockIdx.x;
    if (threadIdx.x != 0 || id >= num_cases) return;
    auto& pair = pairs[id];
    if (pair.stats.failure) return;
    auto* a = traces + id*40;
    auto* b = a + 20;
    float other[4];
    for (int j=0; j<4; ++j) other[j]=joint_sign(j)*pair.commands[t][j];
    const bool accepted_a = runtime::step(pair.worlds[0],pair.commands[t],.005f,a);
    const bool accepted_b = runtime::step(pair.worlds[1],other,.005f,b);
    if (!accepted_a || !accepted_b) { pair.stats.failure=1; return; }
    for (int k=0;k<10;++k) pair.stats.compare_step(a[k],b[k]);
}
}
void replay_cuda(GpuPair* pairs) {
    cudarl::pond_math_profile::validate_host_geometry();
    GpuPair* device = nullptr;
    runtime::SubstepTrace* traces = nullptr;
    check(cudaMalloc(&device,num_cases*sizeof(GpuPair)));
    check(cudaMalloc(&traces,num_cases*40*sizeof(*traces)));
    check(cudaMemcpy(device,pairs,num_cases*sizeof(GpuPair),cudaMemcpyHostToDevice));
    for (int t=0;t<gpu_controls;++t) {
        replay<<<num_cases,1>>>(device,traces,t);
        check(cudaGetLastError());
    }
    check(cudaMemcpy(pairs,device,num_cases*sizeof(GpuPair),cudaMemcpyDeviceToHost));
    check(cudaFree(traces));
    check(cudaFree(device));
}
}
