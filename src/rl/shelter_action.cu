#include "envs/swimmer/shelter_motor.h"
#include <cmath>
#include <stdexcept>

namespace cudarl::shelter_motor {
namespace {
constexpr int action_threads=128;
__global__ void means_kernel(const float* raw,const float* observations,float* means,int n,ActuatorScale scale) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n*kAct) return;
    const bool cover=observations[(i/kAct)*kObs+55]>=.5f;
    means[i]=cover?raw[i]/scale.value[i%kAct]:raw[i];
}
__global__ void mean_gradient_kernel(float* gradients,const float* observations,int n,ActuatorScale scale) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n*kAct && observations[(i/kAct)*kObs+55]>=.5f) gradients[i]/=scale.value[i%kAct];
}
void validate_transform(int n,ActuatorScale scale) {
    if(n<0) throw std::invalid_argument("Negative shelter policy population");
    for(float value:scale.value)
        if(!std::isfinite(value) || value<1.f || value>1048576.f)
            throw std::invalid_argument("Shelter actuator scales must be finite and within the saved protocol range");
}
}
void policy_means(const float* raw,const float* observations,float* means,int n,ActuatorScale scale,cudaStream_t stream) {
    validate_transform(n,scale);if(n==0) return;
    means_kernel<<<(n*kAct+action_threads-1)/action_threads,action_threads,0,stream>>>(raw,observations,means,n,scale);
    CUDA_CHECK_THROW(cudaGetLastError());
}
void policy_mean_grad(float* gradients,const float* observations,int n,ActuatorScale scale,cudaStream_t stream) {
    validate_transform(n,scale);if(n==0) return;
    mean_gradient_kernel<<<(n*kAct+action_threads-1)/action_threads,action_threads,0,stream>>>(gradients,observations,n,scale);
    CUDA_CHECK_THROW(cudaGetLastError());
}
}
