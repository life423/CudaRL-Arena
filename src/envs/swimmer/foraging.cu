#include "foraging.h"
#include <stdexcept>
namespace cudarl::foraging {
namespace {
__global__ void reset_kernel(State* states,uint32_t* episodes,float* obs,int n,uint32_t seed,swimmer::Body body) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    episodes[i]=0;states[i]=foraging::initial(seed,static_cast<uint32_t>(i),0,body);
    observe(states[i],obs+i*kObs);
}
__global__ void step_kernel(State* states,uint32_t* episodes,float* obs,const float* actions,
                            swimmer::Outputs out,int n,uint32_t seed,bool autoreset) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    State s=states[i];Result result{};
    if(autoreset || s.body.steps<kEpisodeSteps) result=advance(s,actions+i*swimmer::kAct);
    if(out.reward) out.reward[i]=result.reward;
    if(out.done) out.done[i]=result.timeout?1.f:0.f;
    if(out.truncated) out.truncated[i]=result.timeout?1.f:0.f;
    if(result.timeout && out.final_obs) observe(s,out.final_obs+i*kObs);
    if(result.timeout && autoreset) s=foraging::initial(seed,static_cast<uint32_t>(i),++episodes[i],s.body.body);
    states[i]=s;observe(s,obs+i*kObs);
}
}
Batch::Batch(int n,uint32_t seed,swimmer::Body body):n_(n),seed_(seed) {
    if(n<=0) throw std::invalid_argument("Foraging batch must be positive");
    states_.allocate(n);episodes_.allocate(n);obs_.allocate(static_cast<size_t>(n)*kObs);
    reset_kernel<<<(n+127)/128,128>>>(states_.get(),episodes_.get(),obs_.get(),n,seed_,body);
    CUDA_CHECK_THROW(cudaGetLastError());
}
void Batch::step(const float* actions,swimmer::Outputs out,cudaStream_t stream,bool autoreset) {
    step_kernel<<<(n_+127)/128,128,0,stream>>>(states_.get(),episodes_.get(),obs_.get(),actions,out,n_,seed_,autoreset);
    CUDA_CHECK_THROW(cudaGetLastError());
}
}
