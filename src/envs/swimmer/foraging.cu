#include "foraging.h"
#include <stdexcept>
namespace cudarl::foraging {
namespace {
__global__ void shuffle_kernel(float* obs,int rows,int environments,uint32_t seed,uint32_t first_tick) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows) return;
    const uint32_t key=rng::bits(seed^0xA6F029B3u,static_cast<uint32_t>(i%environments),first_tick+static_cast<uint32_t>(i/environments));
    permute_food_inputs(obs+i*kObs,static_cast<int>(key%6u));
}
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
void shuffle_food_inputs(float* obs,int rows,int environments,uint32_t seed,uint32_t first_tick,cudaStream_t stream) {
    if(rows<=0 || environments<=0) throw std::invalid_argument("Invalid observation shuffle dimensions");
    shuffle_kernel<<<(rows+255)/256,256,0,stream>>>(obs,rows,environments,seed,first_tick);
    CUDA_CHECK_THROW(cudaGetLastError());
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
void Batch::save(checkpoint::Writer& out) const {
    std::vector<State> states(n_);std::vector<uint32_t> episodes(n_);std::vector<float> obs(static_cast<size_t>(n_)*kObs);
    states_.copy_to_host(states.data(),states.size());episodes_.copy_to_host(episodes.data(),episodes.size());obs_.copy_to_host(obs.data(),obs.size());
    out.array(states);out.array(episodes);out.array(obs);
}
void Batch::load(checkpoint::Reader& in) {
    auto states=in.array<State>(n_);auto episodes=in.array<uint32_t>(n_);auto obs=in.array<float>(static_cast<size_t>(n_)*kObs);
    if(states.size()!=static_cast<size_t>(n_) || episodes.size()!=states.size() || obs.size()!=static_cast<size_t>(n_)*kObs)
        throw std::runtime_error("Incompatible background-world checkpoint");
    for(float v:obs) if(!std::isfinite(v)) throw std::runtime_error("Invalid background-world observation");
    for(int i=0;i<n_;++i) if(states[i].seed!=seed_ || states[i].environment!=static_cast<uint32_t>(i) ||
        states[i].episode!=episodes[i] || states[i].body.steps<0 || states[i].body.steps>=kEpisodeSteps)
        throw std::runtime_error("Invalid background-world counters");
    states_.copy_from_host(states.data(),states.size());episodes_.copy_from_host(episodes.data(),episodes.size());obs_.copy_from_host(obs.data(),obs.size());
}
}
