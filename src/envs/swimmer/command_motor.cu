#include "command_motor.h"
#include <stdexcept>

namespace cudarl::command_motor {
namespace {
__global__ void reset_kernel(State* states,float* observations,uint32_t* invalid,uint32_t* safety,int n,uint32_t seed) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    states[i]=initial(seed,static_cast<uint32_t>(i));invalid[i]=0;safety[i]=0;observe(states[i],observations+i*kObs);
}
__global__ void observe_kernel(const State* states,float* observations,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) observe(states[i],observations+i*kObs);
}
__global__ void step_kernel(State* states,float* observations,uint32_t* invalid,uint32_t* safety,State* first_unsafe,const float* actions,
        swimmer::Outputs out,int n,uint32_t seed,bool autoreset,const deep::Physics* per_world=nullptr,const uint8_t* active=nullptr) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    State state=states[i];Result result;
    const bool stepping=(!active||active[i])&&(autoreset||state.steps<kEpisodeSteps);
    // A physical contact response can move an illegal starting pose back
    // inside during the tick. Preserve that initial failure rather than
    // reporting only its corrected endpoint as a safe rollout.
    const bool unsafe_start=stepping&&finite(state)&&physical_clearance(state.body)<0;
    if(unsafe_start&&safety[i]==0) first_unsafe[i]=state;
    if(stepping) result=advance(state,actions+i*kAct,per_world?per_world[i]:physics());
    const bool end=result.timeout||result.invalid;
    if(out.reward) out.reward[i]=result.reward;
    if(out.done) out.done[i]=end?1.f:0.f;
    // The task exposes its finite600-tick horizon in the observation. Reaching
    // that horizon is a terminal, never an external time-limit truncation.
    if(out.truncated) out.truncated[i]=0;
    if(result.invalid) ++invalid[i];
    if(unsafe_start||result.unsafe) {if(safety[i]==0&&!unsafe_start) first_unsafe[i]=state;++safety[i];}
    if(end&&out.final_obs) {
        if(result.invalid) for(int j=0;j<kObs;++j) out.final_obs[i*kObs+j]=0;
        else observe(state,out.final_obs+i*kObs);
    }
    if(end&&autoreset) state=initial(seed,static_cast<uint32_t>(i),state.episode+1);
    states[i]=state;
    if(result.invalid&&!autoreset) for(int j=0;j<kObs;++j) observations[i*kObs+j]=0;
    else observe(state,observations+i*kObs);
}
}
Batch::Batch(int n,uint32_t seed,swimmer::Body unused):n_(n),seed_(seed) {
    (void)unused;if(n<=0) throw std::invalid_argument("Command motor population must be positive");
    states_.allocate(n);obs_.allocate(static_cast<size_t>(n)*kObs);invalid_.allocate(n);safety_.allocate(n);first_unsafe_.allocate(n);
    CUDA_CHECK_THROW(cudaMemset(first_unsafe_.get(),0,first_unsafe_.size_bytes()));
    reset_kernel<<<(n+63)/64,64>>>(states_.get(),obs_.get(),invalid_.get(),safety_.get(),n,seed_);
    CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());
}
void Batch::step(const float* actions,swimmer::Outputs out,cudaStream_t stream,bool autoreset) {
    step_kernel<<<(n_+63)/64,64,0,stream>>>(states_.get(),obs_.get(),invalid_.get(),safety_.get(),first_unsafe_.get(),actions,out,n_,seed_,autoreset);
    CUDA_CHECK_THROW(cudaGetLastError());
}
void Batch::step_physics(const float* actions,const deep::Physics* per_world,const uint8_t* active,cudaStream_t stream) {
    step_kernel<<<(n_+63)/64,64,0,stream>>>(states_.get(),obs_.get(),invalid_.get(),safety_.get(),first_unsafe_.get(),actions,{},n_,seed_,false,per_world,active);
    CUDA_CHECK_THROW(cudaGetLastError());
}
void Batch::set_states(const std::vector<State>& states,cudaStream_t stream) {
    if(states.size()!=static_cast<size_t>(n_)) throw std::invalid_argument("Command motor state population mismatch");
    for(const auto& state:states) if(!finite(state)||state.steps>kEpisodeSteps) throw std::invalid_argument("Invalid command motor state");
    // Nonblocking streams do not inherit default-stream ordering. Keep the
    // upload, diagnostic resets and observation consumer on this exact stream.
    // The final sync also keeps the caller's stable/pinned host vector alive
    // until its asynchronous upload is complete. This dependency pre-existed
    // the focused regression test; it is not a newly repaired physics defect.
    CUDA_CHECK_THROW(cudaMemcpyAsync(states_.get(),states.data(),states.size()*sizeof(State),cudaMemcpyHostToDevice,stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(invalid_.get(),0,invalid_.size_bytes(),stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(safety_.get(),0,safety_.size_bytes(),stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(first_unsafe_.get(),0,first_unsafe_.size_bytes(),stream));
    observe_kernel<<<(n_+63)/64,64,0,stream>>>(states_.get(),obs_.get(),n_);CUDA_CHECK_THROW(cudaGetLastError());
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
std::vector<State> Batch::host_states(cudaStream_t stream) const {
    std::vector<State> states(n_);
    CUDA_CHECK_THROW(cudaMemcpyAsync(states.data(),states_.get(),states.size()*sizeof(State),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));return states;
}
uint64_t Batch::invalid_count(cudaStream_t stream) const {
    std::vector<uint32_t> counts(n_);
    CUDA_CHECK_THROW(cudaMemcpyAsync(counts.data(),invalid_.get(),counts.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));uint64_t total=0;for(auto count:counts) total+=count;return total;
}
uint64_t Batch::unsafe_count(cudaStream_t stream) const {
    std::vector<uint32_t> counts(n_);
    CUDA_CHECK_THROW(cudaMemcpyAsync(counts.data(),safety_.get(),counts.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));uint64_t total=0;for(auto count:counts) total+=count;return total;
}
std::vector<State> Batch::unsafe_states(cudaStream_t stream) const {
    std::vector<uint32_t> counts(n_);std::vector<State> first(n_),result;
    CUDA_CHECK_THROW(cudaMemcpyAsync(counts.data(),safety_.get(),counts.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK_THROW(cudaMemcpyAsync(first.data(),first_unsafe_.get(),first.size()*sizeof(State),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    for(int i=0;i<n_;++i) if(counts[static_cast<size_t>(i)]>0) result.push_back(first[static_cast<size_t>(i)]);return result;
}
void Batch::save(checkpoint::Writer& out) const {
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    std::vector<State> states(n_);std::vector<uint32_t> invalid(n_);std::vector<float> observations(static_cast<size_t>(n_)*kObs);
    states_.copy_to_host(states.data(),states.size());invalid_.copy_to_host(invalid.data(),invalid.size());obs_.copy_to_host(observations.data(),observations.size());
    std::vector<uint32_t> safety(n_);std::vector<State> first(n_);
    safety_.copy_to_host(safety.data(),safety.size());first_unsafe_.copy_to_host(first.data(),first.size());
    out.value(n_);out.value(seed_);out.array(states);out.array(invalid);out.array(observations);out.array(safety);out.array(first);
}
void Batch::load(checkpoint::Reader& in) {
    if(in.value<int>()!=n_||in.value<uint32_t>()!=seed_) throw std::runtime_error("Command motor population identity mismatch");
    const auto states=in.array<State>(n_);const auto invalid=in.array<uint32_t>(n_);const auto observations=in.array<float>(static_cast<size_t>(n_)*kObs);
    const auto safety=in.array<uint32_t>(n_);const auto first=in.array<State>(n_);
    if(states.size()!=static_cast<size_t>(n_)||invalid.size()!=states.size()||observations.size()!=static_cast<size_t>(n_)*kObs||safety.size()!=states.size()||first.size()!=states.size())
        throw std::runtime_error("Invalid command motor snapshot dimensions");
    for(int i=0;i<n_;++i) if(!finite(states[i])||states[i].steps>=kEpisodeSteps||states[i].seed!=seed_||states[i].environment!=static_cast<uint32_t>(i)||invalid[i]!=0)
        throw std::runtime_error("Command motor snapshot contains invalid world/counters");
    for(uint32_t count:safety) if(count!=0) throw std::runtime_error("Cannot continue an unsafe command motor rollout snapshot");
    for(float value:observations) if(!finite_number(value)) throw std::runtime_error("Nonfinite command motor observation");
    for(int i=0;i<n_;++i) {
        float reconstructed[kObs];observe(states[static_cast<size_t>(i)],reconstructed);
        for(int j=0;j<kObs;++j) if(fabsf(reconstructed[j]-observations[static_cast<size_t>(i)*kObs+j])>1e-5f)
            throw std::runtime_error("Command motor saved observations disagree with their uploaded states");
    }
    states_.copy_from_host(states.data(),states.size());invalid_.copy_from_host(invalid.data(),invalid.size());obs_.copy_from_host(observations.data(),observations.size());
    safety_.copy_from_host(safety.data(),safety.size());first_unsafe_.copy_from_host(first.data(),first.size());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
}
}
