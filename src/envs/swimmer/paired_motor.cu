#include "paired_motor.h"

namespace cudarl::paired_motor {
namespace {
__global__ void reset(State* states,float* obs,uint32_t* invalid,uint32_t* safety,int n,uint32_t seed) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;states[i]=initial(seed,static_cast<uint32_t>(i));invalid[i]=safety[i]=0;observe(states[i],obs+i*kObs);
}
__global__ void observations(const State* states,float* obs,int n) {const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) observe(states[i],obs+i*kObs);}
__global__ void stepping(State* states,float* obs,uint32_t* invalid,uint32_t* safety,State* first,const float* latent,swimmer::Outputs out,
        int n,uint32_t seed,Interface mode,Gains gains,bool autoreset,const deep::Physics* per_world=nullptr,const uint8_t* active=nullptr) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;State state=states[i];command_motor::Result result;
    if((!active||active[i])&&invalid[i]==0&&safety[i]==0&&(autoreset||state.task.steps<kEpisodeSteps)) result=advance(state,latent+i*4,mode,gains,per_world?per_world[i]:command_motor::physics());
    const bool end=result.timeout||result.invalid||result.unsafe;
    if(out.reward) out.reward[i]=result.reward;if(out.done) out.done[i]=end?1.f:0.f;if(out.truncated) out.truncated[i]=0;
    if(result.invalid) ++invalid[i];if(result.unsafe) {if(safety[i]==0) first[i]=state;safety[i]+=static_cast<uint32_t>(state.unsafe_substeps);}
    if(end&&out.final_obs) {if(result.invalid) for(int j=0;j<kObs;++j) out.final_obs[i*kObs+j]=0;else observe(state,out.final_obs+i*kObs);}
    // Only ordinary finite600-tick episode boundaries may reset. A failed
    // physical world remains preserved and frozen until the research abort.
    if(result.timeout&&!result.invalid&&!result.unsafe&&autoreset) state=initial(seed,static_cast<uint32_t>(i),state.task.episode+1);
    states[i]=state;if(result.invalid) for(int j=0;j<kObs;++j) obs[i*kObs+j]=0;else observe(state,obs+i*kObs);
}
uint64_t counts(const device_ptr<uint32_t>& pointer,int n,cudaStream_t stream) {
    std::vector<uint32_t> values(n);CUDA_CHECK_THROW(cudaMemcpyAsync(values.data(),pointer.get(),values.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));uint64_t sum=0;for(uint32_t x:values) sum+=x;return sum;
}
}
Batch::Batch(int n,uint32_t seed,swimmer::Body body):n_(n),seed_(seed) {
    (void)body;if(n<=0) throw std::invalid_argument("Paired motor world count must be positive");states_.allocate(n);first_unsafe_.allocate(n);observations_.allocate(static_cast<size_t>(n)*kObs);invalid_.allocate(n);safety_.allocate(n);
    CUDA_CHECK_THROW(cudaMemset(first_unsafe_.get(),0,first_unsafe_.size_bytes()));reset<<<(n+63)/64,64>>>(states_.get(),observations_.get(),invalid_.get(),safety_.get(),n,seed);CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());
}
void Batch::configure(Interface mode,const Gains& gains,cudaStream_t stream) {
    if(!valid_gains(gains,mode)) throw std::invalid_argument("Invalid paired motor interface/SI gains");CUDA_CHECK_THROW(cudaStreamSynchronize(stream));interface_=mode;gains_=gains;
}
void Batch::step(const float* latent,swimmer::Outputs out,cudaStream_t stream,bool autoreset) {
    stepping<<<(n_+63)/64,64,0,stream>>>(states_.get(),observations_.get(),invalid_.get(),safety_.get(),first_unsafe_.get(),latent,out,n_,seed_,interface_,gains_,autoreset);CUDA_CHECK_THROW(cudaGetLastError());
}
void Batch::step_physics(const float* latent,const deep::Physics* physical,const uint8_t* active,cudaStream_t stream) {
    stepping<<<(n_+63)/64,64,0,stream>>>(states_.get(),observations_.get(),invalid_.get(),safety_.get(),first_unsafe_.get(),latent,{},n_,seed_,interface_,gains_,false,physical,active);CUDA_CHECK_THROW(cudaGetLastError());
}
std::vector<State> Batch::host_states(cudaStream_t stream) const {
    std::vector<State> result(n_);CUDA_CHECK_THROW(cudaMemcpyAsync(result.data(),states_.get(),result.size()*sizeof(State),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));return result;
}
void Batch::set_states(const std::vector<State>& states,cudaStream_t stream) {
    if(states.size()!=static_cast<size_t>(n_)) throw std::invalid_argument("Paired motor population mismatch");for(const auto& s:states) if(!finite(s)||s.task.steps>kEpisodeSteps) throw std::invalid_argument("Invalid paired motor uploaded state");
    CUDA_CHECK_THROW(cudaMemcpyAsync(states_.get(),states.data(),states.size()*sizeof(State),cudaMemcpyHostToDevice,stream));CUDA_CHECK_THROW(cudaMemsetAsync(invalid_.get(),0,invalid_.size_bytes(),stream));CUDA_CHECK_THROW(cudaMemsetAsync(safety_.get(),0,safety_.size_bytes(),stream));CUDA_CHECK_THROW(cudaMemsetAsync(first_unsafe_.get(),0,first_unsafe_.size_bytes(),stream));
    observations<<<(n_+63)/64,64,0,stream>>>(states_.get(),observations_.get(),n_);CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
uint64_t Batch::invalid_count(cudaStream_t stream) const {return counts(invalid_,n_,stream);}
uint64_t Batch::unsafe_count(cudaStream_t stream) const {return counts(safety_,n_,stream);}
std::vector<State> Batch::unsafe_states(cudaStream_t stream) const {
    std::vector<uint32_t> amount(n_);std::vector<State> first(n_),out;CUDA_CHECK_THROW(cudaMemcpyAsync(amount.data(),safety_.get(),amount.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaMemcpyAsync(first.data(),first_unsafe_.get(),first.size()*sizeof(State),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));for(int i=0;i<n_;++i) if(amount[i]) out.push_back(first[i]);return out;
}
void Batch::save(checkpoint::Writer& out) const {
    CUDA_CHECK_THROW(cudaDeviceSynchronize());std::vector<State> states(n_),first(n_);std::vector<float> obs(static_cast<size_t>(n_)*kObs);std::vector<uint32_t> invalid(n_),safety(n_);
    states_.copy_to_host(states.data(),states.size());first_unsafe_.copy_to_host(first.data(),first.size());observations_.copy_to_host(obs.data(),obs.size());invalid_.copy_to_host(invalid.data(),invalid.size());safety_.copy_to_host(safety.data(),safety.size());
    out.value(uint32_t{0x504D4241u});out.value(kProtocol);out.value(uint32_t{sizeof(State)});out.value(n_);out.value(seed_);out.value(interface_);out.value(gains_);out.array(states);out.array(obs);out.array(invalid);out.array(safety);out.array(first);
}
void Batch::load(checkpoint::Reader& in) {
    if(in.value<uint32_t>()!=0x504D4241u||in.value<uint32_t>()!=kProtocol||in.value<uint32_t>()!=sizeof(State)||in.value<int>()!=n_||in.value<uint32_t>()!=seed_) throw std::runtime_error("Incompatible paired motor batch");
    const auto mode=in.value<Interface>();const auto gains=in.value<Gains>();if(mode!=interface_||!same_gains(gains,gains_)||!valid_gains(gains,mode)) throw std::runtime_error("Paired motor saved actuator mapping differs");
    const auto states=in.array<State>(n_);const auto obs=in.array<float>(static_cast<size_t>(n_)*kObs);const auto invalid=in.array<uint32_t>(n_),safety=in.array<uint32_t>(n_);const auto first=in.array<State>(n_);
    if(states.size()!=static_cast<size_t>(n_)||obs.size()!=static_cast<size_t>(n_)*kObs||invalid.size()!=states.size()||safety.size()!=states.size()||first.size()!=states.size()) throw std::runtime_error("Paired motor saved dimensions differ");
    for(int i=0;i<n_;++i) {
        if(!finite(states[i])||states[i].task.steps>=kEpisodeSteps||states[i].task.seed!=seed_||states[i].task.environment!=static_cast<uint32_t>(i)||invalid[i]||safety[i]) throw std::runtime_error("Unsafe or inconsistent paired motor saved world");
        float restored[kObs];observe(states[i],restored);for(int j=0;j<kObs;++j) if(!command_motor::finite_number(obs[static_cast<size_t>(i)*kObs+j])||fabsf(restored[j]-obs[static_cast<size_t>(i)*kObs+j])>1e-5f) throw std::runtime_error("Paired motor saved observation disagrees with state");
    }
    states_.copy_from_host(states.data(),states.size());first_unsafe_.copy_from_host(first.data(),first.size());observations_.copy_from_host(obs.data(),obs.size());invalid_.copy_from_host(invalid.data(),invalid.size());safety_.copy_from_host(safety.data(),safety.size());CUDA_CHECK_THROW(cudaDeviceSynchronize());
}
}
