#include "pond_selector.h"
namespace cudarl::pond_selector {
namespace {
CUDARL_HD deep_pond::State initial(uint32_t seed,uint32_t environment,uint32_t episode,bool recovery) {
    auto s=deep_pond::initial(seed,environment,episode);
    if(recovery) {
        const uint32_t key=episode*23u;const float kind=rng::uniform(seed^0x42AF8B91u,environment,key);
        if(kind<.25f) {
            const float sign=rng::uniform(seed^0x42AF8B91u,environment,key+1)<.5f?-1.f:1.f;
            for(int j=0;j<4;++j) s.body.angle[j]=sign*deep::limit*(.85f+.14f*rng::uniform(seed^0x42AF8B91u,environment,key+2+j));
        } else if(kind<.5f) for(int j=0;j<4;++j) s.body.angle[j]=deep::limit*.98f*(2*rng::uniform(seed^0x42AF8B91u,environment,key+2+j)-1);
        for(int f=0;f<3;++f) {s.food[f]={99,99,99};s.placements[f]=0;}
        for(int f=0;f<3;++f) deep_pond::place_food(s,f);
    }
    return s;
}
__global__ void feature_kernel(const float* observations,float* output,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*3) return;features(observations+(i/3)*kObs,i%3,output+i*kFeatures);
}
__global__ void reset(deep_pond::State* states,uint32_t* episodes,int* targets,float* observations,int n,uint32_t seed,bool recovery) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    states[i]=initial(seed,static_cast<uint32_t>(i),0,recovery);episodes[i]=0;targets[i]=-1;observe(states[i],-1,observations+i*kObs);
}
__global__ void inputs(const deep_pond::State* states,const int* targets,float* output,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    auto* row=output+i*deep_pond::kObs;deep_pond::observe(states[i],row);
    demos::primary_pond_food(row,targets[i]<0?0:targets[i]);
}
__global__ void begin_kernel(const int* actions,int* targets,swimmer::Outputs out,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    targets[i]=actions[i];if(out.reward) out.reward[i]=0;if(out.done) out.done[i]=0;if(out.truncated) out.truncated[i]=0;
}
__global__ void advance_kernel(deep_pond::State* states,uint32_t* episodes,int* targets,float* observations,
                               const float* torques,swimmer::Outputs out,int n,uint32_t seed,bool autoreset,int episode_steps,int decision_steps,bool recovery) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    auto s=states[i];if(!autoreset && s.steps>=deep_pond::kEpisodeSteps) return;
    const auto result=deep_pond::advance(s,torques+i*4);
    const bool failed=autoreset && recovery && s.steps%decision_steps==0 && s.since_food>=400 &&
        deep::norm({s.body.velocity[0],s.body.velocity[1],s.body.velocity[2]})<.005f;
    // High-level PPO receives actual collection only. No nearest-target labels,
    // distance shaping, direct forces or changes to motor/physical dynamics.
    if(out.reward && result.eaten_slot>=0) out.reward[i]+=1.f;
    if(out.reward && failed) out.reward[i]-=5.f;
    if(s.steps>=episode_steps || failed) {
        if(out.done) out.done[i]=1;if(out.truncated) out.truncated[i]=failed?0.f:1.f;
        if(out.final_obs) observe(s,targets[i],out.final_obs+i*kObs);
        if(autoreset) {s=initial(seed,static_cast<uint32_t>(i),++episodes[i],recovery);targets[i]=-1;}
    }
    states[i]=s;observe(s,targets[i],observations+i*kObs);
}
}
void make_features(const float* observations,float* output,int n,cudaStream_t stream) {
    feature_kernel<<<(n*3+127)/128,128,0,stream>>>(observations,output,n);CUDA_CHECK_THROW(cudaGetLastError());
}
Batch::Batch(int n,uint32_t seed,int decision_steps,bool recovery):n_(n),decision_steps_(decision_steps),episode_steps_(recovery?6000:1200),seed_(seed),recovery_(recovery) {
    if(n<1 || decision_steps<1 || episode_steps_%decision_steps!=0) throw std::invalid_argument("Selector decision period must divide the physical episode");
    states_.allocate(n);episodes_.allocate(n);targets_.allocate(n);observations_.allocate(static_cast<size_t>(n)*kObs);motor_inputs_.allocate(static_cast<size_t>(n)*deep_pond::kObs);
    reset<<<(n+63)/64,64>>>(states_.get(),episodes_.get(),targets_.get(),observations_.get(),n,seed,recovery);CUDA_CHECK_THROW(cudaGetLastError());
}
const float* Batch::motor_obs(cudaStream_t stream) {
    inputs<<<(n_+127)/128,128,0,stream>>>(states_.get(),targets_.get(),motor_inputs_.get(),n_);CUDA_CHECK_THROW(cudaGetLastError());return motor_inputs_.get();
}
void Batch::begin(const int* targets,swimmer::Outputs out,cudaStream_t stream) {
    begin_kernel<<<(n_+127)/128,128,0,stream>>>(targets,targets_.get(),out,n_);CUDA_CHECK_THROW(cudaGetLastError());
}
void Batch::step(const float* torques,swimmer::Outputs out,cudaStream_t stream,bool autoreset) {
    advance_kernel<<<(n_+63)/64,64,0,stream>>>(states_.get(),episodes_.get(),targets_.get(),observations_.get(),torques,out,n_,seed_,autoreset,episode_steps_,decision_steps_,recovery_);CUDA_CHECK_THROW(cudaGetLastError());
}
void Batch::save(checkpoint::Writer& out) const {
    std::vector<deep_pond::State> states(n_);std::vector<uint32_t> episodes(n_);std::vector<int> targets(n_);std::vector<float> obs(static_cast<size_t>(n_)*kObs);
    states_.copy_to_host(states.data(),states.size());episodes_.copy_to_host(episodes.data(),episodes.size());targets_.copy_to_host(targets.data(),targets.size());observations_.copy_to_host(obs.data(),obs.size());
    out.array(states);out.array(episodes);out.array(targets);out.array(obs);
}
void Batch::load(checkpoint::Reader& in) {
    auto states=in.array<deep_pond::State>(n_);auto episodes=in.array<uint32_t>(n_);auto targets=in.array<int>(n_);auto obs=in.array<float>(static_cast<size_t>(n_)*kObs);
    if(states.size()!=static_cast<size_t>(n_) || episodes.size()!=states.size() || targets.size()!=states.size() || obs.size()!=static_cast<size_t>(n_)*kObs) throw std::runtime_error("Selector world dimensions do not match");
    for(int i=0;i<n_;++i) if(states[i].seed!=seed_ || states[i].environment!=static_cast<uint32_t>(i) || states[i].episode!=episodes[i] ||
        targets[i]<-1 || targets[i]>2 || states[i].steps<0 || states[i].steps>=episode_steps_ || states[i].steps%decision_steps_!=0) throw std::runtime_error("Invalid selector world counters");
    for(float x:obs) if(!std::isfinite(x)) throw std::runtime_error("Non-finite selector observation");
    states_.copy_from_host(states.data(),states.size());episodes_.copy_from_host(episodes.data(),episodes.size());targets_.copy_from_host(targets.data(),targets.size());observations_.copy_from_host(obs.data(),obs.size());
}
}
