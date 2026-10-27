#include "progress_selector.h"
namespace cudarl::progress_selector {
namespace {
__device__ deep_pond::State start(uint32_t seed,uint32_t environment,uint32_t episode,bool training) {
    auto state=deep_pond::initial(seed,environment,episode);
    if(training) {
        const float kind=rng::uniform(seed^0x42AF8B91u,environment,episode*23u);
        if(kind<.5f) {
            for(int j=0;j<4;++j) state.body.angle[j]=deep::limit*.98f*(2*rng::uniform(seed^0x42AF8B91u,environment,episode*23u+2+j)-1);
            for(int f=0;f<3;++f) {state.food[f]={99,99,99};state.placements[f]=0;}
            for(int f=0;f<3;++f) deep_pond::place_food(state,f);
        }
    }
    return state;
}
__global__ void reset(deep_pond::State* states,uint32_t* episodes,int* targets,History* history,deep::V* current,
                      float* obs,int n,uint32_t seed,float maximum,bool training,deep::V fixed) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    states[i]=start(seed,static_cast<uint32_t>(i),0,training);episodes[i]=0;targets[i]=-1;
    history[i]=initial_history(states[i]);current[i]=training?sample_current(seed,static_cast<uint32_t>(i),0,maximum):fixed;
    observe(states[i],-1,history[i],obs+i*kObs);
}
__global__ void feature_kernel(const float* obs,float* out,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*3) features(obs+(i/3)*kObs,i%3,out+i*kFeatures);
}
__global__ void inputs(const deep_pond::State* states,const int* target,float* out,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    deep_pond::observe(states[i],out+i*33);demos::primary_pond_food(out+i*33,target[i]<0?0:target[i]);
}
__global__ void begin_kernel(const int* selected,int* targets,swimmer::Outputs out,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;targets[i]=selected[i];
    if(out.reward) out.reward[i]=0;if(out.done) out.done[i]=0;if(out.truncated) out.truncated[i]=0;
}
__global__ void advance_kernel(deep_pond::State* states,uint32_t* episodes,int* targets,History* history,deep::V* current,float* obs,
                              const float* torques,swimmer::Outputs out,int n,uint32_t seed,int period,float maximum,bool training,bool autoreset,deep::V fixed) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    auto state=states[i];auto h=history[i];const auto result=deep_pond::advance(state,torques+i*4,current[i]);
    if(state.steps%period==0) update_history(state,h,period);
    const bool failed=autoreset && state.steps%period==0 && state.since_food>=400 &&
        deep::norm({state.body.velocity[0],state.body.velocity[1],state.body.velocity[2]})<.005f;
    if(out.reward && result.eaten_slot>=0) out.reward[i]+=1;
    if(out.reward && failed) out.reward[i]-=5;
    if(state.steps>=6000 || failed) {
        if(out.done) out.done[i]=1;if(out.truncated) out.truncated[i]=failed?0.f:1.f;
        if(out.final_obs) observe(state,targets[i],h,out.final_obs+i*kObs);
        if(autoreset) {
            state=start(seed,static_cast<uint32_t>(i),++episodes[i],training);targets[i]=-1;h=initial_history(state);
            current[i]=training?sample_current(seed,static_cast<uint32_t>(i),episodes[i],maximum):fixed;
        }
    }
    states[i]=state;history[i]=h;observe(state,targets[i],h,obs+i*kObs);
}
}
void make_features(const float* obs,float* out,int n,cudaStream_t stream) {
    feature_kernel<<<(n*3+127)/128,128,0,stream>>>(obs,out,n);CUDA_CHECK_THROW(cudaGetLastError());
}
Batch::Batch(int n,uint32_t seed,int period,float maximum,bool training,deep::V fixed):n_(n),period_(period),seed_(seed),maximum_(maximum),training_(training),fixed_(fixed),
    states_(n),episodes_(n),targets_(n),history_(n),currents_(n),observations_(static_cast<size_t>(n)*kObs),motor_inputs_(static_cast<size_t>(n)*33) {
    if(n<1 || period<1 || 6000%period || maximum<0 || maximum>.01f) throw std::invalid_argument("Invalid progress-selector environment");
    reset<<<(n+63)/64,64>>>(states_.get(),episodes_.get(),targets_.get(),history_.get(),currents_.get(),observations_.get(),n,seed,maximum,training,fixed);CUDA_CHECK_THROW(cudaGetLastError());
}
const float* Batch::motor_obs(cudaStream_t stream) {inputs<<<(n_+127)/128,128,0,stream>>>(states_.get(),targets_.get(),motor_inputs_.get(),n_);CUDA_CHECK_THROW(cudaGetLastError());return motor_inputs_.get();}
void Batch::begin(const int* targets,swimmer::Outputs out,cudaStream_t stream) {begin_kernel<<<(n_+127)/128,128,0,stream>>>(targets,targets_.get(),out,n_);CUDA_CHECK_THROW(cudaGetLastError());}
void Batch::step(const float* torques,swimmer::Outputs out,cudaStream_t stream,bool autoreset) {
    advance_kernel<<<(n_+63)/64,64,0,stream>>>(states_.get(),episodes_.get(),targets_.get(),history_.get(),currents_.get(),observations_.get(),torques,out,n_,seed_,period_,maximum_,training_,autoreset,fixed_);CUDA_CHECK_THROW(cudaGetLastError());
}
void Batch::save(checkpoint::Writer& out) const {
    std::vector<deep_pond::State> states(n_);std::vector<uint32_t> episodes(n_);std::vector<int> targets(n_);std::vector<History> history(n_);std::vector<deep::V> currents(n_);std::vector<float> obs(static_cast<size_t>(n_)*kObs);
    states_.copy_to_host(states.data(),n_);episodes_.copy_to_host(episodes.data(),n_);targets_.copy_to_host(targets.data(),n_);history_.copy_to_host(history.data(),n_);currents_.copy_to_host(currents.data(),n_);observations_.copy_to_host(obs.data(),obs.size());
    out.array(states);out.array(episodes);out.array(targets);out.array(history);out.array(currents);out.array(obs);
}
void Batch::load(checkpoint::Reader& in) {
    auto states=in.array<deep_pond::State>(n_);auto episodes=in.array<uint32_t>(n_);auto targets=in.array<int>(n_);auto history=in.array<History>(n_);auto currents=in.array<deep::V>(n_);auto obs=in.array<float>(static_cast<size_t>(n_)*kObs);
    if(states.size()!=static_cast<size_t>(n_) || episodes.size()!=states.size() || targets.size()!=states.size() || history.size()!=states.size() || currents.size()!=states.size() || obs.size()!=static_cast<size_t>(n_)*kObs) throw std::runtime_error("Progress world save dimensions changed");
    for(float x:obs) if(!std::isfinite(x)) throw std::runtime_error("Invalid progress observation");
    for(int i=0;i<n_;++i) if(states[i].seed!=seed_ || states[i].environment!=static_cast<uint32_t>(i) || states[i].episode!=episodes[i] || targets[i]<-1 || targets[i]>2 || states[i].steps<0 || states[i].steps>=6000 || states[i].steps%period_) throw std::runtime_error("Invalid progress world counters");
    states_.copy_from_host(states.data(),n_);episodes_.copy_from_host(episodes.data(),n_);targets_.copy_from_host(targets.data(),n_);history_.copy_from_host(history.data(),n_);currents_.copy_from_host(currents.data(),n_);observations_.copy_from_host(obs.data(),obs.size());
}
}
