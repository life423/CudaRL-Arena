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
__device__ void spawn(deep_pond::State& state,int& target,History& history,deep::V& current,uint32_t seed,
                      uint32_t environment,uint32_t episode,float maximum,bool training,deep::V fixed,
                      const RecoveryStart* recovery,int count) {
    if(training && count>0 && rng::uniform(seed^0xB41073A1u,environment,episode*2)<.5f) {
        const auto& item=recovery[rng::bits(seed^0xB41073A1u,environment,episode*2+1)%static_cast<uint32_t>(count)];
        state=item.state;target=item.target;history=item.history;current=item.current;
        state.seed=seed;state.environment=environment;state.episode=episode;state.steps=0;state.eaten=0;state.last_eaten=-1;
        // Retain a truthful slow-progress context, leaving ten seconds before
        // the unchanged training failure rule can end the new episode.
        if(state.since_food>200) state.since_food=200;
    } else {
        state=start(seed,environment,episode,training);target=-1;history=initial_history(state);
        current=training?sample_current(seed,environment,episode,maximum):fixed;
    }
}
__global__ void reset(deep_pond::State* states,uint32_t* episodes,int* targets,History* history,deep::V* current,
                      float* obs,int n,uint32_t seed,float maximum,bool training,deep::V fixed,const RecoveryStart* recovery,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    episodes[i]=0;spawn(states[i],targets[i],history[i],current[i],seed,static_cast<uint32_t>(i),0,maximum,training,fixed,recovery,count);
    observe(states[i],targets[i],history[i],obs+i*kObs);
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
                              const float* torques,swimmer::Outputs out,int n,uint32_t seed,int period,float maximum,bool training,bool autoreset,deep::V fixed,
                              const RecoveryStart* recovery,int count) {
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
            spawn(state,targets[i],h,current[i],seed,static_cast<uint32_t>(i),++episodes[i],maximum,training,fixed,recovery,count);
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
    reset<<<(n+63)/64,64>>>(states_.get(),episodes_.get(),targets_.get(),history_.get(),currents_.get(),observations_.get(),n,seed,maximum,training,fixed,nullptr,0);CUDA_CHECK_THROW(cudaGetLastError());
}
const float* Batch::motor_obs(cudaStream_t stream) {inputs<<<(n_+127)/128,128,0,stream>>>(states_.get(),targets_.get(),motor_inputs_.get(),n_);CUDA_CHECK_THROW(cudaGetLastError());return motor_inputs_.get();}
void Batch::begin(const int* targets,swimmer::Outputs out,cudaStream_t stream) {begin_kernel<<<(n_+127)/128,128,0,stream>>>(targets,targets_.get(),out,n_);CUDA_CHECK_THROW(cudaGetLastError());}
void Batch::step(const float* torques,swimmer::Outputs out,cudaStream_t stream,bool autoreset) {
    advance_kernel<<<(n_+63)/64,64,0,stream>>>(states_.get(),episodes_.get(),targets_.get(),history_.get(),currents_.get(),observations_.get(),torques,out,n_,seed_,period_,maximum_,training_,autoreset,fixed_,recovery_.get(),recovery_count_);CUDA_CHECK_THROW(cudaGetLastError());
}
void Batch::set_recovery_starts(const std::vector<RecoveryStart>& starts,bool reset_worlds) {
    if(starts.size()>4096 || (!training_ && !starts.empty())) throw std::invalid_argument("Invalid recovery curriculum");
    for(const auto& r:starts) {
        if(r.target<0 || r.target>2 || !std::isfinite(deep::norm(r.current)) || deep::norm(r.current)>.01001f) throw std::invalid_argument("Invalid recovery target/current");
        float obs[kObs];observe(r.state,r.target,r.history,obs);for(float x:obs) if(!std::isfinite(x)) throw std::invalid_argument("Non-finite recovery state");
        for(int f=0;f<3;++f) if(!deep_pond::water(r.state.food[f]) || r.history.placements[f]!=r.state.placements[f]) throw std::invalid_argument("Recovery food must stay underwater with valid history");
    }
    CUDA_CHECK_THROW(cudaDeviceSynchronize());recovery_count_=static_cast<int>(starts.size());recovery_.allocate(starts.size());
    if(!starts.empty()) recovery_.copy_from_host(starts.data(),starts.size());
    if(reset_worlds) {
        reset<<<(n_+63)/64,64>>>(states_.get(),episodes_.get(),targets_.get(),history_.get(),currents_.get(),observations_.get(),n_,seed_,maximum_,training_,fixed_,recovery_.get(),recovery_count_);
        CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());
    }
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
