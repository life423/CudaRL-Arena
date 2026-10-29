#include "feeding_batch.h"
namespace cudarl::feeding {
namespace {
__global__ void initialize(State* states,int n,uint32_t seed,Context context) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) states[i]=initial(seed,static_cast<uint32_t>(i),0,context,i%4==0);
}
__global__ void make_features(State* states,float* features,int n) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n*capacity) return;
    const int world=row/capacity,food=row%capacity;float* out=features+row*progress_selector::kFeatures;
    const auto& s=states[world];
    if(s.decision && s.pellets[food].active) feeding::features(s,food,out);
    else for(int j=0;j<progress_selector::kFeatures;++j) out[j]=0;
}
__global__ void choose(State* states,const float* scores,float* obs,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;auto& s=states[i];
    if(s.decision) {
        s.target=-1;float best=-1e30f;
        for(int f=0;f<capacity;++f) if(s.pellets[f].active && scores[i*capacity+f]>best) {best=scores[i*capacity+f];s.target=f;}
        s.decision=false;
    }
    motor_observe(s,obs+i*kObs);
}
__global__ void advance_kernel(State* states,const float* actions,swimmer::Outputs out,int n,bool autoreset,Context context) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;auto& s=states[i];
    const float reward=advance(s,actions+i*kAct,context.current);const bool timeout=s.steps>=kEpisodeSteps;
    if(out.reward) out.reward[i]=reward;if(out.done) out.done[i]=timeout?1.f:0.f;if(out.truncated) out.truncated[i]=timeout?1.f:0.f;
    if(autoreset && !timeout) replenish(s,context);
}
__global__ void finish_time_limits(State* states,const float* observations,float* final_obs,int n,Context context) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;auto& s=states[i];
    if(s.steps>=kEpisodeSteps) {
        if(final_obs) for(int j=0;j<kObs;++j) final_obs[i*kObs+j]=observations[i*kObs+j];
        s=initial(s.seed,s.environment,s.episode+1,context,s.retention);
    }
}
}
Batch::Batch(int n,uint32_t seed,swimmer::Body unused):n_(n),seed_(seed),states_(n),observations_(static_cast<size_t>(n)*kObs),features_(static_cast<size_t>(n)*capacity*progress_selector::kFeatures),
    selector_(rl::progress_selector_spec(),n*capacity,seed) {
    (void)unused;if(n<1 || n>2048) throw std::invalid_argument("Invalid feeding batch");
    initialize<<<(n+63)/64,64>>>(states_.get(),n_,seed_,context_);CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());
    prepare(nullptr);CUDA_CHECK_THROW(cudaDeviceSynchronize());
}
void Batch::prepare(cudaStream_t stream) {
    make_features<<<(n_*capacity+127)/128,128,0,stream>>>(states_.get(),features_.get(),n_);CUDA_CHECK_THROW(cudaGetLastError());
    selector_.forward(features_.get(),n_*capacity,stream);
    choose<<<(n_+63)/64,64,0,stream>>>(states_.get(),selector_.output(),observations_.get(),n_);CUDA_CHECK_THROW(cudaGetLastError());
}
void Batch::configure(const std::vector<float>& weights,const Context& context,cudaStream_t stream,bool reset) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));selector_.set_params(weights);context_=context;
    if(reset) {ticks_=0;initialize<<<(n_+63)/64,64,0,stream>>>(states_.get(),n_,seed_,context_);CUDA_CHECK_THROW(cudaGetLastError());}
    prepare(stream);
}
void Batch::step(const float* actions,swimmer::Outputs out,cudaStream_t stream,bool autoreset) {
    advance_kernel<<<(n_+63)/64,64,0,stream>>>(states_.get(),actions,out,n_,autoreset,context_);CUDA_CHECK_THROW(cudaGetLastError());prepare(stream);
    if(autoreset && ++ticks_>=kEpisodeSteps) {
        finish_time_limits<<<(n_+63)/64,64,0,stream>>>(states_.get(),observations_.get(),out.final_obs,n_,context_);CUDA_CHECK_THROW(cudaGetLastError());
        ticks_=0;prepare(stream);
    }
}
State Batch::state(cudaStream_t stream) const {
    if(n_!=1) throw std::runtime_error("Single-world state requested from training batch");State out;
    CUDA_CHECK_THROW(cudaMemcpyAsync(&out,states_.get(),sizeof(out),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));return out;
}
void Batch::set_state(const State& state,cudaStream_t stream) {
    if(n_!=1) throw std::runtime_error("Single-world restore requested from training batch");
    ticks_=state.steps;CUDA_CHECK_THROW(cudaMemcpyAsync(states_.get(),&state,sizeof(state),cudaMemcpyHostToDevice,stream));prepare(stream);CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
void Batch::save(checkpoint::Writer& out) const {
    std::vector<State> states(n_);states_.copy_to_host(states.data(),states.size());out.array(states);out.value(context_);
}
void Batch::load(checkpoint::Reader& in) {
    const auto states=in.array<State>(n_);const auto context=in.value<Context>();if(states.size()!=static_cast<size_t>(n_) || context.count<0 || context.count>32) throw std::runtime_error("Invalid feeding worlds");
    for(const auto& s:states) if(s.steps<0 || s.steps>=kEpisodeSteps || s.target>=capacity || !std::isfinite(s.body.position.x)) throw std::runtime_error("Invalid feeding save state");
    ticks_=states[0].steps;context_=context;states_.copy_from_host(states.data(),states.size());prepare(nullptr);CUDA_CHECK_THROW(cudaDeviceSynchronize());
}
}
