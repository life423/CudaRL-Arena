#include "arch_batch.h"

namespace cudarl::arch {
namespace {
int population(int n) {
    if(n<1 || n>2048) throw std::invalid_argument("Invalid arch batch population");
    return n;
}
__global__ void initialize(State* states,int n,uint32_t seed) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) states[i]=initial(seed,static_cast<uint32_t>(i),0,i%4==0);
}
__global__ void make_features(State* states,float* features,int n) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=n*feeding::capacity) return;
    const int world=row/feeding::capacity,food=row%feeding::capacity;
    float* out=features+row*progress_selector::kFeatures;
    const auto& pond=states[world].pond;
    if(pond.decision && pond.pellets[food].active) feeding::features(pond,food,out);
    else for(int j=0;j<progress_selector::kFeatures;++j) out[j]=0;
}
__global__ void choose(State* states,const float* scores,float* observations,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    auto& state=states[i];auto& pond=state.pond;
    if(pond.decision) {
        pond.target=-1;float best=-1e30f;
        for(int food=0;food<feeding::capacity;++food) {
            const float score=scores[i*feeding::capacity+food];
            if(pond.pellets[food].active && score>best) {best=score;pond.target=food;}
        }
        pond.decision=false;
    }
    arch::observe(state,observations+i*kObs);
}
__global__ void advance_kernel(State* states,const float* actions,swimmer::Outputs out,int n,bool autoreset) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    auto& state=states[i];
    const float reward=arch::advance(state,actions+i*kAct);
    const bool timeout=state.pond.steps>=kEpisodeSteps;
    if(out.reward) out.reward[i]=reward;
    if(out.done) out.done[i]=timeout?1.f:0.f;
    if(out.truncated) out.truncated[i]=timeout?1.f:0.f;
    // Arch food is automatic in both the visible habitat and practice. A time
    // limit preserves its terminal observation before any reset/refill.
    if(!autoreset || !timeout) arch::replenish(state);
}
__global__ void finish_time_limits(State* states,const float* observations,float* final_obs,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    auto& state=states[i];
    if(state.pond.steps>=kEpisodeSteps) {
        if(final_obs) for(int j=0;j<kObs;++j) final_obs[i*kObs+j]=observations[i*kObs+j];
        const uint32_t seed=state.pond.seed,environment=state.pond.environment,episode=state.pond.episode;
        const bool retention=state.pond.retention;
        state=initial(seed,environment,episode+1,retention);
    }
}
}
Batch::Batch(int n,uint32_t seed,swimmer::Body unused):n_(population(n)),seed_(seed),states_(n_),
    observations_(static_cast<size_t>(n_)*kObs),
    features_(static_cast<size_t>(n_)*feeding::capacity*progress_selector::kFeatures),
    selector_(rl::progress_selector_spec(),n_*feeding::capacity,seed) {
    (void)unused;
    initialize<<<(n_+63)/64,64>>>(states_.get(),n_,seed_);CUDA_CHECK_THROW(cudaGetLastError());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());prepare(nullptr);CUDA_CHECK_THROW(cudaDeviceSynchronize());
}
void Batch::prepare(cudaStream_t stream) {
    make_features<<<(n_*feeding::capacity+127)/128,128,0,stream>>>(states_.get(),features_.get(),n_);CUDA_CHECK_THROW(cudaGetLastError());
    selector_.forward(features_.get(),n_*feeding::capacity,stream);
    choose<<<(n_+63)/64,64,0,stream>>>(states_.get(),selector_.output(),observations_.get(),n_);CUDA_CHECK_THROW(cudaGetLastError());
}
void Batch::configure(const std::vector<float>& selector,cudaStream_t stream,bool reset) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));selector_.set_params(selector);
    if(reset) {
        ticks_=0;
        initialize<<<(n_+63)/64,64,0,stream>>>(states_.get(),n_,seed_);CUDA_CHECK_THROW(cudaGetLastError());
    }
    prepare(stream);
}
void Batch::step(const float* actions,swimmer::Outputs out,cudaStream_t stream,bool autoreset) {
    advance_kernel<<<(n_+63)/64,64,0,stream>>>(states_.get(),actions,out,n_,autoreset);CUDA_CHECK_THROW(cudaGetLastError());
    prepare(stream);
    if(autoreset && ++ticks_>=kEpisodeSteps) {
        finish_time_limits<<<(n_+63)/64,64,0,stream>>>(states_.get(),observations_.get(),out.final_obs,n_);CUDA_CHECK_THROW(cudaGetLastError());
        ticks_=0;prepare(stream);
    }
}
State Batch::state(cudaStream_t stream) const {
    if(n_!=1) throw std::runtime_error("Single-world state requested from arch training batch");
    State out;
    CUDA_CHECK_THROW(cudaMemcpyAsync(&out,states_.get(),sizeof(out),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));return out;
}
void Batch::set_state(const State& state,cudaStream_t stream) {
    if(n_!=1) throw std::runtime_error("Single-world restore requested from arch training batch");
    ticks_=state.pond.steps;
    CUDA_CHECK_THROW(cudaMemcpyAsync(states_.get(),&state,sizeof(state),cudaMemcpyHostToDevice,stream));
    prepare(stream);CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
void Batch::save(checkpoint::Writer& out) const {
    std::vector<State> states(static_cast<size_t>(n_));states_.copy_to_host(states.data(),states.size());out.array(states);
}
void Batch::load(checkpoint::Reader& in) {
    const auto states=in.array<State>(static_cast<size_t>(n_));
    if(states.size()!=static_cast<size_t>(n_)) throw std::runtime_error("Invalid arch world count");
    for(const auto& state:states) {
        const auto& pond=state.pond;
        if(pond.steps<0 || pond.steps>=kEpisodeSteps || pond.steps!=states[0].pond.steps ||
           pond.target<-1 || pond.target>=feeding::capacity || !std::isfinite(pond.body.position.x) ||
           !std::isfinite(pond.body.position.y) || !std::isfinite(pond.body.position.z))
            throw std::runtime_error("Invalid arch practice state");
    }
    ticks_=states[0].pond.steps;states_.copy_from_host(states.data(),states.size());
    prepare(nullptr);CUDA_CHECK_THROW(cudaDeviceSynchronize());
}
}
