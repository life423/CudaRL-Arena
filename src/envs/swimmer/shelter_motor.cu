#include "shelter_motor.h"
#include <string>

namespace cudarl::shelter_motor {
namespace {
// Keep the large sampled-geometry helpers behind device call boundaries. Their
// equations remain shared; duplicating them in initialization and every reset
// path needlessly expands CUDA compiler IR and increases compilation memory.
__device__ __noinline__ float sampled_clearance(const deep::BodyState& body,const deep::Habitat& habitat,deep::Basin basin) {
    return predator::body_clearance(body,habitat,basin);
}
__device__ __noinline__ float sampled_exposure(const deep::BodyState& body,deep::V eye,const deep::Habitat& habitat,deep::Basin basin) {
    return predator::exposure(body,eye,habitat,basin);
}
__device__ __noinline__ predator::Cover cover_geometry(int index,deep::V eye,const deep::Habitat& habitat,deep::Basin basin) {
    return predator::cover(index,eye,habitat,basin);
}
__device__ __noinline__ State initial(uint32_t seed,uint32_t environment,uint32_t episode,arch::Context context) {
    State world;world.state=arch::initial(seed,environment,episode,false,context.basin);
    if(environment%2u==0u) return world;
    auto& state=world.state;const int index=static_cast<int>(rng::bits(seed^0x571E11A1u,environment,episode)%predator::kCoverCount);
    const auto point=cover_geometry(index,world.request.eye,state.habitat,context.basin);
    world.request.cover=true;world.request.goal.position=point.position;world.request.goal.cover=index;world.request.goal.valid=point.valid;
    world.request.heading=deep::axis_rotation({0,1,0},point.yaw).rotate({1,0,0});state.pond.target=-1;state.pond.decision=false;
    // Lesson 1: discover quiet, physically clear holding before far approach
    // or cross-current stationkeeping. Only unseen practice resets change.
    // Quarter one starts at the legal pose; quarter three has tiny errors.
    const auto body=predator::cover_pose(point);state.pond.body=body;
    bool placed=environment%4u==1u;
    #pragma unroll 1
    for(uint32_t attempt=0;attempt<48u && !placed;++attempt) {
        const uint32_t key=episode*787u+attempt*7u;
        state.pond.body=body;
        constexpr float amount=.15f;
        state.pond.body.position+=deep::V{(rng::uniform(seed,environment,key)*2-1)*amount,
            (rng::uniform(seed,environment,key+1)*2-1)*amount,(rng::uniform(seed,environment,key+2)*2-1)*amount};
        const float error=(rng::uniform(seed,environment,key+3)*2-1)*.15f;
        state.pond.body.orientation=deep::axis_rotation({0,1,0},point.yaw+error);
        placed=sampled_clearance(state.pond.body,state.habitat,context.basin)>.08f &&
            arch::water(state.pond.body.position,state.habitat,.2f,context.basin);
    }
    if(!placed) state.pond.body=body;
    if(!point.valid || sampled_clearance(state.pond.body,state.habitat,context.basin)<.08f) state.pond.invalid=true;
    state.pond.anchor=state.pond.body.position;
    world.exposure=sampled_exposure(state.pond.body,world.request.eye,state.habitat,context.basin);return world;
}
__global__ void reset(State* states,int n,uint32_t seed,arch::Context context) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) states[i]=initial(seed,static_cast<uint32_t>(i),0,context);
}
__global__ void feature_kernel(const State* states,float* out,int n) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n*feeding::capacity) return;
    const auto& world=states[row/feeding::capacity];const int slot=row%feeding::capacity;
    float* result=out+row*progress_selector::kFeatures;
    if(!world.request.cover && world.state.pond.decision && world.state.pond.pellets[slot].active)
        feeding::features(world.state.pond,slot,result);
    else for(int j=0;j<progress_selector::kFeatures;++j) result[j]=0;
}
__global__ void prepare_kernel(State* states,const float* scores,float* out,int n,arch::Context context) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    auto& world=states[i];auto& pond=world.state.pond;
    if(world.request.cover) {pond.target=-1;pond.decision=false;}
    else if(pond.decision) {
        pond.target=-1;float best=-1e30f;
        for(int slot=0;slot<feeding::capacity;++slot) if(pond.pellets[slot].active && scores[i*feeding::capacity+slot]>best) {
            best=scores[i*feeding::capacity+slot];pond.target=slot;
        }
        pond.decision=false;
    }
    observe(world,out+i*kObs,context.basin);
}
__global__ void advance_kernel(State* states,const float* actions,swimmer::Outputs out,int n,uint32_t seed,
                               arch::Context context,bool autoreset,bool practice) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    auto& world=states[i];auto& state=world.state;auto& pond=state.pond;
    // Evaluation/live requests still use their actual context current.
    // Ordinary food practice retains its existing current distribution.
    const auto water=practice?(world.request.cover?deep::V{}:arch::practice_current(state,context)):context.current;
    const float before=deep::norm(pond.body.position-world.request.goal.position);
    const int contacts=state.contacts;
    float reward=arch::advance(state,actions+i*kAct,water,context.basin);
    arch::replenish(state,context.basin);
    if(world.request.cover) {
        const float distance=deep::norm(pond.body.position-world.request.goal.position);
        if(pond.steps%4==0) world.exposure=sampled_exposure(pond.body,world.request.eye,state.habitat,context.basin);
        const float gap=sampled_clearance(pond.body,state.habitat,context.basin);
        const float alignment=deep::clamp(deep::dot(pond.body.orientation.rotate({1,0,0}),world.request.heading),-1,1);
        const float speed=deep::norm({pond.body.velocity[0],pond.body.velocity[1],pond.body.velocity[2]});
        const float near=expf(-distance*distance/(.45f*.45f));
        const bool safe=world.exposure<=.05f && gap>=.08f;
        const float quiet=expf(-speed/.015f);
        reward=3.f*(before-distance)-.008f*fminf(distance,4.f)-.025f*world.exposure;
        // Dense practice feedback rewards progress toward a clear, concealed,
        // quiet pose. The five-second safety proof below stays strictly binary.
        const float cover_quality=(1.f-deep::clamp(world.exposure,0,1))*deep::clamp((gap+.02f)/.10f,0,1);
        reward+=.025f*fmaxf(0,alignment)+.1f*near*(.35f+.65f*fmaxf(0,alignment))*quiet*cover_quality;
        reward-=.2f*(state.contacts-contacts)+.02f*deep::clamp(-gap/.1f,0,1);
        for(int j=0;j<kAct;++j) {const float torque=deep::clamp(actions[i*kAct+j],-1,1);reward-=.002f*torque*torque;}
        if(safe && distance<=.6f) {++world.held_ticks;world.peak_hold=max(world.peak_hold,world.held_ticks);} else world.held_ticks=0;
    }
    if(out.reward) out.reward[i]=reward;
    const bool timeout=pond.steps>=(world.request.cover?kCoverEpisode:arch::kEpisodeSteps),failed=pond.invalid;
    if(out.done) out.done[i]=timeout || failed?1.f:0.f;
    if(out.truncated) out.truncated[i]=timeout && !failed?1.f:0.f;
    if((timeout || failed) && out.final_obs) observe(world,out.final_obs+i*kObs,context.basin);
    if((timeout || failed) && autoreset) {
        const auto episode=pond.episode;world=initial(seed,static_cast<uint32_t>(i),episode+1,context);
    }
}
__global__ void request_kernel(State* state,Request request) {state[0].request=request;state[0].held_ticks=0;state[0].peak_hold=0;state[0].state.pond.decision=true;}
void validate_context(const arch::Context& context) {
    if(context.basin.layout!=2 || context.version<0 || !std::isfinite(deep::norm(context.current)) ||
       deep::norm(context.current)>.0075001f || fabsf(context.current.y)>1e-8f)
        throw std::invalid_argument("Shelter motor task requires boulder geometry and bounded horizontal current");
}
void validate_state(const State& world) {
    const auto& pond=world.state.pond;
    if(pond.steps<0 || pond.target<-1 || pond.target>=feeding::capacity || !std::isfinite(deep::norm(pond.body.position)) ||
       !std::isfinite(world.exposure) || world.held_ticks<0 || world.peak_hold<world.held_ticks)
        throw std::runtime_error("Invalid shelter motor world");
    if(world.request.cover && (!world.request.goal.valid || world.request.goal.cover<0 || world.request.goal.cover>=predator::kCoverCount ||
       !std::isfinite(deep::norm(world.request.goal.position)) || !std::isfinite(deep::norm(world.request.eye)) ||
       fabsf(deep::norm(world.request.heading)-1.f)>1e-4f)) throw std::runtime_error("Invalid cover movement request");
}
}
Batch::Batch(int n,uint32_t seed,swimmer::Body unused):n_(n),seed_(seed),states_(n),
    observations_(static_cast<size_t>(n)*kObs),features_(static_cast<size_t>(n)*feeding::capacity*progress_selector::kFeatures),
    selector_(rl::progress_selector_spec(),n*feeding::capacity,seed) {
    (void)unused;if(n<1 || n>2048) throw std::invalid_argument("Invalid shelter motor population");
    reset<<<(n+63)/64,64>>>(states_.get(),n,seed,context_);CUDA_CHECK_THROW(cudaGetLastError());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());prepare(nullptr);CUDA_CHECK_THROW(cudaDeviceSynchronize());
}
void Batch::prepare(cudaStream_t stream) {
    feature_kernel<<<(n_*feeding::capacity+127)/128,128,0,stream>>>(states_.get(),features_.get(),n_);CUDA_CHECK_THROW(cudaGetLastError());
    selector_.forward(features_.get(),n_*feeding::capacity,stream);
    prepare_kernel<<<(n_+63)/64,64,0,stream>>>(states_.get(),selector_.output(),observations_.get(),n_,context_);CUDA_CHECK_THROW(cudaGetLastError());
}
void Batch::configure(const std::vector<float>& shared,const arch::Context& context,cudaStream_t stream,bool reset_worlds) {
    validate_context(context);CUDA_CHECK_THROW(cudaStreamSynchronize(stream));selector_.set_params(shared);context_=context;
    if(reset_worlds) {reset<<<(n_+63)/64,64,0,stream>>>(states_.get(),n_,seed_,context_);CUDA_CHECK_THROW(cudaGetLastError());}
    prepare(stream);
}
void Batch::step(const float* torques,swimmer::Outputs out,cudaStream_t stream,bool autoreset,bool practice) {
    advance_kernel<<<(n_+63)/64,64,0,stream>>>(states_.get(),torques,out,n_,seed_,context_,autoreset,practice);CUDA_CHECK_THROW(cudaGetLastError());prepare(stream);
}
std::vector<State> Batch::host_states(cudaStream_t stream) const {
    std::vector<State> out(static_cast<size_t>(n_));
    CUDA_CHECK_THROW(cudaMemcpyAsync(out.data(),states_.get(),states_.size_bytes(),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));return out;
}
State Batch::state(cudaStream_t stream) const {
    if(n_!=1) throw std::runtime_error("Single-world motor snapshot requires population one");return host_states(stream)[0];
}
void Batch::set_state(const State& state,cudaStream_t stream) {
    if(n_!=1) throw std::runtime_error("Single-world motor restore requires population one");validate_state(state);
    CUDA_CHECK_THROW(cudaMemcpyAsync(states_.get(),&state,sizeof(State),cudaMemcpyHostToDevice,stream));prepare(stream);CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
void Batch::set_states(const std::vector<State>& states,cudaStream_t stream) {
    if(states.size()!=static_cast<size_t>(n_)) throw std::runtime_error("Shelter motor replacement population mismatch");
    for(const auto& state:states) validate_state(state);
    CUDA_CHECK_THROW(cudaMemcpyAsync(states_.get(),states.data(),states_.size_bytes(),cudaMemcpyHostToDevice,stream));
    prepare(stream);CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
void Batch::set_request(const Request& request,cudaStream_t stream) {
    if(n_!=1) throw std::runtime_error("Single-world motor request requires population one");
    State validation;validation.request=request;validate_state(validation);
    request_kernel<<<1,1,0,stream>>>(states_.get(),request);CUDA_CHECK_THROW(cudaGetLastError());prepare(stream);
}
void Batch::save(checkpoint::Writer& out) const {
    CUDA_CHECK_THROW(cudaDeviceSynchronize());std::vector<State> states(static_cast<size_t>(n_));states_.copy_to_host(states.data(),states.size());
    out.value(context_);out.value(action_scales_);out.array(selector_.get_params());out.array(states);
}
void Batch::set_action_scales(ActuatorScale scale) {
    for(float value:scale.value) if(!std::isfinite(value) || value<1.f || value>1048576.f)
        throw std::invalid_argument("Invalid shelter action calibration");
    action_scales_=scale;
}
void Batch::load(checkpoint::Reader& in) {
    const auto context=in.value<arch::Context>();validate_context(context);
    const auto scales=in.value<ActuatorScale>();set_action_scales(scales);
    const auto shared=in.array<float>(selector_.num_params());const auto states=in.array<State>(static_cast<size_t>(n_));
    if(shared.size()!=static_cast<size_t>(selector_.num_params()) || states.size()!=static_cast<size_t>(n_)) throw std::runtime_error("Shelter motor snapshot dimensions changed");
    for(float v:shared) if(!std::isfinite(v)) throw std::runtime_error("Non-finite shared selector");
    for(const auto& state:states) {validate_state(state);if(state.state.pond.steps>=(state.request.cover?kCoverEpisode:arch::kEpisodeSteps))
        throw std::runtime_error("Invalid shelter motor practice clock");}
    CUDA_CHECK_THROW(cudaDeviceSynchronize());context_=context;selector_.set_params(shared);states_.copy_from_host(states.data(),states.size());
    prepare(nullptr);CUDA_CHECK_THROW(cudaDeviceSynchronize());
}
}
