#include "shelter_batch.h"
#include "rl/ppo_math.h"
#include <string>

namespace cudarl::shelter {
static_assert(kChoices<=rl::kMaxActions,"Categorical PPO stack capacity must cover every shelter option");
namespace {
__device__ World initial(uint32_t seed,uint32_t environment,uint32_t episode,arch::Context context,Options options) {
    World out;out.state=arch::initial(seed,environment,episode,false,context.basin);
    out.predator.seed=rng::bits(seed^0x571E1A11u,environment,episode);
    out.threats=options.threats && (!options.training || environment%4u!=0u);
    out.exposure=predator::exposure(out.state.pond.body,event(out).eye,out.state.habitat,context.basin);return out;
}
__global__ void reset(World* worlds,int n,uint32_t seed,arch::Context context,Options options) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) worlds[i]=initial(seed,static_cast<uint32_t>(i),0,context,options);
}
__global__ void observations(const World* worlds,float* out,int n,arch::Context context,Options options) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) observe(worlds[i],out+i*kObs,context.basin,options);
}
__global__ void feature_kernel(const float* obs,float* rows,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n*kChoices*kFeatures) rows[i]=obs[(i/(kChoices*kFeatures))*kObs+i%(kChoices*kFeatures)];
}
__global__ void mask_kernel(const float* obs,float* logits,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n*kChoices && obs[(i/kChoices)*kObs+kChoices*kFeatures+i%kChoices]<.5f) logits[i]=-1e9f;
}
__global__ void inputs(World* worlds,float* obs,int n,arch::Context context) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    auto& world=worlds[i];
    if(world.goal.food>=0) world.goal=option(world,world.goal.food,context.basin);
    deep::V heading{0,0,1};
    if(world.goal.cover>=0) {
        const auto cover=predator::cover(world.goal.cover,event(world).eye,world.state.habitat,context.basin);
        heading=deep::axis_rotation({0,1,0},cover.yaw).rotate({1,0,0});
    }
    motor_observe59(world.state,world.goal,heading,obs+i*shelter_motor::kObs,context.basin);
}
__global__ void begin_kernel(World* worlds,const int* choices,swimmer::Outputs out,int n,arch::Context context,Options options) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    auto& world=worlds[i];world.chosen=choices[i];world.goal=option(world,world.chosen,context.basin);
    if(!world.goal.valid || (!options.cover && world.goal.cover>=0)) world.state.pond.invalid=true;
    world.state.pond.target=world.goal.food;
    world.goal_distance=deep::norm(world.goal.position-world.state.pond.body.position);world.goal_progress=0;
    if(world.goal.cover>=0) ++world.metrics.cover_choices;else ++world.metrics.food_choices;
    if(out.reward) out.reward[i]=0;if(out.done) out.done[i]=0;if(out.truncated) out.truncated[i]=0;
}
__global__ void advance_kernel(World* worlds,const float* torques,swimmer::Outputs out,int n,uint32_t seed,
                               arch::Context context,int period,Options options,bool autoreset,float* obs) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    auto& world=worlds[i];auto& state=world.state;const auto threat=event(world);
    const int food_before=state.pond.eaten,contact_before=state.contacts;
    const deep::V water=options.training?arch::practice_current(state,context):context.current;
    // Keep exactly the existing articulated dynamics, solids, food advection and
    // physical collection. Its target-distance shaping is not a safety reward.
    (void)arch::advance(state,torques+i*4,water,context.basin);arch::replenish(state,context.basin);
    predator::advance(world.predator);
    if(state.pond.steps%4==0)
        world.exposure=predator::exposure(state.pond.body,threat.eye,state.habitat,context.basin);
    const int collected=state.pond.eaten-food_before;
    float reward=5.f*collected-.002f-.02f*(state.contacts-contact_before)-.2f*threat.strength*world.exposure;
    if(state.pond.invalid) reward-=5.f;
    if(threat.active) {
        world.metrics.threat_seconds+=deep::dt;world.metrics.exposure_seconds+=world.exposure*deep::dt;
        if(world.exposure<=.05f && predator::body_clearance(state.pond.body,state.habitat,context.basin)>=.08f)
            ++world.metrics.safe_ticks;
    }
    if(threat.passed) world.metrics.post_threat_foods+=collected;
    if(out.reward) out.reward[i]+=reward;
    const bool timeout=state.pond.steps>=arch::kEpisodeSteps;
    const bool failure=state.pond.invalid;
    if((timeout || failure) && out.done) out.done[i]=1;
    if((timeout || failure) && out.truncated) out.truncated[i]=failure?0.f:1.f;
    if(state.pond.steps%period==0) {
        const float distance=deep::norm(world.goal.position-state.pond.body.position);
        world.goal_progress=deep::clamp((world.goal_distance-distance)/(period*deep::dt),-2,2);
        world.goal_distance=distance;
    }
    if((timeout || failure) && autoreset && state.pond.steps%period==0) {
        if(out.final_obs) observe(world,out.final_obs+i*kObs,context.basin,options);
        world=initial(seed,static_cast<uint32_t>(i),state.pond.episode+1,context,options);
    }
    if(world.state.pond.steps%period==0) observe(world,obs+i*kObs,context.basin,options);
}
void validate_context(const arch::Context& context) {
    if(context.basin.layout!=2 || context.version<0 || !std::isfinite(deep::norm(context.current)) ||
       deep::norm(context.current)>.0075001f || fabsf(context.current.y)>1e-8f)
        throw std::invalid_argument("Shelter pilot requires a boulder habitat and bounded horizontal current");
}
void validate_world(const World& world) {
    const auto& pond=world.state.pond;
    if(pond.target<-1 || pond.target>=kFoods || world.chosen<-1 || world.chosen>=kChoices ||
       pond.steps<0 || !std::isfinite(deep::norm(pond.body.position)) || !std::isfinite(world.exposure) ||
       !std::isfinite(world.predator.elapsed) || !std::isfinite(world.predator.manual_age))
        throw std::runtime_error("Invalid shelter pilot world");
    for(int i=kFoods;i<feeding::capacity;++i) if(pond.pellets[i].active)
        throw std::runtime_error("The bounded shelter pilot needs exactly the current boulder food slots; many-pellet worlds are not silently truncated");
}
}
void make_features(const float* obs,float* rows,int n,cudaStream_t stream) {
    feature_kernel<<<(n*kChoices*kFeatures+127)/128,128,0,stream>>>(obs,rows,n);CUDA_CHECK_THROW(cudaGetLastError());
}
void mask_logits(const float* obs,float* logits,int n,cudaStream_t stream) {
    mask_kernel<<<(n*kChoices+127)/128,128,0,stream>>>(obs,logits,n);CUDA_CHECK_THROW(cudaGetLastError());
}
Batch::Batch(int n,uint32_t seed,arch::Context context,int period,Options options):
    n_(n),seed_(seed),context_(context),period_(period),options_(options),worlds_(n),
    observations_(static_cast<size_t>(n)*kObs),motor_inputs_(static_cast<size_t>(n)*shelter_motor::kObs) {
    validate_context(context_);
    if(n<1 || n>2048 || period<1 || arch::kEpisodeSteps%period) throw std::invalid_argument("Invalid shelter pilot population or decision period");
    reset<<<(n+63)/64,64>>>(worlds_.get(),n,seed,context_,options_);CUDA_CHECK_THROW(cudaGetLastError());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());prepare(nullptr);CUDA_CHECK_THROW(cudaDeviceSynchronize());
}
void Batch::prepare(cudaStream_t stream) {
    observations<<<(n_+63)/64,64,0,stream>>>(worlds_.get(),observations_.get(),n_,context_,options_);CUDA_CHECK_THROW(cudaGetLastError());
}
const float* Batch::motor_obs(cudaStream_t stream) {
    inputs<<<(n_+63)/64,64,0,stream>>>(worlds_.get(),motor_inputs_.get(),n_,context_);CUDA_CHECK_THROW(cudaGetLastError());return motor_inputs_.get();
}
void Batch::begin(const int* choices,swimmer::Outputs out,cudaStream_t stream) {
    begin_kernel<<<(n_+63)/64,64,0,stream>>>(worlds_.get(),choices,out,n_,context_,options_);CUDA_CHECK_THROW(cudaGetLastError());
}
void Batch::step(const float* torques,swimmer::Outputs out,cudaStream_t stream,bool autoreset) {
    advance_kernel<<<(n_+63)/64,64,0,stream>>>(worlds_.get(),torques,out,n_,seed_,context_,period_,options_,autoreset,observations_.get());CUDA_CHECK_THROW(cudaGetLastError());
}
std::vector<World> Batch::states(cudaStream_t stream) const {
    std::vector<World> out(static_cast<size_t>(n_));
    CUDA_CHECK_THROW(cudaMemcpyAsync(out.data(),worlds_.get(),worlds_.size_bytes(),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));return out;
}
World Batch::state(cudaStream_t stream) const {
    if(n_!=1) throw std::runtime_error("Single-world shelter snapshot requires population one");return states(stream)[0];
}
void Batch::set_state(const World& world,cudaStream_t stream) {
    if(n_!=1) throw std::runtime_error("Single-world shelter restore requires population one");validate_world(world);
    CUDA_CHECK_THROW(cudaMemcpyAsync(worlds_.get(),&world,sizeof(World),cudaMemcpyHostToDevice,stream));prepare(stream);CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
void Batch::configure(arch::Context context,cudaStream_t stream) {
    validate_context(context);context_=context;prepare(stream);
}
void Batch::set_options(Options options,cudaStream_t stream) {
    options_=options;prepare(stream);
}
void Batch::save(checkpoint::Writer& out) const {
    CUDA_CHECK_THROW(cudaDeviceSynchronize());std::vector<World> worlds(static_cast<size_t>(n_));worlds_.copy_to_host(worlds.data(),worlds.size());
    out.value(context_);out.array(worlds);
}
void Batch::load(checkpoint::Reader& in) {
    const auto context=in.value<arch::Context>();validate_context(context);auto worlds=in.array<World>(static_cast<size_t>(n_));
    if(worlds.size()!=static_cast<size_t>(n_)) throw std::runtime_error("Shelter population changed");
    for(const auto& world:worlds) {validate_world(world);if(world.state.pond.steps>=arch::kEpisodeSteps || world.state.pond.steps%period_)
        throw std::runtime_error("Shelter checkpoint must end on an on-policy macro boundary");}
    CUDA_CHECK_THROW(cudaDeviceSynchronize());context_=context;worlds_.copy_from_host(worlds.data(),worlds.size());prepare(nullptr);CUDA_CHECK_THROW(cudaDeviceSynchronize());
}
}
