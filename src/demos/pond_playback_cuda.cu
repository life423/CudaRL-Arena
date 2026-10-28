#include "pond_cuda_api.h"
#include "pond_playback.h"
#include "rl/pond_selector_policy.h"
#include "rl/progress_selector_policy.h"
#include "rl/ppo_kernels.h"
#include "nn/mlp_gpu.h"
#include "cuda/device.h"
#include "cuda/cuda_memory.h"
#include <climits>
#include <cstdio>
#include <memory>
namespace cudarl::demos {
namespace {
__global__ void selector_observation(const deep_pond::State* state,const int* target,float* output) {pond_selector::observe(*state,*target,output);}
__global__ void progress_observation(const deep_pond::State* state,const int* target,const progress_selector::History* history,float* output) {progress_selector::observe(*state,*target,*history,output);}
__global__ void initialize_history(const deep_pond::State* state,progress_selector::History* history) {*history=progress_selector::initial_history(*state);}
__global__ void refresh_history(const deep_pond::State* state,progress_selector::History* history,int period) {progress_selector::update_history(*state,*history,period);}
__global__ void motor_observation(const float* input,const int* target,float* output) {
    for(int j=0;j<33;++j) output[j]=input[j];primary_pond_food(output,*target);
}
struct GpuPlayback {
    deep_pond::Batch world;nn::GpuMlp motor,selector;cudaStream_t stream=nullptr;
    deep::V current{};
    device_ptr<float> obs,features,motor_obs,logp;device_ptr<int> target;device_ptr<progress_selector::History> history;int tick=0,period;bool progress;
    GpuPlayback(const float* m,int nm,const float* p,int np,int interval,uint32_t seed,uint32_t environment):
        world(1,seed),motor(rl::swimmer_specification(4,33),1,7),selector(np==rl::progress_selector_spec().num_params()?rl::progress_selector_spec():rl::selector_spec(),3,7),period(interval),progress(np==rl::progress_selector_spec().num_params()) {
        if(interval<1 || interval>1200 || nm!=motor.num_params() || np!=selector.num_params()) throw std::invalid_argument("Invalid CUDA playback model dimensions");
        motor.set_params(std::vector<float>(m,m+nm));selector.set_params(std::vector<float>(p,p+np));
        obs.allocate(progress?40:36);features.allocate(progress?3*34:3*32);motor_obs.allocate(33);logp.allocate(1);target.allocate(1);
        const int initial=-1;target.copy_from_host(&initial,1);
        if(environment!=0) {
            const auto state=deep_pond::initial(seed,environment);float o[33];deep_pond::observe(state,o);
            CUDA_CHECK_THROW(cudaMemcpy(const_cast<deep_pond::State*>(world.states()),&state,sizeof(state),cudaMemcpyHostToDevice));
            CUDA_CHECK_THROW(cudaMemcpy(const_cast<float*>(world.obs()),o,sizeof(o),cudaMemcpyHostToDevice));
        }
        if(progress) {history.allocate(1);initialize_history<<<1,1>>>(world.states(),history.get());CUDA_CHECK_THROW(cudaGetLastError());}
        CUDA_CHECK_THROW(cudaDeviceSynchronize());CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    }
    ~GpuPlayback() {if(stream) {cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
    void step(deep_pond::State* state,int* selected) {
        if(tick%period==0) {
            if(progress) {
                progress_observation<<<1,1,0,stream>>>(world.states(),target.get(),history.get(),obs.get());CUDA_CHECK_THROW(cudaGetLastError());
                progress_selector::make_features(obs.get(),features.get(),1,stream);
            } else {
                selector_observation<<<1,1,0,stream>>>(world.states(),target.get(),obs.get());CUDA_CHECK_THROW(cudaGetLastError());
                pond_selector::make_features(obs.get(),features.get(),1,stream);
            }
            selector.forward(features.get(),3,stream);
            rl::sample_actions(selector.output(),1,3,0,0,true,target.get(),logp.get(),stream);
        }
        motor_observation<<<1,1,0,stream>>>(world.obs(),target.get(),motor_obs.get());CUDA_CHECK_THROW(cudaGetLastError());
        motor.forward(motor_obs.get(),1,stream);world.step(motor.output(),{},stream,false,INT_MAX,current);
        if(progress && (tick+1)%period==0) {refresh_history<<<1,1,0,stream>>>(world.states(),history.get(),period);CUDA_CHECK_THROW(cudaGetLastError());}
        CUDA_CHECK_THROW(cudaMemcpyAsync(state,world.states(),sizeof(*state),cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(selected,target.get(),sizeof(*selected),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));++tick;
    }
};
void* create(const float* motor,int motor_count,const float* selector,int selector_count,int period,uint32_t seed,uint32_t environment,char* message,size_t size) {
    try {
        const auto device=probe_device();if(!device.available) throw std::runtime_error(device.error);
        auto playback=std::make_unique<GpuPlayback>(motor,motor_count,selector,selector_count,period,seed,environment);
        std::snprintf(message,size,"%s; CUDA device and kernel check passed",device.name.c_str());return playback.release();
    } catch(const std::exception& e) {std::snprintf(message,size,"%s",e.what());return nullptr;}
}
int step(void* handle,deep_pond::State* state,int* target,char* error,size_t size) {
    try {static_cast<GpuPlayback*>(handle)->step(state,target);return 1;}
    catch(const std::exception& e) {std::snprintf(error,size,"%s",e.what());return 0;}
}
void destroy(void* handle) {delete static_cast<GpuPlayback*>(handle);}
int set_current(void* handle,deep::V current,char* error,size_t size) {
    try {validate_pond_current(current);static_cast<GpuPlayback*>(handle)->current=current;return 1;}
    catch(const std::exception& e) {std::snprintf(error,size,"%s",e.what());return 0;}
}
}
}
#ifdef _WIN32
#define CUDARL_PLAYBACK_EXPORT __declspec(dllexport)
#else
#define CUDARL_PLAYBACK_EXPORT __attribute__((visibility("default")))
#endif
extern "C" CUDARL_PLAYBACK_EXPORT const cudarl::demos::PondCudaApi* cudarl_get_pond_cuda_api() {
    static const cudarl::demos::PondCudaApi api{1,sizeof(cudarl::demos::PondCudaApi),sizeof(cudarl::deep_pond::State),
        cudarl::demos::create,cudarl::demos::step,cudarl::demos::destroy};return &api;
}
extern "C" CUDARL_PLAYBACK_EXPORT const cudarl::demos::PondCudaCurrentApi* cudarl_get_pond_current_api() {
    static const cudarl::demos::PondCudaCurrentApi api{1,sizeof(cudarl::demos::PondCudaCurrentApi),cudarl::demos::set_current};return &api;
}
