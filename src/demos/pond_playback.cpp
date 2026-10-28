#include "pond_playback.h"
#include "pond_cuda_api.h"
#include "nn/mlp_cpu.h"
#include <filesystem>
#include <cstring>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif
namespace cudarl::demos {
namespace {
class CpuPlayback final:public PondPlayback {
    rl::SelectorPolicy policy_;nn::CpuMlp<float> motor_,selector_;PondFrame frame_;std::string detail_;bool progress_;
public:
    CpuPlayback(const rl::SelectorPolicy& policy,const std::vector<float>& weights,bool progress,uint32_t seed,uint32_t environment,std::string reason):
        policy_(policy),motor_(rl::swimmer_specification(4,33)),selector_(progress?rl::progress_selector_spec():rl::selector_spec()),detail_(std::move(reason)),progress_(progress) {
        motor_.params()=policy.motor;selector_.params()=weights;frame_.state=deep_pond::initial(seed,environment);
        if(progress_) frame_.history=progress_selector::initial_history(frame_.state);
    }
    void continue_from(const PondFrame& state) {frame_=state;}
    void set_current(deep::V current) override {validate_pond_current(current);frame_.current=current;}
    const PondFrame& frame() const override {return frame_;}
    const char* backend() const override {return "CPU";}
    const std::string& detail() const override {return detail_;}
    const PondFrame& step() override {
        auto& state=frame_.state;
        if(state.steps%policy_.decision_steps==0) {
            float o[40],scores[3];
            if(progress_) progress_selector::observe(state,frame_.selected,frame_.history,o);else pond_selector::observe(state,frame_.selected,o);
            for(int f=0;f<3;++f) {float features[34];
                if(progress_) progress_selector::features(o,f,features);else pond_selector::features(o,f,features);
                selector_.forward(features,1);scores[f]=selector_.output()[0];}
            frame_.selected=0;for(int f=1;f<3;++f) if(scores[f]>scores[frame_.selected]) frame_.selected=f;
        }
        float o[33];deep_pond::observe(state,o);primary_pond_food(o,frame_.selected);motor_.forward(o,1);deep_pond::advance(state,motor_.output(),frame_.current);
        if(progress_ && state.steps%policy_.decision_steps==0) progress_selector::update_history(state,frame_.history,policy_.decision_steps);
        return frame_;
    }
};
class Module {
    void* handle_=nullptr;
public:
    explicit Module(const std::string& path) {
#ifdef _WIN32
        if(!path.empty()) handle_=LoadLibraryExW(std::filesystem::absolute(std::filesystem::u8path(path)).c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
        if(!path.empty()) handle_=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
#endif
    }
    ~Module() {
#ifdef _WIN32
        if(handle_) FreeLibrary(static_cast<HMODULE>(handle_));
#else
        if(handle_) dlclose(handle_);
#endif
    }
    template<class T> T entry(const char* name) const {
        if(!handle_) return nullptr;
#ifdef _WIN32
        const auto symbol=GetProcAddress(static_cast<HMODULE>(handle_),name);
        T result=nullptr;static_assert(sizeof(result)==sizeof(symbol));std::memcpy(&result,&symbol,sizeof(result));return result;
#else
        return reinterpret_cast<T>(dlsym(handle_,name));
#endif
    }
};
class CudaPlayback final:public PondPlayback {
    std::unique_ptr<Module> module_;const PondCudaApi* api_;const PondCudaCurrentApi* current_api_;void* handle_;
    rl::SelectorPolicy policy_;std::vector<float> selector_weights_;PondFrame frame_;std::string detail_;bool fallback_allowed_,progress_;
    std::unique_ptr<CpuPlayback> fallback_;
public:
    CudaPlayback(std::unique_ptr<Module> module,const PondCudaApi* api,const PondCudaCurrentApi* current_api,void* handle,const rl::SelectorPolicy& policy,const std::vector<float>& weights,bool progress,
        uint32_t seed,uint32_t environment,std::string detail,bool allow_fallback):module_(std::move(module)),api_(api),current_api_(current_api),handle_(handle),
        policy_(policy),selector_weights_(weights),detail_(std::move(detail)),fallback_allowed_(allow_fallback),progress_(progress) {
        frame_.state=deep_pond::initial(seed,environment);if(progress_) frame_.history=progress_selector::initial_history(frame_.state);
    }
    ~CudaPlayback() override {if(handle_) api_->destroy(handle_);}
    const PondFrame& frame() const override {return fallback_?fallback_->frame():frame_;}
    const char* backend() const override {return fallback_?"CPU":"CUDA";}
    const std::string& detail() const override {return fallback_?fallback_->detail():detail_;}
    void set_current(deep::V current) override {
        validate_pond_current(current);
        if(fallback_) {fallback_->set_current(current);return;}
        char error[512]{};
        if(current_api_) {
            if(!current_api_->set_current(handle_,current,error,sizeof(error))) throw std::runtime_error(std::string("Could not change CUDA current: ")+error);
            frame_.current=current;return;
        }
        if(deep::norm(current)==0) return;
        if(!fallback_allowed_) throw std::runtime_error("This CUDA playback module needs rebuilding for current control");
        fallback_=std::make_unique<CpuPlayback>(policy_,selector_weights_,progress_,frame_.state.seed,frame_.state.environment,"CUDA module has no current control; continued on CPU");
        fallback_->continue_from(frame_);fallback_->set_current(current);
    }
    const PondFrame& step() override {
        if(fallback_) return fallback_->step();char error[512]{};
        PondFrame next;
        if(api_->step(handle_,&next.state,&next.selected,error,sizeof(error))) {
            next.history=frame_.history;
            next.current=frame_.current;
            if(progress_ && next.state.steps%policy_.decision_steps==0) progress_selector::update_history(next.state,next.history,policy_.decision_steps);
            frame_=next;return frame_;
        }
        if(!fallback_allowed_) throw std::runtime_error(std::string("CUDA playback failed: ")+error);
        fallback_=std::make_unique<CpuPlayback>(policy_,selector_weights_,progress_,frame_.state.seed,frame_.state.environment,std::string("CUDA playback failed; continued on CPU: ")+error);
        fallback_->continue_from(frame_);return fallback_->step();
    }
};
}
static std::unique_ptr<PondPlayback> create_playback(const rl::SelectorPolicy& policy,const std::vector<float>& weights,bool progress,
    const std::string& preference,const std::string& module_path,uint32_t seed,uint32_t environment) {
    const auto spec=progress?rl::progress_selector_spec():rl::selector_spec();
    if(policy.motor.size()!=static_cast<size_t>(rl::swimmer_specification(4,33).num_params()) || weights.size()!=static_cast<size_t>(spec.num_params()) ||
        policy.decision_steps<1 || policy.decision_steps>1200 || 1200%policy.decision_steps) throw std::invalid_argument("Invalid playback controller dimensions/cadence");
    for(const auto* values:{&policy.motor,&weights}) for(float x:*values) if(!std::isfinite(x)) throw std::invalid_argument("Non-finite playback controller");
    if(preference!="auto" && preference!="cpu" && preference!="cuda") throw std::invalid_argument("Pond backend must be auto, cpu or cuda");
    if(preference=="cpu") return std::make_unique<CpuPlayback>(policy,weights,progress,seed,environment,"CPU selected explicitly");
    std::string reason="Optional CUDA module not installed";
    if(!module_path.empty() && std::filesystem::exists(std::filesystem::u8path(module_path))) {
        auto module=std::make_unique<Module>(module_path);const auto entry=module->entry<GetPondCudaApi>("cudarl_get_pond_cuda_api");
        if(entry) {
            const auto* api=entry();
            if(api && api->version==1 && api->api_size==sizeof(PondCudaApi) && api->state_size==sizeof(deep_pond::State) && api->create && api->step && api->destroy) {
                char message[512]{};void* handle=api->create(policy.motor.data(),static_cast<int>(policy.motor.size()),weights.data(),static_cast<int>(weights.size()),policy.decision_steps,seed,environment,message,sizeof(message));
                if(handle) {
                    const auto extension=module->entry<GetPondCudaCurrentApi>("cudarl_get_pond_current_api");
                    const auto* current_api=extension?extension():nullptr;
                    if(current_api && (current_api->version!=1 || current_api->api_size!=sizeof(PondCudaCurrentApi) || !current_api->set_current)) current_api=nullptr;
                    return std::make_unique<CudaPlayback>(std::move(module),api,current_api,handle,policy,weights,progress,seed,environment,message,preference=="auto");
                }
                reason=std::string("CUDA health check unavailable: ")+message;
            } else reason="CUDA module has an incompatible interface";
        } else reason="CUDA module could not load";
    }
    if(preference=="cuda") throw std::runtime_error(reason);
    return std::make_unique<CpuPlayback>(policy,weights,progress,seed,environment,reason);
}
std::unique_ptr<PondPlayback> create_pond_playback(const rl::SelectorPolicy& policy,const std::string& preference,const std::string& module_path,uint32_t seed,uint32_t environment) {
    return create_playback(policy,policy.selector,false,preference,module_path,seed,environment);
}
std::unique_ptr<PondPlayback> create_pond_playback(const rl::ProgressPolicy& policy,const std::string& preference,const std::string& module_path,uint32_t seed,uint32_t environment) {
    return create_playback(policy.baseline,policy.actor,true,preference,module_path,seed,environment);
}
}
