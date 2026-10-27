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
    rl::SelectorPolicy policy_;nn::CpuMlp<float> motor_,selector_;PondFrame frame_;std::string detail_;
public:
    CpuPlayback(const rl::SelectorPolicy& policy,uint32_t seed,uint32_t environment,std::string reason):
        policy_(policy),motor_(rl::swimmer_specification(4,33)),selector_(rl::selector_spec()),detail_(std::move(reason)) {
        motor_.params()=policy.motor;selector_.params()=policy.selector;frame_.state=deep_pond::initial(seed,environment);
    }
    void continue_from(const PondFrame& state) {frame_=state;}
    const PondFrame& frame() const override {return frame_;}
    const char* backend() const override {return "CPU";}
    const std::string& detail() const override {return detail_;}
    const PondFrame& step() override {
        auto& state=frame_.state;
        if(state.steps%policy_.decision_steps==0) {
            float o[36],scores[3];pond_selector::observe(state,frame_.selected,o);
            for(int f=0;f<3;++f) {float features[32];pond_selector::features(o,f,features);selector_.forward(features,1);scores[f]=selector_.output()[0];}
            frame_.selected=0;for(int f=1;f<3;++f) if(scores[f]>scores[frame_.selected]) frame_.selected=f;
        }
        float o[33];deep_pond::observe(state,o);primary_pond_food(o,frame_.selected);motor_.forward(o,1);deep_pond::advance(state,motor_.output());return frame_;
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
    GetPondCudaApi entry() const {
        if(!handle_) return nullptr;
#ifdef _WIN32
        const auto symbol=GetProcAddress(static_cast<HMODULE>(handle_),"cudarl_get_pond_cuda_api");
        GetPondCudaApi result=nullptr;static_assert(sizeof(result)==sizeof(symbol));std::memcpy(&result,&symbol,sizeof(result));return result;
#else
        return reinterpret_cast<GetPondCudaApi>(dlsym(handle_,"cudarl_get_pond_cuda_api"));
#endif
    }
};
class CudaPlayback final:public PondPlayback {
    std::unique_ptr<Module> module_;const PondCudaApi* api_;void* handle_;
    rl::SelectorPolicy policy_;PondFrame frame_;std::string detail_;bool fallback_allowed_;
    std::unique_ptr<CpuPlayback> fallback_;
public:
    CudaPlayback(std::unique_ptr<Module> module,const PondCudaApi* api,void* handle,const rl::SelectorPolicy& policy,
        uint32_t seed,uint32_t environment,std::string detail,bool allow_fallback):module_(std::move(module)),api_(api),handle_(handle),
        policy_(policy),detail_(std::move(detail)),fallback_allowed_(allow_fallback) {frame_.state=deep_pond::initial(seed,environment);}
    ~CudaPlayback() override {if(handle_) api_->destroy(handle_);}
    const PondFrame& frame() const override {return fallback_?fallback_->frame():frame_;}
    const char* backend() const override {return fallback_?"CPU":"CUDA";}
    const std::string& detail() const override {return fallback_?fallback_->detail():detail_;}
    const PondFrame& step() override {
        if(fallback_) return fallback_->step();char error[512]{};
        PondFrame next;
        if(api_->step(handle_,&next.state,&next.selected,error,sizeof(error))) {frame_=next;return frame_;}
        if(!fallback_allowed_) throw std::runtime_error(std::string("CUDA playback failed: ")+error);
        fallback_=std::make_unique<CpuPlayback>(policy_,frame_.state.seed,frame_.state.environment,std::string("CUDA playback failed; continued on CPU: ")+error);
        fallback_->continue_from(frame_);return fallback_->step();
    }
};
}
std::unique_ptr<PondPlayback> create_pond_playback(const rl::SelectorPolicy& policy,const std::string& preference,
    const std::string& module_path,uint32_t seed,uint32_t environment) {
    if(preference!="auto" && preference!="cpu" && preference!="cuda") throw std::invalid_argument("Pond backend must be auto, cpu or cuda");
    if(preference=="cpu") return std::make_unique<CpuPlayback>(policy,seed,environment,"CPU selected explicitly");
    std::string reason="Optional CUDA module not installed";
    if(!module_path.empty() && std::filesystem::exists(std::filesystem::u8path(module_path))) {
        auto module=std::make_unique<Module>(module_path);const auto entry=module->entry();
        if(entry) {
            const auto* api=entry();
            if(api && api->version==1 && api->api_size==sizeof(PondCudaApi) && api->state_size==sizeof(deep_pond::State) && api->create && api->step && api->destroy) {
                char message[512]{};void* handle=api->create(policy.motor.data(),static_cast<int>(policy.motor.size()),policy.selector.data(),static_cast<int>(policy.selector.size()),policy.decision_steps,seed,environment,message,sizeof(message));
                if(handle) return std::make_unique<CudaPlayback>(std::move(module),api,handle,policy,seed,environment,message,preference=="auto");
                reason=std::string("CUDA health check unavailable: ")+message;
            } else reason="CUDA module has an incompatible interface";
        } else reason="CUDA module could not load";
    }
    if(preference=="cuda") throw std::runtime_error(reason);
    return std::make_unique<CpuPlayback>(policy,seed,environment,reason);
}
}
