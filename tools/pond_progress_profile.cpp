#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cuda_profiler_api.h>
#include "rl/progress_selector_ppo.h"
#ifdef CUDARL_PROFILE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif
namespace rl=cudarl::rl;
namespace {
void push(const char* label) {
#ifdef CUDARL_PROFILE_NVTX
    nvtxRangePushA(label);
#else
    (void)label;
#endif
}
void pop() {
#ifdef CUDARL_PROFILE_NVTX
    nvtxRangePop();
#endif
}
struct Timing {
    cudaStream_t stream;cudaEvent_t event[4]{};rl::Clock::time_point host[5];
    explicit Timing(cudaStream_t s):stream(s) {for(auto& e:event) CUDA_CHECK_THROW(cudaEventCreate(&e));}
    ~Timing() {for(auto e:event) if(e) cudaEventDestroy(e);}
    static void mark(void* context,int phase) {
        auto& self=*static_cast<Timing*>(context);self.host[phase]=rl::Clock::now();
        if(phase>0) pop();
        const char* labels[4]={"rollout","bootstrap_gae","ppo_optimization","stream_wait"};
        if(phase<4) {push(labels[phase]);CUDA_CHECK_THROW(cudaEventRecord(self.event[phase],self.stream));}
    }
    double host_ms(int a,int b) const {return std::chrono::duration<double,std::milli>(host[b]-host[a]).count();}
    float gpu_ms(int a,int b) const {float result=0;CUDA_CHECK_THROW(cudaEventElapsedTime(&result,event[a],event[b]));return result;}
};
}
int main(int argc,char** argv) {
    try {
        std::string baseline_path,checkpoint,output;int n=8192,warmup=2,updates=3;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--help") {std::puts("pond_progress_profile --baseline FROZEN_POLICY --output CSV [--checkpoint READ_ONLY_FULL_SAVE] [--envs 8192] [--warmup 2] [--updates 3]\nDisposable profiling only. No policy/checkpoint writes. CUDA profiler capture brackets only measured updates; optional NVTX labels rollout/bootstrap/PPO/wait. CUDA event spans include any device-idle gaps within a phase.");return 0;}
            if(i+1>=argc) throw std::invalid_argument("Missing argument value");const std::string value=argv[++i];
            if(arg=="--baseline") baseline_path=value;else if(arg=="--checkpoint") checkpoint=value;else if(arg=="--output") output=value;
            else if(arg=="--envs") n=std::stoi(value);else if(arg=="--warmup") warmup=std::stoi(value);else if(arg=="--updates") updates=std::stoi(value);else throw std::invalid_argument("Unknown profile argument");
        }
        if(baseline_path.empty() || output.empty() || n<16 || n>32768 || warmup<0 || warmup>10 || updates<1 || updates>20) throw std::invalid_argument("Invalid profiling configuration");
        const auto output_absolute=std::filesystem::weakly_canonical(output);
        if(output_absolute==std::filesystem::weakly_canonical(baseline_path) || (!checkpoint.empty() && output_absolute==std::filesystem::weakly_canonical(checkpoint))) throw std::invalid_argument("Profiling output must not overwrite an input");
        const auto original=cudarl::checkpoint::read(baseline_path);const auto baseline=rl::SelectorPolicy::load(baseline_path);
        std::vector<uint8_t> saved;uint32_t seed=71;
        if(!checkpoint.empty()) {
            saved=cudarl::checkpoint::read(checkpoint);cudarl::checkpoint::Reader header{saved};
            if(header.value<uint32_t>()!=0x50535431u || header.value<uint32_t>()!=1) throw std::invalid_argument("Unsupported profile input save");
            n=header.value<int>();seed=header.value<uint32_t>();if(n<16 || n>32768) throw std::invalid_argument("Invalid saved batch");
        }
        rl::ProgressTrainer trial(baseline,n,seed);if(!saved.empty()) trial.restore(saved);
        for(int i=0;i<warmup;++i) trial.train_update();
        const auto parent=std::filesystem::path(output).parent_path();if(!parent.empty()) std::filesystem::create_directories(parent);
        std::ofstream csv(output);if(!csv) throw std::runtime_error("Could not write profile timings");
        csv<<"update,environments,host_rollout_ms,host_bootstrap_ms,host_optimization_ms,host_wait_ms,host_total_ms,gpu_rollout_span_ms,gpu_bootstrap_span_ms,gpu_optimization_span_ms,gpu_total_span_ms,simulation_steps_per_second\n";
        Timing timing(trial.stream.value);rl::ProgressInstrumentation instrument{&Timing::mark,&timing};
        CUDA_CHECK_THROW(cudaDeviceSynchronize());CUDA_CHECK_THROW(cudaProfilerStart());push("measured_training");
        for(int i=0;i<updates;++i) {
            push("update");trial.train_update(&instrument);pop();
            const double total=timing.host_ms(0,4);const double rate=double(n)*rl::ProgressTrainer::horizon*baseline.decision_steps/(total*.001);
            csv<<std::setprecision(10)<<trial.update<<','<<n<<','<<timing.host_ms(0,1)<<','<<timing.host_ms(1,2)<<','<<timing.host_ms(2,3)<<','<<timing.host_ms(3,4)<<','<<total<<','<<timing.gpu_ms(0,1)<<','<<timing.gpu_ms(1,2)<<','<<timing.gpu_ms(2,3)<<','<<timing.gpu_ms(0,3)<<','<<rate<<'\n';
            std::printf("PROFILE update=%d envs=%d host_ms=%.3f GPU_span_ms=%.3f rollout_ms=%.3f optimization_ms=%.3f steps_s=%.0f\n",trial.update,n,total,timing.gpu_ms(0,3),timing.gpu_ms(0,1),timing.gpu_ms(2,3),rate);std::fflush(stdout);
        }
        pop();CUDA_CHECK_THROW(cudaProfilerStop());
        if(trial.motor.get_params()!=baseline.motor || original!=cudarl::checkpoint::read(baseline_path) || (!checkpoint.empty() && saved!=cudarl::checkpoint::read(checkpoint))) throw std::runtime_error("Protected profile inputs changed");
        if(!csv) throw std::runtime_error("Profile write failed");std::puts("PROFILE_SAVED no_checkpoint_writes=true original_inputs_unchanged=true");return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Profiling: %s\n",e.what());return 1;}
}
