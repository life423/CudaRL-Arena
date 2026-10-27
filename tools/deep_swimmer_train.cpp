#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include "rl/swimmer_ppo.h"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
namespace {
std::atomic<bool> stopping{false};
void interrupt(int) {stopping.store(true);}
#ifdef _WIN32
BOOL WINAPI console_event(DWORD event) {
    if(event==CTRL_C_EVENT || event==CTRL_BREAK_EVENT) {stopping.store(true);return TRUE;}return FALSE;
}
#endif
std::string default_path() {
#ifdef _MSC_VER
    char* appdata=nullptr;size_t size=0;
    if(_dupenv_s(&appdata,&size,"APPDATA")!=0 || !appdata) throw std::runtime_error("APPDATA is unavailable; pass --checkpoint PATH");
    std::filesystem::path root(appdata);std::free(appdata);
#else
    const char* appdata=std::getenv("APPDATA");
    if(!appdata) throw std::runtime_error("Pass --checkpoint PATH");std::filesystem::path root(appdata);
#endif
    return (root/"Godot"/"app_userdata"/"CudaRL Arena"/"pond-policies"/"deep-v1"/"training.bin").string();
}
void evaluate_actor(cudarl::rl::DeepWaterTrainer& trainer,const std::vector<float>& parameters,uint32_t seed,const char* label) {
    namespace d=cudarl::deep;
    constexpr int n=64;d::Batch trials(n,seed);trainer.actor.set_params(parameters);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    std::vector<d::State> previous(n),now(n);float min_depth[n],max_depth[n];
    CUDA_CHECK_THROW(cudaMemcpy(previous.data(),trials.states(),n*sizeof(d::State),cudaMemcpyDeviceToHost));
    float target_origin[n];for(int i=0;i<n;++i) {min_depth[i]=max_depth[i]=previous[i].body.position.y;target_origin[i]=previous[i].body.position.y;}
    int foods=0,upper=0,lower=0,ascending=0,descending=0,success=0;
    for(int t=0;t<1200;++t) {
        trainer.actor.forward(trials.obs(),n,trainer.stream.value);
        trials.step(trainer.actor.output(),{},trainer.stream.value,false);
        CUDA_CHECK_THROW(cudaMemcpyAsync(now.data(),trials.states(),n*sizeof(d::State),cudaMemcpyDeviceToHost,trainer.stream.value));
        CUDA_CHECK_THROW(cudaStreamSynchronize(trainer.stream.value));
        for(int i=0;i<n;++i) {
            const auto& s=now[i];if(!std::isfinite(s.body.position.y)) throw std::runtime_error("Non-finite evaluation state");
            min_depth[i]=std::min(min_depth[i],s.body.position.y);max_depth[i]=std::max(max_depth[i],s.body.position.y);
            if(s.eaten>previous[i].eaten) {
                ++foods;if(previous[i].food.y>= -2) ++upper;else ++lower;
                if(previous[i].food.y>target_origin[i]) ++ascending;else ++descending;
                target_origin[i]=s.body.position.y;
            }
        }
        previous.swap(now);
    }
    double depth_span=0;for(int i=0;i<n;++i) {if(previous[i].eaten>0) ++success;depth_span+=(max_depth[i]-min_depth[i])*4.;}
    std::printf("HOLDOUT %s seed=%u trials=64 seconds=60 foods=%d food_per_minute=%.6f successful=%d upper=%d lower=%d ascending=%d descending=%d mean_depth_span_cm=%.3f\n",
        label,seed,foods,foods/64.,success,upper,lower,ascending,descending,depth_span/64.);std::fflush(stdout);
}
void evaluate_actor(cudarl::rl::DeepPondTrainer& trainer,const std::vector<float>& parameters,uint32_t seed,const char* label) {
    namespace p=cudarl::deep_pond;
    constexpr int n=64;p::Batch trials(n,seed);trainer.actor.set_params(parameters);CUDA_CHECK_THROW(cudaDeviceSynchronize());
    std::vector<p::State> states(n);int slots[3]{},ascending=0,descending=0;std::vector<float> old_depth(n);
    CUDA_CHECK_THROW(cudaMemcpy(states.data(),trials.states(),n*sizeof(p::State),cudaMemcpyDeviceToHost));
    for(int i=0;i<n;++i) old_depth[i]=states[i].body.position.y;
    for(int t=0;t<1200;++t) {
        trainer.actor.forward(trials.obs(),n,trainer.stream.value);trials.step(trainer.actor.output(),{},trainer.stream.value,false);
        CUDA_CHECK_THROW(cudaMemcpyAsync(states.data(),trials.states(),n*sizeof(p::State),cudaMemcpyDeviceToHost,trainer.stream.value));
        CUDA_CHECK_THROW(cudaStreamSynchronize(trainer.stream.value));
        for(int i=0;i<n;++i) if(states[i].last_eaten>=0) {
            ++slots[states[i].last_eaten];if(states[i].body.position.y>old_depth[i]) ++ascending;else ++descending;
            old_depth[i]=states[i].body.position.y;
        }
    }
    int foods=0,success=0;for(const auto& s:states) {foods+=s.eaten;if(s.eaten>0) ++success;}
    std::printf("POND_HOLDOUT %s seed=%u food_per_minute=%.6f successful=%d slots=%d,%d,%d ascending=%d descending=%d\n",
        label,seed,foods/64.,success,slots[0],slots[1],slots[2],ascending,descending);std::fflush(stdout);
}
template<class Trainer> int run(const cudarl::rl::SwimmerConfig& cfg,bool evaluation,uint32_t evaluation_seed) {
    std::printf("DEEP_WATER_PPO checkpoint=%s\nCtrl+C saves and stops. Keep this console open while training.\n",cfg.training_checkpoint_path.c_str());std::fflush(stdout);
    Trainer trainer(cfg);
    if(evaluation) {
        const auto current=trainer.actor_params(),best=trainer.best_policy().parameters;
        evaluate_actor(trainer,Trainer::initial_actor_params(cfg.seed),evaluation_seed,"untrained");
        evaluate_actor(trainer,current,evaluation_seed,"latest");if(!best.empty()) evaluate_actor(trainer,best,evaluation_seed,"best");return 0;
    }
    const auto& status=trainer.checkpoint_status();
    std::printf("%s at update %d; %d worlds; fixed 5 ms physics\n",status.resumed?"Resumed":(status.imported?"Transferred swimming":"Started"),trainer.current_stats().update,cfg.num_envs);
    int completed=0;const auto start=std::chrono::steady_clock::now();
    while(!stopping.load() && (cfg.total_updates==0 || completed<cfg.total_updates)) {
        const auto stats=trainer.update();++completed;
        if(stats.update%25==0 || completed==1) {
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            const double steps=static_cast<double>(completed)*cfg.num_envs*cudarl::rl::rollout;
            std::printf("update=%d current=%.3f best=%.3f food/min | %.0f steps/s | %.2fx time per world | saved=%d\n",
                stats.update,stats.evaluation_foods*60./(64*stats.evaluation_seconds),trainer.best_policy().food_per_minute(),
                steps/seconds,steps/seconds*0.05/cfg.num_envs,trainer.checkpoint_status().saved_update);std::fflush(stdout);
        }
        if(!trainer.checkpoint_status().error.empty()) throw std::runtime_error(trainer.checkpoint_status().error);
    }
    trainer.save_checkpoint();if(!trainer.checkpoint_status().error.empty()) throw std::runtime_error(trainer.checkpoint_status().error);
    std::printf("Saved full training at update %d. Open Deep water in Godot to resume watching.\n",trainer.current_stats().update);return 0;
}
}
int main(int argc,char** argv) {
    try {
        cudarl::rl::SwimmerConfig cfg;cfg.num_envs=1024;cfg.seed=7;cfg.total_updates=0;cfg.evaluation_every=25;
        cfg.training_checkpoint_path=default_path();
        bool evaluation=false,pond=false,custom_path=false;uint32_t evaluation_seed=0;std::string source;
        for(int i=1;i<argc;++i) {
            const std::string a=argv[i];
            if(a=="--pond") {pond=true;continue;}
            if(a=="--help") {
                std::puts("Deep-water PPO, native C++/CUDA, no renderer.\n"
                    "deep_swimmer_train [--updates N] [--checkpoint PATH] [--envs N] [--seed N]\n"
                    "--pond trains the three-food deep pond, transferring the original 3D swimmer on first use.\n"
                    "--source PATH overrides the original swimming checkpoint for that first transfer.\n"
                    "Evaluation only: --evaluate-seed N compares untrained/latest/best on new 3D trials.\n"
                    "Defaults: unlimited updates, 1024 worlds, seed 7, same save as Godot.\n"
                    "Stop with Ctrl+C to save full training state, then reopen Deep water in Godot.\n"
                    "Close the Godot training scene before starting; one process owns the save.\n"
                    "Physics stays at 5 ms substeps; acceleration comes from faster execution.");return 0;
            }
            if(i+1>=argc) throw std::invalid_argument("Missing command argument");
            const std::string value=argv[++i];
            if(a=="--updates") cfg.total_updates=std::stoi(value);
            else if(a=="--envs") cfg.num_envs=std::stoi(value);
            else if(a=="--seed") cfg.seed=static_cast<uint32_t>(std::stoul(value));
            else if(a=="--checkpoint") {cfg.training_checkpoint_path=value;custom_path=true;}
            else if(a=="--source") source=value;
            else if(a=="--evaluate-seed") {evaluation=true;evaluation_seed=static_cast<uint32_t>(std::stoul(value));}
            else throw std::invalid_argument("Unknown option: "+a);
        }
        if(cfg.total_updates<0 || cfg.num_envs<1 || cfg.num_envs>32768) throw std::invalid_argument("Invalid update budget or population");
        if(pond) {
            cfg.shuffle_food_inputs=false;cfg.deep_source_checkpoint=source.empty()?default_path():source;
            if(!custom_path) cfg.training_checkpoint_path=(std::filesystem::path(default_path()).parent_path().parent_path()/"deep-pond-v1"/"training.bin").string();
        }
        std::signal(SIGINT,interrupt);
#ifdef _WIN32
        SetConsoleCtrlHandler(console_event,TRUE);
#endif
        return pond?run<cudarl::rl::DeepPondTrainer>(cfg,evaluation,evaluation_seed):run<cudarl::rl::DeepWaterTrainer>(cfg,evaluation,evaluation_seed);
    } catch(const std::exception& e) {std::fprintf(stderr,"Deep-water training: %s\n",e.what());return 1;}
}
