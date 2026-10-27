#include <atomic>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include "rl/pond_selector_ppo.h"
#include "nn/mlp_cpu.h"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
namespace rl=cudarl::rl;
namespace {
std::atomic<bool> stopping{false};void stop(int) {stopping.store(true);}
#ifdef _WIN32
BOOL WINAPI console_event(DWORD event) {if(event==CTRL_C_EVENT || event==CTRL_BREAK_EVENT) {stopping.store(true);return TRUE;}return FALSE;}
#endif
float recovery_time(const std::string& fixture,const std::vector<float>& weights,const std::vector<float>& motor_weights,int period) {
    if(fixture.empty()) return 0;
    const auto bytes=cudarl::checkpoint::read(fixture);cudarl::checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x53544131u) throw std::runtime_error("Invalid recovery fixture");
    auto state=in.value<cudarl::deep_pond::State>();int target=in.value<int>();in.finish();
    cudarl::nn::CpuMlp<float> motor(rl::swimmer_specification(4,33)),selector(rl::selector_spec());motor.params()=motor_weights;selector.params()=weights;
    for(int tick=0;tick<400;++tick) {
        float o[36];cudarl::pond_selector::observe(state,target,o);
        if(state.steps%period==0) {
            float scores[3];for(int f=0;f<3;++f) {float x[32];cudarl::pond_selector::features(o,f,x);selector.forward(x,1);scores[f]=selector.output()[0];}
            target=0;for(int f=1;f<3;++f) if(scores[f]>scores[target]) target=f;
        }
        cudarl::demos::primary_pond_food(o,target);motor.forward(o,1);
        if(cudarl::deep_pond::advance(state,motor.output()).eaten_slot>=0) return (tick+1)*.05f;
    }
    return -1;
}
}
int main(int argc,char** argv) {
    try {
        std::string source,path,policy,motor_policy,fixture;int updates=400;rl::SelectorConfig cfg;bool explicit_seed=false,explicit_envs=false,explicit_period=false,explicit_recovery=false;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--recovery") {cfg.recovery=true;explicit_recovery=true;continue;}
            if(arg=="--help") {std::puts("pond_selector_train (--source FROZEN_POND_COPY | --motor-policy BUNDLE) --checkpoint SELECTOR_SAVE --policy BEST_POLICY\n[--updates 400] [--seed 11] [--envs 1024] [--decision-steps 20] [--recovery] [--fixture STALL]\nNative categorical PPO over food choices; frozen motor and unchanged physics. Updates are additional; zero runs until Ctrl+C. Existing full saves supply their configuration automatically. Without a full save, --motor-policy transfers selector weights with fresh optimizer/worlds. --recovery uses mixed resting joint poses, five-minute training episodes and a -5 terminal penalty for a 20-second low-speed food drought; these resets apply only to training.");return 0;}
            if(i+1>=argc) throw std::invalid_argument("Missing argument value");const std::string v=argv[++i];
            if(arg=="--source") source=v;else if(arg=="--checkpoint") path=v;else if(arg=="--policy") policy=v;else if(arg=="--motor-policy") motor_policy=v;else if(arg=="--fixture") fixture=v;
            else if(arg=="--updates") updates=std::stoi(v);else if(arg=="--seed") {cfg.seed=static_cast<uint32_t>(std::stoul(v));explicit_seed=true;}
            else if(arg=="--envs") {cfg.environments=std::stoi(v);explicit_envs=true;}else if(arg=="--decision-steps") {cfg.decision_steps=std::stoi(v);explicit_period=true;}
            else throw std::invalid_argument("Unknown argument: "+arg);
        }
        if((source.empty() && motor_policy.empty()) || path.empty() || policy.empty() || updates<0 || source==path || policy==source || policy==path || motor_policy==path || motor_policy==policy) throw std::invalid_argument("Provide separate input, training and policy files and a nonnegative update budget");
        std::vector<uint8_t> saved;cudarl::checkpoint::Lease lease(path);
        if(cudarl::checkpoint::exists(path)) {
            saved=cudarl::checkpoint::read(path);cudarl::checkpoint::Reader header{saved};
            if(header.value<uint32_t>()!=0x53454C31u) throw std::runtime_error("Unsupported selector checkpoint");
            const auto version=header.value<uint32_t>();if(version!=2 && version!=3) throw std::runtime_error("Unsupported selector checkpoint");
            rl::SelectorConfig previous;previous.environments=header.value<int>();previous.decision_steps=header.value<int>();previous.seed=header.value<uint32_t>();if(version==3) previous.recovery=header.value<bool>();
            if(!explicit_seed) cfg.seed=previous.seed;if(!explicit_envs) cfg.environments=previous.environments;if(!explicit_period) cfg.decision_steps=previous.decision_steps;if(!explicit_recovery) cfg.recovery=previous.recovery;
        }
        std::vector<float> motor_weights;int motor_update=0;rl::SelectorPolicy inherited;
        if(!motor_policy.empty()) {
            inherited=rl::SelectorPolicy::load(motor_policy);motor_weights=inherited.motor;motor_update=inherited.motor_update;
            if(saved.empty() && !explicit_period) cfg.decision_steps=inherited.decision_steps;
        } else {
            rl::SwimmerConfig motor_cfg;motor_cfg.num_envs=1024;motor_cfg.seed=7;motor_cfg.evaluation_every=25;motor_cfg.training_checkpoint_path=source;
            rl::DeepPondTrainer original(motor_cfg);if(!original.checkpoint_status().resumed || !original.best_policy().valid()) throw std::runtime_error("Source must contain a trained best policy");
            motor_weights=original.best_policy().parameters;motor_update=original.best_policy().update;
        }
        rl::SelectorTrainer trainer(cfg,motor_weights,motor_update);
        if(!saved.empty()) trainer.restore(saved);else if(!inherited.selector.empty()) trainer.actor.set_params(inherited.selector);
        const bool resume=trainer.update>0;std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
#ifdef _WIN32
        SetConsoleCtrlHandler(console_event,TRUE);
#endif
        const auto parent=std::filesystem::path(path).parent_path();if(!parent.empty()) std::filesystem::create_directories(parent);
        std::ofstream log(path+".csv",resume?std::ios::app:std::ios::trunc);if(!log) throw std::runtime_error("Could not open learning log");
        if(!resume) log<<"update,fixed_food_per_minute,best_food_per_minute,physics_steps,wall_seconds,recovery_seconds\n";
        const auto start=rl::Clock::now();
        for(int completed=0;!stopping.load() && (updates==0 || completed<=updates);++completed) {
            if(trainer.update%25==0 || (updates>0 && completed==updates) || completed==0) {
                const int foods=trainer.evaluate(0x6A91B23Du,false);const float recovery=recovery_time(fixture,trainer.actor.get_params(),trainer.frozen_motor,cfg.decision_steps);
                if(recovery>=0 && foods>trainer.best.foods) trainer.best={trainer.update,motor_update,cfg.decision_steps,foods,64,trainer.actor.get_params(),trainer.frozen_motor};
                const double seconds=std::chrono::duration<double>(rl::Clock::now()-start).count();
                std::printf("SELECTOR update=%d current=%.6f best=%.6f best_update=%d motor_update=%d seed=%u recovery=%.2fs elapsed=%.1fs\n",trainer.update,foods/64.,trainer.best.food_per_minute(),trainer.best.update,trainer.motor_update,cfg.seed,recovery,seconds);std::fflush(stdout);
                log<<trainer.update<<','<<foods/64.<<','<<trainer.best.food_per_minute()<<','<<static_cast<long long>(trainer.update)*cfg.environments*rl::SelectorTrainer::horizon*cfg.decision_steps<<','<<seconds<<','<<recovery<<'\n';log.flush();
                cudarl::checkpoint::write_atomic(path,trainer.snapshot());if(trainer.best.foods>=0) trainer.best.save(policy);
            }
            if(updates==0 || completed<updates) trainer.train_update();
        }
        cudarl::checkpoint::write_atomic(path,trainer.snapshot());if(trainer.best.foods>=0) trainer.best.save(policy);
        if(trainer.motor.get_params()!=trainer.frozen_motor) throw std::runtime_error("Frozen motor weights changed");
        std::printf("SAVED selector_update=%d best=%.6f motor_unchanged=true checkpoint=%s\n",trainer.update,trainer.best.food_per_minute(),path.c_str());return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Selector training: %s\n",e.what());return 1;}
}
