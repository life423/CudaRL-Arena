#include <atomic>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include "rl/progress_selector_ppo.h"
namespace rl=cudarl::rl;
namespace {std::atomic<bool> stopping{false};void stop(int) {stopping.store(true);}}
int main(int argc,char** argv) {
    try {
        std::string input,output;int environments=4096,updates=128;uint32_t seed=71;bool bench=false,large=false;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];if(arg=="--benchmark") {bench=true;continue;}
            if(arg=="--benchmark-large") {bench=true;large=true;continue;}
            if(arg=="--help") {std::puts("pond_progress_train --baseline FROZEN_BUNDLE --output DIR [--envs 4096] [--seed 71] [--updates 128] [--benchmark]\nIsolated progress-aware categorical PPO, frozen motor, 25% calm and otherwise uniform 0..1 cm/s currents. Full resume from DIR/training.bin. Fresh critic/optimizer for the new input architecture. Validation every 32 updates; candidate files never replace the pond model.");return 0;}
            if(i+1>=argc) throw std::invalid_argument("Missing argument value");const std::string value=argv[++i];
            if(arg=="--baseline") input=value;else if(arg=="--output") output=value;else if(arg=="--envs") environments=std::stoi(value);
            else if(arg=="--seed") seed=static_cast<uint32_t>(std::stoul(value));else if(arg=="--updates") updates=std::stoi(value);else throw std::invalid_argument("Unknown argument: "+arg);
        }
        if(input.empty() || output.empty() || updates<0 || environments<16 || environments>32768) throw std::invalid_argument("Supply baseline, separate output directory, 16..32768 environments and nonnegative updates");
        if(std::filesystem::weakly_canonical(std::filesystem::path(input).parent_path())==std::filesystem::weakly_canonical(output)) throw std::invalid_argument("Keep candidate outputs in a different directory from the preserved baseline");
        const auto original=cudarl::checkpoint::read(input);const auto baseline=rl::SelectorPolicy::load(input);
        std::filesystem::create_directories(output);
        if(bench) {
            std::ofstream csv(std::filesystem::path(output)/(large?"batch-benchmark-large.csv":"batch-benchmark.csv"));csv<<"environments,updates,wall_seconds,simulation_steps_per_second\n";
            const std::vector<int> sizes=large?std::vector<int>{16384,32768}:std::vector<int>{1024,4096,8192};
            for(int n:sizes) {
                rl::ProgressTrainer trial(baseline,n,seed);trial.train_update();const auto start=rl::Clock::now();
                for(int i=0;i<3;++i) trial.train_update();const double seconds=std::chrono::duration<double>(rl::Clock::now()-start).count();
                const double rate=double(n)*rl::ProgressTrainer::horizon*baseline.decision_steps*3/seconds;
                csv<<n<<",3,"<<seconds<<','<<rate<<'\n';std::printf("BATCH envs=%d elapsed=%.3f steps_per_second=%.0f\n",n,seconds,rate);std::fflush(stdout);
                if(trial.motor.get_params()!=baseline.motor) throw std::runtime_error("Frozen motor changed");
            }
            return 0;
        }
        const auto checkpoint=(std::filesystem::path(output)/"training.bin").string();cudarl::checkpoint::Lease lease(checkpoint);
        std::vector<uint8_t> saved;
        if(cudarl::checkpoint::exists(checkpoint)) {
            saved=cudarl::checkpoint::read(checkpoint);cudarl::checkpoint::Reader header{saved};
            if(header.value<uint32_t>()!=0x50535431u || header.value<uint32_t>()!=1) throw std::runtime_error("Unsupported progress save");
            environments=header.value<int>();seed=header.value<uint32_t>();
        }
        rl::ProgressTrainer trainer(baseline,environments,seed);if(!saved.empty()) trainer.restore(saved);
        std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
        const int start_update=trainer.update;const auto start=rl::Clock::now();
        std::ofstream csv(std::filesystem::path(output)/"learning.csv",saved.empty()?std::ios::trunc:std::ios::app);
        if(saved.empty()) csv<<"update,calm_fpm,px_current_fpm,diagonal_current_fpm,wall_seconds\n";
        for(int completed=0;completed<=updates && !stopping.load();++completed) {
            if(completed==0 || trainer.update%32==0 || completed==updates) {
                const auto scores=trainer.evaluate();const auto policy=trainer.policy();
                policy.save((std::filesystem::path(output)/("candidate-"+std::to_string(trainer.update)+".bin")).string());
                policy.save((std::filesystem::path(output)/"latest.bin").string());cudarl::checkpoint::write_atomic(checkpoint,trainer.snapshot());
                const double elapsed=std::chrono::duration<double>(rl::Clock::now()-start).count();
                csv<<trainer.update<<','<<scores[0]<<','<<scores[1]<<','<<scores[2]<<','<<elapsed<<'\n';csv.flush();
                std::printf("PROGRESS seed=%u envs=%d update=%d calm=%.4f current_px=%.4f current_diagonal=%.4f elapsed=%.1f\n",seed,environments,trainer.update,scores[0],scores[1],scores[2],elapsed);std::fflush(stdout);
            }
            if(completed<updates) trainer.train_update();
        }
        cudarl::checkpoint::write_atomic(checkpoint,trainer.snapshot());trainer.policy().save((std::filesystem::path(output)/"latest.bin").string());
        if(trainer.motor.get_params()!=baseline.motor || original!=cudarl::checkpoint::read(input)) throw std::runtime_error("Original baseline or motor changed");
        std::printf("SAVED update=%d additional=%d motor_unchanged=true source_unchanged=true\n",trainer.update,trainer.update-start_update);return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Progress training: %s\n",e.what());return 1;}
}
