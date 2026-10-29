#include "feeding_session.h"
#include "pond_playback.h"
#include "rl/feeding_cpu.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>

namespace cudarl::demos {
namespace {
namespace fd=feeding;
using Clock=std::chrono::steady_clock;
struct Evaluation {int foods=0,calm=0,feeding_worlds=0;double progress=0;bool valid=true;};
Evaluation evaluate_cpu(const std::vector<float>& weights,const rl::ProgressPolicy& source,const fd::Context& context,int count,int steps,uint32_t seed) {
    fd::CpuBatch worlds(count,seed,source.actor,context);
    nn::CpuMlp<float> actor(rl::swimmer_specification(4,fd::kObs));actor.params()=weights;
    for(int tick=0;tick<steps;++tick) {actor.forward(worlds.observations.data(),count);worlds.step(actor.output(),nullptr,nullptr,nullptr,false,true);}
    Evaluation result;
    for(const auto& s:worlds.states) {
        result.valid=result.valid && !s.invalid && s.max_still<100;
        if(s.retention) result.calm+=s.eaten;
        else {result.foods+=s.eaten;++result.feeding_worlds;double nearest=5;
            for(const auto& p:s.pellets) if(p.active) nearest=std::min(nearest,static_cast<double>(deep::norm(p.position-s.body.position)));
            result.progress+=s.eaten*10.-nearest;}
    }
    return result;
}
double rate(const Evaluation& e,int steps) {return e.feeding_worlds?e.foods*60./(e.feeding_worlds*steps*static_cast<double>(deep::dt)):0;}
class CpuSession final:public FeedingSession {
    rl::ProgressPolicy source_;std::string path_;int worlds_;
    std::unique_ptr<checkpoint::Lease> lease_;
    fd::CpuBatch live_;nn::CpuMlp<float> motor_{rl::swimmer_specification(4,fd::kObs)};
    mutable std::mutex mutex_;fd::Context context_;FeedingStats stats_;
    std::vector<float> initial_,adopted_,pending_;std::vector<uint8_t> restored_;
    uint64_t publication_=0,applied_=0;
    std::atomic<bool> stopping_{false},save_{false},paused_{false};std::thread worker_;
    void save(rl::CpuFeedingTrainer& trainer) {
        if(path_.empty()) return;
        try {
            checkpoint::Writer training;trainer.save(training);
            checkpoint::Writer out;out.value(uint32_t{0x46435055});out.value(uint32_t{1});out.value(static_cast<uint32_t>(sizeof(fd::State)));out.value(worlds_);source_.write(out);
            {std::lock_guard<std::mutex> lock(mutex_);out.value(live_.states[0]);out.value(context_);out.array(adopted_);out.value(stats_.adopted_update);}
            out.array(training.data);checkpoint::write_atomic(path_,out.data);
            std::lock_guard<std::mutex> lock(mutex_);stats_.saved_update=trainer.update_id;stats_.save_error.clear();
        } catch(const std::exception& e) {std::lock_guard<std::mutex> lock(mutex_);stats_.save_error=e.what();}
    }
    void run() {
        try {
            fd::Context practice;{std::lock_guard<std::mutex> lock(mutex_);practice=context_;}
            rl::CpuFeedingTrainer trainer(worlds_,source_,practice);
            if(!restored_.empty()) {checkpoint::Reader in{restored_};trainer.load(in);in.finish();practice=trainer.env.context;restored_.clear();}
            const auto baseline=evaluate_cpu(initial_,source_,practice,8,200,0xFEEDC101u);
            {std::lock_guard<std::mutex> lock(mutex_);stats_.ready=true;stats_.training=true;stats_.worlds=worlds_;stats_.update=trainer.update_id;stats_.original_score=rate(baseline,200);stats_.adopted_score=stats_.original_score;}
            std::printf("FEEDING_CPU_REFERENCE worlds=8 seconds=10 food_min=%.3f valid=%d\n",rate(baseline,200),baseline.valid?1:0);std::fflush(stdout);
            auto last_save=Clock::now();
            while(!stopping_.load()) {
                if(save_.exchange(false) || Clock::now()-last_save>=std::chrono::seconds(30)) {save(trainer);last_save=Clock::now();}
                if(paused_.load()) {std::this_thread::sleep_for(std::chrono::milliseconds(10));continue;}
                const auto start=Clock::now();fd::Context next;
                {std::lock_guard<std::mutex> lock(mutex_);next=context_;}
                // Apply interaction context at the rollout boundary without restarting world schedules.
                if(next.version!=practice.version) {practice=next;trainer.env.context=practice;trainer.env.prepare();}
                trainer.update();
                for(float p:trainer.actor.params()) if(!std::isfinite(p)) throw std::runtime_error("Non-finite CPU learner; original controller preserved");
                for(float p:trainer.critic.params()) if(!std::isfinite(p)) throw std::runtime_error("Non-finite CPU value estimator");
                for(float p:trainer.log_std) if(!std::isfinite(p)) throw std::runtime_error("Non-finite CPU exploration");
                const double milliseconds=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                {std::lock_guard<std::mutex> lock(mutex_);stats_.update=trainer.update_id;stats_.update_ms=milliseconds;stats_.trained_context=practice.version;stats_.learner_changed=trainer.actor.params()!=initial_;}
                if(trainer.update_id%16==0 && !stopping_.load()) {
                    std::vector<float> incumbent;{std::lock_guard<std::mutex> lock(mutex_);incumbent=adopted_;}
                    const uint32_t seed=0xFEEDA501u+static_cast<uint32_t>(trainer.update_id);
                    const auto original=evaluate_cpu(initial_,source_,practice,16,600,seed);
                    const auto current=evaluate_cpu(incumbent,source_,practice,16,600,seed);
                    const auto candidate=evaluate_cpu(trainer.actor.params(),source_,practice,16,600,seed);
                    const bool better=candidate.valid && candidate.foods>=current.foods && candidate.progress>current.progress+.02 && candidate.calm>=original.calm;
                    std::printf("FEEDING_CPU_EVAL update=%d original=%.3f incumbent=%.3f candidate=%.3f accepted=%d\n",trainer.update_id,rate(original,600),rate(current,600),rate(candidate,600),better?1:0);std::fflush(stdout);
                    std::lock_guard<std::mutex> lock(mutex_);++stats_.evaluations;stats_.original_score=rate(original,600);stats_.candidate_score=rate(candidate,600);stats_.adopted_score=rate(current,600);
                    if(better) {adopted_=trainer.actor.params();pending_=adopted_;++publication_;stats_.adopted_update=trainer.update_id;stats_.adopted_score=rate(candidate,600);}
                }
                while(!stopping_.load() && Clock::now()-start<std::chrono::milliseconds(500)) std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            save(trainer);std::lock_guard<std::mutex> lock(mutex_);stats_.training=false;
        } catch(const std::exception& e) {std::lock_guard<std::mutex> lock(mutex_);stats_.error=e.what();stats_.training=false;}
    }
public:
    CpuSession(rl::ProgressPolicy source,std::string path,int worlds):source_(std::move(source)),path_(std::move(path)),worlds_(worlds),live_(1,7,source_.actor) {
        if(worlds<8 || worlds>128 || worlds%8) throw std::invalid_argument("Invalid CPU feeding population");
        stats_.backend="CPU";initial_=rl::feeding_motor_parameters(source_.baseline.motor);adopted_=initial_;
        live_.states[0]=fd::initial(7,0,0,context_,false);
        if(!path_.empty()) {
            lease_=std::make_unique<checkpoint::Lease>(path_);
            if(checkpoint::exists(path_)) {
                const auto bytes=checkpoint::read(path_);checkpoint::Reader in{bytes};
                if(in.value<uint32_t>()!=0x46435055u || in.value<uint32_t>()!=1 || in.value<uint32_t>()!=sizeof(fd::State) || in.value<int>()!=worlds_) throw std::runtime_error("Incompatible CPU feeding save; other saves preserved");
                const auto saved=rl::ProgressPolicy::read(in);
                if(saved.actor!=source_.actor || saved.baseline.motor!=source_.baseline.motor) throw std::runtime_error("CPU feeding source mismatch");
                live_.states[0]=in.value<fd::State>();context_=in.value<fd::Context>();adopted_=in.array<float>(initial_.size());stats_.adopted_update=in.value<int>();restored_=in.array<uint8_t>(128000000);in.finish();
                if(adopted_.size()!=initial_.size() || context_.count<0 || context_.count>32 || live_.states[0].target>=fd::capacity) throw std::runtime_error("Invalid CPU feeding snapshot");
                for(float p:adopted_) if(!std::isfinite(p)) throw std::runtime_error("Invalid CPU adopted controller");
                checkpoint::Reader saved_update{restored_};stats_.loaded_update=saved_update.value<int>();stats_.saved_update=stats_.loaded_update;stats_.resumed=true;stats_.applied_update=stats_.adopted_update;
            }
        }
        live_.context=context_;live_.prepare();motor_.params()=adopted_;
        worker_=std::thread([this] {run();});
    }
    ~CpuSession() override {stop();}
    void stop() override {stopping_.store(true);if(worker_.joinable()) worker_.join();}
    const fd::State& state() const override {return live_.states[0];}
    fd::State step() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if(publication_!=applied_) {motor_.params()=pending_;applied_=publication_;stats_.applied_update=stats_.adopted_update;}
        motor_.forward(live_.observations.data(),1);live_.step(motor_.output(),nullptr,nullptr,nullptr,false);return live_.states[0];
    }
    int drop(deep::V position) override {
        std::lock_guard<std::mutex> lock(mutex_);const int slot=fd::drop(live_.states[0],position);if(slot<0) return -1;
        if(context_.count<32) context_.drops[context_.count++]=position;
        else {for(int i=1;i<32;++i) context_.drops[i-1]=context_.drops[i];context_.drops[31]=position;}
        ++context_.version;live_.context=context_;live_.prepare();return slot;
    }
    void set_current(deep::V current) override {
        validate_pond_current(current);if(deep::norm(current)>.0075001f) throw std::invalid_argument("Feeding current exceeds 0.75 cm/s");
        std::lock_guard<std::mutex> lock(mutex_);context_.current=current;++context_.version;live_.context=context_;
    }
    deep::V current() const override {std::lock_guard<std::mutex> lock(mutex_);return context_.current;}
    FeedingStats stats() const override {std::lock_guard<std::mutex> lock(mutex_);auto out=stats_;out.context_version=context_.version;out.paused=paused_.load();return out;}
    void request_save() override {save_.store(true);}
    void set_paused(bool paused) override {paused_.store(paused);}
};
}
std::unique_ptr<FeedingSession> create_cpu_feeding_session(const std::string& policy,const std::string& checkpoint,int worlds) {
    return std::make_unique<CpuSession>(rl::ProgressPolicy::load(policy),checkpoint,worlds);
}
}
