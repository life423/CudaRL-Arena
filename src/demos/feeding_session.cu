#include "feeding_session.h"
#include "pond_playback.h"
#include "envs/swimmer/feeding_batch.h"
#include "rl/swimmer_ppo.h"
#include "rl/progress_selector_policy.h"
#include "rl/feeding_policy.h"
#include "nn/mlp_cpu.h"
#include <atomic>
#include <mutex>
#include <thread>
#include <cstdio>
namespace cudarl::demos {
namespace {
namespace fd=feeding;
struct FeedingTask {
    using State=fd::State;using Batch=fd::Batch;
    static constexpr int inputs=fd::kObs,actions=4,evaluation_steps=200;
    static constexpr float entropy=.002f;
    static int collected(const State& state) {return state.eaten;}
};
using Trainer=rl::BasicSwimmerTrainer<FeedingTask>;
struct Evaluation {int foods=0,calm=0,feeding_worlds=0,calm_worlds=0;double progress=0;bool valid=true;};
Evaluation evaluate(const std::vector<float>& actor,const std::vector<float>& selector,const fd::Context& context,int count,int steps,uint32_t seed) {
    fd::Batch worlds(count,seed);rl::SwimmerStream stream;worlds.configure(selector,context,stream.value,true);
    nn::GpuMlp motor(rl::swimmer_specification(4,fd::kObs),count,7);motor.set_params(actor);
    for(int tick=0;tick<steps;++tick) {motor.forward(worlds.obs(),count,stream.value);worlds.step(motor.output(),{},stream.value,false);}
    std::vector<fd::State> states(static_cast<size_t>(count));CUDA_CHECK_THROW(cudaMemcpyAsync(states.data(),worlds.states(),states.size()*sizeof(fd::State),cudaMemcpyDeviceToHost,stream.value));CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
    Evaluation result;
    for(const auto& s:states) {
        result.valid=result.valid && !s.invalid && s.max_still<100;
        if(s.retention) {result.calm+=s.eaten;++result.calm_worlds;}
        else {
            result.foods+=s.eaten;++result.feeding_worlds;double nearest=5;
            for(const auto& p:s.pellets) if(p.active) nearest=std::min(nearest,static_cast<double>(deep::norm(p.position-s.body.position)));
            result.progress+=s.eaten*10.-nearest;
        }
    }
    return result;
}
double rate(const Evaluation& e,int steps) {return e.feeding_worlds?e.foods*60./(e.feeding_worlds*steps*static_cast<double>(deep::dt)):0;}
class Session final:public FeedingSession {
    rl::ProgressPolicy source_;std::string path_;int worlds_;
    std::unique_ptr<checkpoint::Lease> lease_;
    rl::SwimmerStream live_stream_;fd::Batch live_;nn::GpuMlp motor_;
    fd::State frame_;fd::Context context_;
    mutable std::mutex mutex_;FeedingStats stats_;std::vector<float> adopted_,initial_,pending_;
    uint64_t publication_=0,applied_=0;
    std::vector<uint8_t> restored_;
    std::atomic<bool> stopping_{false},save_{false},paused_{false};std::thread worker_;
    void save(Trainer& trainer) {
        if(path_.empty()) return;
        try {
            checkpoint::Writer training;
            training.value(trainer.update_id);training.array(trainer.actor.get_params());training.array(trainer.critic.get_params());
            std::vector<float> deviation(4);trainer.log_std.copy_to_host(deviation.data(),deviation.size());training.array(deviation);
            trainer.actor_opt.save(training);trainer.critic_opt.save(training);trainer.std_opt.save(training);trainer.env.save(training);
            checkpoint::Writer out;out.value(uint32_t{0x46454544});out.value(uint32_t{2});out.value(static_cast<uint32_t>(sizeof(fd::State)));out.value(worlds_);source_.write(out);
            {std::lock_guard<std::mutex> lock(mutex_);out.value(frame_);out.value(context_);out.array(adopted_);out.value(stats_.adopted_update);}
            out.array(training.data);checkpoint::write_atomic(path_,out.data);
            std::lock_guard<std::mutex> lock(mutex_);stats_.saved_update=trainer.update_id;stats_.save_error.clear();
        } catch(const std::exception& e) {std::lock_guard<std::mutex> lock(mutex_);stats_.save_error=e.what();}
    }
    void run() {
        try {
            Trainer trainer(worlds_,71);trainer.actor.set_params(initial_);trainer.actor_opt.set_learning_rate(.0001f);trainer.critic_opt.set_learning_rate(.0003f);trainer.std_opt.set_learning_rate(.0001f);
            const std::vector<float> deviation(4,-1.5f);trainer.log_std.copy_from_host(deviation.data(),deviation.size());
            fd::Context practice;
            {std::lock_guard<std::mutex> lock(mutex_);practice=context_;}
            trainer.env.configure(source_.actor,practice,trainer.stream.value,true);
            if(!restored_.empty()) {
                checkpoint::Reader in{restored_};trainer.update_id=in.value<int>();trainer.actor.set_params(in.array<float>(trainer.actor.num_params()));trainer.critic.set_params(in.array<float>(trainer.critic.num_params()));
                const auto stds=in.array<float>(4);if(stds.size()!=4 || trainer.update_id<0) throw std::runtime_error("Invalid feeding learner save");trainer.log_std.copy_from_host(stds.data(),stds.size());
                trainer.actor_opt.load(in);trainer.critic_opt.load(in);trainer.std_opt.load(in);trainer.env.load(in);in.finish();
                practice=trainer.env.context();
                if(trainer.actor_opt.steps()!=trainer.update_id*rl::epochs*rl::minibatches || trainer.critic_opt.steps()!=trainer.actor_opt.steps() || trainer.std_opt.steps()!=trainer.actor_opt.steps()) throw std::runtime_error("Inconsistent feeding optimizer counters");
                restored_.clear();
            }
            // Brief starting reference only: eight worlds for ten simulated seconds.
            const auto baseline=evaluate(initial_,source_.actor,practice,8,200,0xFEEDC101u);
            {
                std::lock_guard<std::mutex> lock(mutex_);stats_.ready=true;stats_.training=true;stats_.worlds=worlds_;stats_.update=trainer.update_id;
                stats_.original_score=rate(baseline,200);stats_.adopted_score=stats_.original_score;
            }
            std::printf("FEEDING_BRIEF_REFERENCE worlds=8 seconds=10 food_min=%.3f valid=%d\n",rate(baseline,200),baseline.valid?1:0);std::fflush(stdout);
            auto last_save=rl::Clock::now();
            while(!stopping_.load()) {
                if(save_.exchange(false) || rl::Clock::now()-last_save>=std::chrono::seconds(30)) {save(trainer);last_save=rl::Clock::now();}
                if(paused_.load()) {std::this_thread::sleep_for(std::chrono::milliseconds(10));continue;}
                const auto start=rl::Clock::now();fd::Context next;
                {std::lock_guard<std::mutex> lock(mutex_);next=context_;}
                if(next.version!=practice.version) {practice=next;trainer.env.configure(source_.actor,practice,trainer.stream.value,true);}
                const auto update=trainer.update();
                const auto learned=trainer.actor_params();
                for(float weight:learned) if(!std::isfinite(weight)) throw std::runtime_error("Non-finite learner weights; original controller preserved");
                const bool changed=learned!=initial_;
                {std::lock_guard<std::mutex> lock(mutex_);stats_.update=update.update;stats_.update_ms=update.update_ms;stats_.trained_context=practice.version;stats_.learner_changed=changed;}
                if(update.update%16==0) {
                    const auto candidate=trainer.actor_params();std::vector<float> incumbent;
                    {std::lock_guard<std::mutex> lock(mutex_);incumbent=adopted_;}
                    const uint32_t seed=0xFEEDA501u+static_cast<uint32_t>(update.update);
                    const auto original=evaluate(initial_,source_.actor,practice,16,600,seed);
                    const auto current=evaluate(incumbent,source_.actor,practice,16,600,seed);
                    const auto trial=evaluate(candidate,source_.actor,practice,16,600,seed);
                    const bool better=trial.valid && trial.foods>=current.foods && trial.progress>current.progress+.02 && trial.calm>=original.calm;
                    std::printf("FEEDING_EVAL update=%d context=%d seed=%u original=%.3f incumbent=%.3f candidate=%.3f progress_gain=%.5f calm=%d original_calm=%d accepted=%d\n",update.update,practice.version,seed,rate(original,600),rate(current,600),rate(trial,600),trial.progress-current.progress,trial.calm,original.calm,better?1:0);std::fflush(stdout);
                    std::lock_guard<std::mutex> lock(mutex_);++stats_.evaluations;stats_.original_score=rate(original,600);stats_.candidate_score=rate(trial,600);stats_.adopted_score=rate(current,600);
                    if(better) {adopted_=candidate;pending_=candidate;++publication_;stats_.adopted_update=update.update;stats_.adopted_score=rate(trial,600);}
                }
                if(rl::Clock::now()-last_save>=std::chrono::seconds(30)) {save(trainer);last_save=rl::Clock::now();}
                while(!stopping_.load() && rl::Clock::now()-start<std::chrono::milliseconds(500)) std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            save(trainer);std::lock_guard<std::mutex> lock(mutex_);stats_.training=false;
        } catch(const std::exception& e) {std::lock_guard<std::mutex> lock(mutex_);stats_.error=e.what();stats_.training=false;}
    }
public:
    Session(rl::ProgressPolicy source,std::string path,int worlds):source_(std::move(source)),path_(std::move(path)),worlds_(worlds),live_(1,7),motor_(rl::swimmer_specification(4,fd::kObs),1,7) {
        if(worlds<16 || worlds>2048 || worlds%8) throw std::invalid_argument("Invalid feeding population");
        initial_=rl::feeding_motor_parameters(source_.baseline.motor);adopted_=initial_;frame_=fd::initial(7,0,0,context_,false);
        if(!path_.empty()) {
            lease_=std::make_unique<checkpoint::Lease>(path_);
            if(checkpoint::exists(path_)) {
                const auto bytes=checkpoint::read(path_);checkpoint::Reader in{bytes};
                if(in.value<uint32_t>()!=0x46454544u || in.value<uint32_t>()!=2 || in.value<uint32_t>()!=sizeof(fd::State) || in.value<int>()!=worlds_) throw std::runtime_error("Incompatible feeding save; original saves untouched");
                const auto saved=rl::ProgressPolicy::read(in);
                if(saved.actor!=source_.actor || saved.baseline.motor!=source_.baseline.motor) throw std::runtime_error("Feeding source mismatch");
                frame_=in.value<fd::State>();context_=in.value<fd::Context>();adopted_=in.array<float>(motor_.num_params());stats_.adopted_update=in.value<int>();restored_=in.array<uint8_t>(128000000);in.finish();
                if(adopted_.size()!=initial_.size() || context_.count<0 || context_.count>32 || frame_.target>=fd::capacity) throw std::runtime_error("Invalid feeding snapshot");
                checkpoint::Reader saved_update{restored_};stats_.loaded_update=saved_update.value<int>();stats_.saved_update=stats_.loaded_update;stats_.resumed=true;stats_.applied_update=stats_.adopted_update;
            }
        }
        motor_.set_params(adopted_);live_.configure(source_.actor,context_,live_stream_.value);live_.set_state(frame_,live_stream_.value);
        worker_=std::thread([this] {run();});
    }
    ~Session() override {stop();}
    void stop() override {stopping_.store(true);if(worker_.joinable()) worker_.join();}
    const fd::State& state() const override {return frame_;}
    fd::State step() override {
        std::vector<float> next;int accepted=0;
        {std::lock_guard<std::mutex> lock(mutex_);if(publication_!=applied_) {next=pending_;applied_=publication_;accepted=stats_.adopted_update;}}
        if(!next.empty()) {motor_.set_params(next);std::lock_guard<std::mutex> lock(mutex_);stats_.applied_update=accepted;}
        motor_.forward(live_.obs(),1,live_stream_.value);live_.step(motor_.output(),{},live_stream_.value,false);
        const auto next_frame=live_.state(live_stream_.value);{std::lock_guard<std::mutex> lock(mutex_);frame_=next_frame;}return next_frame;
    }
    int drop(deep::V position) override {
        fd::State updated=frame_;const int slot=fd::drop(updated,position);if(slot<0) return -1;
        fd::Context context;
        {std::lock_guard<std::mutex> lock(mutex_);frame_=updated;
            if(context_.count<32) context_.drops[context_.count++]=position;
            else {for(int i=1;i<32;++i) context_.drops[i-1]=context_.drops[i];context_.drops[31]=position;}
            ++context_.version;context=context_;
        }
        live_.configure(source_.actor,context,live_stream_.value);live_.set_state(updated,live_stream_.value);return slot;
    }
    void set_current(deep::V current) override {
        validate_pond_current(current);if(deep::norm(current)>.0075001f) throw std::invalid_argument("Feeding current exceeds 0.75 cm/s");fd::Context context;
        {std::lock_guard<std::mutex> lock(mutex_);context_.current=current;++context_.version;context=context_;}
        live_.configure(source_.actor,context,live_stream_.value);
    }
    deep::V current() const override {std::lock_guard<std::mutex> lock(mutex_);return context_.current;}
    FeedingStats stats() const override {std::lock_guard<std::mutex> lock(mutex_);auto out=stats_;out.context_version=context_.version;out.paused=paused_.load();return out;}
    void request_save() override {save_.store(true);}
    void set_paused(bool paused) override {paused_.store(paused);}
};
}
std::unique_ptr<FeedingSession> create_feeding_session(const std::string& policy,const std::string& checkpoint,int worlds) {
    return std::make_unique<Session>(rl::ProgressPolicy::load(policy),checkpoint,worlds);
}
}
