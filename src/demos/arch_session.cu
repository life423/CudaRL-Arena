#include "feeding_session.h"
#include "envs/swimmer/arch_batch.h"
#include "rl/arch_policy.h"
#include "rl/swimmer_ppo.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <thread>

namespace cudarl::demos {
namespace {
namespace ar=arch;
struct ArchTask {
    using State=ar::State;using Batch=ar::Batch;
    static constexpr int inputs=ar::kObs,actions=ar::kAct,evaluation_steps=200;
    static constexpr float entropy=.002f;
    static int collected(const State& state) {return state.pond.eaten;}
};
using Trainer=rl::BasicSwimmerTrainer<ArchTask>;
struct Evaluation {
    int foods=0,calm=0,worlds=0,contacts=0,passages=0,clean=0;
    double progress=0;
    bool valid=true;
};
Evaluation evaluate(const std::vector<float>& actor,const std::vector<float>& selector,int count,int steps,uint32_t seed) {
    ar::Batch batch(count,seed);rl::SwimmerStream stream;batch.configure(selector,stream.value,true);
    nn::GpuMlp motor(rl::swimmer_specification(4,ar::kObs),count,7);motor.set_params(actor);
    for(int tick=0;tick<steps;++tick) {motor.forward(batch.obs(),count,stream.value);batch.step(motor.output(),{},stream.value,false);}
    std::vector<ar::State> states(static_cast<size_t>(count));
    CUDA_CHECK_THROW(cudaMemcpyAsync(states.data(),batch.states(),states.size()*sizeof(ar::State),cudaMemcpyDeviceToHost,stream.value));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
    Evaluation result;
    for(const auto& state:states) {
        const auto& pond=state.pond;
        result.valid=result.valid && !pond.invalid && pond.max_still<100;
        if(pond.retention) {result.calm+=pond.eaten;continue;}
        ++result.worlds;result.foods+=pond.eaten;result.contacts+=state.contacts;
        result.passages+=state.passages;result.clean+=state.clean_passages;
        double nearest=10;
        for(const auto& food:pond.pellets) if(food.active) nearest=std::min(nearest,static_cast<double>(deep::norm(food.position-pond.body.position)));
        result.progress+=pond.eaten*10.-nearest;
    }
    return result;
}
double rate(const Evaluation& result,int steps) {
    return result.worlds?result.foods*60./(result.worlds*steps*static_cast<double>(deep::dt)):0;
}
bool finite_frame(const ar::State& state) {
    const auto& pond=state.pond;
    if(pond.steps<0 || pond.eaten<0 || pond.target<-1 || pond.target>=feeding::capacity ||
       state.contacts<0 || state.passages<0 || state.clean_passages<0 || state.clean_passages>state.passages ||
       !std::isfinite(deep::norm(pond.body.position)) || deep::norm(pond.body.position)>100) return false;
    for(float velocity:pond.body.velocity) if(!std::isfinite(velocity)) return false;
    for(float angle:pond.body.angle) if(!std::isfinite(angle)) return false;
    for(const auto& food:pond.pellets) if(food.active && (!std::isfinite(deep::norm(food.position)) || !std::isfinite(deep::norm(food.velocity)))) return false;
    const auto q=pond.body.orientation;
    const float norm=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
    return std::isfinite(norm) && norm>.99f && norm<1.01f;
}
class ArchSession final:public FeedingSession {
    static constexpr uint32_t magic=0x41524348u,version=1;
    rl::ArchSource source_;
    std::string path_;
    int worlds_;
    std::unique_ptr<checkpoint::Lease> lease_;
    rl::SwimmerStream live_stream_;
    ar::Batch live_;
    nn::GpuMlp motor_;
    ar::State frame_;
    mutable std::mutex mutex_;
    FeedingStats stats_;
    std::vector<float> initial_,adopted_,live_actor_,pending_;
    std::vector<uint8_t> restored_;
    uint64_t publication_=0,applied_=0;
    std::atomic<bool> stopping_{false},save_{false},paused_{false};
    std::thread worker_;

    void save(Trainer& trainer) {
        if(path_.empty()) return;
        try {
            checkpoint::Writer training;training.value(trainer.update_id);
            training.array(trainer.actor.get_params());training.array(trainer.critic.get_params());
            std::vector<float> deviation(4);trainer.log_std.copy_to_host(deviation.data(),deviation.size());training.array(deviation);
            trainer.actor_opt.save(training);trainer.critic_opt.save(training);trainer.std_opt.save(training);trainer.env.save(training);
            checkpoint::Writer out;out.value(magic);out.value(version);out.value(static_cast<uint32_t>(sizeof(ar::State)));out.value(worlds_);source_.write(out);
            {std::lock_guard<std::mutex> lock(mutex_);out.value(frame_);out.array(adopted_);out.value(stats_.adopted_update);out.array(live_actor_);out.value(stats_.applied_update);}
            out.array(training.data);checkpoint::write_atomic(path_,out.data);
            std::lock_guard<std::mutex> lock(mutex_);stats_.saved_update=trainer.update_id;stats_.save_error.clear();
        } catch(const std::exception& error) {std::lock_guard<std::mutex> lock(mutex_);stats_.save_error=error.what();}
    }
    void run() {
        try {
            Trainer trainer(worlds_,71);trainer.actor.set_params(initial_);
            trainer.actor_opt.set_learning_rate(.0001f);trainer.critic_opt.set_learning_rate(.0003f);trainer.std_opt.set_learning_rate(.0001f);
            const std::vector<float> deviation(4,-1.5f);trainer.log_std.copy_from_host(deviation.data(),deviation.size());
            trainer.env.configure(source_.policy.actor,trainer.stream.value,true);
            if(!restored_.empty()) {
                checkpoint::Reader in{restored_};trainer.update_id=in.value<int>();
                const auto actor=in.array<float>(trainer.actor.num_params()),critic=in.array<float>(trainer.critic.num_params()),stds=in.array<float>(4);
                if(actor.size()!=static_cast<size_t>(trainer.actor.num_params()) || critic.size()!=static_cast<size_t>(trainer.critic.num_params()) || stds.size()!=4 || trainer.update_id<0)
                    throw std::runtime_error("Invalid arch learner dimensions");
                for(float x:actor) if(!std::isfinite(x)) throw std::runtime_error("Non-finite saved arch actor");
                for(float x:critic) if(!std::isfinite(x)) throw std::runtime_error("Non-finite saved arch critic");
                for(float x:stds) if(!std::isfinite(x)) throw std::runtime_error("Non-finite saved arch deviation");
                trainer.actor.set_params(actor);trainer.critic.set_params(critic);trainer.log_std.copy_from_host(stds.data(),stds.size());
                trainer.actor_opt.load(in);trainer.critic_opt.load(in);trainer.std_opt.load(in);trainer.env.load(in);in.finish();
                if(trainer.actor_opt.steps()!=trainer.update_id*rl::epochs*rl::minibatches || trainer.critic_opt.steps()!=trainer.actor_opt.steps() || trainer.std_opt.steps()!=trainer.actor_opt.steps())
                    throw std::runtime_error("Inconsistent arch optimizer counters");
                restored_.clear();
            }
            // The first reference is intentionally brief. It is not an obstacle
            // mastery claim or a substitute for fresh final evaluation.
            const auto baseline=evaluate(initial_,source_.policy.actor,8,200,0xA2C4C101u);
            const auto approved=adopted_==initial_?baseline:evaluate(adopted_,source_.policy.actor,8,200,0xA2C4C101u);
            {
                std::lock_guard<std::mutex> lock(mutex_);stats_.ready=true;stats_.training=true;stats_.worlds=worlds_;stats_.update=trainer.update_id;
                stats_.original_score=rate(baseline,200);stats_.adopted_score=rate(approved,200);
            }
            std::printf("ARCH_BRIEF_REFERENCE parent=%d fingerprint=%llu worlds=8 seconds=10 food_min=%.3f passages=%d contacts=%d valid=%d\n",source_.parent_update,
                static_cast<unsigned long long>(source_.fingerprint),rate(baseline,200),baseline.passages,baseline.contacts,baseline.valid?1:0);std::fflush(stdout);
            auto last_save=rl::Clock::now();
            while(!stopping_.load()) {
                if(save_.exchange(false) || rl::Clock::now()-last_save>=std::chrono::seconds(30)) {save(trainer);last_save=rl::Clock::now();}
                if(paused_.load()) {std::this_thread::sleep_for(std::chrono::milliseconds(10));continue;}
                const auto start=rl::Clock::now();const auto update=trainer.update();
                const auto learned=trainer.actor_params();
                for(float weight:learned) if(!std::isfinite(weight)) throw std::runtime_error("Non-finite arch actor; inherited controller preserved");
                {std::lock_guard<std::mutex> lock(mutex_);stats_.update=update.update;stats_.update_ms=update.update_ms;stats_.learner_changed=learned!=initial_;}
                if(update.update%16==0) {
                    std::vector<float> incumbent;
                    {std::lock_guard<std::mutex> lock(mutex_);incumbent=adopted_;}
                    const uint32_t seed=0xA2C4A501u+static_cast<uint32_t>(update.update);
                    const auto original=evaluate(initial_,source_.policy.actor,16,600,seed);
                    const auto current=evaluate(incumbent,source_.policy.actor,16,600,seed);
                    const auto trial=evaluate(learned,source_.policy.actor,16,600,seed);
                    const bool retained=trial.calm>=std::max(original.calm,current.calm);
                    const bool improved=trial.progress>current.progress+.02 || trial.clean>current.clean;
                    const bool better=trial.valid && retained && trial.foods>=current.foods && trial.contacts<=current.contacts && improved;
                    std::printf("ARCH_EVAL update=%d seed=%u original=%.3f incumbent=%.3f candidate=%.3f calm=%d reference_calm=%d contacts=%d/%d passages=%d clean=%d accepted=%d\n",
                        update.update,seed,rate(original,600),rate(current,600),rate(trial,600),trial.calm,std::max(original.calm,current.calm),trial.contacts,current.contacts,trial.passages,trial.clean,better?1:0);std::fflush(stdout);
                    std::lock_guard<std::mutex> lock(mutex_);++stats_.evaluations;
                    stats_.original_score=rate(original,600);stats_.candidate_score=rate(trial,600);stats_.adopted_score=rate(current,600);
                    if(better) {adopted_=learned;pending_=learned;++publication_;stats_.adopted_update=update.update;stats_.adopted_score=rate(trial,600);}
                }
                while(!stopping_.load() && rl::Clock::now()-start<std::chrono::milliseconds(500)) std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            save(trainer);std::lock_guard<std::mutex> lock(mutex_);stats_.training=false;
        } catch(const std::exception& error) {std::lock_guard<std::mutex> lock(mutex_);stats_.error=error.what();stats_.training=false;}
    }
public:
    ArchSession(const std::string& source_path,std::string path,int worlds):path_(std::move(path)),worlds_(worlds),live_(1,7),motor_(rl::swimmer_specification(4,ar::kObs),1,7) {
        if(worlds<16 || worlds>2048 || worlds%8) throw std::invalid_argument("Invalid arch population");
        if(!path_.empty() && std::filesystem::weakly_canonical(path_)==std::filesystem::weakly_canonical(source_path))
            throw std::invalid_argument("Arch checkpoint must be separate from its inherited feeding save");
        stats_.arch_mode=true;stats_.input_count=ar::kObs;
        if(!path_.empty()) lease_=std::make_unique<checkpoint::Lease>(path_);
        if(!path_.empty() && checkpoint::exists(path_)) {
            const auto bytes=checkpoint::read(path_);checkpoint::Reader in{bytes};
            if(in.value<uint32_t>()!=magic || in.value<uint32_t>()!=version || in.value<uint32_t>()!=sizeof(ar::State) || in.value<int>()!=worlds_)
                throw std::runtime_error("Incompatible arch save; original feeding save untouched");
            source_=rl::ArchSource::read(in);frame_=in.value<ar::State>();adopted_=in.array<float>(motor_.num_params());stats_.adopted_update=in.value<int>();
            live_actor_=in.array<float>(motor_.num_params());stats_.applied_update=in.value<int>();restored_=in.array<uint8_t>(128000000);in.finish();
            if(!finite_frame(frame_) || adopted_.size()!=static_cast<size_t>(motor_.num_params()) || live_actor_.size()!=adopted_.size() ||
               stats_.adopted_update<0 || stats_.applied_update<0 || stats_.applied_update>stats_.adopted_update || restored_.size()<sizeof(int))
                throw std::runtime_error("Invalid arch live snapshot");
            for(float weight:adopted_) if(!std::isfinite(weight)) throw std::runtime_error("Non-finite adopted arch snapshot");
            for(float weight:live_actor_) if(!std::isfinite(weight)) throw std::runtime_error("Non-finite live arch snapshot");
            if(stats_.applied_update==stats_.adopted_update && live_actor_!=adopted_) throw std::runtime_error("Inconsistent live arch actor");
            if(stats_.applied_update!=stats_.adopted_update) {pending_=adopted_;publication_=1;}
            checkpoint::Reader training{restored_};stats_.loaded_update=training.value<int>();
            if(stats_.loaded_update<stats_.adopted_update) throw std::runtime_error("Invalid arch publication counter");
            stats_.saved_update=stats_.loaded_update;stats_.resumed=true;
        } else {
            source_=rl::load_arch_source(source_path);frame_=ar::initial(7,0,0,false);adopted_=rl::arch_motor_parameters(source_.motor);live_actor_=adopted_;
        }
        initial_=rl::arch_motor_parameters(source_.motor);stats_.parent_update=source_.parent_update;stats_.parent_selector_update=source_.policy.update;
        motor_.set_params(live_actor_);live_.configure(source_.policy.actor,live_stream_.value);live_.set_state(frame_,live_stream_.value);
        worker_=std::thread([this] {run();});
    }
    ~ArchSession() override {stop();}
    void stop() override {stopping_.store(true);if(worker_.joinable()) worker_.join();}
    const feeding::State& state() const override {return frame_.pond;}
    feeding::State step() override {
        std::vector<float> next;int accepted=0;
        {std::lock_guard<std::mutex> lock(mutex_);if(publication_!=applied_) {next=pending_;applied_=publication_;accepted=stats_.adopted_update;}}
        if(!next.empty()) {motor_.set_params(next);std::lock_guard<std::mutex> lock(mutex_);live_actor_=next;stats_.applied_update=accepted;}
        motor_.forward(live_.obs(),1,live_stream_.value);live_.step(motor_.output(),{},live_stream_.value,false);
        const auto next_frame=live_.state(live_stream_.value);
        {std::lock_guard<std::mutex> lock(mutex_);frame_=next_frame;}
        return next_frame.pond;
    }
    int drop(deep::V) override {return -1;} // Controlled automatic-food experiment.
    void set_current(deep::V current) override {
        if(!std::isfinite(deep::norm(current)) || deep::norm(current)>0) throw std::invalid_argument("First arch experiment uses calm water");
    }
    deep::V current() const override {return {};}
    FeedingStats stats() const override {
        std::lock_guard<std::mutex> lock(mutex_);auto out=stats_;out.paused=paused_.load();
        out.arch_contacts=frame_.contacts;out.arch_passages=frame_.passages;out.arch_clean_passages=frame_.clean_passages;return out;
    }
    void request_save() override {save_.store(true);}
    void set_paused(bool paused) override {paused_.store(paused);}
};
}
std::unique_ptr<FeedingSession> create_arch_session(const std::string& source_full_save,const std::string& checkpoint,int worlds) {
    return std::make_unique<ArchSession>(source_full_save,checkpoint,worlds);
}
}
