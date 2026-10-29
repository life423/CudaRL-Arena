#include "feeding_session.h"
#include "rl/shelter_motor_run.h"
#include "rl/shelter_hold_cases.h"
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <mutex>
#include <thread>

namespace cudarl::demos {
namespace {
namespace sm=shelter_motor;
constexpr uint32_t hold_magic=0x484C4431u,hold_version=1;
struct Proof {
    int holds=0,finite=0,cases=0;
    double exposure=0,food=0,old_food=0;
    bool passed() const {return cases==24 && finite==cases && holds>=18 && food>=old_food*.9;}
};
Proof prove(const std::vector<float>& actor,const rl::ShelterMotorRun& source,uint32_t seed) {
    const auto cases=rl::shelter_hold_cases(source.source.frame,seed);
    const auto results=rl::probe_shelter_motor(actor,cases,1,200,source.scales);
    Proof out;out.cases=static_cast<int>(results.size());
    for(const auto& result:results) {
        const auto& final=result.final_state.pond;
        bool finite=result.finite && !final.invalid && std::isfinite(deep::norm(final.body.position)) && deep::norm(final.body.position)<100;
        for(float value:final.body.velocity) finite=finite && std::isfinite(value);
        for(float value:final.body.angle) finite=finite && std::isfinite(value);
        out.finite+=finite?1:0;out.holds+=result.longest_safe_dwell_ticks>=100?1:0;
        out.exposure+=result.mean_exposure;
    }
    if(out.cases) out.exposure/=out.cases;
    const uint32_t food_seed=seed^0xF00D0011u;
    out.food=rl::shelter_motor_food_rate(actor,source.source,food_seed,16,600,{},source.scales);
    out.old_food=rl::shelter_motor_food_rate(rl::transfer_shelter_motor(source.source.motor),source.source,food_seed);
    return out;
}
void report_proof(const char* label,int update,const Proof& proof,bool accepted) {
    std::printf("HOLD_%s update=%d five_second_holds=%d/%d finite=%d exposure=%.6f calm_food_min=%.3f original_food_min=%.3f accepted=%d\n",
        label,update,proof.holds,proof.cases,proof.finite,proof.exposure,proof.food,proof.old_food,accepted?1:0);std::fflush(stdout);
}
bool finite_frame(const sm::State& frame) {
    const auto& p=frame.state.pond;const auto& body=p.body;
    if(p.invalid || p.steps<0 || p.eaten<0 || !std::isfinite(body.work) || !std::isfinite(body.travel) ||
       !std::isfinite(deep::norm(p.anchor)) || !std::isfinite(deep::norm(body.position)) || deep::norm(body.position)>100 ||
       !frame.request.cover || !frame.request.goal.valid || frame.request.goal.cover!=0 || frame.request.goal.food!=-1 ||
       !std::isfinite(deep::norm(frame.request.goal.position)) || !std::isfinite(deep::norm(frame.request.eye)) ||
       fabsf(deep::norm(frame.request.heading)-1.f)>1e-4f) return false;
    const auto& q=body.orientation;const float norm=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
    if(!std::isfinite(norm) || norm<.99f || norm>1.01f) return false;
    for(float value:body.velocity) if(!std::isfinite(value)) return false;
    for(float value:body.angle) if(!std::isfinite(value)) return false;
    for(float value:p.last_action) if(!std::isfinite(value)) return false;
    for(const auto& pellet:p.pellets) if(pellet.active &&
       (!std::isfinite(deep::norm(pellet.position)) || !std::isfinite(deep::norm(pellet.velocity)))) return false;
    return true;
}
class HoldingSession final:public FeedingSession {
    std::string path_;int worlds_;
    std::unique_ptr<checkpoint::Lease> lease_;
    rl::ShelterMotorRun source_;
    std::vector<uint8_t> source_payload_,restored_;
    std::vector<float> initial_,adopted_,live_actor_,pending_;
    int pending_update_=0;
    rl::SwimmerStream live_stream_;
    sm::Batch live_;
    nn::GpuMlp actor_;
    device_ptr<float> means_;
    sm::State frame_{};
    int held_ticks_=0,longest_held_ticks_=0;
    float exposure_=1,clearance_=0,distance_=0;
    mutable std::mutex mutex_;
    FeedingStats stats_;
    std::atomic<bool> stopping_{false},save_{false},paused_{false};
    std::thread worker_;

    void measure(const sm::State& frame,bool advance_time) {
        const float exposure=predator::exposure(frame.state.pond.body,frame.request.eye,frame.state.habitat,{2});
        const float clearance=predator::body_clearance(frame.state.pond.body,frame.state.habitat,{2});
        const float distance=deep::norm(frame.state.pond.body.position-frame.request.goal.position);
        if(!std::isfinite(exposure) || !std::isfinite(clearance) || !std::isfinite(distance))
            throw std::runtime_error("Holding practice produced a non-finite physical measurement");
        exposure_=exposure;clearance_=clearance;distance_=distance;
        if(advance_time) {
            if(exposure<=.05f && clearance>=.08f) {
                if(held_ticks_<std::numeric_limits<int>::max()) ++held_ticks_;
                longest_held_ticks_=std::max(longest_held_ticks_,held_ticks_);
            } else held_ticks_=0;
        }
    }
    void restore(const std::string& path) {
        const auto bytes=checkpoint::read(path);checkpoint::Reader in{bytes};
        if(in.value<uint32_t>()!=hold_magic || in.value<uint32_t>()!=hold_version ||
           in.value<uint32_t>()!=sizeof(sm::State) || in.value<int>()!=worlds_)
            throw std::runtime_error("Incompatible holding practice checkpoint");
        source_payload_=in.array<uint8_t>(256000000);source_=rl::read_shelter_motor_run(source_payload_);
        frame_=in.value<sm::State>();held_ticks_=in.value<int>();longest_held_ticks_=in.value<int>();
        const size_t count=static_cast<size_t>(actor_.num_params());
        adopted_=in.array<float>(count);stats_.adopted_update=in.value<int>();
        live_actor_=in.array<float>(count);stats_.applied_update=in.value<int>();
        restored_=in.array<uint8_t>(256000000);in.finish();
        if(source_.worlds!=worlds_ || !finite_frame(frame_) || held_ticks_<0 || longest_held_ticks_<held_ticks_ ||
           adopted_.size()!=count || live_actor_.size()!=count || stats_.adopted_update<source_.update ||
           stats_.applied_update<source_.update || stats_.applied_update>stats_.adopted_update)
            throw std::runtime_error("Invalid saved holding practice state");
        for(const auto* parameters:{&adopted_,&live_actor_}) for(float value:*parameters)
            if(!std::isfinite(value)) throw std::runtime_error("Non-finite saved holding actor");
        if(stats_.applied_update==stats_.adopted_update && live_actor_!=adopted_)
            throw std::runtime_error("Inconsistent holding actor publication");
        checkpoint::Reader learner{restored_};
        if(learner.value<uint32_t>()!=0x534D5452u || learner.value<uint32_t>()!=rl::shelter_motor_protocol ||
           learner.value<uint32_t>()!=sizeof(sm::State) || learner.value<int>()!=worlds_ || learner.value<uint32_t>()!=source_.seed)
            throw std::runtime_error("Incompatible nested holding learner");
        stats_.update=learner.value<int>();
        if(stats_.update<stats_.adopted_update) throw std::runtime_error("Holding learner predates its published actor");
        stats_.loaded_update=stats_.update;stats_.saved_update=stats_.update;stats_.resumed=true;
        if(stats_.applied_update<stats_.adopted_update) {pending_=adopted_;pending_update_=stats_.adopted_update;}
    }
    void save(rl::ShelterMotorTrainer& trainer) {
        try {
            const auto training=rl::shelter_motor_snapshot(trainer);checkpoint::Writer out;
            out.value(hold_magic);out.value(hold_version);out.value(static_cast<uint32_t>(sizeof(sm::State)));out.value(worlds_);
            out.array(source_payload_);
            {
                std::lock_guard<std::mutex> lock(mutex_);out.value(frame_);out.value(held_ticks_);out.value(longest_held_ticks_);
                out.array(adopted_);out.value(stats_.adopted_update);out.array(live_actor_);out.value(stats_.applied_update);
            }
            out.array(training);checkpoint::write_atomic(path_,out.data);
            std::lock_guard<std::mutex> lock(mutex_);stats_.saved_update=trainer.update_id;stats_.save_error.clear();
        } catch(const std::exception& error) {std::lock_guard<std::mutex> lock(mutex_);stats_.save_error=error.what();}
    }
    void run() {
        try {
            rl::ShelterMotorTrainer trainer(worlds_,source_.seed);
            rl::restore_shelter_motor(trainer,restored_.empty()?source_.training:restored_);restored_.clear();
            for(int joint=0;joint<4;++joint) if(trainer.env.action_scales().value[joint]!=source_.scales.value[joint])
                throw std::runtime_error("Holding action calibration changed during restore");
            {
                std::lock_guard<std::mutex> lock(mutex_);stats_.ready=true;stats_.training=true;stats_.update=trainer.update_id;
                stats_.trained_context=trainer.env.context().version;
            }
            auto last_save=rl::Clock::now();
            while(!stopping_.load()) {
                if(save_.exchange(false) || rl::Clock::now()-last_save>=std::chrono::seconds(30)) {save(trainer);last_save=rl::Clock::now();}
                if(paused_.load()) {std::this_thread::sleep_for(std::chrono::milliseconds(10));continue;}
                const auto elapsed=trainer.update_seconds();const auto learned=trainer.actor.get_params();
                for(float value:learned) if(!std::isfinite(value)) throw std::runtime_error("Non-finite holding learner; approved live actor retained");
                {
                    std::lock_guard<std::mutex> lock(mutex_);stats_.update=trainer.update_id;stats_.update_ms=elapsed*1000.;stats_.learner_changed=learned!=initial_;
                }
                if(trainer.update_id%25!=0) continue;
                const uint32_t seed=rng::bits(0x4810CA71u,static_cast<uint32_t>(trainer.update_id),0x191u);
                const auto candidate=prove(learned,source_,seed);
                std::vector<float> incumbent;{std::lock_guard<std::mutex> lock(mutex_);incumbent=adopted_;}
                const auto prior=prove(incumbent,source_,seed);
                const bool improves=candidate.holds>prior.holds ||
                    (candidate.holds==prior.holds && candidate.exposure<prior.exposure-.002);
                const bool accepted=candidate.passed() && improves;
                {
                    std::lock_guard<std::mutex> lock(mutex_);++stats_.evaluations;
                    stats_.candidate_score=candidate.food;stats_.original_score=candidate.old_food;
                    stats_.shelter_exposure=candidate.exposure;stats_.shelter_reference_exposure=prior.exposure;
                    stats_.calm_retention_score=candidate.food;
                    if(accepted) {
                        adopted_=learned;stats_.adopted_update=trainer.update_id;stats_.adopted_score=candidate.food;
                        pending_=learned;pending_update_=trainer.update_id;
                    }
                }
                report_proof("MATCHED_GATE",trainer.update_id,candidate,accepted);
            }
            save(trainer);std::lock_guard<std::mutex> lock(mutex_);stats_.training=false;
        } catch(const std::exception& error) {std::lock_guard<std::mutex> lock(mutex_);stats_.error=error.what();stats_.training=false;}
    }
public:
    HoldingSession(const std::string& source_path,std::string checkpoint,int worlds):path_(std::move(checkpoint)),worlds_(worlds),
        live_(1,7),actor_(rl::swimmer_specification(4,sm::kObs),1,7),means_(4) {
        if(path_.empty() || worlds_<16 || worlds_>2048 || worlds_%8) throw std::invalid_argument("Holding practice requires a separate full save and a valid population");
        if(std::filesystem::weakly_canonical(path_)==std::filesystem::weakly_canonical(source_path))
            throw std::invalid_argument("Holding practice must not overwrite its source motor run");
        lease_=std::make_unique<checkpoint::Lease>(path_);
        if(checkpoint::exists(path_)) restore(path_);
        else {
            source_payload_=checkpoint::read(source_path);source_=rl::read_shelter_motor_run(source_payload_);
            if(source_.worlds!=worlds_) throw std::runtime_error("Holding practice must resume the source's complete population");
            // This is a newly assigned lesson world, not relocation of the
            // user's existing creature. Its body thereafter moves only by NN
            // torque commands and the unchanged articulated physics.
            const auto cases=rl::shelter_hold_cases(source_.source.frame,0x481011FEu);const auto& lesson=cases.front();
            frame_.state=lesson.state;frame_.request.cover=true;frame_.request.goal=lesson.goal;frame_.request.eye=lesson.predator_eye;
            const auto target=predator::cover(0,lesson.predator_eye,lesson.state.habitat,{2});
            frame_.request.heading=deep::axis_rotation({0,1,0},target.yaw).rotate({1,0,0});
            initial_=source_.actor;adopted_=initial_;live_actor_=initial_;
            stats_.update=source_.update;stats_.loaded_update=source_.update;
            stats_.adopted_update=source_.update;stats_.applied_update=source_.update;
        }
        initial_=source_.actor;
        const auto proof=prove(source_.actor,source_,0x48101011u);report_proof("READINESS",source_.update,proof,proof.passed());
        if(!proof.passed()) throw std::runtime_error("Calm holding lesson is not ready: it needs 18 of 24 perturbed five-second holds and 90% original calm-food retention. The boulder pond is preserved.");
        const arch::Context visible_context{{},source_.source.context.version,{2}};
        live_.configure(source_.source.ancestry.policy.actor,visible_context,live_stream_.value,false);live_.set_action_scales(source_.scales);
        live_.set_state(frame_,live_stream_.value);actor_.set_params(live_actor_);measure(frame_,false);
        stats_.backend="CUDA";stats_.hold_practice_mode=true;stats_.arch_mode=true;stats_.worlds=worlds_;stats_.input_count=sm::kObs;stats_.layout_id=2;
        stats_.habitat_name="Calm holding lesson · assigned goal";stats_.parent_update=source_.source.motor_update;
        stats_.parent_stage_update=source_.update;stats_.parent_selector_update=source_.source.ancestry.policy.update;
        stats_.imported_full_state=true;stats_.context_version=visible_context.version;
        stats_.original_score=proof.old_food;stats_.adopted_score=proof.food;stats_.calm_retention_score=proof.food;
        worker_=std::thread([this] {run();});
    }
    ~HoldingSession() override {stop();}
    feeding::State step() override {
        std::vector<float> pending;int update=0;
        {std::lock_guard<std::mutex> lock(mutex_);if(!pending_.empty()) {pending=std::move(pending_);pending_.clear();update=pending_update_;}}
        if(!pending.empty()) {
            actor_.set_params(pending);std::lock_guard<std::mutex> lock(mutex_);live_actor_=std::move(pending);stats_.applied_update=update;
        }
        actor_.forward(live_.obs(),1,live_stream_.value);
        sm::policy_means(actor_.output(),live_.obs(),means_.get(),1,source_.scales,live_stream_.value);
        live_.step(means_.get(),{},live_stream_.value,false,false);
        const auto frame=live_.state(live_stream_.value);
        if(!finite_frame(frame)) {
            std::lock_guard<std::mutex> lock(mutex_);stats_.error="Holding practice reached an invalid state; no reset or scripted recovery was applied.";
            throw std::runtime_error(stats_.error);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);measure(frame,true);frame_=frame;
            stats_.arch_contacts=frame.state.contacts;stats_.arch_passages=frame.state.passages;stats_.arch_clean_passages=frame.state.clean_passages;
        }
        return frame.state.pond;
    }
    const feeding::State& state() const override {return frame_.state.pond;}
    int drop(deep::V) override {return -1;}
    void set_current(deep::V current) override {
        if(!std::isfinite(deep::norm(current)) || deep::norm(current)>1e-8f)
            throw std::runtime_error("This first holding lesson is calm-water practice; current compensation is a later test.");
    }
    deep::V current() const override {return {};}
    FeedingStats stats() const override {std::lock_guard<std::mutex> lock(mutex_);auto result=stats_;result.paused=paused_.load();return result;}
    void request_save() override {save_.store(true);}
    void set_paused(bool paused) override {paused_.store(paused);}
    PredatorInfo predator_info() const override {
        std::lock_guard<std::mutex> lock(mutex_);PredatorInfo info;info.enabled=true;info.position=frame_.request.eye;
        info.goal=frame_.request.goal.position;info.cover_goal=true;info.exposure=exposure_;info.clearance=clearance_;info.goal_distance=distance_;
        info.held_seconds=held_ticks_*deep::dt;info.longest_safe_seconds=longest_held_ticks_*deep::dt;return info;
    }
    void stop() override {stopping_.store(true);if(worker_.joinable()) worker_.join();}
};
}
std::unique_ptr<FeedingSession> create_hold_session(const std::string& source,const std::string& checkpoint,int worlds) {
    return std::make_unique<HoldingSession>(source,checkpoint,worlds);
}
}
