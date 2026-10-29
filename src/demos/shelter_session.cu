#include "feeding_session.h"
#include "rl/shelter_motor_run.h"
#include "rl/shelter_ppo.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <thread>

namespace cudarl::demos {
namespace {
namespace sh=shelter;
constexpr uint32_t session_magic=0x53484C31u,session_version=1;
uint64_t actor_fingerprint(const std::vector<float>& actor) {
    uint64_t result=14695981039346656037ull;
    for(float value:actor) {
        unsigned char bytes[sizeof(float)];std::memcpy(bytes,&value,sizeof(value));
        for(unsigned char byte:bytes) {result^=byte;result*=1099511628211ull;}
    }
    return result;
}
bool same_context(const arch::Context& a,const arch::Context& b) {
    return a.version==b.version && a.basin.layout==b.basin.layout && a.current.x==b.current.x && a.current.y==b.current.y && a.current.z==b.current.z;
}
void validate_context(const arch::Context& context) {
    if(context.basin.layout!=2 || context.version<0 || !std::isfinite(deep::norm(context.current)) ||
       deep::norm(context.current)>.0075001f || fabsf(context.current.y)>1e-8f)
        throw std::runtime_error("Shelter pond requires boulder geometry and a horizontal current no faster than 0.75 cm/s");
}
struct Watch {
    float minimum_clearance=1e9f;
    float exposure_seconds=0,threat_seconds=0,safe_seconds=0;
    int invalid=0,severe_ticks=0,severe_streak=0,longest_severe=0,safe_streak=0,longest_safe=0;
    bool cue_seen=false,exposed_at_cue=false;
};
__device__ __noinline__ float exact_exposure(const sh::World& world,deep::V eye,deep::Basin basin) {
    return predator::exposure(world.state.pond.body,eye,world.state.habitat,basin);
}
__global__ void watch_kernel(const sh::World* worlds,Watch* reports,int n,deep::Basin basin) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    const auto& world=worlds[i];auto& report=reports[i];const auto threat=sh::event(world);
    const float gap=predator::body_clearance(world.state.pond.body,world.state.habitat,basin);
    const float exposure=exact_exposure(world,threat.eye,basin);
    report.minimum_clearance=fminf(report.minimum_clearance,gap);
    if(world.state.pond.invalid || !isfinite(gap) || !isfinite(exposure)) report.invalid=1;
    if(gap<-.2f) {++report.severe_ticks;++report.severe_streak;report.longest_severe=max(report.longest_severe,report.severe_streak);}
    else report.severe_streak=0;
    // Eligibility is set when the warning first becomes visible. A creature
    // that is already hidden cannot earn an escape; timely warning responses
    // remain eligible even when they reach cover before the active pass.
    if(!report.cue_seen && (threat.warning || threat.active)) {report.cue_seen=true;report.exposed_at_cue=exposure>.5f;}
    if(threat.active) {report.threat_seconds+=deep::dt;report.exposure_seconds+=exposure*deep::dt;}
    if(report.exposed_at_cue && threat.active && exposure<=.05f && gap>=.08f) {
        report.safe_seconds+=deep::dt;++report.safe_streak;report.longest_safe=max(report.longest_safe,report.safe_streak);
    }
    else report.safe_streak=0;
}
struct Evaluation {
    int worlds=0,foods=0,post_foods=0,safe_worlds=0,eligible_worlds=0,invalid=0,contacts=0,severe_ticks=0,longest_severe=0;
    double exposure_seconds=0,threat_seconds=0,safe_seconds=0,seconds=48;
    float minimum_clearance=1e9f;
    double exposure() const {return threat_seconds>0?exposure_seconds/threat_seconds:0;}
    double food_rate() const {return worlds>0?foods*60.0/(worlds*seconds):0;}
};
Evaluation evaluate(const std::vector<float>& actor,const std::vector<float>& motor,const arch::Context& context,
                    uint32_t seed,bool threats,bool allow_cover,shelter_motor::ActuatorScale scales={},int count=16,int ticks=960) {
    sh::Options options;options.training=false;options.threats=threats;options.cover=allow_cover;
    sh::Batch worlds(count,seed,context,sh::kPeriod,options);rl::SwimmerStream stream;
    nn::GpuMlp selector(rl::shelter_selector_spec(),count*sh::kChoices,7),body(rl::swimmer_specification(4,shelter_motor::kObs),count,7);
    selector.set_params(actor);body.set_params(motor);
    device_ptr<float> features(static_cast<size_t>(count)*sh::kChoices*sh::kFeatures),logits(static_cast<size_t>(count)*sh::kChoices),logp(count),means(static_cast<size_t>(count)*shelter_motor::kAct);
    device_ptr<int> choices(count);device_ptr<Watch> watch(count);std::vector<Watch> monitored(static_cast<size_t>(count));watch.copy_from_host(monitored.data(),monitored.size());
    for(int tick=0;tick<ticks;++tick) {
        if(tick%sh::kPeriod==0) {
            sh::make_features(worlds.obs(),features.get(),count,stream.value);selector.forward(features.get(),count*sh::kChoices,stream.value);
            rl::swimmer_copy(logits.get(),selector.output(),static_cast<size_t>(count)*sh::kChoices,stream.value);sh::mask_logits(worlds.obs(),logits.get(),count,stream.value);
            rl::sample_actions(logits.get(),count,sh::kChoices,0,0,true,choices.get(),logp.get(),stream.value);worlds.begin(choices.get(),{},stream.value);
        }
        const auto* motor_input=worlds.motor_obs(stream.value);body.forward(motor_input,count,stream.value);
        shelter_motor::policy_means(body.output(),motor_input,means.get(),count,scales,stream.value);
        worlds.step(means.get(),{},stream.value,false);
        watch_kernel<<<(count+63)/64,64,0,stream.value>>>(worlds.worlds(),watch.get(),count,context.basin);CUDA_CHECK_THROW(cudaGetLastError());
    }
    const auto states=worlds.states(stream.value);watch.copy_to_host(monitored.data(),monitored.size());
    Evaluation result;result.worlds=count;result.seconds=ticks*deep::dt;
    for(int i=0;i<count;++i) {
        const auto& state=states[static_cast<size_t>(i)];const auto& item=monitored[static_cast<size_t>(i)];
        result.foods+=state.state.pond.eaten;result.post_foods+=state.metrics.post_threat_foods;
        result.exposure_seconds+=item.exposure_seconds;result.threat_seconds+=item.threat_seconds;
        result.safe_seconds+=item.safe_seconds;result.safe_worlds+=item.longest_safe>=60?1:0;
        result.eligible_worlds+=item.exposed_at_cue?1:0;
        result.invalid+=item.invalid;result.contacts+=state.state.contacts;result.severe_ticks+=item.severe_ticks;
        result.longest_severe=std::max(result.longest_severe,item.longest_severe);result.minimum_clearance=std::min(result.minimum_clearance,item.minimum_clearance);
    }
    return result;
}
bool retention(const Evaluation& candidate,const Evaluation& reference) {
    return candidate.invalid==0 && candidate.foods>=reference.foods*.9 &&
        candidate.severe_ticks<=reference.severe_ticks+candidate.worlds*10 &&
        candidate.longest_severe<=reference.longest_severe+20 &&
        candidate.contacts<=reference.contacts*1.25+candidate.worlds*3;
}
bool shelter_gate(const Evaluation& candidate,const Evaluation& reference,const Evaluation& calm,const Evaluation& calm_reference) {
    return retention(calm,calm_reference) && candidate.invalid==0 && candidate.eligible_worlds>=candidate.worlds/2 && candidate.safe_worlds>=candidate.worlds/2 &&
        candidate.exposure()<=reference.exposure()*.9 && candidate.exposure()<=reference.exposure()-.05 &&
        candidate.safe_seconds>=reference.safe_seconds+candidate.worlds && candidate.post_foods>=reference.post_foods*.9 &&
        candidate.severe_ticks<=reference.severe_ticks+candidate.worlds*10 && candidate.longest_severe<=reference.longest_severe+20 &&
        candidate.contacts<=reference.contacts*1.25+candidate.worlds*3;
}
void report_evaluation(const char* phase,int update,const Evaluation& trial,const Evaluation& reference,const Evaluation& calm,const Evaluation& calm_reference,bool accepted) {
    std::printf("SHELTER_%s update=%d exposure=%.4f reference=%.4f safe_worlds=%d/%d eligible_exposed=%d safe_seconds=%.2f post_food=%d reference_post=%d calm_food=%d reference_calm=%d invalid=%d severe_ticks=%d reference_severe=%d longest_severe=%d contacts=%d minimum_clearance=%.5f accepted=%d\n",
        phase,update,trial.exposure(),reference.exposure(),trial.safe_worlds,trial.worlds,trial.eligible_worlds,trial.safe_seconds,trial.post_foods,reference.post_foods,
        calm.foods,calm_reference.foods,trial.invalid+calm.invalid,trial.severe_ticks,reference.severe_ticks,trial.longest_severe,trial.contacts,trial.minimum_clearance,accepted?1:0);std::fflush(stdout);
}
class ShelterSession final:public FeedingSession {
    std::string path_;int worlds_;
    std::unique_ptr<checkpoint::Lease> lease_;
    rl::BoulderSource source_;
    std::vector<uint8_t> motor_payload_,restored_;
    std::vector<float> frozen_motor_,baseline_,adopted_,live_actor_,pending_;
    shelter_motor::ActuatorScale frozen_scales_{};
    rl::SwimmerStream live_stream_;
    sh::Batch live_;
    nn::GpuMlp live_selector_,live_motor_;
    device_ptr<float> live_features_,live_logits_,live_logp_,live_motor_means_;
    device_ptr<int> live_choice_;
    sh::World frame_{};
    arch::Context context_{},practice_context_{};
    mutable std::mutex mutex_;
    FeedingStats stats_;
    bool adopted_cover_=false,live_cover_=false,pending_cover_=false,choose_soon_=true;
    int pending_update_=0,motor_hold_successes_=0;
    double motor_calm_reference_=0,motor_calm_=0;
    uint64_t motor_fingerprint_=0;
    std::atomic<bool> stopping_{false},save_{false},paused_{false},trigger_{false};
    std::thread worker_;

    void readiness() {
        const auto cases=rl::shelter_motor_probe_cases(source_.frame,0x5E1E1001u);
        const auto results=rl::probe_shelter_motor(frozen_motor_,cases,240,100,frozen_scales_);
        int finite=0;motor_hold_successes_=0;
        for(const auto& result:results) {finite+=result.finite?1:0;motor_hold_successes_+=result.longest_safe_dwell_ticks>=100?1:0;}
        const auto original=rl::transfer_shelter_motor(source_.motor);
        motor_calm_reference_=rl::shelter_motor_food_rate(original,source_,0x5E1E1002u);
        motor_calm_=rl::shelter_motor_food_rate(frozen_motor_,source_,0x5E1E1002u,16,600,{},frozen_scales_);
        std::printf("SHELTER_MOTOR_READINESS holds=%d/%zu finite=%d original_calm_food_min=%.3f candidate_calm_food_min=%.3f\n",
            motor_hold_successes_,results.size(),finite,motor_calm_reference_,motor_calm_);std::fflush(stdout);
        if(finite!=static_cast<int>(results.size()) || motor_hold_successes_<18 || motor_calm_<motor_calm_reference_*.9)
            throw std::runtime_error("Shelter prototype is still practicing: its body controller has not passed 18 of 24 five-second cover holds and 90% calm-food retention. The working boulder pond and all saves are preserved.");
    }
    void restore(const std::string& path) {
        const auto bytes=checkpoint::read(path);checkpoint::Reader in{bytes};
        if(in.value<uint32_t>()!=session_magic || in.value<uint32_t>()!=session_version || in.value<uint32_t>()!=sizeof(sh::World) || in.value<int>()!=worlds_)
            throw std::runtime_error("Incompatible shelter session checkpoint");
        motor_payload_=in.array<uint8_t>(256000000);const auto motor=rl::read_shelter_motor_run(motor_payload_);source_=motor.source;frozen_scales_=motor.scales;
        const auto parent=in.value<uint64_t>();motor_fingerprint_=in.value<uint64_t>();frozen_motor_=in.array<float>(rl::swimmer_specification(4,shelter_motor::kObs).num_params());
        if(parent!=source_.fingerprint || frozen_motor_!=motor.actor || actor_fingerprint(frozen_motor_)!=motor_fingerprint_)
            throw std::runtime_error("Saved shelter motor or parent identity changed");
        frame_=in.value<sh::World>();context_=in.value<arch::Context>();practice_context_=in.value<arch::Context>();validate_context(context_);validate_context(practice_context_);
        adopted_=in.array<float>(rl::shelter_selector_spec().num_params());stats_.adopted_update=in.value<int>();adopted_cover_=in.value<bool>();
        live_actor_=in.array<float>(rl::shelter_selector_spec().num_params());stats_.applied_update=in.value<int>();live_cover_=in.value<bool>();
        (void)in.value<int>();(void)in.value<double>();(void)in.value<double>();restored_=in.array<uint8_t>(256000000);in.finish();
        if(adopted_.size()!=static_cast<size_t>(rl::shelter_selector_spec().num_params()) || live_actor_.size()!=adopted_.size() ||
           stats_.adopted_update<0 || stats_.applied_update<0 || stats_.applied_update>stats_.adopted_update)
            throw std::runtime_error("Invalid saved shelter publication state");
        for(const auto* weights:{&adopted_,&live_actor_}) for(float value:*weights) if(!std::isfinite(value)) throw std::runtime_error("Non-finite saved shelter selector");
        if(stats_.applied_update==stats_.adopted_update && (live_actor_!=adopted_ || live_cover_!=adopted_cover_))
            throw std::runtime_error("Inconsistent saved shelter live controller");
        checkpoint::Reader learner{restored_};(void)learner.value<uint32_t>();(void)learner.value<uint32_t>();(void)learner.value<uint32_t>();
        if(learner.value<int>()!=worlds_) throw std::runtime_error("Saved shelter practice population changed");
        (void)learner.value<uint32_t>();stats_.loaded_update=learner.value<int>();
        if(stats_.loaded_update<stats_.adopted_update) throw std::runtime_error("Saved shelter learner predates its published controller");
        stats_.update=stats_.loaded_update;stats_.saved_update=stats_.loaded_update;stats_.resumed=true;
        if(stats_.applied_update!=stats_.adopted_update) {pending_=adopted_;pending_cover_=adopted_cover_;pending_update_=stats_.adopted_update;}
    }
    void save(rl::ShelterTrainer& trainer) {
        if(path_.empty()) return;
        try {
            const auto training=trainer.snapshot();checkpoint::Writer out;
            out.value(session_magic);out.value(session_version);out.value(static_cast<uint32_t>(sizeof(sh::World)));out.value(worlds_);
            out.array(motor_payload_);out.value(source_.fingerprint);out.value(motor_fingerprint_);out.array(frozen_motor_);
            {
                std::lock_guard<std::mutex> lock(mutex_);out.value(frame_);out.value(context_);out.value(trainer.context);
                out.array(adopted_);out.value(stats_.adopted_update);out.value(adopted_cover_);out.array(live_actor_);out.value(stats_.applied_update);out.value(live_cover_);
                out.value(motor_hold_successes_);out.value(motor_calm_reference_);out.value(motor_calm_);
            }
            out.array(training);checkpoint::write_atomic(path_,out.data);
            std::lock_guard<std::mutex> lock(mutex_);stats_.saved_update=trainer.update;stats_.save_error.clear();
        } catch(const std::exception& error) {std::lock_guard<std::mutex> lock(mutex_);stats_.save_error=error.what();}
    }
    void run() {
        try {
            rl::ShelterTrainer trainer(source_.ancestry.policy.actor,frozen_motor_,worlds_,79,practice_context_,frozen_scales_);
            if(!restored_.empty()) {trainer.restore(restored_);restored_.clear();}
            const auto original_motor=rl::transfer_shelter_motor(source_.motor);
            {std::lock_guard<std::mutex> lock(mutex_);stats_.ready=true;stats_.training=true;}
            auto last_save=rl::Clock::now();
            while(!stopping_.load()) {
                if(save_.exchange(false) || rl::Clock::now()-last_save>=std::chrono::seconds(30)) {save(trainer);last_save=rl::Clock::now();}
                if(paused_.load()) {std::this_thread::sleep_for(std::chrono::milliseconds(10));continue;}
                arch::Context practice;{std::lock_guard<std::mutex> lock(mutex_);practice=context_;}
                if(!same_context(practice,trainer.context)) {trainer.env.configure(practice,trainer.stream.value);trainer.context=practice;}
                const auto start=rl::Clock::now();trainer.train_update();const auto learned=trainer.policy();
                for(float value:learned) if(!std::isfinite(value)) throw std::runtime_error("Non-finite goal learner; the visible inherited controller was retained");
                {
                    std::lock_guard<std::mutex> lock(mutex_);stats_.update=trainer.update;stats_.update_ms=std::chrono::duration<double,std::milli>(rl::Clock::now()-start).count();
                    stats_.trained_context=practice.version;stats_.learner_changed=learned!=baseline_;
                }
                if(trainer.update%25!=0) continue;
                const uint32_t development_seed=0x5E1EDE01u;
                auto calm_context=practice;calm_context.current={};
                const auto reference=evaluate(baseline_,frozen_motor_,practice,development_seed,true,false,frozen_scales_);
                const auto calm_reference=evaluate(baseline_,original_motor,calm_context,development_seed^0x00101010u,false,false);
                const auto trial=evaluate(learned,frozen_motor_,practice,development_seed,true,true,frozen_scales_);
                const auto calm=evaluate(learned,frozen_motor_,calm_context,development_seed^0x00101010u,false,true,frozen_scales_);
                const bool development_pass=shelter_gate(trial,reference,calm,calm_reference);
                report_evaluation("DEVELOPMENT",trainer.update,trial,reference,calm,calm_reference,development_pass);
                {
                    std::lock_guard<std::mutex> lock(mutex_);++stats_.evaluations;stats_.evaluated_context=practice.version;
                    stats_.original_score=reference.food_rate();stats_.candidate_score=trial.food_rate();stats_.shelter_exposure=trial.exposure();
                    stats_.shelter_reference_exposure=reference.exposure();stats_.calm_retention_score=calm.food_rate();
                }
                if(!development_pass || stopping_.load()) continue;
                const uint32_t gate_seed=rng::bits(0x5E1ECA71u,static_cast<uint32_t>(trainer.update),0x131u);
                const auto gate_reference=evaluate(baseline_,frozen_motor_,practice,gate_seed,true,false,frozen_scales_);
                const auto gate_calm_reference=evaluate(baseline_,original_motor,calm_context,gate_seed^0x00404040u,false,false);
                const auto gate_trial=evaluate(learned,frozen_motor_,practice,gate_seed,true,true,frozen_scales_);
                const auto gate_calm=evaluate(learned,frozen_motor_,calm_context,gate_seed^0x00404040u,false,true,frozen_scales_);
                bool accepted=shelter_gate(gate_trial,gate_reference,gate_calm,gate_calm_reference);
                std::vector<float> incumbent;bool incumbent_cover=false;
                {std::lock_guard<std::mutex> lock(mutex_);incumbent=adopted_;incumbent_cover=adopted_cover_;}
                if(accepted && incumbent_cover) {
                    const auto prior=evaluate(incumbent,frozen_motor_,practice,gate_seed,true,true,frozen_scales_);
                    accepted=gate_trial.exposure()<=prior.exposure()-.01 && gate_trial.post_foods>=prior.post_foods*.9;
                }
                if(accepted) {
                    std::lock_guard<std::mutex> lock(mutex_);accepted=same_context(practice,context_);
                    if(accepted) {
                        adopted_=learned;adopted_cover_=true;stats_.adopted_update=trainer.update;stats_.adopted_score=gate_trial.food_rate();
                        pending_=learned;pending_cover_=true;pending_update_=trainer.update;
                    }
                }
                report_evaluation("MATCHED_GATE",trainer.update,gate_trial,gate_reference,gate_calm,gate_calm_reference,accepted);
            }
            save(trainer);std::lock_guard<std::mutex> lock(mutex_);stats_.training=false;
        } catch(const std::exception& error) {std::lock_guard<std::mutex> lock(mutex_);stats_.error=error.what();stats_.training=false;}
    }
public:
    ShelterSession(const std::string& source_path,std::string checkpoint,int worlds):path_(std::move(checkpoint)),worlds_(worlds),
        live_(1,7,arch::Context{{},0,{2}},sh::kPeriod,sh::Options{false,true,true,false}),
        live_selector_(rl::shelter_selector_spec(),sh::kChoices,7),live_motor_(rl::swimmer_specification(4,shelter_motor::kObs),1,7),
        live_features_(sh::kChoices*sh::kFeatures),live_logits_(sh::kChoices),live_logp_(1),live_motor_means_(shelter_motor::kAct),live_choice_(1) {
        if(worlds<16 || worlds>2048 || worlds%8) throw std::invalid_argument("Invalid shelter practice population");
        if(!path_.empty() && std::filesystem::weakly_canonical(path_)==std::filesystem::weakly_canonical(source_path))
            throw std::invalid_argument("Shelter learning must use its own checkpoint");
        if(!path_.empty()) lease_=std::make_unique<checkpoint::Lease>(path_);
        if(!path_.empty() && checkpoint::exists(path_)) restore(path_);
        else {
            const auto motor_path=(std::filesystem::path(path_).parent_path()/"motor-training.bin").string();
            if(!checkpoint::exists(motor_path)) throw std::runtime_error("Shelter motor practice is not ready. Keep using the working boulder pond; its source save is unchanged.");
            motor_payload_=checkpoint::read(motor_path);const auto motor=rl::read_shelter_motor_run(motor_payload_);source_=motor.source;frozen_scales_=motor.scales;
            const auto requested=rl::load_boulder_source(source_path);
            if(source_.fingerprint!=requested.fingerprint || source_.parent!=requested.parent) throw std::runtime_error("The shelter motor was trained from a different boulder save");
            frozen_motor_=motor.actor;motor_fingerprint_=actor_fingerprint(frozen_motor_);context_=source_.context;practice_context_=context_;
            frame_.state=source_.frame;frame_.predator.seed=7;frame_.chosen=frame_.state.pond.target;
            frame_.goal=sh::option(frame_,frame_.chosen,context_.basin);
            frame_.exposure=predator::exposure(frame_.state.pond.body,sh::event(frame_).eye,frame_.state.habitat,context_.basin);
            adopted_=rl::transfer_shelter_actor(source_.ancestry.policy.actor);live_actor_=adopted_;
        }
        baseline_=rl::transfer_shelter_actor(source_.ancestry.policy.actor);readiness();
        validate_context(context_);live_.configure(context_,live_stream_.value);live_.set_options({false,true,true,live_cover_},live_stream_.value);
        live_.set_state(frame_,live_stream_.value);live_selector_.set_params(live_actor_);live_motor_.set_params(frozen_motor_);
        stats_.backend="CUDA";stats_.shelter_mode=true;stats_.arch_mode=true;stats_.worlds=worlds_;stats_.input_count=shelter_motor::kObs;stats_.layout_id=2;
        stats_.habitat_name="Predator shelter · boulder pond";stats_.parent_update=source_.motor_update;stats_.parent_stage_update=source_.learner_update;
        stats_.parent_selector_update=source_.ancestry.policy.update;stats_.imported_full_state=true;stats_.context_version=context_.version;
        worker_=std::thread([this] {run();});
    }
    ~ShelterSession() override {stop();}
    feeding::State step() override {
        std::vector<float> next;int applied=0;bool allow_cover=false;
        {std::lock_guard<std::mutex> lock(mutex_);if(!pending_.empty()) {next=std::move(pending_);pending_.clear();applied=pending_update_;allow_cover=pending_cover_;}}
        if(!next.empty()) {
            live_selector_.set_params(next);live_.set_options({false,true,true,allow_cover},live_stream_.value);
            std::lock_guard<std::mutex> lock(mutex_);live_actor_=std::move(next);live_cover_=allow_cover;stats_.applied_update=applied;choose_soon_=true;
        }
        if(trigger_.exchange(false)) {
            sh::World world;{std::lock_guard<std::mutex> lock(mutex_);world=frame_;}predator::trigger(world.predator);
            live_.set_state(world,live_stream_.value);choose_soon_=true;
        }
        int steps;{std::lock_guard<std::mutex> lock(mutex_);steps=frame_.state.pond.steps;}
        if(choose_soon_ || steps%sh::kPeriod==0) {
            sh::make_features(live_.obs(),live_features_.get(),1,live_stream_.value);live_selector_.forward(live_features_.get(),sh::kChoices,live_stream_.value);
            rl::swimmer_copy(live_logits_.get(),live_selector_.output(),sh::kChoices,live_stream_.value);sh::mask_logits(live_.obs(),live_logits_.get(),1,live_stream_.value);
            rl::sample_actions(live_logits_.get(),1,sh::kChoices,0,0,true,live_choice_.get(),live_logp_.get(),live_stream_.value);live_.begin(live_choice_.get(),{},live_stream_.value);choose_soon_=false;
        }
        const auto* motor_input=live_.motor_obs(live_stream_.value);live_motor_.forward(motor_input,1,live_stream_.value);
        shelter_motor::policy_means(live_motor_.output(),motor_input,live_motor_means_.get(),1,frozen_scales_,live_stream_.value);
        live_.step(live_motor_means_.get(),{},live_stream_.value,false);
        const auto frame=live_.state(live_stream_.value);
        {std::lock_guard<std::mutex> lock(mutex_);frame_=frame;stats_.arch_contacts=frame.state.contacts;stats_.arch_passages=frame.state.passages;stats_.arch_clean_passages=frame.state.clean_passages;}
        return frame.state.pond;
    }
    const feeding::State& state() const override {return frame_.state.pond;}
    int drop(deep::V) override {return -1;}
    void set_current(deep::V current) override {
        arch::Context context;{std::lock_guard<std::mutex> lock(mutex_);context=context_;}
        context.current=current;
        if(context.version==std::numeric_limits<int>::max()) throw std::runtime_error("Shelter current revision exhausted");
        ++context.version;validate_context(context);live_.configure(context,live_stream_.value);
        std::lock_guard<std::mutex> lock(mutex_);context_=context;stats_.context_version=context.version;choose_soon_=true;
    }
    deep::V current() const override {std::lock_guard<std::mutex> lock(mutex_);return context_.current;}
    FeedingStats stats() const override {std::lock_guard<std::mutex> lock(mutex_);auto result=stats_;result.paused=paused_.load();return result;}
    void request_save() override {save_.store(true);}
    void set_paused(bool paused) override {paused_.store(paused);}
    bool trigger_predator() override {trigger_.store(true);return true;}
    PredatorInfo predator_info() const override {
        std::lock_guard<std::mutex> lock(mutex_);const auto threat=sh::event(frame_);PredatorInfo result;
        result.enabled=true;result.warning=threat.warning;result.active=threat.active;result.position=threat.eye;result.shadow=threat.shadow;
        result.direction=threat.velocity;result.strength=threat.strength;result.exposure=frame_.exposure;result.goal=frame_.goal.position;result.cover_goal=frame_.goal.cover>=0;return result;
    }
    void stop() override {stopping_.store(true);if(worker_.joinable()) worker_.join();}
};
}
std::unique_ptr<FeedingSession> create_shelter_session(const std::string& source,const std::string& checkpoint,int worlds) {
    return std::make_unique<ShelterSession>(source,checkpoint,worlds);
}
}
