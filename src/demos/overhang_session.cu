#include "feeding_session.h"
#include "envs/swimmer/arch_batch.h"
#include "rl/boulder_source.h"
#include <limits>

namespace cudarl::demos {
namespace {
bool finite_vector(deep::V value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
bool valid_current(deep::V current) {
    return finite_vector(current) && deep::norm(current)<=.0075001f && fabsf(current.y)<=1e-8f;
}
rl::BoulderSource preview_source(const std::string& path) {
    auto source=rl::load_boulder_source(path);const auto& frame=source.frame;
    const auto& body=frame.pond.body;const auto& habitat=frame.habitat;
    const auto q=body.orientation;const float quaternion_norm=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
    if(frame.pond.retention || frame.pond.steps<0 || frame.pond.eaten<0 ||
       frame.pond.target<-1 || frame.pond.target>=feeding::capacity || frame.contacts<0 || frame.passages<0 ||
       frame.clean_passages<0 || frame.clean_passages>frame.passages ||
       !finite_vector(body.position) || !std::isfinite(quaternion_norm) ||
       quaternion_norm<.99f || quaternion_norm>1.01f || !std::isfinite(body.work) || !std::isfinite(body.travel) ||
       !std::isfinite(habitat.scale) || habitat.scale<=0 || !finite_vector(habitat.center) ||
       !std::isfinite(habitat.major) || habitat.major<=0 || !std::isfinite(habitat.minor) || habitat.minor<=0 ||
       !std::isfinite(habitat.yaw) || !valid_current(source.context.current) || source.context.version<0)
        throw std::runtime_error("Invalid live boulder frame for the overhang preview; original save preserved");
    for(float velocity:body.velocity) if(!std::isfinite(velocity))
        throw std::runtime_error("Non-finite saved boulder velocity; original save preserved");
    for(float angle:body.angle) if(!std::isfinite(angle))
        throw std::runtime_error("Non-finite saved boulder joint; original save preserved");
    // Reject a newly conflicting layout instead of shifting the personal
    // creature, changing its joints or silently deleting/moving real food.
    const float clearance=deep::overhang_body_clearance(body,habitat);
    if(!std::isfinite(clearance) || clearance<0)
        throw std::runtime_error("Saved swimmer overlaps the new overhang; original boulder save preserved");
    for(const auto& food:frame.pond.pellets) if(food.active) {
        if(!finite_vector(food.position) || !finite_vector(food.velocity))
            throw std::runtime_error("Non-finite saved boulder food; original save preserved");
        if(deep::overhang_distance(food.position,habitat)<feeding::radius/deep::scale)
            throw std::runtime_error("Saved food overlaps the new overhang; original boulder save preserved");
    }
    return source;
}
struct PreviewStream {
    cudaStream_t value=nullptr;
    PreviewStream() {CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~PreviewStream() {if(value) {cudaStreamSynchronize(value);cudaStreamDestroy(value);}}
};
// This isolated viewer runs inherited neural weights in the current native
// physical world. It has no learner, replay timeline, save lease or writer.
class OverhangSession final:public FeedingSession {
    rl::BoulderSource source_;
    arch::Batch live_;
    nn::GpuMlp motor_;
    arch::State frame_;
    arch::Context context_;
    FeedingStats stats_;
    bool paused_=false;
    // Destroy/synchronize the stream before releasing the CUDA actors/world.
    PreviewStream stream_;
public:
    explicit OverhangSession(const std::string& path):source_(preview_source(path)),
        live_(1,7),motor_(rl::swimmer_specification(arch::kAct,arch::kObs),1,7),
        frame_(source_.frame),context_(source_.context) {
        context_.basin.layout=3;motor_.set_params(source_.motor);
        live_.configure(source_.ancestry.policy.actor,context_,stream_.value,false);
        live_.set_state(frame_,stream_.value);frame_=live_.state(stream_.value);
        stats_.arch_mode=true;stats_.layout_id=3;stats_.input_count=arch::kObs;
        stats_.habitat_name="Overhang pond · live preview";
        stats_.ready=true;stats_.training=false;stats_.worlds=0;
        stats_.update=source_.learner_update;stats_.loaded_update=source_.learner_update;
        stats_.parent_stage_update=source_.learner_update;
        stats_.adopted_update=source_.motor_update;stats_.applied_update=source_.motor_update;
        stats_.parent_update=source_.ancestry.parent_update;
        stats_.parent_selector_update=source_.ancestry.policy.update;
    }
    feeding::State step() override {
        if(paused_) return frame_.pond;
        motor_.forward(live_.obs(),1,stream_.value);
        live_.step(motor_.output(),{},stream_.value,false,false);
        frame_=live_.state(stream_.value);return frame_.pond;
    }
    const feeding::State& state() const override {return frame_.pond;}
    int drop(deep::V) override {return -1;} // The retained automatic-food task.
    void set_current(deep::V current) override {
        if(!valid_current(current))
            throw std::invalid_argument("Overhang current must be finite, horizontal and at most 0.75 cm/s");
        if(current.x==context_.current.x && current.y==context_.current.y && current.z==context_.current.z) return;
        if(context_.version==std::numeric_limits<int>::max())
            throw std::runtime_error("Overhang current context counter exhausted");
        auto next=context_;next.current=current;++next.version;
        live_.configure(source_.ancestry.policy.actor,next,stream_.value,false);
        context_=next;
    }
    deep::V current() const override {return context_.current;}
    FeedingStats stats() const override {
        auto result=stats_;result.paused=paused_;result.context_version=context_.version;
        result.arch_contacts=frame_.contacts;result.arch_passages=frame_.passages;
        result.arch_clean_passages=frame_.clean_passages;return result;
    }
    void request_save() override {} // Inference does not own a training save.
    void set_paused(bool paused) override {paused_=paused;}
    void stop() override {} // No worker and no checkpoint writes.
};
}
std::unique_ptr<FeedingSession> create_overhang_session(const std::string& source_boulder_full_save) {
    return std::make_unique<OverhangSession>(source_boulder_full_save);
}
}
