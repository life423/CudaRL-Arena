#include "hold_lab_preview.h"
#include "core/checkpoint.h"
#include "motor/frozen_controller.h"
#include "motor/hold_assessment.h"
#include "../../tools/hold_learned_amplitude_math.h"
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <array>
#include <cmath>
#include <stdexcept>

namespace godot {
namespace sm = cudarl::straight_motor;
namespace model = cudarl::movement;
namespace p = cudarl::physics;
namespace cp = cudarl::checkpoint;
namespace nn = cudarl::nn;
namespace {
Vector3 point(const p::V& v) { return Vector3(v.x,v.y,v.z); }
constexpr char canonical[] = "810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0";
}
struct HoldLabPreview::Impl {
    nn::CpuMlp<float> actor{{{41,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}}};
    sm::World world{};
    sm::ForwardTask task{};
    sm::ForwardConfig config{};
    sm::RestHoldAssessment hold_score;
    std::array<float,4> gains{1,1,1,1};
    std::array<float,2> head_parameters{0,0};
    uint32_t package_version=0,policy_method=0;
    float learned_sigma=0;
    bool loaded=false,paused=true,holding=true,qualified=false;
    int case_id=0,lesson_updates=0;
    uint64_t controls=0,substeps=0,protocol=0;
    uint64_t attempted_controls=0;
    p::BodyState failed_before{},failed_candidate{};
    bool failure_recorded=false;
    std::string error;
};
HoldLabPreview::HoldLabPreview() : impl_(std::make_unique<Impl>()) {}
HoldLabPreview::~HoldLabPreview() = default;
void HoldLabPreview::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_controller","path"),&HoldLabPreview::load_controller);
    ClassDB::bind_method(D_METHOD("reset","case_id","holding"),&HoldLabPreview::reset);
    ClassDB::bind_method(D_METHOD("start"),&HoldLabPreview::start);
    ClassDB::bind_method(D_METHOD("pause"),&HoldLabPreview::pause);
    ClassDB::bind_method(D_METHOD("step"),&HoldLabPreview::step);
    ClassDB::bind_method(D_METHOD("get_state"),&HoldLabPreview::get_state);
    ClassDB::bind_method(D_METHOD("get_geometry"),&HoldLabPreview::get_geometry);
    ClassDB::bind_method(D_METHOD("get_error"),&HoldLabPreview::get_error);
}
bool HoldLabPreview::load_controller(const String& path) {
    auto& s=*impl_; s.loaded=false; s.paused=true; s.error.clear();
    s.qualified=false;s.lesson_updates=0;s.protocol=0;s.failure_recorded=false;
    s.package_version=0;s.policy_method=0;s.head_parameters={0,0};s.learned_sigma=0;
    try {
        const auto global=ProjectSettings::get_singleton()->globalize_path(path);
        const auto bytes=cp::read(global.utf8().get_data()); cp::Reader r{bytes};
        if(r.value<uint32_t>()!=0x31564c48) throw std::runtime_error("Wrong research motor package magic");
        const auto version=r.value<uint32_t>();
        if(version!=1 && version!=2) throw std::runtime_error("Unsupported research motor package version");
        if(r.value<uint32_t>()!=model::kVersion || r.value<uint32_t>()!=sm::kTaskVersion ||
           r.value<uint32_t>()!=cudarl::joint_motor::kActionVersion || r.value<uint32_t>()!=4 ||
           r.value<uint32_t>()!=41 || r.value<uint32_t>()!=4 || r.value<uint32_t>()!=336)
            throw std::runtime_error("Incompatible research motor package");
        const auto method=version==2?r.value<uint32_t>():0u;
        if(version==2 && method!=1) throw std::runtime_error("Unsupported learned motor method");
        const auto updates=r.value<uint32_t>(),qualified=r.value<uint32_t>();
        if(updates>32 || qualified>1) throw std::runtime_error("Invalid lesson metadata");
        const auto protocol=r.value<uint64_t>();
        if(protocol==0) throw std::runtime_error("Missing research protocol identity");
        const auto parent=r.array<uint8_t>(64);
        if(std::string(parent.begin(),parent.end())!=canonical) throw std::runtime_error("Wrong preserved parent");
        const auto contracts=cudarl::motor::FrozenController::source_contract_hashes();
        if(r.value<uint32_t>()!=contracts.size()) throw std::runtime_error("Source contract count mismatch");
        for(auto expected:contracts) {
            const auto text=r.array<uint8_t>(64);
            if(std::string(text.begin(),text.end())!=expected) throw std::runtime_error("Research motor physics/source mismatch");
        }
        auto parameters=r.array<float>(2532);std::array<float,4> gains{};std::array<float,2> head{};float sigma=0;
        if(version==1){
            const auto old_gains=r.array<float>(4);
            if(old_gains.size()!=4)throw std::runtime_error("Wrong legacy motor parameter shape");
            std::copy(old_gains.begin(),old_gains.end(),gains.begin());
        }else{
            const auto raw=r.array<float>(2);const float stored_gain=r.value<float>(),stored_sigma=r.value<float>();
            if(raw.size()!=2 || !std::isfinite(raw[0]) || !std::isfinite(raw[1]))throw std::runtime_error("Invalid learned amplitude parameters");
            std::copy(raw.begin(),raw.end(),head.begin());
            const float gain=cudarl::learned_amplitude::sigmoid(head[0]);
            sigma=cudarl::learned_amplitude::standard_deviation(head[1]);
            if(!std::isfinite(stored_gain) || !std::isfinite(stored_sigma) || std::abs(stored_gain-gain)>2e-7f ||
               std::abs(stored_sigma-sigma)>2e-6f || gain<0 || gain>1 || sigma<.05f || sigma>8.f)
                throw std::runtime_error("Learned amplitude mapping does not match its saved parameters");
            gains.fill(gain);
        }
        r.finish();
        if(parameters.size()!=2532) throw std::runtime_error("Wrong motor parameter shape");
        for(float x:parameters) if(!std::isfinite(x)) throw std::runtime_error("Nonfinite backbone weight");
        for(float x:gains) if(!std::isfinite(x)) throw std::runtime_error("Nonfinite learned gain");
        s.actor.params()=std::move(parameters);std::copy(gains.begin(),gains.end(),s.gains.begin());
        s.package_version=version;s.policy_method=method;s.head_parameters=head;s.learned_sigma=sigma;
        s.lesson_updates=static_cast<int>(updates);s.qualified=qualified!=0;s.protocol=protocol;s.loaded=true;
        if(!reset(0,true)) {s.loaded=false;return false;}
        return true;
    } catch(const std::exception& e) {s.error=e.what();return false;}
}
bool HoldLabPreview::reset(int id,bool holding) {
    auto& s=*impl_;s.paused=true;
    if(!s.loaded) {s.error="Load a research motor package first";return false;}
    if(id<0 || id>(holding?5:3)) {s.error="Case outside the recorded lesson";return false;}
    s.error.clear();s.case_id=id;s.holding=holding;s.controls=s.substeps=s.attempted_controls=0;s.failure_recorded=false;
    s.world=sm::initial_world(holding?(id%2?3:0):id);s.world.arena=sm::long_swim_arena();
    s.config=sm::config_for_speed(.008f);s.config.observation_schema=4;
    s.config.maximum_controls=holding?1280:160;
    s.config.target_speed_m_s=holding?0.f:.008f;
    s.config.stationary_request=holding;s.config.position_hold=holding;s.task={};
    if(!sm::begin_forward(s.world,s.config,s.task) ||
       (holding && (!sm::offset_episode_start(s.world,s.task,s.config,id/2==1?-.0005f:id/2==2?.0005f:0.f) ||
                    !sm::update_velocity_request(s.world,s.config,s.task)))) {
        s.error="Native preview reset rejected";return false;
    }
    s.hold_score=sm::RestHoldAssessment{};
    if(holding && !s.hold_score.record(0,sm::measure(s.task,s.world))) {s.error="Invalid holding reset";return false;}
    return true;
}
bool HoldLabPreview::start() {
    auto& s=*impl_;if(!s.loaded || !s.error.empty() || s.controls>=static_cast<uint64_t>(s.config.maximum_controls)) return false;
    s.paused=false;return true;
}
void HoldLabPreview::pause() {impl_->paused=true;}
bool HoldLabPreview::step() {
    auto& s=*impl_;if(!s.loaded || s.paused || !s.error.empty() || s.controls>=static_cast<uint64_t>(s.config.maximum_controls)) return false;
    if(!sm::in_domain(s.world,s.config) || (s.holding && !sm::update_velocity_request(s.world,s.config,s.task))) {
        s.error="Native preview domain/request stop";s.paused=true;return false;
    }
    float observation[41]{};
    if(!sm::observe_appended(s.task,s.world,s.config,observation)) {s.error="Invalid native observation";s.paused=true;return false;}
    s.actor.forward(observation,1);float means[4];
    for(int j=0;j<4;++j) means[j]=s.holding?s.gains[j]*s.actor.output()[j]:s.actor.output()[j];
    sm::Action action;
    if(!sm::action_from_latent(means,action)){s.error="Invalid learned joint command";s.paused=true;return false;}
    const auto before=s.world.attempted_substeps;
    s.failed_before=s.failed_candidate=s.world.physical.body;
    ++s.attempted_controls;
    const bool advanced=cudarl::joint_motor::step_state(s.world,action.command,nullptr,&s.failed_before,&s.failed_candidate);
    s.substeps+=s.world.attempted_substeps-before;
    if(!advanced || s.world.attempted_substeps-before!=10) {s.failure_recorded=true;s.error="Native physical stop: "+std::to_string(s.world.failure_code);s.paused=true;return false;}
    ++s.controls;const auto transition=s.holding?sm::record_hold_control(s.task,s.world,s.config):sm::record_control(s.task,s.world,s.config);
    if(!transition.accepted || (s.holding && !s.hold_score.record(static_cast<int>(s.controls),transition.metrics))) {
        s.error="Native preview interval rejected";s.paused=true;return false;
    }
    if(s.controls>=static_cast<uint64_t>(s.config.maximum_controls)) s.paused=true;
    return true;
}
String HoldLabPreview::get_error() const {return String(impl_->error.c_str());}
Dictionary HoldLabPreview::get_geometry() const {
    Dictionary out;out["model"]=6;out["scale_m"]=p::scale;out["rod_radius_m"]=model::Geometry::rod_radius;
    out["fin_extension_m"]=model::Geometry::fin_extension;out["fin_thickness_m"]=model::Geometry::fin_thickness;
    out["arena_width_m"]=1.2f;out["arena_depth_m"]=.8f;out["floor_y_m"]=cudarl::contact::bounds_min_model().y*p::scale;
    Array links;for(int i=0;i<3;++i) {const auto g=model::body_geometry(i);Dictionary d;d["length"]=g.length;d["half_height"]=g.half_height;d["half_width"]=g.half_width;links.append(d);}out["links"]=links;return out;
}
Dictionary HoldLabPreview::get_state() const {
    const auto& s=*impl_;const auto& body=s.world.physical.body;const auto z=sm::measure(s.task,s.world);Dictionary out;
    out["loaded"]=s.loaded;out["paused"]=s.paused;out["finished"]=s.loaded && s.controls>=static_cast<uint64_t>(s.config.maximum_controls);
    out["training"]=false;out["holding"]=s.holding;out["qualified_candidate"]=s.qualified;out["error"]=get_error();out["case"]=s.case_id;
    out["model"]=6;out["controls"]=static_cast<int64_t>(s.controls);out["substeps"]=static_cast<int64_t>(s.substeps);out["time_s"]=s.controls*p::dt;
    out["attempted_controls"]=static_cast<int64_t>(s.attempted_controls);out["failure_recorded"]=s.failure_recorded;
    out["live_physics"]=true;out["stochastic_exploration"]=false;out["runtime_adopted"]=false;
    if(s.failure_recorded){Dictionary first,candidate;
        first["position"]=point(s.failed_before.position);first["orientation"]=Quaternion(s.failed_before.orientation.x,s.failed_before.orientation.y,s.failed_before.orientation.z,s.failed_before.orientation.w);
        candidate["position"]=point(s.failed_candidate.position);candidate["orientation"]=Quaternion(s.failed_candidate.orientation.x,s.failed_candidate.orientation.y,s.failed_candidate.orientation.z,s.failed_candidate.orientation.w);
        for(int j=0;j<4;++j){first[String("j")+String::num_int64(j)]=s.failed_before.angle[j];candidate[String("j")+String::num_int64(j)]=s.failed_candidate.angle[j];}
        out["failure_before"]=first;out["failure_candidate"]=candidate;}
    out["training_update"]=336;out["holding_updates"]=s.lesson_updates;out["observation_schema"]=4;out["action_mapping"]=cudarl::joint_motor::kActionVersion;
    out["package_version"]=s.package_version;out["policy_method"]=String(s.policy_method==1?"learned_persistent_amplitude":s.loaded?"four_axis_hold_gains":"unloaded");
    out["holding_parameter_count"]=s.policy_method==1?2:4;out["backbone_frozen"]=true;
    if(s.policy_method==1){out["policy_mu"]=s.head_parameters[0];out["policy_variance_coordinate"]=s.head_parameters[1];
        out["policy_sigma"]=s.learned_sigma;out["deterministic_amplitude"]=s.gains[0];}
    out["maximum_controls"]=s.config.maximum_controls;out["duration_s"]=static_cast<double>(s.config.maximum_controls)*.05;
    out["target_speed_m_s"]=s.holding?0.f:.008f;out["speed_m_s"]=p::norm(z.velocity_m_s);out["depth_m"]=z.depth_m;out["lateral_m"]=z.lateral_m;
    out["anchor_error_m"]=s.holding?p::norm(z.hold_error_m):0.f;out["tilt_deg"]=z.tilt_deg;out["forward_m"]=z.forward_m;
    out["anchor_position"]=point(s.holding?s.task.hold_anchor_model:s.task.start_com_model);out["com_position"]=point(cudarl::motor::motion(body).com_model);
    out["head_x"]=body.position.x;out["head_y"]=body.position.y;out["head_z"]=body.position.z;
    out["qx"]=body.orientation.x;out["qy"]=body.orientation.y;out["qz"]=body.orientation.z;out["qw"]=body.orientation.w;
    out["passed"]=s.holding?s.hold_score.passed():s.task.outcome==sm::Outcome::Success;
    for(int j=0;j<4;++j) {out[String("j")+String::num_int64(j)]=body.angle[j];out[String("gain")+String::num_int64(j)]=s.gains[j];}
    Array links;for(const auto& link:model::shape(body).link) {Dictionary d;d["position"]=point(link.center/p::scale);d["orientation"]=Quaternion(link.q.x,link.q.y,link.q.z,link.q.w);links.append(d);}out["links"]=links;
    return out;
}
}
