#include "motor_control_lab.h"
#include "motor_benchmark.h"
#include "rl/boulder_source.h"
#include "rl/command_motor_policy.h"
#include "rl/paired_motor_policy.h"
#include <filesystem>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/quaternion.hpp>

namespace godot {
namespace {
namespace bench=cudarl::motor_benchmark;
namespace pm=cudarl::paired_motor;
using cudarl::deep::V;
Vector3 vector(V p) {return {p.x,p.y,p.z};}
std::string file_path(const String& path) {
    return ProjectSettings::get_singleton()->globalize_path(path).utf8().get_data();
}
std::vector<float> calibrated_actor(const std::string& path,uint64_t original) {
    const auto bytes=cudarl::checkpoint::read(path);cudarl::checkpoint::Reader file{bytes};
    if(file.value<uint32_t>()!=0x534D4631u||file.value<uint32_t>()!=1u||file.value<uint64_t>()!=original)
        throw std::runtime_error("Motor benchmark needs the matching immutable calibrated source");
    const auto parent=file.array<uint8_t>(128000000);const auto training=file.array<uint8_t>(256000000);file.finish();
    if(cudarl::rl::read_boulder_source(parent).fingerprint!=original) throw std::runtime_error("Motor benchmark parent mismatch");
    cudarl::checkpoint::Reader actor{training};
    if(actor.value<uint32_t>()!=0x534D5452u||actor.value<uint32_t>()!=5u||actor.value<uint32_t>()!=sizeof(cudarl::shelter_motor::State))
        throw std::runtime_error("Motor benchmark requires the protocol-5 source");
    (void)actor.value<int>();(void)actor.value<uint32_t>();(void)actor.value<int>();
    const auto scales=actor.value<cudarl::shelter_motor::ActuatorScale>();
    for(float value:scales.value) if(!std::isfinite(value)||value<1) throw std::runtime_error("Invalid saved actuator units");
    auto result=actor.array<float>(cudarl::rl::swimmer_specification(4,59).num_params());
    for(float value:result) if(!std::isfinite(value)) throw std::runtime_error("Invalid frozen actor");return result;
}
const char* controller_name(int id) {
    switch(id) {
    case 0:return "Passive zero torque";case 1:return "Conventional reference";
    case 2:return "Original saved swimmer";case 3:return "Calibrated saved motor";
    case 4:return "Legacy command motor experiment";
    case 5:return "Paired PPO · direct torque";
    default:return "Paired PPO · joint target";
    }
}
bench::Suite suite_id(int value) {
    if(value<0||value>2) throw std::invalid_argument("Unknown motor benchmark suite");return static_cast<bench::Suite>(value);
}
uint32_t suite_seed(bench::Suite value) {
    return value==bench::Suite::Reference?bench::reference_seed:(value==bench::Suite::Development?bench::development_seed:bench::final_seed);
}
}
struct MotorControlLab::Impl {
    cudarl::rl::BoulderSource original;
    std::vector<float> calibrated;
    cudarl::rl::CommandMotorPolicy candidate;
    cudarl::rl::PairedMotorPolicy paired[2];
    bool paired_ready[2]{};
    String paired_error[2],paired_path[2];
    pm::State paired_state;
    bool ready=false,candidate_ready=false,started=false;
    int controller=0;
    bench::Fixture fixture;
    bench::State state;
    bench::Sample sample;
    bench::Progress progress;
    bench::Four last{};
    bench::ConventionalReference conventional;
    std::unique_ptr<bench::FrozenMotor> actor;
};
MotorControlLab::MotorControlLab():impl_(std::make_unique<Impl>()) {}
MotorControlLab::~MotorControlLab()=default;
void MotorControlLab::_bind_methods() {
    ClassDB::bind_method(D_METHOD("configure","original","calibrated","candidate","paired_root"),&MotorControlLab::configure,DEFVAL(""),DEFVAL(""));
    ClassDB::bind_method(D_METHOD("start_trial","suite","index","controller"),&MotorControlLab::start_trial);
    ClassDB::bind_method(D_METHOD("step"),&MotorControlLab::step);
    ClassDB::bind_method(D_METHOD("get_state"),&MotorControlLab::get_state);
    ClassDB::bind_method(D_METHOD("get_cases","suite"),&MotorControlLab::get_cases);
    ClassDB::bind_method(D_METHOD("has_candidate"),&MotorControlLab::has_candidate);
    ClassDB::bind_method(D_METHOD("get_controller_info","controller"),&MotorControlLab::get_controller_info);
    ClassDB::bind_method(D_METHOD("get_error"),&MotorControlLab::get_error);
}
bool MotorControlLab::configure(String original,String calibrated,String candidate,String paired_root) {
    error_="";impl_->started=false;impl_->ready=false;impl_->candidate_ready=false;
    for(int arm=0;arm<2;++arm) {impl_->paired_ready[arm]=false;impl_->paired_error[arm]="No saved trained policy for this paired arm";impl_->paired_path[arm]="";}
    try {
        impl_->original=cudarl::rl::load_boulder_source(file_path(original));
        impl_->calibrated=calibrated_actor(file_path(calibrated),impl_->original.fingerprint);
        if(!candidate.is_empty()&&std::filesystem::is_regular_file(file_path(candidate))) {
            impl_->candidate=cudarl::rl::CommandMotorPolicy::load(file_path(candidate));
            if(impl_->candidate.source_fingerprint!=impl_->original.fingerprint) throw std::runtime_error("Candidate source differs from frozen comparison source");
            impl_->candidate_ready=true;
        }
        if(!paired_root.is_empty()) for(int arm=0;arm<2;++arm) {
            const auto path=std::filesystem::path(file_path(paired_root))/(arm==0?"torque":"joint_target")/"candidate-policy.bin";
            impl_->paired_path[arm]=String(path.string().c_str());
            if(!std::filesystem::is_regular_file(path)) continue;
            try {
                auto policy=cudarl::rl::PairedMotorPolicy::load(path.string());
                const auto expected=arm==0?pm::Interface::Torque:pm::Interface::JointTarget;
                if(policy.interface_type!=expected) throw std::runtime_error("Paired policy arm does not match its folder/interface");
                if(policy.source_fingerprint!=impl_->original.fingerprint) throw std::runtime_error("Paired policy source differs from the frozen comparison source");
                if(!policy.has_trained_weights()) throw std::runtime_error("Paired policy has zero training updates; it is not a learned result");
                impl_->paired[arm]=std::move(policy);impl_->paired_ready[arm]=true;impl_->paired_error[arm]="";
            } catch(const std::exception& e) {impl_->paired_error[arm]=String(e.what());}
        }
        if(impl_->paired_ready[0]&&impl_->paired_ready[1]&&impl_->paired[0].regulation_gate!=impl_->paired[1].regulation_gate) {
            for(int arm=0;arm<2;++arm) {impl_->paired_ready[arm]=false;impl_->paired_error[arm]="Paired arms do not share the exact same source/catalog/gain preflight";}
        }
        impl_->ready=true;return true;
    } catch(const std::exception& e) {error_=String(e.what());return false;}
}
bool MotorControlLab::has_candidate() const {return impl_->candidate_ready;}
Dictionary MotorControlLab::get_controller_info(int controller) const {
    Dictionary out;out["id"]=controller;out["name"]=String(controller_name(controller));
    const bool paired=controller==5||controller==6;
    bool available=impl_->ready&&controller>=0&&controller<=3;
    if(controller==4) available=impl_->ready&&impl_->candidate_ready;
    if(paired) available=impl_->ready&&impl_->paired_ready[controller-5];
    out["available"]=available;out["paired"]=paired;out["learning_updates"]=0;
    if(paired) {
        const int arm=controller-5;out["error"]=impl_->paired_error[arm];out["path"]=impl_->paired_path[arm];out["budget"]=64;
        if(available) {
            const auto& policy=impl_->paired[arm];out["update"]=policy.update;out["training_complete"]=policy.update==64;
            out["passed_final_gate"]=policy.passed_final_gate;out["source_fingerprint"]=String::num_uint64(policy.source_fingerprint);
            out["environment_fingerprint"]=String::num_uint64(policy.environment_fingerprint);
            out["interface"]=arm==0?"torque":"joint_target";
            PackedFloat32Array kp,kd;for(int j=0;j<4;++j) {kp.append(policy.gains.kp[j]);kd.append(policy.gains.kd[j]);}out["kp_Nm_per_rad"]=kp;out["kd_Nm_per_rad_s"]=kd;
        }
    }
    return out;
}
Array MotorControlLab::get_cases(int suite) const {
    Array result;
    try {
        const auto group=suite_id(suite);
        for(int id=0;id<bench::suite_count(group);++id) {
            const auto fixture=bench::make_fixture(suite_seed(group),group,id);Dictionary item;
            item["id"]=id;item["kind"]=String(bench::name(fixture.kind));item["name"]=String::num_int64(id+1)+" - "+String(bench::name(fixture.kind));result.append(item);
        }
    } catch(const std::exception&) {}
    return result;
}
bool MotorControlLab::start_trial(int suite,int index,int controller) {
    error_="";impl_->started=false;
    try {
        if(!impl_->ready||controller<0||controller>6) throw std::runtime_error("Motor benchmark sources are not ready");
        if(controller==4&&!impl_->candidate_ready) throw std::runtime_error("No experimental motor was trained; the original pond is preserved");
        if(controller>=5&&!impl_->paired_ready[controller-5]) throw std::runtime_error(impl_->paired_error[controller-5].utf8().get_data());
        const auto group=suite_id(suite);impl_->fixture=bench::make_fixture(suite_seed(group),group,index);
        if(impl_->fixture.needs_warmup) {
            bench::FrozenMotor capture(impl_->original.motor,bench::TargetMode::Original55);
            if(!bench::prepare_braking(impl_->fixture,capture)) throw std::runtime_error("The saved swimmer could not supply this braking start. No velocity was injected. Choose another trial.");
        }
        impl_->controller=controller;impl_->actor.reset();
        if(controller==2) impl_->actor=std::make_unique<bench::FrozenMotor>(impl_->original.motor,bench::TargetMode::Original55);
        if(controller==3) impl_->actor=std::make_unique<bench::FrozenMotor>(impl_->calibrated,bench::TargetMode::Calibrated59);
        if(controller==4) impl_->actor=std::make_unique<bench::FrozenMotor>(impl_->candidate.actor,bench::TargetMode::Candidate59);
        if(controller>=5) impl_->actor=std::make_unique<bench::FrozenMotor>(impl_->paired[controller-5].actor,bench::TargetMode::Candidate59);
        impl_->state=impl_->fixture.initial;impl_->paired_state={};impl_->paired_state.task=impl_->state;
        impl_->progress={};impl_->last={};impl_->sample=bench::measure(impl_->state);
        bench::assess(impl_->progress,impl_->state,impl_->sample,impl_->fixture);impl_->started=true;return true;
    } catch(const std::exception& e) {error_=String(e.what());return false;}
}
Dictionary MotorControlLab::step() {
    if(!impl_->started||impl_->state.steps>=impl_->fixture.evaluation_ticks||!impl_->sample.finite) return get_state();
    if(impl_->controller==0) impl_->last={};
    else if(impl_->controller==1) impl_->last=impl_->conventional.action(impl_->state);
    else impl_->last=impl_->actor->action(impl_->state);
    cudarl::command_motor::Result result;
    if(impl_->controller>=5) {
        const auto& policy=impl_->paired[impl_->controller-5];
        result=pm::advance(impl_->paired_state,impl_->last.data(),policy.interface_type,policy.gains,bench::physics_for(impl_->fixture,impl_->state.steps));
        impl_->state=impl_->paired_state.task;
    } else result=bench::advance(impl_->state,impl_->last.data(),impl_->fixture);
    if(result.unsafe) {impl_->progress.penetrated=true;if(impl_->progress.first_failure_step<0) impl_->progress.first_failure_step=impl_->state.steps;}
    impl_->sample=bench::measure(impl_->state);if(result.invalid) {impl_->sample.finite=false;error_="Physical trial stopped: nonfinite state or command";}
    bench::assess(impl_->progress,impl_->state,impl_->sample,impl_->fixture);return get_state();
}
Dictionary MotorControlLab::get_state() const {
    Dictionary out;if(!impl_->started) return out;
    const auto& s=impl_->state;const auto& m=impl_->sample;const auto& p=impl_->progress;
    const auto status=bench::finish(p,impl_->fixture);const bool complete=s.steps>=impl_->fixture.evaluation_ticks||!m.finite;
    out["tick"]=s.steps;out["elapsed"]=s.steps*cudarl::deep::dt;out["complete"]=complete;out["finite"]=m.finite;
    out["controller"]=String(controller_name(impl_->controller));out["controller_id"]=impl_->controller;
    out["experimental"]=impl_->controller>=4;out["qualified_candidate"]=impl_->controller==4&&impl_->candidate.passed_final_gate;
    out["paired"]=impl_->controller>=5;
    if(impl_->controller>=5) {
        const auto& policy=impl_->paired[impl_->controller-5];out["qualified_candidate"]=policy.passed_final_gate;
        out["policy_update"]=policy.update;out["policy_budget"]=64;out["training_complete"]=policy.update==64;
        out["interface"]=impl_->controller==5?"torque":"joint_target";
        out["source_fingerprint"]=String::num_uint64(policy.source_fingerprint);out["environment_fingerprint"]=String::num_uint64(policy.environment_fingerprint);
        PackedFloat32Array target,actual,requested;
        if(impl_->controller==6) for(float value:impl_->paired_state.held_target) target.append(value);
        for(int j=0;j<4;++j) {actual.append(impl_->paired_state.actual_torques[36+j]*10.f);requested.append(impl_->paired_state.unclamped_torques[36+j]*1e6f);}
        out["joint_targets_rad"]=target;out["actual_torque_uNm"]=actual;
        out["requested_torque_uNm"]=requested;
        out["joint_limit_rad"]=cudarl::deep::limit;
        out["torque_rms_uNm"]=std::sqrt(impl_->paired_state.mean_squared_torque)*10.f;
    }
    out["kind"]=String(bench::name(impl_->fixture.kind));out["case_id"]=impl_->fixture.id;out["passed_trial"]=status.passed;
    out["position"]=vector(s.body.position);out["goal"]=vector(s.request.position);out["heading"]=vector(s.request.heading);
    out["speed_mm_s"]=m.head_speed*1000;out["com_speed_mm_s"]=m.com_speed*1000;out["body_speed_mm_s"]=m.point_speed*1000;
    out["goal_error_mm"]=m.goal_error*1000;out["heading_error_deg"]=m.heading_error*57.295779513f;
    out["angular_speed"]=m.angular_speed;out["joint_rate"]=m.joint_rate;out["clearance_mm"]=m.clearance*1000;
    out["safety_failed"]=p.penetrated||!m.finite;
    out["best_hold_seconds"]=p.hold_seconds;out["path_mm"]=p.path*1000;out["entry_speed_mm_s"]=impl_->fixture.actual_entry_speed*1000;
    out["warmup_ticks"]=impl_->fixture.warmup_ticks;out["budget_ticks"]=impl_->fixture.evaluation_ticks;
    PackedFloat32Array commands,raw;for(int j=0;j<4;++j) {
        raw.append(impl_->last[static_cast<size_t>(j)]);
        commands.append(impl_->controller>=5?impl_->paired_state.actual_torques[36+j]:cudarl::deep::clamp(impl_->last[static_cast<size_t>(j)],-1,1));
    }out["commands"]=commands;out["raw_actions"]=raw;
    Array links;const auto shape=cudarl::deep::shape(s.body);
    for(int link=0;link<3;++link) {Dictionary item;const auto& l=shape.link[link];item["position"]=vector(l.center/cudarl::deep::scale);
        item["quaternion"]=Quaternion(l.q.x,l.q.y,l.q.z,l.q.w);item["length"]=l.length/cudarl::deep::scale;links.append(item);}
    out["links"]=links;out["learning_updates"]=0;out["physics_backend"]="Native C++ · same physical equations";return out;
}
}
