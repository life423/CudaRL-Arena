#include "motor_v2_lab.h"
#include "motor_v2_benchmark.h"
#include "rl/motor_v2_policy.h"
#include "rl/boulder_source.h"
#include "nn/mlp_cpu.h"
#include <filesystem>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/quaternion.hpp>

namespace godot {
namespace {
namespace mv=cudarl::motor_v2;namespace mb=cudarl::motor_v2_benchmark;
namespace mc=cudarl::motor_contact_v2;namespace fs=std::filesystem;
Vector3 vector(cudarl::deep::V value) {return {value.x,value.y,value.z};}
std::string resolved(const String& path) {return ProjectSettings::get_singleton()->globalize_path(path).utf8().get_data();}
const char* arm_name(int arm) {return arm==0?"Direct-torque PPO":"Joint-target PPO";}
}
struct MotorV2Lab::Impl {
    fs::path root;
    cudarl::rl::MotorV2Policy policy[2];
    bool ready[2]{},catalog_ready=false,started=false;
    String policy_error[2],policy_path[2];
    uint64_t source=0;
    std::vector<mb::Fixture> catalog;
    std::unique_ptr<cudarl::nn::CpuMlp<float>> model;
    mv::State state;
    mb::Fixture fixture;
    mb::Sample sample;
    mb::Progress progress;
    std::array<float,4> raw{};
    int arm=0;
};
MotorV2Lab::MotorV2Lab():impl_(std::make_unique<Impl>()) {}
MotorV2Lab::~MotorV2Lab()=default;
void MotorV2Lab::_bind_methods() {
    ClassDB::bind_method(D_METHOD("configure","study_root"),&MotorV2Lab::configure);
    ClassDB::bind_method(D_METHOD("get_controller_info","arm"),&MotorV2Lab::get_controller_info);
    ClassDB::bind_method(D_METHOD("get_cases"),&MotorV2Lab::get_cases);
    ClassDB::bind_method(D_METHOD("start_trial","case_id","arm"),&MotorV2Lab::start_trial);
    ClassDB::bind_method(D_METHOD("step"),&MotorV2Lab::step);
    ClassDB::bind_method(D_METHOD("get_state"),&MotorV2Lab::get_state);
    ClassDB::bind_method(D_METHOD("get_error"),&MotorV2Lab::get_error);
}
bool MotorV2Lab::configure(String study_root) {
    error_="";impl_->started=false;impl_->catalog_ready=false;impl_->catalog.clear();impl_->source=0;
    for(int arm=0;arm<2;++arm) {impl_->ready[arm]=false;impl_->policy_error[arm]="No saved trained motor-v2 policy";}
    try {
        impl_->root=fs::path(resolved(study_root));
        const auto source_path=impl_->root/"sources"/"source-boulder.bin";
        if(fs::is_regular_file(source_path)) impl_->source=cudarl::rl::load_boulder_source(source_path.string()).fingerprint;
        const auto catalog_path=impl_->root/"sources"/"final-fixtures.bin";
        if(fs::is_regular_file(catalog_path)) {
            const auto bytes=cudarl::checkpoint::read(catalog_path.string());cudarl::checkpoint::Reader in{bytes};
            if(in.value<uint32_t>()!=0x4D563246u||in.value<uint32_t>()!=2||in.value<uint32_t>()!=sizeof(mb::Fixture)||
               in.value<uint64_t>()!=impl_->source||in.value<uint64_t>()!=mv::physics_fingerprint()||in.value<uint64_t>()!=mb::rules_fingerprint())
                throw std::runtime_error("Motor-v2 catalog source/physics/task binding differs");
            impl_->catalog=in.array<mb::Fixture>(64);in.finish();
            if(impl_->catalog.size()!=64) throw std::runtime_error("Motor-v2 catalog must contain exactly64 frozen cases");
            for(const auto& fixture:impl_->catalog) if(!mv::finite(fixture.initial)||fixture.initial.task.steps!=0)
                throw std::runtime_error("Invalid motor-v2 initial state");
            impl_->catalog_ready=true;
        }
        for(int arm=0;arm<2;++arm) {
            const auto path=impl_->root/"run"/(arm==0?"torque":"joint_target")/"candidate-policy.bin";
            impl_->policy_path[arm]=String(path.string().c_str());if(!fs::is_regular_file(path)) continue;
            try {
                auto policy=cudarl::rl::MotorV2Policy::load(path.string());
                const auto expected=arm==0?mv::Interface::Torque:mv::Interface::JointTarget;
                if(policy.interface_type!=expected||!policy.has_trained_weights()||policy.source_fingerprint!=impl_->source)
                    throw std::runtime_error("Motor-v2 actor must match its arm/source and have real PPO updates");
                if(!impl_->catalog_ready) throw std::runtime_error("No validated frozen motor-v2 catalog is available");
                impl_->policy[arm]=std::move(policy);impl_->ready[arm]=true;impl_->policy_error[arm]="";
            } catch(const std::exception& error) {impl_->policy_error[arm]=String(error.what());}
        }
        if(impl_->ready[0]&&impl_->ready[1]&&
           (impl_->policy[0].contact_gate!=impl_->policy[1].contact_gate||impl_->policy[0].historical_regulation_gate!=impl_->policy[1].historical_regulation_gate))
            throw std::runtime_error("Motor-v2 arms have different physical readiness/gain records");
        return true;
    } catch(const std::exception& error) {
        error_=String(error.what());impl_->ready[0]=impl_->ready[1]=false;return false;
    }
}
Dictionary MotorV2Lab::get_controller_info(int arm) const {
    Dictionary out;out["arm"]=arm;out["physics_version"]=2;out["protocol_version"]=2;out["learning_updates"]=0;
    out["available"]=arm>=0&&arm<2&&impl_->ready[arm];
    if(arm<0||arm>1) {out["error"]="Unknown motor-v2 arm";return out;}
    out["name"]=String(arm_name(arm));out["error"]=impl_->policy_error[arm];out["path"]=impl_->policy_path[arm];out["budget"]=64;
    if(impl_->ready[arm]) {
        const auto& policy=impl_->policy[arm];out["update"]=policy.update;out["completed_budget"]=policy.completed_budget;
        out["passed_final_gate"]=policy.passed_final_gate;out["interface"]=arm==0?"torque":"joint_target";
        out["source_fingerprint"]=String::num_uint64(policy.source_fingerprint);out["environment_fingerprint"]=String::num_uint64(policy.environment_fingerprint);
        out["evaluation_complete"]=fs::is_regular_file(impl_->root/"final-summary.json")&&
            fs::is_regular_file(impl_->root/"run"/(arm==0?"torque":"joint_target")/"final-cases.csv");
    }
    return out;
}
Array MotorV2Lab::get_cases() const {
    Array result;for(const auto& fixture:impl_->catalog) {
        Dictionary entry;entry["id"]=fixture.id;entry["kind"]=String(mb::name(fixture.kind));
        entry["name"]=String::num_int64(fixture.id+1)+" · "+String(mb::name(fixture.kind));result.append(entry);
    }return result;
}
bool MotorV2Lab::start_trial(int case_id,int arm) {
    error_="";impl_->started=false;
    try {
        if(arm<0||arm>1||!impl_->ready[arm]) throw std::runtime_error("No saved trained actor for this new-physics arm");
        if(!impl_->catalog_ready||case_id<0||case_id>=static_cast<int>(impl_->catalog.size())) throw std::runtime_error("Frozen motor-v2 case is unavailable");
        impl_->arm=arm;impl_->fixture=impl_->catalog[static_cast<size_t>(case_id)];impl_->state=impl_->fixture.initial;
        impl_->model=std::make_unique<cudarl::nn::CpuMlp<float>>(cudarl::rl::motor_v2_specification());
        impl_->model->params()=impl_->policy[arm].actor;impl_->raw={};impl_->progress={};
        impl_->sample=mb::measure(impl_->state);mb::assess(impl_->progress,impl_->state,impl_->sample,impl_->fixture);
        impl_->started=true;return true;
    } catch(const std::exception& error) {error_=String(error.what());return false;}
}
Dictionary MotorV2Lab::step() {
    if(!impl_->started||impl_->state.task.steps>=impl_->fixture.steps||!impl_->progress.finite||impl_->progress.unsafe) return get_state();
    float observation[59];mv::observe(impl_->state,observation);impl_->model->forward(observation,1);
    for(int axis=0;axis<4;++axis) impl_->raw[axis]=impl_->model->output()[axis];
    const auto& policy=impl_->policy[impl_->arm];const auto result=mv::advance(impl_->state,impl_->raw.data(),policy.interface_type,policy.gains);
    impl_->sample=mb::measure(impl_->state);
    if(result.invalid) {impl_->sample.finite=false;error_="Motor-v2 trial stopped on nonfinite dynamics or action";}
    mb::assess(impl_->progress,impl_->state,impl_->sample,impl_->fixture);return get_state();
}
Dictionary MotorV2Lab::get_state() const {
    Dictionary out;if(!impl_->started) return out;
    const auto& state=impl_->state;const auto& sample=impl_->sample;const auto& progress=impl_->progress;
    const auto& policy=impl_->policy[impl_->arm];const auto result=mb::finish(progress,impl_->fixture);
    out["arm"]=impl_->arm;out["case_id"]=impl_->fixture.id;out["kind"]=String(mb::name(impl_->fixture.kind));
    out["tick"]=state.task.steps;out["elapsed"]=state.task.steps*cudarl::deep::dt;out["learning_updates"]=0;
    out["physics_version"]=2;out["policy_update"]=policy.update;out["policy_budget"]=64;out["completed_budget"]=policy.completed_budget;
    out["accepted"]=policy.passed_final_gate;out["complete"]=state.task.steps>=impl_->fixture.steps||!progress.finite||progress.unsafe;
    out["finite"]=sample.finite;out["safety_failed"]=progress.unsafe||!progress.finite;out["passed_trial"]=result.passed;
    out["position"]=vector(state.task.body.position);out["goal"]=vector(state.task.request.position);out["heading"]=vector(state.task.request.heading);
    out["water_velocity"]=vector(state.water_velocity);out["goal_error_mm"]=sample.goal_error*1000;
    out["heading_error_deg"]=sample.heading_error*57.295779513f;out["head_mm_s"]=sample.head_speed*1000;out["com_mm_s"]=sample.com_speed*1000;
    out["angular_rad_s"]=sample.angular_speed;out["point_mm_s"]=sample.point_speed*1000;out["joint_rate_rad_s"]=sample.joint_rate;
    out["floor_gap_mm"]=sample.floor_gap*1000;out["other_gap_mm"]=sample.other_gap*1000;out["overlap_limit_mm"]=mc::kOverlapTolerance*1000;
    out["floor_contact"]=sample.floor_contact;out["floor_force_N"]=state.substep_max_floor_force;out["best_hold_seconds"]=progress.hold_seconds;
    out["geometric_hold_seconds"]=progress.geometric_hold_seconds;out["positive_work_J"]=progress.positive_work_J;
    out["torque_rms_uNm"]=sample.torque_rms_uNm;out["supported_task"]=impl_->fixture.kind==mb::Kind::BottomRest;
    PackedFloat32Array raw,targets,actual;
    for(int axis=0;axis<4;++axis) {
        raw.append(impl_->raw[axis]);actual.append(state.task.previous_action[axis]*10.f);
        if(impl_->arm==1) targets.append(state.held_target[axis]);
    }
    out["raw_actions"]=raw;out["joint_targets_rad"]=targets;out["actual_torque_uNm"]=actual;
    Array links;const auto shape=cudarl::deep::shape(state.task.body);
    for(int link=0;link<3;++link) {const auto& physical=shape.link[link];Dictionary entry;
        entry["position"]=vector(physical.center/cudarl::deep::scale);entry["quaternion"]=Quaternion(physical.q.x,physical.q.y,physical.q.z,physical.q.w);
        links.append(entry);
    }out["links"]=links;return out;
}
}
