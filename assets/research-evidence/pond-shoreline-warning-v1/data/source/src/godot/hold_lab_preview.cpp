#include "hold_lab_preview.h"
#include "simulation/pond_runtime.h"
#include "motor/pond_motor_observation.h"
#include "motor/heading_request_observation.h"
#include "motor/saved_motor_contract.h"
#include "nn/mlp_cpu.h"
#include "core/checkpoint.h"
#include "rl/learned_amplitude_math.h"
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/packed_float64_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace godot {
namespace p=cudarl::physics;
namespace model=cudarl::movement;
namespace runtime=cudarl::pond_runtime;
namespace obs=cudarl::pond_motor_observation;
namespace pond=cudarl::pond_geometry_v1;
namespace cp=cudarl::checkpoint;
namespace {
constexpr char canonical[]="810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0";
Vector3 point(p::V value) {return Vector3(value.x,value.y,value.z);}
Dictionary body_record(const p::BodyState& body) {
    Dictionary result;result["position"]=point(body.position);
    result["orientation"]=Quaternion(body.orientation.x,body.orientation.y,body.orientation.z,body.orientation.w);
    Array rates;for(float rate:body.velocity)rates.append(rate);result["velocity"]=rates;
    for(int j=0;j<4;++j)result[String("j")+String::num_int64(j)]=body.angle[j];
    result["work_J"]=body.work;result["travel_model"]=body.travel;return result;
}
std::string physical_stop(int code) {
    if(code==102)return "Paused at the water-surface physics boundary";
    if(code==103)return "Paused at the dry-land physics boundary";
    if(code==104)return "The whole-body terrain query could not be certified";
    if(code==110)return "Shore contact exceeded the physical overlap limit";
    return "Native pond physical check stopped: "+std::to_string(code);
}
}
struct HoldLabPreview::Impl {
    cudarl::nn::CpuMlp<float> actor{{{41,32,32,4},{cudarl::nn::Activation::Tanh,cudarl::nn::Activation::Tanh,cudarl::nn::Activation::Identity}}};
    runtime::World world{};
    obs::RequestState request{};
    std::array<float,2> head{};
    float gain=0,sigma=0;
    uint32_t lesson_updates=0;
    uint64_t protocol=0,controls=0;
    int case_id=0;
    bool heading_only=false;
    float requested_turn_degrees=0;
    bool loaded=false,paused=true,holding=true,untimed=false,qualified=false,failure_recorded=false;
    std::string error;
    uint64_t maximum_controls() const {return untimed?0u:holding?1280u:160u;}
    bool finished() const {return !untimed&&controls>=maximum_controls();}
};
HoldLabPreview::HoldLabPreview():impl_(std::make_unique<Impl>()){}
HoldLabPreview::~HoldLabPreview()=default;
void HoldLabPreview::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_controller","path"),&HoldLabPreview::load_controller);
    ClassDB::bind_method(D_METHOD("reset","case_id","holding"),&HoldLabPreview::reset,DEFVAL(0),DEFVAL(true));
    ClassDB::bind_method(D_METHOD("reset_untimed_swim","case_id"),&HoldLabPreview::reset_untimed_swim,DEFVAL(0));
    ClassDB::bind_method(D_METHOD("request_swim_turn","direction"),&HoldLabPreview::request_swim_turn);
    ClassDB::bind_method(D_METHOD("start"),&HoldLabPreview::start);
    ClassDB::bind_method(D_METHOD("pause"),&HoldLabPreview::pause);
    ClassDB::bind_method(D_METHOD("step"),&HoldLabPreview::step);
    ClassDB::bind_method(D_METHOD("get_state"),&HoldLabPreview::get_state);
    ClassDB::bind_method(D_METHOD("get_geometry"),&HoldLabPreview::get_geometry);
    ClassDB::bind_method(D_METHOD("get_pond_geometry_descriptor"),&HoldLabPreview::get_pond_geometry_descriptor);
    ClassDB::bind_method(D_METHOD("sample_pond_surface","positions_m"),&HoldLabPreview::sample_pond_surface);
    ClassDB::bind_method(D_METHOD("sample_pond_outline","angle_radius"),&HoldLabPreview::sample_pond_outline);
    ClassDB::bind_method(D_METHOD("get_error"),&HoldLabPreview::get_error);
}
bool HoldLabPreview::load_controller(const String& path) {
    auto& s=*impl_;s.loaded=false;s.paused=true;s.qualified=false;s.error.clear();
    s.world.motor.physical.paused=true;s.lesson_updates=0;s.protocol=0;s.head={0,0};s.gain=s.sigma=0;
    try {
        const auto global=ProjectSettings::get_singleton()->globalize_path(path);
        const auto bytes=cp::read(global.utf8().get_data());cp::Reader reader{bytes};
        if(reader.value<uint32_t>()!=0x31564c48)throw std::runtime_error("Wrong saved motor package magic");
        if(reader.value<uint32_t>()!=cudarl::motor::saved_motor_package_version)
            throw std::runtime_error("Obsolete motor package; this app requires saved amplitude package 2");
        if(reader.value<uint32_t>()!=6 || reader.value<uint32_t>()!=6 ||
           reader.value<uint32_t>()!=cudarl::joint_motor::kActionVersion || reader.value<uint32_t>()!=4 ||
           reader.value<uint32_t>()!=41 || reader.value<uint32_t>()!=4 || reader.value<uint32_t>()!=336)
            throw std::runtime_error("Incompatible saved motor shape or observation contract");
        if(reader.value<uint32_t>()!=1)throw std::runtime_error("Unsupported saved motor method; learned persistent amplitude is required");
        const auto updates=reader.value<uint32_t>(),qualified=reader.value<uint32_t>();
        if(updates!=7 || qualified!=1)throw std::runtime_error("This app requires the qualified seven-update saved amplitude motor");
        const auto protocol=reader.value<uint64_t>();
        if(protocol==0)throw std::runtime_error("Missing saved research protocol identity");
        const auto parent=reader.array<uint8_t>(64);
        if(std::string(parent.begin(),parent.end())!=canonical)throw std::runtime_error("Wrong preserved parent checkpoint");
        const auto& contracts=cudarl::motor::saved_motor_source_contract_hashes();
        if(reader.value<uint32_t>()!=contracts.size())throw std::runtime_error("Saved source provenance count mismatch");
        for(auto expected:contracts) {
            const auto stored=reader.array<uint8_t>(64);
            if(std::string(stored.begin(),stored.end())!=expected)throw std::runtime_error("Saved motor source provenance mismatch");
        }
        auto parameters=reader.array<float>(2532);
        const auto raw_head=reader.array<float>(2);
        const float stored_gain=reader.value<float>(),stored_sigma=reader.value<float>();
        reader.finish();
        if(parameters.size()!=2532 || raw_head.size()!=2)throw std::runtime_error("Wrong saved motor parameter shape");
        for(float value:parameters)if(!std::isfinite(value))throw std::runtime_error("Nonfinite backbone weight");
        for(float value:raw_head)if(!std::isfinite(value))throw std::runtime_error("Nonfinite learned amplitude parameter");
        const float gain=cudarl::learned_amplitude::sigmoid(raw_head[0]);
        const float sigma=cudarl::learned_amplitude::standard_deviation(raw_head[1]);
        if(!std::isfinite(stored_gain)||!std::isfinite(stored_sigma)||std::abs(stored_gain-gain)>2e-7f||
           std::abs(stored_sigma-sigma)>2e-6f||gain<0||gain>1||sigma<.05f||sigma>8.f)
            throw std::runtime_error("Learned amplitude mapping does not match its saved parameters");
        s.actor.params()=std::move(parameters);s.head={raw_head[0],raw_head[1]};s.gain=gain;s.sigma=sigma;
        s.lesson_updates=updates;s.qualified=qualified!=0;s.protocol=protocol;s.loaded=true;
        if(!reset(0,true)){s.loaded=false;return false;}
        return true;
    }catch(const std::exception& error){s.error=error.what();return false;}
}
bool HoldLabPreview::reset(int id,bool holding) {
    auto& s=*impl_;s.paused=true;s.world.motor.physical.paused=true;
    if(!s.loaded){s.error="Load the saved motor first";return false;}
    runtime::World next;obs::RequestState request;
    auto& physical=next.motor.physical;
    if(!obs::initialize(id,holding,physical.body,physical.held_targets,next.motor.reference_rate,
                        physical.last_actual_torques,request)) {
        s.error="Case outside the saved starting fixtures";return false;
    }
    physical.paused=false;
    if(!runtime::prepare(next)){s.error="Native pond reset could not certify the starting body";return false;}
    physical.paused=true;
    s.world=next;s.request=request;s.case_id=id;s.holding=holding;s.untimed=false;
    s.controls=0;s.heading_only=false;s.requested_turn_degrees=0;s.failure_recorded=false;s.error.clear();return true;
}
bool HoldLabPreview::reset_untimed_swim(int id) {
    if(!reset(id,false))return false;impl_->untimed=true;return true;
}
bool HoldLabPreview::request_swim_turn(int direction) {
    auto& s=*impl_;
    if(!s.loaded||s.holding||!s.error.empty()||(direction!=1&&direction!=-1))return false;
    const float degrees=5.f*static_cast<float>(direction);
    const auto rotation=p::axis_rotation({0,1,0},degrees*obs::pi/180.f);
    auto& request=s.request.request;
    request.heading=rotation.rotate(request.heading);
    request.terminal_velocity=rotation.rotate(request.terminal_velocity);
    request.position=s.request.start_com_model+rotation.rotate(request.position-s.request.start_com_model);
    s.heading_only=true;s.requested_turn_degrees+=degrees;
    return true;
}
bool HoldLabPreview::start() {
    auto& s=*impl_;
    if(!s.loaded||!s.error.empty()||s.finished())return false;
    s.paused=false;s.world.motor.physical.paused=false;return true;
}
void HoldLabPreview::pause() {impl_->paused=true;impl_->world.motor.physical.paused=true;}
bool HoldLabPreview::step() {
    auto& s=*impl_;if(!s.loaded||s.paused||!s.error.empty()||s.finished())return false;
    if(s.controls==std::numeric_limits<uint64_t>::max()){s.error="Swimming paused at the session counter limit";pause();return false;}
    auto& physical=s.world.motor.physical;
    float observation[obs::observations]{};
    const bool observed=s.heading_only?
        cudarl::heading_request_observation::observe(s.request,physical.body,physical.held_targets,s.world.motor.reference_rate,physical.last_actual_torques,observation):
        obs::observe(s.request,physical.body,physical.held_targets,s.world.motor.reference_rate,physical.last_actual_torques,observation);
    if(!observed) {
        s.error="Invalid saved motor observation";pause();return false;
    }
    s.actor.forward(observation,1);
    float command[4]{};const float amplitude=s.holding?s.gain:1.f;
    for(int j=0;j<4;++j) {
        const float latent=amplitude*s.actor.output()[j];
        if(!std::isfinite(latent)){s.error="Invalid learned joint request";pause();return false;}
        command[j]=tanhf(latent);
    }
    if(!runtime::step(s.world,command)) {
        s.failure_recorded=true;s.error=physical_stop(s.world.failure);pause();return false;
    }
    ++s.controls;++s.request.controls;
    if(s.finished())pause();
    return true;
}
String HoldLabPreview::get_error() const {return String(impl_->error.c_str());}
Dictionary HoldLabPreview::get_pond_geometry_descriptor() const {
    const auto& terrain=impl_->world.terrain;Dictionary out;
    out["schema"]=1;out["version"]=pond::version;out["runtime_version"]=runtime::version;out["units"]="metres";
    out["collision_envelope_version"]=cudarl::pond_body_geometry::collision_envelope_version;
    out["query_arithmetic_version"]=cudarl::pond_body_geometry::query_arithmetic_version;
    out["water_radii_m"]=Vector2(static_cast<real_t>(terrain.water_radius_x_m),static_cast<real_t>(terrain.water_radius_z_m));
    out["flat_radii_m"]=Vector2(static_cast<real_t>(terrain.flat_radius_x_m),static_cast<real_t>(terrain.flat_radius_z_m));
    out["floor_y_m"]=terrain.floor_y_m;out["reference_scale_m"]=terrain.reference_scale_m;
    out["used_by_contact"]=true;out["used_by_policy"]=false;out["geometry_only"]=false;
    out["active_motion_domain"]="shared_pond_terrain";out["contact_overlap_limit_m"]=runtime::overlap_limit_m;
    out["whole_body_query_tolerance_m"]=runtime::query_tolerance_m;
    return out;
}
Dictionary HoldLabPreview::sample_pond_surface(const PackedVector2Array& positions_m) const {
    Dictionary out;out["schema"]=1;out["version"]=pond::version;out["valid"]=false;
    if(positions_m.size()>100000){out["error"]="Geometry query batch is too large";return out;}
    PackedFloat64Array heights,coordinates;PackedVector3Array normals;PackedInt32Array regions;
    heights.resize(positions_m.size());coordinates.resize(positions_m.size());normals.resize(positions_m.size());regions.resize(positions_m.size());
    for(int64_t i=0;i<positions_m.size();++i) {
        const auto at=positions_m[i];const auto query=pond::sample(impl_->world.terrain,at.x,at.y);
        if(!query.valid){out["error"]="Invalid geometry query";out["invalid_index"]=i;return out;}
        heights[i]=query.height_m;coordinates[i]=query.shoreline_coordinate;
        normals[i]=Vector3(static_cast<real_t>(query.normal.x),static_cast<real_t>(query.normal.y),static_cast<real_t>(query.normal.z));
        regions[i]=static_cast<int32_t>(query.region);
    }
    out["valid"]=true;out["heights_m"]=heights;out["normals"]=normals;out["regions"]=regions;
    out["shoreline_coordinates"]=coordinates;return out;
}
Dictionary HoldLabPreview::sample_pond_outline(const PackedVector2Array& angle_radius) const {
    Dictionary out;out["schema"]=1;out["version"]=pond::version;out["valid"]=false;
    if(angle_radius.size()>100000){out["error"]="Geometry query batch is too large";return out;}
    PackedVector3Array positions;positions.resize(angle_radius.size());
    for(int64_t i=0;i<angle_radius.size();++i) {
        const auto at=angle_radius[i];const auto query=pond::outline(impl_->world.terrain,at.x,at.y);
        if(!query.valid){out["error"]="Invalid outline query";out["invalid_index"]=i;return out;}
        positions[i]=Vector3(static_cast<real_t>(query.point_m.x),static_cast<real_t>(query.point_m.y),static_cast<real_t>(query.point_m.z));
    }
    out["valid"]=true;out["positions_m"]=positions;return out;
}
Dictionary HoldLabPreview::get_geometry() const {
    const auto& terrain=impl_->world.terrain;Dictionary out;
    out["model"]=6;out["runtime_version"]=runtime::version;out["scale_m"]=p::scale;
    out["collision_envelope_version"]=cudarl::pond_body_geometry::collision_envelope_version;
    out["query_arithmetic_version"]=cudarl::pond_body_geometry::query_arithmetic_version;
    out["rod_radius_m"]=model::Geometry::rod_radius;out["fin_extension_m"]=model::Geometry::fin_extension;
    out["fin_thickness_m"]=model::Geometry::fin_thickness;out["floor_y_m"]=terrain.floor_y_m;
    out["arena_width_m"]=2*terrain.water_radius_x_m;out["arena_depth_m"]=2*terrain.water_radius_z_m;
    out["pond_flat_radii_m"]=Vector2(static_cast<real_t>(terrain.flat_radius_x_m),static_cast<real_t>(terrain.flat_radius_z_m));
    out["pond_water_radii_m"]=Vector2(static_cast<real_t>(terrain.water_radius_x_m),static_cast<real_t>(terrain.water_radius_z_m));
    out["pond_geometry"]=get_pond_geometry_descriptor();
    Array links;
    for(int i=0;i<3;++i){const auto dimensions=model::body_geometry(i);Dictionary link;link["length"]=dimensions.length;
        link["half_height"]=dimensions.half_height;link["half_width"]=dimensions.half_width;links.append(link);}
    out["links"]=links;return out;
}
Dictionary HoldLabPreview::get_state() const {
    const auto& s=*impl_;const auto& physical=s.world.motor.physical;const auto& body=physical.body;
    const auto measured=obs::measure(s.request,body);const auto motion=obs::motion(body);Dictionary out;
    out["loaded"]=s.loaded;out["paused"]=s.paused;out["finished"]=s.finished();out["untimed_swim"]=s.loaded&&s.untimed;
    out["heading_only_request"]=s.heading_only;out["requested_turn_degrees"]=s.requested_turn_degrees;
    out["body"]=body_record(body);
    out["training"]=false;out["holding"]=s.holding;out["qualified_candidate"]=s.qualified;out["error"]=get_error();out["case"]=s.case_id;
    out["model"]=6;out["runtime_version"]=runtime::version;out["live_physics"]=true;out["stochastic_exploration"]=false;
    out["collision_envelope_version"]=cudarl::pond_body_geometry::collision_envelope_version;
    out["query_arithmetic_version"]=cudarl::pond_body_geometry::query_arithmetic_version;
    out["saved_motor_provenance_version"]=cudarl::motor::saved_motor_provenance_version;
    out["runtime_adopted"]=false;out["runtime_qualification"]="external_evidence_required";
    out["controls"]=static_cast<int64_t>(s.controls);out["attempted_controls"]=static_cast<int64_t>(s.world.attempted_controls);
    out["substeps"]=static_cast<int64_t>(s.world.motor.attempted_substeps);
    out["attempted_substeps"]=static_cast<int64_t>(s.world.motor.attempted_substeps);
    out["accepted_substeps"]=static_cast<int64_t>(physical.substeps);out["time_s"]=physical.time_seconds;
    out["world_controls"]=static_cast<int64_t>(physical.control_steps);out["task_controls"]=static_cast<int64_t>(s.request.controls);
    out["maximum_controls"]=static_cast<int64_t>(s.maximum_controls());out["duration_s"]=static_cast<double>(s.maximum_controls())*.05;
    out["failure_recorded"]=s.failure_recorded;out["physical_failure_code"]=s.world.failure;
    out["pond_domain_valid"]=runtime::valid_state(s.world)&&s.world.inspection.valid&&s.world.inspection.certified&&
        s.world.inspection.lower_gap_m>=-runtime::overlap_limit_m;
    out["pond_domain_version"]=runtime::version;out["shared_shoreline"]=true;out["bank_navigation_implemented"]=false;
    out["whole_body_clearance_lower_m"]=s.world.inspection.lower_gap_m;
    out["whole_body_clearance_upper_m"]=s.world.inspection.upper_gap_m;
    out["minimum_control_clearance_m"]=s.world.minimum_control_gap_m;
    out["minimum_lifetime_clearance_m"]=s.world.minimum_gap_m;
    out["whole_body_query_certified"]=s.world.inspection.certified;
    out["contact_spring_work_J"]=s.world.contact_spring_work;out["contact_damping_loss_J"]=s.world.contact_damping_loss;
    out["contact_potential_J"]=s.world.contact_potential_J;
    out["contact_elastic_residual_J"]=s.world.contact_elastic_residual_J;
    out["maximum_passive_energy_excess_J"]=s.world.maximum_passive_energy_excess_J;
    if(s.failure_recorded){out["failure_before"]=body_record(s.world.failure_before);out["failure_candidate"]=body_record(s.world.failure_candidate);}
    out["training_update"]=336;out["holding_updates"]=s.lesson_updates;out["observation_schema"]=obs::observation_schema;
    out["action_mapping"]=cudarl::joint_motor::kActionVersion;out["package_version"]=s.loaded?2:0;
    out["policy_method"]=s.loaded?"learned_persistent_amplitude":"unloaded";
    out["holding_parameter_count"]=s.loaded?2:0;out["backbone_frozen"]=s.loaded;
    out["policy_mu"]=s.head[0];out["policy_variance_coordinate"]=s.head[1];out["policy_sigma"]=s.sigma;out["deterministic_amplitude"]=s.gain;
    out["request_origin"]=point(s.request.start_com_model);out["request_heading"]=point(s.request.request.heading);
    out["request_position"]=point(s.request.request.position);out["terminal_velocity"]=point(s.request.request.terminal_velocity);
    Array targets,rates,torques,commands;
    for(int j=0;j<4;++j){targets.append(physical.held_targets[j]);rates.append(s.world.motor.reference_rate[j]);
        torques.append(physical.last_actual_torques[j]);commands.append(physical.raw_action[j]);}
    out["held_targets"]=targets;out["reference_rates"]=rates;out["previous_torques"]=torques;out["last_action"]=commands;
    out["target_speed_m_s"]=s.holding?0.f:obs::swim_speed_m_s;out["speed_m_s"]=p::norm(measured.velocity_m_s);
    const float horizontal_speed=sqrtf(motion.velocity_m_s.x*motion.velocity_m_s.x+motion.velocity_m_s.z*motion.velocity_m_s.z);
    out["horizontal_speed_m_s"]=horizontal_speed;out["course_valid"]=horizontal_speed>=1e-6f;
    out["course_deg"]=atan2f(motion.velocity_m_s.z,motion.velocity_m_s.x)*180/obs::pi;
    out["depth_m"]=measured.depth_m;out["signed_depth_m"]=measured.signed_depth_m;out["lateral_m"]=measured.lateral_m;
    out["anchor_error_m"]=s.holding?p::norm(measured.hold_error_m):0.f;out["tilt_deg"]=measured.tilt_deg;
    out["forward_m"]=measured.forward_m;out["anchor_position"]=point(s.holding?s.request.hold_anchor_model:s.request.start_com_model);
    out["com_position"]=point(motion.com_model);out["head_x"]=body.position.x;out["head_y"]=body.position.y;out["head_z"]=body.position.z;
    out["qx"]=body.orientation.x;out["qy"]=body.orientation.y;out["qz"]=body.orientation.z;out["qw"]=body.orientation.w;
    out["passed"]=false;
    for(int j=0;j<4;++j){out[String("j")+String::num_int64(j)]=body.angle[j];out[String("gain")+String::num_int64(j)]=s.gain;}
    Array links;for(const auto& link:model::shape(body).link){Dictionary item;item["position"]=point(link.center/p::scale);
        item["orientation"]=Quaternion(link.q.x,link.q.y,link.q.z,link.q.w);links.append(item);}out["links"]=links;return out;
}
}
