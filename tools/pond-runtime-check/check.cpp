// Explicit opt-in CPU product qualification; no training or optimizer path.
#include "simulation/pond_runtime.h"
#include "motor/pond_motor_observation.h"
#include "motor/straight_task.h"
#include "motor/hold_assessment.h"
#include "motor/saved_motor_contract.h"
#include "rl/learned_amplitude_math.h"
#include "nn/mlp_cpu.h"
#include "core/checkpoint.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
namespace p=cudarl::physics;
namespace old=cudarl::straight_motor;
namespace po=cudarl::pond_motor_observation;
namespace pr=cudarl::pond_runtime;
namespace pg=cudarl::pond_geometry_v1;
namespace bg=cudarl::pond_body_geometry;
namespace cp=cudarl::checkpoint;
namespace nn=cudarl::nn;
namespace {
constexpr uint64_t control_cap=68000,substep_cap=688000;
constexpr double seconds_cap=900,body_tolerance=2e-6;
constexpr const char* protocol_id="shoreline-product-v4-2026-11-23";
static_assert(pr::version==4,"Corrected product qualification requires runtime4");
static_assert(bg::query_arithmetic_version==2,"Corrected query arithmetic must be bound explicitly");
using Keys=std::set<std::pair<int,int>>;
struct Options {bool run=false,preflight=false;fs::path policy,out,protocol,reference;};
void require(bool yes,const std::string& why) {if(!yes)throw std::runtime_error(why);}
struct Package {std::vector<float> weights;float gain=0;uint64_t checksum=0,training_protocol=0;};
std::string escaped(const std::string& value) {
    std::string result;
    for(char c:value){if(c=='"'||c=='\\')result+='\\';if(c=='\n')result+="\\n";else if(c=='\r')result+="\\r";else result+=c;}
    return result;
}
std::string optional_number(double value,bool present) {
    if(!present)return "null";
    std::ostringstream out;out<<std::setprecision(17)<<value;return out.str();
}
Package load_package(const fs::path& path) {
    const auto bytes=cp::read(path.string());cp::Reader r{bytes};Package result;result.checksum=cp::checksum(bytes);
    require(r.value<uint32_t>()==0x31564c48&&r.value<uint32_t>()==2,"saved_package_magic_or_version");
    require(r.value<uint32_t>()==6&&r.value<uint32_t>()==6&&r.value<uint32_t>()==3&&r.value<uint32_t>()==4&&
        r.value<uint32_t>()==41&&r.value<uint32_t>()==4&&r.value<uint32_t>()==336,"saved_package_training_contract");
    require(r.value<uint32_t>()==1&&r.value<uint32_t>()==7&&r.value<uint32_t>()==1,"saved_amplitude_qualification");
    result.training_protocol=r.value<uint64_t>();require(result.training_protocol!=0,"saved_training_protocol_missing");
    const auto parent=r.array<uint8_t>(64);
    require(std::string(parent.begin(),parent.end())=="810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0","saved_parent_mismatch");
    const auto& contracts=cudarl::motor::saved_motor_source_contract_hashes();
    require(r.value<uint32_t>()==contracts.size(),"saved_source_count");
    for(const auto expected:contracts){const auto hash=r.array<uint8_t>(64);require(std::string(hash.begin(),hash.end())==expected,"saved_source_provenance");}
    result.weights=r.array<float>(2532);const auto head=r.array<float>(2);
    const float stored_gain=r.value<float>(),stored_sigma=r.value<float>();r.finish();
    require(result.weights.size()==2532&&head.size()==2,"saved_parameter_count");
    for(float value:result.weights)require(std::isfinite(value),"saved_nonfinite_weight");
    require(std::isfinite(head[0])&&std::isfinite(head[1]),"saved_nonfinite_amplitude");
    result.gain=cudarl::learned_amplitude::sigmoid(head[0]);const float sigma=cudarl::learned_amplitude::standard_deviation(head[1]);
    require(std::isfinite(stored_gain)&&std::isfinite(stored_sigma)&&std::abs(stored_gain-result.gain)<=2e-7f&&
        std::abs(stored_sigma-sigma)<=2e-6f&&result.gain>=0&&result.gain<=1&&sigma>=.05f&&sigma<=8.f,"saved_amplitude_mapping");
    return result;
}
nn::CpuMlp<float> actor(const Package& package) {
    nn::CpuMlp<float> result({{41,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}});
    result.params()=package.weights;return result;
}
struct Counts {
    uint64_t attempted_controls=0,accepted_controls=0,attempted_substeps=0;
    uint64_t old_controls=0,new_controls=0,bank_controls=0,half_controls=0,outward_controls=0,pure_inspections=0;
    uint64_t product_endpoint_inspections=0;
    double minimum_observed_gap_m=1e30,maximum_abs_elastic_residual_J=0,maximum_passive_excess_J=0;
    int attempted_cases=0,completed_cases=0,failed_cases=0;
    std::chrono::steady_clock::time_point began=std::chrono::steady_clock::now();
    void guard(int substeps=10) const {
        require(attempted_controls<control_cap&&attempted_substeps+static_cast<uint64_t>(substeps)<=substep_cap,"finite_physics_cap");
        require(std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count()<seconds_cap,"finite_wall_cap");
    }
    void observe(const pr::World& w) {
        minimum_observed_gap_m=std::min(minimum_observed_gap_m,w.minimum_gap_m);
        maximum_abs_elastic_residual_J=std::max(maximum_abs_elastic_residual_J,std::abs(w.contact_elastic_residual_J));
        maximum_passive_excess_J=std::max(maximum_passive_excess_J,w.maximum_passive_energy_excess_J);
    }
};
struct Evidence {
    fs::path root;std::ofstream controls,cases,windows,physics_cases,prefixes;bool failure_written=false;int failures=0;
    std::vector<std::pair<std::string,int>> planned;
    std::set<std::pair<std::string,int>> attempted;
    explicit Evidence(const fs::path& path):root(path),controls(path/"controls.csv"),cases(path/"cases.csv"),windows(path/"windows.csv"),
        physics_cases(path/"physics-cases.csv"),prefixes(path/"matched-product-endpoints.csv") {
        controls<<std::setprecision(17);cases<<std::setprecision(17);windows<<std::setprecision(17);
        physics_cases<<std::setprecision(17);prefixes<<std::setprecision(17);
        controls<<"suite,case,world,control,attempted,accepted,substeps,physical_control_steps,actor_observation_control,request_state_controls,px_model,py_model,pz_model,qx,qy,qz,qw";
        for(int j=0;j<4;++j)controls<<",angle"<<j;
        for(int j=0;j<10;++j)controls<<",velocity"<<j;
        for(int j=0;j<4;++j)controls<<",target"<<j<<",reference_rate"<<j<<",last_actual_torque"<<j<<",command"<<j;
        controls<<",work_J,travel_model,COM_vx,COM_vy,COM_vz,forward_m,lane_m,depth_m,tilt_deg,minimum_control_gap_m,minimum_lifetime_gap_m,contact_spring_work_J,contact_damping_loss_J,contact_potential_J,energy_error_J,total_resistance_loss_J,water_loss_J,contact_elastic_residual_J,maximum_passive_energy_excess_J\n";
        cases<<"suite,case,pose,offset_m,duration_controls,attempted_controls,accepted_controls,attempted_substeps,completed,passed,max_body_difference,max_filter_difference,max_observation_difference,reason\n";
        windows<<"world,pose,end_s,speed_m_s,RMSE_m_s,max_lane_m,max_depth_m,max_tilt_deg,path_deg,work_J_per_m,passed\n";
        physics_cases<<"suite,case,minimum_whole_body_gap_m,total_resistance_loss_J,water_loss_J,contact_damping_loss_J,contact_spring_work_J,contact_potential_J,contact_elastic_residual_J,maximum_passive_energy_excess_J\n";
        prefixes<<"suite,case,control,endpoint_only,reference_valid,reference_certified,reference_gap_lower_m,reference_failure,new_valid,new_certified,new_gap_lower_m,new_failure,positive_clearance_endpoint_prefix\n";
        require(bool(controls)&&bool(cases)&&bool(windows)&&bool(physics_cases)&&bool(prefixes),"evidence_open_failed");
        for(int i=0;i<4;++i)planned.emplace_back("short",i);
        for(int i:{0,3})planned.emplace_back("long",i);
        for(const char* suite:{"original_hold","withheld_hold"})for(int i=0;i<6;++i)planned.emplace_back(suite,i);
        for(int i:{0,3})planned.emplace_back("endurance_hold",i);
        for(int i=0;i<8;++i)planned.emplace_back("bank_contact",i);
        for(const char* suite:{"half_step","outward_swim"})for(int i=0;i<4;++i)planned.emplace_back(suite,i);
        std::ofstream plan(path/"planned-cases.csv");plan<<"suite,case\n";for(const auto& key:planned)plan<<key.first<<','<<key.second<<'\n';
    }
    void begin(const char* suite,int id) {
        attempted.emplace(suite,id);cases<<suite<<','<<id<<",,,,0,0,0,0,0,0,0,0,attempt_started\n";cases.flush();
    }
    ~Evidence() {for(const auto& key:planned)if(attempted.count(key)==0)cases<<key.first<<','<<key.second<<",,,,0,0,0,0,0,0,0,0,skipped_after_stop\n";}
    void record(const char* suite,int id,const char* world,int control,const cudarl::joint_motor::World& w,
                const po::RequestState& request,const float* command,bool attempted_interval,bool accepted,
                double gap,double spring=0,double damping=0,double potential=0,const pr::World* pond=nullptr) {
        const auto& b=w.physical.body;const auto z=po::measure(request,b);
        controls<<suite<<','<<id<<','<<world<<','<<control<<','<<attempted_interval<<','<<accepted<<','<<w.attempted_substeps
            <<','<<w.physical.control_steps<<','<<control-1<<','<<request.controls
            <<','<<b.position.x<<','<<b.position.y<<','<<b.position.z<<','<<b.orientation.x<<','<<b.orientation.y<<','<<b.orientation.z<<','<<b.orientation.w;
        for(float a:b.angle)controls<<','<<a;
        for(float v:b.velocity)controls<<','<<v;
        for(int j=0;j<4;++j)controls<<','<<w.physical.held_targets[j]<<','<<w.reference_rate[j]<<','<<w.physical.last_actual_torques[j]<<','<<(command?command[j]:0);
        controls<<','<<b.work<<','<<b.travel<<','<<z.velocity_m_s.x<<','<<z.velocity_m_s.y<<','<<z.velocity_m_s.z
            <<','<<z.forward_m<<','<<z.lateral_m<<','<<z.depth_m<<','<<z.tilt_deg<<','<<gap<<','<<(pond?pond->minimum_gap_m:gap)
            <<','<<spring<<','<<damping<<','<<potential<<','<<w.last_energy_error<<','<<w.fluid_loss<<','<<w.fluid_loss-damping
            <<','<<(pond?pond->contact_elastic_residual_J:0)<<','<<(pond?pond->maximum_passive_energy_excess_J:0)<<'\n';
        require(bool(controls),"control_evidence_write_failed");
    }
    void physics_case(const char* suite,int id,const pr::World& w) {
        physics_cases<<suite<<','<<id<<','<<w.minimum_gap_m<<','<<w.motor.fluid_loss<<','<<w.motor.fluid_loss-w.contact_damping_loss
            <<','<<w.contact_damping_loss<<','<<w.contact_spring_work<<','<<w.contact_potential_J<<','<<w.contact_elastic_residual_J
            <<','<<w.maximum_passive_energy_excess_J<<'\n';physics_cases.flush();
        require(bool(physics_cases),"physics_case_evidence_write_failed");
    }
    bool endpoint(const char* suite,int id,int control,const p::BodyState& reference,const p::BodyState& candidate,Counts& count) {
        const auto terrain=pg::current_visual_parameters();
        const auto a=bg::inspect(reference,terrain,pr::query_tolerance_m,pr::query_visit_limit);
        const auto b=bg::inspect(candidate,terrain,pr::query_tolerance_m,pr::query_visit_limit);
        count.product_endpoint_inspections+=2;
        const bool clear=a.valid&&a.certified&&b.valid&&b.certified&&a.lower_gap_m>0&&b.lower_gap_m>0;
        prefixes<<suite<<','<<id<<','<<control<<",1,"<<a.valid<<','<<a.certified<<','<<a.lower_gap_m<<','<<a.failure_reason
            <<','<<b.valid<<','<<b.certified<<','<<b.lower_gap_m<<','<<b.failure_reason<<','<<clear<<'\n';prefixes.flush();
        require(bool(prefixes),"endpoint_evidence_write_failed");
        return clear;
    }
    void failure(const std::string& reason,const p::BodyState& before,const p::BodyState& candidate,int control) {
        const auto stem="failure-"+std::to_string(++failures);cp::Writer data;
        data.value(uint32_t{0x31465050});data.value(uint32_t{1});data.value(control);data.value(before);data.value(candidate);
        cp::write_atomic((root/(stem+".bin")).string(),data.data);
        std::ofstream note(root/(stem+".json"));note<<"{\"reason\":\""<<escaped(reason)<<"\",\"control\":"<<control
            <<",\"body_bytes\":"<<sizeof(p::BodyState)<<",\"binary_order\":\"magic,version,control,before,candidate\"}\n";
        note.close();
        if(!failure_written) {
            failure_written=true;cp::write_atomic((root/"first-failure.bin").string(),data.data);
            fs::copy_file(root/(stem+".json"),root/"first-failure.json");
        }
    }
};
struct CaseSpec {const char* suite;int id,pose;float offset;int controls;bool holding,long_swim=false;};
std::vector<CaseSpec> retention_specs() {
    std::vector<CaseSpec> out;
    for(int pose=0;pose<4;++pose)out.push_back({"short",pose,pose,0,160,false});
    for(int pose:{0,3})out.push_back({"long",pose,pose,0,1200,false,true});
    for(int id=0;id<6;++id)out.push_back({"original_hold",id,id%2?3:0,id/2==0?0.f:id/2==1?-.0005f:.0005f,1280,true});
    int id=0;for(float depth:{0.f,-.00025f,.00025f})for(int pose:{1,2})out.push_back({"withheld_hold",id++,pose,depth,1280,true});
    for(int pose:{0,3})out.push_back({"endurance_hold",pose,pose,0,6000,true});
    return out;
}
pg::Parameters compatibility_terrain() {
    auto parameters=pg::current_visual_parameters();parameters.water_radius_x_m=.9;parameters.water_radius_z_m=.7;
    parameters.flat_radius_x_m=.65;parameters.flat_radius_z_m=.45;return parameters;
}
struct Pair {old::World baseline;old::ForwardTask old_request;old::ForwardConfig config;pr::World candidate;po::RequestState request;};
Pair initialize_pair(const CaseSpec& spec) {
    Pair pair;pair.baseline=old::initial_world(spec.pose);pair.baseline.arena=old::long_swim_arena();
    pair.config=old::config_for_speed(.008f);pair.config.observation_schema=4;pair.config.maximum_controls=spec.controls;
    pair.config.target_speed_m_s=spec.holding?0.f:.008f;pair.config.stationary_request=spec.holding;pair.config.position_hold=spec.holding;
    require(old::begin_forward(pair.baseline,pair.config,pair.old_request),"old_fixture_begin");
    require(old::offset_episode_start(pair.baseline,pair.old_request,pair.config,spec.offset),"old_fixture_offset");
    if(spec.holding)require(old::update_velocity_request(pair.baseline,pair.config,pair.old_request),"old_hold_request");
    auto& motor=pair.candidate.motor;
    require(po::initialize(spec.pose,false,motor.physical.body,motor.physical.held_targets,motor.reference_rate,
        motor.physical.last_actual_torques,pair.request),"new_fixture_begin");
    motor.physical.paused=false;motor.arena=pair.baseline.arena;
    if(spec.holding) {
        pair.request.holding=true;pair.request.request.terminal_velocity=pair.request.request.heading*0.f;
        if(spec.offset!=0)motor.physical.body.position.y+=spec.offset/p::scale;
        pair.request.hold_anchor_model=po::motion(motor.physical.body).com_model;
        pair.request.hold_anchor_model.y=pair.request.start_com_model.y;pair.request.hold_anchor_set=true;
    }
    pair.candidate.terrain=compatibility_terrain();require(pr::prepare(pair.candidate),"new_fixture_prepare");return pair;
}
double body_difference(const p::BodyState& a,const p::BodyState& b) {
    double result=0;const auto add=[&](float x,float y){result=std::max(result,std::abs(double(x)-y));};
    add(a.position.x,b.position.x);add(a.position.y,b.position.y);add(a.position.z,b.position.z);
    add(a.orientation.x,b.orientation.x);add(a.orientation.y,b.orientation.y);add(a.orientation.z,b.orientation.z);add(a.orientation.w,b.orientation.w);
    for(int j=0;j<4;++j)add(a.angle[j],b.angle[j]);
    for(int j=0;j<10;++j)add(a.velocity[j],b.velocity[j]);
    add(a.work,b.work);add(a.travel,b.travel);return result;
}
double filter_difference(const old::World& a,const old::World& b) {
    double result=0;for(int j=0;j<4;++j) {
        result=std::max(result,std::abs(double(a.reference_rate[j])-b.reference_rate[j]));
        result=std::max(result,std::abs(double(a.physical.held_targets[j])-b.physical.held_targets[j]));
        result=std::max(result,std::abs(double(a.physical.last_actual_torques[j])-b.physical.last_actual_torques[j]));
    }return result;
}
struct HoldScore {
    int total,next=0;float peak_speed=0,peak_anchor=0,peak_depth=0,peak_lane=0,peak_tilt=0,window_forward=0,excursion=0;
    explicit HoldScore(int n):total(n){}
    template<class M>void record(int control,const M& z) {
        require(control==next&&control<=total,"hold_score_order");++next;
        const float speed=p::norm(z.velocity_m_s),anchor=p::norm(z.hold_error_m);
        require(std::isfinite(speed)&&std::isfinite(anchor)&&std::isfinite(z.depth_m)&&std::isfinite(z.lateral_m)&&std::isfinite(z.tilt_deg)&&std::isfinite(z.forward_m),"hold_score_nonfinite");
        peak_speed=std::max(peak_speed,speed);peak_anchor=std::max(peak_anchor,anchor);peak_depth=std::max(peak_depth,z.depth_m);
        peak_lane=std::max(peak_lane,z.lateral_m);peak_tilt=std::max(peak_tilt,z.tilt_deg);
        if(control==total-160)window_forward=z.forward_m;
        if(control>=total-160)excursion=std::max(excursion,std::abs(z.forward_m-window_forward));
    }
    bool passed()const{return next==total+1&&peak_speed<=.0005f&&peak_anchor<=.002f&&peak_depth<=.001f&&peak_lane<=.002f&&peak_tilt<=10.f&&excursion<=.002f;}
};
struct SwimScore {
    float max_lane=0,max_depth=0,max_tilt=0,start_work=0,window_work=0;double error2=0,window_error2=0;
    p::V steady_origin{},window_origin{};Keys keys;
    template<class M>void record(int control,const M& z,const p::BodyState& b,p::V heading,bool windows,int pose,const char* name,std::ostream& out) {
        max_lane=std::max(max_lane,z.lateral_m);max_depth=std::max(max_depth,z.depth_m);max_tilt=std::max(max_tilt,z.tilt_deg);
        const auto com=po::motion(b).com_model;
        if(control==40){steady_origin=window_origin=com;window_work=b.work;}
        if(control>40){error2+=z.speed_error_m_s*z.speed_error_m_s;window_error2+=z.speed_error_m_s*z.speed_error_m_s;}
        if(windows&&control%80==0) {
            const auto delta=(com-window_origin)*p::scale;const float forward=p::dot(delta,heading);
            const auto side=delta-heading*forward;const int samples=control==80?40:80;
            const float speed=forward/(samples*p::dt),rmse=sqrtf(static_cast<float>(window_error2/samples));
            const float path=atan2f(p::norm({side.x,0,side.z}),forward)*180/po::pi;
            const float work=(b.work-window_work)/fmaxf(forward,1e-9f);
            const bool pass=speed>=.9f*.008f&&speed<=1.1f*.008f&&rmse<=.3125f*.008f&&max_lane<=.002f&&max_depth<=.001f&&max_tilt<=10.f&&path<=5&&work<=.001f;
            if(pass)keys.emplace(pose,control/20);
            out<<name<<','<<pose<<','<<control/20<<','<<speed<<','<<rmse<<','<<max_lane<<','<<max_depth<<','<<max_tilt<<','<<path<<','<<work<<','<<pass<<'\n';
            window_origin=com;window_work=b.work;window_error2=0;
        }
    }
    bool short_passed(int controls,const p::BodyState& b,p::V heading,float forward_total)const {
        const auto delta=(po::motion(b).com_model-steady_origin)*p::scale;const float forward=p::dot(delta,heading);
        const auto side=delta-heading*forward;const float speed=forward/(float(controls-40)*p::dt);
        const float rmse=sqrtf(static_cast<float>(error2/(controls-40))),path=atan2f(p::norm({side.x,0,side.z}),forward)*180/po::pi;
        const float work=(b.work-start_work)/fmaxf(forward_total,1e-9f);
        return speed>=.9f*.008f&&speed<=1.1f*.008f&&rmse<=.3125f*.008f&&max_lane<=.002f&&max_depth<=.001f&&max_tilt<=10.f&&path<=5&&work<=.001f;
    }
};
Keys expected_long_keys(const fs::path& path) {
    std::ifstream input(path);require(bool(input),"missing_retained_long_window_reference");std::string line;std::getline(input,line);Keys passed,all;
    while(std::getline(input,line)) {
        std::stringstream row(line);std::vector<double> values;std::string value;while(std::getline(row,value,','))values.push_back(std::stod(value));
        require(values.size()==10,"long_reference_shape");const int pose=static_cast<int>(values[0]),end=static_cast<int>(values[1]);
        require((pose==0||pose==3)&&end>=4&&end<=60&&end%4==0&&all.emplace(pose,end).second,"long_reference_key");
        if(values[9]==1)passed.emplace(pose,end);
    }
    require(all.size()==30&&passed.size()==28,"long_reference_pass_keys");return passed;
}
struct Fixture {pr::World world;po::RequestState request;};
Fixture outward_fixture(int id) {
    Fixture f;auto& motor=f.world.motor;
    require(po::initialize(0,false,motor.physical.body,motor.physical.held_targets,motor.reference_rate,motor.physical.last_actual_torques,f.request),"outward_initialize");
    const float angle=float(id)*po::pi*.5f;motor.physical.body.orientation=p::axis_rotation({0,1,0},-angle);
    motor.physical.body.position.x=0;motor.physical.body.position.z=0;motor.physical.paused=false;
    f.request.start_com_model=po::motion(motor.physical.body).com_model;
    auto heading=motor.physical.body.orientation.rotate({1,0,0});heading=heading/p::norm(heading);
    f.request.request={f.request.start_com_model+heading*(.072f/p::scale),heading*.008f,heading};
    require(pr::prepare(f.world),"outward_prepare");return f;
}
Fixture bank_fixture(int id,Counts& counts) {
    Fixture f;auto& motor=f.world.motor;const float angle=float(id)*po::pi*.25f;
    const p::V direction{cosf(angle),0,sinf(angle)};
    motor.physical.body.orientation=p::axis_rotation({0,1,0},-angle);motor.physical.paused=false;
    const auto query=[&](double radius) {
        auto body=motor.physical.body;body.position.x=direction.x*static_cast<float>(radius/p::scale);body.position.z=direction.z*static_cast<float>(radius/p::scale);
        ++counts.pure_inspections;return bg::inspect(body,f.world.terrain,pr::query_tolerance_m);
    };
    const double ellipse_angle=atan2(double(direction.z)/f.world.terrain.water_radius_z_m,double(direction.x)/f.world.terrain.water_radius_x_m);
    const double shore_radius=pg::edge(ellipse_angle)/hypot(double(direction.x)/f.world.terrain.water_radius_x_m,double(direction.z)/f.world.terrain.water_radius_z_m);
    double low=0,high=shore_radius*.8;
    auto low_query=query(low),high_query=query(high);
    require(low_query.valid&&low_query.certified&&high_query.valid&&high_query.certified&&low_query.lower_gap_m>.0001&&high_query.lower_gap_m<.0001,"bank_fixture_bracket");
    for(int i=0;i<48;++i){const double middle=(low+high)*.5;const auto q=query(middle);require(q.valid&&q.certified,"bank_fixture_query");if(q.lower_gap_m>=.0001)low=middle;else high=middle;}
    motor.physical.body.position.x=direction.x*static_cast<float>(low/p::scale);motor.physical.body.position.z=direction.z*static_cast<float>(low/p::scale);
    motor.physical.body.velocity[0]=direction.x*.005f;motor.physical.body.velocity[2]=direction.z*.005f;
    f.request.start_com_model=po::motion(motor.physical.body).com_model;f.request.request.heading=direction;
    f.request.request.position=f.request.start_com_model+direction*(.072f/p::scale);f.request.request.terminal_velocity=direction*.008f;
    require(pr::prepare(f.world)&&f.world.inspection.lower_gap_m>=.000099,"bank_fixture_prepare");return f;
}
void commands(nn::CpuMlp<float>& net,const po::RequestState& request,const old::World& world,float gain,float* output,float* observations=nullptr) {
    float values[41]{};require(po::observe(request,world.physical.body,world.physical.held_targets,world.reference_rate,world.physical.last_actual_torques,values),"product_observation_invalid");
    if(observations)std::copy(values,values+41,observations);
    net.forward(values,1);for(int j=0;j<4;++j)output[j]=tanhf((request.holding?gain:1.f)*net.output()[j]);
}
void write_result(const Options& option,const Counts& count,const Package& package,bool pass,const std::string& reason,uint64_t protocol_hash=0) {
    std::ofstream out(option.out/(option.preflight?"preflight.json":"result.json"));
    out<<std::setprecision(17)<<"{\"schema\":1,\"protocol_id\":\""<<protocol_id<<"\",\"runtime_version\":"<<pr::version
        <<",\"query_arithmetic_version\":"<<bg::query_arithmetic_version<<",\"collision_envelope_version\":"<<bg::collision_envelope_version
        <<",\"preflight_only\":"<<(option.preflight?"true":"false")
        <<",\"passed\":"<<(pass?"true":"false")<<",\"reason\":\""<<escaped(reason)<<"\",\"package_checksum_fnv1a64\":"<<package.checksum
        <<",\"protocol_checksum_fnv1a64\":"<<protocol_hash<<",\"saved_gain\":"<<package.gain
        <<",\"attempted_controls\":"<<count.attempted_controls<<",\"accepted_controls\":"<<count.accepted_controls
        <<",\"attempted_substeps\":"<<count.attempted_substeps<<",\"retention_old_accepted_controls\":"<<count.old_controls
        <<",\"retention_new_accepted_controls\":"<<count.new_controls<<",\"bank_accepted_controls\":"<<count.bank_controls
        <<",\"half_step_accepted_controls\":"<<count.half_controls
        <<",\"outward_accepted_controls\":"<<count.outward_controls<<",\"bank_placement_inspections\":"<<count.pure_inspections
        <<",\"product_endpoint_inspections\":"<<count.product_endpoint_inspections
        <<",\"minimum_observed_whole_body_gap_m\":"<<optional_number(count.minimum_observed_gap_m,count.attempted_substeps!=0)
        <<",\"maximum_abs_elastic_residual_J\":"<<count.maximum_abs_elastic_residual_J
        <<",\"maximum_passive_energy_excess_J\":"<<count.maximum_passive_excess_J
        <<",\"planned_cases\":36,\"skipped_cases\":"<<36-count.attempted_cases
        <<",\"attempted_cases\":"<<count.attempted_cases<<",\"completed_cases\":"<<count.completed_cases<<",\"failed_cases\":"<<count.failed_cases
        <<",\"CUDA_kernels\":0,\"optimizer_steps\":0,\"autonomous_navigation_qualified\":false,\"wall_seconds\":"
        <<std::chrono::duration<double>(std::chrono::steady_clock::now()-count.began).count()<<"}\n";
    out.flush();require(bool(out),"result_write_failed");
}
void retention_case(const CaseSpec& spec,const Package& package,Counts& count,Evidence& evidence,Keys& old_keys,Keys& new_keys) {
    ++count.attempted_cases;evidence.begin(spec.suite,spec.id);
    const auto before_count=count.attempted_controls,before_accepted=count.accepted_controls,before_steps=count.attempted_substeps;
    Pair pair=initialize_pair(spec);auto old_actor=actor(package),new_actor=actor(package);
    HoldScore old_hold(spec.controls),new_hold(spec.controls);SwimScore old_swim,new_swim;
    double max_body=0,max_filter=0,max_observation=0;std::string reason;bool completed=false,passed=false;
    try {
        bool product_prefix_open=evidence.endpoint(spec.suite,spec.id,0,pair.baseline.physical.body,pair.candidate.motor.physical.body,count);
        if(spec.holding){old_hold.record(0,old::measure(pair.old_request,pair.baseline));new_hold.record(0,po::measure(pair.request,pair.candidate.motor.physical.body));}
        for(int k=1;k<=spec.controls;++k) {
            if(spec.holding)require(old::update_velocity_request(pair.baseline,pair.config,pair.old_request),"reference_request_update");
            float old_observation[41]{},new_observation[41]{},old_command[4]{},new_command[4]{};
            require(old::observe_appended(pair.old_request,pair.baseline,pair.config,old_observation),"reference_observation_invalid");
            old_actor.forward(old_observation,1);
            for(int j=0;j<4;++j)old_command[j]=tanhf((spec.holding?package.gain:1.f)*old_actor.output()[j]);
            commands(new_actor,pair.request,pair.candidate.motor,package.gain,new_command,new_observation);
            for(int j=0;j<41;++j)max_observation=std::max(max_observation,std::abs(double(old_observation[j])-new_observation[j]));
            const auto old_before=pair.baseline.physical.body;auto candidate=old_before;auto old_pre=old_before;
            count.guard();++count.attempted_controls;const auto old_substeps=pair.baseline.attempted_substeps;
            const bool old_ok=cudarl::joint_motor::step_state(pair.baseline,old_command,nullptr,&old_pre,&candidate);
            count.attempted_substeps+=pair.baseline.attempted_substeps-old_substeps;
            if(old_ok){++count.accepted_controls;++count.old_controls;}
            evidence.record(spec.suite,spec.id,"reference",k,pair.baseline,pair.request,old_command,true,old_ok,cudarl::movement::clearance(pair.baseline.physical.body,true,pair.baseline.arena));
            if(!old_ok){evidence.failure("reference_physics_"+std::to_string(pair.baseline.failure_code),old_pre,candidate,k);throw std::runtime_error("reference_physics_failure");}
            count.guard();++count.attempted_controls;const auto new_substeps=pair.candidate.motor.attempted_substeps;
            const bool new_ok=pr::step(pair.candidate,new_command);count.attempted_substeps+=pair.candidate.motor.attempted_substeps-new_substeps;
            count.observe(pair.candidate);
            if(new_ok){++count.accepted_controls;++count.new_controls;++pair.request.controls;}
            evidence.record(spec.suite,spec.id,"pond_compatibility",k,pair.candidate.motor,pair.request,new_command,true,new_ok,pair.candidate.minimum_control_gap_m,
                pair.candidate.contact_spring_work,pair.candidate.contact_damping_loss,pair.candidate.contact_potential_J,&pair.candidate);
            if(!new_ok){evidence.failure("candidate_physics_"+std::to_string(pair.candidate.failure),pair.candidate.failure_before,pair.candidate.failure_candidate,k);throw std::runtime_error("candidate_physics_failure");}
            max_body=std::max(max_body,body_difference(pair.baseline.physical.body,pair.candidate.motor.physical.body));
            max_filter=std::max(max_filter,filter_difference(pair.baseline,pair.candidate.motor));
            if(max_body>body_tolerance||max_filter>body_tolerance) {
                evidence.failure("no_contact_retention_divergence",old_before,pair.candidate.motor.physical.body,k);
                throw std::runtime_error("no_contact_retention_divergence");
            }
            if(product_prefix_open)product_prefix_open=evidence.endpoint(spec.suite,spec.id,k,pair.baseline.physical.body,pair.candidate.motor.physical.body,count);
            const auto interval=spec.holding?old::record_hold_control(pair.old_request,pair.baseline,pair.config):old::record_control(pair.old_request,pair.baseline,pair.config);
            require(interval.accepted,"reference_scorer_rejected");const auto new_metrics=po::measure(pair.request,pair.candidate.motor.physical.body);
            if(spec.holding){old_hold.record(k,interval.metrics);new_hold.record(k,new_metrics);}
            else {
                old_swim.record(k,interval.metrics,pair.baseline.physical.body,pair.old_request.request.heading,spec.long_swim,spec.pose,"reference",evidence.windows);
                new_swim.record(k,new_metrics,pair.candidate.motor.physical.body,pair.request.request.heading,spec.long_swim,spec.pose,"pond_compatibility",evidence.windows);
            }
        }
        completed=true;
        if(spec.holding)passed=old_hold.passed()&&new_hold.passed();
        else if(spec.long_swim){old_keys.insert(old_swim.keys.begin(),old_swim.keys.end());new_keys.insert(new_swim.keys.begin(),new_swim.keys.end());passed=old_swim.keys==new_swim.keys;}
        else passed=pair.old_request.outcome==old::Outcome::Success&&new_swim.short_passed(spec.controls,pair.candidate.motor.physical.body,pair.request.request.heading,po::measure(pair.request,pair.candidate.motor.physical.body).forward_m);
        require(passed,"retained_behavior_gate_failed");++count.completed_cases;
    }catch(const std::exception& error){reason=error.what();++count.failed_cases;}
    evidence.cases<<spec.suite<<','<<spec.id<<','<<spec.pose<<','<<spec.offset<<','<<spec.controls<<','<<count.attempted_controls-before_count
        <<','<<count.accepted_controls-before_accepted<<','<<count.attempted_substeps-before_steps<<','<<completed<<','<<passed<<','<<max_body<<','<<max_filter<<','<<max_observation<<','<<reason<<'\n';
    evidence.physics_case(spec.suite,spec.id,pair.candidate);
    evidence.cases.flush();evidence.controls.flush();require(reason.empty(),reason);
}
std::vector<p::BodyState> product_case(bool bank,int id,const Package& package,Counts& count,Evidence& evidence) {
    const char* suite=bank?"bank_contact":"outward_swim";const int total=bank?200:1200;
    ++count.attempted_cases;evidence.begin(suite,id);
    const auto before_count=count.attempted_controls,before_accepted=count.accepted_controls,before_steps=count.attempted_substeps;
    auto fixture=bank?bank_fixture(id,count):outward_fixture(id);auto net=actor(package);std::string reason;bool passed=false;int completed=0;
    const double initial_gap=fixture.world.inspection.lower_gap_m;bool touched=false;std::vector<p::BodyState> states{fixture.world.motor.physical.body};
    try {
        for(int k=1;k<=total;++k) {
            float command[4]{};if(!bank)commands(net,fixture.request,fixture.world.motor,package.gain,command);
            count.guard();++count.attempted_controls;const auto before=fixture.world.motor.attempted_substeps;
            const bool ok=pr::step(fixture.world,command);count.attempted_substeps+=fixture.world.motor.attempted_substeps-before;
            count.observe(fixture.world);
            if(ok){++count.accepted_controls;++fixture.request.controls;++completed;if(bank)++count.bank_controls;else ++count.outward_controls;}
            evidence.record(suite,id,"product",k,fixture.world.motor,fixture.request,command,true,ok,fixture.world.minimum_control_gap_m,
                fixture.world.contact_spring_work,fixture.world.contact_damping_loss,fixture.world.contact_potential_J,&fixture.world);
            touched=touched||fixture.world.contact_spring_work!=0||fixture.world.contact_potential_J>0;
            if(!ok) {
                evidence.failure(std::string(suite)+"_physics_"+std::to_string(fixture.world.failure),fixture.world.failure_before,fixture.world.failure_candidate,k);
                if(!bank&&fixture.world.failure==100+static_cast<int>(bg::SurfaceScope)){reason="unsupported_surface_stop";passed=true;break;}
                throw std::runtime_error("product_physics_failure_"+std::to_string(fixture.world.failure));
            }
            require(fixture.world.inspection.certified&&fixture.world.inspection.lower_gap_m>=-pr::overlap_limit_m&&fixture.world.contact_damping_loss>=0,"product_clearance_or_damping_gate");
            states.push_back(fixture.world.motor.physical.body);
        }
        require(!bank||touched,"bank_fixture_never_contacted");passed=true;++count.completed_cases;
    }catch(const std::exception& error){reason=error.what();++count.failed_cases;}
    evidence.cases<<suite<<','<<id<<','<<id<<','<<initial_gap<<','<<total<<','<<count.attempted_controls-before_count<<','<<count.accepted_controls-before_accepted
        <<','<<count.attempted_substeps-before_steps<<','<<(completed==total)<<','<<passed<<",0,0,0,"<<reason<<'\n';
    evidence.physics_case(suite,id,fixture.world);
    evidence.cases.flush();evidence.controls.flush();require(passed,reason);return states;
}
void half_case(int id,const std::vector<p::BodyState>& reference,Counts& count,Evidence& evidence) {
    ++count.attempted_cases;evidence.begin("half_step",id);require(reference.size()==201,"half_reference_incomplete");
    const auto before_count=count.attempted_controls,before_accepted=count.accepted_controls,before_steps=count.attempted_substeps;
    auto fixture=bank_fixture(id*2,count);std::string reason;int completed=0;double max_position=0,max_velocity=0,max_orientation=0;
    try {
        for(int k=1;k<=200;++k) {
            const float command[4]{};count.guard(20);++count.attempted_controls;const auto before=fixture.world.motor.attempted_substeps;
            const bool ok=pr::step(fixture.world,command,.0025f);count.attempted_substeps+=fixture.world.motor.attempted_substeps-before;
            count.observe(fixture.world);
            if(ok){++count.accepted_controls;++count.half_controls;++fixture.request.controls;++completed;}
            evidence.record("half_step",id,"product_half",k,fixture.world.motor,fixture.request,command,true,ok,fixture.world.minimum_control_gap_m,
                fixture.world.contact_spring_work,fixture.world.contact_damping_loss,fixture.world.contact_potential_J,&fixture.world);
            if(!ok){evidence.failure("half_step_physics_"+std::to_string(fixture.world.failure),fixture.world.failure_before,fixture.world.failure_candidate,k);throw std::runtime_error("half_step_physics_failure");}
            const auto expected=po::motion(reference[static_cast<size_t>(k)]),actual=po::motion(fixture.world.motor.physical.body);
            max_position=std::max(max_position,double(p::norm((expected.com_model-actual.com_model)*p::scale)));
            max_velocity=std::max(max_velocity,double(p::norm(expected.velocity_m_s-actual.velocity_m_s)));
            const auto& a=reference[static_cast<size_t>(k)].orientation;const auto& b=fixture.world.motor.physical.body.orientation;
            const double norm_a=sqrt(double(a.x)*a.x+double(a.y)*a.y+double(a.z)*a.z+double(a.w)*a.w);
            const double norm_b=sqrt(double(b.x)*b.x+double(b.y)*b.y+double(b.z)*b.z+double(b.w)*b.w);
            const double cosine=std::min(1.,std::abs(double(a.x)*b.x+double(a.y)*b.y+double(a.z)*b.z+double(a.w)*b.w)/(norm_a*norm_b));
            max_orientation=std::max(max_orientation,2*acos(cosine)*180/po::pi);
            require(max_position<=.00025&&max_velocity<=.00025&&max_orientation<=.5,"half_step_convergence_gate");
        }
        ++count.completed_cases;
    }catch(const std::exception& error){reason=error.what();++count.failed_cases;}
    evidence.cases<<"half_step,"<<id<<','<<id*2<<",0,200,"<<count.attempted_controls-before_count<<','<<count.accepted_controls-before_accepted
        <<','<<count.attempted_substeps-before_steps<<','<<(completed==200)<<','<<reason.empty()<<','<<max_position<<','<<max_velocity<<','<<max_orientation<<','<<reason<<'\n';
    evidence.physics_case("half_step",id,fixture.world);
    evidence.cases.flush();evidence.controls.flush();require(reason.empty(),reason);
}
Options options(int argc,char** argv) {
    Options out;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];if(arg=="--preflight")out.preflight=true;else if(arg=="--run")out.run=true;
        else if(arg=="--policy"&&i+1<argc)out.policy=argv[++i];else if(arg=="--out"&&i+1<argc)out.out=argv[++i];
        else if(arg=="--protocol"&&i+1<argc)out.protocol=argv[++i];else if(arg=="--long-reference"&&i+1<argc)out.reference=argv[++i];
        else throw std::runtime_error("unknown_or_incomplete_argument");
    }
    require(out.preflight!=out.run&&!out.policy.empty()&&!out.out.empty()&&!out.reference.empty(),"require_one_mode_policy_out_long_reference");
    if(out.run)require(!out.protocol.empty()&&fs::is_regular_file(out.protocol),"run_requires_frozen_protocol_file");
    return out;
}
}
int main(int argc,char** argv) {
    Options option;Counts counts;Package package;uint64_t protocol_hash=0;bool owns_output=false;
    try {
        option=options(argc,argv);require(!fs::exists(option.out)&&fs::create_directories(option.out),"output_identity_already_exists");owns_output=true;
        package=load_package(option.policy);const auto expected=expected_long_keys(option.reference);
        if(option.preflight) {
            std::ofstream fixture_evidence(option.out/"fixtures.csv");
            fixture_evidence<<std::setprecision(17)<<"suite,case,px_model,py_model,pz_model,qx,qy,qz,qw,gap_lower_m,gap_upper_m,max_body_y_m,certificate_visits,certified\n";
            const auto record_fixture=[&](const char* suite,int id,const pr::World& w) {
                const auto& b=w.motor.physical.body;fixture_evidence<<suite<<','<<id<<','<<b.position.x<<','<<b.position.y<<','<<b.position.z
                    <<','<<b.orientation.x<<','<<b.orientation.y<<','<<b.orientation.z<<','<<b.orientation.w<<','<<w.inspection.lower_gap_m<<','
                    <<w.inspection.upper_gap_m<<','<<w.inspection.max_body_y_m<<','<<w.inspection.visits<<','<<w.inspection.certified<<'\n';fixture_evidence.flush();
            };
            for(const auto& spec:retention_specs()) {
                auto pair=initialize_pair(spec);float a[41]{},b[41]{};
                require(old::observe_appended(pair.old_request,pair.baseline,pair.config,a)&&po::observe(pair.request,pair.candidate.motor.physical.body,
                    pair.candidate.motor.physical.held_targets,pair.candidate.motor.reference_rate,pair.candidate.motor.physical.last_actual_torques,b),"preflight_observation");
                for(int j=0;j<41;++j)require(a[j]==b[j],"preflight_initial_input_difference");
                require(body_difference(pair.baseline.physical.body,pair.candidate.motor.physical.body)==0,"preflight_initial_body_difference");
                record_fixture(spec.suite,spec.id,pair.candidate);
            }
            for(int id=0;id<8;++id){const auto fixture=bank_fixture(id,counts);require(fixture.world.prepared,"preflight_bank_fixture");record_fixture("bank_contact",id,fixture.world);}
            for(int id=0;id<4;++id){const auto fixture=outward_fixture(id);require(fixture.world.prepared,"preflight_outward_fixture");record_fixture("outward_swim",id,fixture.world);}
            write_result(option,counts,package,true,"zero_integration_preflight_passed");return 0;
        }
        std::ifstream protocol(option.protocol,std::ios::binary);const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(protocol)),{});
        require(!bytes.empty(),"empty_protocol_file");protocol_hash=cp::checksum(bytes);
        std::ofstream started(option.out/"run-started.json");started<<"{\"protocol_id\":\""<<protocol_id<<"\",\"attempts_allowed\":1,\"control_cap\":"<<control_cap<<",\"substep_cap\":"<<substep_cap<<",\"optimizer_steps_allowed\":0}\n";started.close();
        Evidence evidence(option.out);Keys old_keys,new_keys;
        for(const auto& spec:retention_specs())retention_case(spec,package,counts,evidence,old_keys,new_keys);
        require(old_keys==expected&&new_keys==expected,"retained_long_window_key_gate");
        require(counts.old_controls==30400&&counts.new_controls==30400,"retention_accounting_gate");
        std::array<std::vector<p::BodyState>,4> reference;
        for(int id=0;id<8;++id){auto states=product_case(true,id,package,counts,evidence);if(id%2==0)reference[static_cast<size_t>(id/2)]=std::move(states);}
        for(int id=0;id<4;++id)half_case(id,reference[static_cast<size_t>(id)],counts,evidence);
        for(int id=0;id<4;++id)product_case(false,id,package,counts,evidence);
        require(counts.bank_controls==1600&&counts.half_controls==800&&counts.attempted_controls<=control_cap&&counts.attempted_substeps<=substep_cap,"complete_protocol_accounting");
        require(cp::checksum(cp::read(option.policy.string()))==package.checksum,"saved_policy_changed");
        write_result(option,counts,package,true,"product_physics_and_retention_passed_navigation_unqualified",protocol_hash);return 0;
    }catch(const std::exception& error) {
        if(owns_output)write_result(option,counts,package,false,error.what(),protocol_hash);
        std::cerr<<error.what()<<'\n';return 2;
    }
}
