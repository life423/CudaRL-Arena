// Native frozen-candidate assessment. No CUDA allocation or optimizer exists here.
#include "rl/pond_ppo_contract.h"
#include "rl/pond_ppo_task.h"
#include "rl/pond_ppo_schedule.h"
#include "motor/saved_motor_contract.h"
#include "rl/learned_amplitude_math.h"
#include "nn/mlp_cpu.h"
#include "pond_candidate_identity.h"
#include <windows.h>
#include <bcrypt.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
namespace pp=cudarl::rl::pond_ppo;
namespace pt=cudarl::rl::pond_ppo_task;
namespace pr=cudarl::pond_runtime;
namespace po=cudarl::pond_motor_observation;
namespace pg=cudarl::pond_geometry_v1;
namespace p=cudarl::physics;
namespace cp=cudarl::checkpoint;
namespace nn=cudarl::nn;
namespace {
constexpr uint64_t evaluation_controls=36800,evaluation_substeps=368000;
constexpr uint64_t aggregate_controls=184000,aggregate_substeps=1840000,evidence_cap=1024ull*1024*1024;
constexpr double aggregate_seconds=900;
constexpr const char* protocol_id=pp::default_study_protocol_id;
static_assert(pr::version==pp::runtime_version,"Candidate evaluator must use the exact qualified runtime");
using Keys=std::set<std::pair<int,int>>;
void require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
std::vector<uint8_t> file(const fs::path& path){std::ifstream in(path,std::ios::binary);require(bool(in),"Missing file: "+path.string());return {(std::istreambuf_iterator<char>(in)),{}};}
uint64_t eval_source(){const std::string text=pond_candidate_source_sha256;return cp::checksum({text.begin(),text.end()});}
std::string escape(const std::string& text){std::string out;for(char c:text){if(c=='"'||c=='\\')out+='\\';if(c=='\n')out+="\\n";else if(c=='\r')out+="\\r";else out+=c;}return out;}
std::string sha256(const std::vector<uint8_t>& bytes) {
    require(bytes.size()<=ULONG_MAX,"SHA input too large");BCRYPT_ALG_HANDLE provider=nullptr;
    require(BCryptOpenAlgorithmProvider(&provider,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"SHA provider failed");
    std::array<unsigned char,32> digest{};const auto status=BCryptHash(provider,nullptr,0,const_cast<PUCHAR>(bytes.data()),static_cast<ULONG>(bytes.size()),digest.data(),32);
    BCryptCloseAlgorithmProvider(provider,0);require(status>=0,"SHA operation failed");
    std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto value:digest)out<<std::setw(2)<<unsigned(value);return out.str();
}
std::string scalar(const std::string& text,const std::string& name) {
    size_t i=0;const auto ws=[&](){while(i<text.size()&&std::isspace(static_cast<unsigned char>(text[i])))++i;};
    const auto quoted=[&](){require(i<text.size()&&text[i++]=='"',"Expected JSON string");std::string out;bool escaped=false;
        while(i<text.size()){const char c=text[i++];if(escaped){out+=c;escaped=false;}else if(c=='\\')escaped=true;else if(c=='"')return out;else out+=c;}throw std::runtime_error("Unclosed JSON string");};
    ws();require(i<text.size()&&text[i++]=='{',"Expected root JSON object");
    while(i<text.size()){ws();if(text[i]=='}')break;const auto key=quoted();ws();require(i<text.size()&&text[i++]==':',"Expected JSON colon");ws();std::string value;
        if(i<text.size()&&text[i]=='"')value=quoted();
        else {const auto begin=i;int depth=0;bool in_string=false,escaped=false;while(i<text.size()){const char c=text[i];
            if(in_string){if(escaped)escaped=false;else if(c=='\\')escaped=true;else if(c=='"')in_string=false;}
            else if(c=='"')in_string=true;else if(c=='{'||c=='[')++depth;else if(c=='}'||c==']'){if(depth==0)break;--depth;}else if(c==','&&depth==0)break;++i;}
            value=text.substr(begin,i-begin);while(!value.empty()&&std::isspace(static_cast<unsigned char>(value.back())))value.pop_back();}
        if(key==name)return value;ws();if(i<text.size()&&text[i]==','){++i;continue;}break;}
    throw std::runtime_error("Missing root JSON field: "+name);
}
struct Options {bool describe=false,self_test=false,preflight=false,run=false,baseline=false,final_partial=false;fs::path parent,candidate,protocol,out,ledger,reference;std::string study_id=protocol_id;};
std::string validated_study_id(const std::string& text){const auto id=scalar(text,"protocol_id");
    require(id==pp::default_study_protocol_id||id==pp::curriculum_study_protocol_id||id==pp::course_study_protocol_id||id==pp::repeated_study_protocol_id,"Wrong evaluation protocol");
    if(id==pp::curriculum_study_protocol_id)require(scalar(text,"turn_curriculum_program")=="paired-gentle-moderate-v1","Unbound curriculum study");
    if(id==pp::course_study_protocol_id)require(scalar(text,"study_program")=="course-direction-v1","Unbound course-reward study");
    if(id==pp::repeated_study_protocol_id)require(scalar(text,"study_program")=="repeated-headings-v1","Unbound repeated-request study");return id;}
uint32_t expected_task_stride(const std::string& id) {
    return static_cast<uint32_t>(id==pp::repeated_study_protocol_id?sizeof(cudarl::rl::pond_ppo_schedule::SavedTask):sizeof(pt::Task));
}
Options options(int argc,char** argv) {
    Options out;for(int i=1;i<argc;++i){const std::string arg=argv[i];const auto next=[&](){require(i+1<argc,"Missing argument value");return fs::path(argv[++i]);};
        if(arg=="--describe")out.describe=true;else if(arg=="--self-test")out.self_test=true;else if(arg=="--preflight")out.preflight=true;else if(arg=="--run")out.run=true;else if(arg=="--baseline")out.baseline=true;else if(arg=="--final-partial")out.final_partial=true;
        else if(arg=="--parent")out.parent=next();else if(arg=="--candidate")out.candidate=next();else if(arg=="--protocol")out.protocol=next();
        else if(arg=="--out")out.out=next();else if(arg=="--ledger")out.ledger=next();else if(arg=="--long-reference")out.reference=next();else throw std::runtime_error("Unknown argument: "+arg);}
    require(int(out.describe)+int(out.self_test)+int(out.preflight)+int(out.run)==1,"Choose one mode");
    require(!(out.baseline&&out.final_partial),"Baseline and final-partial modes are exclusive");
    if(!out.describe&&!out.self_test)require(!out.parent.empty()&&!out.candidate.empty()&&!out.protocol.empty()&&!out.out.empty()&&!out.reference.empty(),"Candidate, parent, protocol, output and reference required");
    if(out.run)require(!out.ledger.empty(),"Aggregate ledger required");return out;
}
struct Parent {uint64_t checksum=0;float gain=0;};
Parent load_parent(const fs::path& path) {
    require(sha256(file(path))==pp::parent_package_sha256,"Parent package bytes changed");const auto bytes=cp::read(path.string());cp::Reader in{bytes};Parent out;out.checksum=cp::checksum(bytes);
    for(uint32_t value:{0x31564c48u,2u,6u,6u,3u,4u,41u,4u,336u,1u,7u,1u})require(in.value<uint32_t>()==value,"Parent package contract changed");
    require(in.value<uint64_t>()!=0,"Parent protocol missing");const auto parent=in.array<uint8_t>(64);
    require(std::string(parent.begin(),parent.end())=="810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0","Parent checkpoint mismatch");
    const auto& hashes=cudarl::motor::saved_motor_source_contract_hashes();require(in.value<uint32_t>()==hashes.size(),"Parent contract count");
    for(auto hash:hashes){const auto bytes_hash=in.array<uint8_t>(64);require(std::string(bytes_hash.begin(),bytes_hash.end())==hash,"Parent provenance mismatch");}
    const auto actor=in.array<float>(pp::actor_weights),head=in.array<float>(2);const float gain=in.value<float>(),sigma=in.value<float>();in.finish();
    require(actor.size()==pp::actor_weights&&head.size()==2,"Parent parameter count");out.gain=cudarl::learned_amplitude::sigmoid(head[0]);
    require(out.gain==pp::frozen_holding_gain&&std::abs(gain-out.gain)<=2e-7f&&std::abs(sigma-cudarl::learned_amplitude::standard_deviation(head[1]))<=2e-6f,"Parent gain changed");return out;
}
struct Ledger {uint64_t parent=0,protocol=0,source=0,mask=0,controls=0,substeps=0,bytes=0;double seconds=0;bool active=false,closed=false;};
void save_ledger(const fs::path& path,const Ledger& ledger) {
    cp::Writer out;out.value(uint32_t{0x314c4550});out.value(uint32_t{2});for(uint64_t value:{ledger.parent,ledger.protocol,ledger.source,ledger.mask,ledger.controls,ledger.substeps,ledger.bytes})out.value(value);
    out.value(ledger.seconds);out.value(uint32_t{ledger.active?1u:0u});out.value(uint32_t{ledger.closed?1u:0u});cp::write_atomic(path.string(),out.data);
}
Ledger load_ledger(const fs::path& path,uint64_t parent,uint64_t protocol) {
    Ledger out;out.parent=parent;out.protocol=protocol;out.source=eval_source();if(!fs::exists(path))return out;
    const auto bytes=cp::read(path.string());cp::Reader in{bytes};require(in.value<uint32_t>()==0x314c4550&&in.value<uint32_t>()==2,"Evaluation ledger format");
    out.parent=in.value<uint64_t>();out.protocol=in.value<uint64_t>();out.source=in.value<uint64_t>();out.mask=in.value<uint64_t>();out.controls=in.value<uint64_t>();out.substeps=in.value<uint64_t>();out.bytes=in.value<uint64_t>();out.seconds=in.value<double>();out.active=in.value<uint32_t>()!=0;out.closed=in.value<uint32_t>()!=0;in.finish();
    require(out.parent==parent&&out.protocol==protocol&&out.source==eval_source()&&!out.active&&!out.closed,"Evaluation ledger identity, interrupted attempt or final assessment closed");
    require(out.controls<=aggregate_controls&&out.substeps<=aggregate_substeps&&out.bytes<=evidence_cap&&std::isfinite(out.seconds)&&out.seconds<=aggregate_seconds,"Evaluation ledger caps");return out;
}
int assessment_slot(uint64_t update,bool baseline,bool final_partial,uint64_t prior_mask) {
    require(baseline==(update==0),"Candidate0 requires explicit --baseline; trained checkpoints forbid it");
    const bool scheduled=update==0||update==8||update==16||update==24||update==32;
    require(final_partial?(!scheduled&&update>0&&update<32):scheduled,"Checkpoint is outside predeclared evaluation schedule");
    const int slot=final_partial?static_cast<int>(update/8)+1:update==0?0:static_cast<int>(update/8);
    require(slot<5&&prior_mask==(uint64_t{1}<<slot)-1,"Evaluation schedule repeated, skipped or reordered");return slot;
}
struct StoppedTraining {
    uint64_t checkpoint=0;pp::Counters counters{};double seconds=0;uint64_t bytes=0;bool accounting_known=false;uint64_t controls_upper=0,substeps_upper=0;
};
StoppedTraining stopped_training(const std::vector<uint8_t>& payload,uint64_t protocol,uint64_t source,uint64_t checkpoint,const pp::CandidateState& candidate) {
    cp::Reader in{payload};require(in.value<uint32_t>()==0x314c5050&&in.value<uint32_t>()==1,"Stopped training ledger format");
    require(in.value<uint64_t>()==protocol&&in.value<uint64_t>()==source,"Stopped training ledger source/protocol mismatch");
    require(in.value<uint32_t>()==6,"Final-partial assessment requires STOPPED training, not paused/active/complete");
    StoppedTraining out;out.checkpoint=in.value<uint64_t>();out.counters=in.value<pp::Counters>();out.seconds=in.value<double>();out.bytes=in.value<uint64_t>();
    const auto known=in.value<uint32_t>();require(known<=1,"Stopped training accounting flag");out.accounting_known=known==1;
    out.controls_upper=in.value<uint64_t>();out.substeps_upper=in.value<uint64_t>();in.finish();
    require(checkpoint!=0&&out.checkpoint==checkpoint,"Final-partial candidate is not the exact last-good stopped checkpoint");
    const auto& saved=candidate.counters;
    require(out.counters.completed_updates==saved.completed_updates,"Stopped training completed update does not match candidate");
    require(out.counters.attempted_controls>=saved.attempted_controls&&out.counters.attempted_controls<=pp::total_controls&&
        out.counters.attempted_substeps>=saved.attempted_substeps&&out.counters.attempted_substeps<=pp::total_controls*10&&
        out.counters.actor_adam_calls>=saved.actor_adam_calls&&out.counters.actor_adam_calls<=pp::total_adam_calls/2&&
        out.counters.critic_adam_calls>=saved.critic_adam_calls&&out.counters.critic_adam_calls<=pp::total_adam_calls/2,"Stopped training accounting is inconsistent");
    require(std::isfinite(out.seconds)&&out.seconds>=candidate.wall_seconds_used&&out.bytes>=candidate.evidence_bytes_used&&
        out.controls_upper>=out.counters.attempted_controls&&out.controls_upper<=pp::total_controls&&
        out.substeps_upper>=out.counters.attempted_substeps&&out.substeps_upper<=pp::total_controls*10,"Stopped training resource/accounting bounds invalid");
    require(!out.accounting_known||(out.controls_upper==out.counters.attempted_controls&&out.substeps_upper==out.counters.attempted_substeps),"Known stopped accounting differs from exact counters");return out;
}
struct StoppedTrainingLease {
    fs::path path,lock;uint64_t checksum=0;bool locked=false;
    StoppedTrainingLease(const fs::path& bound_path,bool acquire):path(bound_path),lock(bound_path.string()+".running") {
        require(path.is_absolute(),"Stopped training ledger path must be protocol-bound and absolute");
        if(acquire){require(fs::create_directory(lock),"Trainer is active or its process state is unresolved");locked=true;}
        else require(!fs::exists(lock),"Trainer is active or its process state is unresolved");
        try{checksum=cp::checksum(cp::read(path.string()));}catch(...){if(locked){std::error_code ignored;fs::remove(lock,ignored);}throw;}
    }
    void unchanged()const{require(cp::checksum(cp::read(path.string()))==checksum,"Stopped training ledger changed during assessment");}
    ~StoppedTrainingLease(){if(locked){std::error_code ignored;fs::remove(lock,ignored);}}
};
void assessment_pure_tests(uint32_t& assertions);
void pure_tests() {
    uint32_t assertions=0;const auto check=[&](bool value){require(value,"Pure evaluator schedule test failed");++assertions;};
    const auto rejects=[&](const auto& fn){bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}check(rejected);};
    for(uint64_t update:{0ull,8ull,16ull,24ull,32ull}) {
        const int slot=static_cast<int>(update/8);check(assessment_slot(update,update==0,false,(uint64_t{1}<<slot)-1)==slot);
        rejects([&](){assessment_slot(update,update==0,true,(uint64_t{1}<<slot)-1);});
        rejects([&](){assessment_slot(update,update==0,false,uint64_t{1}<<slot);});
    }
    for(uint64_t update=1;update<32;++update)if(update%8!=0) {
        const int slot=static_cast<int>(update/8)+1;const uint64_t mask=(uint64_t{1}<<slot)-1;
        check(assessment_slot(update,false,true,mask)==slot);
        rejects([&](){assessment_slot(update,false,false,mask);});
        rejects([&](){assessment_slot(update,false,true,mask>>1);});
    }
    pp::CandidateState candidate;candidate.counters.completed_updates=12;candidate.counters.attempted_controls=12*pp::controls_per_update;
    candidate.counters.attempted_substeps=candidate.counters.attempted_controls*10;
    candidate.counters.actor_adam_calls=candidate.counters.critic_adam_calls=12*pp::adam_calls_per_update;
    candidate.wall_seconds_used=100;candidate.evidence_bytes_used=1000;
    const auto payload=[&](uint32_t status,uint64_t checkpoint,uint64_t update,uint64_t protocol=17,uint64_t source=19) {
        cp::Writer out;out.value(uint32_t{0x314c5050});out.value(uint32_t{1});out.value(protocol);out.value(source);out.value(status);out.value(checkpoint);
        auto counters=candidate.counters;counters.completed_updates=update;counters.attempted_controls+=512;counters.attempted_substeps+=5120;
        out.value(counters);out.value(110.);out.value(uint64_t{2000});out.value(uint32_t{1});out.value(counters.attempted_controls);out.value(counters.attempted_substeps);return out.data;
    };
    check(stopped_training(payload(6,23,12),17,19,23,candidate).counters.completed_updates==12);
    for(uint32_t status:{2u,3u,4u,5u})rejects([&](){stopped_training(payload(status,23,12),17,19,23,candidate);});
    rejects([&](){stopped_training(payload(6,0,12),17,19,23,candidate);});
    rejects([&](){stopped_training(payload(6,22,12),17,19,23,candidate);});
    rejects([&](){stopped_training(payload(6,23,13),17,19,23,candidate);});
    rejects([&](){stopped_training(payload(6,23,12,18),17,19,23,candidate);});
    rejects([&](){stopped_training(payload(6,23,12,17,20),17,19,23,candidate);});
    auto truncated=payload(6,23,12);truncated.pop_back();rejects([&](){stopped_training(truncated,17,19,23,candidate);});
    check(validated_study_id(R"({"protocol_id":"pond-motor-ppo-v4-repeated-2026-11-24","study_program":"repeated-headings-v1"})")==pp::repeated_study_protocol_id);
    rejects([&](){validated_study_id(R"({"protocol_id":"pond-motor-ppo-v4-repeated-2026-11-24","study_program":"course-direction-v1"})");});
    check(expected_task_stride(pp::default_study_protocol_id)==sizeof(pt::Task)&&expected_task_stride(pp::repeated_study_protocol_id)==sizeof(cudarl::rl::pond_ppo_schedule::SavedTask)&&sizeof(cudarl::rl::pond_ppo_schedule::SavedTask)>sizeof(pt::Task));
    assessment_pure_tests(assertions);
    std::cout<<"{\"passed\":true,\"assertions\":"<<assertions<<",\"physical_controls\":0,\"physical_substeps\":0,\"CUDA_kernels\":0,\"optimizer_steps\":0}\n";
}
struct Budget {
    uint64_t controls=0,substeps=0,retention_controls=0,retention_substeps=0,turn_controls=0,turn_substeps=0;
    std::chrono::steady_clock::time_point began=std::chrono::steady_clock::now();Ledger prior;
    double seconds()const{return std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();}
    void guard()const{require(controls<evaluation_controls&&substeps+10<=evaluation_substeps&&prior.controls+controls<aggregate_controls&&prior.substeps+substeps+10<=aggregate_substeps,"Evaluation physical cap");require(prior.seconds+seconds()<aggregate_seconds,"Aggregate evaluation wall cap");}
};
struct Spec {const char* suite;int id,pose,controls;float offset=0,turn_degrees=0;bool holding=false,long_swim=false,turn=false;};
std::vector<Spec> specs() {
    std::vector<Spec> out;for(int pose=0;pose<4;++pose)out.push_back({"short",pose,pose,160});
    for(int pose:{0,3}){Spec c{"long",pose,pose,1200};c.long_swim=true;out.push_back(c);}
    for(int id=0;id<6;++id){Spec c{"original_hold",id,id%2?3:0,1280};c.offset=id/2==0?0.f:id/2==1?-.0005f:.0005f;c.holding=true;out.push_back(c);}
    int id=0;for(float depth:{0.f,-.00025f,.00025f})for(int pose:{1,2}){Spec c{"withheld_hold",id++,pose,1280};c.offset=depth;c.holding=true;out.push_back(c);}
    for(int pose:{0,3}){Spec c{"endurance_hold",pose,pose,6000};c.holding=true;out.push_back(c);}
    id=0;for(float sign:{-1.f,1.f})for(float angle:{5.f,15.f,30.f,60.f,90.f}){Spec c{"turn",id++,0,400};c.turn=true;c.turn_degrees=sign*angle;out.push_back(c);}
    for(float sign:{-1.f,1.f})for(float angle:{30.f,60.f,90.f}){Spec c{"turn",id++,3,400};c.turn=true;c.turn_degrees=sign*angle;out.push_back(c);}
    return out;
}
struct Fixture {pr::World world;pt::Task task;};
Fixture fixture(const Spec& spec) {
    Fixture out;auto& state=out.world.motor.physical;require(po::initialize(spec.pose,false,state.body,state.held_targets,out.world.motor.reference_rate,state.last_actual_torques,out.task.request),"Evaluation fixture initialization");
    state.paused=false;out.task.kind=spec.holding?pt::Hold:spec.turn?pt::Turn:pt::Straight;out.task.turn_radians=spec.turn_degrees*po::pi/180.f;
    if(spec.holding){out.task.request.holding=true;out.task.request.request.terminal_velocity=out.task.request.request.heading*0.f;
        if(spec.offset!=0)state.body.position.y+=spec.offset/p::scale;out.task.request.hold_anchor_model=po::motion(state.body).com_model;
        out.task.request.hold_anchor_model.y=out.task.request.start_com_model.y;out.task.request.hold_anchor_set=true;}
    if(!spec.turn){out.world.terrain.water_radius_x_m=.9;out.world.terrain.water_radius_z_m=.7;out.world.terrain.flat_radius_x_m=.65;out.world.terrain.flat_radius_z_m=.45;}
    require(pr::prepare(out.world),"Evaluation fixture prepare");return out;
}
struct HoldScore {
    int total,next=0;float speed=0,anchor=0,depth=0,lane=0,tilt=0,window=0,excursion=0;
    explicit HoldScore(int controls):total(controls){}
    void record(int control,const po::Metrics& sample) {
        require(control==next++&&control<=total,"Hold record order");
        for(float value:{p::norm(sample.velocity_m_s),p::norm(sample.hold_error_m),sample.depth_m,sample.lateral_m,sample.tilt_deg,sample.forward_m})require(std::isfinite(value),"Nonfinite hold metric");
        speed=std::max(speed,p::norm(sample.velocity_m_s));anchor=std::max(anchor,p::norm(sample.hold_error_m));
        depth=std::max(depth,sample.depth_m);lane=std::max(lane,sample.lateral_m);tilt=std::max(tilt,sample.tilt_deg);
        if(control==total-160)window=sample.forward_m;if(control>=total-160)excursion=std::max(excursion,std::abs(sample.forward_m-window));
    }
    bool passed()const{return next==total+1&&speed<=.0005f&&anchor<=.002f&&depth<=.001f&&lane<=.002f&&tilt<=10.f&&excursion<=.002f;}
};
struct SwimScore {
    float lane=0,depth=0,tilt=0,window_work=0;double error=0,window_error=0;p::V origin{},window_origin{};Keys keys;
    void record(int control,const po::Metrics& z,const p::BodyState& body,p::V heading,bool windows,int pose,std::ostream& trace) {
        lane=std::max(lane,z.lateral_m);depth=std::max(depth,z.depth_m);tilt=std::max(tilt,z.tilt_deg);const auto center=po::motion(body).com_model;
        if(control==40){origin=window_origin=center;window_work=body.work;}
        if(control>40){error+=z.speed_error_m_s*z.speed_error_m_s;window_error+=z.speed_error_m_s*z.speed_error_m_s;}
        if(windows&&control%80==0){const auto d=(center-window_origin)*p::scale;const float forward=p::dot(d,heading);const auto side=d-heading*forward;const int n=control==80?40:80;
            const float speed=forward/(n*p::dt),rmse=sqrtf(static_cast<float>(window_error/n)),angle=atan2f(p::norm({side.x,0,side.z}),forward)*180/po::pi,work=(body.work-window_work)/fmaxf(forward,1e-9f);
            const bool pass=speed>=.9f*.008f&&speed<=1.1f*.008f&&rmse<=.3125f*.008f&&lane<=.002f&&depth<=.001f&&tilt<=10.f&&angle<=5&&work<=.001f;
            if(pass)keys.emplace(pose,control/20);trace<<pose<<','<<control/20<<','<<speed<<','<<rmse<<','<<lane<<','<<depth<<','<<tilt<<','<<angle<<','<<work<<','<<pass<<'\n';
            window_origin=center;window_work=body.work;window_error=0;}
    }
    bool passed(int controls,const p::BodyState& body,p::V heading,float total_forward)const {
        const auto d=(po::motion(body).com_model-origin)*p::scale;const float forward=p::dot(d,heading);const auto side=d-heading*forward;
        const float speed=forward/(float(controls-40)*p::dt),rmse=sqrtf(static_cast<float>(error/(controls-40))),angle=atan2f(p::norm({side.x,0,side.z}),forward)*180/po::pi;
        return speed>=.9f*.008f&&speed<=1.1f*.008f&&rmse<=.3125f*.008f&&lane<=.002f&&depth<=.001f&&tilt<=10.f&&angle<=5&&body.work/fmaxf(total_forward,1e-9f)<=.001f;
    }
};
Keys long_reference(const fs::path& path) {
    std::ifstream in(path);require(bool(in),"Long-window reference missing");std::string line;std::getline(in,line);Keys all,passed;
    while(std::getline(in,line)){std::stringstream row(line);std::vector<double> values;std::string field;while(std::getline(row,field,','))values.push_back(std::stod(field));require(values.size()==10,"Long-window reference shape");
        const int pose=static_cast<int>(values[0]),end=static_cast<int>(values[1]);require((pose==0||pose==3)&&end>=4&&end<=60&&end%4==0&&all.emplace(pose,end).second,"Long-window reference keys");if(values[9]==1)passed.emplace(pose,end);}
    require(all.size()==30&&passed.size()==28,"Original long-window passes changed");return passed;
}
struct TurnScore {
    double late_x=0,late_y=0,late_z=0,arc_m=0;int late_samples=0;
    float maximum_depth=0,maximum_tilt=0;double minimum_gap_m=1e30;
    p::V previous_com{};bool previous_set=false;
    void record(int control,const po::Metrics& z,const p::BodyState& body,double gap) {
        maximum_depth=std::max(maximum_depth,z.depth_m);maximum_tilt=std::max(maximum_tilt,z.tilt_deg);
        minimum_gap_m=std::min(minimum_gap_m,gap);const auto com=po::motion(body).com_model*p::scale;
        if(previous_set&&control>40){const auto d=com-previous_com;arc_m+=std::hypot(double(d.x),double(d.z));}
        previous_com=com;previous_set=true;
        if(control>320){late_x+=z.velocity_m_s.x;late_y+=z.velocity_m_s.y;late_z+=z.velocity_m_s.z;++late_samples;}
    }
    double horizontal_speed()const{return late_samples?std::hypot(late_x,late_z)/late_samples:0;}
    double course_error(p::V heading)const {
        if(late_samples==0||horizontal_speed()<1e-9)return 180;
        const double actual=atan2(-late_z,late_x),wanted=atan2(-double(heading.z),double(heading.x));
        return std::abs(atan2(sin(actual-wanted),cos(actual-wanted)))*180/po::pi;
    }
    double projection(p::V heading)const{return late_samples?(late_x*heading.x+late_z*heading.z)/late_samples:0;}
    bool passed(int completed,p::V heading)const{return completed==400&&late_samples==800&&projection(heading)>=.006&&course_error(heading)<=5&&maximum_depth<=.001f&&maximum_tilt<=10.f;}
};
struct Output {
    fs::path root;std::ofstream controls,cases,windows,turn_substeps;std::set<std::pair<std::string,int>> attempted;
    explicit Output(const fs::path& path):root(path),controls(path/"controls.csv"),cases(path/"cases.csv"),windows(path/"long-windows.csv"),turn_substeps(path/"turn-substeps.csv") {
        for(auto* stream:{&controls,&cases,&windows,&turn_substeps})*stream<<std::setprecision(17);
        controls<<"suite,case,control,time_s,command0,command1,command2,command3,COM_vx,COM_vy,COM_vz,forward_m,lane_m,depth_m,tilt_deg,anchor_error_m,minimum_control_gap_m,minimum_lifetime_gap_m,heading_x,heading_y,heading_z,energy_residual_J,passive_excess_J";
        body_header(controls);for(const char* name:{"held","reference_rate","last_actual"})for(int j=0;j<4;++j)controls<<','<<name<<j;controls<<'\n';
        cases<<"suite,case,pose,offset_m,turn_degrees,planned_controls,attempted_controls,accepted_controls,attempted_substeps,completed,passed,hard_failure,late_horizontal_speed_m_s,late_projected_speed_m_s,late_course_error_deg,max_depth_m,max_tilt_deg,minimum_body_gap_m,arc_m,reason\n";
        windows<<"pose,end_s,mean_m_s,RMSE_m_s,lateral_m,depth_m,tilt_deg,path_deg,work_J_per_m,passed\n";
        turn_substeps<<"case,control,substep,depth_m,tilt_deg,COM_vx,COM_vy,COM_vz,minimum_body_gap_m";body_header(turn_substeps);turn_substeps<<'\n';
        require(bool(controls)&&bool(cases)&&bool(windows)&&bool(turn_substeps),"Evaluation evidence open failed");
    }
    static void body_header(std::ostream& out) {
        for(const char* name:{"px_model","py_model","pz_model","qx","qy","qz","qw"})out<<','<<name;
        for(int j=0;j<4;++j)out<<",joint"<<j;for(int j=0;j<10;++j)out<<",velocity"<<j;out<<",work_J,travel_model";
    }
    static void body(std::ostream& out,const p::BodyState& value) {
        out<<','<<value.position.x<<','<<value.position.y<<','<<value.position.z<<','<<value.orientation.x<<','<<value.orientation.y<<','<<value.orientation.z<<','<<value.orientation.w;
        for(float v:value.angle)out<<','<<v;for(float v:value.velocity)out<<','<<v;out<<','<<value.work<<','<<value.travel;
    }
    void evidence_guard(uint64_t prior) {
        uint64_t bytes=0;for(auto* stream:{&controls,&cases,&windows,&turn_substeps}){const auto position=stream->tellp();require(position>=0,"Evidence stream position failed");bytes+=static_cast<uint64_t>(position);}
        require(prior+bytes+1024*1024<=evidence_cap,"Aggregate evaluator evidence cap");
    }
    void failure(const Spec& spec,int control,const Fixture& f) {
        const std::string stem=std::string(spec.suite)+"-"+std::to_string(spec.id)+"-control-"+std::to_string(control);
        cp::Writer out;out.value(uint32_t{0x31464350});out.value(uint32_t{1});out.value(control);out.value(f.world);out.value(f.task);
        cp::write_atomic((root/(stem+"-native.bin")).string(),out.data);
    }
    ~Output() {
        for(const auto& spec:specs())if(attempted.count({spec.suite,spec.id})==0)
            cases<<spec.suite<<','<<spec.id<<','<<spec.pose<<','<<spec.offset<<','<<spec.turn_degrees<<','<<spec.controls<<",0,0,0,0,0,0,0,0,180,0,0,0,0,skipped_after_stop\n";
    }
};
struct Outcome {bool completed=false,passed=false,hard=false;std::string reason;Keys long_keys;};
Outcome evaluate(const Spec& spec,const pp::CandidateState& candidate,nn::CpuMlp<float>& actor,Budget& budget,Output& output,const Keys& expected_keys) {
    output.attempted.emplace(spec.suite,spec.id);auto f=fixture(spec);HoldScore hold(spec.controls);SwimScore swim;TurnScore turn;Outcome result;
    const uint64_t before_controls=budget.controls,before_substeps=budget.substeps;int accepted=0;
    const auto first=po::measure(f.task.request,f.world.motor.physical.body);if(spec.holding)hold.record(0,first);
    try {
        for(int control=1;control<=spec.controls;++control) {
            float observation[41]{},command[4]{};
            require(pt::observe(f.task,f.world,observation),"Candidate observation failed");actor.forward(observation,1);
            for(int j=0;j<4;++j)command[j]=tanhf((spec.holding?candidate.frozen_gain:1.f)*actor.output()[j]);
            budget.guard();++budget.controls;const auto prior=f.world.motor.attempted_substeps;
            pr::SubstepTrace trace[20];const bool advanced=pr::step(f.world,command,.005f,spec.turn?trace:nullptr);
            const auto steps=f.world.motor.attempted_substeps-prior;budget.substeps+=steps;
            if(spec.turn){++budget.turn_controls;budget.turn_substeps+=steps;}else{++budget.retention_controls;budget.retention_substeps+=steps;}
            if(!advanced) {
                output.failure(spec,control,f);result.reason="runtime_stop_"+std::to_string(f.world.failure);
                result.hard=!(spec.turn&&(f.world.failure==100+static_cast<int>(cudarl::pond_body_geometry::SurfaceScope)||
                    f.world.failure==100+static_cast<int>(cudarl::pond_body_geometry::DryLandScope)));
                break;
            }
            ++accepted;++f.task.episode_controls;++f.task.request.controls;
            const auto metrics=po::measure(f.task.request,f.world.motor.physical.body);
            if(spec.holding)hold.record(control,metrics);
            else if(spec.turn) {
                for(int substep=0;substep<10;++substep) {
                    require(trace[substep].recorded&&trace[substep].accepted,"Missing accepted 5ms turn trace");
                    const auto z=po::measure(f.task.request,trace[substep].after);
                    turn.record(control,z,trace[substep].after,trace[substep].minimum_gap_m);
                    output.turn_substeps<<spec.id<<','<<control<<','<<substep+1<<','<<z.depth_m<<','<<z.tilt_deg
                        <<','<<z.velocity_m_s.x<<','<<z.velocity_m_s.y<<','<<z.velocity_m_s.z<<','<<trace[substep].minimum_gap_m;
                    Output::body(output.turn_substeps,trace[substep].after);output.turn_substeps<<'\n';
                }
            } else swim.record(control,metrics,f.world.motor.physical.body,f.task.request.request.heading,spec.long_swim,spec.pose,output.windows);
            output.controls<<spec.suite<<','<<spec.id<<','<<control<<','<<control*.05;
            for(float value:command)output.controls<<','<<value;
            output.controls<<','<<metrics.velocity_m_s.x<<','<<metrics.velocity_m_s.y<<','<<metrics.velocity_m_s.z<<','<<metrics.forward_m<<','<<metrics.lateral_m
                <<','<<metrics.depth_m<<','<<metrics.tilt_deg<<','<<p::norm(metrics.hold_error_m)<<','<<f.world.minimum_control_gap_m<<','<<f.world.minimum_gap_m;
            const auto heading=f.task.request.request.heading;output.controls<<','<<heading.x<<','<<heading.y<<','<<heading.z<<','<<f.world.motor.last_energy_error<<','<<f.world.maximum_passive_energy_excess_J;
            Output::body(output.controls,f.world.motor.physical.body);
            for(const auto* values:{f.world.motor.physical.held_targets,f.world.motor.reference_rate,f.world.motor.physical.last_actual_torques})for(int j=0;j<4;++j)output.controls<<','<<values[j];
            output.controls<<'\n';require(bool(output.controls)&&bool(output.windows)&&bool(output.turn_substeps),"Evaluation evidence write failed");output.evidence_guard(budget.prior.bytes);
            // These are the training task's genuine depth/tilt terminal limits.
            // The stricter movement qualifications remain separate below.
            if(metrics.depth_m>.005f||metrics.tilt_deg>30.f) {result.reason=metrics.depth_m>.005f?"depth_task_terminal":"tilt_task_terminal";break;}
        }
        result.completed=accepted==spec.controls;
        if(spec.holding)result.passed=result.completed&&hold.passed();
        else if(spec.turn)result.passed=turn.passed(accepted,f.task.request.request.heading);
        else if(spec.long_swim){
            result.long_keys=swim.keys;Keys required;for(const auto& key:expected_keys)if(key.first==spec.pose)required.insert(key);
            result.passed=result.completed&&std::includes(swim.keys.begin(),swim.keys.end(),required.begin(),required.end());}
        else result.passed=result.completed&&swim.passed(spec.controls,f.world.motor.physical.body,f.task.request.request.heading,po::measure(f.task.request,f.world.motor.physical.body).forward_m);
        if(!result.passed&&result.reason.empty())result.reason="movement_qualification_failed";
    }catch(const std::exception& error){result.reason=error.what();result.hard=true;output.failure(spec,accepted+1,f);}
    output.cases<<spec.suite<<','<<spec.id<<','<<spec.pose<<','<<spec.offset<<','<<spec.turn_degrees<<','<<spec.controls<<','<<budget.controls-before_controls
        <<','<<accepted<<','<<budget.substeps-before_substeps<<','<<result.completed<<','<<result.passed<<','<<result.hard
        <<','<<turn.horizontal_speed()<<','<<turn.projection(f.task.request.request.heading)<<','<<turn.course_error(f.task.request.request.heading)
        <<','<<(spec.turn?turn.maximum_depth:spec.holding?hold.depth:swim.depth)<<','<<(spec.turn?turn.maximum_tilt:spec.holding?hold.tilt:swim.tilt)
        <<','<<f.world.minimum_gap_m<<','<<turn.arc_m<<','<<result.reason<<'\n';
    output.cases.flush();output.controls.flush();output.windows.flush();output.turn_substeps.flush();require(bool(output.cases)&&bool(output.controls)&&bool(output.windows)&&bool(output.turn_substeps),"Evaluation flush failed");
    return result;
}
struct Summary {
    uint32_t retention_passed=0,retention_attempted=0,turn_passed=0,turn_attempted=0,actual_long_passes=0,retained_long_passes=0;
    uint32_t retention_completed=0,turn_completed=0,hold_passed=0,hold_attempted=0,short_passed=0,short_attempted=0,long_passed=0,long_attempted=0;
    uint64_t stopped_training_checksum=0;bool retention=false,hard=false;std::string reason,first_retention_failure;
};
bool retention_qualified(const Summary& summary,const Budget& budget) {
    return !summary.hard&&summary.retention_attempted==20&&summary.retention_completed==20&&summary.retention_passed==20&&
        summary.hold_attempted==14&&summary.hold_passed==14&&summary.short_attempted==4&&summary.short_passed==4&&
        summary.long_attempted==2&&summary.long_passed==2&&summary.retained_long_passes==28&&
        budget.retention_controls==30400&&budget.retention_substeps==304000;
}
bool assessment_completed(const Summary& summary) {
    return summary.retention_attempted==20&&summary.turn_attempted==16&&!summary.hard;
}
void assessment_pure_tests(uint32_t& assertions) {
    const auto check=[&](bool passed){require(passed,"Pure complete-retention admission test failed");++assertions;};
    Summary good;good.retention_attempted=good.retention_completed=good.retention_passed=20;
    good.hold_attempted=good.hold_passed=14;good.short_attempted=good.short_passed=4;good.long_attempted=good.long_passed=2;
    good.retained_long_passes=28;good.turn_attempted=16;
    Budget budget;budget.retention_controls=30400;budget.retention_substeps=304000;
    check(retention_qualified(good,budget));check(assessment_completed(good));
    check(good.turn_passed==0&&retention_qualified(good,budget)); // Turning is the lesson, not a prerequisite.
    auto bad=good;bad.retention_passed=18;bad.long_passed=0;bad.retained_long_passes=6;
    check(!retention_qualified(bad,budget)&&assessment_completed(bad)); // All soft failures still get assessed.
    bad=good;bad.hold_passed=13;check(!retention_qualified(bad,budget));
    bad=good;bad.short_passed=3;check(!retention_qualified(bad,budget));
    bad=good;bad.long_passed=1;check(!retention_qualified(bad,budget));
    bad=good;bad.retained_long_passes=27;check(!retention_qualified(bad,budget));
    bad=good;bad.retention_completed=19;check(!retention_qualified(bad,budget));
    bad=good;bad.turn_attempted=15;check(!assessment_completed(bad));
    bad=good;bad.hard=true;check(!retention_qualified(bad,budget)&&!assessment_completed(bad));
    auto incomplete=budget;--incomplete.retention_controls;check(!retention_qualified(good,incomplete));
    incomplete=budget;--incomplete.retention_substeps;check(!retention_qualified(good,incomplete));
}
uint64_t directory_bytes(const fs::path& path) {uint64_t count=0;for(const auto& entry:fs::recursive_directory_iterator(path))if(entry.is_regular_file())count+=entry.file_size();return count;}
void write_result(const Options& options,const pp::CandidateState& candidate,const Budget& budget,const Summary& summary,bool preflight,uint64_t checkpoint) {
    std::ofstream out(options.out/(preflight?"preflight.json":"result.json"));
    out<<std::setprecision(17)<<"{\"schema\":2,\"protocol_id\":\""<<options.study_id<<"\",\"runtime_version\":"<<pp::runtime_version<<",\"query_arithmetic_version\":"<<pp::query_arithmetic_version<<",\"preflight_only\":"<<(preflight?"true":"false")
        <<",\"passed\":"<<((preflight||(summary.retention&&assessment_completed(summary)))&&!summary.hard?"true":"false")
        <<",\"baseline_only\":"<<(options.baseline?"true":"false")<<",\"assessment_completed\":"<<(assessment_completed(summary)?"true":"false")
        <<",\"training_admission_qualified\":"<<(!preflight&&summary.retention&&assessment_completed(summary)?"true":"false")
        <<",\"final_partial\":"<<(options.final_partial?"true":"false")<<",\"stopped_training_ledger_checksum\":"<<summary.stopped_training_checksum
        <<",\"training_continuation_authorized\":false"
        <<",\"reason\":\""<<escape(summary.reason)
        <<"\",\"checkpoint_checksum\":"<<checkpoint<<",\"update\":"<<candidate.counters.completed_updates<<",\"eval_source_identity\":"<<eval_source()
        <<",\"candidate_actor_adam_calls\":"<<candidate.counters.actor_adam_calls<<",\"candidate_critic_adam_calls\":"<<candidate.counters.critic_adam_calls
        <<",\"retention_passed\":"<<(summary.retention?"true":"false")<<",\"retention_cases_passed\":"<<summary.retention_passed<<",\"retention_cases_attempted\":"<<summary.retention_attempted
        <<",\"retention_cases_total\":20,\"retention_controls\":"<<budget.retention_controls<<",\"retention_substeps\":"<<budget.retention_substeps
        <<",\"retention_cases_completed\":"<<summary.retention_completed<<",\"rest_cases_passed\":"<<summary.hold_passed<<",\"rest_cases_attempted\":"<<summary.hold_attempted<<",\"rest_cases_total\":14"
        <<",\"short_swim_cases_passed\":"<<summary.short_passed<<",\"short_swim_cases_attempted\":"<<summary.short_attempted<<",\"short_swim_cases_total\":4"
        <<",\"long_swim_cases_passed\":"<<summary.long_passed<<",\"long_swim_cases_attempted\":"<<summary.long_attempted<<",\"long_swim_cases_total\":2"
        <<",\"parent_long_window_passes\":28,\"long_window_total\":30,\"candidate_long_window_passes\":"<<summary.actual_long_passes<<",\"retained_parent_long_window_passes\":"<<summary.retained_long_passes
        <<",\"turn_cases_passed\":"<<summary.turn_passed<<",\"turn_cases_attempted\":"<<summary.turn_attempted<<",\"turn_cases_completed\":"<<summary.turn_completed<<",\"turn_cases_total\":16"
        <<",\"turning_qualified\":"<<(summary.turn_passed==16&&!summary.hard?"true":"false")<<",\"hard_failure\":"<<(summary.hard?"true":"false")
        <<",\"turn_controls\":"<<budget.turn_controls<<",\"turn_substeps\":"<<budget.turn_substeps<<",\"attempted_controls\":"<<budget.controls<<",\"attempted_substeps\":"<<budget.substeps
        <<",\"wall_seconds\":"<<budget.seconds()<<",\"optimizer_steps\":0,\"PPO_updates\":0,\"CUDA_kernels\":0,\"autonomous_navigation_qualified\":false}\n";
    out.flush();require(bool(out),"Evaluation result write failed");
}
}
int main(int argc,char** argv) {
    Options o;pp::CandidateState candidate;Budget budget;Summary summary;bool owns_output=false,ledger_started=false;uint64_t checkpoint=0,parent_checksum=0,protocol_checksum=0;
    Ledger ledger;std::unique_ptr<cp::Lease> ledger_lease;std::unique_ptr<StoppedTrainingLease> stopped_lease;
    try {
        o=options(argc,argv);
        if(o.self_test){pure_tests();return 0;}
        if(o.describe){if(!o.protocol.empty()){const auto bytes=file(o.protocol);o.study_id=validated_study_id({bytes.begin(),bytes.end()});}std::cout<<"{\"protocol_id\":\""<<o.study_id<<"\",\"supported_protocol_ids\":[\""<<pp::default_study_protocol_id<<"\",\""<<pp::curriculum_study_protocol_id<<"\",\""<<pp::course_study_protocol_id<<"\",\""<<pp::repeated_study_protocol_id<<"\"],\"assessment_schema\":2,\"eval_source_identity_fnv1a64\":\""<<eval_source()<<"\",\"eval_source_sha256\":\""<<pond_candidate_source_sha256
            <<"\",\"runtime_version\":"<<pp::runtime_version<<",\"query_arithmetic_version\":"<<pp::query_arithmetic_version<<",\"world_stride\":"<<sizeof(pr::World)<<",\"task_stride\":"<<expected_task_stride(o.study_id)
            <<",\"retention_cases\":20,\"retention_controls\":30400,\"rest_cases\":14,\"short_swim_cases\":4,\"required_parent_long_window_keys\":28,\"turn_cases\":16,\"complete_baseline_retention_required\":true,\"assess_all_soft_failures\":true,\"turning_is_training_prerequisite\":false,\"max_controls_per_checkpoint\":36800,\"max_assessments\":5,\"max_aggregate_controls\":184000,\"max_aggregate_substeps\":1840000,\"max_wall_seconds\":900,\"final_partial_requires_stopped_last_good\":true,\"final_partial_closes_ledger\":true,\"optimizer_steps\":0}\n";return 0;}
        require(!fs::exists(o.out)&&fs::create_directories(o.out),"Evaluation output already exists");owns_output=true;
        const auto parent=load_parent(o.parent);parent_checksum=parent.checksum;const auto protocol=file(o.protocol);protocol_checksum=cp::checksum(protocol);
        const std::string text(protocol.begin(),protocol.end());o.study_id=validated_study_id(text);
        const auto training_source=std::stoull(scalar(text,"source_identity_fnv1a64"));const auto payload=cp::read(o.candidate.string());checkpoint=cp::checksum(payload);
        candidate=pp::read_candidate(payload,parent.checksum,protocol_checksum,training_source,static_cast<uint32_t>(sizeof(pr::World)),expected_task_stride(o.study_id));
        const uint64_t update=candidate.counters.completed_updates;
        const int expected_slot=o.final_partial?static_cast<int>(update/8)+1:static_cast<int>(update/8);
        require(expected_slot<5,"Checkpoint is outside the five assessment slots");
        assessment_slot(update,o.baseline,o.final_partial,(uint64_t{1}<<expected_slot)-1);
        if(o.final_partial) {
            stopped_lease=std::make_unique<StoppedTrainingLease>(fs::path(scalar(text,"ledger_path_absolute")),o.run);
            stopped_training(cp::read(stopped_lease->path.string()),protocol_checksum,training_source,checkpoint,candidate);
            summary.stopped_training_checksum=stopped_lease->checksum;
        }
        const auto expected_keys=long_reference(o.reference);
        nn::CpuMlp<float> actor({{41,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}});actor.params()=candidate.actor;
        if(o.preflight) {
            float input[41]{};actor.forward(input,1);for(int j=0;j<4;++j)require(std::isfinite(actor.output()[j]),"Candidate zero-input inference nonfinite");
            summary.reason="candidate_and_Adam_bytes_parsed_without_optimizer_or_physics";
            write_result(o,candidate,budget,summary,true,checkpoint);return 0;
        }
        ledger_lease=std::make_unique<cp::Lease>(o.ledger.string()+".assessment");
        ledger=load_ledger(o.ledger,parent.checksum,protocol_checksum);budget.prior=ledger;
        const int slot=assessment_slot(update,o.baseline,o.final_partial,ledger.mask);
        require(ledger.controls+evaluation_controls<=aggregate_controls&&ledger.substeps+evaluation_substeps<=aggregate_substeps,"Aggregate evaluation reservation exceeded");
        ledger.mask|=uint64_t{1}<<slot;ledger.active=true;ledger.closed=o.final_partial||update==32;save_ledger(o.ledger,ledger);ledger_started=true;
        Keys candidate_keys;
        // Windows can report a stale file size until the last writer closes.
        {
            Output output(o.out);
            for(const auto& spec:specs()) {
                const auto outcome=evaluate(spec,candidate,actor,budget,output,expected_keys);
                if(spec.turn){++summary.turn_attempted;summary.turn_passed+=outcome.passed?1u:0u;summary.turn_completed+=outcome.completed?1u:0u;}
                else {
                    ++summary.retention_attempted;summary.retention_passed+=outcome.passed?1u:0u;summary.retention_completed+=outcome.completed?1u:0u;
                    candidate_keys.insert(outcome.long_keys.begin(),outcome.long_keys.end());
                    if(spec.holding){++summary.hold_attempted;summary.hold_passed+=outcome.passed?1u:0u;}
                    else if(spec.long_swim){++summary.long_attempted;summary.long_passed+=outcome.passed?1u:0u;}
                    else {++summary.short_attempted;summary.short_passed+=outcome.passed?1u:0u;}
                    if(!outcome.passed&&summary.first_retention_failure.empty())summary.first_retention_failure=std::string(spec.suite)+"_"+std::to_string(spec.id);
                }
                if(outcome.hard){summary.hard=true;summary.reason=outcome.reason;break;}
                // Every predeclared skill is assessed after a soft miss. A failed
                // retention assessment still blocks the next training chunk below.
            }
        }
        summary.actual_long_passes=static_cast<uint32_t>(candidate_keys.size());
        for(const auto& key:expected_keys)summary.retained_long_passes+=candidate_keys.count(key)?1u:0u;
        summary.retention=retention_qualified(summary,budget);
        if(!summary.hard&&!summary.retention)summary.reason=summary.first_retention_failure.empty()?"exact_retention_or_long_window_keys_failed":"retention_failed_"+summary.first_retention_failure;
        require(cp::checksum(cp::read(o.candidate.string()))==checkpoint&&cp::checksum(cp::read(o.parent.string()))==parent_checksum&&cp::checksum(file(o.protocol))==protocol_checksum,"Evaluation immutable input changed");
        if(stopped_lease)stopped_lease->unchanged();
        require(budget.prior.bytes+directory_bytes(o.out)<=evidence_cap,"Aggregate evaluation evidence cap");
        if(summary.retention&&assessment_completed(summary)) {
            pp::RetentionReceipt receipt;receipt.checkpoint_checksum=checkpoint;receipt.parent_package_checksum=parent_checksum;receipt.protocol_checksum=protocol_checksum;receipt.eval_source_identity=eval_source();
            receipt.cases_passed=summary.retention_passed;receipt.total=20;receipt.completed_cases=summary.retention_attempted;receipt.controls=budget.retention_controls;receipt.substeps=budget.retention_substeps;receipt.passed=summary.retention;receipt.baseline_only=o.baseline;
            cp::write_atomic((o.out/(o.baseline?"baseline-receipt.bin":"retention-receipt.bin")).string(),pp::write_retention_receipt(receipt));
            summary.reason=o.baseline?"baseline_retention_passed_turning_measured":summary.turn_passed==16?"retention_and_turning_qualified":"retention_passed_turning_still_learning";
        }
        write_result(o,candidate,budget,summary,false,checkpoint);
        ledger.controls+=budget.controls;ledger.substeps+=budget.substeps;ledger.seconds+=budget.seconds();ledger.bytes+=directory_bytes(o.out);ledger.active=false;save_ledger(o.ledger,ledger);
        return summary.retention&&assessment_completed(summary)?0:2;
    }catch(const std::exception& error) {
        summary.hard=true;summary.reason=error.what();
        if(owns_output)write_result(o,candidate,budget,summary,o.preflight,checkpoint);
        if(ledger_started){ledger.controls+=budget.controls;ledger.substeps+=budget.substeps;ledger.seconds+=budget.seconds();ledger.bytes+=directory_bytes(o.out);ledger.active=false;save_ledger(o.ledger,ledger);}
        std::cerr<<error.what()<<'\n';return 2;
    }
}
