// Preparation is pure inference. Physical entry is a separately approved,
// one-shot frozen-controller diagnostic; no optimizer is constructed or called.
#include "motor/straight_task.h"
#include "motor/turning_task_v1.h"
#include "rl/append_hold_checkpoint.h"
#include "nn/mlp_cpu.h"
#include "nn/mlp_gpu.h"
#include "cuda/cuda_memory.h"
#include "frozen_requested_turn_authorization.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace probe {
namespace fs=std::filesystem;namespace p=cudarl::physics;namespace cp=cudarl::checkpoint;
namespace old=cudarl::straight_motor;namespace joint=cudarl::joint_motor;namespace model=cudarl::movement;
namespace bank=cudarl::turning_task_v1;namespace nn=cudarl::nn;
namespace grant=cudarl::frozen_requested_turn_authorization;
constexpr int maximum_controls=7360,maximum_substeps=73600,warm_controls=80,case_controls=320;
constexpr uint64_t evidence_cap=128ull*1024*1024;
constexpr double time_cap=120;
const auto process_start=std::chrono::steady_clock::now();
constexpr int offsets[11]={0,5,-5,15,-15,30,-30,60,-60,90,-90};
struct Sample {p::BodyState body{};double energy[6]{};uint64_t attempted=0,accepted=0;};
struct Frame {
    joint::World world{};old::ForwardTask task{};p::V initial_heading{};
    float observation[41]{},base[4]{},command[4]{};
    Sample substep[10]{};p::BodyState first_before{},candidate{};
    int valid=0,advanced=0,accepted=0,attempted=0,reason=0,failure=0;
};
struct Budget {
    uint64_t controls=0,substeps=0,accepted=0,bytes=0;
    std::chrono::steady_clock::time_point start=process_start;
    double seconds()const{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
    void before(){if(controls>=maximum_controls||substeps>=maximum_substeps||seconds()>=time_cap)throw std::runtime_error("Probe physical/time cap reached");}
    void charge(uint64_t n){if(bytes+n>evidence_cap-131072)throw std::runtime_error("Probe evidence cap reached");bytes+=n;}
};
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
old::ForwardConfig config(){auto c=old::config_for_speed(.008f);c.observation_schema=4;c.maximum_controls=case_controls;return c;}
nn::MlpSpec spec(){return {{41,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}};}
CUDARL_HD inline void energy(const joint::World& w,double* e){e[0]=w.signed_work;e[1]=w.positive_work;e[2]=w.fluid_loss;e[3]=w.numerical_loss;e[4]=w.external_work;e[5]=w.last_energy_error;}
CUDARL_HD inline bool initialize(int pose,Frame& f){
    f={};f.world=old::initial_world(pose);f.world.arena=old::long_swim_arena();
    auto c=old::config_for_speed(.008f);c.observation_schema=4;c.maximum_controls=case_controls;
    if(!old::begin_forward(f.world,c,f.task))return false;
    f.initial_heading=f.task.request.heading;f.valid=1;return true;
}
CUDARL_HD inline bool latch(Frame& f,int offset){
    if(offset==0)return true; // Deliberate exact identity, including target/lane.
    const auto before=f.world.physical.body;const auto com=cudarl::motor::motion(before).com_model;
    const auto heading=p::axis_rotation({0,1,0},float(offset)*model::pi/180.f).rotate(f.initial_heading);
    f.task.start_com_model.x=com.x;f.task.start_com_model.z=com.z;
    f.task.request.heading=heading;f.task.request.terminal_velocity=heading*.008f;
    f.task.request.position=f.task.start_com_model+heading*(.072f/p::scale);
    return p::finite_vector(heading);
}
CUDARL_HD inline bool observe(Frame& f){
    auto c=old::config_for_speed(.008f);c.observation_schema=4;c.maximum_controls=case_controls;
    return old::observe_appended(f.task,f.world,c,f.observation);
}
// Same original mechanical sequence, with accepted-state behavioral guards.
// No boundary force, state projection, rate reset, or action replacement.
CUDARL_HD inline bool advance(Frame& f){
    f.advanced=f.accepted=f.attempted=f.reason=f.failure=0;auto& w=f.world;auto& s=w.physical;
    f.first_before=f.candidate=s.body;
    if(!joint::in_domain(w)||s.paused){f.failure=101;return false;}
    const auto initial=bank::bounds(s.body);
    if(!initial.finite||initial.active_radius_squared>1||initial.inset_radius_squared>=1){f.failure=102;return false;}
    for(int j=0;j<4;++j)if(!p::finite_number(f.command[j])||fabsf(f.command[j])>1){f.failure=103;return false;}
    s.telemetry={};s.mean_squared_torque=0;
    for(int j=0;j<4;++j)s.raw_action[j]=f.command[j];
    for(int j=0;j<40;++j){s.actual_torques[j]=0;s.requested_torques[j]=0;}
    for(int k=0;k<10;++k){
        f.first_before=f.candidate=s.body;model::Reference reference;
        if(!joint::filter_targets(s.held_targets,w.reference_rate,f.command,reference)){w.failure_code=f.failure=10;s.invalid=true;return false;}
        model::Telemetry t;++w.attempted_substeps;++f.attempted;
        if(!model::substep(s.body,reference,.005f,s.fluid,t,true,w.arena,&f.candidate)){w.failure_code=f.failure=t.failure_code;s.invalid=true;return false;}
        w.signed_work+=t.signed_work;w.positive_work+=t.positive_work;w.fluid_loss+=t.fluid_loss;
        w.numerical_loss+=t.numerical_loss;w.external_work+=t.external_work;
        w.last_energy_error=model::kinetic(s.body)-w.signed_work+w.fluid_loss+w.numerical_loss-w.external_work;
        if(fabs(w.last_energy_error)>.02*(w.positive_work+w.fluid_loss)+2e-10){w.failure_code=f.failure=9;s.invalid=true;return false;}
        ++s.substeps;++s.telemetry.executed_substeps;++f.accepted;s.time_seconds=double(s.substeps)*.005f;
        s.telemetry.minimum_floor_gap=fminf(s.telemetry.minimum_floor_gap,t.minimum_gap);
        s.telemetry.minimum_other_gap=fminf(s.telemetry.minimum_other_gap,t.minimum_gap);
        for(int j=0;j<4;++j){s.held_targets[j]=reference.angle[j];const float n=t.torque[j]/p::torque;
            s.last_actual_torques[j]=n;s.actual_torques[k*4+j]=n;s.requested_torques[k*4+j]=t.requested[j];s.mean_squared_torque+=n*n;}
        auto& saved=f.substep[k];saved.body=s.body;energy(w,saved.energy);saved.attempted=w.attempted_substeps;saved.accepted=s.substeps;
        const auto b=bank::bounds(s.body);
        if(!b.finite||b.active_radius_squared>1){w.failure_code=f.failure=104;s.unsafe=true;return false;}
        const float depth=(cudarl::motor::motion(s.body).com_model.y-f.task.start_com_model.y)*p::scale;
        const float tilt=acosf(p::clamp(s.body.orientation.rotate({0,1,0}).y,-1,1))*180/model::pi;
        f.reason=b.inset_radius_squared>=1?1:fabsf(depth)>.005f?4:tilt>20?5:0;
        if(f.reason)break;
    }
    s.mean_squared_torque/=float(f.accepted*4);++s.control_steps;
    if(!joint::in_domain(w)){f.failure=105;return false;}
    // Bookkeeping observes the real state. Legacy straight outcome is diagnostic.
    auto c=old::config_for_speed(.008f);c.observation_schema=4;c.maximum_controls=case_controls;
    const auto transition=old::record_control(f.task,w,c);
    if(!transition.accepted){f.failure=106;return false;}
    f.advanced=1;return true;
}
__global__ void initial(Frame* f,int n){int i=threadIdx.x;if(i>=n)return;initialize(i<4?i:(i-4)/11?3:0,f[i]);}
__global__ void observations(Frame* f,int n){int i=threadIdx.x;if(i<n)f[i].valid=observe(f[i])?1:0;}
__global__ void gather(const Frame* f,float* x,int n){int i=threadIdx.x;if(i<n)for(int j=0;j<41;++j)x[i*41+j]=f[i].observation[j];}
__global__ void decode(Frame* f,const float* z,int n){int i=threadIdx.x;if(i<n)for(int j=0;j<4;++j){f[i].base[j]=z[i*4+j];f[i].command[j]=tanhf(z[i*4+j]);}}
__global__ void stepping(Frame* f){advance(*f);}
__global__ void request_latch(Frame* f,int offset){f->valid=latch(*f,offset)?1:0;}
void sync(){CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());}
std::vector<float> load(const fs::path& root){
    require(cp::checksum(cp::read((root/"parent-checkpoint.bin").string()))==2039539270511780368ull&&
        cp::checksum(cp::read((root/"parent-protocol.bin").string()))==14874979637633386686ull&&
        cp::checksum(cp::read((root/"holding-parent.bin").string()))==16983070322273901339ull,"Exact frozen parent checksum differs");
    const auto state=cudarl::rl::read_ppo_training_state(cp::read((root/"parent-checkpoint.bin").string()),cp::checksum(cp::read((root/"parent-protocol.bin").string())));
    require(state.actor.update==336&&state.actor.observation_schema==2&&state.actor.reward_version==5,"Wrong frozen parent");
    const auto weights=cudarl::rl::append_zero_input_column(state.actor.parameters);
    const auto bytes=cp::read((root/"holding-parent.bin").string());cp::Reader h{bytes};
    require(h.value<uint32_t>()==0x314d4148&&h.value<uint32_t>()==1&&h.value<uint64_t>()!=0,"Wrong holding parent");
    require(h.value<uint64_t>()==cp::checksum(cp::read((root/"parent-checkpoint.bin").string())),"Holding parent binding differs");
    for(uint32_t v:{6u,6u,3u,4u,41u,4u,336u})require(h.value<uint32_t>()==v,"Holding contract differs");
    require(h.value<int>()==7&&h.value<int>()==112&&h.array<float>(2532)==weights,"Holding backbone/history differs");
    const auto head=h.array<float>(2);const float gain=h.value<float>(),sigma=h.value<float>();(void)h.array<uint8_t>();h.finish();
    require(head.size()==2&&std::isfinite(gain)&&gain>0&&gain<1&&std::isfinite(sigma)&&sigma>0,"Holding metadata invalid");
    return weights;
}
uint64_t protocol(const fs::path& root){return cp::checksum(cp::read((root/"protocol.bin").string()));}
uint64_t projected_run_bytes(){return uint64_t(maximum_substeps)*700+uint64_t(maximum_controls+30)*2200+
    uint64_t(maximum_controls+30)*(sizeof(joint::World)+sizeof(old::ForwardTask)+240)+1024*1024;}
cp::Writer binding(const fs::path& root,const fs::path& exe){cp::Writer w;w.value(uint32_t{0x31505446});w.value(uint32_t{1});w.value(protocol(root));
    for(const char* n:{"approved-protocol.md","parent-checkpoint.bin","parent-protocol.bin","holding-parent.bin"})w.value(grant::whole_file(root/n));
    w.value(grant::whole_file(exe));return w;}
void body(std::ostream& out,const p::BodyState& b){
    out<<std::setprecision(9);for(float v:{b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w})out<<','<<v;
    for(float v:b.angle)out<<','<<v;for(float v:b.velocity)out<<','<<v;out<<','<<b.work<<','<<b.travel;
}
void body_header(std::ostream& o){for(const char* n:{"x","y","z","qx","qy","qz","qw","j0","j1","j2","j3","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","body_work","body_travel"})o<<','<<n;}
void checked_line(std::ofstream& stream,const std::string& text,Budget& budget){budget.charge(text.size());stream<<text;require(bool(stream),"Probe evidence write failed");}
void capture(std::ofstream& control,std::ofstream& substeps,cp::Writer& raw,Budget& budget,int group,int id,int backend,int tick,const Frame& f){
    std::ostringstream o;o<<std::setprecision(17)<<group<<','<<id<<','<<backend<<','<<tick<<','<<f.world.physical.time_seconds<<','<<f.world.physical.control_steps<<','<<f.world.attempted_substeps<<','<<f.world.physical.substeps<<','<<f.attempted<<','<<f.accepted<<','<<f.reason<<','<<f.failure<<','<<int(f.task.outcome)<<','<<f.task.controls;
    body(o,f.world.physical.body);for(float v:{f.task.start_com_model.x,f.task.start_com_model.y,f.task.start_com_model.z,f.task.request.position.x,f.task.request.position.y,f.task.request.position.z,f.task.request.heading.x,f.task.request.heading.y,f.task.request.heading.z,f.task.request.terminal_velocity.x,f.task.request.terminal_velocity.y,f.task.request.terminal_velocity.z})o<<','<<v;
    for(float v:f.world.physical.held_targets)o<<','<<v;for(float v:f.world.reference_rate)o<<','<<v;for(float v:f.world.physical.last_actual_torques)o<<','<<v;
    for(float v:f.observation)o<<','<<v;for(float v:f.base)o<<','<<v;for(float v:f.command)o<<','<<v;
    double e[6];energy(f.world,e);o<<std::setprecision(17);for(double v:e)o<<','<<v;o<<'\n';checked_line(control,o.str(),budget);
    if(tick)for(int k=0;k<f.accepted;++k){const auto& z=f.substep[k];std::ostringstream row;row<<group<<','<<id<<','<<backend<<','<<tick<<','<<k+1<<','<<z.attempted<<','<<z.accepted;body(row,z.body);row<<std::setprecision(17);for(double v:z.energy)row<<','<<v;row<<'\n';checked_line(substeps,row.str(),budget);}
    const auto before=raw.data.size();raw.value(group);raw.value(id);raw.value(backend);raw.value(tick);raw.value(f.world);raw.value(f.task);
    for(float v:f.observation)raw.value(v);for(float v:f.base)raw.value(v);for(float v:f.command)raw.value(v);raw.value(f.attempted);raw.value(f.accepted);raw.value(f.reason);raw.value(f.failure);
    budget.charge(raw.data.size()-before);
}
void agreement(const Frame& a,const Frame& b,double& max_position){
    const auto& x=a.world.physical.body;const auto& y=b.world.physical.body;
    const float* left=reinterpret_cast<const float*>(&x);const float* right=reinterpret_cast<const float*>(&y);
    for(int j=0;j<21;++j){const double d=std::abs(double(left[j])-right[j]);require(std::isfinite(d),"Nonfinite backend difference");
        const double tolerance=j<3?2e-5:j<11?2e-5:j<14?2e-7:2e-5;
        if(j<3)max_position=std::max(max_position,d);require(d<=tolerance,"Matched-command physics disagreement");}
    for(int j=0;j<4;++j)for(const auto pair:{std::pair<float,float>{a.world.physical.held_targets[j],b.world.physical.held_targets[j]},
        {a.world.reference_rate[j],b.world.reference_rate[j]},{a.world.physical.last_actual_torques[j],b.world.physical.last_actual_torques[j]}})
        require(std::isfinite(pair.first)&&std::isfinite(pair.second)&&std::abs(double(pair.first)-pair.second)<=2e-5,"Matched-command motor difference");
    for(int j=0;j<40;++j)require(std::isfinite(a.world.physical.actual_torques[j])&&std::isfinite(b.world.physical.actual_torques[j])&&
        std::abs(double(a.world.physical.actual_torques[j])-b.world.physical.actual_torques[j])<=2e-5,"Matched-command substep torque difference");
    double ae[6],be[6];energy(a.world,ae);energy(b.world,be);
    for(int j=0;j<6;++j)require(std::isfinite(ae[j])&&std::isfinite(be[j])&&std::abs(ae[j]-be[j])<=1e-12+1e-4*std::max(std::abs(ae[j]),std::abs(be[j])),"Matched-command energy difference");
}
void infer(nn::GpuMlp& network,device_ptr<Frame>& d_state,device_ptr<float>& d_input,int n){observations<<<1,32>>>(d_state.get(),n);gather<<<1,32>>>(d_state.get(),d_input.get(),n);network.forward(d_input.get(),n);decode<<<1,32>>>(d_state.get(),network.output(),n);sync();}
int prepare(const fs::path& root,const fs::path& parents,const fs::path& document,const fs::path& exe){
    require(!fs::exists(root),"Prepare a fresh probe root; never overwrite evidence");require(fs::create_directories(root),"Cannot create probe root");
    for(const char* name:{"parent-checkpoint.bin","parent-protocol.bin","holding-parent.bin"})fs::copy_file(parents/name,root/name);
    fs::copy_file(document,root/"approved-protocol.md");const auto weights=load(root);
    cp::Writer p;p.value(uint32_t{0x31525446});p.value(uint32_t{1});p.value(grant::whole_file(root/"approved-protocol.md"));
    p.value(cp::checksum(cp::read((root/"parent-checkpoint.bin").string())));p.value(cp::checksum(cp::read((root/"holding-parent.bin").string())));
    for(int v:{6,4,41,4,336,22,80,320,320,7360,73600,120})p.value(v);
    for(float v:{.008f,.005f,20.f,.20f,.12f,.195f,.115f})p.value(v);p.value(evidence_cap);for(int v:offsets)p.value(v);
    cp::write_atomic((root/"protocol.bin").string(),p.data);
    nn::CpuMlp<float> cpu(spec());cpu.params()=weights;nn::GpuMlp gpu(spec(),26,1);gpu.set_params(weights);
    device_ptr<Frame> d_state(26);device_ptr<float> d_input(26*41);initial<<<1,32>>>(d_state.get(),26);sync();infer(gpu,d_state,d_input,26);
    std::vector<Frame> rows(26);d_state.copy_to_host(rows.data(),26);double max_latent=0,max_command=0;
    cp::Writer fixture;fixture.value(uint32_t{0x315A5446});fixture.value(uint32_t{1});fixture.value(protocol(root));fixture.value(uint32_t{sizeof(Frame)});fixture.array(rows);
    cp::write_atomic((root/"pure-inference.bin").string(),fixture.data);
    for(const auto& row:rows){require(row.valid&&row.world.physical.control_steps==0&&row.world.attempted_substeps==0&&row.world.physical.substeps==0,"Pure initialization failed");cpu.forward(row.observation,1);
        for(int j=0;j<4;++j){const double z=std::abs(double(cpu.output()[j])-row.base[j]),a=std::abs(double(tanhf(cpu.output()[j]))-row.command[j]);max_latent=std::max(max_latent,z);max_command=std::max(max_command,a);require(z<=2e-5&&a<=2e-5,"Pure pointwise inference differs");}}
    cp::Writer latches;latches.value(uint32_t{0x314C5446});latches.value(uint32_t{1});latches.value(protocol(root));
    cp::Writer gpu_latches;gpu_latches.value(uint32_t{0x31475446});gpu_latches.value(uint32_t{1});gpu_latches.value(protocol(root));
    double maximum_latch_difference=0;
    for(int id=0;id<22;++id){Frame row=rows[id+4],before=row;const int off=offsets[id%11];require(latch(row,off),"Pure request change rejected");
        require(std::memcmp(&before.world,&row.world,sizeof(row.world))==0,"Request mutated physical state");
        require(row.task.controls==before.task.controls&&row.task.last_world_control==before.task.last_world_control&&row.task.start_com_model.y==before.task.start_com_model.y,"Request reset clocks/depth");
        if(!off)require(std::memcmp(&before.task,&row.task,sizeof(row.task))==0,"Zero offset changed original intent");
        d_state.copy_from_host(&before,1);request_latch<<<1,1>>>(d_state.get(),off);sync();Frame gpu_latch;d_state.copy_to_host(&gpu_latch,1);
        require(std::memcmp(&before.world,&gpu_latch.world,sizeof(before.world))==0,"GPU request mutated physical state");
        if(!off)require(std::memcmp(&before.task,&gpu_latch.task,sizeof(before.task))==0,"GPU zero offset changed intent");
        const double d=p::norm(row.task.request.heading-gpu_latch.task.request.heading);maximum_latch_difference=std::max(maximum_latch_difference,d);require(d<=2e-5,"Pointwise request rotation differs");
        latches.value(id);latches.value(off);latches.value(before.task);latches.value(row.task);
        gpu_latches.value(id);gpu_latches.value(off);gpu_latches.value(before.task);gpu_latches.value(gpu_latch.task);}
    cp::write_atomic((root/"pure-latches.bin").string(),latches.data);
    cp::write_atomic((root/"pure-gpu-latches.bin").string(),gpu_latches.data);
    std::ofstream layout(root/"layout.json");layout<<"{\"frame\":{\"size\":"<<sizeof(Frame);
#define FIELD(Type,name) <<",\"" #name "\":"<<offsetof(Type,name)
    layout FIELD(Frame,world) FIELD(Frame,task) FIELD(Frame,initial_heading) FIELD(Frame,observation) FIELD(Frame,base) FIELD(Frame,command)
        FIELD(Frame,substep) FIELD(Frame,first_before) FIELD(Frame,candidate) FIELD(Frame,valid) FIELD(Frame,advanced) FIELD(Frame,accepted)
        FIELD(Frame,attempted) FIELD(Frame,reason) FIELD(Frame,failure)<<"},\"task\":{\"size\":"<<sizeof(old::ForwardTask);
    layout FIELD(old::ForwardTask,request) FIELD(old::ForwardTask,start_com_model) FIELD(old::ForwardTask,controls)
        FIELD(old::ForwardTask,last_world_control) FIELD(old::ForwardTask,outcome)<<"},\"request\":{\"size\":"<<sizeof(cudarl::simulation::MotorRequest);
    layout FIELD(cudarl::simulation::MotorRequest,position) FIELD(cudarl::simulation::MotorRequest,terminal_velocity)
        FIELD(cudarl::simulation::MotorRequest,heading)<<"},\"sample\":{\"size\":"<<sizeof(Sample);
    layout FIELD(Sample,body) FIELD(Sample,energy) FIELD(Sample,attempted) FIELD(Sample,accepted)<<"},\"World\":{\"size\":"<<sizeof(joint::World)
        <<"},\"BodyState\":{\"size\":"<<sizeof(p::BodyState)<<"}}\n";layout.close();require(bool(layout),"Layout write failed");
#undef FIELD
    cp::write_atomic((root/"prepared.bin").string(),binding(root,exe).data);
    std::ofstream report(root/"preparation.json");report<<std::setprecision(17)<<"{\"schema\":1,\"passed\":true,\"physical_controls\":0,\"physical_substeps\":0,\"optimizer_steps\":0,\"initial_fixtures\":26,\"request_latch_fixtures\":22,\"maximum_same_observation_latent_difference\":"<<max_latent<<",\"maximum_same_observation_command_difference\":"<<max_command<<",\"maximum_latch_heading_difference\":"<<maximum_latch_difference<<",\"world_bytes\":"<<sizeof(joint::World)<<",\"task_bytes\":"<<sizeof(old::ForwardTask)<<",\"frame_bytes\":"<<sizeof(Frame)<<",\"body_bytes\":"<<sizeof(p::BodyState)<<",\"conservative_generated_evidence_bytes\":"<<projected_run_bytes()<<",\"evidence_cap_bytes\":"<<evidence_cap<<",\"physical_run_approved\":false}\n";report.close();require(bool(report),"Preparation report write failed");
    std::cout<<"Prepared frozen requested-turn probe: zero physical controls, zero optimizer steps\n";return 0;
}
struct CaseResult {
    int id=0,pose=0,offset=0,controls=0,reason=0,stable_run=0;double seconds=0,progress=0,path=0,max_depth=0,max_tilt=0,max_lane=0,work=0;
    double last_seconds=0,last_vx=0,last_vz=0,last_fx=0,last_fz=0,last_head_sq=0,settled_seconds=-1,displacement_course=0;
    double turn_seconds=0,turn_vx=0,turn_vz=0,min_head_error=1e6,max_head_error=-1e6;p::V heading{};
};
double course_error(p::V heading,p::V vector){return std::atan2(double(heading.z)*vector.x-double(heading.x)*vector.z,double(heading.x)*vector.x+double(heading.z)*vector.z)*180/model::pi;}
int run(const fs::path& root,const fs::path& exe){
    require(cp::read((root/"prepared.bin").string())==binding(root,exe).data,"Prepared executable/parents/protocol changed");
    grant::require_approved(root,protocol(root));require(!fs::exists(root/"result.json")&&!fs::exists(root/"probe-started"),"Probe already attempted; no retry");
    Budget budget;for(const auto& file:fs::recursive_directory_iterator(root))if(file.is_regular_file())budget.bytes+=file.file_size();
    // Conservative float/double textual widths, raw endpoints, close reserve.
    require(budget.bytes+projected_run_bytes()<=evidence_cap,"Frozen inputs plus worst-case evidence do not fit cap");
    require(fs::create_directory(root/"probe-started"),"Cannot create one-shot probe directory");
    const auto out=root/"probe-started";std::ofstream states(out/"controls.csv"),substeps(out/"substeps.csv"),pointwise(out/"same-observation.csv");cp::Writer raw;raw.value(uint32_t{1});raw.value(uint32_t{sizeof(joint::World)});raw.value(uint32_t{sizeof(old::ForwardTask)});
    std::ostringstream h;h<<"group,case,backend,control,seconds,world_controls,attempted_substeps,accepted_substeps,last_attempted,last_accepted,behavior_reason,hard_failure,legacy_outcome,task_controls";body_header(h);
    for(const char* prefix:{"lane_origin","target","heading","terminal_velocity"})for(char axis:{'x','y','z'})h<<','<<prefix<<'_'<<axis;
    for(const auto item:{std::pair<const char*,int>{"held",4},{"reference_rate",4},{"previous_torque",4},{"observation",41},{"base",4},{"command",4},{"energy",6}})for(int j=0;j<item.second;++j)h<<','<<item.first<<j;h<<'\n';
    checked_line(states,h.str(),budget);std::ostringstream sh;sh<<"group,case,backend,control,substep,attempted_substeps,accepted_substeps";body_header(sh);for(int j=0;j<6;++j)sh<<",energy"<<j;sh<<'\n';checked_line(substeps,sh.str(),budget);
    std::ostringstream ph;ph<<"pose,control";for(int j=0;j<41;++j)ph<<",observation"<<j;
    for(const char* n:{"cpu_base","gpu_base","cpu_command","gpu_command"})for(int j=0;j<4;++j)ph<<','<<n<<j;ph<<'\n';checked_line(pointwise,ph.str(),budget);
    std::string fault;int completed_gate=0,completed_cases=0;double maximum_position=0,maximum_inference=0;Frame active{},other{};int active_group=0,active_id=0,active_tick=0;std::vector<CaseResult> summaries;
    try{
        const auto weights=load(root);nn::CpuMlp<float> cpu(spec());cpu.params()=weights;nn::GpuMlp gpu(spec(),1,1);gpu.set_params(weights);
        device_ptr<Frame> d_state(1);device_ptr<float> d_input(41);
        for(int pose=0;pose<4;++pose){active_group=0;active_id=pose;initialize(pose,active);other=active;capture(states,substeps,raw,budget,0,pose,0,0,active);capture(states,substeps,raw,budget,0,pose,1,0,other);
            for(int tick=1;tick<=40;++tick){active_tick=tick;require(observe(active),"CPU precheck observation failed");cpu.forward(active.observation,1);for(int j=0;j<4;++j){active.base[j]=cpu.output()[j];active.command[j]=tanhf(active.base[j]);other.observation[j]=0;}
                d_input.copy_from_host(active.observation,41);gpu.forward(d_input.get(),1);std::array<float,4> gpu_base;CUDA_CHECK_THROW(cudaDeviceSynchronize());CUDA_CHECK_THROW(cudaMemcpy(gpu_base.data(),gpu.output(),4*sizeof(float),cudaMemcpyDeviceToHost));
                bool same=true;for(int j=0;j<4;++j){const double z=std::abs(double(gpu_base[j])-active.base[j]),a=std::abs(double(tanhf(gpu_base[j]))-active.command[j]);maximum_inference=std::max({maximum_inference,z,a});same=same&&z<=2e-5&&a<=2e-5;other.base[j]=active.base[j];other.command[j]=active.command[j];}for(int j=0;j<41;++j)other.observation[j]=active.observation[j];
                std::ostringstream pr;pr<<std::setprecision(9)<<pose<<','<<tick;for(float v:active.observation)pr<<','<<v;
                for(float v:active.base)pr<<','<<v;for(float v:gpu_base)pr<<','<<v;for(float v:active.command)pr<<','<<v;for(float v:gpu_base)pr<<','<<tanhf(v);pr<<'\n';checked_line(pointwise,pr.str(),budget);require(same,"Pointwise precheck inference failed");
                budget.before();++budget.controls;advance(active);budget.substeps+=active.attempted;budget.accepted+=active.accepted;capture(states,substeps,raw,budget,0,pose,0,tick,active);require(active.advanced&&!active.reason,"CPU precheck failed/terminated");
                budget.before();++budget.controls;d_state.copy_from_host(&other,1);stepping<<<1,1>>>(d_state.get());sync();d_state.copy_to_host(&other,1);budget.substeps+=other.attempted;budget.accepted+=other.accepted;capture(states,substeps,raw,budget,0,pose,1,tick,other);require(other.advanced&&!other.reason,"CUDA precheck failed/terminated");agreement(active,other,maximum_position);
            }++completed_gate;
        }
        for(int id=0;id<22;++id){active_group=1;active_id=id;const int pose=id/11?3:0,off=offsets[id%11];initialize(pose,active);d_state.copy_from_host(&active,1);capture(states,substeps,raw,budget,1,id,1,0,active);
            CaseResult result;result.id=id;result.pose=pose;result.offset=off;p::V turn_start{};bool turned=false;
            for(int tick=1;tick<=case_controls;++tick){active_tick=tick;budget.before();if(tick==warm_controls+1){turn_start=cudarl::motor::motion(active.world.physical.body).com_model;turned=true;request_latch<<<1,1>>>(d_state.get(),off);sync();}
                infer(gpu,d_state,d_input,1);d_state.copy_to_host(&active,1);require(active.valid,"CUDA original observer failed");const auto before=cudarl::motor::motion(active.world.physical.body).com_model;result.heading=active.task.request.heading;
                ++budget.controls;stepping<<<1,1>>>(d_state.get());sync();d_state.copy_to_host(&active,1);budget.substeps+=active.attempted;budget.accepted+=active.accepted;capture(states,substeps,raw,budget,1,id,1,tick,active);require(active.advanced,"Probe hard physical fault");
                result.controls=tick;result.reason=active.reason;result.seconds=active.world.physical.time_seconds;
                const auto measured=old::measure(active.task,active.world);result.max_depth=std::max(result.max_depth,double(measured.depth_m));result.max_tilt=std::max(result.max_tilt,double(measured.tilt_deg));result.max_lane=std::max(result.max_lane,double(measured.lateral_m));
                if(turned){const auto motion=cudarl::motor::motion(active.world.physical.body);const auto com=motion.com_model;const auto d=com-before;result.path+=p::norm({d.x,0,d.z})*p::scale;result.progress=p::dot((com-turn_start)*p::scale,active.task.request.heading);
                    result.displacement_course=course_error(active.task.request.heading,com-turn_start);
                    const auto head=bank::forward(active.world.physical.body);const double error=course_error(active.task.request.heading,head),dt=active.accepted*.005;
                    result.turn_seconds+=dt;result.turn_vx+=motion.velocity_m_s.x*dt;result.turn_vz+=motion.velocity_m_s.z*dt;
                    result.min_head_error=std::min(result.min_head_error,error);result.max_head_error=std::max(result.max_head_error,error);
                    result.stable_run=std::abs(error)<=10&&active.accepted==10?result.stable_run+1:0;
                    if(result.stable_run>=20&&result.settled_seconds<0)result.settled_seconds=(tick-warm_controls-20)*.05;
                    if(tick>case_controls-80){result.last_seconds+=dt;result.last_vx+=motion.velocity_m_s.x*dt;result.last_vz+=motion.velocity_m_s.z*dt;
                        result.last_fx+=head.x*dt;result.last_fz+=head.z*dt;result.last_head_sq+=error*error*dt;}}
                if(active.reason)break;
            }result.work=active.world.positive_work;summaries.push_back(result);++completed_cases;
        }
        require(budget.controls<=maximum_controls&&budget.substeps<=maximum_substeps&&budget.seconds()<=time_cap,"Final probe cap violated");
    }catch(const std::exception& e){fault=e.what();cp::Writer failed;failed.value(uint32_t{1});failed.value(active_group);failed.value(active_id);failed.value(active_tick);failed.value(active);failed.value(other);cp::write_atomic((out/"first-failure.bin").string(),failed.data);}
    cp::write_atomic((out/"states.bin").string(),raw.data);states.close();substeps.close();pointwise.close();if((!states||!substeps||!pointwise)&&fault.empty())fault="Evidence flush failed";
    std::ofstream summary(out/"cases.csv");summary<<"case,pose,offset_degrees,controls,seconds,behavior_reason,turn_progress_m,turn_horizontal_arc_m,max_depth_m,max_tilt_deg,max_lane_m,positive_work_J,turn_displacement_course_error_deg,first_20_control_samples_head_error_below10_start_seconds,last4_velocity_x_m_s,last4_velocity_z_m_s,last4_velocity_course_error_deg,last4_head_circular_mean_error_deg,last4_head_error_RMS_deg,turn_mean_velocity_x_m_s,turn_mean_velocity_z_m_s,turn_mean_velocity_course_error_deg,turn_head_error_min_deg,turn_head_error_max_deg\n"<<std::setprecision(17);
    for(const auto& r:summaries){summary<<r.id<<','<<r.pose<<','<<r.offset<<','<<r.controls<<','<<r.seconds<<','<<r.reason<<','<<r.progress<<','<<r.path<<','<<r.max_depth<<','<<r.max_tilt<<','<<r.max_lane<<','<<r.work<<',';
        if(r.turn_seconds>0)summary<<r.displacement_course;else summary<<"null";summary<<',';
        if(r.settled_seconds>=0)summary<<r.settled_seconds;else summary<<"null";
        if(r.controls==320&&!r.reason&&r.last_seconds>3.99){const auto heading=r.heading;
            summary<<','<<r.last_vx/r.last_seconds<<','<<r.last_vz/r.last_seconds<<',';
            if(std::hypot(r.last_vx,r.last_vz)>1e-9)summary<<course_error(heading,{float(r.last_vx),0,float(r.last_vz)});else summary<<"null";
            summary<<',';if(std::hypot(r.last_fx,r.last_fz)>1e-9)summary<<course_error(heading,{float(r.last_fx),0,float(r.last_fz)});else summary<<"null";
            summary<<','<<std::sqrt(r.last_head_sq/r.last_seconds);
        }else summary<<",null,null,null,null,null";
        if(r.turn_seconds>0){summary<<','<<r.turn_vx/r.turn_seconds<<','<<r.turn_vz/r.turn_seconds<<',';
            if(std::hypot(r.turn_vx,r.turn_vz)>1e-9)summary<<course_error(r.heading,{float(r.turn_vx),0,float(r.turn_vz)});else summary<<"null";
            summary<<','<<r.min_head_error<<','<<r.max_head_error;
        }else summary<<",null,null,null,null,null";summary<<'\n';}summary.close();if(!summary&&fault.empty())fault="Case summary write failed";
    uint64_t actual_bytes=0;for(const auto& f:fs::recursive_directory_iterator(root))if(f.is_regular_file())actual_bytes+=f.file_size();
    if(fault.empty()&&(actual_bytes+16384>evidence_cap||budget.seconds()>time_cap))fault="Final evidence/time cap exceeded";
    std::ofstream result(root/"result.json");result<<std::setprecision(17)<<"{\"schema\":1,\"study\":\"frozen_requested_turn_probe\",\"closed\":true,\"hard_fault\":"<<std::quoted(fault)<<",\"precheck_completed_cases\":"<<completed_gate<<",\"probe_completed_cases\":"<<completed_cases<<",\"controls\":"<<budget.controls<<",\"attempted_substeps\":"<<budget.substeps<<",\"accepted_substeps\":"<<budget.accepted<<",\"process_seconds\":"<<budget.seconds()<<",\"evidence_bytes_before_close_report\":"<<actual_bytes<<",\"maximum_position_difference_model\":"<<maximum_position<<",\"maximum_same_observation_inference_difference\":"<<maximum_inference<<",\"PPO_updates\":0,\"optimizer_steps\":0,\"qualified\":false,\"runtime_adopted\":false,\"descriptive_probe_only\":true}\n";result.close();require(bool(result),"Close report write failed");
    std::cout<<"Frozen turn probe closed controls="<<budget.controls<<" steps="<<budget.substeps<<" fault="<<fault<<'\n';return fault.empty()?0:3;
}
}
int main(int argc,char** argv){try{
    const auto exe=std::filesystem::absolute(argv[0]);
    if(argc==5&&std::string(argv[1])=="--prepare")return probe::prepare(std::filesystem::absolute(argv[2]),std::filesystem::absolute(argv[3]),std::filesystem::absolute(argv[4]),exe);
    if(argc==2)return probe::run(std::filesystem::absolute(argv[1]),exe);
    throw std::runtime_error("Use --prepare RUN PARENTS PROTOCOL_MD, or RUN after separate explicit approval");
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
