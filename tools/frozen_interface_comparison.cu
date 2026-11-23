// One finite frozen-motor interface diagnostic. No optimizer or training path.
#include "motor/straight_task.h"
#include "motor/turning_task_v1.h"
#include "rl/append_hold_checkpoint.h"
#include "nn/mlp_cpu.h"
#include "nn/mlp_gpu.h"
#include "cuda/cuda_memory.h"
#include "frozen_interface_authorization.h"
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

namespace comparison {
namespace fs=std::filesystem;namespace p=cudarl::physics;namespace cp=cudarl::checkpoint;
namespace old=cudarl::straight_motor;namespace joint=cudarl::joint_motor;namespace model=cudarl::movement;
namespace bank=cudarl::turning_task_v1;namespace nn=cudarl::nn;
namespace grant=cudarl::frozen_interface_authorization;
constexpr int case_count=88,gate_count=4,case_controls=430;
constexpr int maximum_controls=35200,maximum_substeps=352000;
constexpr uint64_t evidence_cap=512ull*1024*1024;
constexpr double time_cap=240;
const auto process_start=std::chrono::steady_clock::now();
struct Case {int id=0,fixture=0,pose=0,warm=80,scenario=0,arm=0,maximum=400;};
struct Sample {p::BodyState body{};double energy[6]{};uint64_t attempted=0,accepted=0;};
struct Frame {
    joint::World world{};old::ForwardTask task{};p::V initial_heading{};
    float observation[41]{},base[4]{},command[4]{};
    Sample substep[10]{};p::BodyState first_before{},candidate{};
    int valid=0,advanced=0,accepted=0,attempted=0,reason=0,failure=0;
    Case context{};int request_active=0,offset_degrees=0;
};
struct Budget {
    uint64_t controls=0,substeps=0,accepted=0,bytes=0;
    double seconds()const{return std::chrono::duration<double>(std::chrono::steady_clock::now()-process_start).count();}
    void before(){if(controls>=maximum_controls||substeps>=maximum_substeps||seconds()>=time_cap-10)
        throw std::runtime_error("Comparison physical/time cap reached");}
    void charge(uint64_t n){if(bytes+n>evidence_cap-131072)throw std::runtime_error("Comparison evidence cap reached");bytes+=n;}
};
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
old::ForwardConfig config(){auto c=old::config_for_speed(.008f);c.observation_schema=4;c.maximum_controls=case_controls;return c;}
nn::MlpSpec spec(){return {{41,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}};}
CUDARL_HD inline int post_controls(int scenario){return scenario<=4?320:scenario<=6?260:scenario<=8?270:340;}
CUDARL_HD inline Case case_for(int id){
    Case c;c.id=id;c.arm=id%2;c.scenario=(id/2)%11;c.fixture=id/22;
    c.pose=c.fixture==0?0:c.fixture==1?3:c.fixture==2?1:2;
    c.warm=c.fixture<2?80:90;c.maximum=c.warm+post_controls(c.scenario);return c;
}
CUDARL_HD inline Case gate_for(int id){
    Case c;c.id=id;c.fixture=id/2;c.arm=id%2;c.pose=id<2?0:3;c.warm=id<2?80:90;
    c.scenario=id<2?0:1;c.maximum=id<2?120:130;return c;
}
CUDARL_HD inline bool event(const Case& c,int tick,int& offset){
    const int relative=tick-c.warm-1;if(relative<0)return false;
    const int sign=c.scenario%2?1:-1;
    if(c.scenario==0){offset=0;return relative==0;}
    if(c.scenario<=4){offset=sign*(c.scenario<=2?5:15);return relative==0;}
    if(c.scenario<=6){offset=sign*5*(relative/20+1);return relative<=100&&relative%20==0;}
    if(c.scenario<=8){offset=sign*5*(relative/10+1);return relative<=110&&relative%10==0;}
    if(relative<=100&&relative%20==0){offset=sign*5*(relative/20+1);return true;}
    if(relative>=160&&relative<=260&&relative%20==0){offset=sign*(30-5*((relative-160)/20+1));return true;}
    return false;
}
CUDARL_HD inline void energy(const joint::World& w,double* e){
    e[0]=w.signed_work;e[1]=w.positive_work;e[2]=w.fluid_loss;e[3]=w.numerical_loss;e[4]=w.external_work;e[5]=w.last_energy_error;
}
CUDARL_HD inline bool initialize(Case context,Frame& f){
    f={};f.context=context;f.world=old::initial_world(context.pose);f.world.arena=old::long_swim_arena();
    auto c=old::config_for_speed(.008f);c.observation_schema=4;c.maximum_controls=case_controls;
    if(!old::begin_forward(f.world,c,f.task))return false;
    f.initial_heading=f.task.request.heading;f.valid=1;return true;
}
CUDARL_HD inline bool request(Frame& f,int tick){
    if(tick>=f.context.warm+1)f.request_active=1;
    int offset=0;if(!event(f.context,tick,offset))return true;
    if(offset==f.offset_degrees)return true; // The zero program keeps the entire task identical.
    const auto com=cudarl::motor::motion(f.world.physical.body).com_model;
    const auto heading=p::axis_rotation({0,1,0},float(offset)*model::pi/180.f).rotate(f.initial_heading);
    f.task.start_com_model.x=com.x;f.task.start_com_model.z=com.z;
    f.task.request.heading=heading;f.task.request.terminal_velocity=heading*.008f;
    f.task.request.position=f.task.start_com_model+heading*(.072f/p::scale);
    f.offset_degrees=offset;return p::finite_vector(heading);
}
CUDARL_HD inline bool observe(Frame& f){
    auto c=old::config_for_speed(.008f);c.observation_schema=4;c.maximum_controls=case_controls;
    auto view=f.task;
    if(f.context.arm==1&&f.request_active){const auto com=cudarl::motor::motion(f.world.physical.body).com_model;
        view.start_com_model.x=com.x;view.start_com_model.z=com.z;}
    return old::observe_appended(view,f.world,c,f.observation);
}
CUDARL_HD inline int point_guard(const Frame& f){
    const auto b=bank::bounds(f.world.physical.body);
    if(!b.finite||b.active_radius_squared>1)return 104;
    const float depth=(cudarl::motor::motion(f.world.physical.body).com_model.y-f.task.start_com_model.y)*p::scale;
    const float tilt=acosf(p::clamp(f.world.physical.body.orientation.rotate({0,1,0}).y,-1,1))*180/model::pi;
    return b.inset_radius_squared>=1?1:fabsf(depth)>.005f?4:tilt>20?5:0;
}
// Identical shared 1Hz target filter, 5ms mechanics and torque limits.
// Accepted-state guards inspect real state before another substep can hide it.
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
        const int guard=point_guard(f);
        if(guard==104){w.failure_code=f.failure=104;s.unsafe=true;return false;}
        f.reason=guard;
        if(f.reason)break;
    }
    s.mean_squared_torque/=float(f.accepted*4);++s.control_steps;
    if(!joint::in_domain(w)){f.failure=105;return false;}
    auto c=old::config_for_speed(.008f);c.observation_schema=4;c.maximum_controls=case_controls;
    const auto transition=old::record_control(f.task,w,c);
    if(!transition.accepted){f.failure=106;return false;}
    f.advanced=1;return true;
}
__global__ void initial(Frame* f,int n){int i=threadIdx.x;if(i<n)initialize(case_for(i),f[i]);}
__global__ void observations(Frame* f,int n){int i=threadIdx.x;if(i<n)f[i].valid=f[i].valid&&observe(f[i])?1:0;}
__global__ void gather(const Frame* f,float* x,int n){int i=threadIdx.x;if(i<n)for(int j=0;j<41;++j)x[i*41+j]=f[i].observation[j];}
__global__ void decode(Frame* f,const float* z,int n){int i=threadIdx.x;if(i<n)for(int j=0;j<4;++j){f[i].base[j]=z[i*4+j];f[i].command[j]=tanhf(z[i*4+j]);}}
__global__ void stepping(Frame* f){advance(*f);}
__global__ void request_latch(Frame* f,int tick){f->valid=request(*f,tick)?1:0;}
__global__ void guard_query(const Frame* f,int* output,int n){int i=threadIdx.x;if(i<n)output[i]=point_guard(f[i]);}
void sync(){CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());}

#include "frozen_interface_comparison_common.inc"

void case_json(std::ostream& out,const Case& c){
    out<<"{\"case\":"<<c.id<<",\"fixture\":"<<c.fixture<<",\"pose\":"<<c.pose<<",\"warm_controls\":"<<c.warm
       <<",\"scenario\":"<<c.scenario<<",\"arm\":\""<<(c.arm?"heading_only":"lane_following")<<"\",\"maximum_controls\":"<<c.maximum<<",\"events\":[";
    bool first=true;int previous=0;
    for(int tick=1;tick<=c.maximum;++tick){int offset=0;if(!event(c,tick,offset))continue;
        if(!first)out<<',';first=false;out<<"{\"before_control\":"<<tick<<",\"offset_degrees\":"<<offset
            <<",\"rebase_lane\":"<<(offset!=previous?"true":"false")<<'}';previous=offset;}
    out<<"]}";
}
void write_cases(const fs::path& root){
    std::ofstream out(root/"cases.json");out<<"{\"schema\":1,\"study\":\"frozen_interface_comparison\",\"cases\":[";
    for(int i=0;i<case_count;++i){if(i)out<<',';case_json(out,case_for(i));}
    out<<"],\"gate_cases\":[";for(int i=0;i<gate_count;++i){if(i)out<<',';case_json(out,gate_for(i));}out<<"]}\n";
    out.close();require(bool(out),"Case catalog write failed");
}
void layout(const fs::path& root){
    std::ofstream out(root/"layout.json");out<<"{\"frame\":{\"size\":"<<sizeof(Frame);
#define FIELD(Type,name) <<",\"" #name "\":"<<offsetof(Type,name)
    out FIELD(Frame,world) FIELD(Frame,task) FIELD(Frame,initial_heading) FIELD(Frame,observation) FIELD(Frame,base) FIELD(Frame,command)
        FIELD(Frame,substep) FIELD(Frame,first_before) FIELD(Frame,candidate) FIELD(Frame,valid) FIELD(Frame,advanced) FIELD(Frame,accepted)
        FIELD(Frame,attempted) FIELD(Frame,reason) FIELD(Frame,failure) FIELD(Frame,context) FIELD(Frame,request_active) FIELD(Frame,offset_degrees)
        <<"},\"task\":{\"size\":"<<sizeof(old::ForwardTask);
    out FIELD(old::ForwardTask,request) FIELD(old::ForwardTask,start_com_model) FIELD(old::ForwardTask,controls)
        FIELD(old::ForwardTask,last_world_control) FIELD(old::ForwardTask,outcome)<<"},\"request\":{\"size\":"<<sizeof(cudarl::simulation::MotorRequest);
    out FIELD(cudarl::simulation::MotorRequest,position) FIELD(cudarl::simulation::MotorRequest,terminal_velocity)
        FIELD(cudarl::simulation::MotorRequest,heading)<<"},\"sample\":{\"size\":"<<sizeof(Sample);
    out FIELD(Sample,body) FIELD(Sample,energy) FIELD(Sample,attempted) FIELD(Sample,accepted)<<"},\"case\":{\"size\":"<<sizeof(Case);
    out FIELD(Case,id) FIELD(Case,fixture) FIELD(Case,pose) FIELD(Case,warm) FIELD(Case,scenario) FIELD(Case,arm) FIELD(Case,maximum)
        <<"},\"World\":{\"size\":"<<sizeof(joint::World)<<"},\"BodyState\":{\"size\":"<<sizeof(p::BodyState)<<"}}\n";
#undef FIELD
    out.close();require(bool(out),"Layout write failed");
}
int prepare(const fs::path& root,const fs::path& parents,const fs::path& document,const fs::path& exe){
    require(!fs::exists(root),"Prepare a fresh comparison root; never overwrite evidence");
    require(fs::create_directories(root),"Cannot create comparison root");
    for(const char* name:{"parent-checkpoint.bin","parent-protocol.bin","holding-parent.bin"})fs::copy_file(parents/name,root/name);
    fs::copy_file(document,root/"approved-protocol.md");const auto weights=load(root);
    write_cases(root);layout(root);
    cp::Writer payload;payload.value(uint32_t{0x31524946});payload.value(uint32_t{1});
    payload.value(grant::whole_file(root/"approved-protocol.md"));payload.value(grant::whole_file(root/"cases.json"));
    for(const char* name:{"parent-checkpoint.bin","parent-protocol.bin","holding-parent.bin"})payload.value(cp::checksum(cp::read((root/name).string())));
    for(int v:{6,4,41,4,336,case_count,gate_count,case_controls,maximum_controls,maximum_substeps,240})payload.value(v);
    for(float v:{.008f,.005f,20.f,.20f,.12f,.195f,.115f})payload.value(v);payload.value(evidence_cap);
    cp::write_atomic((root/"protocol.bin").string(),payload.data);
    nn::CpuMlp<float> cpu(spec());cpu.params()=weights;nn::GpuMlp gpu(spec(),case_count,1);gpu.set_params(weights);
    device_ptr<Frame> d_state(case_count);device_ptr<float> d_input(case_count*41);
    initial<<<1,128>>>(d_state.get(),case_count);sync();infer(gpu,d_state,d_input,case_count);
    std::vector<Frame> initial_rows(case_count);d_state.copy_to_host(initial_rows.data(),case_count);
    double maximum_inference=0,maximum_latch=0;
    auto check_inference=[&](const Frame& f){require(f.valid&&f.world.physical.control_steps==0&&f.world.attempted_substeps==0&&f.world.physical.substeps==0,"Pure fixture integrated or failed");
        cpu.forward(f.observation,1);for(int j=0;j<4;++j){const double z=std::abs(double(cpu.output()[j])-f.base[j]);
            const double a=std::abs(double(tanhf(cpu.output()[j]))-f.command[j]);maximum_inference=std::max({maximum_inference,z,a});require(z<=2e-5&&a<=2e-5,"Pure same-observation inference differs");}};
    for(const auto& f:initial_rows)check_inference(f);
    cp::Writer latches;latches.value(uint32_t{0x314C4946});latches.value(uint32_t{1});latches.value(protocol(root));latches.value(uint32_t{520});
    std::vector<Frame> before(case_count),after(case_count);
    int latch_count=0;
    for(int i=0;i<case_count;++i){auto pre=initial_rows[i];bool first=true;
        for(int tick=1;tick<=pre.context.maximum;++tick){int offset=0;if(!event(pre.context,tick,offset))continue;auto post=pre;
            require(request(post,tick),"CPU pure request failed");
            require(std::memcmp(&pre.world,&post.world,sizeof(pre.world))==0,"Request mutated world");
            require(pre.task.controls==post.task.controls&&pre.task.last_world_control==post.task.last_world_control&&pre.task.start_com_model.y==post.task.start_com_model.y,"Request changed depth or clocks");
            if(pre.context.scenario==0)require(std::memcmp(&pre.task,&post.task,sizeof(pre.task))==0,"Zero request changed task");
            d_state.copy_from_host(&pre,1);request_latch<<<1,1>>>(d_state.get(),tick);sync();Frame device;d_state.copy_to_host(&device,1);
            require(std::memcmp(&pre.world,&device.world,sizeof(pre.world))==0,"GPU request mutated world");
            const double delta=p::norm(post.task.request.heading-device.task.request.heading);maximum_latch=std::max(maximum_latch,delta);require(delta<=2e-5,"Pure request backend difference");
            latches.value(i);latches.value(tick);latches.value(pre);latches.value(post);latches.value(device);++latch_count;
            if(first){before[i]=post;first=false;}pre=post;
        }
        auto& synthetic=before[i];synthetic.world.physical.body.position.x+=.025f;synthetic.world.physical.body.position.y+=.025f;synthetic.world.physical.body.position.z+=.025f;
        synthetic.world.physical.body.orientation=p::axis_rotation({1,0,0},.15f);
        for(float& v:synthetic.observation)v=0;for(float& v:synthetic.base)v=0;for(float& v:synthetic.command)v=0;
    }
    require(latch_count==520,"Pure scheduled event count differs");
    cp::write_atomic((root/"pure-latches.bin").string(),latches.data);
    d_state.copy_from_host(before.data(),case_count);infer(gpu,d_state,d_input,case_count);d_state.copy_to_host(after.data(),case_count);
    for(int i=0;i<case_count;++i){require(std::memcmp(&before[i].world,&after[i].world,sizeof(joint::World))==0&&
        std::memcmp(&before[i].task,&after[i].task,sizeof(old::ForwardTask))==0,"Observer mutated world or task");check_inference(after[i]);}
    cp::Writer fixture;fixture.value(uint32_t{0x31464946});fixture.value(uint32_t{1});fixture.value(protocol(root));fixture.value(uint32_t{sizeof(Frame)});
    fixture.array(initial_rows);fixture.array(before);fixture.array(after);cp::write_atomic((root/"fixtures.bin").string(),fixture.data);
    std::vector<Frame> guard_rows(12);const int expected_guards[12]={0,0,4,0,4,0,5,0,1,104,4,1};
    for(auto& row:guard_rows)require(initialize(case_for(0),row),"Pure guard initialization failed");
    const float depth_offsets[5]={0,.0049f,.0051f,-.0049f,-.0051f};
    for(int i=1;i<5;++i)guard_rows[i].world.physical.body.position.y+=depth_offsets[i]/p::scale;
    guard_rows[5].world.physical.body.orientation=p::axis_rotation({1,0,0},19.f*model::pi/180.f);
    guard_rows[6].world.physical.body.orientation=p::axis_rotation({1,0,0},21.f*model::pi/180.f);
    guard_rows[7].world.physical.body.position.x+=.170f/p::scale;
    guard_rows[8].world.physical.body.position.x+=.188f/p::scale;
    guard_rows[9].world.physical.body.position.x+=.198f/p::scale;
    guard_rows[10].world.physical.body.position.y+=.0051f/p::scale;
    guard_rows[10].world.physical.body.orientation=p::axis_rotation({1,0,0},21.f*model::pi/180.f);
    guard_rows[11]=guard_rows[10];guard_rows[11].world.physical.body.position.x+=.188f/p::scale;
    d_state.copy_from_host(guard_rows.data(),12);device_ptr<int> d_guards(12);guard_query<<<1,128>>>(d_state.get(),d_guards.get(),12);sync();int device_guards[12];d_guards.copy_to_host(device_guards,12);
    cp::Writer guards;guards.value(uint32_t{0x31474946});guards.value(uint32_t{1});guards.value(protocol(root));guards.value(uint32_t{12});
    for(int i=0;i<12;++i){const int host=point_guard(guard_rows[i]);require(host==expected_guards[i]&&device_guards[i]==host,"Pure guard classification differs");
        guards.value(i);guards.value(expected_guards[i]);guards.value(guard_rows[i]);guards.value(host);guards.value(device_guards[i]);}
    cp::write_atomic((root/"pure-guards.bin").string(),guards.data);
    cp::write_atomic((root/"prepared.bin").string(),binding(root,exe).data);
    std::ofstream report(root/"preparation.json");report<<std::setprecision(17)<<"{\"schema\":1,\"study\":\"frozen_interface_comparison\",\"passed\":true,\"physical_controls\":0,\"physical_substeps\":0,\"PPO_updates\":0,\"optimizer_steps\":0,\"initial_fixtures\":"<<case_count
        <<",\"observer_fixtures\":"<<case_count<<",\"request_latch_fixtures\":520,\"guard_fixtures\":12,\"maximum_same_observation_inference_difference\":"<<maximum_inference<<",\"maximum_latch_heading_difference\":"<<maximum_latch
        <<",\"world_bytes\":"<<sizeof(joint::World)<<",\"task_bytes\":"<<sizeof(old::ForwardTask)<<",\"frame_bytes\":"<<sizeof(Frame)<<",\"conservative_generated_evidence_bytes\":"<<projected_run_bytes()<<",\"evidence_cap_bytes\":"<<evidence_cap<<",\"physical_run_approved\":false}\n";
    report.close();require(bool(report),"Preparation report write failed");
    std::cout<<"Prepared frozen interface comparison: zero physical controls and optimizer steps\n";return 0;
}

int run(const fs::path& root,const fs::path& exe){
    require(cp::read((root/"prepared.bin").string())==binding(root,exe).data,"Prepared executable, parents or protocol changed");
    grant::require_approved(root,protocol(root));
    require(!fs::exists(root/"result.json")&&!fs::exists(root/"probe-started"),"Comparison already attempted; no retry");
    Budget budget;for(const auto& file:fs::recursive_directory_iterator(root))if(file.is_regular_file())budget.bytes+=file.file_size();
    require(budget.bytes+projected_run_bytes()<=evidence_cap,"Frozen inputs and projected evidence exceed cap");
    require(fs::space(root).available>=evidence_cap+1024ull*1024*1024,"Comparison evidence and reserve space unavailable");
    require(fs::create_directory(root/"probe-started"),"Cannot create one-shot comparison directory");
    const auto out=root/"probe-started";
    std::ofstream states(out/"controls.csv"),substeps(out/"substeps.csv"),pointwise(out/"same-observation.csv"),summary(out/"cases.csv");
    cp::Writer raw;raw.value(uint32_t{1});raw.value(uint32_t{sizeof(joint::World)});raw.value(uint32_t{sizeof(old::ForwardTask)});
    std::ostringstream h;h<<"group,case,backend,control,seconds,world_controls,attempted_substeps,accepted_substeps,last_attempted,last_accepted,behavior_reason,hard_failure,legacy_outcome,task_controls";body_header(h);
    for(const char* prefix:{"lane_origin","target","heading","terminal_velocity"})for(char axis:{'x','y','z'})h<<','<<prefix<<'_'<<axis;
    for(const auto item:{std::pair<const char*,int>{"held",4},{"reference_rate",4},{"previous_torque",4},{"observation",41},{"base",4},{"command",4},{"energy",6}})for(int j=0;j<item.second;++j)h<<','<<item.first<<j;h<<'\n';
    checked_line(states,h.str(),budget);
    std::ostringstream sh;sh<<"group,case,backend,control,substep,attempted_substeps,accepted_substeps";body_header(sh);for(int j=0;j<6;++j)sh<<",energy"<<j;sh<<'\n';checked_line(substeps,sh.str(),budget);
    std::ostringstream ph;ph<<"case,control";for(int j=0;j<41;++j)ph<<",observation"<<j;
    for(const char* n:{"cpu_base","gpu_base","cpu_command","gpu_command"})for(int j=0;j<4;++j)ph<<','<<n<<j;ph<<'\n';checked_line(pointwise,ph.str(),budget);
    checked_line(summary,"case,fixture,pose,warm_controls,scenario,arm,maximum_controls,controls,seconds,behavior_reason\n",budget);
    std::string fault;int completed_gate=0,cases_started=0,completed_cases=0;
    double maximum_position=0,maximum_inference=0;Frame active{},other{};int active_group=0,active_id=0,active_tick=0;
    try{
        const auto weights=load(root);nn::CpuMlp<float> cpu(spec());cpu.params()=weights;
        nn::GpuMlp gpu(spec(),1,1);gpu.set_params(weights);device_ptr<Frame> d_state(1);device_ptr<float> d_input(41);
        for(int id=0;id<gate_count;++id){active_group=0;active_id=id;const auto context=gate_for(id);
            require(initialize(context,active),"Gate initialization failed");other=active;
            capture(states,substeps,raw,budget,0,id,0,0,active);capture(states,substeps,raw,budget,0,id,1,0,other);
            for(int tick=1;tick<=context.maximum;++tick){active_tick=tick;budget.before();
                require(request(active,tick)&&request(other,tick),"Matched request rejected");
                require(observe(active),"CPU gate observer rejected");cpu.forward(active.observation,1);
                for(int j=0;j<4;++j){active.base[j]=cpu.output()[j];active.command[j]=tanhf(active.base[j]);}
                d_input.copy_from_host(active.observation,41);gpu.forward(d_input.get(),1);sync();
                std::array<float,4> gpu_base;CUDA_CHECK_THROW(cudaMemcpy(gpu_base.data(),gpu.output(),4*sizeof(float),cudaMemcpyDeviceToHost));
                bool same=true;
                for(int j=0;j<4;++j){const double z=std::abs(double(gpu_base[j])-active.base[j]);
                    const double a=std::abs(double(tanhf(gpu_base[j]))-active.command[j]);maximum_inference=std::max({maximum_inference,z,a});
                    same=same&&z<=2e-5&&a<=2e-5;other.base[j]=active.base[j];other.command[j]=active.command[j];}
                for(int j=0;j<41;++j)other.observation[j]=active.observation[j];
                std::ostringstream pr;pr<<std::setprecision(9)<<id<<','<<tick;for(float v:active.observation)pr<<','<<v;
                for(float v:active.base)pr<<','<<v;for(float v:gpu_base)pr<<','<<v;
                for(float v:active.command)pr<<','<<v;for(float v:gpu_base)pr<<','<<tanhf(v);pr<<'\n';
                checked_line(pointwise,pr.str(),budget);require(same,"Gate same-observation inference mismatch");
                ++budget.controls;advance(active);budget.substeps+=active.attempted;budget.accepted+=active.accepted;
                capture(states,substeps,raw,budget,0,id,0,tick,active);require(active.advanced&&!active.reason,"CPU gate failed or terminated");
                budget.before();++budget.controls;d_state.copy_from_host(&other,1);stepping<<<1,1>>>(d_state.get());sync();d_state.copy_to_host(&other,1);
                budget.substeps+=other.attempted;budget.accepted+=other.accepted;
                capture(states,substeps,raw,budget,0,id,1,tick,other);require(other.advanced&&!other.reason,"CUDA gate failed or terminated");
                agreement(active,other,maximum_position);
            }++completed_gate;
        }
        for(int id=0;id<case_count;++id){active_group=1;active_id=id;active_tick=0;++cases_started;const auto context=case_for(id);
            require(initialize(context,active),"Comparison case initialization failed");d_state.copy_from_host(&active,1);
            capture(states,substeps,raw,budget,1,id,1,0,active);
            int completed_controls=0;
            for(int tick=1;tick<=context.maximum;++tick){active_tick=tick;budget.before();
                request_latch<<<1,1>>>(d_state.get(),tick);sync();infer(gpu,d_state,d_input,1);d_state.copy_to_host(&active,1);
                require(active.valid,"CUDA comparison observer failed");++budget.controls;
                stepping<<<1,1>>>(d_state.get());sync();d_state.copy_to_host(&active,1);
                budget.substeps+=active.attempted;budget.accepted+=active.accepted;
                capture(states,substeps,raw,budget,1,id,1,tick,active);require(active.advanced,"Comparison hard physical fault");
                completed_controls=tick;if(active.reason)break;
            }
            std::ostringstream row;row<<std::setprecision(17)<<id<<','<<context.fixture<<','<<context.pose<<','<<context.warm<<','<<context.scenario<<','<<context.arm
                <<','<<context.maximum<<','<<completed_controls<<','<<active.world.physical.time_seconds<<','<<active.reason<<'\n';checked_line(summary,row.str(),budget);++completed_cases;
        }
        require(budget.controls<=maximum_controls&&budget.substeps<=maximum_substeps&&budget.seconds()<=time_cap,"Final comparison cap violated");
    }catch(const std::exception& e){fault=e.what();cp::Writer failed;failed.value(uint32_t{1});failed.value(active_group);failed.value(active_id);failed.value(active_tick);
        failed.value(active);failed.value(other);cp::write_atomic((out/"first-failure.bin").string(),failed.data);}
    cp::write_atomic((out/"states.bin").string(),raw.data);states.close();substeps.close();pointwise.close();summary.close();
    if((!states||!substeps||!pointwise||!summary)&&fault.empty())fault="Evidence flush failed";
    uint64_t actual_bytes=0;for(const auto& file:fs::recursive_directory_iterator(root))if(file.is_regular_file())actual_bytes+=file.file_size();
    if(fault.empty()&&(actual_bytes+16384>evidence_cap||budget.seconds()>time_cap))fault="Final evidence/time cap exceeded";
    std::ofstream result(root/"result.json");result<<std::setprecision(17)<<"{\"schema\":1,\"study\":\"frozen_interface_comparison\",\"closed\":true,\"hard_fault\":"<<std::quoted(fault)
        <<",\"precheck_completed_cases\":"<<completed_gate<<",\"cases_started\":"<<cases_started<<",\"probe_completed_cases\":"<<completed_cases<<",\"controls\":"<<budget.controls
        <<",\"attempted_substeps\":"<<budget.substeps<<",\"accepted_substeps\":"<<budget.accepted<<",\"process_seconds\":"<<budget.seconds()<<",\"evidence_bytes_before_close_report\":"<<actual_bytes
        <<",\"maximum_position_difference_model\":"<<maximum_position<<",\"maximum_same_observation_inference_difference\":"<<maximum_inference
        <<",\"PPO_updates\":0,\"optimizer_steps\":0,\"training\":false,\"qualified\":false,\"runtime_adopted\":false,\"descriptive_comparison_only\":true}\n";
    result.close();require(bool(result),"Comparison close report write failed");
    std::cout<<"Frozen interface comparison closed controls="<<budget.controls<<" steps="<<budget.substeps<<" fault="<<fault<<'\n';return fault.empty()?0:3;
}
}
int main(int argc,char** argv){try{
    const auto exe=std::filesystem::absolute(argv[0]);
    if(argc==5&&std::string(argv[1])=="--prepare")return comparison::prepare(std::filesystem::absolute(argv[2]),std::filesystem::absolute(argv[3]),std::filesystem::absolute(argv[4]),exe);
    if(argc==2)return comparison::run(std::filesystem::absolute(argv[1]),exe);
    throw std::runtime_error("Use --prepare RUN PARENTS PROTOCOL_MD, or RUN after frozen preparation and explicit comparison approval");
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
