// PREPARATION: explicit --run and sealed prerequisite bindings are mandatory.
// No work runs at build time. The original package is read-only.
#include "cuda/pond_math_profile.h"
#include "rl/pond_ppo_task.h"
#include "rl/pond_ppo_schedule.h"
#include "rl/pond_ppo_contract.h"
#include "motor/saved_motor_contract.h"
#include "rl/learned_amplitude_math.h"
#include "nn/mlp_gpu.h"
#include "nn/adam.h"
#include "rl/gaussian_kernels.h"
#include "rl/gaussian_math.h"
#include "rl/ppo_kernels.h"
#include "admission.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace fs=std::filesystem;
namespace pp=cudarl::rl::pond_ppo;
namespace pt=cudarl::rl::pond_ppo_task;
namespace ps=cudarl::rl::pond_ppo_schedule;
namespace pr=cudarl::pond_runtime;
namespace po=cudarl::pond_motor_observation;
namespace cp=cudarl::checkpoint;
namespace nn=cudarl::nn;
namespace rl=cudarl::rl;
namespace p=cudarl::physics;
namespace ph=cudarl::rl::pond_ppo_host;
using ph::whole_file;using ph::text_file;using ph::sha256;using ph::scalar;using ph::source_identity;
namespace {
constexpr int N=static_cast<int>(pp::worlds),T=static_cast<int>(pp::controls_per_world);
constexpr int Rows=N*T,D=pp::observations,A=pp::actions,MB=static_cast<int>(pp::minibatch);
constexpr uint32_t default_seed=0x504f4e44u;
constexpr float maximum_approx_kl=.05f;
static_assert(std::is_trivially_copyable_v<pr::World>&&std::is_trivially_copyable_v<pt::Task>);
static_assert(pr::version==pp::runtime_version,"Qualify the exact candidate runtime before building the trainer");
struct Options {
    bool run=false,preflight=false,describe=false,initialize_only=false;
    fs::path policy,out,protocol,cpu_gate,cuda_gate,resume,retention;
    uint64_t evaluator_source=0;
    uint32_t to_update=8,study_program=pt::curriculum_off;
};
struct Package {std::vector<float> weights;float gain=0;uint64_t checksum=0;};
const char* study_id(uint32_t program){
    if(program==pt::half_second_course_v1)return pp::half_second_study_protocol_id;
    return program==pt::repeated_headings_v1?pp::repeated_study_protocol_id:(program==pt::course_direction_v1?pp::course_study_protocol_id:(program==pt::paired_gentle_moderate_v1?pp::curriculum_study_protocol_id:pp::default_study_protocol_id));
}
const char* program_name(uint32_t program){
    if(program==pt::half_second_course_v1)return "half-second-course-v1";
    return program==pt::repeated_headings_v1?"repeated-headings-v1":(program==pt::course_direction_v1?"course-direction-v1":(program==pt::paired_gentle_moderate_v1?"paired-gentle-moderate-v1":"off"));
}
uint32_t update_budget(uint32_t program){return program==pt::half_second_course_v1?8:pp::updates;}
uint32_t task_stride(uint32_t program){
    if(program==pt::half_second_course_v1)return sizeof(ps::HalfSecondTask);
    return static_cast<uint32_t>(program==pt::repeated_headings_v1?sizeof(ps::SavedTask):sizeof(pt::Task));
}
void require(bool yes,const std::string& reason){if(!yes)throw std::runtime_error(reason);}
std::string json_escape(const std::string& value) {
    std::string result;
    for(char c:value){if(c=='"'||c=='\\')result+='\\';if(c=='\n')result+="\\n";else if(c=='\r')result+="\\r";else result+=c;}
    return result;
}
Package load_package(const fs::path& path) {
    require(sha256(whole_file(path))==pp::parent_package_sha256,"Original saved package SHA256 mismatch");
    const auto payload=cp::read(path.string());cp::Reader in{payload};Package result;result.checksum=cp::checksum(payload);
    for(uint32_t expected:{0x31564c48u,2u,6u,6u,3u,4u,41u,4u,336u,1u,7u,1u})require(in.value<uint32_t>()==expected,"Original saved motor contract mismatch");
    require(in.value<uint64_t>()!=0,"Missing original protocol");const auto parent=in.array<uint8_t>(64);
    require(std::string(parent.begin(),parent.end())=="810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0","Wrong original parent");
    const auto& hashes=cudarl::motor::saved_motor_source_contract_hashes();require(in.value<uint32_t>()==hashes.size(),"Wrong source count");
    for(auto expected:hashes){const auto stored=in.array<uint8_t>(64);require(std::string(stored.begin(),stored.end())==expected,"Wrong saved provenance");}
    result.weights=in.array<float>(pp::actor_weights);const auto head=in.array<float>(2);
    const float gain=in.value<float>(),sigma=in.value<float>();in.finish();
    require(result.weights.size()==pp::actor_weights&&head.size()==2,"Original parameter shape mismatch");
    for(float value:result.weights)require(std::isfinite(value),"Nonfinite original weight");
    result.gain=cudarl::learned_amplitude::sigmoid(head[0]);
    require(result.gain==pp::frozen_holding_gain&&std::abs(gain-result.gain)<=2e-7f&&
        std::abs(sigma-cudarl::learned_amplitude::standard_deviation(head[1]))<=2e-6f,"Original holding mapping mismatch");return result;
}
Options options(int argc,char** argv) {
    Options result;
    for(int i=1;i<argc;++i){const std::string arg=argv[i];const auto next=[&](){require(i+1<argc,"Missing value for "+arg);return std::string(argv[++i]);};
        if(arg=="--run")result.run=true;else if(arg=="--preflight")result.preflight=true;else if(arg=="--describe")result.describe=true;else if(arg=="--initialize-only")result.initialize_only=true;
        else if(arg=="--policy")result.policy=next();else if(arg=="--out")result.out=next();else if(arg=="--protocol")result.protocol=next();
        else if(arg=="--cpu-gate")result.cpu_gate=next();else if(arg=="--cuda-gate")result.cuda_gate=next();else if(arg=="--resume")result.resume=next();
        else if(arg=="--retention-receipt")result.retention=next();else if(arg=="--evaluator-source")result.evaluator_source=std::stoull(next());
        else if(arg=="--turn-curriculum"){require(result.study_program==0,"Study programs are mutually exclusive");require(next()=="paired-gentle-moderate-v1","Unknown training-only curriculum");result.study_program=pt::paired_gentle_moderate_v1;}
        else if(arg=="--turn-reward"){require(result.study_program==0,"Study programs are mutually exclusive");require(next()=="course-direction-v1","Unknown turn reward program");result.study_program=pt::course_direction_v1;}
        else if(arg=="--turn-schedule"){
            require(result.study_program==0,"Study programs are mutually exclusive");
            const auto name=next();
            if(name=="half-second-course-v1")result.study_program=pt::half_second_course_v1;
            else {require(name=="repeated-headings-v1","Unknown turn schedule");result.study_program=pt::repeated_headings_v1;}
        }
        else if(arg=="--to-update"){const auto value=std::stoull(next());require(value<=32,"Update budget exceeds32");result.to_update=static_cast<uint32_t>(value);}
        else throw std::runtime_error("Unknown argument: "+arg);}
    require(int(result.run)+int(result.preflight)+int(result.describe)+int(result.initialize_only)==1,"Use exactly --describe, --preflight, --initialize-only or --run");
    require(result.to_update<=update_budget(result.study_program),"Update target exceeds this study's allocation");
    if(!result.describe)require(!result.policy.empty()&&!result.out.empty(),"Policy and unique output required");return result;
}
void prerequisites(const Options& o,uint64_t parent_checksum) {
    ph::prerequisites(o.protocol,o.cpu_gate,o.cuda_gate,o.evaluator_source,parent_checksum,o.study_program);
}
struct TransitionTrace {
    float x=0,z=0,vx=0,vy=0,vz=0,depth=0,tilt=0;
    double minimum_gap=0;
    uint32_t episode=0,control=0,kind=0,occupied_cells=0,occupied_grid[8]{};
    float bank_near_seconds=0,contact_seconds=0,postwarmup_stall_seconds=0;
    float heading_x=0,heading_z=0,requested_angle=0;
    uint32_t request_serial=0,onset_control=0,request_age=0,qualifying_streak=0,events=0,active=0,succeeded=0,long_hold=0;
    uint64_t onsets=0,successes=0,timeouts=0,interrupted=0,early_terminal=0,long_holds=0;
};
struct Buffer {
    device_ptr<pr::World> worlds{N};device_ptr<pt::Task> tasks{N};device_ptr<ps::State> schedules{N};
    device_ptr<ps::CourseHistory> histories{N};
    device_ptr<float> observation{Rows*D},latent{Rows*A},logp{Rows},value{Rows},reward{Rows},done{Rows},truncated{Rows};
    device_ptr<float> final_value{Rows},final_observation{N*D},last_observation{N*D},last_value{N};
    device_ptr<float> advantages{Rows},returns{Rows},adv_stats{2},log_std{A};
    device_ptr<float> kl{Rows};
    device_ptr<uint64_t> attempted{Rows},substeps{Rows};device_ptr<int> reason{Rows},fault{1};
    device_ptr<TransitionTrace> trace{Rows};
    device_ptr<float> mb_observation{MB*D},mb_latent{MB*A},mb_logp{MB},mb_advantage{MB},mb_return{MB};
    device_ptr<float> dmean{MB*A},dlog_std{MB*A},dvalue{MB},row_stats{MB*rl::kGaussianStats},stats{rl::kGaussianStats};
};
__device__ void fail(int* fault,int env){atomicMin(fault,env);}
__global__ void initialize_kernel(pr::World* worlds,pt::Task* tasks,ps::State* schedules,ps::CourseHistory* histories,uint32_t seed,uint32_t curriculum,int* fault) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
    schedules[i]=ps::State{};
    if(curriculum==pt::half_second_course_v1)histories[i]=ps::CourseHistory{};
    if(!pt::initialize(worlds[i],tasks[i],static_cast<uint32_t>(i),0,seed,curriculum))fail(fault,i);
}
__global__ void observe_kernel(pr::World* worlds,pt::Task* tasks,ps::State* schedules,float* output,int* fault,uint32_t program,uint32_t seed,bool collecting) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
    if(pt::repeating_program(program)){
        if(collecting){ps::before(schedules[i],tasks[i],static_cast<uint32_t>(i),seed,program);
            if(!pr::valid_state(worlds[i])||!pt::observe(tasks[i],worlds[i],output+i*D,false))fail(fault,i);}
        else {auto task=tasks[i];auto schedule=schedules[i];ps::before(schedule,task,static_cast<uint32_t>(i),seed,program);
            if(!pr::valid_state(worlds[i])||!pt::observe(task,worlds[i],output+i*D,false))fail(fault,i);}
    }else if(!pr::valid_state(worlds[i])||!pt::observe(tasks[i],worlds[i],output+i*D))fail(fault,i);
}
__global__ void step_kernel(pr::World* worlds,pt::Task* tasks,ps::State* schedules,ps::CourseHistory* histories,const float* latent,float gain,float* reward,
        float* done,float* truncated,float* final_observation,uint64_t* attempted,uint64_t* substeps,int* reason,TransitionTrace* trace,int* fault,uint32_t program,uint32_t seed) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
    float command[A]{};const float amplitude=tasks[i].request.holding?gain:1.f;
    for(int j=0;j<A;++j){const float z=latent[i*A+j];if(!p::finite_number(z)){reason[i]=20;fail(fault,i);return;}command[j]=tanhf(amplitude*z);}
    attempted[i]=1;const auto before=worlds[i].motor.attempted_substeps;
    const bool accepted=pr::step(worlds[i],command);substeps[i]=worlds[i].motor.attempted_substeps-before;
    if(!accepted){reason[i]=1000+worlds[i].failure;fail(fault,i);return;}
    const auto motion=po::motion(worlds[i].motor.physical.body);
    p::V course_velocity{};
    const p::V* feedback=nullptr;
    if(program==pt::half_second_course_v1 && tasks[i].kind==pt::Turn){
        auto& history=histories[i];
        if(tasks[i].episode_controls==0){
            const auto start=tasks[i].request.start_com_model * p::scale;
            history.append(start.x,start.z,course_velocity);
        }
        const auto position=motion.com_model * p::scale;
        if(history.append(position.x,position.z,course_velocity)) feedback=&course_velocity;
        if(tasks[i].request_changed && (!feedback || !p::finite_vector(course_velocity))){
            reason[i]=21;fail(fault,i);return;
        }
    }
    const auto transition=pt::record(tasks[i],worlds[i],program,feedback);reward[i]=transition.reward;
    const auto measured=po::measure(tasks[i].request,worlds[i].motor.physical.body);
    TransitionTrace row;row.x=motion.com_model.x*p::scale;row.z=motion.com_model.z*p::scale;
    row.vx=motion.velocity_m_s.x;row.vy=motion.velocity_m_s.y;row.vz=motion.velocity_m_s.z;row.depth=measured.depth_m;row.tilt=measured.tilt_deg;
    row.minimum_gap=worlds[i].minimum_control_gap_m;row.episode=tasks[i].episode_index;row.control=tasks[i].episode_controls;
    row.kind=static_cast<uint32_t>(tasks[i].kind);row.occupied_cells=tasks[i].occupied_cells;
    for(int word=0;word<8;++word)row.occupied_grid[word]=tasks[i].occupied_grid[word];
    row.bank_near_seconds=tasks[i].bank_near_seconds;row.contact_seconds=tasks[i].contact_seconds;row.postwarmup_stall_seconds=tasks[i].postwarmup_stall_seconds;
    if(pt::repeating_program(program)){auto& schedule=schedules[i];ps::after(schedule,tasks[i],{row.vx,row.vz,row.depth,row.tilt},transition.terminal,transition.truncated,feedback);
        row.heading_x=tasks[i].request.request.heading.x;row.heading_z=tasks[i].request.request.heading.z;row.requested_angle=schedule.angle;
        row.request_serial=schedule.serial;row.onset_control=schedule.onset_control;row.request_age=schedule.age;row.qualifying_streak=schedule.streak;row.events=schedule.events;row.active=schedule.active;row.succeeded=schedule.succeeded;row.long_hold=schedule.long_hold;
        row.onsets=schedule.onsets;row.successes=schedule.successes;row.timeouts=schedule.timeouts;row.interrupted=schedule.interrupted;row.early_terminal=schedule.early_terminal;row.long_holds=schedule.long_holds;}
    trace[i]=row;
    done[i]=transition.terminal?1.f:0.f;truncated[i]=transition.truncated?1.f:0.f;reason[i]=transition.reason;
    bool observed=false;
    if(pt::repeating_program(program)){auto task=tasks[i];auto schedule=schedules[i];
        if(!transition.terminal)ps::before(schedule,task,static_cast<uint32_t>(i),seed,program);
        observed=pt::observe(task,worlds[i],final_observation+i*D,false);
    }else observed=pt::observe(tasks[i],worlds[i],final_observation+i*D);
    if(!p::finite_number(reward[i])||!observed)fail(fault,i);
}
__global__ void reset_terminal_kernel(pr::World* worlds,pt::Task* tasks,ps::State* schedules,ps::CourseHistory* histories,const float* done,uint32_t seed,uint32_t curriculum,int* fault) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N||done[i]==0)return;
    const auto episode=tasks[i].episode_index+1;
    if(pt::repeating_program(curriculum))ps::reset_episode(schedules[i]);
    if(curriculum==pt::half_second_course_v1)histories[i]=ps::CourseHistory{};
    if(!pt::initialize(worlds[i],tasks[i],static_cast<uint32_t>(i),episode,seed,curriculum))fail(fault,i);
}
__global__ void audit_kernel(const float* obs,const float* latent,const float* logp,const float* value,const float* reward,
        const float* done,const float* truncated,const float* final_value,int* fault) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=Rows)return;
    bool valid=p::finite_number(logp[row])&&p::finite_number(value[row])&&p::finite_number(reward[row])&&p::finite_number(final_value[row]);
    valid=valid&&(done[row]==0||done[row]==1)&&(truncated[row]==0||truncated[row]==1)&&truncated[row]<=done[row];
    for(int j=0;j<D;++j)valid=valid&&p::finite_number(obs[row*D+j]);
    for(int j=0;j<A;++j)valid=valid&&p::finite_number(latent[row*A+j]);
    if(!valid)fail(fault,row);
}
__global__ void finite_kernel(const float* values,int size,int* fault) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<size&&!p::finite_number(values[i]))fail(fault,i);
}
__global__ void likelihood_kernel(const float* means,const float* logs,const float* latent,const float* old_logp,int n,int* fault) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
    const float actual=rl::gaussian_logp(means+row*A,logs,latent+row*A,A);
    if(!p::finite_number(actual)||fabsf(actual-old_logp[row])>1e-4f)fail(fault,row);
}
__global__ void kl_kernel(const float* means,const float* logs,const float* latent,const float* old_logp,float* output,int n,int* fault) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
    const float delta=rl::gaussian_logp(means+row*A,logs,latent+row*A,A)-old_logp[row];
    output[row]=expf(delta)-1-delta;if(!p::finite_number(output[row]))fail(fault,row);
}
struct Stream {
    cudaStream_t value=nullptr;Stream(){CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~Stream(){if(value)cudaStreamDestroy(value);}
};
struct Timer {
    cudaEvent_t first=nullptr,last=nullptr;
    Timer(){CUDA_CHECK_THROW(cudaEventCreate(&first));CUDA_CHECK_THROW(cudaEventCreate(&last));}
    ~Timer(){if(first)cudaEventDestroy(first);if(last)cudaEventDestroy(last);}
    void start(cudaStream_t stream){CUDA_CHECK_THROW(cudaEventRecord(first,stream));}
    float stop(cudaStream_t stream){CUDA_CHECK_THROW(cudaEventRecord(last,stream));CUDA_CHECK_THROW(cudaEventSynchronize(last));float ms=0;CUDA_CHECK_THROW(cudaEventElapsedTime(&ms,first,last));return ms;}
};
template<class Type> std::vector<Type> download(const device_ptr<Type>& input,size_t count,cudaStream_t stream) {
    require(count<=input.size(),"Readback shape exceeded");std::vector<Type> result(count);
    if(count==0)return result;
    CUDA_CHECK_THROW(cudaMemcpyAsync(result.data(),input.get(),count*sizeof(Type),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));return result;
}
template<class Type> std::vector<uint8_t> raw_bytes(const std::vector<Type>& values) {
    std::vector<uint8_t> result(values.size()*sizeof(Type));std::memcpy(result.data(),values.data(),result.size());return result;
}
template<class Type> void restore(device_ptr<Type>& target,const std::vector<uint8_t>& bytes) {
    require(bytes.size()==target.size()*sizeof(Type),"Carried-state shape mismatch");
    CUDA_CHECK_THROW(cudaMemcpy(target.get(),bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
}
std::vector<uint8_t> saved_tasks(Buffer& b,uint32_t program,cudaStream_t stream) {
    const auto tasks=download(b.tasks,N,stream);if(!pt::repeating_program(program))return raw_bytes(tasks);
    const auto schedules=download(b.schedules,N,stream);
    if(program==pt::half_second_course_v1) {
        const auto histories=download(b.histories,N,stream);std::vector<ps::HalfSecondTask> saved(N);
        for(int i=0;i<N;++i)saved[i]={tasks[i],schedules[i],histories[i]};
        return raw_bytes(saved);
    }
    std::vector<ps::SavedTask> saved(N);
    for(int i=0;i<N;++i){saved[i].task=tasks[i];saved[i].schedule=schedules[i];}return raw_bytes(saved);
}
void restore_tasks(Buffer& b,const std::vector<uint8_t>& bytes,uint32_t program) {
    if(!pt::repeating_program(program)){restore(b.tasks,bytes);return;}
    std::vector<pt::Task> tasks(N);std::vector<ps::State> schedules(N);
    if(program==pt::half_second_course_v1) {
        require(bytes.size()==N*sizeof(ps::HalfSecondTask),"Half-second course history shape mismatch");
        std::vector<ps::HalfSecondTask> saved(N);std::memcpy(saved.data(),bytes.data(),bytes.size());
        std::vector<ps::CourseHistory> histories(N);
        for(int i=0;i<N;++i){
            tasks[i]=saved[i].task;schedules[i]=saved[i].schedule;histories[i]=saved[i].history;
            require(histories[i].next<ps::CourseHistory::intervals && histories[i].count<=ps::CourseHistory::intervals,"Invalid saved COM history");
        }
        restore(b.tasks,raw_bytes(tasks));restore(b.schedules,raw_bytes(schedules));restore(b.histories,raw_bytes(histories));return;
    }
    require(bytes.size()==N*sizeof(ps::SavedTask),"Saved repeated-request context stride mismatch");
    std::vector<ps::SavedTask> saved(N);std::memcpy(saved.data(),bytes.data(),bytes.size());
    for(int i=0;i<N;++i){tasks[i]=saved[i].task;schedules[i]=saved[i].schedule;}
    restore(b.tasks,raw_bytes(tasks));restore(b.schedules,raw_bytes(schedules));
}
void clear_fault(Buffer& b,cudaStream_t stream){const int good=INT_MAX;CUDA_CHECK_THROW(cudaMemcpyAsync(b.fault.get(),&good,sizeof(good),cudaMemcpyHostToDevice,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));}
int fault(Buffer& b,cudaStream_t stream){return download(b.fault,1,stream)[0];}
void check_cuda(){CUDA_CHECK_THROW(cudaGetLastError());}
struct Resources {
    double prior_wall=0;uint64_t bytes=0,text_remaining=0;
    std::ofstream save_times;
    const std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
    double wall()const{return prior_wall+std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
    void guard(uint64_t extra=0)const{require(wall()<pp::maximum_wall_seconds,"Aggregate1200-second cap reached");require(extra<=pp::maximum_evidence_bytes-bytes,"Aggregate4GiB evidence cap reached");}
    void work_guard()const{guard();require(wall()<pp::maximum_wall_seconds-10,"Time reserved for bounded failure/checkpoint finalization");}
    void reserve_text(){guard(8192);bytes+=8192;text_remaining=8192;}
    void text(std::ostream& stream,const std::string& value){require(value.size()<=text_remaining,"Reserved text evidence exhausted");stream<<value;stream.flush();require(bool(stream),"Text evidence write failed");text_remaining-=value.size();}
    void enable_save_timing(const fs::path& path){require(!fs::exists(path),"Save timing evidence exists");save_times.open(path);require(bool(save_times),"Save timing file open failed");text(save_times,"file,payload_bytes,checksum_atomic_write_wall_ms\n");}
    void save(const fs::path& path,const std::vector<uint8_t>& payload){
        guard(static_cast<uint64_t>(payload.size())+24);require(!fs::exists(path),"Evidence target already exists");
        const bool timed=save_times.is_open();const auto began=timed?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        cp::write_atomic(path.string(),payload);bytes+=static_cast<uint64_t>(payload.size())+24;
        if(timed){const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();
            std::ostringstream row;row<<std::setprecision(17)<<path.filename().string()<<','<<payload.size()<<','<<elapsed<<'\n';text(save_times,row.str());}
    }
};
struct Ledger {
    fs::path path,lock;uint64_t protocol=0;bool closed=false,physical_started=false;
    Resources& resources;pp::CandidateState& state;
    Ledger(const Options& o,uint64_t document,Resources& budget,pp::CandidateState& current,uint64_t resumed_checksum)
        :protocol(document),resources(budget),state(current) {
        path=fs::path(scalar(text_file(o.protocol),"ledger_path_absolute"));require(path.is_absolute(),"Ledger path must be absolute and protocol-bound");
        lock=path;lock+=".running";require(fs::create_directory(lock),"Study already running or stopped with unresolved process state");
        try {
            if(o.resume.empty())require(!fs::exists(path),"This protocol allocation was already initialized; no automatic restart");
            else {const auto bytes=cp::read(path.string());cp::Reader in{bytes};require(in.value<uint32_t>()==0x314c5050&&in.value<uint32_t>()==1,"Wrong study ledger");
                require(in.value<uint64_t>()==protocol&&in.value<uint64_t>()==source_identity(),"Study ledger source/protocol mismatch");
                const auto status=in.value<uint32_t>();require(status==2||status==4,"Study stopped/active/complete; no automatic restart");
                require(in.value<uint64_t>()==resumed_checksum,"Resume is not the last completed checkpoint");
                const auto recorded=in.value<pp::Counters>();require(recorded.completed_updates==state.counters.completed_updates&&recorded.attempted_controls==state.counters.attempted_controls,"Ledger accounting differs");
                resources.prior_wall=std::max(resources.prior_wall,in.value<double>());resources.bytes=std::max(resources.bytes,in.value<uint64_t>());
                require(in.value<uint32_t>()==1,"Unresolved physical accounting cannot resume");(void)in.value<uint64_t>();(void)in.value<uint64_t>();in.finish();resources.guard();}
            save(3,resumed_checksum,true,state.counters.attempted_controls,state.counters.attempted_substeps);
        }catch(...){std::error_code ignored;fs::remove(lock,ignored);throw;}
    }
    ~Ledger(){if(!closed){try{save(6,0,!physical_started,physical_started?pp::total_controls:state.counters.attempted_controls,physical_started?pp::total_controls*10:state.counters.attempted_substeps);}catch(...) {}}std::error_code ignored;fs::remove(lock,ignored);}
    void save(uint32_t status,uint64_t candidate,bool known,uint64_t upper_controls,uint64_t upper_steps) {
        cp::Writer out;out.value(uint32_t{0x314c5050});out.value(uint32_t{1});out.value(protocol);out.value(source_identity());out.value(status);out.value(candidate);out.value(state.counters);
        out.value(resources.wall());out.value(resources.bytes);out.value(static_cast<uint32_t>(known));out.value(upper_controls);out.value(upper_steps);
        require(out.data.size()+24<=resources.text_remaining,"Auxiliary ledger reserve exhausted");cp::write_atomic(path.string(),out.data);resources.text_remaining-=out.data.size()+24;
    }
    void finish(uint32_t status,uint64_t candidate,bool known,uint64_t upper_controls,uint64_t upper_steps){save(status,candidate,known,upper_controls,upper_steps);closed=true;}
};
struct HostBatch {
    std::vector<float> observation,latent,logp,value,reward,done,truncated,final_value,last_value;
    std::vector<uint64_t> attempted,substeps;std::vector<int> reason;
    std::vector<TransitionTrace> trace;
};
struct ActiveBatch {
    bool active=false,accounted=false,saved=false;int steps=0;uint32_t update=0;uint64_t serial=0;
    pp::Counters before{};
    std::vector<uint8_t> world,task;std::vector<float> actor,critic;
};
void account(const HostBatch& h,pp::Counters& counters,ActiveBatch& active) {
    if(active.accounted)return;auto next=counters;
    for(size_t i=0;i<h.attempted.size();++i){require(h.attempted[i]<=1&&h.substeps[i]<=h.attempted[i]*10,"Corrupt physical accounting row");next.attempted_controls+=h.attempted[i];next.attempted_substeps+=h.substeps[i];}
    counters=next;active.accounted=true;
}
HostBatch batch(Buffer& b,cudaStream_t stream,int steps) {
    const auto rows=static_cast<size_t>(steps)*N;HostBatch result;
    result.observation=download(b.observation,rows*D,stream);result.latent=download(b.latent,rows*A,stream);
    result.logp=download(b.logp,rows,stream);result.value=download(b.value,rows,stream);result.reward=download(b.reward,rows,stream);
    result.done=download(b.done,rows,stream);result.truncated=download(b.truncated,rows,stream);result.final_value=download(b.final_value,rows,stream);
    result.last_value=download(b.last_value,N,stream);result.attempted=download(b.attempted,rows,stream);result.substeps=download(b.substeps,rows,stream);result.reason=download(b.reason,rows,stream);
    result.trace=download(b.trace,rows,stream);return result;
}
std::vector<uint8_t> batch_payload(const HostBatch& h,uint32_t update,int steps,uint64_t serial,
        const std::vector<uint8_t>& initial_world,const std::vector<uint8_t>& initial_task,
        const std::vector<float>& actor,const std::vector<float>& critic) {
    cp::Writer out;out.value(uint32_t{0x31425450});out.value(uint32_t{2});out.value(update);out.value(steps);out.value(serial);
    out.value(source_identity());out.array(initial_world);out.array(initial_task);out.array(actor);out.array(critic);
    for(const auto* values:{&h.observation,&h.latent,&h.logp,&h.value,&h.reward,&h.done,&h.truncated,&h.final_value,&h.last_value})out.array(*values);
    out.array(h.attempted);out.array(h.substeps);out.array(h.reason);out.value(static_cast<uint32_t>(sizeof(TransitionTrace)));out.array(raw_bytes(h.trace));return out.data;
}
nn::MlpSpec actor_spec(){return {{D,32,32,A},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}};}
nn::MlpSpec critic_spec(){return {{D,32,32,1},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}};}
void save_failure(Resources& resources,const fs::path& out,Buffer& b,nn::GpuMlp& actor,nn::GpuMlp& critic,
        nn::GpuAdam& actor_adam,nn::GpuAdam& critic_adam,const pp::Counters& counters,cudaStream_t stream,uint32_t program) {
    cp::Writer data;data.value(uint32_t{0x31465050});data.value(counters);data.array(actor.get_params());data.array(critic.get_params());
    cp::Writer a,c;actor_adam.save(a);critic_adam.save(c);data.array(a.data);data.array(c.data);
    data.array(raw_bytes(download(b.worlds,N,stream)));data.array(saved_tasks(b,program,stream));
    resources.save(out/"failed-state.bin",data.data);
}
pp::CandidateState checkpoint(Buffer& b,nn::GpuMlp& actor,nn::GpuMlp& critic,nn::GpuAdam& actor_adam,nn::GpuAdam& critic_adam,
        pp::CandidateState state,Resources& resources,cudaStream_t stream,uint32_t program) {
    state.actor=actor.get_params();state.critic=critic.get_params();state.log_std=download(b.log_std,A,stream);
    cp::Writer a,c;actor_adam.save(a);critic_adam.save(c);state.actor_adam=std::move(a.data);state.critic_adam=std::move(c.data);
    state.world_state=raw_bytes(download(b.worlds,N,stream));state.task_state=saved_tasks(b,program,stream);
    // Conservatively reserve five seconds for this checkpoint's write and final
    // report. The reserve is carried once at the next process boundary.
    state.wall_seconds_used=resources.wall()+5;state.evidence_bytes_used=resources.bytes;return state;
}
void run(const Options& o,const Package& package) {
    prerequisites(o,package.checksum);cudarl::pond_math_profile::validate_host_geometry();
    require(o.initialize_only==o.resume.empty(),"Initialize candidate0 once, then resume its exact carried state after baseline assessment");
    const auto protocol_checksum=cp::checksum(whole_file(o.protocol));pp::CandidateState state;uint64_t last_checkpoint=0;
    state.parent_package_checksum=package.checksum;state.protocol_checksum=protocol_checksum;state.source_identity_checksum=source_identity();
    state.seed=default_seed;state.world_stride=static_cast<uint32_t>(sizeof(pr::World));state.task_stride=task_stride(o.study_program);state.frozen_gain=package.gain;
    if(!o.resume.empty()) {
        const auto saved=cp::read(o.resume.string());state=pp::read_candidate(saved,package.checksum,protocol_checksum,source_identity(),static_cast<uint32_t>(sizeof(pr::World)),task_stride(o.study_program));
        last_checkpoint=cp::checksum(saved);require(state.seed==default_seed,"Saved sampler seed differs from this study");
        require(!o.retention.empty()&&o.evaluator_source!=0,"Specific candidate retention receipt required before continuation");
        const auto receipt=pp::read_retention_receipt(cp::read(o.retention.string()),cp::checksum(saved),package.checksum,protocol_checksum,o.evaluator_source,state.counters.completed_updates==0);
        require(receipt.passed&&receipt.cases_passed==20&&receipt.controls==30400&&receipt.substeps==304000,
            "Genuine full inherited-skill retention is required before every optimizer chunk, including candidate0");
    }
    require(state.counters.completed_updates%8==0&&o.to_update==state.counters.completed_updates+8&&o.to_update<=update_budget(o.study_program),"Only the next authorized eight-update checkpoint chunk is permitted");
    Resources resources;resources.prior_wall=state.wall_seconds_used;resources.bytes=state.evidence_bytes_used;resources.guard();
    require(!fs::exists(o.out)&&fs::create_directories(o.out),"Unique output directory required");
    resources.reserve_text();std::ofstream metrics(o.out/"updates.csv");resources.text(metrics,"update,attempted_controls,attempted_substeps,actor_Adam,critic_Adam,mean_reward,collection_ms,physics_ms,optimization_ms,transfer_ms,approx_KL,aggregate_wall_s\n");
    if(o.study_program!=pt::curriculum_off)resources.enable_save_timing(o.out/"save-times.csv");
    Ledger ledger(o,protocol_checksum,resources,state,last_checkpoint);
    CUDA_CHECK_THROW(cudaSetDevice(0));Stream stream;Buffer b;clear_fault(b,stream.value);
    nn::GpuMlp actor(actor_spec(),MB,state.seed),critic(critic_spec(),MB,state.seed+1);
    nn::AdamConfig config;config.lr=pp::learning_rate;config.max_grad_norm=pp::gradient_clip;
    nn::GpuAdam actor_adam(pp::actor_weights,config),critic_adam(pp::critic_weights,config);
    std::vector<float> logs(A,pp::fixed_log_std());b.log_std.copy_from_host(logs.data(),logs.size());
    if(o.resume.empty()) {
        actor.set_params(package.weights);initialize_kernel<<<(N+31)/32,32,0,stream.value>>>(b.worlds.get(),b.tasks.get(),b.schedules.get(),b.histories.get(),state.seed,o.study_program,b.fault.get());check_cuda();
        if(fault(b,stream.value)!=INT_MAX){save_failure(resources,o.out,b,actor,critic,actor_adam,critic_adam,state.counters,stream.value,o.study_program);
            std::ofstream failed(o.out/"result.json");resources.text(failed,"{\"passed\":false,\"reason\":\"initial_GPU_certificate_failed\",\"attempted_controls\":0,\"attempted_substeps\":0,\"optimizer_calls\":0}\n");
            throw std::runtime_error("Initial GPU world certificate failed before collection");}
        state=checkpoint(b,actor,critic,actor_adam,critic_adam,state,resources,stream.value,o.study_program);
        auto initial=pp::write_candidate(state);state.evidence_bytes_used=resources.bytes+initial.size()+24;initial=pp::write_candidate(state);resources.save(o.out/"candidate-0.bin",initial);
        std::ofstream initialized(o.out/"result.json");resources.text(initialized,"{\"initialized\":true,\"baseline_assessment_required\":true,\"physical_controls\":0,\"physical_substeps\":0,\"optimizer_calls\":0}\n");
        ledger.finish(2,cp::checksum(initial),true,0,0);
        return;
    } else {
        actor.set_params(state.actor);critic.set_params(state.critic);cp::Reader a{state.actor_adam},c{state.critic_adam};actor_adam.load(a);critic_adam.load(c);a.finish();c.finish();
        restore(b.worlds,state.world_state);restore_tasks(b,state.task_state,o.study_program);
    }
    const int physical_threads=o.study_program>=pt::course_direction_v1?8:32;
    bool success=false,accounting_known=true;std::string reason="chunk_complete_retention_required";ActiveBatch active;
    uint64_t control_upper=state.counters.attempted_controls,substep_upper=state.counters.attempted_substeps;
    try {
        for(uint32_t update=static_cast<uint32_t>(state.counters.completed_updates)+1;update<=o.to_update;++update) {
            resources.guard(32ull*1024*1024);require(state.counters.attempted_controls+Rows<=pp::total_controls,"Physical training budget exhausted");clear_fault(b,stream.value);
            const auto start_world=raw_bytes(download(b.worlds,N,stream.value)),start_task=saved_tasks(b,o.study_program,stream.value);
            const auto start_actor=actor.get_params(),start_critic=critic.get_params();const auto first_serial=state.sampler_decision_serial;
            active={};active.update=update;active.serial=first_serial;active.before=state.counters;active.world=start_world;active.task=start_task;active.actor=start_actor;active.critic=start_critic;
            for(auto* values:{&b.observation,&b.latent,&b.logp,&b.value,&b.reward,&b.done,&b.truncated,&b.final_value,&b.last_value})CUDA_CHECK_THROW(cudaMemsetAsync(values->get(),0,values->size_bytes(),stream.value));
            CUDA_CHECK_THROW(cudaMemsetAsync(b.attempted.get(),0,b.attempted.size_bytes(),stream.value));CUDA_CHECK_THROW(cudaMemsetAsync(b.substeps.get(),0,b.substeps.size_bytes(),stream.value));CUDA_CHECK_THROW(cudaMemsetAsync(b.reason.get(),0,b.reason.size_bytes(),stream.value));CUDA_CHECK_THROW(cudaMemsetAsync(b.trace.get(),0,b.trace.size_bytes(),stream.value));
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));active.active=true;
            Timer collection,physics;collection.start(stream.value);float physics_ms=0;int steps=0;bool collection_failed=false;
            for(int t=0;t<T;++t) {
                resources.work_guard();float* observation=b.observation.get()+t*N*D;
                observe_kernel<<<(N+31)/32,32,0,stream.value>>>(b.worlds.get(),b.tasks.get(),b.schedules.get(),observation,b.fault.get(),o.study_program,state.seed,true);check_cuda();
                if(fault(b,stream.value)!=INT_MAX){collection_failed=true;steps=t;break;}
                actor.forward(observation,N,stream.value);critic.forward(observation,N,stream.value);
                CUDA_CHECK_THROW(cudaMemcpyAsync(b.value.get()+t*N,critic.output(),N*sizeof(float),cudaMemcpyDeviceToDevice,stream.value));
                rl::sample_gaussian(actor.output(),b.log_std.get(),N,A,state.seed,static_cast<uint32_t>(state.sampler_decision_serial++),b.latent.get()+t*N*A,b.logp.get()+t*N,stream.value);
                physics.start(stream.value);
                active.steps=t+1;ledger.physical_started=true;
                step_kernel<<<(N+physical_threads-1)/physical_threads,physical_threads,0,stream.value>>>(b.worlds.get(),b.tasks.get(),b.schedules.get(),b.histories.get(),b.latent.get()+t*N*A,state.frozen_gain,
                    b.reward.get()+t*N,b.done.get()+t*N,b.truncated.get()+t*N,b.final_observation.get(),b.attempted.get()+t*N,b.substeps.get()+t*N,b.reason.get()+t*N,b.trace.get()+t*N,b.fault.get(),o.study_program,state.seed);check_cuda();
                physics_ms+=physics.stop(stream.value);steps=t+1;
                if(fault(b,stream.value)!=INT_MAX){collection_failed=true;break;}
                critic.forward(b.final_observation.get(),N,stream.value);
                CUDA_CHECK_THROW(cudaMemcpyAsync(b.final_value.get()+t*N,critic.output(),N*sizeof(float),cudaMemcpyDeviceToDevice,stream.value));
                reset_terminal_kernel<<<(N+31)/32,32,0,stream.value>>>(b.worlds.get(),b.tasks.get(),b.schedules.get(),b.histories.get(),b.done.get()+t*N,state.seed,o.study_program,b.fault.get());check_cuda();
                if(fault(b,stream.value)!=INT_MAX){collection_failed=true;break;}
            }
            if(!collection_failed) {
                observe_kernel<<<(N+31)/32,32,0,stream.value>>>(b.worlds.get(),b.tasks.get(),b.schedules.get(),b.last_observation.get(),b.fault.get(),o.study_program,state.seed,false);check_cuda();
                critic.forward(b.last_observation.get(),N,stream.value);CUDA_CHECK_THROW(cudaMemcpyAsync(b.last_value.get(),critic.output(),N*sizeof(float),cudaMemcpyDeviceToDevice,stream.value));
                audit_kernel<<<(Rows+255)/256,256,0,stream.value>>>(b.observation.get(),b.latent.get(),b.logp.get(),b.value.get(),b.reward.get(),b.done.get(),b.truncated.get(),b.final_value.get(),b.fault.get());check_cuda();
                finite_kernel<<<(N+255)/256,256,0,stream.value>>>(b.last_value.get(),N,b.fault.get());check_cuda();
                collection_failed=fault(b,stream.value)!=INT_MAX;
            }
            const float collection_ms=collection.stop(stream.value);const auto transfer_started=std::chrono::steady_clock::now();const auto h=batch(b,stream.value,steps);
            double mean_reward=0;for(float value:h.reward)mean_reward+=value;
            if(!h.reward.empty())mean_reward/=static_cast<double>(h.reward.size());
            const double transfer_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-transfer_started).count();
            resources.save(o.out/("batch-"+std::to_string(update)+".bin"),batch_payload(h,update,steps,first_serial,start_world,start_task,start_actor,start_critic));
            active.saved=true;account(h,state.counters,active);
            require(!collection_failed,"Collection rejected before GAE/Adam; preserve failed-state and all attempts");
            require(steps==T,"Incomplete rollout");for(size_t i=0;i<h.attempted.size();++i)require(h.attempted[i]==1&&h.substeps[i]==10,"Batch physical accounting mismatch");
            // Check the entire batch's actual unscaled latent likelihood under
            // unchanged collection weights before ANY optimizer call.
            for(int first=0;first<Rows;first+=MB){actor.forward(b.observation.get()+first*D,MB,stream.value);
                likelihood_kernel<<<(MB+255)/256,256,0,stream.value>>>(actor.output(),b.log_std.get(),b.latent.get()+first*A,b.logp.get()+first,MB,b.fault.get());check_cuda();}
            require(fault(b,stream.value)==INT_MAX,"Stored unscaled latent likelihood failed before GAE/Adam");
            rl::compute_gae(T,N,b.reward.get(),b.value.get(),b.done.get(),b.truncated.get(),b.final_value.get(),b.last_value.get(),pp::gamma,pp::lambda,b.advantages.get(),b.returns.get(),stream.value);
            finite_kernel<<<(Rows+255)/256,256,0,stream.value>>>(b.advantages.get(),Rows,b.fault.get());finite_kernel<<<(Rows+255)/256,256,0,stream.value>>>(b.returns.get(),Rows,b.fault.get());check_cuda();
            require(fault(b,stream.value)==INT_MAX,"Invalid GAE rejected before Adam");rl::mean_std(b.advantages.get(),Rows,b.adv_stats.get(),stream.value);
            Timer optimize;optimize.start(stream.value);double last_kl=0;
            for(uint32_t epoch=0;epoch<pp::epochs;++epoch) {
                const auto permutation=rl::make_permutation(Rows,state.seed^(update*1664525u+epoch*1013904223u));
                for(int first=0;first<Rows;first+=MB) {
                    resources.work_guard();
                    rl::gather_rows(permutation,first,MB,D,b.observation.get(),b.mb_observation.get(),stream.value);
                    rl::gather_rows(permutation,first,MB,A,b.latent.get(),b.mb_latent.get(),stream.value);
                    rl::gather_rows(permutation,first,MB,1,b.logp.get(),b.mb_logp.get(),stream.value);
                    rl::gather_rows(permutation,first,MB,1,b.advantages.get(),b.mb_advantage.get(),stream.value);
                    rl::gather_rows(permutation,first,MB,1,b.returns.get(),b.mb_return.get(),stream.value);
                    actor.forward(b.mb_observation.get(),MB,stream.value);critic.forward(b.mb_observation.get(),MB,stream.value);
                    CUDA_CHECK_THROW(cudaMemsetAsync(b.stats.get(),0,b.stats.size_bytes(),stream.value));
                    rl::gaussian_ppo_loss_grad(actor.output(),b.log_std.get(),critic.output(),MB,A,b.mb_latent.get(),b.mb_logp.get(),b.mb_advantage.get(),b.mb_return.get(),b.adv_stats.get(),
                        {pp::clip,0,.5f},b.dmean.get(),b.dlog_std.get(),b.dvalue.get(),b.row_stats.get(),b.stats.get(),stream.value);
                    const auto stats=download(b.stats,rl::kGaussianStats,stream.value);for(float value:stats)require(std::isfinite(value),"Nonfinite PPO statistics");
                    last_kl=stats[3]/MB;require(last_kl<=maximum_approx_kl,"Predeclared KL stop; partial update is not a completed candidate");
                    actor.zero_grad(stream.value);critic.zero_grad(stream.value);
                    actor.backward(b.dmean.get(),MB,nullptr,stream.value);critic.backward(b.dvalue.get(),MB,nullptr,stream.value);
                    finite_kernel<<<(pp::actor_weights+255)/256,256,0,stream.value>>>(actor.grads(),pp::actor_weights,b.fault.get());
                    finite_kernel<<<(pp::critic_weights+255)/256,256,0,stream.value>>>(critic.grads(),pp::critic_weights,b.fault.get());check_cuda();
                    require(fault(b,stream.value)==INT_MAX,"Nonfinite gradient rejected before Adam");
                    actor_adam.step(actor.params(),actor.grads(),stream.value);++state.counters.actor_adam_calls;
                    critic_adam.step(critic.params(),critic.grads(),stream.value);++state.counters.critic_adam_calls;
                    finite_kernel<<<(pp::actor_weights+255)/256,256,0,stream.value>>>(actor.params(),pp::actor_weights,b.fault.get());
                    finite_kernel<<<(pp::critic_weights+255)/256,256,0,stream.value>>>(critic.params(),pp::critic_weights,b.fault.get());check_cuda();
                    require(fault(b,stream.value)==INT_MAX,"Nonfinite updated parameters; candidate rejected");
                }
            }
            // Audit final actor against this whole rollout after the last Adam
            // call as well; a final-step KL jump cannot become a good candidate.
            for(int first=0;first<Rows;first+=MB){actor.forward(b.observation.get()+first*D,MB,stream.value);
                kl_kernel<<<(MB+255)/256,256,0,stream.value>>>(actor.output(),b.log_std.get(),b.latent.get()+first*A,b.logp.get()+first,b.kl.get()+first,MB,b.fault.get());check_cuda();}
            require(fault(b,stream.value)==INT_MAX,"Nonfinite final rollout KL");rl::mean_std(b.kl.get(),Rows,b.adv_stats.get(),stream.value);
            last_kl=download(b.adv_stats,2,stream.value)[0];require(std::isfinite(last_kl)&&last_kl<=maximum_approx_kl,"Final whole-rollout approximate KL stop");
            const float optimization_ms=optimize.stop(stream.value);state.counters.completed_updates=update;
            state=checkpoint(b,actor,critic,actor_adam,critic_adam,state,resources,stream.value,o.study_program);
            auto payload=pp::write_candidate(state);state.evidence_bytes_used=resources.bytes+payload.size()+24;payload=pp::write_candidate(state);
            resources.save(o.out/("candidate-"+std::to_string(update)+".bin"),payload);
            last_checkpoint=cp::checksum(payload);active.active=false;
            std::ostringstream row;row<<update<<','<<state.counters.attempted_controls<<','<<state.counters.attempted_substeps<<','<<state.counters.actor_adam_calls<<','<<state.counters.critic_adam_calls
                <<','<<mean_reward<<','<<collection_ms<<','<<physics_ms<<','<<optimization_ms<<','<<transfer_ms<<','<<last_kl<<','<<resources.wall()<<'\n';resources.text(metrics,row.str());
        }
        require(sha256(whole_file(o.policy))==pp::parent_package_sha256,"Original saved motor changed");success=true;
    } catch(const std::exception& error) {
        reason=error.what();
        if(active.active){
            try {
                const auto partial=batch(b,stream.value,active.steps);account(partial,state.counters,active);
                if(!active.saved){resources.save(o.out/("partial-batch-"+std::to_string(active.update)+".bin"),batch_payload(partial,active.update,active.steps,active.serial,active.world,active.task,active.actor,active.critic));active.saved=true;}
            }catch(const std::exception& salvage){accounting_known=active.accounted;reason+="; partial batch readback: "+std::string(salvage.what());}
            control_upper=active.before.attempted_controls+static_cast<uint64_t>(active.steps)*N;
            substep_upper=active.before.attempted_substeps+static_cast<uint64_t>(active.steps)*N*10;
        }
        try{save_failure(resources,o.out,b,actor,critic,actor_adam,critic_adam,state.counters,stream.value,o.study_program);}catch(const std::exception& dump){reason+="; failure dump: "+std::string(dump.what());}
    }
    if(accounting_known){control_upper=state.counters.attempted_controls;substep_upper=state.counters.attempted_substeps;}
    std::ostringstream result;result<<std::setprecision(17)<<"{\"passed\":"<<(success?"true":"false")<<",\"reason\":\""<<json_escape(reason)
        <<"\",\"completed_updates\":"<<state.counters.completed_updates<<",\"attempted_controls\":"<<state.counters.attempted_controls
        <<",\"attempted_substeps\":"<<state.counters.attempted_substeps<<",\"actor_Adam_calls\":"<<state.counters.actor_adam_calls<<",\"critic_Adam_calls\":"<<state.counters.critic_adam_calls
        <<",\"physical_accounting_readback_complete\":"<<(accounting_known?"true":"false")<<",\"attempted_controls_upper_bound\":"<<control_upper<<",\"attempted_substeps_upper_bound\":"<<substep_upper
        <<",\"aggregate_wall_seconds\":"<<resources.wall()<<",\"aggregate_evidence_bytes_upper_bound\":"<<resources.bytes<<",\"per_chunk_text_reserve_bytes\":8192,\"retention_required\":true,\"navigation_qualified\":false}\n";
    std::ofstream result_file(o.out/"result.json");resources.text(result_file,result.str());
    ledger.finish(success?(state.counters.completed_updates==update_budget(o.study_program)?5u:4u):6u,last_checkpoint,accounting_known,control_upper,substep_upper);
    require(success,reason);
}
}
int main(int argc,char** argv) {
    try {
        const auto o=options(argc,argv);
        if(o.describe){std::cout<<"{\"protocol_id\":\""<<study_id(o.study_program)<<"\",\"source_sha256\":\""<<ph::source_sha256()<<"\",\"source_identity_fnv1a64\":\""<<source_identity()<<"\",\"runtime\":"<<pp::runtime_version<<",\"query_arithmetic_version\":"<<pp::query_arithmetic_version<<",\"observation_schema\":"<<pp::observation_schema<<",\"task_version\":"<<pt::version<<",\"retention_receipt_schema\":"<<pp::retention_schema
            <<",\"study_program\":\""<<program_name(o.study_program)<<"\",\"study_program_id\":"<<o.study_program<<",\"physical_kernel_block_threads\":"<<(o.study_program>=pt::course_direction_v1?8:32)<<",\"turn_curriculum_program\":\""<<(o.study_program==pt::paired_gentle_moderate_v1?"paired-gentle-moderate-v1":"off")<<"\",\"qualified_physics_source_sha256\":\""<<ph::physics_source_sha256()<<"\",\"world_stride\":"<<sizeof(pr::World)<<",\"task_stride\":"<<task_stride(o.study_program)<<",\"base_task_stride\":"<<sizeof(pt::Task)<<",\"schedule_stride\":"<<sizeof(ps::State)<<",\"trace_stride\":"<<sizeof(TransitionTrace)<<",\"trace_heading_offset\":"<<offsetof(TransitionTrace,heading_x)<<",\"trace_serial_offset\":"<<offsetof(TransitionTrace,request_serial)<<",\"trace_counters_offset\":"<<offsetof(TransitionTrace,onsets)<<",\"course_history_stride\":"<<(o.study_program==pt::half_second_course_v1?sizeof(ps::CourseHistory):0)<<",\"max_updates\":"<<update_budget(o.study_program)<<",\"optimizer_calls\":0,\"physical_controls\":0}\n";return 0;}
        const auto package=load_package(o.policy);
        if(o.preflight){if(!o.protocol.empty())prerequisites(o,package.checksum);cudarl::pond_math_profile::validate_host_geometry();uint64_t visits=0;uint32_t observation_checks=0;
            for(uint32_t env=0;env<32;++env){pr::World world;pt::Task task;
                require(pt::initialize(world,task,env,0,default_seed,o.study_program),"Pure initial task/certificate failed");visits+=world.inspection.visits;
                float original[D]{},moving[D]{};const auto& physical=world.motor.physical;
                require(po::observe(task.request,physical.body,physical.held_targets,world.motor.reference_rate,physical.last_actual_torques,original)&&pt::observe(task,world,moving),"Pure task observation failed");
                for(int j=0;j<D;++j){require(original[j]==moving[j],"Initial heading-only observation changed inherited input");++observation_checks;}
                const auto controls=world.motor.physical.control_steps,steps=world.motor.attempted_substeps;
                task.episode_controls=40;task.request.controls=40;require(pt::observe(task,world,moving),"Persistent request preparation failed");
                require(world.motor.physical.control_steps==controls&&world.motor.attempted_substeps==steps&&controls==0&&steps==0,"Pure task preparation advanced physics");
            }
            require(!fs::exists(o.out)&&fs::create_directories(o.out),"Unique preflight output required");
            std::ofstream out(o.out/"preflight.json");out<<"{\"passed\":true,\"protocol_id\":\""<<study_id(o.study_program)<<"\",\"parent_package_unchanged\":true,\"source_sha256\":\""<<ph::source_sha256()
                <<"\",\"actor_weights\":2532,\"critic_weights\":2433,\"prepared_worlds\":32,\"pure_geometry_visits\":"<<visits<<",\"observation_checks\":"<<observation_checks
                <<",\"study_program\":\""<<program_name(o.study_program)<<"\",\"study_program_id\":"<<o.study_program<<",\"physical_kernel_block_threads\":"<<(o.study_program>=pt::course_direction_v1?8:32)<<",\"training_control_cap\":"<<update_budget(o.study_program)*pp::controls_per_update<<",\"Adam_call_cap\":"<<2*update_budget(o.study_program)*pp::adam_calls_per_update<<",\"physics_prerequisites_checked\":"<<(!o.protocol.empty()?"true":"false")<<",\"physics_controls\":0,\"physics_substeps\":0,\"optimizer_calls\":0,\"CUDA_calls\":0}\n";return 0;}
        run(o,package);return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
