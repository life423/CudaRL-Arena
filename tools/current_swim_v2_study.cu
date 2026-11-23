// Opt-in versioned current study. No entry runs training implicitly.
#include "motor/joint_motor.h"
namespace current_study_v2 {
void calm_failure(const cudarl::joint_motor::World&,const cudarl::physics::BodyState&,
                  const cudarl::physics::BodyState&,int,int);
void calm_step(const cudarl::joint_motor::World&,const cudarl::physics::BodyState&,int,int);
}
#define CUDARL_PHYSICAL_FAILURE_HOOK current_study_v2::calm_failure
#define CUDARL_PHYSICAL_STEP_HOOK current_study_v2::calm_step
#define CUDARL_REST_HOLD_MEASURED_PHASE_B 1
#define CUDARL_REST_HOLD_MEASURED_BASELINE 1
#define CUDARL_REST_HOLD_RESET_STUDY 1
#include "rest_hold_study.cu"
#include "current_swim_math.h"
#include "hold_learned_amplitude_math.h"
#include "motor/current_task_v2.h"
#include <cstddef>
#include <set>

namespace current_study_v2 {
namespace task=cudarl::current_task_v2;namespace motor=cudarl::current_motor_v2;
namespace policy=cudarl::current_swim_policy_v1;
constexpr int maximum_worlds=16384,episode_controls=160,head_parameters=16;
constexpr int supported_sizes[4]={2048,4096,8192,16384};
constexpr uint32_t profile_seed=0x57c0a199u;
bool supported_world_count(int n) {for(int x:supported_sizes)if(n==x)return true;return false;}
constexpr int maximum_updates=32,maximum_adam_steps=512,maximum_epochs=4;
constexpr int calm_controls=10720,current_suite_controls=2560;
constexpr int gate_controls=3760,gate_substeps=37600;
constexpr int baseline_controls=gate_controls+10720+2560,profile_controls=2949120;
constexpr int controls_cap=89571440,substeps_cap=controls_cap*10;
constexpr double seconds_cap=900;
constexpr uint32_t calibration_seed=0x36b91a57u,training_seed=0x84e62c13u;
constexpr uint32_t policy_magic=0x32575343,protocol_magic=0x32505343,batch_magic=0x32425343;
constexpr uint32_t scale_magic=0x32535343,release_magic=0x32525343;
constexpr uint64_t gib=1024ull*1024*1024,initial_disk_free=52ull*gib,working_disk_reserve=8ull*gib;
uint64_t study_protocol=0;
uint64_t selected_size_binding=0;
fs::path study_root;
fs::path failure_directory;
bool failure_saved=false;
float saved_hold_gain=0;
std::vector<float> saved_hold_parameters;
std::vector<uint8_t> hold_checkpoint_payload;
double prior_gate_seconds=0;
std::ofstream calm_states;
std::string calm_category;
int calm_case=-1,calm_expected=0;

void body_csv(std::ostream& out,const p::BodyState& body) {
    out<<','<<body.position.x<<','<<body.position.y<<','<<body.position.z
       <<','<<body.orientation.x<<','<<body.orientation.y<<','<<body.orientation.z<<','<<body.orientation.w;
    for(float x:body.angle)out<<','<<x;for(float x:body.velocity)out<<','<<x;out<<','<<body.work;
}
void body_csv_header(std::ostream& out) {
    out<<",position_x,position_y,position_z,qx,qy,qz,qw";
    for(int j=0;j<4;++j)out<<",angle"<<j;for(int j=0;j<10;++j)out<<",velocity"<<j;out<<",positive_work_J";
}
void calm_step(const m::World& world,const p::BodyState& before,int,int) {
    if(!calm_states.is_open())return;
    const int control=int(world.physical.control_steps);if(control==1)++calm_case;
    require(calm_case>=0&&calm_case<calm_expected,"Calm supplemental case count differs");
    if(control==1){calm_states<<calm_category<<','<<calm_case<<",0";body_csv(calm_states,before);calm_states<<'\n';}
    calm_states<<calm_category<<','<<calm_case<<','<<control;body_csv(calm_states,world.physical.body);calm_states<<'\n';
    require(bool(calm_states),"Calm supplemental evidence write failed");
}
void calm_group(const char* category,int expected) {calm_category=category;calm_case=-1;calm_expected=expected;}
void close_calm_group() {require(calm_case+1==calm_expected,"Calm supplemental cases incomplete");}

void capture_failure(const m::World& world,const p::BodyState& before,const p::BodyState& candidate,
                     int controls,int substeps,const char* file) {
    if(failure_saved)return;
    require(!failure_directory.empty()&&fs::is_directory(failure_directory),"Failure directory unavailable");
    const auto path=failure_directory/file;require(!fs::exists(path),"Failure evidence already exists");
    failure_saved=true;cp::Writer out;out.value(uint32_t{0x31465343});out.value(uint32_t{1});
    out.value(controls);out.value(substeps);out.value(uint32_t{sizeof(m::World)});
    out.value(world);out.value(before);out.value(candidate);cp::write_atomic(path.string(),out.data);
}
void calm_failure(const m::World& world,const p::BodyState& before,const p::BodyState& candidate,
                  int controls,int substeps) {capture_failure(world,before,candidate,controls,substeps,"first-calm-failure.bin");}
uint64_t file_hash(const fs::path& path) {
    std::ifstream in(path,std::ios::binary);require(bool(in),"Missing bound file");std::vector<uint8_t> bytes;
    for(char c;in.get(c);)bytes.push_back(uint8_t(c));require(in.eof(),"Bound file read failed");return cp::checksum(bytes);
}
void load_holding_parent(const fs::path& path) {
    hold_checkpoint_payload=cp::read(path.string());cp::Reader in{hold_checkpoint_payload};
    require(in.value<uint32_t>()==0x314d4148&&in.value<uint32_t>()==1,"Wrong learned holding parent");
    const auto parent_protocol=in.value<uint64_t>();require(parent_protocol!=0,"Holding protocol absent");
    require(in.value<uint64_t>()==cp::checksum(parent_checkpoint_payload),"Holding/swimming parents differ");
    for(uint32_t expected:{6u,6u,3u,4u,41u,4u,336u})require(in.value<uint32_t>()==expected,"Holding parent contract differs");
    require(in.value<int>()==7&&in.value<int>()==112,"Holding parent is not the qualified seven-update policy");
    require(in.array<float>(2532)==resume_state->actor.parameters,"Holding backbone differs from migrated336");
    saved_hold_parameters=in.array<float>(2);require(saved_hold_parameters.size()==2,"Holding parameters absent");
    saved_hold_gain=in.value<float>();const float sigma=in.value<float>();const auto adam=in.array<uint8_t>();in.finish();
    require(saved_hold_gain==cudarl::learned_amplitude::sigmoid(saved_hold_parameters[0])&&
            sigma==cudarl::learned_amplitude::standard_deviation(saved_hold_parameters[1]),"Holding parameter mapping differs");
    cp::Reader optimizer{adam};const auto config=optimizer.value<nn::AdamConfig>();
    require(optimizer.value<int>()==112&&config.lr==.05f,"Holding optimizer history differs");
    const auto first=optimizer.array<float>(2),second=optimizer.array<float>(2);optimizer.finish();
    require(first.size()==2&&second.size()==2&&saved_hold_gain>0&&saved_hold_gain<1,"Holding parent incomplete");
    finite(saved_hold_parameters);finite(first);finite(second);
}
void configure(const fs::path& root,bool copied) {
    study_root=root;selected_size_binding=0;
    configure_append(root,copied);control_cap=controls_cap;substep_cap=substeps_cap;active_wall_cap=seconds_cap;
    require(resume_state->actor.parameters.size()==2532&&original40.actor.update==336,"Expected preserved336 body");
    const auto hold=copied?root/"holding-parent.bin":
      fs::absolute("assets/ppo-four-axis-checkpoints/learned-amplitude-2026-11-22-qualified/qualified-policy-update-7.bin");
    load_holding_parent(hold);
}
struct Case {int pose=0;float current_x=0,current_z=0;int controls=160;};
CUDARL_HD inline Case training_case(int id) {
    Case result;result.pose=id/8?3:0;const float magnitude=(id/4)%2?.002f:.001f;
    const int direction=id%4;if(direction<2)result.current_x=direction==0?magnitude:-magnitude;
    else result.current_z=direction==2?magnitude:-magnitude;return result;
}
std::vector<Case> training_cases() {std::vector<Case> out;for(int i=0;i<16;++i)out.push_back(training_case(i));return out;}
std::vector<Case> withheld_cases() {
    std::vector<Case> out;for(int pose:{1,2})for(float magnitude:{.0015f,.0025f})for(int direction=0;direction<4;++direction) {
        const float component=magnitude/sqrtf(2.f);
        out.push_back({pose,direction/2?-component:component,direction%2?-component:component});
    }return out;
}
std::vector<Case> endurance_cases() {return {{0,-.002f,0,1280},{3,0,.002f,1280}};}
std::vector<uint8_t> protocol_bytes(const fs::path& root) {
    cp::Writer out;out.value(protocol_magic);out.value(uint32_t{2});out.value(file_hash(root/"approved-protocol.md"));
    out.value(cp::checksum(parent_checkpoint_payload));out.value(cp::checksum(hold_checkpoint_payload));
    for(const auto& file:{root/"gate-frozen-inputs.json",root/"physics-gate"/"gate-passed.bin",root/"independent-gate-verification.json"})
        out.value(fs::is_regular_file(file)?file_hash(file):uint64_t{0});
    for(int value:{6,2,3,5,44,41,4,16,336,7,maximum_worlds,episode_controls,maximum_updates,maximum_adam_steps,
                  calm_controls,current_suite_controls,baseline_controls,profile_controls,controls_cap,substeps_cap,900})out.value(value);
    for(int n:supported_sizes)out.value(n);out.value(profile_seed);out.value(initial_disk_free);out.value(working_disk_reserve);
    for(float value:{.5f,.5f,.002f,.0025f,.01f,.5f,.2f,.01f,8.f})out.value(value);
    out.value(calibration_seed);out.value(training_seed);
    for(const auto& cases:{training_cases(),withheld_cases(),endurance_cases()})
        for(const auto& c:cases){out.value(c.pose);out.value(c.current_x);out.value(c.current_z);out.value(c.controls);}
    const std::string method="frozen41body plus sixteen flow-conditioned trim/phase coefficients .5*tanh(u); u Gaussian fixedsigma.5 oncewhole8sepisode; LOO same16case; fixedglobalcalibrationRMS; nativePPOonceepisode; calmexactpassthrough; no currenthold or adoption";
    out.array(std::vector<uint8_t>(method.begin(),method.end()));return out.data;
}
void validate_protocol(const fs::path& root) {
    const auto bytes=protocol_bytes(root);require(cp::read((root/"protocol.bin").string())==bytes,"Frozen current protocol differs");
    study_protocol=cp::checksum(bytes);
}
void validate_physics_gate(const fs::path& root) {
    const auto gate=root/"physics-gate",marker=gate/"gate-passed.bin";
    const auto bytes=cp::read(marker.string());cp::Reader in{bytes};
    require(in.value<uint32_t>()==0x32474343&&in.value<uint32_t>()==2&&in.value<uint32_t>()==1,"Physics gate release missing or failed");
    require(in.value<uint64_t>()==gate_controls&&in.value<uint64_t>()==gate_substeps,"Physical gate counters differ");
    const auto gate_protocol=cp::read((gate/"protocol.bin").string());
    require(in.value<uint64_t>()==cp::checksum(gate_protocol),"Physics gate protocol changed");in.finish();
    require(file_hash(gate/"approved-protocol.md")==file_hash(root/"approved-protocol.md"),"Gate and learner approved protocols differ");
    std::ifstream active(gate/"run-started"/"active-seconds.txt");require(bool(active),"Physical gate active-time receipt missing");
    active>>prior_gate_seconds;require(bool(active)&&std::isfinite(prior_gate_seconds)&&prior_gate_seconds>0&&prior_gate_seconds<seconds_cap,"Invalid physical gate active time");
}
struct EpisodeCounts {
    unsigned long long controls=0,substeps=0;
    double reward_sum=0;
    int failure_code=0;
    p::BodyState first_before{},rejected_candidate{};
};
struct Data {
    int n=0,trace_rows=0,minibatch=0;
    std::unique_ptr<Stream> stream;
    std::unique_ptr<nn::GpuMlp> backbone;
    std::unique_ptr<nn::GpuAdam> optimizer;
    device_ptr<float> means{head_parameters},gradient{head_parameters};
    device_ptr<m::World> worlds;device_ptr<m::ForwardTask> tasks;
    device_ptr<EpisodeCounts> counts;device_ptr<int> fault{1};
    device_ptr<float> latent,old_logp;
    device_ptr<float> observations,flow,base,joint_latent;
    device_ptr<float> reward,metrics,pre_state,post_state;
    device_ptr<float> advantages,mb_latent,mb_oldlogp,mb_advantage;
    device_ptr<float> row_gradient,row_stats,stats{5};
    unsigned long long collected_controls=0,collected_substeps=0;
    int completed=0;
    Data(int worlds_count,bool training):n(worlds_count),trace_rows(n*episode_controls),minibatch(n/4),
      stream(std::make_unique<Stream>()),backbone(std::make_unique<nn::GpuMlp>(spec(4),n,seed)),
      worlds(n),tasks(n),counts(n),latent(n*head_parameters),old_logp(n),
      observations(trace_rows*41),flow(trace_rows*3),base(trace_rows*4),joint_latent(trace_rows*4),
      reward(trace_rows),metrics(trace_rows*16),pre_state(trace_rows*33),post_state(trace_rows*23),
      advantages(n),mb_latent(minibatch*head_parameters),mb_oldlogp(minibatch),mb_advantage(minibatch),
      row_gradient(minibatch*head_parameters),row_stats(minibatch*5) {
        require(supported_world_count(n),"Unsupported v2 world count");
        backbone->set_params(resume_state->actor.parameters);const std::vector<float> zero(head_parameters,0);
        means.copy_from_host(zero.data(),zero.size());
        CUDA_CHECK_THROW(cudaMemsetAsync(counts.get(),0,counts.size_bytes(),stream->value));
        CUDA_CHECK_THROW(cudaMemsetAsync(fault.get(),0,sizeof(int),stream->value));
        if(training){auto config=initial_optimizer_config(true);config.lr=.01f;
            optimizer=std::make_unique<nn::GpuAdam>(head_parameters,config);require(optimizer->steps()==0,"Current Adam is not fresh");}
    }
};
__global__ void initialize_episodes(int n,m::World* worlds,m::ForwardTask* tasks,const float* means,float* latent,
                                   float* logp,int* fault,uint32_t action_seed,uint32_t counter) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||*fault)return;
    for(int j=0;j<head_parameters;++j)latent[i*head_parameters+j]=means[j]+policy::fixed_sigma*
      cudarl::rng::normal(action_seed,counter,uint32_t(i*head_parameters+j));
    logp[i]=policy::log_probability(means,latent+i*head_parameters);
    if(!p::finite_number(logp[i])){atomicCAS(fault,0,1);return;}
    const auto c=training_case(i%16);
    if(!task::initialize(c.pose,{c.current_x,0,c.current_z},worlds[i],tasks[i]))atomicCAS(fault,0,2);
}
__global__ void observe_episodes(int n,const m::World* worlds,const m::ForwardTask* tasks,float* observation,float* flow,int* fault) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(*fault){for(int j=0;j<41;++j)observation[i*41+j]=0;for(int j=0;j<3;++j)flow[i*3+j]=0;return;}
    if(!task::observe(tasks[i],worlds[i],observation+i*41,flow+i*3))atomicCAS(fault,0,3);
}
CUDARL_HD inline void record_metrics(const m::ForwardTask& t,const m::World& world,float* values) {
    const auto z=m::measure(t,world);values[0]=z.forward_m;values[1]=z.lateral_m;values[2]=z.depth_m;
    values[3]=z.tilt_deg;values[4]=z.signed_depth_m;values[5]=z.velocity_m_s.x;values[6]=z.velocity_m_s.y;
    values[7]=z.velocity_m_s.z;values[8]=z.roll_deg;values[9]=z.pitch_deg;values[10]=z.head_yaw_deg;
    values[11]=world.physical.mean_squared_torque;values[12]=t.steady_speed_m_s;
    values[13]=t.speed_rmse;values[14]=t.path_angle_deg;values[15]=float(t.controls);
}
CUDARL_HD inline void record_pre(const m::World& world,float* values) {
    const auto& body=world.physical.body;values[0]=body.position.x;values[1]=body.position.y;values[2]=body.position.z;
    values[3]=body.orientation.x;values[4]=body.orientation.y;values[5]=body.orientation.z;values[6]=body.orientation.w;
    for(int j=0;j<4;++j)values[7+j]=body.angle[j];for(int j=0;j<10;++j)values[11+j]=body.velocity[j];
    for(int j=0;j<4;++j){values[21+j]=world.physical.held_targets[j];values[25+j]=world.physical.last_actual_torques[j];values[29+j]=world.reference_rate[j];}
}
__global__ void advance_episodes(int n,m::World* worlds,m::ForwardTask* tasks,EpisodeCounts* counts,const float* episode_latent,
        const float* flow,const float* network_output,float* base,float* joint_latent,float* reward,float* metrics,
        float* pre_state,float* post_state,int* fault,int control) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||*fault)return;
    record_pre(worlds[i],pre_state+i*33);
    float raw[4];policy::joint_latents(episode_latent+i*head_parameters,network_output+i*4,flow+i*3,raw);
    for(int j=0;j<4;++j){base[i*4+j]=network_output[i*4+j];joint_latent[i*4+j]=raw[j];}
    m::Action action;if(!m::action_from_latent(raw,action)){atomicCAS(fault,0,4);return;}
    const auto before=worlds[i].attempted_substeps;++counts[i].controls;
    p::BodyState pre=worlds[i].physical.body,candidate=pre;
    const bool advanced=motor::step_state(worlds[i],action.command,nullptr,&pre,&candidate);
    const auto attempted=worlds[i].attempted_substeps-before;counts[i].substeps+=attempted;
    if(!advanced||attempted!=10){counts[i].first_before=pre;counts[i].rejected_candidate=candidate;
        counts[i].failure_code=worlds[i].failure_code;atomicCAS(fault,0,5);return;}
    const auto interval=task::record(tasks[i],worlds[i]);
    if(!interval.accepted||interval.done!=(control==episode_controls)||interval.truncated||tasks[i].controls!=control||
       worlds[i].physical.control_steps!=uint64_t(control)) {
        counts[i].first_before=pre;counts[i].rejected_candidate=worlds[i].physical.body;atomicCAS(fault,0,6);return;
    }
    reward[i]=interval.reward;counts[i].reward_sum+=double(reward[i]);record_metrics(tasks[i],worlds[i],metrics+i*16);
    float scratch[33];record_pre(worlds[i],scratch);for(int j=0;j<21;++j)post_state[i*23+j]=scratch[j];
    post_state[i*23+21]=worlds[i].physical.body.work;post_state[i*23+22]=worlds[i].physical.mean_squared_torque;
    for(int j=0;j<16;++j)if(!p::finite_number(metrics[i*16+j])){atomicCAS(fault,0,7);return;}
}
__global__ void episode_loss(int n,const float* means,const float* latent,const float* oldp,const float* advantage,
                             float* gradient,float* stats) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto terms=policy::loss(means,latent+i*head_parameters,oldp[i],advantage[i],1.f/float(n),gradient+i*head_parameters);
    stats[i*5]=terms.pg_loss;stats[i*5+1]=terms.v_loss;stats[i*5+2]=terms.entropy;
    stats[i*5+3]=terms.approx_kl;stats[i*5+4]=terms.clipped;
}
struct Collected {
    int n=0,purpose=0,fault=0;
    std::vector<float> means,latent,logp,returns;
    std::vector<EpisodeCounts> counts;
    std::vector<float> observations,flow,base,joint_latent,reward,metrics,pre_state,post_state;
    std::vector<m::World> final_worlds;
    std::vector<m::ForwardTask> final_tasks;
    int ticks=0,passed=0;
    double maximum_logp_error=0,maximum_return_error=0,maximum_action_error=0;
    double observe_ms=0,network_ms=0,physics_ms=0,readback_seconds=0,validation_seconds=0,evidence_seconds=0,end_to_end_seconds=0;
};
void account(Data& data,Budget& budget) {
    const auto counts=download(data.counts.get(),data.n,data.stream->value);
    unsigned long long controls=0,substeps=0;for(const auto& count:counts){controls+=count.controls;substeps+=count.substeps;}
    require(controls>=data.collected_controls&&substeps>=data.collected_substeps,"Current counters moved backwards");
    budget.controls+=int(controls-data.collected_controls);budget.substeps+=int(substeps-data.collected_substeps);
    data.collected_controls=controls;data.collected_substeps=substeps;
    require(budget.controls<=budget.maximum_controls&&budget.substeps<=budget.maximum_substeps,"Current physical cap exceeded");
}
uint64_t collection_bytes(int n,int ticks) {return uint64_t(n)*ticks*125*4+uint64_t(n)*2048+1024*1024;}
uint64_t projected_study_bytes() {
    const uint64_t trace=uint64_t(profile_controls+(maximum_updates+1)*maximum_worlds*episode_controls)*125*4;
    uint64_t profile_worlds=0;for(int n:supported_sizes)profile_worlds+=uint64_t(n)*3;
    return trace+(profile_worlds+uint64_t(maximum_updates+1)*maximum_worlds)*2048+45ull*1024*1024+gib;
}
void disk_guard(const fs::path& directory,uint64_t next_bytes,bool initial=false) {
    const auto free=fs::space(directory).available;
    require(free>=next_bytes+working_disk_reserve,"Insufficient evidence disk space before collection; attempt stops");
    if(initial)require(free>=initial_disk_free&&free>=projected_study_bytes()+working_disk_reserve,"Initial GPU-study disk reserve unavailable");
}
struct StageEvents {
    std::vector<cudaEvent_t> event;
    explicit StageEvents(int ticks):event(size_t(ticks)*4) {for(auto& e:event)CUDA_CHECK_THROW(cudaEventCreate(&e));}
    ~StageEvents(){for(auto e:event)cudaEventDestroy(e);}
    void mark(int control,int stage,cudaStream_t stream){CUDA_CHECK_THROW(cudaEventRecord(event[size_t(control)*4+stage],stream));}
    void totals(int ticks,Collected& batch) {
        for(int k=0;k<ticks;++k)for(int stage=0;stage<3;++stage){float ms=0;
            CUDA_CHECK_THROW(cudaEventElapsedTime(&ms,event[size_t(k)*4+stage],event[size_t(k)*4+stage+1]));
            if(stage==0)batch.observe_ms+=ms;else if(stage==1)batch.network_ms+=ms;else batch.physics_ms+=ms;}
    }
};
template<class T>void queue_read(std::vector<T>& output,const T* input,size_t count,cudaStream_t stream) {
    output.resize(count);if(count)CUDA_CHECK_THROW(cudaMemcpyAsync(output.data(),input,count*sizeof(T),cudaMemcpyDeviceToHost,stream));
}
void snapshot(Data& data,Collected& batch,const std::vector<EpisodeCounts>& before) {
    const auto began=std::chrono::steady_clock::now();const auto stream=data.stream->value;const int n=data.n,collection_rows=batch.ticks*n;
    queue_read(batch.counts,data.counts.get(),n,stream);queue_read(batch.observations,data.observations.get(),size_t(collection_rows)*41,stream);
    queue_read(batch.flow,data.flow.get(),size_t(collection_rows)*3,stream);queue_read(batch.base,data.base.get(),size_t(collection_rows)*4,stream);
    queue_read(batch.joint_latent,data.joint_latent.get(),size_t(collection_rows)*4,stream);queue_read(batch.reward,data.reward.get(),collection_rows,stream);
    queue_read(batch.metrics,data.metrics.get(),size_t(collection_rows)*16,stream);queue_read(batch.pre_state,data.pre_state.get(),size_t(collection_rows)*33,stream);
    queue_read(batch.post_state,data.post_state.get(),size_t(collection_rows)*23,stream);queue_read(batch.final_worlds,data.worlds.get(),n,stream);
    queue_read(batch.final_tasks,data.tasks.get(),n,stream);CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    for(int i=0;i<n;++i){batch.counts[i].controls-=before[i].controls;batch.counts[i].substeps-=before[i].substeps;}
    batch.readback_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
}
void save_batch(const fs::path& directory,const char* phase,int index,const Collected& batch) {
    const auto began=std::chrono::steady_clock::now();const int n=batch.n,rows_saved=batch.ticks*n;
    const auto target=directory/(std::string(phase)+"-batch-"+std::to_string(index));
    require(fs::create_directory(target),"V2 batch identity exists");
    cp::Writer meta;meta.value(batch_magic);meta.value(uint32_t{2});meta.value(study_protocol);
    for(int value:{n,episode_controls,batch.ticks,41,3,4,16,head_parameters,batch.purpose,index,batch.fault})meta.value(value);
    meta.array(batch.means);meta.array(batch.latent);meta.array(batch.logp);meta.array(batch.returns);
    meta.value(uint32_t{sizeof(EpisodeCounts)});meta.array(batch.counts);
    meta.value(uint32_t{sizeof(m::World)});meta.array(batch.final_worlds);
    meta.value(uint32_t{sizeof(m::ForwardTask)});meta.array(batch.final_tasks);
    const auto meta_checksum=cp::checksum(meta.data);cp::write_atomic((target/"meta.bin").string(),meta.data);
    const char* names[8]={"observations","flow","base","joint-latent","reward","metrics","pre-state","post-state"};
    const int columns[8]={41,3,4,4,1,16,33,23};
    const std::vector<float>* arrays[8]={&batch.observations,&batch.flow,&batch.base,&batch.joint_latent,
      &batch.reward,&batch.metrics,&batch.pre_state,&batch.post_state};
    std::ofstream manifest(target/"manifest.json");manifest<<"{\"schema\":2,\"meta_checksum\":"<<meta_checksum
      <<",\"worlds\":"<<n<<",\"ticks\":"<<batch.ticks<<",\"purpose\":"<<batch.purpose<<",\"parts\":[";
    for(int field=0;field<8;++field) {
        require(arrays[field]->size()==size_t(rows_saved)*columns[field],"V2 batch part shape differs");
        cp::Writer part;part.value(batch_magic);part.value(uint32_t{2});part.value(field);part.value(meta_checksum);
        part.value(n);part.value(batch.ticks);part.value(columns[field]);part.array(*arrays[field]);
        require(part.data.size()<=512ull*1024*1024,"V2 part exceeds unchanged native envelope limit");
        const auto file=target/(std::string(names[field])+".bin");cp::write_atomic(file.string(),part.data);
        uint64_t header[3]{};std::ifstream check(file,std::ios::binary);check.read(reinterpret_cast<char*>(header),sizeof(header));
        require(bool(check)&&header[0]==0x31504b434c524443ull&&header[1]==part.data.size(),"V2 part envelope header differs");
        if(field)manifest<<',';manifest<<"{\"field\":"<<field<<",\"name\":"<<std::quoted(file.filename().string())
          <<",\"columns\":"<<columns[field]<<",\"payload_bytes\":"<<header[1]<<",\"payload_checksum\":"<<header[2]<<"}";
    }
    manifest<<"]}\n";manifest.close();require(bool(manifest),"V2 batch manifest write failed");
    std::ofstream csv(target/"episodes.csv");csv<<std::setprecision(12);
    csv<<"episode,case,pose,current_x_mm_s,current_z_mm_s,old_logp,reward_sum,return_8s,controls,substeps,failure_code";
    for(int j=0;j<head_parameters;++j)csv<<",latent"<<j;csv<<'\n';
    for(int i=0;i<n;++i){const auto c=training_case(i%16);const auto& count=batch.counts[i];
        csv<<i<<','<<i%16<<','<<c.pose<<','<<c.current_x*1000<<','<<c.current_z*1000<<','<<batch.logp[i]<<','<<count.reward_sum<<',';
        if(!batch.returns.empty())csv<<batch.returns[i];
        csv<<','<<count.controls<<','<<count.substeps<<','<<count.failure_code;
        for(int j=0;j<head_parameters;++j)csv<<','<<batch.latent[i*head_parameters+j];csv<<'\n';
    }
    csv.close();require(bool(csv),"V2 episode evidence write failed");
    (void)began;
}
void validate_batch(Collected& batch) {
    const auto began=std::chrono::steady_clock::now();const int n=batch.n,ticks=batch.ticks;
    require(batch.fault==0,"Unsafe current collection rejected before optimizer");
    require((batch.purpose==2&&ticks==32)||(batch.purpose!=2&&ticks==episode_controls),"Wrong V2 collection horizon");
    if(batch.purpose!=2)batch.returns.resize(n);else require(batch.returns.empty(),"Profiling data must not contain PPO returns");
    double means[head_parameters];for(int j=0;j<head_parameters;++j)means[j]=batch.means[j];
    for(int i=0;i<n;++i) {
        const auto c=training_case(i%16);const auto& count=batch.counts[i];const auto& world=batch.final_worlds[i];const auto& t=batch.final_tasks[i];
        require(count.controls==uint64_t(ticks)&&count.substeps==uint64_t(ticks*10)&&t.controls==ticks&&
          world.physical.control_steps==uint64_t(ticks)&&world.physical.substeps==uint64_t(ticks*10),"V2 whole/prefix episode count differs");
        require(world.physical.current.x==c.current_x&&world.physical.current.y==0&&world.physical.current.z==c.current_z,"Current changed during V2 collection");
        if(batch.purpose==2)require(t.outcome==m::Outcome::Running,"Profiling prefix falsely marked terminal");
        else require(t.outcome==m::Outcome::Success||t.outcome==m::Outcome::Timeout,"Full training episode is not terminal");
        double episode[head_parameters];for(int j=0;j<head_parameters;++j)episode[j]=batch.latent[i*head_parameters+j];
        batch.maximum_logp_error=std::max(batch.maximum_logp_error,std::abs(policy::log_probability(means,episode)-batch.logp[i]));
        double sum=0;for(int k=0;k<ticks;++k) {
            const int row=k*n+i;sum+=batch.reward[row];
            require(batch.observations[row*41+34]==p::clamp(1.f-float(k)/40,0,1),"V2 observation clock reset");
            float expected[4];policy::joint_latents(batch.latent.data()+i*head_parameters,batch.base.data()+row*4,batch.flow.data()+row*3,expected);
            for(int j=0;j<4;++j)batch.maximum_action_error=std::max(batch.maximum_action_error,double(std::abs(expected[j]-batch.joint_latent[row*4+j])));
        }
        batch.maximum_return_error=std::max(batch.maximum_return_error,std::abs(sum-count.reward_sum));
        require(std::abs(sum-count.reward_sum)<1e-6+1e-7*std::abs(sum),"V2 whole/prefix reward sum differs");
        if(batch.purpose!=2){batch.returns[i]=float(sum/8.);require(std::isfinite(batch.returns[i]),"Nonfinite V2 return");batch.passed+=t.outcome==m::Outcome::Success;}
    }
    require(batch.maximum_logp_error<.0001&&batch.maximum_action_error<.00002,"V2 stored probability/action mismatch");
    batch.validation_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
}
Collected collect(Data& data,Budget& budget,const fs::path& directory,const char* phase,int index,int purpose,int ticks=episode_controls) {
    const auto began=std::chrono::steady_clock::now();const int n=data.n,collection_rows=ticks*n,grid=(n+127)/128;
    require(ticks==episode_controls||(purpose==2&&ticks==32),"Invalid V2 collection horizon");
    require(budget.controls+collection_rows<=budget.maximum_controls&&budget.substeps+collection_rows*10<=budget.maximum_substeps,"No budget for V2 collection");
    disk_guard(directory,collection_bytes(n,ticks));const auto stream=data.stream->value;
    const auto before=download(data.counts.get(),n,stream);std::vector<EpisodeCounts> reset(n);
    for(int i=0;i<n;++i){reset[i].controls=before[i].controls;reset[i].substeps=before[i].substeps;}
    data.counts.copy_from_host(reset.data(),n);CUDA_CHECK_THROW(cudaMemsetAsync(data.fault.get(),0,sizeof(int),stream));
    float* pointers[8]={data.observations.get(),data.flow.get(),data.base.get(),data.joint_latent.get(),data.reward.get(),data.metrics.get(),data.pre_state.get(),data.post_state.get()};
    const int columns[8]={41,3,4,4,1,16,33,23};for(int field=0;field<8;++field)CUDA_CHECK_THROW(cudaMemsetAsync(pointers[field],0,size_t(collection_rows)*columns[field]*sizeof(float),stream));
    const uint32_t action_seed=purpose==2?profile_seed:purpose==0?calibration_seed:training_seed;
    initialize_episodes<<<grid,128,0,stream>>>(n,data.worlds.get(),data.tasks.get(),data.means.get(),data.latent.get(),data.old_logp.get(),data.fault.get(),action_seed,uint32_t(index));
    Collected batch;batch.n=n;batch.purpose=purpose;batch.means=download(data.means.get(),head_parameters,stream);
    batch.latent=download(data.latent.get(),size_t(n)*head_parameters,stream);batch.logp=download(data.old_logp.get(),n,stream);
    StageEvents timing(ticks);int fault=download(data.fault.get(),1,stream)[0];
    try {
        require(fault==0,"V2 episode initialization failed");
        for(int k=0;k<ticks;++k) {
            wall_guard();const int offset=k*n;timing.mark(k,0,stream);
            observe_episodes<<<grid,128,0,stream>>>(n,data.worlds.get(),data.tasks.get(),data.observations.get()+offset*41,data.flow.get()+offset*3,data.fault.get());
            timing.mark(k,1,stream);data.backbone->forward(data.observations.get()+offset*41,n,stream);timing.mark(k,2,stream);
            advance_episodes<<<grid,128,0,stream>>>(n,data.worlds.get(),data.tasks.get(),data.counts.get(),data.latent.get(),data.flow.get()+offset*3,
              data.backbone->output(),data.base.get()+offset*4,data.joint_latent.get()+offset*4,data.reward.get()+offset,data.metrics.get()+offset*16,
              data.pre_state.get()+offset*33,data.post_state.get()+offset*23,data.fault.get(),k+1);
            timing.mark(k,3,stream);CUDA_CHECK_THROW(cudaGetLastError());batch.ticks=k+1;
            if((k+1)%16==0) {
                fault=download(data.fault.get(),1,stream)[0];const auto live_counts=download(data.counts.get(),n,stream);
                unsigned long long controls=0,substeps=0;for(const auto& count:live_counts){controls+=count.controls;substeps+=count.substeps;}
                std::ofstream live(study_root/"status.json");live<<std::setprecision(17)<<"{\"phase\":"<<std::quoted(phase)
                  <<",\"batch\":"<<index<<",\"worlds\":"<<n<<",\"completed_updates\":"<<data.completed<<",\"tick\":"<<k+1
                  <<",\"ticks_per_episode\":"<<ticks<<",\"unix_seconds\":"<<unix_now()
                  <<",\"controls\":"<<budget.controls+controls-data.collected_controls
                  <<",\"substeps\":"<<budget.substeps+substeps-data.collected_substeps
                  <<",\"optimizer_steps\":"<<(data.optimizer?data.optimizer->steps():0)<<",\"fault\":"<<fault<<"}\n";
                if(fault)break;
            }
        }
    } catch(...) {
        batch.fault=download(data.fault.get(),1,stream)[0];account(data,budget);snapshot(data,batch,before);
        const auto save_start=std::chrono::steady_clock::now();save_batch(directory,phase,index,batch);
        batch.evidence_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-save_start).count();throw;
    }
    account(data,budget);batch.fault=download(data.fault.get(),1,stream)[0];snapshot(data,batch,before);timing.totals(batch.ticks,batch);
    // Returns are absent from performance prefixes. Complete PPO episodes receive
    // their whole8s returns only after independent host-side collection checks.
    if(batch.purpose!=2){batch.returns.resize(n);for(int i=0;i<n;++i)batch.returns[i]=float(batch.counts[i].reward_sum/8.);}
    const auto save_start=std::chrono::steady_clock::now();save_batch(directory,phase,index,batch);
    batch.evidence_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-save_start).count();
    require(batch.ticks==ticks,"Incomplete V2 collection");validate_batch(batch);
    require(data.backbone->get_params()==resume_state->actor.parameters,"Inherited body changed during V2 collection");
    batch.end_to_end_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
    const auto target=directory/(std::string(phase)+"-batch-"+std::to_string(index));
    std::ofstream report(target/"timing.json");report<<std::setprecision(17)
      <<"{\"worlds\":"<<n<<",\"ticks\":"<<ticks<<",\"purpose\":"<<purpose<<",\"sample\":"<<index
      <<",\"controls\":"<<collection_rows<<",\"observe_ms\":"<<batch.observe_ms<<",\"network_ms\":"<<batch.network_ms<<",\"physics_ms\":"<<batch.physics_ms
      <<",\"readback_wall_seconds\":"<<batch.readback_seconds<<",\"validation_seconds\":"<<batch.validation_seconds
      <<",\"evidence_seconds\":"<<batch.evidence_seconds<<",\"end_to_end_seconds\":"<<batch.end_to_end_seconds
      <<",\"optimizer_steps\":0,\"PPO_data\":"<<(purpose==2?"false":"true")<<"}\n";
    report.close();require(bool(report),"V2 pipeline timing write failed");return batch;
}
struct Scale {double value=0;double means[16]{};int counts[16]{};};
Scale fit_scale(const std::vector<float>& returns) {
    const int n=int(returns.size());require(supported_world_count(n),"Calibration size differs");Scale scale;
    for(int i=0;i<n;++i){require(std::isfinite(returns[i]),"Nonfinite calibration return");scale.means[i%16]+=returns[i];++scale.counts[i%16];}
    for(int c=0;c<16;++c){require(scale.counts[c]==n/16,"Current calibration case coverage differs");scale.means[c]/=scale.counts[c];}
    double square=0;for(int i=0;i<n;++i){const double e=returns[i]-scale.means[i%16];square+=e*e;}
    scale.value=std::sqrt(square/n);require(std::isfinite(scale.value)&&scale.value>0,"No finite nonzero calibration signal");return scale;
}
std::vector<float> leave_one_out(const std::vector<float>& returns,double scale) {
    const int collection_rows=int(returns.size());require(supported_world_count(collection_rows)&&std::isfinite(scale)&&scale>0,"Invalid current LOO input");
    double sum[16]{};int n[16]{};
    for(int i=0;i<collection_rows;++i){require(std::isfinite(returns[i]),"Nonfinite current return");sum[i%16]+=returns[i];++n[i%16];}
    std::vector<float> result(collection_rows);for(int i=0;i<collection_rows;++i){require(n[i%16]==collection_rows/16,"LOO case coverage differs");
        result[i]=float((returns[i]-(sum[i%16]-returns[i])/(n[i%16]-1))/scale);require(std::isfinite(result[i]),"Nonfinite current advantage");}
    return result;
}
std::vector<uint8_t> scale_bytes(const Scale& scale,uint64_t protocol) {
    cp::Writer out;out.value(scale_magic);out.value(uint32_t{2});out.value(protocol);out.value(scale.value);
    for(double x:scale.means)out.value(x);for(int n:scale.counts)out.value(n);return out.data;
}
Scale read_scale(const fs::path& root,uint64_t protocol,int n) {
    const auto bytes=cp::read((root/"fixed-scale.bin").string());cp::Reader in{bytes};
    require(in.value<uint32_t>()==scale_magic&&in.value<uint32_t>()==2&&in.value<uint64_t>()==protocol,"Wrong fixed current scale");
    Scale scale;scale.value=in.value<double>();for(double& x:scale.means)x=in.value<double>();for(int& count:scale.counts)count=in.value<int>();in.finish();
    require(std::isfinite(scale.value)&&scale.value>0,"Invalid fixed current scale");for(int c=0;c<16;++c)require(scale.counts[c]==n/16&&std::isfinite(scale.means[c]),"Invalid fixed scale cases");return scale;
}
void save_policy(Data& data,const fs::path& directory,uint64_t protocol,int update) {
    const auto path=directory/("current-policy-update-"+std::to_string(update)+".bin");require(!fs::exists(path),"Current policy identity exists");
    const auto means=download(data.means.get(),head_parameters,data.stream->value);cp::Writer out,adam;data.optimizer->save(adam);
    out.value(policy_magic);out.value(uint32_t{2});out.value(protocol);out.value(cp::checksum(parent_checkpoint_payload));out.value(cp::checksum(hold_checkpoint_payload));
    out.value(selected_size_binding);
    for(uint32_t value:{6u,2u,3u,5u,44u,41u,4u,16u,336u,7u})out.value(value);
    out.value(data.n);out.value(update);out.value(data.optimizer->steps());out.array(data.backbone->get_params());
    out.array(saved_hold_parameters);out.value(saved_hold_gain);out.array(means);out.value(policy::fixed_sigma);out.value(policy::coefficient_limit);out.array(adam.data);
    cp::write_atomic(path.string(),out.data);
    const auto bytes=cp::read(path.string());require(bytes==out.data,"Current checkpoint exact roundtrip failed");
    cp::Reader read_adam{adam.data};const auto config=read_adam.value<nn::AdamConfig>();const int steps=read_adam.value<int>();
    const auto first=read_adam.array<float>(head_parameters),second=read_adam.array<float>(head_parameters);read_adam.finish();
    require(config.lr==.01f&&config.beta1==.9f&&config.beta2==.999f&&config.eps==.00001f&&config.max_grad_norm==.5f&&
            steps==data.optimizer->steps()&&steps<=maximum_adam_steps&&first.size()==16&&second.size()==16,"Current Adam save contract differs");
    finite(means);finite(first);finite(second);for(float x:second)require(x>=0,"Invalid current Adam variance");
}
double optimize(Data& data,const Collected& batch,const Scale& scale,int update,const fs::path& directory) {
    const int n=data.n,mb=data.minibatch;require(batch.n==n&&batch.purpose==1&&batch.ticks==160,"Only complete fresh training episodes reach PPO");
    const auto advantage=leave_one_out(batch.returns,scale.value);const auto stream=data.stream->value;
    data.advantages.copy_from_host(advantage.data(),advantage.size());
    cp::Writer evidence;evidence.value(uint32_t{0x32415343});evidence.value(uint32_t{2});evidence.value(update);evidence.value(n);evidence.value(scale.value);
    evidence.array(batch.returns);evidence.array(advantage);cp::write_atomic((directory/("advantages-"+std::to_string(update)+".bin")).string(),evidence.data);
    const int before=data.optimizer->steps();double kl=0;
    for(int epoch=0;epoch<maximum_epochs;++epoch) {
        CUDA_CHECK_THROW(cudaMemsetAsync(data.stats.get(),0,5*sizeof(float),stream));
        const auto permutation=rl::make_permutation(n,0x1c72931u+uint32_t(update*maximum_epochs+epoch));
        for(int first=0;first<n;first+=mb) {
            wall_guard();rl::gather_rows(permutation,first,mb,head_parameters,data.latent.get(),data.mb_latent.get(),stream);
            rl::gather_rows(permutation,first,mb,1,data.old_logp.get(),data.mb_oldlogp.get(),stream);
            rl::gather_rows(permutation,first,mb,1,data.advantages.get(),data.mb_advantage.get(),stream);
            episode_loss<<<(mb+127)/128,128,0,stream>>>(mb,data.means.get(),data.mb_latent.get(),data.mb_oldlogp.get(),data.mb_advantage.get(),data.row_gradient.get(),data.row_stats.get());
            CUDA_CHECK_THROW(cudaMemsetAsync(data.gradient.get(),0,head_parameters*sizeof(float),stream));
            rl::sum_columns(data.row_gradient.get(),mb,head_parameters,data.gradient.get(),stream);
            rl::sum_columns(data.row_stats.get(),mb,5,data.stats.get(),stream);
            finite(download(data.gradient.get(),head_parameters,stream));data.optimizer->step(data.means.get(),data.gradient.get(),stream);
            finite(download(data.means.get(),head_parameters,stream));
        }
        const auto stats=download(data.stats.get(),5,stream);finite(stats);kl=stats[3]/n;if(kl>.01)break;
    }
    require(data.optimizer->steps()-before>=4&&data.optimizer->steps()-before<=16&&data.optimizer->steps()<=maximum_adam_steps,"Current optimizer step cap exceeded");
    require(data.backbone->get_params()==resume_state->actor.parameters,"Frozen body changed during current PPO");return kl;
}
void current_step(Budget& budget,m::World& world,const float* commands) {
    wall_guard();require(budget.controls<budget.maximum_controls&&budget.substeps+10<=budget.maximum_substeps,"Current CPU budget exhausted");
    ++budget.controls;const auto before_count=world.attempted_substeps;p::BodyState before=world.physical.body,candidate=before;
    const bool advanced=motor::step_state(world,commands,nullptr,&before,&candidate);
    budget.substeps+=int(world.attempted_substeps-before_count);
    if(!advanced||world.attempted_substeps-before_count!=10) {
        capture_failure(world,before,candidate,budget.controls,budget.substeps,"first-current-cpu-failure.bin");
        throw std::runtime_error("Current physical assessment stop code="+std::to_string(world.failure_code));
    }
}
struct Assessment {int passed=0,completed=0;double mean_cost=0;int window_passed=0,window_total=0;};
Assessment assess_current(const std::vector<float>& means,const std::vector<Case>& cases,const fs::path& directory,Budget& budget) {
    require(fs::create_directory(directory),"Current assessment identity exists");nn::CpuMlp<float> backbone(spec(4));backbone.params()=resume_state->actor.parameters;
    std::ofstream summary(directory/"summary.csv"),trace(directory/"controls.csv"),poses(directory/"poses.csv"),configurations(directory/"cases.csv"),
      states(directory/"states.csv"),windows(directory/"windows.csv"),initial(directory/"initial.csv");
    summary<<std::setprecision(12);trace<<std::setprecision(12);poses<<std::setprecision(12);configurations<<std::setprecision(12);
    states<<std::setprecision(12);windows<<std::setprecision(12);initial<<std::setprecision(12);
    summary<<"case,pose,current_x_mm_s,current_z_mm_s,passed,steady_mm_s,RMSE_mm_s,lateral_mm,depth_mm,tilt_deg,path_deg,work_J_per_m,forward_mm,return,controls,substeps\n";
    trace<<"case,control,time_s,forward_mm,lane_mm,depth_mm,tilt_deg,velocity_x_mm_s,velocity_y_mm_s,velocity_z_mm_s,reward";
    for(int j=0;j<4;++j)trace<<",base"<<j<<",latent"<<j<<",command"<<j;trace<<'\n';
    poses<<"category,case,control,time_s,position_x_model,position_y_model,position_z_model,quat_w,quat_x,quat_y,quat_z,angle0,angle1,angle2,angle3\n";
    configurations<<"case,pose,current_x_m_s,current_z_m_s,heading_x,heading_y,heading_z,maximum_controls\n";
    states<<"case,control";body_csv_header(states);states<<",mean_squared_torque\n";
    windows<<"case,end_s,mean_mm_s,RMSE_mm_s,passed\n";
    initial<<"case,forward_mm,lane_mm,depth_mm,tilt_deg,speed_mm_s\n";
    Assessment assessment;
    for(int id=0;id<int(cases.size());++id) {
        const auto c=cases[id];m::World world;m::ForwardTask t;
        require(task::initialize(c.pose,{c.current_x,0,c.current_z},world,t,c.controls),"Current assessment initialization failed");
        configurations<<id<<','<<c.pose<<','<<c.current_x<<','<<c.current_z<<','<<t.request.heading.x<<','<<t.request.heading.y<<','<<t.request.heading.z<<','<<c.controls<<'\n';
        const auto reset_metrics=m::measure(t,world);
        initial<<id<<','<<reset_metrics.forward_m*1000<<','<<reset_metrics.lateral_m*1000<<','<<reset_metrics.depth_m*1000
          <<','<<reset_metrics.tilt_deg<<','<<p::norm(reset_metrics.velocity_m_s)*1000<<'\n';
        require(reset_metrics.lateral_m<=.002f&&reset_metrics.depth_m<=.001f&&reset_metrics.tilt_deg<=10,"Current reset violates acceptance");
        double reward_sum=0,window_error_square=0;auto window_origin=cudarl::motor::motion(world.physical.body).com_model;
        bool all_windows=true;int case_windows=0;
        for(int k=0;k<=c.controls;++k) {
            const auto& body=world.physical.body;poses<<"current,"<<id<<','<<k<<','<<k*.05<<','<<body.position.x<<','<<body.position.y<<','<<body.position.z
              <<','<<body.orientation.w<<','<<body.orientation.x<<','<<body.orientation.y<<','<<body.orientation.z;
            for(float a:body.angle)poses<<','<<a;poses<<'\n';
            states<<id<<','<<k;body_csv(states,world.physical.body);states<<','<<world.physical.mean_squared_torque<<'\n';
            if(k==c.controls)break;
            float observation[41],flow[3],raw[4];require(task::observe(t,world,observation,flow,c.controls),"Current assessment observation rejected");
            backbone.forward(observation,1);policy::joint_latents(means.data(),backbone.output(),flow,raw);
            m::Action action;require(m::action_from_latent(raw,action),"Current assessment latent invalid");current_step(budget,world,action.command);
            const auto interval=task::record(t,world,c.controls);require(interval.accepted&&interval.done==(k+1==c.controls)&&!interval.truncated,"Current assessment task/terminal mismatch");
            reward_sum+=interval.reward;const auto& z=interval.metrics;
            trace<<id<<','<<k+1<<','<<(k+1)*.05<<','<<z.forward_m*1000<<','<<z.lateral_m*1000<<','<<z.depth_m*1000<<','<<z.tilt_deg
              <<','<<z.velocity_m_s.x*1000<<','<<z.velocity_m_s.y*1000<<','<<z.velocity_m_s.z*1000<<','<<interval.reward;
            for(int j=0;j<4;++j)trace<<','<<backbone.output()[j]<<','<<raw[j]<<','<<action.command[j];trace<<'\n';
            if(c.controls>160) {
                const int control=k+1;
                if(control==80)window_origin=cudarl::motor::motion(world.physical.body).com_model;
                if(control>80)window_error_square+=double(z.speed_error_m_s)*z.speed_error_m_s;
                if(control>=160&&control%80==0) {
                    const auto com=cudarl::motor::motion(world.physical.body).com_model;
                    const auto displacement=(com-window_origin)*p::scale;
                    const float speed=p::dot(displacement,t.request.heading)/(80*p::dt);
                    const float rmse=float(std::sqrt(window_error_square/80));
                    const bool pass=speed>=.0072f&&speed<=.0088f&&rmse<=.0025f;
                    windows<<id<<','<<control*.05<<','<<speed*1000<<','<<rmse*1000<<','<<pass<<'\n';
                    ++case_windows;++assessment.window_total;assessment.window_passed+=pass;all_windows=all_windows&&pass;
                    window_origin=com;window_error_square=0;
                }
            }
        }
        if(c.controls>160)require(case_windows==15,"Current endurance window count differs");
        const bool passed=t.outcome==m::Outcome::Success&&all_windows;const auto last=m::measure(t,world);
        const float work=world.physical.body.work/fmaxf(last.forward_m,1e-9f);
        summary<<id<<','<<c.pose<<','<<c.current_x*1000<<','<<c.current_z*1000<<','<<passed<<','<<t.steady_speed_m_s*1000
          <<','<<t.speed_rmse*1000<<','<<t.maximum_lateral*1000<<','<<t.maximum_depth*1000<<','<<t.maximum_tilt<<','<<t.path_angle_deg
          <<','<<work<<','<<last.forward_m*1000<<','<<reward_sum/(c.controls*.05)<<','<<world.physical.control_steps<<','<<world.attempted_substeps<<'\n';
        ++assessment.completed;assessment.passed+=passed;assessment.mean_cost-=reward_sum/(c.controls*.05)/double(cases.size());
    }
    require(bool(trace)&&bool(summary)&&bool(poses)&&bool(configurations)&&bool(states)&&bool(windows)&&bool(initial),"Current assessment write failed");return assessment;
}
using Keys=std::set<std::pair<int,int>>;
Keys long_keys(const fs::path& path) {
    std::ifstream input(path);require(bool(input),"Missing calm long-window evidence");std::string line;std::getline(input,line);Keys all,passed;int rows_seen=0;
    while(std::getline(input,line)){std::stringstream stream(line);std::string item;std::vector<double> value;while(std::getline(stream,item,','))value.push_back(std::stod(item));
        require(value.size()==10,"Calm long-window format differs");const int pose=int(value[0]),end=int(value[1]);require(all.emplace(pose,end).second,"Repeated long-window key");
        ++rows_seen;if(value[9]==1)passed.emplace(pose,end);}
    require(rows_seen==30,"Calm long-window count differs");return passed;
}
struct Calm {int short_pass=0,hold_pass=0,long_pass=0;Keys keys;};
Calm assess_calm(const fs::path& directory,Budget& budget) {
    require(fs::create_directory(directory),"Calm assessment identity exists");const int before=budget.controls;
    calm_states.open(directory/"states.csv");calm_states<<std::setprecision(12)<<"category,case,control";body_csv_header(calm_states);calm_states<<'\n';
    calm_group("short",4);const auto swim=evaluate(resume_state->actor.parameters,directory,"short",budget);close_calm_group();
    nn::CpuMlp<float> body(spec(4));body.params()=resume_state->actor.parameters;
    std::ofstream summary(directory/"holds.csv"),trace(directory/"hold-controls.csv");summary<<std::setprecision(12);trace<<std::setprecision(12);
    summary<<"case,pose,offset_mm,passed,max_speed_mm_s,max_anchor_mm,max_depth_mm,max_lane_mm,max_tilt_deg,final8_excursion_mm\n";
    trace<<"case,control,time_s,forward_mm,speed_mm_s,anchor_mm,depth_mm,lane_mm,tilt_deg\n";
    int holds=0;calm_group("hold",6);
    for(int id=0;id<6;++id) {
        const int pose=id%2?3:0;const float offset=id/2==0?0.f:id/2==1?-.0005f:.0005f;
        auto world=m::initial_world(pose);world.arena=m::long_swim_arena();m::ForwardTask t;
        auto config=m::config_for_speed(.008f);config.observation_schema=4;config.maximum_controls=1280;
        config.target_speed_m_s=0;config.stationary_request=true;config.position_hold=true;
        require(m::begin_forward(world,config,t)&&m::offset_episode_start(world,t,config,offset)&&m::update_velocity_request(world,config,t),"Calm hold initialization rejected");
        m::RestHoldAssessment score;require(score.record(0,m::measure(t,world)),"Calm hold initial sample invalid");
        for(int k=1;k<=1280;++k) {
            float observation[41],raw[4];require(m::update_velocity_request(world,config,t)&&m::observe_appended(t,world,config,observation),"Calm hold observation invalid");
            body.forward(observation,1);for(int j=0;j<4;++j)raw[j]=saved_hold_gain*body.output()[j];
            m::Action action;require(m::action_from_latent(raw,action),"Calm hold action invalid");budget.step(world,action.command);
            const auto interval=m::record_hold_control(t,world,config);require(interval.accepted&&score.record(k,interval.metrics),"Calm hold scoring rejected");
            const auto& z=interval.metrics;trace<<id<<','<<k<<','<<k*.05<<','<<z.forward_m*1000<<','<<p::norm(z.velocity_m_s)*1000
              <<','<<p::norm(z.hold_error_m)*1000<<','<<z.depth_m*1000<<','<<z.lateral_m*1000<<','<<z.tilt_deg<<'\n';
        }
        require(score.complete(),"Calm hold incomplete");holds+=score.passed();
        summary<<id<<','<<pose<<','<<offset*1000<<','<<score.passed()<<','<<score.peak_speed_m_s()*1000<<','<<score.peak_anchor_m()*1000
          <<','<<score.peak_depth_m()*1000<<','<<score.peak_lane_m()*1000<<','<<score.peak_tilt_degrees()<<','<<score.final_excursion_m()*1000<<'\n';
    }
    close_calm_group();require(fs::create_directory(directory/"long"),"Calm long identity exists");
    calm_group("long",2);const auto longer=sustained(resume_state->actor.parameters,directory/"long",budget);close_calm_group();
    calm_states.close();
    require(longer.safety_fault.empty()&&budget.controls-before==calm_controls,"Calm retention physical/count fault");
    require(bool(summary)&&bool(trace),"Calm hold evidence write failed");const auto keys=long_keys(directory/"long"/"sustained-windows.csv");
    require(keys.size()==size_t(longer.windows_passed),"Calm long score count differs");return {swim.passed,holds,longer.windows_passed,keys};
}
void require_calm(const Calm& calm) {require(calm.short_pass==4&&calm.hold_pass==6&&calm.long_pass==28,"Preserved calm abilities changed");}
void zero_flow_policy_check(const std::vector<float>& means) {
    require(means.size()==head_parameters,"Wrong current head size");finite(means);const float flow[3]={0,0,0};
    for(int sample=0;sample<64;++sample){float base[4],actual[4];for(int j=0;j<4;++j)base[j]=float(sample-32)*.07f+float(j)*.13f;
        policy::joint_latents(means.data(),base,flow,actual);for(int j=0;j<4;++j)require(actual[j]==base[j],"Calm passthrough changed");}
}
__global__ void math_fixtures(float* output) {
    const int fixture=threadIdx.x;if(fixture>=24)return;float means[head_parameters],latent[head_parameters],gradient[head_parameters];
    for(int j=0;j<head_parameters;++j){means[j]=float((fixture+j)%7-3)*.17f;latent[j]=means[j]+policy::fixed_sigma*(float((fixture*3+j)%11)-5)*.19f;}
    const float shift=fixture/8==0?0.f:fixture/8==1?-logf(2.f):logf(2.f);
    const float oldp=policy::log_probability(means,latent)+shift,advantage=fixture%2?.8f:-.8f;
    const auto terms=policy::loss(means,latent,oldp,advantage,1.f,gradient);float base[4],flow[3]={.5f,-.03f,-.8f},raw[4];
    for(int j=0;j<4;++j)base[j]=float(j-2)*.37f;policy::joint_latents(latent,base,flow,raw);
    const int stride=head_parameters*3+6;const int at=fixture*stride;
    for(int j=0;j<head_parameters;++j){output[at+j]=means[j];output[at+16+j]=latent[j];output[at+32+j]=gradient[j];}
    output[at+48]=oldp;output[at+49]=terms.pg_loss;for(int j=0;j<4;++j)output[at+50+j]=raw[j];
}
void numerical(const fs::path& directory) {
    require(fs::create_directory(directory),"Numerical current fixture identity exists");Stream stream;device_ptr<float> fixture(24*54);
    math_fixtures<<<1,32,0,stream.value>>>(fixture.get());const auto observed=download(fixture.get(),24*54,stream.value);
    double maximum_probability=0,maximum_gradient=0,maximum_fd=0,maximum_decode=0;
    for(int i=0;i<24;++i) {
        double means[16],latent[16],gradient[16];for(int j=0;j<16;++j){means[j]=observed[i*54+j];latent[j]=observed[i*54+16+j];}
        const double oldp=observed[i*54+48],advantage=i%2?.8:-.8;
        const double shift=i/8==0?0.:i/8==1?-double(logf(2.f)):double(logf(2.f));
        maximum_probability=std::max(maximum_probability,std::abs(policy::log_probability(means,latent)-(oldp-shift)));
        (void)policy::loss(means,latent,oldp,advantage,1.,gradient);
        for(int j=0;j<16;++j){maximum_gradient=std::max(maximum_gradient,std::abs(gradient[j]-observed[i*54+32+j]));
            const double saved=means[j],epsilon=1e-6;double unused[16];means[j]=saved+epsilon;const auto plus=policy::loss(means,latent,oldp,advantage,1.,unused);
            means[j]=saved-epsilon;const auto minus=policy::loss(means,latent,oldp,advantage,1.,unused);means[j]=saved;
            maximum_fd=std::max(maximum_fd,std::abs((plus.pg_loss-minus.pg_loss)/(2*epsilon)-gradient[j]));}
        double base[4],flow[3]={double(.5f),double(-.03f),double(-.8f)},raw[4];for(int j=0;j<4;++j)base[j]=double(float(j-2)*.37f);
        policy::joint_latents(latent,base,flow,raw);for(int j=0;j<4;++j)maximum_decode=std::max(maximum_decode,std::abs(raw[j]-observed[i*54+50+j]));
    }
    require(maximum_probability<2e-5&&maximum_gradient<.0001&&maximum_fd<.0001&&maximum_decode<2e-6,"Current Gaussian/decode/gradient numerical gate failed");
    for(int n:supported_sizes) {
        std::vector<float> returns(n);for(int i=0;i<n;++i)returns[i]=float(i%16)*.3f+float(i/16)*.1f;
        const auto before=leave_one_out(returns,2);const int selected=17;const double old_baseline=returns[selected]-2*before[selected];
        returns[selected]+=7;const auto after=leave_one_out(returns,2);
        require(std::abs(old_baseline-(returns[selected]-2*after[selected]))<1e-5,"Own return entered V2 LOO baseline");
        const auto scale=fit_scale(returns);require(scale.value>0,"Synthetic V2 scale failed");
    }
    std::vector<float> extreme(16,1e6f);zero_flow_policy_check(extreme);
    std::ofstream report(directory/"math-zero.json");report<<std::setprecision(17)
      <<"{\"passed\":true,\"actual_CUDA_cases\":24,\"max_logp_error\":"<<maximum_probability<<",\"max_gradient_error\":"<<maximum_gradient
      <<",\"max_finite_difference_error\":"<<maximum_fd<<",\"max_decode_error\":"<<maximum_decode
      <<",\"LOO_own_return_excluded\":true,\"exact_calm_passthrough\":true,\"physical_controls\":0,\"optimizer_steps\":0}\n";
}
void bulk_policy_zero(Data& data,const std::vector<float>& latent,const std::vector<float>& old_logp,const fs::path& directory) {
    const int n=data.n,mb=data.minibatch;const auto stream=data.stream->value;std::vector<float> advantage(n);
    for(int i=0;i<n;++i)advantage[i]=.25f+float(i%19-9)*.07f;
    data.advantages.copy_from_host(advantage.data(),n);const auto permutation=rl::make_permutation(n,0x7138b2a9u);
    std::vector<int> seen(n,0);double maximum_row=0,maximum_sum=0,maximum_stat=0,maximum_last=0;int checked=0,last_rows=0;
    for(int first=0;first<n;first+=mb) {
        rl::gather_rows(permutation,first,mb,head_parameters,data.latent.get(),data.mb_latent.get(),stream);
        rl::gather_rows(permutation,first,mb,1,data.old_logp.get(),data.mb_oldlogp.get(),stream);
        rl::gather_rows(permutation,first,mb,1,data.advantages.get(),data.mb_advantage.get(),stream);
        CUDA_CHECK_THROW(cudaMemsetAsync(data.row_gradient.get(),0,size_t(mb)*16*sizeof(float),stream));
        CUDA_CHECK_THROW(cudaMemsetAsync(data.row_stats.get(),0,size_t(mb)*5*sizeof(float),stream));
        episode_loss<<<(mb+127)/128,128,0,stream>>>(mb,data.means.get(),data.mb_latent.get(),data.mb_oldlogp.get(),data.mb_advantage.get(),data.row_gradient.get(),data.row_stats.get());
        CUDA_CHECK_THROW(cudaMemsetAsync(data.gradient.get(),0,16*sizeof(float),stream));CUDA_CHECK_THROW(cudaMemsetAsync(data.stats.get(),0,5*sizeof(float),stream));
        rl::sum_columns(data.row_gradient.get(),mb,16,data.gradient.get(),stream);rl::sum_columns(data.row_stats.get(),mb,5,data.stats.get(),stream);
        const auto gathered=download(data.mb_latent.get(),size_t(mb)*16,stream),gp=download(data.mb_oldlogp.get(),mb,stream),
          ga=download(data.mb_advantage.get(),mb,stream),rows_gradient=download(data.row_gradient.get(),size_t(mb)*16,stream),
          sum=download(data.gradient.get(),16,stream),stats=download(data.stats.get(),5,stream);
        double expected_sum[16]{},expected_loss=0;const double fixed_log_sigma=std::log(.5),entropy=16*(fixed_log_sigma+.5*(1.83787706640934548+1));
        for(int row=0;row<mb;++row) {
            const int original=int(rl::permute(permutation,uint32_t(first+row)));require(original>=0&&original<n&&++seen[original]==1,"V2 bulk permutation coverage defect");
            require(gp[row]==old_logp[original]&&ga[row]==advantage[original],"V2 bulk scalar gather defect");
            double logp=-16*fixed_log_sigma-8*1.83787706640934548;for(int j=0;j<16;++j) {
                require(gathered[row*16+j]==latent[original*16+j],"V2 bulk latent gather defect");
                const double z=double(gathered[row*16+j])/.5;logp-=.5*z*z;
            }
            const double ratio=std::exp(logp-double(gp[row]));require(std::abs(ratio-1)<.001,"Synthetic V2 ratio left unclipped fixture");
            expected_loss-=double(ga[row])*ratio;
            double last_magnitude=0;
            for(int j=0;j<16;++j) {
                const double expected=-double(ga[row])*ratio*double(gathered[row*16+j])/.25/mb;
                const double error=std::abs(expected-rows_gradient[row*16+j]);maximum_row=std::max(maximum_row,error);expected_sum[j]+=expected;
                if(row==mb-1){maximum_last=std::max(maximum_last,error);last_magnitude+=std::abs(expected);}
            }
            if(row==mb-1){require(last_magnitude>1e-8,"V2 bulk last-row fixture is vacuous");++last_rows;}
            ++checked;
        }
        for(int j=0;j<16;++j)maximum_sum=std::max(maximum_sum,std::abs(expected_sum[j]-sum[j]));
        maximum_stat=std::max(maximum_stat,std::abs(expected_loss/mb-double(stats[0])/mb));
        maximum_stat=std::max(maximum_stat,std::abs(entropy-double(stats[2])/mb));
        require(stats[1]==0&&std::abs(stats[3]/mb)<1e-5&&stats[4]==0,"V2 bulk critic/KL/clipping fixture defect");
    }
    for(int count:seen)require(count==1,"V2 bulk minibatches missed an episode");
    require(checked==n&&last_rows==4&&maximum_row<1e-5&&maximum_sum<2e-5&&maximum_stat<2e-5,"V2 bulk native gradient/reduction gate failed");
    const auto means=download(data.means.get(),16,stream);for(float x:means)require(x==0,"V2 bulk gradient fixture updated policy");
    std::ofstream report(directory/"bulk-policy-zero.json");report<<std::setprecision(17)<<"{\"passed\":true,\"worlds\":"<<n
      <<",\"minibatch_episodes\":"<<mb<<",\"loss_grid_blocks\":"<<(mb+127)/128<<",\"checked_rows\":"<<checked<<",\"checked_last_rows\":"<<last_rows
      <<",\"max_row_gradient_error\":"<<maximum_row<<",\"max_reduced_gradient_error\":"<<maximum_sum<<",\"max_stats_error\":"<<maximum_stat
      <<",\"max_last_row_gradient_error\":"<<maximum_last<<",\"physical_controls\":0,\"optimizer_steps\":0}\n";
}
void initialization_zero(const fs::path& directory,int n) {
    require(fs::create_directory(directory),"V2 initialization zero identity exists");Data data(n,false);const auto stream=data.stream->value;const int grid=(n+127)/128;
    initialize_episodes<<<grid,128,0,stream>>>(n,data.worlds.get(),data.tasks.get(),data.means.get(),data.latent.get(),data.old_logp.get(),data.fault.get(),calibration_seed,0);
    observe_episodes<<<grid,128,0,stream>>>(n,data.worlds.get(),data.tasks.get(),data.observations.get(),data.flow.get(),data.fault.get());
    data.backbone->forward(data.observations.get(),n,stream);
    const auto worlds=download(data.worlds.get(),n,stream);const auto tasks=download(data.tasks.get(),n,stream);
    const auto inputs=download(data.observations.get(),size_t(n)*41,stream),flow=download(data.flow.get(),size_t(n)*3,stream),outputs=download(data.backbone->output(),size_t(n)*4,stream);
    const auto latent=download(data.latent.get(),size_t(n)*16,stream),logp=download(data.old_logp.get(),n,stream);
    bulk_policy_zero(data,latent,logp,directory);
    nn::CpuMlp<float> body(spec(4));body.params()=resume_state->actor.parameters;double maximum_input=0,maximum_body=0,maximum_sample=0,maximum_logp=0;
    int coverage[16]{};
    for(int i=0;i<n;++i) {
        const auto c=training_case(i%16);++coverage[i%16];m::World world;m::ForwardTask t;
        require(task::initialize(c.pose,{c.current_x,0,c.current_z},world,t),"CPU V2 zero initialize failed");
        float observation[41],local[3];require(task::observe(t,world,observation,local),"CPU V2 zero observation failed");
        require(worlds[i].attempted_substeps==0&&worlds[i].physical.control_steps==0&&worlds[i].physical.substeps==0&&tasks[i].controls==0,"V2 zero fixture advanced physics");
        for(int j=0;j<41;++j)maximum_input=std::max(maximum_input,double(std::abs(observation[j]-inputs[i*41+j])));
        for(int j=0;j<3;++j)maximum_input=std::max(maximum_input,double(std::abs(local[j]-flow[i*3+j])));
        body.forward(observation,1);for(int j=0;j<4;++j)maximum_body=std::max(maximum_body,double(std::abs(body.output()[j]-outputs[i*4+j])));
        double mean[16]{},sample[16];for(int j=0;j<16;++j){const float expected=policy::fixed_sigma*cudarl::rng::normal(calibration_seed,0,uint32_t(i*16+j));
            maximum_sample=std::max(maximum_sample,double(std::abs(expected-latent[i*16+j])));sample[j]=latent[i*16+j];}
        maximum_logp=std::max(maximum_logp,std::abs(policy::log_probability(mean,sample)-logp[i]));
    }
    for(int count:coverage)require(count==n/16,"V2 initialization case coverage failed");
    require(download(data.fault.get(),1,stream)[0]==0&&maximum_input<1e-6&&maximum_body<2e-5&&maximum_sample<2e-5&&maximum_logp<.0001,"V2 CPU/CUDA initialization gate failed");
    cp::Writer proof;proof.value(uint32_t{0x32495343});proof.value(uint32_t{2});proof.value(n);
    proof.array(inputs);proof.array(flow);proof.array(outputs);proof.array(latent);proof.array(logp);
    proof.value(uint32_t{sizeof(m::World)});proof.array(worlds);proof.value(uint32_t{sizeof(m::ForwardTask)});proof.array(tasks);
    cp::write_atomic((directory/"initialization.bin").string(),proof.data);
    std::ofstream report(directory/"initialization-zero.json");report<<std::setprecision(17)
      <<"{\"passed\":true,\"worlds\":"<<n<<",\"max_input_error\":"<<maximum_input<<",\"CPU_CUDA_body_error\":"<<maximum_body
      <<",\"sample_error\":"<<maximum_sample<<",\"old_logp_error\":"<<maximum_logp<<",\"physical_controls\":0,\"optimizer_steps\":0}\n";
}
void kernel_metadata(const fs::path& directory) {
    cudaDeviceProp device{};CUDA_CHECK_THROW(cudaGetDeviceProperties(&device,0));size_t free=0,total=0;
    CUDA_CHECK_THROW(cudaMemGetInfo(&free,&total));cudaFuncAttributes attributes{};
    CUDA_CHECK_THROW(cudaFuncGetAttributes(&attributes,advance_episodes));int active_blocks=0;
    CUDA_CHECK_THROW(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,advance_episodes,128,0));
    std::ofstream out(directory/"gpu-kernel-metadata.json");out<<std::setprecision(17)
      <<"{\"device\":"<<std::quoted(device.name)<<",\"SMs\":"<<device.multiProcessorCount<<",\"compute_major\":"<<device.major
      <<",\"compute_minor\":"<<device.minor<<",\"total_bytes\":"<<total<<",\"free_bytes\":"<<free<<",\"block_threads\":128"
      <<",\"registers_per_thread\":"<<attributes.numRegs<<",\"local_bytes_per_thread\":"<<attributes.localSizeBytes
      <<",\"shared_bytes_per_block\":"<<attributes.sharedSizeBytes<<",\"active_blocks_per_SM\":"<<active_blocks
      <<",\"maximum_threads_per_SM\":"<<device.maxThreadsPerMultiProcessor
      <<",\"theoretical_occupancy\":"<<double(active_blocks*128)/device.maxThreadsPerMultiProcessor
      <<",\"projected_study_bytes\":"<<projected_study_bytes()<<",\"initial_disk_guard_bytes\":"<<initial_disk_free
      <<",\"working_reserve_bytes\":"<<working_disk_reserve<<",\"sizes\":[";
    for(int i=0;i<4;++i){if(i)out<<',';const int n=supported_sizes[i];
        out<<"{\"worlds\":"<<n<<",\"grid_blocks\":"<<(n+127)/128<<",\"minibatch_episodes\":"<<n/4
          <<",\"episodes_per_case\":"<<n/16<<",\"trace_bytes\":"<<uint64_t(n)*160*125*4<<"}";}
    out<<"],\"physical_controls\":0,\"optimizer_steps\":0}\n";
}
void zero_fixtures(const fs::path& directory) {
    require(fs::create_directory(directory),"V2 zero fixture identity exists");numerical(directory/"math");
    for(int n:supported_sizes)initialization_zero(directory/("N-"+std::to_string(n)),n);
    int checked=0;for(const auto& c:withheld_cases()) {
        m::World world;m::ForwardTask t;float observation[41],local[3];
        require(task::initialize(c.pose,{c.current_x,0,c.current_z},world,t,c.controls)&&
          task::observe(t,world,observation,local,c.controls),"V2 withheld reset/domain/observation invalid");
        require(world.attempted_substeps==0&&world.physical.control_steps==0&&world.physical.substeps==0&&t.controls==0,"V2 withheld zero check advanced physics");++checked;
    }
    auto config=initial_optimizer_config(true);config.lr=.01f;nn::GpuAdam first(16,config),second(16,config);
    cp::Writer a,b;first.save(a);cp::Reader reader{a.data};second.load(reader);reader.finish();second.save(b);
    require(first.steps()==0&&second.steps()==0&&a.data==b.data,"Fresh V2 Adam zero roundtrip failed");
    kernel_metadata(directory);std::ofstream out(directory/"summary.json");
    out<<"{\"passed\":true,\"supported_sizes\":4,\"withheld_configurations_zero_step\":"<<checked
      <<",\"preserved_holding_gain\":"<<std::setprecision(17)<<saved_hold_gain<<",\"physical_controls\":0,\"optimizer_steps\":0}\n";
}
void write_layout(const fs::path& root) {
    std::ofstream out(root/"layout.json");out<<"{\"episode_counts\":{\"size\":"<<sizeof(EpisodeCounts)
      <<",\"controls\":"<<offsetof(EpisodeCounts,controls)<<",\"substeps\":"<<offsetof(EpisodeCounts,substeps)
      <<",\"reward_sum\":"<<offsetof(EpisodeCounts,reward_sum)<<",\"failure_code\":"<<offsetof(EpisodeCounts,failure_code)
      <<",\"first_before\":"<<offsetof(EpisodeCounts,first_before)<<",\"rejected_candidate\":"<<offsetof(EpisodeCounts,rejected_candidate)
      <<"},\"world_size\":"<<sizeof(m::World)<<",\"task_size\":"<<sizeof(m::ForwardTask)
      <<",\"body_size\":"<<sizeof(p::BodyState)<<",\"pre_floats\":33,\"post_floats\":23,\"metric_floats\":16}\n";
    require(bool(out),"Current ABI layout evidence write failed");
}
void result(const fs::path& file,const Budget& budget,const std::string& fault,const std::string& gate,int updates_done,
            int optimizer_steps,double prior,int baseline_pass,int final_pass,int withheld,int endurance,bool qualified,
            bool baseline_only,int selected_n=0,bool attempt_closed=true) {
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::ofstream out(file);out<<std::setprecision(17)<<"{\"schema\":2,\"controls\":"<<budget.controls<<",\"substeps\":"<<budget.substeps
      <<",\"active_seconds\":"<<seconds<<",\"prior_active_seconds\":"<<prior<<",\"PPO_updates\":"<<updates_done
      <<",\"actor_steps\":"<<optimizer_steps<<",\"critic_steps\":0,\"critic_forward_calls\":0,\"fault\":"<<std::quoted(fault)
      <<",\"gate\":"<<std::quoted(gate)<<",\"baseline_current_passes\":"<<baseline_pass<<",\"final_current_passes\":"<<final_pass
      <<",\"withheld_passes\":"<<withheld<<",\"endurance_passes\":"<<endurance<<",\"qualified\":"<<(qualified?"true":"false")
      <<",\"baseline_only\":"<<(baseline_only?"true":"false")<<",\"selected_worlds\":"<<selected_n
      <<",\"phase_closed\":true,\"closed\":"<<(attempt_closed?"true":"false")<<",\"runtime_adopted\":false,\"retries\":0}\n";
    out.close();require(bool(out),"Cannot save V2 terminal phase result");
    std::ofstream live(file.parent_path()/"status.json");live<<"{\"phase\":"<<std::quoted(file.stem().string()+"_closed")
      <<",\"closed\":"<<(attempt_closed?"true":"false")<<",\"PPO_updates\":"<<updates_done<<",\"controls\":"<<budget.controls
      <<",\"substeps\":"<<budget.substeps<<",\"worlds\":"<<selected_n<<",\"qualified\":"<<(qualified?"true":"false")<<",\"unix_seconds\":"<<unix_now()<<"}\n";
}
struct Prior {int controls=0,substeps=0,n=0;double seconds=0;};
double elapsed() {return std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();}
std::vector<fs::path> baseline_bindings(const fs::path& root) {
    return {root/"baseline-result.json",root/"baseline-current"/"summary.csv",root/"baseline-calm"/"long"/"sustained-windows.csv",root/"physics-gate"/"gate-passed.bin"};
}
std::vector<fs::path> profile_bindings(const fs::path& root) {
    return {root/"profile-result.json",root/"profile-manifest.json",root/"selected-size.bin",root/"baseline-ready.bin"};
}
std::vector<fs::path> calibration_bindings(const fs::path& root) {
    return {root/"calibration-result.json",root/"calibration-report.json",root/"fixed-scale.bin",root/"selected-size.bin",root/"profile-ready.bin",root/"physics-gate"/"gate-passed.bin"};
}
void write_ready(const fs::path& root,const char* phase,int phase_id,const Budget& budget,double prior,int n,const std::vector<fs::path>& bindings) {
    cp::Writer ready;ready.value(release_magic);ready.value(uint32_t{2});ready.value(study_protocol);ready.value(phase_id);
    ready.value(n);ready.value(budget.controls);ready.value(budget.substeps);ready.value(prior+elapsed());
    std::vector<uint64_t> hashes;for(const auto& file:bindings)hashes.push_back(file_hash(file));ready.array(hashes);
    const auto file=root/(std::string(phase)+"-ready.bin");require(!fs::exists(file),"V2 phase release already exists");cp::write_atomic(file.string(),ready.data);
}
Prior read_ready(const fs::path& root,const char* phase,int phase_id,const std::vector<fs::path>& bindings) {
    const auto bytes=cp::read((root/(std::string(phase)+"-ready.bin")).string());cp::Reader in{bytes};
    require(in.value<uint32_t>()==release_magic&&in.value<uint32_t>()==2&&in.value<uint64_t>()==study_protocol&&in.value<int>()==phase_id,"Wrong V2 phase release");
    Prior prior;prior.n=in.value<int>();prior.controls=in.value<int>();prior.substeps=in.value<int>();prior.seconds=in.value<double>();
    const auto hashes=in.array<uint64_t>(32);in.finish();require(hashes.size()==bindings.size(),"V2 release binding count differs");
    for(size_t i=0;i<bindings.size();++i)require(hashes[i]==file_hash(bindings[i]),"V2 preceding-phase evidence changed");
    require(prior.controls>=gate_controls&&prior.controls<=controls_cap&&prior.substeps==prior.controls*10&&
      std::isfinite(prior.seconds)&&prior.seconds>0&&prior.seconds<seconds_cap,"V2 phase accounting/time invalid");
    return prior;
}
Budget begin_phase(const fs::path& root,const char* phase,const Prior& prior) {
    require(fs::create_directory(root/(std::string(phase)+"-started")),"V2 phase already attempted; no retry");
    failure_directory=root/(std::string(phase)+"-started");failure_saved=false;
    started=std::chrono::steady_clock::now();active_wall_cap=seconds_cap-prior.seconds;Budget budget;
    budget.maximum_controls=controls_cap;budget.maximum_substeps=substeps_cap;budget.seconds=active_wall_cap;
    budget.controls=prior.controls;budget.substeps=prior.substeps;return budget;
}
Assessment read_baseline_current(const fs::path& root) {
    std::ifstream input(root/"baseline-current"/"summary.csv");require(bool(input),"Missing V2 current baseline");std::string line;std::getline(input,line);Assessment baseline;
    while(std::getline(input,line)){std::stringstream row(line);std::string item;std::vector<double> v;while(std::getline(row,item,','))v.push_back(std::stod(item));
        require(v.size()==16&&int(v[0])==baseline.completed&&v[14]==160&&v[15]==1600,"Wrong V2 current baseline shape");
        ++baseline.completed;baseline.passed+=int(v[4]);baseline.mean_cost-=v[13]/16;}
    require(baseline.completed==16,"Incomplete V2 current baseline");return baseline;
}
int read_selection(const fs::path& root) {
    const auto bytes=cp::read((root/"selected-size.bin").string());cp::Reader in{bytes};
    require(in.value<uint32_t>()==0x325a5343&&in.value<uint32_t>()==2&&in.value<uint64_t>()==study_protocol,"Wrong V2 selected-size identity");
    const int n=in.value<int>();require(supported_world_count(n),"Selected V2 size unsupported");
    require(in.value<uint64_t>()==file_hash(root/"profile-manifest.json")&&in.value<uint64_t>()==file_hash(root/"baseline-ready.bin"),"Selected V2 profile or baseline changed");in.finish();
    selected_size_binding=cp::checksum(bytes);return n;
}
int prepare(const fs::path& root,const fs::path& protocol_file) {
    require(fs::is_regular_file(protocol_file),"Approved V2 protocol unavailable");
    const bool existing=fs::exists(root);
    if(existing) {
        require(fs::is_directory(root),"V2 study root is not a directory");
        const std::set<std::string> allowed={"physics-gate","gate-source","gate-frozen-inputs.json","process-execution.json",
          "Gate.stdout.log","Gate.stderr.log","gpu-telemetry-Gate.csv","gpu-telemetry-Gate.err",
          "independent-envelope-paths.txt","independent-gate-verification.json","approved-protocol.md"};
        for(const auto& entry:fs::directory_iterator(root))
            require(allowed.count(entry.path().filename().string())==1,"Existing V2 root contains learner work or unrelated files; preserve it");
        require(fs::is_regular_file(root/"gate-frozen-inputs.json")&&fs::is_regular_file(root/"physics-gate"/"gate-passed.bin")&&
          fs::is_regular_file(root/"independent-gate-verification.json"),"Existing V2 root is not a sealed independently verified passed gate");
    } else require(fs::create_directories(root),"Cannot create V2 study root");
    if(fs::exists(root/"approved-protocol.md"))require(file_hash(root/"approved-protocol.md")==file_hash(protocol_file),"Existing gate-parent approved protocol differs");
    else fs::copy_file(protocol_file,root/"approved-protocol.md");
    configure(root,false);if(existing)validate_physics_gate(root);
    fs::copy_file(parent_checkpoint_path,root/"parent-checkpoint.bin");fs::copy_file(parent_protocol_path,root/"parent-protocol.bin");
    fs::copy_file(fs::absolute("assets/ppo-four-axis-checkpoints/learned-amplitude-2026-11-22-qualified/qualified-policy-update-7.bin"),root/"holding-parent.bin");
    const auto protocol=protocol_bytes(root);study_protocol=cp::checksum(protocol);cp::write_atomic((root/"protocol.bin").string(),protocol);
    cp::write_atomic((root/"migrated-initial.bin").string(),rl::write_appended_hold(migrated41,study_protocol));
    require(fs::create_directory(root/"initial-policy"),"V2 initial policy identity exists");
    {Data data(supported_sizes[0],true);save_policy(data,root/"initial-policy",study_protocol,0);}
    zero_fixtures(root/"prepare-zero");write_layout(root);
    std::ofstream info(root/"prepared.json");info<<"{\"schema\":2,\"prepared\":true,\"world_sizes\":[2048,4096,8192,16384],\"controls_per_episode\":160"
      <<",\"profile_controls\":2949120,\"updates_max\":32,\"controls_cap\":89571440,\"substeps_cap\":895714400"
      <<",\"initial_disk_free_bytes\":"<<initial_disk_free<<",\"projected_study_bytes\":"<<projected_study_bytes()
      <<",\"physical_controls\":0,\"optimizer_steps\":0}\n";return 0;
}
int preflight(const fs::path& root) {
    configure(root,true);validate_protocol(root);zero_fixtures(root/"verified-zero");
    std::ofstream marker(root/"preflight-passed.txt");marker<<"V2 runtime-N Gaussian, policy, body/flow, LOO, source/checkpoint and kernel-metadata checks passed with zero physical controls/optimizer steps.\n";return 0;
}
int baseline(const fs::path& root) {
    configure(root,true);validate_protocol(root);validate_physics_gate(root);require(fs::exists(root/"preflight-passed.txt"),"V2 numerical preflight missing");
    disk_guard(root,0,true);const Prior prior{gate_controls,gate_substeps,0,prior_gate_seconds};auto budget=begin_phase(root,"baseline",prior);
    std::string fault,gate;int original_pass=0,withheld_pass=0,endurance_pass=0;bool qualified=false,baseline_only=false;
    try {
        require_calm(assess_calm(root/"baseline-calm",budget));
        const std::vector<float> zero(16,0);const auto before=assess_current(zero,training_cases(),root/"baseline-current",budget);original_pass=before.passed;
        if(before.passed==16) {
            baseline_only=true;require_calm(assess_calm(root/"final-calm",budget));
            const auto withheld=assess_current(zero,withheld_cases(),root/"reserved-withheld",budget);withheld_pass=withheld.passed;
            if(withheld.passed==16){const auto endurance=assess_current(zero,endurance_cases(),root/"reserved-endurance",budget);endurance_pass=endurance.passed;}
            qualified=withheld_pass==16&&endurance_pass==2;gate=qualified?"baseline_already_qualified_no_PPO":"baseline_reserved_validation_failed_no_PPO";
        } else require(budget.controls==baseline_controls,"V2 baseline accounting differs");
        wall_guard();
    }catch(const std::exception& error){fault=error.what();std::cerr<<fault<<std::endl;}
    const bool closed=!fault.empty()||!gate.empty();
    result(root/"baseline-result.json",budget,fault,gate,0,0,prior.seconds,original_pass,original_pass,withheld_pass,endurance_pass,qualified,baseline_only,0,closed);
    if(!closed)write_ready(root,"baseline",1,budget,prior.seconds,0,baseline_bindings(root));
    return fault.empty()?0:3;
}
int profile(const fs::path& root) {
    configure(root,true);validate_protocol(root);validate_physics_gate(root);
    const auto prior=read_ready(root,"baseline",1,baseline_bindings(root));const auto before=read_baseline_current(root);
    require(before.passed<16,"Passing inherited response does not authorize unnecessary profiling");auto budget=begin_phase(root,"profile",prior);
    std::string fault,gate;int winner=0;double best=-1;
    try {
        std::ofstream manifest(root/"profile-manifest.json");manifest<<std::setprecision(17)
          <<"{\"schema\":2,\"protocol\":"<<study_protocol<<",\"profile_seed\":"<<profile_seed<<",\"ticks\":32,\"sizes\":[";
        bool first_size=true;
        for(int n:supported_sizes) {
            const auto directory=root/"profile-started"/("N-"+std::to_string(n));require(fs::create_directory(directory),"V2 profile size identity exists");
            disk_guard(root,collection_bytes(n,32));Data data(n,false);
            double measured_seconds=0;std::vector<uint64_t> evidence_hash,timing_hash;std::vector<double> seconds;
            for(int sample=0;sample<3;++sample) {
                const auto batch=collect(data,budget,directory,"profile",sample,2,32);seconds.push_back(batch.end_to_end_seconds);
                if(sample)measured_seconds+=batch.end_to_end_seconds;
                const auto saved=directory/("profile-batch-"+std::to_string(sample));
                evidence_hash.push_back(file_hash(saved/"manifest.json"));timing_hash.push_back(file_hash(saved/"timing.json"));
                std::cout<<"PROFILE_V2 N="<<n<<" sample="<<sample<<" seconds="<<batch.end_to_end_seconds<<" controls="<<budget.controls<<std::endl;
            }
            const double score=double(2*n*32)/measured_seconds;require(std::isfinite(score)&&score>0,"Invalid V2 pipeline profile timing");
            if(score>best){best=score;winner=n;}
            if(!first_size)manifest<<',';first_size=false;
            manifest<<"{\"worlds\":"<<n<<",\"timed_seconds\":"<<measured_seconds<<",\"controls_per_second\":"<<score<<",\"samples\":[";
            for(int sample=0;sample<3;++sample){if(sample)manifest<<',';manifest<<"{\"sample\":"<<sample<<",\"warmup\":"<<(sample==0?"true":"false")
              <<",\"end_to_end_seconds\":"<<seconds[sample]<<",\"batch_manifest_checksum\":"<<evidence_hash[sample]<<",\"timing_checksum\":"<<timing_hash[sample]<<"}";}
            manifest<<"]}";manifest.flush();require(bool(manifest),"V2 profile manifest write failed");
        }
        manifest<<"],\"selected_worlds\":"<<winner<<",\"selection\":\"highest total timed controls divided by total timed end-to-end seconds; exact tie smaller N\"}\n";
        manifest.close();require(bool(manifest)&&supported_world_count(winner),"V2 profile selection missing");
        require(budget.controls-prior.controls==profile_controls,"V2 profiling physical accounting differs");
        cp::Writer selected;selected.value(uint32_t{0x325a5343});selected.value(uint32_t{2});selected.value(study_protocol);selected.value(winner);
        selected.value(file_hash(root/"profile-manifest.json"));selected.value(file_hash(root/"baseline-ready.bin"));
        cp::write_atomic((root/"selected-size.bin").string(),selected.data);selected_size_binding=cp::checksum(selected.data);wall_guard();
    } catch(const std::exception& error){fault=error.what();std::cerr<<fault<<std::endl;}
    result(root/"profile-result.json",budget,fault,gate,0,0,prior.seconds,before.passed,before.passed,0,0,false,false,winner,!fault.empty());
    if(fault.empty())write_ready(root,"profile",2,budget,prior.seconds,winner,profile_bindings(root));
    return fault.empty()?0:3;
}
int calibrate(const fs::path& root) {
    configure(root,true);validate_protocol(root);validate_physics_gate(root);
    const auto prior=read_ready(root,"profile",2,profile_bindings(root));const int n=read_selection(root);
    require(prior.n==n,"V2 profile/selection size differs");const auto before=read_baseline_current(root);auto budget=begin_phase(root,"calibration",prior);
    std::string fault,gate;
    try {
        disk_guard(root,collection_bytes(n,160));Data data(n,false);
        const auto batch=collect(data,budget,root/"calibration-started","calibration",0,0);const auto scale=fit_scale(batch.returns);
        const auto bytes=scale_bytes(scale,study_protocol);cp::write_atomic((root/"fixed-scale.bin").string(),bytes);
        require(scale_bytes(read_scale(root,study_protocol,n),study_protocol)==bytes,"V2 scale exact roundtrip failed");
        std::vector<float> absolute;for(float x:batch.returns)absolute.push_back(std::abs(x));std::sort(absolute.begin(),absolute.end());
        std::ofstream report(root/"calibration-report.json");report<<std::setprecision(17)
          <<"{\"schema\":2,\"passed\":true,\"episodes\":"<<n<<",\"cases\":16,\"episodes_per_case\":"<<n/16
          <<",\"fixed_global_residual_RMS\":"<<scale.value<<",\"sampled_complete_passes\":"<<batch.passed
          <<",\"baseline_current_passes\":"<<before.passed<<",\"baseline_current_mean_cost\":"<<before.mean_cost
          <<",\"return_abs_p99\":"<<absolute[size_t(std::ceil(.99*absolute.size()))-1]<<",\"return_abs_max\":"<<absolute.back()
          <<",\"max_old_logp_error\":"<<batch.maximum_logp_error<<",\"max_reward_sum_error\":"<<batch.maximum_return_error
          <<",\"max_action_error\":"<<batch.maximum_action_error<<",\"controls\":"<<budget.controls<<",\"substeps\":"<<budget.substeps
          <<",\"actor_steps\":0,\"critic_steps\":0}\n";
        report.close();require(bool(report)&&budget.controls-prior.controls==n*160,"V2 calibration report/count differs");wall_guard();
    }catch(const std::exception& error){fault=error.what();std::cerr<<fault<<std::endl;}
    result(root/"calibration-result.json",budget,fault,gate,0,0,prior.seconds,before.passed,before.passed,0,0,false,false,n,!fault.empty());
    if(fault.empty())write_ready(root,"calibration",3,budget,prior.seconds,n,calibration_bindings(root));
    return fault.empty()?0:3;
}
int train(const fs::path& root) {
    configure(root,true);validate_protocol(root);validate_physics_gate(root);
    const auto prior=read_ready(root,"calibration",3,calibration_bindings(root));const int n=read_selection(root);
    require(prior.n==n,"V2 calibration/selected size differs");const auto scale=read_scale(root,study_protocol,n);const auto before=read_baseline_current(root);
    require(before.passed<16,"Passing inherited response does not authorize unnecessary PPO");auto budget=begin_phase(root,"training",prior);
    std::string fault,gate;int updates_done=0,steps=0,final_pass=before.passed,withheld_pass=0,endurance_pass=0;bool qualified=false;
    std::unique_ptr<Data> active;std::ofstream progress(root/"updates.jsonl");progress<<std::setprecision(17);
    try {
        disk_guard(root,collection_bytes(n,160));active=std::make_unique<Data>(n,true);auto& data=*active;
        save_policy(data,root/"training-started",study_protocol,0);
        for(int update=1;update<=maximum_updates;++update) {
            require(budget.controls+n*episode_controls+current_suite_controls+calm_controls+current_suite_controls+2560<=controls_cap,"V2 reserved final budget exhausted");
            const auto batch=collect(data,budget,root/"training-started","training",update,1);
            const auto optimize_start=std::chrono::steady_clock::now();const double kl=optimize(data,batch,scale,update,root/"training-started");
            const double optimize_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-optimize_start).count();
            updates_done=update;data.completed=update;steps=data.optimizer->steps();save_policy(data,root/"training-started",study_protocol,update);
            const auto means=download(data.means.get(),head_parameters,data.stream->value);zero_flow_policy_check(means);
            const auto measured=assess_current(means,training_cases(),root/"training-started"/("assessment-"+std::to_string(update)),budget);final_pass=measured.passed;
            progress<<"{\"update\":"<<update<<",\"worlds\":"<<n<<",\"minibatch_episodes\":"<<n/4
              <<",\"current_passes\":"<<measured.passed<<",\"current_mean_cost\":"<<measured.mean_cost
              <<",\"baseline_current_passes\":"<<before.passed<<",\"baseline_mean_cost\":"<<before.mean_cost
              <<",\"calm_passthrough_exact\":true,\"KL\":"<<kl<<",\"actor_steps\":"<<steps<<",\"controls\":"<<budget.controls
              <<",\"collection_seconds\":"<<batch.end_to_end_seconds<<",\"optimize_seconds\":"<<optimize_seconds<<",\"means\":[";
            for(int j=0;j<16;++j){if(j)progress<<',';progress<<means[j];}progress<<"]}\n";progress.flush();require(bool(progress),"V2 update report write failed");
            std::cout<<"CURRENT_V2_UPDATE "<<update<<" N="<<n<<" current="<<measured.passed<<"/16 controls="<<budget.controls<<std::endl;
            if(measured.passed==16) {
                const auto calm=assess_calm(root/"final-calm",budget);require_calm(calm);
                require(calm.keys==long_keys(root/"baseline-calm"/"long"/"sustained-windows.csv"),"V2 calm retained-window keys changed");
                const auto withheld=assess_current(means,withheld_cases(),root/"reserved-withheld",budget);withheld_pass=withheld.passed;
                if(withheld_pass==16){const auto endurance=assess_current(means,endurance_cases(),root/"reserved-endurance",budget);endurance_pass=endurance.passed;}
                qualified=withheld_pass==16&&endurance_pass==2&&measured.passed>before.passed;
                gate=qualified?"qualified_current_swimming_no_adoption":"reserved_current_validation_failed";break;
            }
        }
        if(gate.empty()){require_calm(assess_calm(root/"final-calm",budget));gate="bounded_updates_exhausted_current_not_qualified";}
        steps=data.optimizer->steps();require(steps<=512&&data.backbone->get_params()==resume_state->actor.parameters,"V2 final parameter/counter integrity fault");wall_guard();
    } catch(const std::exception& error){fault=error.what();std::cerr<<fault<<std::endl;}
    if(active&&active->optimizer)steps=active->optimizer->steps();
    result(root/"training-result.json",budget,fault,gate,updates_done,steps,prior.seconds,before.passed,final_pass,withheld_pass,endurance_pass,qualified,false,n,true);
    return fault.empty()?0:3;
}
}
int main(int argc,char** argv) {try {
    require(argc>=3,"Use mode isolated-v2-run-directory [approved-protocol]");const std::string mode=argv[1];const auto root=fs::absolute(argv[2]);
    if(mode=="--numerical"){require(argc==3&&!fs::exists(root),"V2 numerical identity exists");fs::create_directories(root.parent_path());current_study_v2::numerical(root);return 0;}
    if(mode=="--prepare"){require(argc==4,"V2 prepare requires approved protocol");return current_study_v2::prepare(root,fs::absolute(argv[3]));}
    require(argc==3,"Unexpected V2 study argument");
    if(mode=="--preflight")return current_study_v2::preflight(root);
    if(mode=="--baseline")return current_study_v2::baseline(root);
    if(mode=="--profile")return current_study_v2::profile(root);
    if(mode=="--calibrate")return current_study_v2::calibrate(root);
    if(mode=="--train")return current_study_v2::train(root);
    throw std::runtime_error("Unknown V2 study mode");
} catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 2;}}
