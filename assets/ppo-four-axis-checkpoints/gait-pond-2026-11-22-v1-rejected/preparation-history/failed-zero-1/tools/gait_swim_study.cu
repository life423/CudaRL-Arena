// Isolated, finite native PPO gait study. No entry runs implicitly.
#include "motor/joint_motor.h"
namespace gait_study {void physical_failure(const cudarl::joint_motor::World&,const cudarl::physics::BodyState&,const cudarl::physics::BodyState&,int,int);}
#define CUDARL_PHYSICAL_FAILURE_HOOK gait_study::physical_failure
#define CUDARL_REST_HOLD_MEASURED_PHASE_B 1
#define CUDARL_REST_HOLD_MEASURED_BASELINE 1
#define CUDARL_REST_HOLD_RESET_STUDY 1
#include "rest_hold_study.cu"
#include "hold_learned_amplitude_math.h"
#include "motor/gait_policy_v1.h"
#include "motor/gait_task_v1.h"
#include <cstddef>
#include <set>
namespace gait_study {
namespace policy=cudarl::gait_policy_v1;namespace task=cudarl::gait_task_v1;
constexpr int worlds_count=8192,episode_controls=160,head_parameters=8,minibatch_episodes=2048;
static_assert(policy::parameters==head_parameters&&task::observations==41&&task::actions==4,
              "Gait study requires eight policy parameters and the existing four-output body");
constexpr int maximum_updates=64,maximum_adam_steps=1024,maximum_epochs=4;
constexpr int controls_cap=85295280,substeps_cap=852952800,gate_controls=1200;
constexpr double seconds_cap=900;
constexpr uint32_t calibration_seed=0x67a13c85u,training_seed=0xa2d54913u;
constexpr uint32_t protocol_magic=0x31524747,policy_magic=0x31504747,batch_magic=0x31424747,scale_magic=0x31534747,ready_magic=0x31594747;
constexpr uint64_t gib=1024ull*1024*1024,initial_disk_free=35*gib,working_disk_reserve=8*gib,projected_study_bytes=24*gib;
uint64_t study_protocol=0;fs::path study_root,failure_directory;bool failure_saved=false;
float saved_hold_gain=0;std::vector<float> saved_hold_parameters;std::vector<uint8_t> hold_checkpoint_payload;
double prior_seconds=0;
void body_csv(std::ostream& out,const p::BodyState& body) {
    out<<','<<body.position.x<<','<<body.position.y<<','<<body.position.z
       <<','<<body.orientation.x<<','<<body.orientation.y<<','<<body.orientation.z<<','<<body.orientation.w;
    for(float x:body.angle)out<<','<<x;for(float x:body.velocity)out<<','<<x;out<<','<<body.work;
}

void body_csv_header(std::ostream& out) {
    out<<",position_x,position_y,position_z,qx,qy,qz,qw";
    for(int j=0;j<4;++j)out<<",angle"<<j;for(int j=0;j<10;++j)out<<",velocity"<<j;out<<",positive_work_J";
}

void capture_failure(const m::World& world,const p::BodyState& before,const p::BodyState& candidate,
                     int controls,int substeps,const char* file) {
    if(failure_saved)return;
    require(!failure_directory.empty()&&fs::is_directory(failure_directory),"Failure directory unavailable");
    const auto path=failure_directory/file;require(!fs::exists(path),"Failure evidence already exists");
    failure_saved=true;cp::Writer out;out.value(uint32_t{0x31465343});out.value(uint32_t{1});
    out.value(controls);out.value(substeps);out.value(uint32_t{sizeof(m::World)});
    out.value(world);out.value(before);out.value(candidate);cp::write_atomic(path.string(),out.data);
}

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

void physical_failure(const m::World& world,const p::BodyState& before,const p::BodyState& candidate,int controls,int substeps){capture_failure(world,before,candidate,controls,substeps,"first-physical-failure.bin");}
void configure(const fs::path& root,bool copied){study_root=root;configure_append(root,copied);control_cap=controls_cap;substep_cap=substeps_cap;active_wall_cap=seconds_cap-prior_seconds;
 require(resume_state->actor.parameters.size()==2532&&original40.actor.update==336,"Wrong preserved swimming body");
 load_holding_parent(copied?root/"holding-parent.bin":fs::absolute("assets/ppo-four-axis-checkpoints/learned-amplitude-2026-11-22-qualified/qualified-policy-update-7.bin"));}
void time_guard(){require(prior_seconds+std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<seconds_cap,"Combined gait native runtime cap reached");}
void disk_guard(const fs::path& root,bool initial=false){const auto free=fs::space(root).available;
 const uint64_t bytes=uint64_t(worlds_count)*episode_controls*65*4+uint64_t(worlds_count)*1400+1024*1024;
 require(free>=working_disk_reserve+bytes,"Insufficient evidence space before gait allocation/collection");
 if(initial)require(free>=initial_disk_free&&free>=projected_study_bytes+working_disk_reserve,"Gait study requires35GiB free and projected whole-study evidence plus8GiB reserve before physical execution");}
std::vector<uint8_t> protocol_bytes(const fs::path& root){cp::Writer out;out.value(protocol_magic);out.value(uint32_t{1});out.value(file_hash(root/"approved-protocol.md"));
 out.value(cp::checksum(parent_checkpoint_payload));out.value(cp::checksum(hold_checkpoint_payload));
 for(int v:{6,1,3,4,41,4,336,8,worlds_count,episode_controls,80,maximum_updates,maximum_adam_steps,maximum_epochs,minibatch_episodes,controls_cap,substeps_cap,900})out.value(v);
 for(float v:{policy::fixed_sigma,policy::bias_limit,.5f,2.f,.01f,.5f,.2f,.01f,.05f,1.f,10.f,5.f})out.value(v);
 out.value(calibration_seed);out.value(training_seed);out.value(initial_disk_free);out.value(working_disk_reserve);out.value(projected_study_bytes);return out.data;}
void validate_protocol(const fs::path& root){const auto bytes=protocol_bytes(root);require(cp::read((root/"protocol.bin").string())==bytes,"Frozen gait protocol differs");study_protocol=cp::checksum(bytes);}
struct EpisodeCounts {unsigned long long controls=0,substeps=0;double reward_sum=0;task::Measurements gait;int failure_code=0;p::BodyState first_before{},rejected_candidate{};};
struct Data {
 std::unique_ptr<Stream> stream=std::make_unique<Stream>();std::unique_ptr<nn::GpuMlp> backbone=std::make_unique<nn::GpuMlp>(spec(4),worlds_count,seed);std::unique_ptr<nn::GpuAdam> optimizer;
 device_ptr<m::World> worlds{worlds_count};device_ptr<m::ForwardTask> tasks{worlds_count};device_ptr<EpisodeCounts> counts{worlds_count};device_ptr<int> fault{1};
 device_ptr<float> means{8},latent{worlds_count*8},logp{worlds_count},observation{worlds_count*41};
 device_ptr<float> pre{worlds_count*episode_controls*33},post{worlds_count*episode_controls*23},base{worlds_count*episode_controls*4},joint_latent{worlds_count*episode_controls*4},reward{worlds_count*episode_controls};
 device_ptr<float> advantages{worlds_count},mb_latent{minibatch_episodes*8},mb_logp{minibatch_episodes},mb_advantage{minibatch_episodes};
 device_ptr<float> row_gradient{minibatch_episodes*8},row_stats{minibatch_episodes*5},gradient{8},stats{5};
 Data(bool training=false){backbone->set_params(resume_state->actor.parameters);const std::vector<float> zeros(8,0);means.copy_from_host(zeros.data(),zeros.size());if(training){auto c=initial_optimizer_config(true);c.lr=.01f;optimizer=std::make_unique<nn::GpuAdam>(8,c);require(optimizer->steps()==0,"Fresh gait optimizer required");}}
};
__global__ void initialize_episodes(m::World* worlds,m::ForwardTask* tasks,EpisodeCounts* counts,const float* means,float* latent,float* logp,int* fault,uint32_t action_seed,uint32_t counter){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=worlds_count)return;counts[i]=EpisodeCounts{};
 for(int j=0;j<8;++j)latent[i*8+j]=means[j]+policy::fixed_sigma*cudarl::rng::normal(action_seed,counter,uint32_t(i*8+j));
 logp[i]=policy::log_probability(means,latent+i*8);if(!p::finite_number(logp[i])||!task::initialize(i%4,worlds[i],tasks[i]))atomicCAS(fault,0,1);
}
__global__ void observe_episodes(const m::World* worlds,const m::ForwardTask* tasks,float* observation,int* fault){const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=worlds_count||*fault)return;
 if(!m::observe_appended(tasks[i],worlds[i],task::config(),observation+i*41))atomicCAS(fault,0,2);}
CUDARL_HD inline void record_pre(const m::World& world,float* values) {
    const auto& body=world.physical.body;values[0]=body.position.x;values[1]=body.position.y;values[2]=body.position.z;
    values[3]=body.orientation.x;values[4]=body.orientation.y;values[5]=body.orientation.z;values[6]=body.orientation.w;
    for(int j=0;j<4;++j)values[7+j]=body.angle[j];for(int j=0;j<10;++j)values[11+j]=body.velocity[j];
    for(int j=0;j<4;++j){values[21+j]=world.physical.held_targets[j];values[25+j]=world.physical.last_actual_torques[j];values[29+j]=world.reference_rate[j];}
}
__global__ void advance_episodes(m::World* worlds,m::ForwardTask* tasks,EpisodeCounts* counts,
        const float* episode_latent,const float* network_output,float* base,float* joint_latent,
        float* reward,float* pre_state,float* post_state,int* fault,int control) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=worlds_count||*fault)return;
    record_pre(worlds[i],pre_state+i*33);
    float raw[4];policy::swim_latents(episode_latent+i*8,network_output+i*4,raw);
    for(int j=0;j<4;++j){base[i*4+j]=network_output[i*4+j];joint_latent[i*4+j]=raw[j];}
    m::Action action;if(!m::action_from_latent(raw,action)){atomicCAS(fault,0,3);return;}
    const auto before=worlds[i].attempted_substeps;++counts[i].controls;
    p::BodyState pre=worlds[i].physical.body,candidate=pre;
    const bool advanced=cudarl::joint_motor::step_state(worlds[i],action.command,nullptr,&pre,&candidate);
    const auto attempted=worlds[i].attempted_substeps-before;counts[i].substeps+=attempted;
    if(!advanced||attempted!=10){counts[i].first_before=pre;counts[i].rejected_candidate=candidate;
        counts[i].failure_code=worlds[i].failure_code;atomicCAS(fault,0,4);return;}
    const auto interval=m::record_control(tasks[i],worlds[i],task::config());
    if(!interval.accepted||interval.done!=(control==episode_controls)||interval.truncated||
       tasks[i].controls!=control||worlds[i].physical.control_steps!=uint64_t(control)||
       !counts[i].gait.record(control,worlds[i].physical.body)) {
        counts[i].first_before=pre;counts[i].rejected_candidate=worlds[i].physical.body;atomicCAS(fault,0,5);return;
    }
    reward[i]=interval.reward;counts[i].reward_sum+=double(reward[i]);
    float scratch[33];record_pre(worlds[i],scratch);for(int j=0;j<21;++j)post_state[i*23+j]=scratch[j];
    post_state[i*23+21]=worlds[i].physical.body.work;post_state[i*23+22]=worlds[i].physical.mean_squared_torque;
}
__global__ void episode_loss(int n,const float* means,const float* latent,const float* oldp,
        const float* advantage,float* gradient,float* stats) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto terms=policy::loss(means,latent+i*8,oldp[i],advantage[i],1.f/n,gradient+i*8);
    stats[i*5]=terms.pg_loss;stats[i*5+1]=terms.v_loss;stats[i*5+2]=terms.entropy;
    stats[i*5+3]=terms.approx_kl;stats[i*5+4]=terms.clipped;
}
struct Collected {
    int purpose=0,index=0,ticks=0,fault=0;
    std::vector<float> means,latent,logp,returns,pre,post,base,joint_latent,reward;
    std::vector<EpisodeCounts> counts;std::vector<m::World> worlds;std::vector<m::ForwardTask> tasks;
    double probability_error=0,decode_error=0,collect_seconds=0,write_seconds=0;
};
void save_batch(const Collected& batch,const fs::path& directory) {
    cp::Writer meta;meta.value(batch_magic);meta.value(uint32_t{1});meta.value(study_protocol);
    for(int value:{worlds_count,episode_controls,batch.ticks,batch.purpose,batch.index,batch.fault})meta.value(value);
    meta.array(batch.means);meta.array(batch.latent);meta.array(batch.logp);meta.array(batch.returns);
    meta.value(uint32_t{sizeof(EpisodeCounts)});meta.array(batch.counts);
    meta.value(uint32_t{sizeof(m::World)});meta.array(batch.worlds);
    meta.value(uint32_t{sizeof(m::ForwardTask)});meta.array(batch.tasks);
    cp::write_atomic((directory/"meta.bin").string(),meta.data);const auto binding=cp::checksum(meta.data);
    const char* names[5]={"pre-state","post-state","base","joint-latent","reward"};
    const int columns[5]={33,23,4,4,1};const std::vector<float>* arrays[5]={&batch.pre,&batch.post,&batch.base,&batch.joint_latent,&batch.reward};
    std::ofstream manifest(directory/"manifest.json");manifest<<"{\"schema\":1,\"meta_checksum\":"<<binding
      <<",\"worlds\":"<<worlds_count<<",\"ticks\":"<<batch.ticks<<",\"purpose\":"<<batch.purpose<<",\"parts\":[";
    for(int field=0;field<5;++field){cp::Writer part;part.value(batch_magic);part.value(uint32_t{1});part.value(field);
        part.value(binding);part.value(worlds_count);part.value(batch.ticks);part.value(columns[field]);part.array(*arrays[field]);
        require(part.data.size()<=512*1024*1024,"Gait evidence part exceeds512MiB");
        const std::string name=std::string(names[field])+".bin";cp::write_atomic((directory/name).string(),part.data);
        if(field)manifest<<',';manifest<<"{\"field\":"<<field<<",\"name\":"<<std::quoted(name)<<",\"columns\":"<<columns[field]
          <<",\"payload_bytes\":"<<part.data.size()<<",\"payload_checksum\":"<<cp::checksum(part.data)<<'}';}
    manifest<<"]}\n";manifest.close();require(bool(manifest),"Cannot save gait part manifest");
    std::ofstream csv(directory/"episodes.csv");csv<<std::setprecision(12)
      <<"episode,case,pose,old_logp,reward_sum,return,controls,substeps,failure_code,movement_passed,mean_pitch1_deg,mean_pitch3_deg,mean_tail_bend_deg,centered_joint_RMS_deg";
    for(int j=0;j<8;++j)csv<<",latent"<<j;csv<<'\n';
    for(int i=0;i<worlds_count;++i){const auto& c=batch.counts[i];csv<<i<<','<<i%4<<','<<i%4<<','<<batch.logp[i]<<','<<c.reward_sum<<',';
        if(!batch.returns.empty())csv<<batch.returns[i];csv<<','<<c.controls<<','<<c.substeps<<','<<c.failure_code<<','
          <<(batch.tasks[i].outcome==m::Outcome::Success)<<','<<c.gait.mean_pitch(0)<<','<<c.gait.mean_pitch(1)<<','
          <<c.gait.mean_bend()<<','<<c.gait.centered_joint_rms_degrees();for(int j=0;j<8;++j)csv<<','<<batch.latent[i*8+j];csv<<'\n';}
    csv.close();require(bool(csv),"Cannot save gait episode evidence");
}
Collected collect(Data& data,Budget& budget,const fs::path& directory,int purpose,int index) {
    time_guard();disk_guard(study_root);require(fs::create_directory(directory),"Gait batch identity already exists");
    require(budget.controls+worlds_count*160<=controls_cap&&budget.substeps+worlds_count*1600<=substeps_cap,"Next gait batch exceeds closed cap");
    const auto begin=std::chrono::steady_clock::now();const auto stream=data.stream->value;
    CUDA_CHECK_THROW(cudaMemsetAsync(data.fault.get(),0,sizeof(int),stream));
    for(auto* a:{&data.pre,&data.post,&data.base,&data.joint_latent,&data.reward})CUDA_CHECK_THROW(cudaMemsetAsync(a->get(),0,a->size_bytes(),stream));
    initialize_episodes<<<64,128,0,stream>>>(data.worlds.get(),data.tasks.get(),data.counts.get(),data.means.get(),data.latent.get(),data.logp.get(),data.fault.get(),purpose==0?calibration_seed:training_seed,uint32_t(index));
    Collected batch;batch.purpose=purpose;batch.index=index;
    for(int control=1;control<=160;++control){time_guard();const size_t offset=size_t(control-1)*worlds_count;
        observe_episodes<<<64,128,0,stream>>>(data.worlds.get(),data.tasks.get(),data.observation.get(),data.fault.get());
        data.backbone->forward(data.observation.get(),worlds_count,stream);
        advance_episodes<<<64,128,0,stream>>>(data.worlds.get(),data.tasks.get(),data.counts.get(),data.latent.get(),
          data.backbone->output(),data.base.get()+offset*4,data.joint_latent.get()+offset*4,data.reward.get()+offset,
          data.pre.get()+offset*33,data.post.get()+offset*23,data.fault.get(),control);
        CUDA_CHECK_THROW(cudaGetLastError());batch.ticks=control;
        batch.fault=download(data.fault.get(),1,stream)[0];if(batch.fault)break;
    }
    batch.means=download(data.means.get(),8,stream);batch.latent=download(data.latent.get(),worlds_count*8,stream);
    batch.logp=download(data.logp.get(),worlds_count,stream);batch.counts=download(data.counts.get(),worlds_count,stream);
    batch.worlds=download(data.worlds.get(),worlds_count,stream);batch.tasks=download(data.tasks.get(),worlds_count,stream);
    const size_t rows_saved=size_t(batch.ticks)*worlds_count;
    batch.pre=download(data.pre.get(),rows_saved*33,stream);batch.post=download(data.post.get(),rows_saved*23,stream);
    batch.base=download(data.base.get(),rows_saved*4,stream);batch.joint_latent=download(data.joint_latent.get(),rows_saved*4,stream);
    batch.reward=download(data.reward.get(),rows_saved,stream);
    for(const auto& count:batch.counts){budget.controls+=int(count.controls);budget.substeps+=int(count.substeps);}
    require(budget.controls<=controls_cap&&budget.substeps<=substeps_cap,"Actual gait physical cap exceeded");
    if(!batch.fault){require(batch.ticks==160,"Gait batch is incomplete");batch.returns.resize(worlds_count);
        double means[8];for(int j=0;j<8;++j)means[j]=batch.means[j];
        for(int i=0;i<worlds_count;++i){const auto& c=batch.counts[i];require(c.controls==160&&c.substeps==1600&&c.gait.samples==80&&c.failure_code==0,"Gait episode count/schema mismatch");
            const bool movement=batch.tasks[i].outcome==m::Outcome::Success;
            batch.returns[i]=float(c.reward_sum/8.-c.gait.cost()-(movement?0.:task::movement_failure_cost));
            double u[8];for(int j=0;j<8;++j)u[j]=batch.latent[i*8+j];
            batch.probability_error=std::max(batch.probability_error,std::abs(policy::log_probability(means,u)-batch.logp[i]));
            for(int control=0;control<160;++control){const size_t row=size_t(control)*worlds_count+i;float expected[4];
                policy::swim_latents(batch.latent.data()+i*8,batch.base.data()+row*4,expected);
                for(int j=0;j<4;++j)batch.decode_error=std::max(batch.decode_error,double(std::abs(expected[j]-batch.joint_latent[row*4+j])));}
        }
        finite(batch.returns);require(batch.probability_error<1e-4&&batch.decode_error<2e-6,"Gait latent probability/decode mismatch");}
    batch.collect_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    const auto write_begin=std::chrono::steady_clock::now();save_batch(batch,directory);
    batch.write_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-write_begin).count();
    std::ofstream timing(directory/"timing.json");timing<<std::setprecision(17)<<"{\"worlds\":"<<worlds_count<<",\"ticks\":"<<batch.ticks
      <<",\"collection_readback_validation_seconds\":"<<batch.collect_seconds<<",\"evidence_seconds\":"<<batch.write_seconds
      <<",\"controls\":"<<worlds_count*batch.ticks<<",\"actual_controls\":";
    unsigned long long attempted=0;for(const auto& c:batch.counts)attempted+=c.controls;
    timing<<attempted<<",\"fault\":"<<batch.fault<<",\"optimizer_steps\":0}\n";timing.close();
    require(bool(timing),"Cannot save gait timing");require(batch.fault==0,"Unsafe/invalid gait collection rejected before advantage or optimizer");
    return batch;
}
struct Scale {double value=0;double means[4]{};};
Scale fit_scale(const std::vector<float>& returns){require(returns.size()==worlds_count,"Wrong gait calibration size");Scale scale;
    for(int i=0;i<worlds_count;++i)scale.means[i%4]+=returns[i];for(double& value:scale.means)value/=worlds_count/4;
    double square=0;for(int i=0;i<worlds_count;++i){const double e=returns[i]-scale.means[i%4];square+=e*e;}
    scale.value=sqrt(square/worlds_count);require(std::isfinite(scale.value)&&scale.value>0,"Invalid independent gait calibration RMS");return scale;}
std::vector<float> leave_one_out(const std::vector<float>& returns,double scale){require(returns.size()==worlds_count&&std::isfinite(scale)&&scale>0,"Invalid gait LOO inputs");
    double sums[4]{};for(int i=0;i<worlds_count;++i)sums[i%4]+=returns[i];std::vector<float> result(worlds_count);
    for(int i=0;i<worlds_count;++i)result[i]=float((returns[i]-(sums[i%4]-returns[i])/(worlds_count/4-1))/scale);return result;}
void save_policy(Data& data,const fs::path& directory,int update){const auto path=directory/("gait-policy-update-"+std::to_string(update)+".bin");require(!fs::exists(path),"Gait checkpoint identity exists");
    cp::Writer out,adam;if(data.optimizer)data.optimizer->save(adam);else{auto cfg=initial_optimizer_config(true);cfg.lr=.01f;nn::GpuAdam fresh(8,cfg);fresh.save(adam);}
    out.value(policy_magic);out.value(uint32_t{1});out.value(study_protocol);out.value(cp::checksum(parent_checkpoint_payload));out.value(cp::checksum(hold_checkpoint_payload));
    for(uint32_t v:{6u,1u,3u,4u,41u,4u,336u,8u})out.value(v);
    out.value(worlds_count);out.value(update);out.value(data.optimizer?data.optimizer->steps():0);
    out.array(data.backbone->get_params());out.array(saved_hold_parameters);out.value(saved_hold_gain);
    out.array(download(data.means.get(),8,data.stream->value));out.value(policy::fixed_sigma);out.value(policy::bias_limit);out.value(2.f);out.array(adam.data);
    cp::write_atomic(path.string(),out.data);require(cp::read(path.string())==out.data,"Gait checkpoint exact roundtrip differs");
}
double optimize(Data& data,const Collected& batch,const Scale& scale,int update,const fs::path& directory){
    require(batch.fault==0&&batch.ticks==160&&batch.purpose==1,"Only complete safe fresh gait data can reach PPO");
    const auto advantage=leave_one_out(batch.returns,scale.value);data.advantages.copy_from_host(advantage.data(),advantage.size());
    cp::Writer proof;proof.value(uint32_t{0x31414747});proof.value(uint32_t{1});proof.value(update);proof.value(worlds_count);proof.value(scale.value);
    proof.array(batch.returns);proof.array(advantage);cp::write_atomic((directory/("advantages-"+std::to_string(update)+".bin")).string(),proof.data);
    const auto stream=data.stream->value;const int before=data.optimizer->steps();double kl=0;
    for(int epoch=0;epoch<4;++epoch){CUDA_CHECK_THROW(cudaMemsetAsync(data.stats.get(),0,5*sizeof(float),stream));
        const auto order=rl::make_permutation(worlds_count,0x2c64931u+uint32_t(update*4+epoch));
        for(int first=0;first<worlds_count;first+=minibatch_episodes){time_guard();
            rl::gather_rows(order,first,minibatch_episodes,8,data.latent.get(),data.mb_latent.get(),stream);
            rl::gather_rows(order,first,minibatch_episodes,1,data.logp.get(),data.mb_logp.get(),stream);
            rl::gather_rows(order,first,minibatch_episodes,1,data.advantages.get(),data.mb_advantage.get(),stream);
            episode_loss<<<16,128,0,stream>>>(minibatch_episodes,data.means.get(),data.mb_latent.get(),data.mb_logp.get(),data.mb_advantage.get(),data.row_gradient.get(),data.row_stats.get());
            CUDA_CHECK_THROW(cudaMemsetAsync(data.gradient.get(),0,8*sizeof(float),stream));
            rl::sum_columns(data.row_gradient.get(),minibatch_episodes,8,data.gradient.get(),stream);
            rl::sum_columns(data.row_stats.get(),minibatch_episodes,5,data.stats.get(),stream);
            finite(download(data.gradient.get(),8,stream));data.optimizer->step(data.means.get(),data.gradient.get(),stream);finite(download(data.means.get(),8,stream));}
        const auto stats=download(data.stats.get(),5,stream);finite(stats);kl=stats[3]/worlds_count;if(kl>.01)break;}
    require(data.optimizer->steps()-before>=4&&data.optimizer->steps()-before<=16&&data.optimizer->steps()<=maximum_adam_steps,"Gait Adam cap exceeded");
    require(data.backbone->get_params()==resume_state->actor.parameters,"Frozen gait backbone changed");return kl;
}
struct SwimCase {int pose=0,controls=160;float offset=0;};
struct SwimResult {int movement_passed=0,gait_passed=0,completed=0,window_passed=0,window_total=0,gait_window_passed=0,gait_window_total=0,recovery_passed=0;double mean_cost=0;};
void state_header(std::ostream& out){out<<"case,control";for(int j=0;j<33;++j)out<<",pre"<<j;
    for(int j=0;j<4;++j)out<<",base"<<j;for(int j=0;j<4;++j)out<<",latent"<<j;
    for(int j=0;j<23;++j)out<<",post"<<j;out<<",reward\n";}
void state_row(std::ostream& out,int id,int control,const float* pre,const float* base,const float* latent,const m::World& world,float reward){
    out<<id<<','<<control;for(int j=0;j<33;++j)out<<','<<pre[j];for(int j=0;j<4;++j)out<<','<<base[j];for(int j=0;j<4;++j)out<<','<<latent[j];
    float post[33];record_pre(world,post);for(int j=0;j<21;++j)out<<','<<post[j];out<<','<<world.physical.body.work<<','<<world.physical.mean_squared_torque<<','<<reward<<'\n';}
void gait_columns(std::ostream& out){out<<",mean_pitch1_deg,mean_pitch3_deg,mean_tail_vertical_bend_deg,centered_joint_RMS_deg,gait_passed";}
void gait_values(std::ostream& out,const task::Measurements& score){out<<','<<score.mean_pitch(0)<<','<<score.mean_pitch(1)<<','<<score.mean_bend()<<','<<score.centered_joint_rms_degrees()<<','<<score.passed();}
SwimResult assess_swim(const std::vector<float>& means,const std::vector<SwimCase>& cases,const fs::path& directory,Budget& budget,bool long_windows=false,bool recovery=false){
    require(fs::create_directory(directory),"Gait swimming assessment identity exists");nn::CpuMlp<float> body(spec(4));body.params()=resume_state->actor.parameters;
    std::ofstream trace(directory/"controls.csv"),summary(directory/"summary.csv"),windows(directory/"windows.csv"),initial(directory/"initial.csv");
    trace<<std::setprecision(12);summary<<std::setprecision(12);windows<<std::setprecision(12);initial<<std::setprecision(12);state_header(trace);
    summary<<"case,pose,offset_mm,controls,movement_passed,steady_mm_s,RMSE_mm_s,lateral_mm,depth_mm,tilt_deg,path_deg,work_J_per_m,return,recovery_passed";gait_columns(summary);summary<<'\n';
    windows<<"case,pose,end_s,steady_mm_s,RMSE_mm_s,lateral_mm,depth_mm,tilt_deg,path_deg,work_J_per_m,movement_passed,gait_scored";gait_columns(windows);windows<<'\n';
    initial<<"case,pose,offset_mm,controls,arena_half_width_m,arena_half_depth_m,start_com_x,start_com_y,start_com_z,heading_x,heading_y,heading_z\n";
    SwimResult result;
    for(int id=0;id<int(cases.size());++id){const auto c=cases[id];auto world=m::initial_world(c.pose);world.arena=c.controls==6000?cudarl::contact::Arena{3.f,.4f}:m::long_swim_arena();
        auto config=task::config(c.controls);m::ForwardTask t;require(m::begin_forward(world,config,t)&&m::offset_episode_start(world,t,config,c.offset),"Gait swimming reset rejected");
        initial<<id<<','<<c.pose<<','<<c.offset*1000<<','<<c.controls<<','<<world.arena.half_width_m<<','<<world.arena.half_depth_m<<','<<t.start_com_model.x<<','<<t.start_com_model.y<<','<<t.start_com_model.z<<','<<t.request.heading.x<<','<<t.request.heading.y<<','<<t.request.heading.z<<'\n';
        task::Measurements gait,window_gait;double sum_reward=0,error2=0;auto origin=cudarl::motor::motion(world.physical.body).com_model;float work0=world.physical.body.work;bool recovered=true;
        for(int control=1;control<=c.controls;++control){time_guard();float observation[41],pre[33],raw[4];record_pre(world,pre);
            require(m::observe_appended(t,world,config,observation),"Gait CPU observation rejected");body.forward(observation,1);policy::swim_latents(means.data(),body.output(),raw);
            m::Action action;require(m::action_from_latent(raw,action),"Gait CPU action rejected");budget.step(world,action.command);
            const auto interval=m::record_control(t,world,config);require(interval.accepted&&gait.record(control,world.physical.body)&&window_gait.record(control,world.physical.body),"Gait CPU task/measurement rejected");
            sum_reward+=interval.reward;state_row(trace,id,control,pre,body.output(),raw,world,interval.reward);
            if(control>40)error2+=interval.metrics.speed_error_m_s*interval.metrics.speed_error_m_s;
            if(control==40){origin=cudarl::motor::motion(world.physical.body).com_model;work0=world.physical.body.work;}
            if(recovery&&c.offset!=0&&control>c.controls-160)recovered=recovered&&interval.metrics.depth_m<fabsf(c.offset)*.5f;
            if(long_windows&&control%80==0){const auto center=cudarl::motor::motion(world.physical.body).com_model;
                const auto delta=(center-origin)*p::scale;const float forward=p::dot(delta,t.request.heading);const auto side=delta-t.request.heading*forward;
                const int samples=control==80?40:80;const float speed=forward/(samples*p::dt),rmse=sqrtf(float(error2/samples)),
                  path=atan2f(p::norm({side.x,0,side.z}),forward)*180/cudarl::movement::pi,work=(world.physical.body.work-work0)/fmaxf(forward,1e-9f);
                const bool pass=speed>=config.minimum_speed_m_s&&speed<=config.maximum_speed_m_s&&rmse<=config.speed_rmse_limit&&
                  t.maximum_lateral<=config.lateral_limit_m&&t.maximum_depth<=config.depth_limit_m&&t.maximum_tilt<=config.tilt_limit_deg&&
                  path<=config.path_angle_limit_deg&&work<=config.work_per_distance_limit;
                ++result.window_total;result.window_passed+=pass;
                if(control>=160){++result.gait_window_total;result.gait_window_passed+=window_gait.passed();}
                windows<<id<<','<<c.pose<<','<<control*.05<<','<<speed*1000<<','<<rmse*1000<<','<<t.maximum_lateral*1000<<','<<t.maximum_depth*1000<<','<<t.maximum_tilt<<','<<path<<','<<work<<','<<pass<<','<<(control>=160);gait_values(windows,window_gait);windows<<'\n';
                origin=center;work0=world.physical.body.work;error2=0;window_gait={};}
        }
        const bool movement=t.outcome==m::Outcome::Success;result.movement_passed+=movement;result.gait_passed+=gait.passed();++result.completed;
        result.recovery_passed+=movement&&recovered;const double episode_return=sum_reward/(c.controls*.05)-gait.cost()-(movement?0:task::movement_failure_cost);result.mean_cost-=episode_return/cases.size();
        const auto last=m::measure(t,world);const float work=(world.physical.body.work-t.start_work_J)/fmaxf(last.forward_m,1e-9f);
        summary<<id<<','<<c.pose<<','<<c.offset*1000<<','<<c.controls<<','<<movement<<','<<t.steady_speed_m_s*1000<<','<<t.speed_rmse*1000<<','<<t.maximum_lateral*1000<<','<<t.maximum_depth*1000<<','<<t.maximum_tilt<<','<<t.path_angle_deg<<','<<work<<','<<episode_return<<','<<(movement&&recovered);gait_values(summary,gait);summary<<'\n';
    }
    for(auto* out:{&trace,&summary,&windows,&initial}){out->close();require(bool(*out),"Gait CPU swim evidence write failed");}return result;
}
class HoldScore {
    int total_,next_=0;bool valid_=true;float peak_[5]{},forward_window_=0,excursion_=0;double square_speed_=0;
public:
    explicit HoldScore(int total):total_(total){valid_=total>=160;}
    bool record(int control,const m::Metrics& z){if(!valid_||control!=next_||control>total_){valid_=false;return false;}
        const float v[5]={p::norm(z.velocity_m_s),p::norm(z.hold_error_m),z.depth_m,z.lateral_m,z.tilt_deg};
        for(int j=0;j<5;++j){if(!std::isfinite(v[j])||v[j]<0){valid_=false;return false;}peak_[j]=fmaxf(peak_[j],v[j]);}
        if(!std::isfinite(z.forward_m)){valid_=false;return false;}if(control)square_speed_+=double(v[0])*v[0];
        if(control==total_-160)forward_window_=z.forward_m;if(control>=total_-160)excursion_=fmaxf(excursion_,fabsf(z.forward_m-forward_window_));++next_;return true;}
    bool passed()const{return valid_&&next_==total_+1&&peak_[0]<=.0005f&&peak_[1]<=.002f&&peak_[2]<=.001f&&peak_[3]<=.002f&&peak_[4]<=10.f&&excursion_<=.002f;}
    const float* peaks()const{return peak_;}float excursion()const{return excursion_;}
};
int assess_holds(const std::vector<float>& means,const std::vector<SwimCase>& cases,const fs::path& directory,Budget& budget){
    require(fs::create_directory(directory),"Gait hold assessment identity exists");nn::CpuMlp<float> body(spec(4));body.params()=resume_state->actor.parameters;
    std::ofstream trace(directory/"controls.csv"),summary(directory/"summary.csv");trace<<std::setprecision(12);summary<<std::setprecision(12);state_header(trace);
    summary<<"case,pose,offset_mm,controls,passed,max_speed_mm_s,max_anchor_mm,max_depth_mm,max_lane_mm,max_tilt_deg,final8_excursion_mm\n";int passed=0;
    for(int id=0;id<int(cases.size());++id){const auto c=cases[id];auto world=m::initial_world(c.pose);world.arena=m::long_swim_arena();auto config=task::config(c.controls);
        config.target_speed_m_s=0;config.stationary_request=true;config.position_hold=true;m::ForwardTask t;
        require(m::begin_forward(world,config,t)&&m::offset_episode_start(world,t,config,c.offset)&&m::update_velocity_request(world,config,t),"Gait hold reset rejected");
        HoldScore score(c.controls);require(score.record(0,m::measure(t,world)),"Gait hold reset scoring rejected");
        for(int control=1;control<=c.controls;++control){time_guard();float observation[41],pre[33],raw[4];record_pre(world,pre);
            require(m::update_velocity_request(world,config,t)&&m::observe_appended(t,world,config,observation),"Gait hold observation rejected");
            body.forward(observation,1);policy::motor_latents(true,means.data(),body.output(),saved_hold_gain,raw);m::Action action;
            require(m::action_from_latent(raw,action),"Gait hold action rejected");budget.step(world,action.command);
            const auto interval=m::record_hold_control(t,world,config);require(interval.accepted&&score.record(control,interval.metrics),"Gait hold score rejected");
            state_row(trace,id,control,pre,body.output(),raw,world,interval.reward);}
        passed+=score.passed();summary<<id<<','<<c.pose<<','<<c.offset*1000<<','<<c.controls<<','<<score.passed();for(int j=0;j<5;++j)summary<<','<<score.peaks()[j]*(j==4?1:1000);summary<<','<<score.excursion()*1000<<'\n';}
    trace.close();summary.close();require(bool(trace)&&bool(summary),"Gait hold evidence write failed");return passed;
}
std::vector<SwimCase> short_cases(){return {{0,160,0},{1,160,0},{2,160,0},{3,160,0}};}
std::vector<SwimCase> original_holds(){std::vector<SwimCase> out;for(float offset:{0.f,-.0005f,.0005f})for(int pose:{0,3})out.push_back({pose,1280,offset});return out;}
std::vector<SwimCase> withheld_holds(){std::vector<SwimCase> out;for(float offset:{0.f,-.00025f,.00025f})for(int pose:{1,2})out.push_back({pose,1280,offset});return out;}
bool retained_windows(const fs::path& csv,int last){std::ifstream input(csv);std::string line;std::getline(input,line);std::set<std::pair<int,int>> passed;
    while(std::getline(input,line)){std::stringstream row(line);std::string value;std::vector<double> values;while(std::getline(row,value,','))values.push_back(std::stod(value));
        require(values.size()==17,"Gait long-window schema differs");if(values[10]==1)passed.emplace(int(values[1]),int(values[2]));}
    for(int pose:{0,3})for(int end=8;end<=last;end+=4)if(!passed.count({pose,end}))return false;return true;}
void identity_check(const std::vector<float>& means){require(means.size()==8,"Wrong gait parameter count");finite(means);
    for(int row=0;row<40;++row){float base[4],raw[4],expected[4],zero[8]{};for(int j=0;j<4;++j)base[j]=float(row-20)*.1f+float(j)*.03f;
        policy::swim_latents(zero,base,raw);for(int j=0;j<4;++j)require(raw[j]==base[j],"Identity gait changed original output");
        policy::motor_latents(true,means.data(),base,saved_hold_gain,raw);for(int j=0;j<4;++j)expected[j]=saved_hold_gain*base[j];
        for(int j=0;j<4;++j)require(raw[j]==expected[j],"Untrained holding branch changed");}}
void write_layout(const fs::path& root){std::ofstream out(root/"layout.json");out<<"{\"episode_counts\":{\"size\":"<<sizeof(EpisodeCounts)
      <<",\"controls\":"<<offsetof(EpisodeCounts,controls)<<",\"substeps\":"<<offsetof(EpisodeCounts,substeps)<<",\"reward_sum\":"<<offsetof(EpisodeCounts,reward_sum)
      <<",\"gait\":"<<offsetof(EpisodeCounts,gait)<<",\"failure_code\":"<<offsetof(EpisodeCounts,failure_code)<<",\"first_before\":"<<offsetof(EpisodeCounts,first_before)
      <<",\"rejected_candidate\":"<<offsetof(EpisodeCounts,rejected_candidate)<<"},\"gait\":{\"size\":"<<sizeof(task::Measurements)
      <<",\"pitch_sum\":"<<offsetof(task::Measurements,pitch_sum)<<",\"tail_bend_sum\":"<<offsetof(task::Measurements,tail_bend_sum)
      <<",\"pitch_square_sum\":"<<offsetof(task::Measurements,pitch_square_sum)<<",\"tail_bend_square_sum\":"<<offsetof(task::Measurements,tail_bend_square_sum)
      <<",\"angle_sum\":"<<offsetof(task::Measurements,angle_sum)<<",\"angle_square_sum\":"<<offsetof(task::Measurements,angle_square_sum)
      <<",\"pitch_peak\":"<<offsetof(task::Measurements,pitch_peak)<<",\"tail_bend_peak\":"<<offsetof(task::Measurements,tail_bend_peak)
      <<",\"yaw_min\":"<<offsetof(task::Measurements,yaw_min)<<",\"yaw_max\":"<<offsetof(task::Measurements,yaw_max)<<",\"samples\":"<<offsetof(task::Measurements,samples)
      <<"},\"world_size\":"<<sizeof(m::World)<<",\"task_size\":"<<sizeof(m::ForwardTask)<<",\"body_size\":"<<sizeof(p::BodyState)<<",\"pre_floats\":33,\"post_floats\":23,\"trace_floats\":65}\n";require(bool(out),"Cannot save gait ABI");}
__global__ void decode_fixture(float* output,float hold_gain){const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=96)return;
    float u[8],base[4],swim[4],hold[4];const float values[5]={-80.f,-.35f,0.f,.35f,80.f};
    for(int j=0;j<8;++j)u[j]=i==0?0.f:values[(i+j)%5];for(int j=0;j<4;++j)base[j]=float((i+j*3)%23-11)*.17f;
    policy::swim_latents(u,base,swim);policy::motor_latents(true,u,base,hold_gain,hold);
    for(int j=0;j<8;++j)output[i*20+j]=u[j];for(int j=0;j<4;++j){output[i*20+8+j]=base[j];output[i*20+12+j]=swim[j];output[i*20+16+j]=hold[j];}}
void numerical(const fs::path& directory){require(fs::create_directory(directory),"Gait zero math identity exists");identity_check(std::vector<float>(8,1e6f));
    Stream decode_stream;device_ptr<float> decoded(96*20);decode_fixture<<<1,128,0,decode_stream.value>>>(decoded.get(),saved_hold_gain);
    const auto decode=download(decoded.get(),96*20,decode_stream.value);double maximum_decode=0;
    for(int i=0;i<96;++i){double u[8],base[4],swim[4];for(int j=0;j<8;++j)u[j]=decode[i*20+j];for(int j=0;j<4;++j)base[j]=decode[i*20+8+j];policy::swim_latents(u,base,swim);
        for(int j=0;j<4;++j){maximum_decode=std::max(maximum_decode,std::abs(swim[j]-decode[i*20+12+j]));
            require(decode[i*20+16+j]==saved_hold_gain*decode[i*20+8+j],"Actual CUDA holding branch changed exact saved output");
            if(i==0)require(decode[i*20+12+j]==decode[i*20+8+j],"Actual CUDA identity changed original output");}}
    require(maximum_decode<2e-6,"Gait CPU/CUDA bounded decode mismatch");cp::Writer decoded_proof;decoded_proof.value(uint32_t{0x31444747});decoded_proof.value(uint32_t{1});decoded_proof.value(saved_hold_gain);decoded_proof.array(decode);cp::write_atomic((directory/"decode-zero.bin").string(),decoded_proof.data);
    Stream stream;device_ptr<float> dmean(8),dlatent(minibatch_episodes*8),dold(minibatch_episodes),dadv(minibatch_episodes),dgradient(minibatch_episodes*8),dstats(minibatch_episodes*5);
    std::vector<float> means(8),latent(minibatch_episodes*8),oldp(minibatch_episodes),adv(minibatch_episodes);
    for(int j=0;j<8;++j)means[j]=float(j-4)*.07f;
    for(int i=0;i<minibatch_episodes;++i){for(int j=0;j<8;++j)latent[i*8+j]=means[j]+float((i*7+j*3)%19-9)*.035f;
        const float ratios[3]={.7f,1.f,1.3f};oldp[i]=policy::log_probability(means.data(),latent.data()+i*8)-logf(ratios[i%3]);adv[i]=(i%2?.7f:-.8f);}
    dmean.copy_from_host(means.data(),means.size());dlatent.copy_from_host(latent.data(),latent.size());dold.copy_from_host(oldp.data(),oldp.size());dadv.copy_from_host(adv.data(),adv.size());
    episode_loss<<<16,128,0,stream.value>>>(minibatch_episodes,dmean.get(),dlatent.get(),dold.get(),dadv.get(),dgradient.get(),dstats.get());
    const auto gradients=download(dgradient.get(),minibatch_episodes*8,stream.value),stats=download(dstats.get(),minibatch_episodes*5,stream.value);
    double maximum_gradient=0,maximum_stats=0,maximum_fd=0;double parameter[8];for(int j=0;j<8;++j)parameter[j]=means[j];
    for(int i=0;i<minibatch_episodes;++i){double u[8],gradient[8];for(int j=0;j<8;++j)u[j]=latent[i*8+j];
        const auto expected=policy::loss(parameter,u,double(oldp[i]),double(adv[i]),1./minibatch_episodes,gradient);
        const double values[5]={expected.pg_loss,expected.v_loss,expected.entropy,expected.approx_kl,expected.clipped};
        for(int j=0;j<8;++j)maximum_gradient=std::max(maximum_gradient,std::abs(gradient[j]-gradients[i*8+j]));
        for(int j=0;j<5;++j)maximum_stats=std::max(maximum_stats,std::abs(values[j]-stats[i*5+j]));
        if(i<24)for(int j=0;j<8;++j){const double original=parameter[j],eps=1e-6;double scratch[8];parameter[j]=original+eps;
            const auto plus=policy::loss(parameter,u,double(oldp[i]),double(adv[i]),1./minibatch_episodes,scratch);parameter[j]=original-eps;
            const auto minus=policy::loss(parameter,u,double(oldp[i]),double(adv[i]),1./minibatch_episodes,scratch);parameter[j]=original;
            maximum_fd=std::max(maximum_fd,std::abs((plus.pg_loss-minus.pg_loss)/(2*eps)-gradient[j]));}}
    require(maximum_gradient<2e-7&&maximum_stats<2e-4&&maximum_fd<2e-7,"Gait bulk PPO/finite difference math failed");
    std::vector<float> returns(worlds_count);for(int i=0;i<worlds_count;++i)returns[i]=float(i%4)+float(i/4)*.0001f;
    const auto before=leave_one_out(returns,2.);const int selected=19;const double a=returns[selected]-before[selected]*2;returns[selected]+=7;
    const auto after=leave_one_out(returns,2.);require(std::abs(a-(returns[selected]-after[selected]*2))<2e-5,"Gait own return entered LOO baseline");
    auto cfg=initial_optimizer_config(true);cfg.lr=.01f;nn::GpuAdam first(8,cfg),second(8,cfg);cp::Writer one,two;first.save(one);cp::Reader in{one.data};second.load(in);in.finish();second.save(two);
    require(first.steps()==0&&second.steps()==0&&one.data==two.data,"Fresh gait Adam roundtrip failed");
    cp::Writer evidence;evidence.value(uint32_t{0x314d4747});evidence.value(uint32_t{1});evidence.array(means);evidence.array(latent);evidence.array(oldp);evidence.array(adv);evidence.array(gradients);evidence.array(stats);cp::write_atomic((directory/"bulk-gradient.bin").string(),evidence.data);
    std::ofstream out(directory/"math-zero.json");out<<std::setprecision(17)<<"{\"passed\":true,\"actual_bulk_rows\":"<<minibatch_episodes<<",\"max_gradient_error\":"<<maximum_gradient<<",\"max_stats_error\":"<<maximum_stats<<",\"max_FD_error\":"<<maximum_fd<<",\"CUDA_decode_cases\":96,\"max_decode_error\":"<<maximum_decode<<",\"holding_exact\":true,\"identity_exact\":true,\"physical_controls\":0,\"optimizer_steps\":0}\n";
}
void bulk_policy_zero(Data& data,const std::vector<float>& latent,const std::vector<float>& old_logp,const fs::path& directory) {
    const int n=worlds_count,mb=minibatch_episodes;const auto stream=data.stream->value;std::vector<float> advantage(n);
    for(int i=0;i<n;++i)advantage[i]=.25f+float(i%19-9)*.07f;
    data.advantages.copy_from_host(advantage.data(),n);const auto permutation=rl::make_permutation(n,0x7138b2a9u);
    std::vector<int> seen(n,0);double maximum_row=0,maximum_sum=0,maximum_stat=0,maximum_last=0;int checked=0,last_rows=0;
    for(int first=0;first<n;first+=mb) {
        rl::gather_rows(permutation,first,mb,head_parameters,data.latent.get(),data.mb_latent.get(),stream);
        rl::gather_rows(permutation,first,mb,1,data.logp.get(),data.mb_logp.get(),stream);
        rl::gather_rows(permutation,first,mb,1,data.advantages.get(),data.mb_advantage.get(),stream);
        CUDA_CHECK_THROW(cudaMemsetAsync(data.row_gradient.get(),0,size_t(mb)*8*sizeof(float),stream));
        CUDA_CHECK_THROW(cudaMemsetAsync(data.row_stats.get(),0,size_t(mb)*5*sizeof(float),stream));
        episode_loss<<<(mb+127)/128,128,0,stream>>>(mb,data.means.get(),data.mb_latent.get(),data.mb_logp.get(),data.mb_advantage.get(),data.row_gradient.get(),data.row_stats.get());
        CUDA_CHECK_THROW(cudaMemsetAsync(data.gradient.get(),0,8*sizeof(float),stream));CUDA_CHECK_THROW(cudaMemsetAsync(data.stats.get(),0,5*sizeof(float),stream));
        rl::sum_columns(data.row_gradient.get(),mb,8,data.gradient.get(),stream);rl::sum_columns(data.row_stats.get(),mb,5,data.stats.get(),stream);
        const auto gathered=download(data.mb_latent.get(),size_t(mb)*8,stream),gp=download(data.mb_logp.get(),mb,stream),
          ga=download(data.mb_advantage.get(),mb,stream),rows_gradient=download(data.row_gradient.get(),size_t(mb)*8,stream),
          sum=download(data.gradient.get(),8,stream),stats=download(data.stats.get(),5,stream);
        double expected_sum[8]{},expected_loss=0;const double fixed_log_sigma=std::log(double(policy::fixed_sigma)),entropy=8*(fixed_log_sigma+.5*(1.83787706640934548+1));
        for(int row=0;row<mb;++row) {
            const int original=int(rl::permute(permutation,uint32_t(first+row)));require(original>=0&&original<n&&++seen[original]==1,"Gait bulk permutation coverage defect");
            require(gp[row]==old_logp[original]&&ga[row]==advantage[original],"Gait bulk scalar gather defect");
            double logp=-8*fixed_log_sigma-4*1.83787706640934548;for(int j=0;j<8;++j) {
                require(gathered[row*8+j]==latent[original*8+j],"Gait bulk latent gather defect");
                const double z=double(gathered[row*8+j])/double(policy::fixed_sigma);logp-=.5*z*z;
            }
            const double ratio=std::exp(logp-double(gp[row]));require(std::abs(ratio-1)<.001,"Synthetic V2 ratio left unclipped fixture");
            expected_loss-=double(ga[row])*ratio;
            double last_magnitude=0;
            for(int j=0;j<8;++j) {
                const double expected=-double(ga[row])*ratio*double(gathered[row*8+j])/(double(policy::fixed_sigma)*policy::fixed_sigma)/mb;
                const double error=std::abs(expected-rows_gradient[row*8+j]);maximum_row=std::max(maximum_row,error);expected_sum[j]+=expected;
                if(row==mb-1){maximum_last=std::max(maximum_last,error);last_magnitude+=std::abs(expected);}
            }
            if(row==mb-1){require(last_magnitude>1e-8,"Gait bulk last-row fixture is vacuous");++last_rows;}
            ++checked;
        }
        for(int j=0;j<8;++j)maximum_sum=std::max(maximum_sum,std::abs(expected_sum[j]-sum[j]));
        maximum_stat=std::max(maximum_stat,std::abs(expected_loss/mb-double(stats[0])/mb));
        maximum_stat=std::max(maximum_stat,std::abs(entropy-double(stats[2])/mb));
        require(stats[1]==0&&std::abs(stats[3]/mb)<1e-5&&stats[4]==0,"Gait bulk critic/KL/clipping fixture defect");
    }
    for(int count:seen)require(count==1,"Gait bulk minibatches missed an episode");
    require(checked==n&&last_rows==4&&maximum_row<1e-5&&maximum_sum<2e-5&&maximum_stat<2e-5,"Gait bulk native gradient/reduction gate failed");
    const auto means=download(data.means.get(),8,stream);for(float x:means)require(x==0,"Gait bulk gradient fixture updated policy");
    std::ofstream report(directory/"bulk-policy-zero.json");report<<std::setprecision(17)<<"{\"passed\":true,\"worlds\":"<<n
      <<",\"minibatch_episodes\":"<<mb<<",\"loss_grid_blocks\":"<<(mb+127)/128<<",\"checked_rows\":"<<checked<<",\"checked_last_rows\":"<<last_rows
      <<",\"max_row_gradient_error\":"<<maximum_row<<",\"max_reduced_gradient_error\":"<<maximum_sum<<",\"max_stats_error\":"<<maximum_stat
      <<",\"max_last_row_gradient_error\":"<<maximum_last<<",\"physical_controls\":0,\"optimizer_steps\":0}\n";
}

void zero_initialization(const fs::path& directory){require(fs::create_directory(directory),"Gait zero initialization identity exists");disk_guard(study_root);Data data;const auto stream=data.stream->value;
    CUDA_CHECK_THROW(cudaMemsetAsync(data.fault.get(),0,sizeof(int),stream));initialize_episodes<<<64,128,0,stream>>>(data.worlds.get(),data.tasks.get(),data.counts.get(),data.means.get(),data.latent.get(),data.logp.get(),data.fault.get(),calibration_seed,0);
    observe_episodes<<<64,128,0,stream>>>(data.worlds.get(),data.tasks.get(),data.observation.get(),data.fault.get());data.backbone->forward(data.observation.get(),worlds_count,stream);
    const auto worlds=download(data.worlds.get(),worlds_count,stream);const auto tasks=download(data.tasks.get(),worlds_count,stream);
    const auto input=download(data.observation.get(),worlds_count*41,stream),output=download(data.backbone->output(),worlds_count*4,stream),latent=download(data.latent.get(),worlds_count*8,stream),logp=download(data.logp.get(),worlds_count,stream);
    require(download(data.fault.get(),1,stream)[0]==0,"Gait zero initialization fault");bulk_policy_zero(data,latent,logp,directory);nn::CpuMlp<float> cpu(spec(4));cpu.params()=resume_state->actor.parameters;double input_error=0,output_error=0,sample_error=0;
    for(int i=0;i<worlds_count;++i){m::World world;m::ForwardTask t;require(task::initialize(i%4,world,t),"Gait CPU zero initialization failed");float expected[41];require(m::observe_appended(t,world,task::config(),expected),"Gait CPU zero observation failed");
        require(worlds[i].attempted_substeps==0&&worlds[i].physical.control_steps==0&&tasks[i].controls==0,"Gait zero fixture advanced dynamics");
        for(int j=0;j<41;++j)input_error=std::max(input_error,double(std::abs(expected[j]-input[i*41+j])));cpu.forward(expected,1);
        for(int j=0;j<4;++j)output_error=std::max(output_error,double(std::abs(cpu.output()[j]-output[i*4+j])));
        for(int j=0;j<8;++j)sample_error=std::max(sample_error,double(std::abs(latent[i*8+j]-policy::fixed_sigma*cudarl::rng::normal(calibration_seed,0,uint32_t(i*8+j)))));}
    require(input_error<1e-6&&output_error<2e-5&&sample_error<1e-6,"Gait CPU/CUDA zero observation/body/RNG mismatch");
    cp::Writer proof;proof.value(uint32_t{0x314e4747});proof.value(uint32_t{1});proof.value(worlds_count);proof.value(uint32_t{sizeof(m::World)});proof.array(worlds);proof.value(uint32_t{sizeof(m::ForwardTask)});proof.array(tasks);proof.array(input);proof.array(output);proof.array(latent);proof.array(logp);cp::write_atomic((directory/"initialization.bin").string(),proof.data);
    cudaDeviceProp properties{};CUDA_CHECK_THROW(cudaGetDeviceProperties(&properties,0));cudaFuncAttributes attributes{};CUDA_CHECK_THROW(cudaFuncGetAttributes(&attributes,advance_episodes));int active=0;CUDA_CHECK_THROW(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,advance_episodes,128,0));
    std::ofstream out(directory/"initialization-zero.json");out<<std::setprecision(17)<<"{\"passed\":true,\"worlds\":"<<worlds_count<<",\"observation_error\":"<<input_error<<",\"backbone_error\":"<<output_error<<",\"RNG_error\":"<<sample_error<<",\"GPU_SM_count\":"<<properties.multiProcessorCount<<",\"kernel_registers\":"<<attributes.numRegs<<",\"kernel_local_bytes\":"<<attributes.localSizeBytes<<",\"active_blocks_per_SM\":"<<active<<",\"grid_blocks\":64,\"physical_controls\":0,\"optimizer_steps\":0}\n";
}
struct GateCount {unsigned long long controls=0,substeps=0;int failure=0;p::BodyState before{},candidate{};};
CUDARL_HD inline bool initialize_gate(int id,m::World& world,m::ForwardTask& t){const bool hold=id>=4;
    const int index=id-4,pose=hold?(index%2?3:0):id;world=m::initial_world(pose);world.arena=m::long_swim_arena();auto c=task::config(hold?1280:160);
    if(hold){c.target_speed_m_s=0;c.stationary_request=true;c.position_hold=true;}
    return m::begin_forward(world,c,t)&&(!hold||(m::offset_episode_start(world,t,c,index/2==0?0.f:index/2==1?-.0005f:.0005f)&&m::update_velocity_request(world,c,t)));}
__global__ void gate_observe(const m::World* worlds,const m::ForwardTask* tasks,float* observation,int* fault){const int i=threadIdx.x;if(i>=10)return;
    auto c=task::config(i>=4?1280:160);if(i>=4){c.target_speed_m_s=0;c.stationary_request=true;c.position_hold=true;}
    if(!m::observe_appended(tasks[i],worlds[i],c,observation+i*41))atomicCAS(fault,0,1);}
__global__ void gate_step(m::World* worlds,m::ForwardTask* tasks,GateCount* counts,const float* output,float hold_gain,int* fault){const int i=threadIdx.x;if(i>=10||*fault)return;
    const float zero[8]{};float raw[4];policy::motor_latents(i>=4,zero,output+i*4,hold_gain,raw);m::Action action;
    if(!m::action_from_latent(raw,action)){atomicCAS(fault,0,2);return;}const auto before=worlds[i].attempted_substeps;++counts[i].controls;
    p::BodyState pre=worlds[i].physical.body,candidate=pre;const bool ok=cudarl::joint_motor::step_state(worlds[i],action.command,nullptr,&pre,&candidate);
    counts[i].substeps+=worlds[i].attempted_substeps-before;
    if(!ok||worlds[i].attempted_substeps-before!=10){counts[i].before=pre;counts[i].candidate=candidate;counts[i].failure=worlds[i].failure_code;atomicCAS(fault,0,3);return;}
    auto c=task::config(i>=4?1280:160);if(i>=4){c.target_speed_m_s=0;c.stationary_request=true;c.position_hold=true;}
    const auto interval=i>=4?m::record_hold_control(tasks[i],worlds[i],c):m::record_control(tasks[i],worlds[i],c);
    if(!interval.accepted)atomicCAS(fault,0,4);}
double compare_worlds(const m::World& a,const m::World& b,bool exact){float pa[33],pb[33];record_pre(a,pa);record_pre(b,pb);double maximum=0;
    for(int j=0;j<33;++j){const double difference=std::abs(double(pa[j])-pb[j]);maximum=std::max(maximum,difference);
        const double tolerance=exact?0:j<3?2e-5:j<11?2e-5:j<14?2e-7:2e-5;require(difference<=tolerance,"Gait compatibility state mismatch");}
    const double ea[6]={a.signed_work,a.positive_work,a.fluid_loss,a.numerical_loss,a.external_work,a.last_energy_error};
    const double eb[6]={b.signed_work,b.positive_work,b.fluid_loss,b.numerical_loss,b.external_work,b.last_energy_error};
    for(int j=0;j<6;++j)require(std::abs(ea[j]-eb[j])<=(exact?0:1e-12+1e-4*std::max(std::abs(ea[j]),std::abs(eb[j]))),"Gait compatibility cumulative-energy mismatch");
    require(a.physical.control_steps==b.physical.control_steps&&a.physical.substeps==b.physical.substeps&&a.attempted_substeps==b.attempted_substeps,"Gait compatibility counters differ");
    require(a.physical.time_seconds==b.physical.time_seconds&&a.physical.paused==b.physical.paused&&a.physical.unsafe==b.physical.unsafe&&a.physical.invalid==b.physical.invalid&&a.failure_code==b.failure_code,"Gait compatibility status/time differs");
    if(exact){const auto& x=a.physical;const auto& y=b.physical;
        require(x.body.work==y.body.work&&x.mean_squared_torque==y.mean_squared_torque&&x.telemetry.minimum_floor_gap==y.telemetry.minimum_floor_gap&&x.telemetry.minimum_other_gap==y.telemetry.minimum_other_gap&&x.telemetry.floor_spring_force_max==y.telemetry.floor_spring_force_max&&x.telemetry.total_spring_force_max==y.telemetry.total_spring_force_max&&x.telemetry.contact_samples==y.telemetry.contact_samples&&x.telemetry.executed_substeps==y.telemetry.executed_substeps&&x.telemetry.first_unsafe_substep==y.telemetry.first_unsafe_substep,"Identity gait changed defined physical telemetry");
        for(int j=0;j<4;++j)require(x.raw_action[j]==y.raw_action[j],"Identity gait changed requested action");
        for(int j=0;j<40;++j)require(x.requested_torques[j]==y.requested_torques[j],"Identity gait changed requested torque");}
    for(int j=0;j<40;++j){const double torque=std::abs(double(a.physical.actual_torques[j])-b.physical.actual_torques[j]);require(torque<=(exact?0:2e-5),"Gait compatibility actual torque mismatch");}
    return maximum;}
void compatibility_gate(const fs::path& root,Budget& budget){const auto directory=root/"gate-started";require(fs::create_directory(directory),"Gait gate already attempted");failure_directory=directory;
    std::vector<m::World> legacy(10),identity(10);std::vector<m::ForwardTask> legacy_task(10),identity_task(10);
    for(int i=0;i<10;++i){require(initialize_gate(i,legacy[i],legacy_task[i]),"Gait gate reset rejected");identity[i]=legacy[i];identity_task[i]=legacy_task[i];}
    Stream stream;device_ptr<m::World> worlds(10);device_ptr<m::ForwardTask> tasks(10);device_ptr<GateCount> counts(10);device_ptr<float> observation(410);device_ptr<int> fault(1);
    worlds.copy_from_host(legacy.data(),legacy.size());tasks.copy_from_host(legacy_task.data(),legacy_task.size());CUDA_CHECK_THROW(cudaMemsetAsync(counts.get(),0,counts.size_bytes(),stream.value));CUDA_CHECK_THROW(cudaMemsetAsync(fault.get(),0,sizeof(int),stream.value));
    nn::GpuMlp gpu_body(spec(4),10,seed);gpu_body.set_params(resume_state->actor.parameters);nn::CpuMlp<float> cpu_body(spec(4));cpu_body.params()=resume_state->actor.parameters;
    cp::Writer raw;raw.value(uint32_t{0x31454747});raw.value(uint32_t{1});raw.value(study_protocol);raw.value(uint32_t{sizeof(m::World)});raw.value(uint32_t{sizeof(m::ForwardTask)});raw.array(legacy);raw.array(legacy_task);
    double maximum_difference=0;unsigned long long previous_gpu_controls=0,previous_gpu_substeps=0;
    for(int control=1;control<=40;++control){time_guard();
        for(int backend=0;backend<2;++backend)for(int id=0;id<10;++id){auto& world=backend?identity[id]:legacy[id];auto& t=backend?identity_task[id]:legacy_task[id];auto c=task::config(id>=4?1280:160);if(id>=4){c.target_speed_m_s=0;c.stationary_request=true;c.position_hold=true;}
            float obs[41],z[4],zero[8]{};require(m::observe_appended(t,world,c,obs),"Gait gate CPU observation rejected");cpu_body.forward(obs,1);
            if(backend)policy::motor_latents(id>=4,zero,cpu_body.output(),saved_hold_gain,z);else for(int j=0;j<4;++j)z[j]=(id>=4?saved_hold_gain:1.f)*cpu_body.output()[j];
            m::Action action;require(m::action_from_latent(z,action),"Gait gate CPU action rejected");budget.step(world,action.command);const auto interval=id>=4?m::record_hold_control(t,world,c):m::record_control(t,world,c);require(interval.accepted,"Gait gate CPU task rejected");}
        gate_observe<<<1,32,0,stream.value>>>(worlds.get(),tasks.get(),observation.get(),fault.get());gpu_body.forward(observation.get(),10,stream.value);
        gate_step<<<1,32,0,stream.value>>>(worlds.get(),tasks.get(),counts.get(),gpu_body.output(),saved_hold_gain,fault.get());
        const auto actual=download(worlds.get(),10,stream.value);const auto actual_task=download(tasks.get(),10,stream.value);const auto actual_count=download(counts.get(),10,stream.value);
        unsigned long long gc=0,gs=0;for(const auto& x:actual_count){gc+=x.controls;gs+=x.substeps;}budget.controls+=int(gc-previous_gpu_controls);budget.substeps+=int(gs-previous_gpu_substeps);previous_gpu_controls=gc;previous_gpu_substeps=gs;
        raw.value(control);raw.array(legacy);raw.array(identity);raw.array(actual);raw.array(legacy_task);raw.array(identity_task);raw.array(actual_task);raw.array(actual_count);
        cp::write_atomic((directory/"endpoints.bin").string(),raw.data);
        require(download(fault.get(),1,stream.value)[0]==0,"Gait GPU compatibility physical/task fault");
        for(int id=0;id<10;++id){compare_worlds(legacy[id],identity[id],true);maximum_difference=std::max(maximum_difference,compare_worlds(legacy[id],actual[id],false));}}
    require(budget.controls==gate_controls&&budget.substeps==gate_controls*10,"Gait compatibility exact accounting differs");
    std::ofstream report(directory/"result.json");report<<std::setprecision(17)<<"{\"passed\":true,\"controls\":1200,\"substeps\":12000,\"fixtures\":10,\"backends\":3,\"CPU_identity_exact\":true,\"maximum_state_difference\":"<<maximum_difference<<"}\n";report.close();require(bool(report),"Cannot save gait gate result");}
struct Prior {int controls=0,substeps=0;double seconds=0;};
std::vector<fs::path> bindings(const fs::path& root,int phase){
    if(phase==0)return {root/"gate-result.json",root/"gate-started/endpoints.bin",root/"gate-started/result.json"};
    if(phase==1)return {root/"baseline-result.json",root/"baseline-short/summary.csv",root/"baseline-hold/summary.csv",root/"baseline-long/windows.csv",root/"gate-ready.bin"};
    return {root/"calibration-result.json",root/"calibration-report.json",root/"fixed-scale.bin",root/"calibration-started/calibration-batch-0/manifest.json",root/"baseline-ready.bin"};}
const char* phase_name(int phase){return phase==0?"gate":phase==1?"baseline":"calibration";}
void write_ready(const fs::path& root,int phase,const Budget& budget){cp::Writer out;out.value(ready_magic);out.value(uint32_t{1});out.value(study_protocol);out.value(phase);out.value(budget.controls);out.value(budget.substeps);
    out.value(prior_seconds+std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count());std::vector<uint64_t> hashes;
    for(const auto& path:bindings(root,phase))hashes.push_back(file_hash(path));out.array(hashes);const auto path=root/(std::string(phase_name(phase))+"-ready.bin");require(!fs::exists(path),"Gait phase release already exists");cp::write_atomic(path.string(),out.data);}
Prior read_ready(const fs::path& root,int phase){const auto bytes=cp::read((root/(std::string(phase_name(phase))+"-ready.bin")).string());cp::Reader in{bytes};
    require(in.value<uint32_t>()==ready_magic&&in.value<uint32_t>()==1&&in.value<uint64_t>()==study_protocol&&in.value<int>()==phase,"Wrong gait phase release");Prior prior;
    prior.controls=in.value<int>();prior.substeps=in.value<int>();prior.seconds=in.value<double>();const auto hashes=in.array<uint64_t>(16);in.finish();const auto files=bindings(root,phase);require(hashes.size()==files.size(),"Gait release binding count differs");
    for(size_t i=0;i<files.size();++i)require(hashes[i]==file_hash(files[i]),"Preceding gait evidence changed");
    require(prior.controls>=1200&&prior.controls<=controls_cap&&prior.substeps==prior.controls*10&&std::isfinite(prior.seconds)&&prior.seconds>0&&prior.seconds<seconds_cap,"Invalid gait preceding accounting");return prior;}
Budget begin_phase(const fs::path& root,const char* phase,const Prior& prior,bool create=true){prior_seconds=prior.seconds;started=std::chrono::steady_clock::now();active_wall_cap=seconds_cap-prior_seconds;
    const auto path=root/(std::string(phase)+"-started");if(create)require(fs::create_directory(path),"Gait phase already attempted; no retry");
    failure_directory=path;failure_saved=false;Budget budget;budget.controls=prior.controls;budget.substeps=prior.substeps;budget.maximum_controls=controls_cap;budget.maximum_substeps=substeps_cap;budget.seconds=active_wall_cap;return budget;}
void result(const fs::path& path,const Budget& budget,const std::string& fault,const std::string& gate,int completed_updates,int steps,int movement,int gait,bool qualified,
        int long60=-1,int recovery=-1,int long300=-1,int holds=-1,int withheld=-1,int hold300=-1,bool closed=false){
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();std::ofstream out(path);out<<std::setprecision(17)
      <<"{\"schema\":1,\"study\":\"gait_pond_v1\",\"controls\":"<<budget.controls<<",\"substeps\":"<<budget.substeps
      <<",\"active_seconds\":"<<seconds<<",\"prior_active_seconds\":"<<prior_seconds<<",\"PPO_updates\":"<<completed_updates<<",\"actor_steps\":"<<steps
      <<",\"critic_steps\":0,\"critic_forward_calls\":0,\"fault\":"<<std::quoted(fault)<<",\"gate\":"<<std::quoted(gate)
      <<",\"short_movement_passes\":"<<movement<<",\"short_gait_passes\":"<<gait<<",\"qualified\":"<<(qualified?"true":"false")
      <<",\"long60_passing_windows\":"<<long60<<",\"recovery_passes\":"<<recovery<<",\"long300_passing_windows\":"<<long300
      <<",\"original_hold_passes\":"<<holds<<",\"withheld_hold_passes\":"<<withheld<<",\"hold300_passes\":"<<hold300
      <<",\"worlds\":8192,\"phase_closed\":true,\"closed\":"<<((closed||!fault.empty())?"true":"false")<<",\"runtime_adopted\":false,\"retries\":0}\n";
    out.close();require(bool(out),"Cannot save gait result");std::ofstream status(path.parent_path()/"status.json");
    status<<"{\"phase\":"<<std::quoted(path.stem().string())<<",\"controls\":"<<budget.controls<<",\"substeps\":"<<budget.substeps
      <<",\"PPO_updates\":"<<completed_updates<<",\"qualified\":"<<(qualified?"true":"false")<<",\"closed\":"<<((closed||!fault.empty())?"true":"false")<<"}\n";
}
void verify_calibration_release(const fs::path& root){const auto bytes=cp::read((root/"calibration-verified.bin").string());cp::Reader in{bytes};
    require(in.value<uint32_t>()==0x31564747&&in.value<uint32_t>()==1&&in.value<uint64_t>()==study_protocol,"Independent gait calibration release missing/incompatible");
    for(const auto& name:{"independent-calibration-verification.json","fixed-scale.bin","calibration-ready.bin"})require(in.value<uint64_t>()==file_hash(root/name),"Independent gait calibration evidence changed");in.finish();}
int prepare(const fs::path& root,const fs::path& approved){require(!fs::exists(root),"Gait run identity already exists");fs::create_directories(root);configure(root,false);
    fs::copy_file(approved,root/"approved-protocol.md");fs::copy_file(parent_checkpoint_path,root/"parent-checkpoint.bin");fs::copy_file(parent_protocol_path,root/"parent-protocol.bin");
    fs::copy_file(fs::absolute("assets/ppo-four-axis-checkpoints/learned-amplitude-2026-11-22-qualified/qualified-policy-update-7.bin"),root/"holding-parent.bin");
    const auto bytes=protocol_bytes(root);cp::write_atomic((root/"protocol.bin").string(),bytes);study_protocol=cp::checksum(bytes);
    cp::write_atomic((root/"migrated-initial.bin").string(),rl::write_appended_hold(migrated41,study_protocol));write_layout(root);identity_check(std::vector<float>(8,0));
    require(fs::create_directory(root/"initial-policy"),"Gait initial checkpoint identity exists");disk_guard(root);{Data data;save_policy(data,root/"initial-policy",0);}
    numerical(root/"prepare-math");zero_initialization(root/"prepare-zero");
    std::ofstream out(root/"prepared.json");out<<"{\"schema\":1,\"prepared\":true,\"worlds\":8192,\"updates_max\":64,\"controls_cap\":85295280,\"substeps_cap\":852952800,\"Adam_max\":1024,\"physical_controls\":0,\"optimizer_steps\":0}\n";out.close();require(bool(out),"Cannot save gait preparation");return 0;}
int preflight(const fs::path& root){configure(root,true);validate_protocol(root);write_layout(root);numerical(root/"verified-math");zero_initialization(root/"verified-zero");
    std::ofstream out(root/"preflight-passed.txt");out<<"Gait bulk PPO/FD, identity, exact holding branch, CPU/CUDA body/input/RNG/ABI and fresh Adam checked; zero physical controls and optimizer steps.\n";return 0;}
int run_gate(const fs::path& root){configure(root,true);validate_protocol(root);require(fs::exists(root/"preflight-passed.txt"),"Gait preflight absent");disk_guard(root,true);
    auto budget=begin_phase(root,"gate",{},false);std::string fault;
    try{compatibility_gate(root,budget);}catch(const std::exception& e){fault=e.what();std::cerr<<fault<<std::endl;}
    result(root/"gate-result.json",budget,fault,fault.empty()?"passed":"compatibility_failed",0,0,0,0,false);
    if(fault.empty())write_ready(root,0,budget);return fault.empty()?0:3;}
int baseline(const fs::path& root){configure(root,true);validate_protocol(root);const auto prior=read_ready(root,0);auto budget=begin_phase(root,"baseline",prior);std::string fault;SwimResult short_score,long_score;int holds=-1;
    try{const std::vector<float> means(8,0);short_score=assess_swim(means,short_cases(),root/"baseline-short",budget);
        holds=assess_holds(means,original_holds(),root/"baseline-hold",budget);long_score=assess_swim(means,{{0,1200,0},{3,1200,0}},root/"baseline-long",budget,true);
        require(short_score.movement_passed==4&&holds==6&&retained_windows(root/"baseline-long/windows.csv",60),"Inherited gait baseline lost a declared skill");
        require(budget.controls-prior.controls==10720,"Gait baseline accounting differs");}
    catch(const std::exception& e){fault=e.what();std::cerr<<fault<<std::endl;}
    result(root/"baseline-result.json",budget,fault,fault.empty()?"baseline_retained":"baseline_failed",0,0,short_score.movement_passed,short_score.gait_passed,false,long_score.window_passed,-1,-1,holds);
    if(fault.empty())write_ready(root,1,budget);return fault.empty()?0:3;}
int calibrate(const fs::path& root){configure(root,true);validate_protocol(root);const auto prior=read_ready(root,1);auto budget=begin_phase(root,"calibration",prior);std::string fault;
    try{disk_guard(root);Data data;const auto batch=collect(data,budget,root/"calibration-started/calibration-batch-0",0,0);const auto scale=fit_scale(batch.returns);
        cp::Writer out;out.value(scale_magic);out.value(uint32_t{1});out.value(study_protocol);out.value(scale.value);for(double mean:scale.means)out.value(mean);cp::write_atomic((root/"fixed-scale.bin").string(),out.data);
        int movement=0,gait=0;for(int i=0;i<worlds_count;++i){movement+=batch.tasks[i].outcome==m::Outcome::Success;gait+=batch.counts[i].gait.passed();}
        std::ofstream report(root/"calibration-report.json");report<<std::setprecision(17)<<"{\"schema\":1,\"passed\":true,\"episodes\":8192,\"cases\":4,\"episodes_per_case\":2048,\"fixed_global_residual_RMS\":"<<scale.value
          <<",\"sampled_movement_passes\":"<<movement<<",\"sampled_gait_passes\":"<<gait<<",\"max_old_logp_error\":"<<batch.probability_error<<",\"max_decode_error\":"<<batch.decode_error<<",\"controls\":"<<budget.controls<<",\"substeps\":"<<budget.substeps<<",\"actor_steps\":0}\n";report.close();require(bool(report)&&budget.controls-prior.controls==1310720,"Gait calibration report/count differs");}
    catch(const std::exception& e){fault=e.what();std::cerr<<fault<<std::endl;}
    result(root/"calibration-result.json",budget,fault,fault.empty()?"await_independent_readiness":"calibration_failed",0,0,0,0,false);
    if(fault.empty())write_ready(root,2,budget);return fault.empty()?0:3;}
Scale read_scale(const fs::path& root){const auto bytes=cp::read((root/"fixed-scale.bin").string());cp::Reader in{bytes};
    require(in.value<uint32_t>()==scale_magic&&in.value<uint32_t>()==1&&in.value<uint64_t>()==study_protocol,"Gait fixed scale protocol differs");Scale scale;scale.value=in.value<double>();for(double& mean:scale.means)mean=in.value<double>();in.finish();require(std::isfinite(scale.value)&&scale.value>0,"Gait scale invalid");return scale;}
int train(const fs::path& root){configure(root,true);validate_protocol(root);const auto prior=read_ready(root,2);verify_calibration_release(root);const auto scale=read_scale(root);auto budget=begin_phase(root,"training",prior);
    std::unique_ptr<Data> active;std::string fault,gate;int completed_updates=0,steps=0,holds=-1,withheld=-1,hold300=-1;SwimResult short_score,long60,long300,recovery;bool qualified=false;
    std::ofstream progress(root/"updates.jsonl");progress<<std::setprecision(17);
    try{disk_guard(root);active=std::make_unique<Data>(true);auto& data=*active;save_policy(data,root/"training-started",0);
        for(int update=1;update<=64;++update){const auto batch=collect(data,budget,root/"training-started"/("training-batch-"+std::to_string(update)),1,update);
            const double kl=optimize(data,batch,scale,update,root/"training-started");completed_updates=update;steps=data.optimizer->steps();save_policy(data,root/"training-started",update);
            const auto means=download(data.means.get(),8,data.stream->value);identity_check(means);short_score=assess_swim(means,short_cases(),root/"training-started"/("assessment-"+std::to_string(update)),budget);
            progress<<"{\"update\":"<<update<<",\"worlds\":8192,\"minibatch_episodes\":2048,\"movement_passes\":"<<short_score.movement_passed<<",\"gait_passes\":"<<short_score.gait_passed<<",\"mean_cost\":"<<short_score.mean_cost<<",\"KL\":"<<kl<<",\"actor_steps\":"<<steps<<",\"controls\":"<<budget.controls<<",\"means\":[";
            for(int j=0;j<8;++j){if(j)progress<<',';progress<<means[j];}progress<<"]}\n";progress.flush();require(bool(progress),"Cannot save gait update ledger");
            std::cout<<"GAIT_UPDATE "<<update<<" movement="<<short_score.movement_passed<<" gait="<<short_score.gait_passed<<" controls="<<budget.controls<<std::endl;
            if(short_score.movement_passed==4&&short_score.gait_passed==4)break;}
        const auto means=download(data.means.get(),8,data.stream->value);require(fs::create_directory(root/"final-skills"),"Gait final suite already attempted");
        long60=assess_swim(means,{{0,1200,0},{3,1200,0}},root/"final-skills/long60",budget,true);
        recovery=assess_swim(means,{{0,1280,0},{0,1280,-.0005f},{0,1280,.0005f}},root/"final-skills/recovery64",budget,false,true);
        long300=assess_swim(means,{{0,6000,0},{3,6000,0}},root/"final-skills/long300",budget,true);
        holds=assess_holds(means,original_holds(),root/"final-skills/original-holds",budget);
        withheld=assess_holds(means,withheld_holds(),root/"final-skills/withheld-holds",budget);
        hold300=assess_holds(means,{{0,6000,0},{3,6000,0}},root/"final-skills/hold300",budget);
        qualified=short_score.movement_passed==4&&short_score.gait_passed==4&&retained_windows(root/"final-skills/long60/windows.csv",60)&&
          long60.gait_window_passed==28&&recovery.recovery_passed==3&&retained_windows(root/"final-skills/long300/windows.csv",300)&&
          long300.gait_window_passed==148&&holds==6&&withheld==6&&hold300==2;
        gate=qualified?"qualified_balanced_gait_no_adoption":"gait_or_retention_not_qualified";
        require(budget.controls==1200+10720+1310720+completed_updates*(worlds_count*160+640)+45600,"Gait full final accounting differs");}
    catch(const std::exception& e){fault=e.what();gate="physical_data_or_numerical_stop";
        if(active&&active->optimizer)steps=active->optimizer->steps();
        std::cerr<<fault<<std::endl;}
    result(root/"training-result.json",budget,fault,gate,completed_updates,steps,short_score.movement_passed,short_score.gait_passed,qualified,long60.window_passed,recovery.recovery_passed,long300.window_passed,holds,withheld,hold300,true);
    return fault.empty()?0:3;}
}
int main(int argc,char** argv){try{require(argc>=3,"Use --prepare/--preflight/--gate/--baseline/--calibrate/--train RUN [protocol]");const std::string mode=argv[1];const auto root=fs::absolute(argv[2]);
    if(mode=="--prepare"){require(argc==4,"Gait prepare requires approved protocol");return gait_study::prepare(root,fs::absolute(argv[3]));}
    require(argc==3,"Unexpected gait argument");if(mode=="--preflight")return gait_study::preflight(root);if(mode=="--gate")return gait_study::run_gate(root);
    if(mode=="--baseline")return gait_study::baseline(root);if(mode=="--calibrate")return gait_study::calibrate(root);if(mode=="--train")return gait_study::train(root);
    throw std::runtime_error("Unknown gait mode");}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}}

