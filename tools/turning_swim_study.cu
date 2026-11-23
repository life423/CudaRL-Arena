// Fresh opt-in native PPO: food-free autonomous swimming inside a pond.
#include "motor/joint_motor.h"
namespace turning_study {void physical_failure(const cudarl::joint_motor::World&,const cudarl::physics::BodyState&,const cudarl::physics::BodyState&,int,int);}
#define CUDARL_PHYSICAL_FAILURE_HOOK turning_study::physical_failure
#define CUDARL_REST_HOLD_MEASURED_PHASE_B 1
#define CUDARL_REST_HOLD_MEASURED_BASELINE 1
#define CUDARL_REST_HOLD_RESET_STUDY 1
#include "rest_hold_study.cu"
#include "hold_learned_amplitude_math.h"
#include "motor/turning_policy_v1.h"
#include "motor/turning_task_v1.h"
#include <cstddef>
#include <set>
namespace turning_study {
namespace policy=cudarl::turning_policy_v1;namespace task=cudarl::turning_task_v1;
constexpr int training_worlds=4096,episode_controls=320,head_parameters=16,minibatch_episodes=1024;
constexpr int maximum_updates=64,maximum_adam_steps=1024,maximum_epochs=4;
constexpr int controls_cap=86811440,substeps_cap=868114400,gate_controls=2480;
constexpr double seconds_cap=900;
constexpr uint32_t calibration_seed=0x4e23b197u,training_seed=0xa715c2d3u;
constexpr uint32_t protocol_magic=0x31525054,policy_magic=0x31505054,batch_magic=0x31425054,scale_magic=0x31535054,ready_magic=0x31595054;
constexpr uint64_t gib=1024ull*1024*1024,projected_study_bytes=20*gib,working_disk_reserve=8*gib;
uint64_t study_protocol=0;fs::path study_root,failure_directory;bool failure_saved=false;
float saved_hold_gain=0;std::vector<float> saved_hold_parameters;std::vector<uint8_t> hold_checkpoint_payload;
double prior_seconds=0;
static_assert(policy::parameters==head_parameters&&policy::actions==4,"Turning policy/body contract differs");
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
void time_guard(){require(prior_seconds+std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<seconds_cap,"Combined turning native runtime cap reached");}
void disk_guard(const fs::path& root,int n,int episode_limit,bool initial=false){const uint64_t bytes=uint64_t(n)*episode_limit*(40*4+1)+uint64_t(n)*2400+1024*1024;
 const auto free=fs::space(root).available;require(free>=working_disk_reserve+bytes,"Insufficient turning evidence space before allocation/collection");
 if(initial)require(free>=projected_study_bytes+working_disk_reserve,"Turning whole-study evidence plus8GiB reserve does not fit");}
struct EpisodeCounts {unsigned long long controls=0,attempted_substeps=0,accepted_substeps=0;double reward_sum=0,seconds=0;int last_substeps=0,reason=0;bool done=false,truncated=false;p::BodyState first_before{},rejected_candidate{};};
struct Data {
 int n,horizon,mb;std::unique_ptr<Stream> stream=std::make_unique<Stream>();std::unique_ptr<nn::GpuMlp> backbone;std::unique_ptr<nn::GpuAdam> optimizer;
 device_ptr<m::World> worlds,initial_worlds;device_ptr<task::Task> tasks,initial_tasks;device_ptr<EpisodeCounts> counts;device_ptr<int> fault,active;
 device_ptr<float> means,latent,logp,observation,raw_features,pre_filter,post_body,joint_latent,reward;
 device_ptr<uint8_t> row_substeps;
 device_ptr<float> advantages,mb_latent,mb_logp,mb_advantage,row_gradient,row_stats,gradient,stats;
 Data(int worlds_count=training_worlds,int controls=episode_controls,bool training=false):n(worlds_count),horizon(controls),mb(worlds_count/4),
  backbone(std::make_unique<nn::GpuMlp>(spec(4),worlds_count,seed)),worlds(worlds_count),initial_worlds(worlds_count),tasks(worlds_count),initial_tasks(worlds_count),counts(worlds_count),fault(1),active(1),
  means(16),latent(size_t(worlds_count)*16),logp(worlds_count),observation(size_t(worlds_count)*41),raw_features(size_t(worlds_count)*8),
  pre_filter(size_t(worlds_count)*controls*12),post_body(size_t(worlds_count)*controls*23),joint_latent(size_t(worlds_count)*controls*4),reward(size_t(worlds_count)*controls),row_substeps(size_t(worlds_count)*controls),
  advantages(worlds_count),mb_latent(size_t(worlds_count/4)*16),mb_logp(worlds_count/4),mb_advantage(worlds_count/4),row_gradient(size_t(worlds_count/4)*16),row_stats(size_t(worlds_count/4)*5),gradient(16),stats(5){
  backbone->set_params(resume_state->actor.parameters);const std::vector<float> zero(16,0);means.copy_from_host(zero.data(),zero.size());
  if(training){require(n==training_worlds&&horizon==episode_controls,"Wrong turning training dimensions");auto c=initial_optimizer_config(true);c.lr=.005f;optimizer=std::make_unique<nn::GpuAdam>(16,c);require(optimizer->steps()==0,"Fresh turning Adam required");}}
};
CUDARL_HD inline void record_filter(const m::World& world,float* values){for(int j=0;j<4;++j){values[j]=world.physical.held_targets[j];values[4+j]=world.reference_rate[j];values[8+j]=world.physical.last_actual_torques[j];}}
CUDARL_HD inline void record_body(const p::BodyState& body,float* values){values[0]=body.position.x;values[1]=body.position.y;values[2]=body.position.z;
 values[3]=body.orientation.x;values[4]=body.orientation.y;values[5]=body.orientation.z;values[6]=body.orientation.w;
 for(int j=0;j<4;++j)values[7+j]=body.angle[j];for(int j=0;j<10;++j)values[11+j]=body.velocity[j];values[21]=body.work;values[22]=body.travel;}
__global__ void initialize_episodes(int n,int horizon,int case_count,bool withheld,bool sampled,m::World* worlds,m::World* initial_worlds,task::Task* tasks,task::Task* initial_tasks,EpisodeCounts* counts,const float* means,float* latent,float* logp,int* fault,uint32_t action_seed,uint32_t counter){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;counts[i]=EpisodeCounts{};
 for(int j=0;j<16;++j)latent[i*16+j]=means[j]+(sampled?policy::fixed_sigma*cudarl::rng::normal(action_seed,counter,uint32_t(i*16+j)):0.f);
 logp[i]=policy::log_probability(means,latent+i*16);if(!p::finite_number(logp[i])||!task::initialize(i%case_count,worlds[i],tasks[i],sampled?horizon:0,withheld))atomicCAS(fault,0,1);
 initial_worlds[i]=worlds[i];initial_tasks[i]=tasks[i];}
__global__ void observe_episodes(int n,const m::World* worlds,const task::Task* tasks,float* observation,float* raw,int* fault){const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(*fault||tasks[i].done){for(int j=0;j<41;++j)observation[i*41+j]=0;for(int j=0;j<8;++j)raw[i*8+j]=0;return;}
 if(!task::observe(tasks[i],worlds[i],observation+i*41,raw+i*8))atomicCAS(fault,0,2);}
__global__ void advance_episodes(int n,m::World* worlds,task::Task* tasks,EpisodeCounts* counts,const float* latent,const float* network_output,const float* raw,
 float* pre_filter,float* post_body,float* applied,float* reward,uint8_t* row_substeps,int* fault,int* active){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||*fault||tasks[i].done)return;record_filter(worlds[i],pre_filter+i*12);
 float z[4];policy::autonomous_latents(latent+i*16,network_output+i*4,raw+i*8,z);for(int j=0;j<4;++j)applied[i*4+j]=z[j];m::Action action;
 if(!m::action_from_latent(z,action)){atomicCAS(fault,0,3);return;}++counts[i].controls;task::StepInfo info;task::step_control(worlds[i],action.command,info);
 counts[i].attempted_substeps+=info.substeps;counts[i].accepted_substeps+=info.accepted_substeps;counts[i].last_substeps=info.accepted_substeps;counts[i].seconds+=info.seconds;
 record_body(worlds[i].physical.body,post_body+i*23);row_substeps[i]=uint8_t(info.accepted_substeps);
 if(info.hard_fault||!info.valid){counts[i].reason=3;counts[i].first_before=info.before;counts[i].rejected_candidate=info.candidate;atomicCAS(fault,0,4);return;}
 const auto interval=task::record(tasks[i],worlds[i],info);if(!interval.accepted||interval.truncated){counts[i].reason=3;atomicCAS(fault,0,5);return;}
 reward[i]=interval.reward;counts[i].reward_sum+=double(interval.reward);counts[i].done=interval.done;counts[i].truncated=interval.truncated;counts[i].reason=int(tasks[i].reason);
 if(!interval.done)atomicAdd(active,1);
}
__global__ void episode_loss(int n,const float* means,const float* latent,const float* oldp,const float* advantage,float* gradient,float* stats){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const auto row=policy::loss(means,latent+i*16,oldp[i],advantage[i],1.f/n,gradient+i*16);
 stats[i*5]=row.pg_loss;stats[i*5+1]=row.v_loss;stats[i*5+2]=row.entropy;stats[i*5+3]=row.approx_kl;stats[i*5+4]=row.clipped;}
}

namespace turning_study {
struct Collected {
 int n=0,horizon=0,ticks=0,purpose=0,index=0,case_count=0,fault=0;bool withheld=false;
 std::vector<float> means,latent,logp,returns,pre_filter,post_body,joint_latent,reward;
 std::vector<uint8_t> row_substeps;std::vector<EpisodeCounts> counts;
 std::vector<m::World> worlds,initial_worlds;std::vector<task::Task> tasks,initial_tasks;
 double probability_error=0,collection_seconds=0,evidence_seconds=0;
};
void save_batch(const Collected& batch,const fs::path& directory){cp::Writer meta;meta.value(batch_magic);meta.value(uint32_t{1});meta.value(study_protocol);
 for(int v:{batch.n,batch.horizon,batch.ticks,batch.purpose,batch.index,batch.case_count,int(batch.withheld),batch.fault})meta.value(v);
 meta.array(batch.means);meta.array(batch.latent);meta.array(batch.logp);meta.array(batch.returns);meta.value(uint32_t{sizeof(EpisodeCounts)});meta.array(batch.counts);
 meta.value(uint32_t{sizeof(m::World)});meta.array(batch.initial_worlds);meta.array(batch.worlds);meta.value(uint32_t{sizeof(task::Task)});meta.array(batch.initial_tasks);meta.array(batch.tasks);
 cp::write_atomic((directory/"meta.bin").string(),meta.data);const auto binding=cp::checksum(meta.data);
 const char* names[4]={"pre-filter","post-body","joint-latent","reward"};const int columns[4]={12,23,4,1};
 const std::vector<float>* arrays[4]={&batch.pre_filter,&batch.post_body,&batch.joint_latent,&batch.reward};
 std::ofstream manifest(directory/"manifest.json");manifest<<"{\"schema\":1,\"meta_checksum\":"<<binding<<",\"worlds\":"<<batch.n<<",\"horizon\":"<<batch.horizon<<",\"ticks\":"<<batch.ticks<<",\"purpose\":"<<batch.purpose<<",\"parts\":[";
 for(int field=0;field<4;++field){cp::Writer part;part.value(batch_magic);part.value(uint32_t{1});part.value(field);part.value(binding);part.value(batch.n);part.value(batch.ticks);part.value(columns[field]);part.array(*arrays[field]);
  require(part.data.size()<=512*1024*1024,"Turning evidence part exceeds512MiB");const std::string name=std::string(names[field])+".bin";cp::write_atomic((directory/name).string(),part.data);
  if(field)manifest<<',';manifest<<"{\"field\":"<<field<<",\"name\":"<<std::quoted(name)<<",\"columns\":"<<columns[field]<<",\"payload_bytes\":"<<part.data.size()<<",\"payload_checksum\":"<<cp::checksum(part.data)<<'}';}
 cp::Writer flags;flags.value(uint32_t{0x31465054});flags.value(uint32_t{1});flags.value(binding);flags.value(batch.n);flags.value(batch.ticks);flags.array(batch.row_substeps);cp::write_atomic((directory/"row-substeps.bin").string(),flags.data);
 manifest<<"],\"row_substeps_checksum\":"<<cp::checksum(flags.data)<<"}\n";manifest.close();require(bool(manifest),"Cannot save turning evidence manifest");
 std::ofstream csv(directory/"episodes.csv");csv<<std::setprecision(12)<<"episode,case,withheld,old_logp,reward_sum,return,actual_controls,attempted_substeps,accepted_substeps,last_substeps,elapsed_seconds,reason,done,truncated";
 for(int j=0;j<16;++j)csv<<",latent"<<j;csv<<'\n';
 for(int i=0;i<batch.n;++i){const auto& c=batch.counts[i];csv<<i<<','<<i%batch.case_count<<','<<batch.withheld<<','<<batch.logp[i]<<','<<c.reward_sum<<',';
  if(!batch.returns.empty())csv<<batch.returns[i];csv<<','<<c.controls<<','<<c.attempted_substeps<<','<<c.accepted_substeps<<','<<c.last_substeps<<','<<c.seconds<<','<<c.reason<<','<<c.done<<','<<c.truncated;
  for(int j=0;j<16;++j)csv<<','<<batch.latent[i*16+j];csv<<'\n';}
 csv.close();require(bool(csv),"Cannot save turning episode receipt");
}
Collected collect(Data& data,Budget& budget,const fs::path& directory,int purpose,int index,int case_count=16,bool withheld=false){
 time_guard();disk_guard(study_root,data.n,data.horizon);require(fs::create_directory(directory),"Turning collection identity already exists");
 require(budget.controls+data.n*data.horizon<=controls_cap&&budget.substeps+data.n*data.horizon*10<=substeps_cap,"Next turning collection exceeds cap");
 const auto begin=std::chrono::steady_clock::now();const auto stream=data.stream->value;const int grid=(data.n+127)/128,advance_grid=(data.n+63)/64;
 CUDA_CHECK_THROW(cudaMemsetAsync(data.fault.get(),0,sizeof(int),stream));
 for(auto* a:{&data.pre_filter,&data.post_body,&data.joint_latent,&data.reward})CUDA_CHECK_THROW(cudaMemsetAsync(a->get(),0,a->size_bytes(),stream));CUDA_CHECK_THROW(cudaMemsetAsync(data.row_substeps.get(),0,data.row_substeps.size_bytes(),stream));
 initialize_episodes<<<grid,128,0,stream>>>(data.n,data.horizon,case_count,withheld,purpose<2,data.worlds.get(),data.initial_worlds.get(),data.tasks.get(),data.initial_tasks.get(),data.counts.get(),data.means.get(),data.latent.get(),data.logp.get(),data.fault.get(),purpose==0?calibration_seed:training_seed,uint32_t(index));
 Collected batch;batch.n=data.n;batch.horizon=data.horizon;batch.purpose=purpose;batch.index=index;batch.case_count=case_count;batch.withheld=withheld;
 for(int control=1;control<=data.horizon;++control){time_guard();const size_t offset=size_t(control-1)*data.n;CUDA_CHECK_THROW(cudaMemsetAsync(data.active.get(),0,sizeof(int),stream));
  observe_episodes<<<grid,128,0,stream>>>(data.n,data.worlds.get(),data.tasks.get(),data.observation.get(),data.raw_features.get(),data.fault.get());
  data.backbone->forward(data.observation.get(),data.n,stream);
  advance_episodes<<<advance_grid,64,0,stream>>>(data.n,data.worlds.get(),data.tasks.get(),data.counts.get(),data.latent.get(),data.backbone->output(),data.raw_features.get(),data.pre_filter.get()+offset*12,data.post_body.get()+offset*23,data.joint_latent.get()+offset*4,data.reward.get()+offset,data.row_substeps.get()+offset,data.fault.get(),data.active.get());
  CUDA_CHECK_THROW(cudaGetLastError());batch.ticks=control;batch.fault=download(data.fault.get(),1,stream)[0];const auto alive=download(data.active.get(),1,stream)[0];if(batch.fault||alive==0)break;}
 batch.means=download(data.means.get(),16,stream);batch.latent=download(data.latent.get(),size_t(data.n)*16,stream);batch.logp=download(data.logp.get(),data.n,stream);
 batch.counts=download(data.counts.get(),data.n,stream);batch.worlds=download(data.worlds.get(),data.n,stream);batch.initial_worlds=download(data.initial_worlds.get(),data.n,stream);
 batch.tasks=download(data.tasks.get(),data.n,stream);batch.initial_tasks=download(data.initial_tasks.get(),data.n,stream);const size_t stored_rows=size_t(batch.ticks)*data.n;
 batch.pre_filter=download(data.pre_filter.get(),stored_rows*12,stream);batch.post_body=download(data.post_body.get(),stored_rows*23,stream);batch.joint_latent=download(data.joint_latent.get(),stored_rows*4,stream);batch.reward=download(data.reward.get(),stored_rows,stream);batch.row_substeps=download(data.row_substeps.get(),stored_rows,stream);
 for(const auto& c:batch.counts){budget.controls+=int(c.controls);budget.substeps+=int(c.attempted_substeps);}
 require(budget.controls<=controls_cap&&budget.substeps<=substeps_cap,"Actual turning counters exceed cap");
 if(!batch.fault){batch.returns.resize(data.n);double means[16];for(int j=0;j<16;++j)means[j]=batch.means[j];
  for(int i=0;i<data.n;++i){auto& c=batch.counts[i];const auto& t=batch.tasks[i];if(purpose>=2&&!t.done){c.truncated=true;c.done=false;}
   const bool terminal=purpose<2?(c.done&&!c.truncated&&t.done&&(c.reason==1||c.reason==2)):
     ((c.reason==1&&c.done&&t.done&&!c.truncated)||(c.reason==0&&!c.done&&!t.done&&c.truncated&&c.controls==uint64_t(data.horizon)));
   require(c.controls>0&&c.controls<=uint64_t(batch.ticks)&&c.controls==t.controls&&terminal&&c.reason==t.reason,"Turning episode terminal/count schema differs");
   require(c.attempted_substeps==c.accepted_substeps&&c.last_substeps>=1&&c.last_substeps<=10&&c.accepted_substeps==(c.controls-1)*10+uint64_t(c.last_substeps),"Turning partial-control accounting differs");
   unsigned long long steps=0;double rewards=0;for(int k=0;k<batch.ticks;++k){const size_t row=size_t(k)*data.n+i;const bool active=uint64_t(k)<c.controls;
    require(batch.row_substeps[row]==(active?(uint64_t(k)+1==c.controls?c.last_substeps:10):0),"Turning active/padding flags differ");
    if(active){steps+=batch.row_substeps[row];rewards+=batch.reward[row];}else require(batch.reward[row]==0,"Inactive turning reward is not zero");}
   require(steps==c.accepted_substeps&&std::abs(rewards-c.reward_sum)<1e-6&&c.reward_sum==t.reward_sum,"Turning reward/substep sums differ");
   batch.returns[i]=float(c.reward_sum/(data.horizon*.05));double u[16];for(int j=0;j<16;++j)u[j]=batch.latent[i*16+j];batch.probability_error=std::max(batch.probability_error,std::abs(policy::log_probability(means,u)-batch.logp[i]));}
  finite(batch.returns);require(batch.probability_error<1e-4,"Turning sampled episode probability mismatch");}
 batch.collection_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();const auto write_begin=std::chrono::steady_clock::now();save_batch(batch,directory);batch.evidence_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-write_begin).count();
 unsigned long long controls=0,substeps=0;for(const auto& c:batch.counts){controls+=c.controls;substeps+=c.attempted_substeps;}
 std::ofstream timing(directory/"timing.json");timing<<std::setprecision(17)<<"{\"worlds\":"<<data.n<<",\"horizon\":"<<data.horizon<<",\"stored_ticks\":"<<batch.ticks<<",\"actual_controls\":"<<controls<<",\"actual_substeps\":"<<substeps<<",\"collection_readback_validation_seconds\":"<<batch.collection_seconds<<",\"evidence_seconds\":"<<batch.evidence_seconds<<",\"fault\":"<<batch.fault<<",\"optimizer_steps\":0}\n";timing.close();require(bool(timing),"Cannot save turning timing");
 require(batch.fault==0,"Hard physical/numerical turning collection rejected before advantage/Adam");return batch;
}
struct Scale {double value=0;double means[16]{};};
Scale fit_scale(const std::vector<float>& returns){require(returns.size()==training_worlds,"Wrong turning calibration count");Scale scale;
 for(int i=0;i<training_worlds;++i)scale.means[i%16]+=returns[i];for(double& x:scale.means)x/=training_worlds/16;
 double square=0;for(int i=0;i<training_worlds;++i){const double e=returns[i]-scale.means[i%16];square+=e*e;}scale.value=sqrt(square/training_worlds);require(std::isfinite(scale.value)&&scale.value>0,"Invalid independent turning calibration RMS");return scale;}
std::vector<float> leave_one_out(const std::vector<float>& returns,double scale){require(returns.size()==training_worlds&&std::isfinite(scale)&&scale>0,"Wrong turning LOO input");double sums[16]{};for(int i=0;i<training_worlds;++i)sums[i%16]+=returns[i];std::vector<float> result(training_worlds);
 for(int i=0;i<training_worlds;++i)result[i]=float((returns[i]-(sums[i%16]-returns[i])/(training_worlds/16-1))/scale);return result;}
void save_policy(Data& data,const fs::path& directory,int update){const auto path=directory/("turning-policy-update-"+std::to_string(update)+".bin");require(!fs::exists(path),"Turning checkpoint identity exists");cp::Writer out,adam;
 if(data.optimizer)data.optimizer->save(adam);else{auto c=initial_optimizer_config(true);c.lr=.005f;nn::GpuAdam fresh(16,c);fresh.save(adam);}
 out.value(policy_magic);out.value(uint32_t{1});out.value(study_protocol);out.value(cp::checksum(parent_checkpoint_payload));out.value(cp::checksum(hold_checkpoint_payload));
 for(uint32_t v:{6u,1u,3u,4u,1u,41u,8u,4u,336u,16u})out.value(v);out.value(training_worlds);out.value(update);out.value(data.optimizer?data.optimizer->steps():0);
 out.array(data.backbone->get_params());out.array(saved_hold_parameters);out.value(saved_hold_gain);out.array(download(data.means.get(),16,data.stream->value));out.value(policy::fixed_sigma);out.value(policy::steering_limit);out.array(adam.data);
 cp::write_atomic(path.string(),out.data);require(cp::read(path.string())==out.data,"Turning checkpoint exact roundtrip differs");}
void check_identity(const std::vector<float>& means){require(means.size()==16,"Wrong turning mean count");finite(means);const float zero[16]{};
 for(int row=0;row<32;++row){float base[4],raw[8],output[4];for(int j=0;j<4;++j)base[j]=float(row-16)*.1f+j*.13f;for(int j=0;j<8;++j)raw[j]=j<3?.1f*row:float(j-5)*.15f;
  policy::autonomous_latents(zero,base,raw,output);for(int j=0;j<4;++j)require(output[j]==base[j],"Turning zero identity changed body output");
  policy::motor_latents(false,true,means.data(),base,saved_hold_gain,raw,output);for(int j=0;j<4;++j)require(output[j]==saved_hold_gain*base[j],"Turning head changed saved holding branch");
  policy::motor_latents(false,false,means.data(),base,saved_hold_gain,raw,output);for(int j=0;j<4;++j)require(output[j]==base[j],"Turning head changed requested swimming branch");}}
double optimize(Data& data,const Collected& batch,const Scale& scale,int update,const fs::path& directory){require(batch.n==training_worlds&&batch.horizon==episode_controls&&batch.purpose==1&&batch.fault==0,"Only fresh complete turning episodes can reach PPO");
 const auto advantage=leave_one_out(batch.returns,scale.value);data.advantages.copy_from_host(advantage.data(),advantage.size());cp::Writer evidence;evidence.value(uint32_t{0x31415054});evidence.value(uint32_t{1});evidence.value(update);evidence.value(training_worlds);evidence.value(scale.value);evidence.array(batch.returns);evidence.array(advantage);cp::write_atomic((directory/("advantages-"+std::to_string(update)+".bin")).string(),evidence.data);
 const auto stream=data.stream->value;const int before=data.optimizer->steps();double kl=0;
 for(int epoch=0;epoch<4;++epoch){CUDA_CHECK_THROW(cudaMemsetAsync(data.stats.get(),0,5*sizeof(float),stream));const auto order=rl::make_permutation(training_worlds,0x0316412du+uint32_t(update*4+epoch));
  for(int first=0;first<training_worlds;first+=minibatch_episodes){time_guard();rl::gather_rows(order,first,minibatch_episodes,16,data.latent.get(),data.mb_latent.get(),stream);rl::gather_rows(order,first,minibatch_episodes,1,data.logp.get(),data.mb_logp.get(),stream);rl::gather_rows(order,first,minibatch_episodes,1,data.advantages.get(),data.mb_advantage.get(),stream);
   episode_loss<<<8,128,0,stream>>>(minibatch_episodes,data.means.get(),data.mb_latent.get(),data.mb_logp.get(),data.mb_advantage.get(),data.row_gradient.get(),data.row_stats.get());CUDA_CHECK_THROW(cudaMemsetAsync(data.gradient.get(),0,16*sizeof(float),stream));rl::sum_columns(data.row_gradient.get(),minibatch_episodes,16,data.gradient.get(),stream);rl::sum_columns(data.row_stats.get(),minibatch_episodes,5,data.stats.get(),stream);
   finite(download(data.gradient.get(),16,stream));data.optimizer->step(data.means.get(),data.gradient.get(),stream);finite(download(data.means.get(),16,stream));}
  const auto stats=download(data.stats.get(),5,stream);finite(stats);kl=stats[3]/training_worlds;if(kl>.01)break;}
 require(data.optimizer->steps()-before>=4&&data.optimizer->steps()-before<=16&&data.optimizer->steps()<=maximum_adam_steps,"Turning Adam budget exceeded");require(data.backbone->get_params()==resume_state->actor.parameters,"Frozen swimming backbone changed during turning PPO");return kl;}
}

namespace turning_study {
p::BodyState decode_recorded_body(const float* x){p::BodyState b;b.position={x[0],x[1],x[2]};b.orientation={x[3],x[4],x[5],x[6]};for(int j=0;j<4;++j)b.angle[j]=x[7+j];for(int j=0;j<10;++j)b.velocity[j]=x[11+j];b.work=x[21];b.travel=x[22];return b;}
struct Evaluation {int passes=0,survivors=0,completed=0;double minimum_mean_speed=0,minimum_window_speed=0;};
Evaluation score_evaluation(const Collected& batch,const fs::path& directory,bool long_validation){require(batch.purpose>=2&&batch.n==batch.case_count,"Wrong deterministic turning evaluation shape");
 std::ofstream summary(directory/"summary.csv"),windows(directory/"motion-windows.csv");summary<<std::setprecision(12);windows<<std::setprecision(12);
 summary<<"case,withheld,passed,survived,actual_controls,attempted_substeps,accepted_substeps,elapsed_seconds,task_done,external_truncated,reason,horizontal_COM_arc_m,mean_speed_mm_s,min_window_speed_mm_s,x_span_m,z_span_m,maximum_depth_mm,maximum_tilt_deg,positive_turn_rad,negative_turn_rad,net_unwrapped_yaw_rad,yaw_range_rad,raw_absolute_yaw_rad\n";
 windows<<"case,start_s,end_s,actual_seconds,horizontal_COM_arc_m,mean_speed_mm_s,passed\n";Evaluation result;
 for(int i=0;i<batch.n;++i){const auto& c=batch.counts[i];const auto& t=batch.tasks[i];auto previous=batch.initial_worlds[i].physical.body;auto previous_com=cudarl::motor::motion(previous).com_model*p::scale;
  const auto first_forward=task::forward(previous);double previous_yaw=atan2(first_forward.z,first_forward.x),unwrapped=previous_yaw,anchor=unwrapped,first_yaw=unwrapped,min_yaw=unwrapped,max_yaw=unwrapped,raw_absolute=0,positive=0,negative=0;
  double min_x=previous_com.x,max_x=min_x,min_z=previous_com.z,max_z=min_z,arc=0,window_arc=0,window_seconds=0,min_window=1e30;int window_count=0;
  for(unsigned long long k=0;k<c.controls;++k){const size_t row=size_t(k)*batch.n+i;const auto body=decode_recorded_body(batch.post_body.data()+row*23);const auto center=cudarl::motor::motion(body).com_model*p::scale;
   const auto displacement=center-previous_com;const double length=sqrt(double(displacement.x)*displacement.x+double(displacement.z)*displacement.z);arc+=length;
   min_x=std::min(min_x,double(center.x));max_x=std::max(max_x,double(center.x));min_z=std::min(min_z,double(center.z));max_z=std::max(max_z,double(center.z));
   const auto f=task::forward(body);const double yaw=atan2(f.z,f.x);double delta=yaw-previous_yaw;while(delta>cudarl::movement::pi)delta-=2*cudarl::movement::pi;while(delta< -cudarl::movement::pi)delta+=2*cudarl::movement::pi;
   unwrapped+=delta;raw_absolute+=fabs(delta);min_yaw=std::min(min_yaw,unwrapped);max_yaw=std::max(max_yaw,unwrapped);const double event=cudarl::movement::pi/4;
   while(unwrapped-anchor>=event){positive+=event;anchor+=event;}while(unwrapped-anchor<= -event){negative-=event;anchor-=event;}
   const int control=int(k+1);if(control>80){window_arc+=length;window_seconds+=float(batch.row_substeps[row])*.005f;}
   if(control>=240&&(control-80)%160==0){const double speed=window_arc/window_seconds;min_window=std::min(min_window,speed);++window_count;
    windows<<i<<','<<(control-160)*.05<<','<<control*.05<<','<<window_seconds<<','<<window_arc<<','<<speed*1000<<','<<(speed>=.003)<<'\n';window_arc=window_seconds=0;}
   previous_com=center;previous_yaw=yaw;
  }
  const bool survived=c.controls==uint64_t(batch.horizon)&&c.reason==0&&!c.done&&c.truncated&&!t.done&&t.maximum_controls==0;
  const double elapsed=c.seconds,mean_speed=elapsed>0?arc/elapsed:0,min_speed=window_count?min_window:0,x_span=max_x-min_x,z_span=max_z-min_z;
  const bool passed=survived&&t.maximum_depth_m<=.001f&&t.maximum_tilt_deg<=10.f&&mean_speed>=.006&&min_speed>=.003&&
    x_span>=(long_validation?.20:.12)&&z_span>=(long_validation?.12:.06)&&(!long_validation||positive+fabs(negative)>=2*cudarl::movement::pi);
  result.passes+=passed;result.survivors+=survived;++result.completed;if(i==0){result.minimum_mean_speed=mean_speed;result.minimum_window_speed=min_speed;}else{result.minimum_mean_speed=std::min(result.minimum_mean_speed,mean_speed);result.minimum_window_speed=std::min(result.minimum_window_speed,min_speed);}
  summary<<i<<','<<batch.withheld<<','<<passed<<','<<survived<<','<<c.controls<<','<<c.attempted_substeps<<','<<c.accepted_substeps<<','<<elapsed<<','<<c.done<<','<<c.truncated<<','<<c.reason<<','<<arc<<','<<mean_speed*1000<<','<<min_speed*1000<<','<<x_span<<','<<z_span<<','<<t.maximum_depth_m*1000<<','<<t.maximum_tilt_deg<<','<<positive<<','<<negative<<','<<unwrapped-first_yaw<<','<<max_yaw-min_yaw<<','<<raw_absolute<<'\n';
 }
 summary.close();windows.close();require(bool(summary)&&bool(windows),"Cannot save continuous turning scores");return result;
}
Evaluation assess_continuous(const std::vector<float>& means,const fs::path& directory,Budget& budget,bool withheld=false,bool ablation=false){
 const int n=withheld?8:16,probe_limit=withheld?36000:1200;disk_guard(study_root,n,probe_limit);Data data(n,probe_limit,false);auto applied_means=means;
 if(ablation)for(int j=0;j<16;++j)if(j!=0&&j!=8)applied_means[j]=0;
 data.means.copy_from_host(applied_means.data(),applied_means.size());
 const auto batch=collect(data,budget,directory,ablation?4:2,0,n,withheld);return score_evaluation(batch,directory,withheld);
}
}

namespace turning_study {
__global__ void decode_fixture(float* output,float hold_gain){const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=96)return;float u[16],base[4],raw[8],autonomous[4],hold[4],requested[4];const float samples[5]={-80.f,-.2f,0,.2f,80.f};
 for(int j=0;j<16;++j)u[j]=i==0?0.f:samples[(i+j)%5];for(int j=0;j<4;++j)base[j]=float((i+j*3)%23-11)*.17f;
 for(int j=0;j<8;++j)raw[j]=j<3?float((i+j)%17)*.2f:float((i+j)%11-5)*.2f;
 policy::autonomous_latents(u,base,raw,autonomous);policy::motor_latents(false,true,u,base,hold_gain,raw,hold);policy::motor_latents(false,false,u,base,hold_gain,raw,requested);
 for(int j=0;j<16;++j)output[i*40+j]=u[j];for(int j=0;j<4;++j){output[i*40+16+j]=base[j];output[i*40+28+j]=autonomous[j];output[i*40+32+j]=hold[j];output[i*40+36+j]=requested[j];}for(int j=0;j<8;++j)output[i*40+20+j]=raw[j];}
void numerical(const fs::path& directory){require(fs::create_directory(directory),"Turning numerical identity exists");check_identity(std::vector<float>(16,1e6f));
 Stream stream;device_ptr<float> decoded(96*40);decode_fixture<<<1,128,0,stream.value>>>(decoded.get(),saved_hold_gain);const auto decode=download(decoded.get(),96*40,stream.value);double max_decode=0;
 for(int i=0;i<96;++i){double u[16],base[4],raw[8],expected[4];for(int j=0;j<16;++j)u[j]=decode[i*40+j];for(int j=0;j<4;++j)base[j]=decode[i*40+16+j];for(int j=0;j<8;++j)raw[j]=decode[i*40+20+j];policy::autonomous_latents(u,base,raw,expected);
  for(int j=0;j<4;++j){max_decode=std::max(max_decode,std::abs(expected[j]-decode[i*40+28+j]));require(decode[i*40+32+j]==saved_hold_gain*decode[i*40+16+j]&&decode[i*40+36+j]==decode[i*40+16+j],"Actual CUDA requested/hold identity changed");if(i==0)require(decode[i*40+28+j]==decode[i*40+16+j],"Actual CUDA zero autonomous identity changed");}}
 cp::Writer decode_proof;decode_proof.value(uint32_t{0x31445054});decode_proof.value(uint32_t{1});decode_proof.value(saved_hold_gain);decode_proof.array(decode);cp::write_atomic((directory/"decode-zero.bin").string(),decode_proof.data);require(max_decode<2e-6,"Turning CPU/CUDA decode failed");
 device_ptr<float> dmeans(16),dlatent(minibatch_episodes*16),dold(minibatch_episodes),dadv(minibatch_episodes),dgradient(minibatch_episodes*16),dstats(minibatch_episodes*5);
 std::vector<float> means(16),latent(minibatch_episodes*16),oldp(minibatch_episodes),adv(minibatch_episodes);for(int j=0;j<16;++j)means[j]=float(j-8)*.025f;
 for(int i=0;i<minibatch_episodes;++i){for(int j=0;j<16;++j)latent[i*16+j]=means[j]+float((i*7+j*3)%19-9)*.02f;const float ratios[3]={.7f,1,1.3f};oldp[i]=policy::log_probability(means.data(),latent.data()+i*16)-logf(ratios[i%3]);adv[i]=i%2?.7f:-.8f;}
 dmeans.copy_from_host(means.data(),means.size());dlatent.copy_from_host(latent.data(),latent.size());dold.copy_from_host(oldp.data(),oldp.size());dadv.copy_from_host(adv.data(),adv.size());episode_loss<<<8,128,0,stream.value>>>(minibatch_episodes,dmeans.get(),dlatent.get(),dold.get(),dadv.get(),dgradient.get(),dstats.get());
 const auto gradients=download(dgradient.get(),minibatch_episodes*16,stream.value),stats=download(dstats.get(),minibatch_episodes*5,stream.value);double max_gradient=0,max_stats=0,max_fd=0;double parameter[16];for(int j=0;j<16;++j)parameter[j]=means[j];
 for(int i=0;i<minibatch_episodes;++i){double u[16],g[16];for(int j=0;j<16;++j)u[j]=latent[i*16+j];const auto expected=policy::loss(parameter,u,double(oldp[i]),double(adv[i]),1./minibatch_episodes,g);
  const double fields[5]={expected.pg_loss,expected.v_loss,expected.entropy,expected.approx_kl,expected.clipped};for(int j=0;j<16;++j)max_gradient=std::max(max_gradient,std::abs(g[j]-gradients[i*16+j]));for(int j=0;j<5;++j)max_stats=std::max(max_stats,std::abs(fields[j]-stats[i*5+j]));
  if(i<24)for(int j=0;j<16;++j){const double original=parameter[j],eps=1e-6;double scratch[16];parameter[j]=original+eps;const auto plus=policy::loss(parameter,u,double(oldp[i]),double(adv[i]),1./minibatch_episodes,scratch);parameter[j]=original-eps;const auto minus=policy::loss(parameter,u,double(oldp[i]),double(adv[i]),1./minibatch_episodes,scratch);parameter[j]=original;max_fd=std::max(max_fd,std::abs((plus.pg_loss-minus.pg_loss)/(2*eps*minibatch_episodes)-g[j]));}}
 cp::Writer proof;proof.value(uint32_t{0x314d5054});proof.value(uint32_t{1});proof.array(means);proof.array(latent);proof.array(oldp);proof.array(adv);proof.array(gradients);proof.array(stats);cp::write_atomic((directory/"gradient-zero.bin").string(),proof.data);
 std::ofstream diagnostic(directory/"diagnostic.json");diagnostic<<std::setprecision(17)<<"{\"max_decode_error\":"<<max_decode<<",\"max_gradient_error\":"<<max_gradient<<",\"max_stats_error\":"<<max_stats<<",\"max_mean_loss_FD_error\":"<<max_fd<<",\"physical_controls\":0,\"optimizer_steps\":0}\n";diagnostic.close();require(bool(diagnostic),"Cannot preserve turning numerical diagnostics");require(max_gradient<2e-7&&max_stats<2e-4&&max_fd<2e-7,"Turning numerical PPO/mean-loss FD failed");
 std::vector<float> returns(training_worlds);for(int i=0;i<training_worlds;++i)returns[i]=float(i%16)+float(i/16)*.001f;const auto before=leave_one_out(returns,2.);const int selected=19;const double own_baseline=returns[selected]-before[selected]*2;returns[selected]+=7;const auto after=leave_one_out(returns,2.);require(std::abs(own_baseline-(returns[selected]-after[selected]*2))<2e-5,"Own turning return entered LOO baseline");
 auto cfg=initial_optimizer_config(true);cfg.lr=.005f;nn::GpuAdam first(16,cfg),second(16,cfg);cp::Writer one,two;first.save(one);cp::Reader reader{one.data};second.load(reader);reader.finish();second.save(two);require(first.steps()==0&&second.steps()==0&&one.data==two.data,"Fresh turning Adam roundtrip failed");
 std::ofstream report(directory/"math-zero.json");report<<"{\"passed\":true,\"CUDA_decode_cases\":96,\"PPO_rows\":1024,\"LOO_own_return_excluded\":true,\"physical_controls\":0,\"optimizer_steps\":0}\n";
}
void bulk_policy_zero(Data& data,const std::vector<float>& latent,const std::vector<float>& old_logp,const fs::path& directory) {
    const int n=training_worlds,mb=minibatch_episodes;const auto stream=data.stream->value;std::vector<float> advantage(n);
    for(int i=0;i<n;++i)advantage[i]=.25f+float(i%19-9)*.07f;
    data.advantages.copy_from_host(advantage.data(),n);const auto permutation=rl::make_permutation(n,0x7138b2a9u);
    std::vector<int> seen(n,0);double maximum_row=0,maximum_sum=0,maximum_stat=0,maximum_last=0;int checked=0,last_rows=0;
    for(int first=0;first<n;first+=mb) {
        rl::gather_rows(permutation,first,mb,head_parameters,data.latent.get(),data.mb_latent.get(),stream);
        rl::gather_rows(permutation,first,mb,1,data.logp.get(),data.mb_logp.get(),stream);
        rl::gather_rows(permutation,first,mb,1,data.advantages.get(),data.mb_advantage.get(),stream);
        CUDA_CHECK_THROW(cudaMemsetAsync(data.row_gradient.get(),0,size_t(mb)*16*sizeof(float),stream));
        CUDA_CHECK_THROW(cudaMemsetAsync(data.row_stats.get(),0,size_t(mb)*5*sizeof(float),stream));
        episode_loss<<<(mb+127)/128,128,0,stream>>>(mb,data.means.get(),data.mb_latent.get(),data.mb_logp.get(),data.mb_advantage.get(),data.row_gradient.get(),data.row_stats.get());
        CUDA_CHECK_THROW(cudaMemsetAsync(data.gradient.get(),0,16*sizeof(float),stream));CUDA_CHECK_THROW(cudaMemsetAsync(data.stats.get(),0,5*sizeof(float),stream));
        rl::sum_columns(data.row_gradient.get(),mb,16,data.gradient.get(),stream);rl::sum_columns(data.row_stats.get(),mb,5,data.stats.get(),stream);
        const auto gathered=download(data.mb_latent.get(),size_t(mb)*16,stream),gp=download(data.mb_logp.get(),mb,stream),
          ga=download(data.mb_advantage.get(),mb,stream),rows_gradient=download(data.row_gradient.get(),size_t(mb)*16,stream),
          sum=download(data.gradient.get(),16,stream),stats=download(data.stats.get(),5,stream);
        double expected_sum[16]{},expected_loss=0;const double fixed_log_sigma=std::log(double(policy::fixed_sigma)),entropy=16*(fixed_log_sigma+.5*(1.83787706640934548+1));
        for(int row=0;row<mb;++row) {
            const int original=int(rl::permute(permutation,uint32_t(first+row)));require(original>=0&&original<n&&++seen[original]==1,"Turning bulk permutation coverage defect");
            require(gp[row]==old_logp[original]&&ga[row]==advantage[original],"Turning bulk scalar gather defect");
            double logp=-16*fixed_log_sigma-8*1.83787706640934548;for(int j=0;j<16;++j) {
                require(gathered[row*16+j]==latent[original*16+j],"Turning bulk latent gather defect");
                const double z=double(gathered[row*16+j])/double(policy::fixed_sigma);logp-=.5*z*z;
            }
            const double ratio=std::exp(logp-double(gp[row]));require(std::abs(ratio-1)<.001,"Synthetic turning ratio left unclipped fixture");
            expected_loss-=double(ga[row])*ratio;
            double last_magnitude=0;
            for(int j=0;j<16;++j) {
                const double expected=-double(ga[row])*ratio*double(gathered[row*16+j])/(double(policy::fixed_sigma)*policy::fixed_sigma)/mb;
                const double error=std::abs(expected-rows_gradient[row*16+j]);maximum_row=std::max(maximum_row,error);expected_sum[j]+=expected;
                if(row==mb-1){maximum_last=std::max(maximum_last,error);last_magnitude+=std::abs(expected);}
            }
            if(row==mb-1){require(last_magnitude>1e-8,"Turning bulk last-row fixture is vacuous");++last_rows;}
            ++checked;
        }
        for(int j=0;j<16;++j)maximum_sum=std::max(maximum_sum,std::abs(expected_sum[j]-sum[j]));
        maximum_stat=std::max(maximum_stat,std::abs(expected_loss/mb-double(stats[0])/mb));
        maximum_stat=std::max(maximum_stat,std::abs(entropy-double(stats[2])/mb));
        require(stats[1]==0&&std::abs(stats[3]/mb)<1e-5&&stats[4]==0,"Turning bulk critic/KL/clipping fixture defect");
    }
    for(int count:seen)require(count==1,"Turning bulk minibatches missed an episode");
    require(checked==n&&last_rows==4&&maximum_row<1e-5&&maximum_sum<2e-5&&maximum_stat<2e-5,"Turning bulk native gradient/reduction gate failed");
    const auto means=download(data.means.get(),16,stream);for(float x:means)require(x==0,"Turning bulk gradient fixture updated policy");
    std::ofstream report(directory/"bulk-policy-zero.json");report<<std::setprecision(17)<<"{\"passed\":true,\"worlds\":"<<n
      <<",\"minibatch_episodes\":"<<mb<<",\"loss_grid_blocks\":"<<(mb+127)/128<<",\"checked_rows\":"<<checked<<",\"checked_last_rows\":"<<last_rows
      <<",\"max_row_gradient_error\":"<<maximum_row<<",\"max_reduced_gradient_error\":"<<maximum_sum<<",\"max_stats_error\":"<<maximum_stat
      <<",\"max_last_row_gradient_error\":"<<maximum_last<<",\"physical_controls\":0,\"optimizer_steps\":0}\n";
}


void zero_initialization(const fs::path& directory){require(fs::create_directory(directory),"Turning zero initialization identity exists");disk_guard(study_root,training_worlds,episode_controls);Data data;const auto stream=data.stream->value;CUDA_CHECK_THROW(cudaMemsetAsync(data.fault.get(),0,sizeof(int),stream));
 initialize_episodes<<<32,128,0,stream>>>(data.n,data.horizon,16,false,true,data.worlds.get(),data.initial_worlds.get(),data.tasks.get(),data.initial_tasks.get(),data.counts.get(),data.means.get(),data.latent.get(),data.logp.get(),data.fault.get(),calibration_seed,0);
 observe_episodes<<<32,128,0,stream>>>(data.n,data.worlds.get(),data.tasks.get(),data.observation.get(),data.raw_features.get(),data.fault.get());data.backbone->forward(data.observation.get(),data.n,stream);
 const auto worlds=download(data.worlds.get(),data.n,stream);const auto tasks=download(data.tasks.get(),data.n,stream);const auto input=download(data.observation.get(),size_t(data.n)*41,stream),raw=download(data.raw_features.get(),size_t(data.n)*8,stream),output=download(data.backbone->output(),size_t(data.n)*4,stream),latent=download(data.latent.get(),size_t(data.n)*16,stream),logp=download(data.logp.get(),data.n,stream);
 require(download(data.fault.get(),1,stream)[0]==0,"Turning zero reset/observation fault");bulk_policy_zero(data,latent,logp,directory);nn::CpuMlp<float> cpu(spec(4));cpu.params()=resume_state->actor.parameters;double input_error=0,sensor_error=0,backbone_error=0,rng_error=0;
 for(int i=0;i<data.n;++i){m::World world;task::Task t;require(task::initialize(i%16,world,t,320,false),"CPU turning zero reset rejected");float expected[41],sensors[8];require(task::observe(t,world,expected,sensors),"CPU turning zero observe rejected");require(worlds[i].attempted_substeps==0&&worlds[i].physical.substeps==0&&tasks[i].controls==0,"Turning zero fixture advanced physics");
  for(int j=0;j<41;++j)input_error=std::max(input_error,double(std::abs(expected[j]-input[i*41+j])));for(int j=0;j<8;++j)sensor_error=std::max(sensor_error,double(std::abs(sensors[j]-raw[i*8+j])));cpu.forward(expected,1);for(int j=0;j<4;++j)backbone_error=std::max(backbone_error,double(std::abs(cpu.output()[j]-output[i*4+j])));
  for(int j=0;j<16;++j)rng_error=std::max(rng_error,double(std::abs(latent[i*16+j]-policy::fixed_sigma*cudarl::rng::normal(calibration_seed,0,uint32_t(i*16+j)))));}
 cp::Writer proof;proof.value(uint32_t{0x314e5054});proof.value(uint32_t{1});proof.value(data.n);proof.value(uint32_t{sizeof(m::World)});proof.array(worlds);proof.value(uint32_t{sizeof(task::Task)});proof.array(tasks);proof.array(input);proof.array(raw);proof.array(output);proof.array(latent);proof.array(logp);cp::write_atomic((directory/"initialization.bin").string(),proof.data);
 std::ofstream report(directory/"initialization-zero.json");report<<std::setprecision(17)<<"{\"passed\":"<<((input_error<2e-5&&sensor_error<2e-5&&backbone_error<2e-5&&rng_error<1e-6)?"true":"false")<<",\"worlds\":4096,\"input_error\":"<<input_error<<",\"sensor_error\":"<<sensor_error<<",\"backbone_error\":"<<backbone_error<<",\"rng_error\":"<<rng_error<<",\"physical_controls\":0,\"optimizer_steps\":0}\n";report.close();require(bool(report),"Cannot preserve turning zero diagnostics");require(input_error<2e-5&&sensor_error<2e-5&&backbone_error<2e-5&&rng_error<1e-6,"Turning CPU/CUDA zero observation/policy/RNG failed");
 cudaDeviceProp properties{};CUDA_CHECK_THROW(cudaGetDeviceProperties(&properties,0));cudaFuncAttributes attributes{};CUDA_CHECK_THROW(cudaFuncGetAttributes(&attributes,advance_episodes));int active=0;CUDA_CHECK_THROW(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,advance_episodes,64,0));std::ofstream gpu(directory/"gpu-kernel-metadata.json");gpu<<"{\"device\":"<<std::quoted(properties.name)<<",\"SM_count\":"<<properties.multiProcessorCount<<",\"VRAM_bytes\":"<<properties.totalGlobalMem<<",\"registers\":"<<attributes.numRegs<<",\"local_bytes\":"<<attributes.localSizeBytes<<",\"active_blocks_per_SM\":"<<active<<",\"advance_block_threads\":64,\"grid_blocks\":64,\"physical_controls\":0,\"optimizer_steps\":0}\n";
}
}

#include "turning_retention.h"
namespace turning_study {
std::vector<uint8_t> protocol_bytes(const fs::path& root){cp::Writer out;out.value(protocol_magic);out.value(uint32_t{1});out.value(file_hash(root/"approved-protocol.md"));out.value(cp::checksum(parent_checkpoint_payload));out.value(cp::checksum(hold_checkpoint_payload));
 for(int v:{6,1,3,4,1,41,8,4,336,16,training_worlds,episode_controls,maximum_updates,maximum_adam_steps,maximum_epochs,minibatch_episodes,gate_controls,10720,46240,16,1200,8,36000,controls_cap,substeps_cap,900})out.value(v);
 for(float v:{policy::fixed_sigma,policy::steering_limit,.005f,.5f,.2f,.01f,.20f,.12f,.195f,.115f,.008f,.001f,10.f,.006f,.003f,.12f,.06f,.20f,.12f,4.f,16.f,6250.f})out.value(v);
 out.value(calibration_seed);out.value(training_seed);out.value(uint32_t{0x0316412d});out.value(projected_study_bytes);out.value(working_disk_reserve);
 for(bool withheld:{false,true})for(int i=0;i<(withheld?8:16);++i){const auto c=task::case_for(i,withheld);out.value(c.com_m);out.value(c.yaw);out.value(c.pose);out.value(c.id);out.value(uint32_t(withheld));}return out.data;}
void validate_protocol(const fs::path& root){const auto bytes=protocol_bytes(root);require(cp::read((root/"protocol.bin").string())==bytes,"Frozen continuous pond protocol differs");study_protocol=cp::checksum(bytes);}
void write_layout(const fs::path& root){std::ofstream out(root/"layout.json");out<<"{\"counts\":{\"size\":"<<sizeof(EpisodeCounts)
 <<",\"controls\":"<<offsetof(EpisodeCounts,controls)<<",\"attempted_substeps\":"<<offsetof(EpisodeCounts,attempted_substeps)<<",\"accepted_substeps\":"<<offsetof(EpisodeCounts,accepted_substeps)
 <<",\"reward_sum\":"<<offsetof(EpisodeCounts,reward_sum)<<",\"seconds\":"<<offsetof(EpisodeCounts,seconds)<<",\"last_substeps\":"<<offsetof(EpisodeCounts,last_substeps)<<",\"reason\":"<<offsetof(EpisodeCounts,reason)
 <<",\"done\":"<<offsetof(EpisodeCounts,done)<<",\"truncated\":"<<offsetof(EpisodeCounts,truncated)<<",\"first_before\":"<<offsetof(EpisodeCounts,first_before)<<",\"rejected_candidate\":"<<offsetof(EpisodeCounts,rejected_candidate)
 <<"},\"task\":{\"size\":"<<sizeof(task::Task)<<",\"controls\":"<<offsetof(task::Task,controls)<<",\"last_world_control\":"<<offsetof(task::Task,last_world_control)<<",\"maximum_controls\":"<<offsetof(task::Task,maximum_controls)
 <<",\"reason\":"<<offsetof(task::Task,reason)<<",\"fixed_depth_model\":"<<offsetof(task::Task,fixed_depth_model)<<",\"maximum_depth_m\":"<<offsetof(task::Task,maximum_depth_m)<<",\"maximum_tilt_deg\":"<<offsetof(task::Task,maximum_tilt_deg)
 <<",\"minimum_clearance_m\":"<<offsetof(task::Task,minimum_clearance_m)<<",\"last_com_model\":"<<offsetof(task::Task,last_com_model)<<",\"elapsed_seconds\":"<<offsetof(task::Task,elapsed_seconds)<<",\"reward_sum\":"<<offsetof(task::Task,reward_sum)
 <<",\"path_length_m\":"<<offsetof(task::Task,path_length_m)<<",\"done\":"<<offsetof(task::Task,done)<<",\"boundary_terminal\":"<<offsetof(task::Task,boundary_terminal)
 <<"},\"world_size\":"<<sizeof(m::World)<<",\"body_size\":"<<sizeof(p::BodyState)<<",\"step_info_size\":"<<sizeof(task::StepInfo)<<",\"float_columns\":40,\"substep_flag_bytes\":1}\n";out.close();require(bool(out),"Cannot save continuous ABI layout");}
struct Prior {int controls=0,substeps=0;double seconds=0;};
Prior passed_gate(const fs::path& root){const auto bytes=cp::read((root/"gate-passed.bin").string());cp::Reader in{bytes};require(in.value<uint32_t>()==0x31504754&&in.value<uint32_t>()==1&&in.value<uint32_t>()==1,"Matched-command gate has not passed");
 require(in.value<uint64_t>()==2480&&in.value<uint64_t>()==24800&&in.value<uint64_t>()==study_protocol,"Matched-command gate protocol/count differs");in.finish();Prior prior;prior.controls=2480;prior.substeps=24800;
 std::ifstream active(root/"gate-started/active-seconds.txt");active>>prior.seconds;require(bool(active)&&std::isfinite(prior.seconds)&&prior.seconds>0&&prior.seconds<seconds_cap,"Invalid gate active-time receipt");return prior;}
std::vector<fs::path> phase_bindings(const fs::path& root,int phase){if(phase==1)return {root/"baseline-result.json",root/"baseline-retention/retention.json",root/"baseline-free/summary.csv",root/"baseline-free/manifest.json",root/"gate-passed.bin",root/"gate-result.json"};
 return {root/"calibration-result.json",root/"calibration-report.json",root/"fixed-scale.bin",root/"calibration-started/calibration-batch-0/manifest.json",root/"baseline-ready.bin"};}
void write_ready(const fs::path& root,int phase,const Budget& budget){cp::Writer out;out.value(ready_magic);out.value(uint32_t{1});out.value(study_protocol);out.value(phase);out.value(budget.controls);out.value(budget.substeps);out.value(prior_seconds+std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count());std::vector<uint64_t> hashes;
 for(const auto& path:phase_bindings(root,phase))hashes.push_back(file_hash(path));out.array(hashes);const auto path=root/(phase==1?"baseline-ready.bin":"calibration-ready.bin");require(!fs::exists(path),"Continuous phase release exists");cp::write_atomic(path.string(),out.data);}
Prior read_ready(const fs::path& root,int phase){const auto bytes=cp::read((root/(phase==1?"baseline-ready.bin":"calibration-ready.bin")).string());cp::Reader in{bytes};require(in.value<uint32_t>()==ready_magic&&in.value<uint32_t>()==1&&in.value<uint64_t>()==study_protocol&&in.value<int>()==phase,"Wrong continuous phase release");Prior prior;
 prior.controls=in.value<int>();prior.substeps=in.value<int>();prior.seconds=in.value<double>();const auto hashes=in.array<uint64_t>(16);in.finish();const auto paths=phase_bindings(root,phase);require(paths.size()==hashes.size(),"Continuous release hash count differs");for(size_t i=0;i<paths.size();++i)require(hashes[i]==file_hash(paths[i]),"Preceding continuous evidence changed");
 require(prior.controls>=2480&&prior.controls<=controls_cap&&prior.substeps>=prior.controls&&prior.substeps<=prior.controls*10&&prior.substeps<=substeps_cap&&std::isfinite(prior.seconds)&&prior.seconds>0&&prior.seconds<900,"Continuous preceding counters/time invalid");return prior;}
Budget begin_phase(const fs::path& root,const char* phase,const Prior& prior){prior_seconds=prior.seconds;started=std::chrono::steady_clock::now();active_wall_cap=seconds_cap-prior_seconds;const auto directory=root/(std::string(phase)+"-started");require(fs::create_directory(directory),"Continuous phase already attempted; no retry");failure_directory=directory;failure_saved=false;Budget budget;
 budget.controls=prior.controls;budget.substeps=prior.substeps;budget.maximum_controls=controls_cap;budget.maximum_substeps=substeps_cap;budget.seconds=active_wall_cap;return budget;}
void nullable(std::ostream& out,bool measured,int value){if(measured)out<<value;else out<<"null";}
void result(const fs::path& path,const Budget& budget,const std::string& fault,const std::string& gate,int completed_updates,int adam_steps,int selection,int withheld,bool qualified,const Retention* retention=nullptr,bool full_retention=false,int ablation=-1,bool closed=false,bool inherited_only=false){
 std::ofstream out(path);out<<std::setprecision(17)<<"{\"schema\":1,\"study\":\"continuous_pond_v1\",\"controls\":"<<budget.controls<<",\"substeps\":"<<budget.substeps<<",\"active_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<",\"prior_active_seconds\":"<<prior_seconds
 <<",\"PPO_updates\":"<<completed_updates<<",\"actor_steps\":"<<adam_steps<<",\"critic_steps\":0,\"critic_forward_calls\":0,\"fault\":"<<std::quoted(fault)<<",\"gate\":"<<std::quoted(gate)<<",\"selection_passes\":"<<selection<<",\"withheld_continuous_passes\":";nullable(out,withheld>=0,withheld);
 out<<",\"withheld_seconds\":1800,\"short_swim_passes\":";nullable(out,retention,retention?retention->short_passes:0);out<<",\"long60_retained_windows\":";nullable(out,retention,retention?retention->long60_retained:0);
 out<<",\"recovery_passes\":";nullable(out,retention&&full_retention,retention?retention->recovery_passes:0);out<<",\"long300_retained_windows\":";nullable(out,retention&&full_retention,retention?retention->long300_retained:0);
 out<<",\"original_hold_passes\":";nullable(out,retention,retention?retention->original_holds:0);out<<",\"withheld_hold_passes\":";nullable(out,retention&&full_retention,retention?retention->withheld_holds:0);out<<",\"hold300_passes\":";nullable(out,retention&&full_retention,retention?retention->endurance_holds:0);
 out<<",\"ablation_passes\":";nullable(out,ablation>=0,ablation);out<<",\"qualified\":"<<(qualified?"true":"false")<<",\"final_policy_update\":"<<completed_updates<<",\"worlds\":4096,\"inherited_only\":"<<(inherited_only?"true":"false")<<",\"phase_closed\":true,\"closed\":"<<((closed||!fault.empty())?"true":"false")<<",\"runtime_adopted\":false,\"retries\":0}\n";
 out.close();require(bool(out),"Cannot save continuous native result");std::ofstream status(path.parent_path()/"status.json");status<<"{\"phase\":"<<std::quoted(path.stem().string())<<",\"controls\":"<<budget.controls<<",\"substeps\":"<<budget.substeps<<",\"PPO_updates\":"<<completed_updates<<",\"qualified\":"<<(qualified?"true":"false")<<",\"closed\":"<<((closed||!fault.empty())?"true":"false")<<"}\n";}
void validate_calibration_audit(const fs::path& root){const auto bytes=cp::read((root/"calibration-verified.bin").string());cp::Reader in{bytes};require(in.value<uint32_t>()==0x31565054&&in.value<uint32_t>()==1&&in.value<uint64_t>()==study_protocol,"Independent continuous readiness release absent/incompatible");for(const auto& file:{"independent-calibration-verification.json","fixed-scale.bin","calibration-ready.bin"})require(in.value<uint64_t>()==file_hash(root/file),"Independent calibration release evidence changed");in.finish();}
int baseline_passes(const fs::path& root){std::ifstream input(root/"baseline-free/summary.csv");std::string line;std::getline(input,line);int count=0,passes=0;while(std::getline(input,line)){std::stringstream row(line);std::string value;std::vector<double> fields;while(std::getline(row,value,','))fields.push_back(std::stod(value));require(fields.size()==23&&int(fields[0])==count,"Baseline continuous summary schema differs");passes+=int(fields[2]);++count;}require(count==16,"Baseline continuous probes incomplete");return passes;}
int prepare(const fs::path& root,const fs::path& approved){require(!fs::exists(root),"Continuous run identity exists");fs::create_directories(root);configure(root,false);fs::copy_file(approved,root/"approved-protocol.md");fs::copy_file(parent_checkpoint_path,root/"parent-checkpoint.bin");fs::copy_file(parent_protocol_path,root/"parent-protocol.bin");fs::copy_file(fs::absolute("assets/ppo-four-axis-checkpoints/learned-amplitude-2026-11-22-qualified/qualified-policy-update-7.bin"),root/"holding-parent.bin");
 const auto bytes=protocol_bytes(root);cp::write_atomic((root/"protocol.bin").string(),bytes);study_protocol=cp::checksum(bytes);cp::write_atomic((root/"migrated-initial.bin").string(),rl::write_appended_hold(migrated41,study_protocol));write_layout(root);check_identity(std::vector<float>(16,0));disk_guard(root,4096,320,true);
 require(fs::create_directory(root/"initial-policy"),"Initial continuous policy identity exists");{Data data;save_policy(data,root/"initial-policy",0);}numerical(root/"prepare-math");zero_initialization(root/"prepare-zero");
 std::ofstream out(root/"prepared.json");out<<"{\"schema\":1,\"prepared\":true,\"worlds\":4096,\"maximum_controls\":320,\"PPO_updates_max\":64,\"Adam_steps_max\":1024,\"controls_cap\":86811440,\"substeps_cap\":868114400,\"projected_study_bytes\":"<<projected_study_bytes<<",\"working_reserve_bytes\":"<<working_disk_reserve<<",\"physical_controls\":0,\"optimizer_steps\":0}\n";out.close();require(bool(out),"Cannot save continuous preparation");return 0;}
int preflight(const fs::path& root){configure(root,true);validate_protocol(root);write_layout(root);numerical(root/"verified-math");zero_initialization(root/"verified-zero");std::ofstream out(root/"preflight-passed.txt");out<<"Continuous turning policy, actual CUDA decode/PPO/gather/reduction/FD, unchanged requested/hold branches, geometry/observation/body/RNG and fresh Adam checked with zero physical controls/optimizer steps.\n";return 0;}
int baseline(const fs::path& root){configure(root,true);validate_protocol(root);require(fs::exists(root/"preflight-passed.txt"),"Continuous preflight missing");const auto prior=passed_gate(root);disk_guard(root,4096,320,true);auto budget=begin_phase(root,"baseline",prior);std::string fault;Retention old;Evaluation free;bool retained=false;
 try{const std::vector<float> zero(16,0);old=assess_retention(zero,root/"baseline-retention",budget,false);retained=true;require(retention_passed(old,false),"Canonical learned baseline changed");free=assess_continuous(zero,root/"baseline-free",budget);}
 catch(const std::exception& error){fault=error.what();std::cerr<<fault<<std::endl;}result(root/"baseline-result.json",budget,fault,fault.empty()?"baseline_measured":"baseline_failed",0,0,free.passes,-1,false,retained?&old:nullptr);
 if(fault.empty())write_ready(root,1,budget);return fault.empty()?0:3;}
int calibrate(const fs::path& root){configure(root,true);validate_protocol(root);passed_gate(root);const auto prior=read_ready(root,1);require(baseline_passes(root)<16,"Inherited free swimmer already passes selection; validate it without PPO");auto budget=begin_phase(root,"calibration",prior);std::string fault;
 try{disk_guard(root,4096,320);Data data;const auto batch=collect(data,budget,root/"calibration-started/calibration-batch-0",0,0);const auto scale=fit_scale(batch.returns);cp::Writer out;out.value(scale_magic);out.value(uint32_t{1});out.value(study_protocol);out.value(scale.value);for(double x:scale.means)out.value(x);cp::write_atomic((root/"fixed-scale.bin").string(),out.data);
  int boundary=0,deadline=0;for(const auto& c:batch.counts){boundary+=c.reason==1;deadline+=c.reason==2;}std::ofstream report(root/"calibration-report.json");report<<std::setprecision(17)<<"{\"schema\":1,\"passed\":true,\"episodes\":4096,\"cases\":16,\"episodes_per_case\":256,\"fixed_global_residual_RMS\":"<<scale.value<<",\"ordinary_boundary_terminals\":"<<boundary<<",\"deadline_terminals\":"<<deadline<<",\"max_old_logp_error\":"<<batch.probability_error<<",\"controls\":"<<budget.controls<<",\"substeps\":"<<budget.substeps<<",\"actor_steps\":0}\n";report.close();require(bool(report),"Cannot save continuous calibration report");}
 catch(const std::exception& error){fault=error.what();std::cerr<<fault<<std::endl;}result(root/"calibration-result.json",budget,fault,fault.empty()?"await_independent_readiness":"calibration_failed",0,0,baseline_passes(root),-1,false);if(fault.empty())write_ready(root,2,budget);return fault.empty()?0:3;}
Scale read_scale(const fs::path& root){const auto bytes=cp::read((root/"fixed-scale.bin").string());cp::Reader in{bytes};require(in.value<uint32_t>()==scale_magic&&in.value<uint32_t>()==1&&in.value<uint64_t>()==study_protocol,"Continuous calibration scale contract differs");Scale scale;scale.value=in.value<double>();for(double& x:scale.means)x=in.value<double>();in.finish();require(std::isfinite(scale.value)&&scale.value>0,"Invalid continuous fixed scale");return scale;}
int train(const fs::path& root){configure(root,true);validate_protocol(root);passed_gate(root);const bool inherited=baseline_passes(root)==16;const auto prior=read_ready(root,inherited?1:2);if(!inherited)validate_calibration_audit(root);Scale scale;if(!inherited)scale=read_scale(root);auto budget=begin_phase(root,"training",prior);
 std::unique_ptr<Data> active;std::string fault,gate;int completed_updates=0,adam_steps=0,selection=baseline_passes(root),withheld=-1,ablation=-1;bool qualified=false,retained=false;Retention old;std::ofstream progress(root/"updates.jsonl");progress<<std::setprecision(17);
 try{disk_guard(root,4096,320);active=std::make_unique<Data>(4096,320,!inherited);auto& data=*active;save_policy(data,root/"training-started",0);
  if(!inherited)for(int update=1;update<=64;++update){const auto batch=collect(data,budget,root/"training-started"/("training-batch-"+std::to_string(update)),1,update);const double kl=optimize(data,batch,scale,update,root/"training-started");completed_updates=update;adam_steps=data.optimizer->steps();save_policy(data,root/"training-started",update);
   const auto means=download(data.means.get(),16,data.stream->value);check_identity(means);const auto measured=assess_continuous(means,root/"training-started"/("assessment-"+std::to_string(update)),budget);selection=measured.passes;
   progress<<"{\"update\":"<<update<<",\"worlds\":4096,\"minibatch_episodes\":1024,\"selection_passes\":"<<selection<<",\"selection_survivors\":"<<measured.survivors<<",\"minimum_mean_speed_mm_s\":"<<measured.minimum_mean_speed*1000<<",\"minimum_window_speed_mm_s\":"<<measured.minimum_window_speed*1000<<",\"KL\":"<<kl<<",\"actor_steps\":"<<adam_steps<<",\"controls\":"<<budget.controls<<",\"substeps\":"<<budget.substeps<<",\"means\":[";
   for(int j=0;j<16;++j){if(j)progress<<',';progress<<means[j];}progress<<"]}\n";progress.flush();require(bool(progress),"Cannot save continuous update ledger");std::cout<<"TURN_UPDATE "<<update<<" selection="<<selection<<" controls="<<budget.controls<<" substeps="<<budget.substeps<<std::endl;if(selection==16)break;}
  const auto means=download(data.means.get(),16,data.stream->value);if(selection==16)withheld=assess_continuous(means,root/"withheld-validation",budget,true).passes;
  old=assess_retention(means,root/"final-retention",budget,true);retained=true;qualified=selection==16&&withheld==8&&retention_passed(old,true);
  if(qualified)ablation=assess_continuous(means,root/"constant-only-ablation",budget,false,true).passes;gate=qualified?"qualified_finite_continuous_swimming_no_adoption":"continuous_or_retention_not_qualified";
 }catch(const std::exception& error){fault=error.what();gate="physical_data_or_numerical_stop";qualified=false;if(active&&active->optimizer)adam_steps=active->optimizer->steps();std::cerr<<fault<<std::endl;}
 result(root/"training-result.json",budget,fault,gate,completed_updates,adam_steps,selection,withheld,qualified,retained?&old:nullptr,true,ablation,true,inherited);return fault.empty()?0:3;}
}
int main(int argc,char** argv){try{require(argc>=3,"Use --prepare/--preflight/--baseline/--calibrate/--train RUN [protocol]");const std::string mode=argv[1];const auto root=fs::absolute(argv[2]);if(mode=="--prepare"){require(argc==4,"Continuous prepare requires approved protocol");return turning_study::prepare(root,fs::absolute(argv[3]));}require(argc==3,"Unexpected continuous argument");
 if(mode=="--preflight")return turning_study::preflight(root);if(mode=="--baseline")return turning_study::baseline(root);if(mode=="--calibrate")return turning_study::calibrate(root);if(mode=="--train")return turning_study::train(root);throw std::runtime_error("Unknown continuous mode");}catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 2;}}
