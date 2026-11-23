// Opt-in whole-episode amplitude PPO. Physics and the 336 backbone are immutable.
#include "motor/joint_motor.h"
namespace amplitude {
void physical_failure(const cudarl::joint_motor::World&,const cudarl::physics::BodyState&,
                      const cudarl::physics::BodyState&,int,int);
void physical_step(const cudarl::joint_motor::World&,const cudarl::physics::BodyState&,int,int);
}
#define CUDARL_PHYSICAL_FAILURE_HOOK amplitude::physical_failure
#define CUDARL_PHYSICAL_STEP_HOOK amplitude::physical_step
#define CUDARL_REST_HOLD_MEASURED_PHASE_B 1
#define CUDARL_REST_HOLD_MEASURED_BASELINE 1
#define CUDARL_REST_HOLD_RESET_STUDY 1
#include "rest_hold_study.cu"
#include "hold_learned_amplitude_math.h"
#include "core/rng.h"
#include <set>

namespace amplitude {
namespace math=cudarl::learned_amplitude;
constexpr int worlds_count=256,episode_controls=1280,trace_rows=worlds_count*episode_controls;
constexpr int maximum_updates=32,maximum_adam_steps=512,minibatch_episodes=64,maximum_epochs=4;
constexpr int calibration_controls=346080,ordinary_controls_cap=11174880,controls_cap=11194560;
constexpr uint32_t policy_magic=0x314d4148,protocol_magic=0x31504148,batch_magic=0x31424148,ready_magic=0x31524148;
constexpr uint32_t calibration_seed=314159265u,training_seed=0x6192a781u;
constexpr double seconds_cap=900;
static_assert(observations==41&&actions==4,"Requires appended41 frozen backbone");
fs::path failure_directory;
bool failure_saved=false;
std::ofstream pose_output;
std::string pose_category;
int pose_case=-1,pose_expected=0;
float pose_gain=1;

void physical_failure(const m::World& world,const p::BodyState& before,const p::BodyState& candidate,int controls,int substeps){
    if(failure_saved)return;
    require(!failure_directory.empty()&&fs::is_directory(failure_directory),"Missing native failure evidence directory");
    const auto path=failure_directory/"first-cpu-physical-failure.bin";require(!fs::exists(path),"Failure evidence already exists");failure_saved=true;
    cp::Writer w;w.value(uint32_t{0x32434648});w.value(uint32_t{1});w.value(controls);w.value(substeps);
    w.value(uint32_t{sizeof(m::World)});w.value(world);w.value(before);w.value(candidate);cp::write_atomic(path.string(),w.data);
}
void pose_header(std::ostream& out){out<<"category,case,control,time_s,position_x_model,position_y_model,position_z_model,quat_w,quat_x,quat_y,quat_z,angle0,angle1,angle2,angle3,gain0,gain1,gain2,gain3\n";}
void write_pose(std::ostream& out,const char* category,int id,int control,const p::BodyState& b,float gain){
    out<<category<<','<<id<<','<<control<<','<<control*.05<<','<<b.position.x<<','<<b.position.y<<','<<b.position.z
       <<','<<b.orientation.w<<','<<b.orientation.x<<','<<b.orientation.y<<','<<b.orientation.z;
    for(float a:b.angle)out<<','<<a;for(int j=0;j<4;++j)out<<','<<gain;out<<'\n';
}
void physical_step(const m::World& world,const p::BodyState& before,int,int){
    if(!pose_output.is_open())return;
    const int k=int(world.physical.control_steps);if(k==1)++pose_case;
    require(pose_case>=0&&pose_case<pose_expected,"Unexpected native pose case");
    if(k==1)write_pose(pose_output,pose_category.c_str(),pose_case,0,before,pose_gain);
    write_pose(pose_output,pose_category.c_str(),pose_case,k,world.physical.body,pose_gain);
    require(bool(pose_output),"Native pose log write failed");
}
void pose_group(const char* category,int cases,float gain){pose_category=category;pose_case=-1;pose_expected=cases;pose_gain=gain;}
void close_pose_group(){require(pose_case+1==pose_expected,"Native pose case capture incomplete");}
std::array<float,2> initial_parameters(){return {logf(99.f),logf((6.f-.05f)/(8.f-6.f))};}
CUDARL_HD inline m::ForwardConfig hold_config(int controls=episode_controls){
    auto c=m::config_for_speed(.008f);c.observation_schema=4;c.maximum_controls=controls;
    c.target_speed_m_s=0;c.stationary_request=true;c.position_hold=true;return c;
}
CUDARL_HD inline int case_pose(int id){return id%2?3:0;}
CUDARL_HD inline float case_offset(int id){return id/2==0?0.f:id/2==1?-.0005f:.0005f;}

struct EpisodeCounts {
    unsigned long long controls=0,substeps=0;
    double reward_sum=0;
    float peak[5]{},rms_speed_square=0,forward_at_window=0,final_excursion=0;
    int failure_code=0;
    p::BodyState first_before{},rejected_candidate{};
};
struct Data {
    std::unique_ptr<Stream> stream;
    std::unique_ptr<nn::GpuMlp> backbone;
    std::unique_ptr<nn::GpuAdam> optimizer;
    device_ptr<float> parameters{2},gradient{2};
    device_ptr<m::World> worlds{worlds_count};device_ptr<m::ForwardTask> tasks{worlds_count};
    device_ptr<EpisodeCounts> counts{worlds_count};device_ptr<int> fault{1};
    device_ptr<float> latent{worlds_count},gain{worlds_count},old_logp{worlds_count},initial_metrics{worlds_count*5};
    device_ptr<float> observations{trace_rows*41},base{trace_rows*4},joint_latent{trace_rows*4};
    device_ptr<float> reward{trace_rows},metrics{trace_rows*6};
    device_ptr<float> advantages{worlds_count},mb_latent{64},mb_oldlogp{64},mb_advantage{64};
    device_ptr<float> row_gradient{64*2},row_stats{64*5},stats{5};
    int completed=0;
    unsigned long long collected_controls=0,collected_substeps=0;
    Data(bool training):stream(std::make_unique<Stream>()),backbone(std::make_unique<nn::GpuMlp>(spec(4),worlds_count,seed)){
        backbone->set_params(resume_state->actor.parameters);const auto p0=initial_parameters();parameters.copy_from_host(p0.data(),2);
        CUDA_CHECK_THROW(cudaMemsetAsync(counts.get(),0,counts.size_bytes(),stream->value));
        CUDA_CHECK_THROW(cudaMemsetAsync(fault.get(),0,sizeof(int),stream->value));
        if(training){auto cfg=initial_optimizer_config(true);cfg.lr=.05f;optimizer=std::make_unique<nn::GpuAdam>(2,cfg);require(optimizer->steps()==0,"Amplitude Adam must start fresh");}
    }
};
CUDARL_HD inline void physical_metrics(const m::ForwardTask& task,const m::World& world,float* output,bool forward=false){
    const auto z=m::measure(task,world);output[0]=p::norm(z.velocity_m_s);output[1]=p::norm(z.hold_error_m);
    output[2]=z.depth_m;output[3]=z.lateral_m;output[4]=z.tilt_deg;if(forward)output[5]=z.forward_m;
}
__global__ void initialize_episodes(m::World* worlds,m::ForwardTask* tasks,EpisodeCounts* counts,
                                   const float* parameters,float* latent,float* gain,float* logp,
                                   float* metrics,int* fault,uint32_t action_seed,uint32_t counter){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=worlds_count||*fault)return;
    const float sigma=math::standard_deviation(parameters[1]);
    const float u=parameters[0]+sigma*cudarl::rng::normal(action_seed,counter,uint32_t(i));
    latent[i]=u;gain[i]=math::sigmoid(u);logp[i]=math::log_probability(parameters,u);
    if(!p::finite_number(u)||!p::finite_number(gain[i])||!p::finite_number(logp[i])||gain[i]<0||gain[i]>1){atomicCAS(fault,0,1);return;}
    const int id=i%6;worlds[i]=m::initial_world(case_pose(id));worlds[i].arena=m::long_swim_arena();tasks[i]={};
    const auto config=hold_config();
    if(!m::begin_forward(worlds[i],config,tasks[i])||!m::offset_episode_start(worlds[i],tasks[i],config,case_offset(id))||
       !m::update_velocity_request(worlds[i],config,tasks[i])){atomicCAS(fault,0,2);return;}
    physical_metrics(tasks[i],worlds[i],metrics+i*5);
    for(int j=0;j<5;++j)counts[i].peak[j]=metrics[i*5+j];
}
__global__ void observe_episodes(const m::World* worlds,const m::ForwardTask* tasks,float* obs,int* fault){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=worlds_count)return;
    if(*fault){for(int j=0;j<41;++j)obs[i*41+j]=0;return;}
    if(!m::observe_appended(tasks[i],worlds[i],hold_config(),obs+i*41))atomicCAS(fault,0,3);
}
__global__ void advance_episodes(m::World* worlds,m::ForwardTask* tasks,EpisodeCounts* counts,
                                const float* base,const float* gain,float* recorded_base,float* joint_latent,
                                float* reward,float* metrics,int* fault,int control){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=worlds_count||*fault)return;
    float latent[4];for(int j=0;j<4;++j){recorded_base[i*4+j]=base[i*4+j];latent[j]=gain[i]*base[i*4+j];joint_latent[i*4+j]=latent[j];}
    m::Action action;if(!m::action_from_latent(latent,action)){atomicCAS(fault,0,4);return;}
    const auto before=worlds[i].attempted_substeps;++counts[i].controls;
    p::BodyState pre=worlds[i].physical.body,candidate=pre;
    const bool advanced=cudarl::joint_motor::step_state(worlds[i],action.command,nullptr,&pre,&candidate);
    const auto attempted=worlds[i].attempted_substeps-before;counts[i].substeps+=attempted;
    if(!advanced||attempted!=10){counts[i].first_before=pre;counts[i].rejected_candidate=candidate;
        counts[i].failure_code=worlds[i].failure_code;atomicCAS(fault,0,5);return;}
    const auto transition=m::record_hold_control(tasks[i],worlds[i],hold_config());
    if(!transition.accepted||transition.done!=(control==episode_controls)||transition.truncated||tasks[i].controls!=control||worlds[i].physical.control_steps!=uint64_t(control)){
        counts[i].first_before=pre;counts[i].rejected_candidate=worlds[i].physical.body;atomicCAS(fault,0,6);return;}
    reward[i]=transition.reward*(1.f/1024);counts[i].reward_sum+=double(reward[i]);
    physical_metrics(tasks[i],worlds[i],metrics+i*6,true);
    for(int j=0;j<5;++j){if(!p::finite_number(metrics[i*6+j])){atomicCAS(fault,0,7);return;}
        counts[i].peak[j]=fmaxf(counts[i].peak[j],metrics[i*6+j]);}
    counts[i].rms_speed_square+=metrics[i*6]*metrics[i*6];
    if(control==episode_controls-160)counts[i].forward_at_window=transition.metrics.forward_m;
    if(control>=episode_controls-160)counts[i].final_excursion=fmaxf(counts[i].final_excursion,
        fabsf(transition.metrics.forward_m-counts[i].forward_at_window));
}
__global__ void episode_loss(const float* parameters,const float* latent,const float* oldp,const float* advantage,
                             int n,float* gradient,float* stats){
    const int i=threadIdx.x;if(i>=n)return;
    const auto terms=math::loss(parameters,latent[i],oldp[i],advantage[i],1.f/n,gradient+i*2);
    stats[i*5]=terms.pg_loss;stats[i*5+1]=terms.v_loss;stats[i*5+2]=terms.entropy;
    stats[i*5+3]=terms.approx_kl;stats[i*5+4]=terms.clipped;
}
struct Collected {
    std::vector<float> parameters,latent,gain,logp,returns;
    std::vector<EpisodeCounts> counts;
    int ticks=0,quiet=0,quiet_pose[2]{};
    double maximum_probability_error=0,maximum_return_error=0;
};
bool passed(const EpisodeCounts& x){return x.controls==episode_controls&&x.substeps==episode_controls*10&&
    x.peak[0]<=.0005f&&x.peak[1]<=.002f&&x.peak[2]<=.001f&&x.peak[3]<=.002f&&x.peak[4]<=10.f&&x.final_excursion<=.002f;}
void account(Data& d,Budget& budget){
    const auto count=download(d.counts.get(),worlds_count,d.stream->value);unsigned long long controls=0,substeps=0;
    for(const auto& x:count){controls+=x.controls;substeps+=x.substeps;}
    require(controls>=d.collected_controls&&substeps>=d.collected_substeps,"Persistent physical accounting moved backwards");
    budget.controls+=int(controls-d.collected_controls);budget.substeps+=int(substeps-d.collected_substeps);
    d.collected_controls=controls;d.collected_substeps=substeps;
    require(budget.controls<=budget.maximum_controls&&budget.substeps<=budget.maximum_substeps,"Whole-episode physical cap reached");
}
uint64_t file_hash(const fs::path& path){std::ifstream in(path,std::ios::binary);require(bool(in),"Bound file missing");
    std::vector<uint8_t> data;for(char c;in.get(c);)data.push_back(uint8_t(c));require(in.eof(),"Bound file read failed");return cp::checksum(data);}
void configure(const fs::path& root,bool copied){
    configure_rest(root,copied);control_cap=controls_cap;substep_cap=controls_cap*10;active_wall_cap=seconds_cap;
    require(resume_state->actor.parameters.size()==2532&&original40.actor.update==336,"Expected frozen336 backbone");
}
std::vector<uint8_t> protocol_bytes(const fs::path& root){
    cp::Writer w;w.value(protocol_magic);w.value(uint32_t{1});w.array(rest_protocol());w.value(file_hash(root/"approved-protocol.md"));
    for(int v:{worlds_count,episode_controls,maximum_updates,maximum_adam_steps,minibatch_episodes,maximum_epochs,
               calibration_controls,ordinary_controls_cap,controls_cap,controls_cap*10,900})w.value(v);
    w.value(calibration_seed);w.value(training_seed);
    for(float v:{.99f,6.f,.05f,7.95f,.05f,.2f,.01f,1.f/1024,64.f})w.value(v);
    const std::string method="one Gaussian latent per whole64s episode; sigmoid gain; bounded sigma; two raw trainable parameters; frozen336; LOO six-case baseline; independent fixed global calibration residual RMS; no amplitude/bin conditioning; no per-control noise or score; no adoption";
    w.array(std::vector<uint8_t>(method.begin(),method.end()));return w.data;
}
void validate_protocol(const fs::path& root){require(cp::read((root/"protocol.bin").string())==protocol_bytes(root),"Whole-episode protocol changed");}
void save_batch(const fs::path& root,const char* phase,int index,const Collected& c,Data& d){
    const auto stream=d.stream->value;const int rows_saved=c.ticks*worlds_count;
    cp::Writer w;w.value(batch_magic);w.value(uint32_t{1});w.value(uint32_t{worlds_count});w.value(uint32_t{episode_controls});w.value(c.ticks);w.value(uint32_t{41});w.value(uint32_t{4});w.value(uint32_t{6});
    w.array(c.parameters);w.array(c.latent);w.array(c.gain);w.array(c.logp);w.array(c.returns);
    w.value(uint32_t{sizeof(EpisodeCounts)});w.array(c.counts);
    w.array(download(d.initial_metrics.get(),worlds_count*5,stream));
    w.array(download(d.observations.get(),rows_saved*41,stream));w.array(download(d.base.get(),rows_saved*4,stream));
    w.array(download(d.joint_latent.get(),rows_saved*4,stream));w.array(download(d.reward.get(),rows_saved,stream));w.array(download(d.metrics.get(),rows_saved*6,stream));
    w.value(uint32_t{sizeof(m::World)});w.array(download(d.worlds.get(),worlds_count,stream));
    w.value(uint32_t{sizeof(m::ForwardTask)});w.array(download(d.tasks.get(),worlds_count,stream));
    const auto name=std::string(phase)+"-batch-"+std::to_string(index)+".bin";require(!fs::exists(root/name),"Episode batch identity exists");
    cp::write_atomic((root/name).string(),w.data);
    std::ofstream csv(root/(std::string(phase)+"-episodes-"+std::to_string(index)+".csv"));csv<<std::setprecision(12)
      <<"episode,case,pose,offset_mm,latent_u,gain,old_logp,reward_sum,return,controls,substeps,passed,max_speed_m_s,max_anchor_m,max_depth_m,max_lane_m,max_tilt_deg,final8_excursion_m\n";
    for(int i=0;i<worlds_count;++i){const auto& x=c.counts[i];csv<<i<<','<<i%6<<','<<case_pose(i%6)<<','<<case_offset(i%6)*1000<<','
      <<c.latent[i]<<','<<c.gain[i]<<','<<c.logp[i]<<','<<x.reward_sum<<','<<c.returns[i]<<','<<x.controls<<','<<x.substeps<<','<<passed(x);
      for(float peak:x.peak)csv<<','<<peak;csv<<','<<x.final_excursion<<'\n';}
    require(bool(csv),"Episode summary write failed");
}
Collected collect(Data& d,Budget& budget,const fs::path& root,const char* phase,int index,bool calibration){
    require(budget.controls+trace_rows<=budget.maximum_controls&&budget.substeps+trace_rows*10<=budget.maximum_substeps,"Next whole batch exceeds budget");
    const auto stream=d.stream->value;CUDA_CHECK_THROW(cudaMemsetAsync(d.fault.get(),0,sizeof(int),stream));
    // Counters are persistent across full episodes/batches; physical episode state resets explicitly.
    const auto before_count=download(d.counts.get(),worlds_count,stream);
    std::vector<EpisodeCounts> reset(worlds_count);
    for(int i=0;i<worlds_count;++i){reset[i].controls=before_count[i].controls;reset[i].substeps=before_count[i].substeps;}
    d.counts.copy_from_host(reset.data(),worlds_count);
    CUDA_CHECK_THROW(cudaMemsetAsync(d.observations.get(),0,d.observations.size_bytes(),stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(d.base.get(),0,d.base.size_bytes(),stream));CUDA_CHECK_THROW(cudaMemsetAsync(d.joint_latent.get(),0,d.joint_latent.size_bytes(),stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(d.reward.get(),0,d.reward.size_bytes(),stream));CUDA_CHECK_THROW(cudaMemsetAsync(d.metrics.get(),0,d.metrics.size_bytes(),stream));
    initialize_episodes<<<2,128,0,stream>>>(d.worlds.get(),d.tasks.get(),d.counts.get(),d.parameters.get(),d.latent.get(),d.gain.get(),d.old_logp.get(),d.initial_metrics.get(),d.fault.get(),calibration?calibration_seed:training_seed,uint32_t(index));
    Collected c;c.parameters=download(d.parameters.get(),2,stream);c.latent=download(d.latent.get(),worlds_count,stream);
    c.gain=download(d.gain.get(),worlds_count,stream);c.logp=download(d.old_logp.get(),worlds_count,stream);
    c.returns.assign(worlds_count,0);int fault=download(d.fault.get(),1,stream)[0];
    try{
        require(fault==0,"Episode reset/probability fault");
        for(int k=0;k<episode_controls;++k){wall_guard();const int offset=k*worlds_count;
            observe_episodes<<<2,128,0,stream>>>(d.worlds.get(),d.tasks.get(),d.observations.get()+offset*41,d.fault.get());
            d.backbone->forward(d.observations.get()+offset*41,worlds_count,stream);
            advance_episodes<<<2,128,0,stream>>>(d.worlds.get(),d.tasks.get(),d.counts.get(),d.backbone->output(),d.gain.get(),
                d.base.get()+offset*4,d.joint_latent.get()+offset*4,d.reward.get()+offset,d.metrics.get()+offset*6,d.fault.get(),k+1);
            CUDA_CHECK_THROW(cudaGetLastError());c.ticks=k+1;
            if((k+1)%32==0){fault=download(d.fault.get(),1,stream)[0];
                const auto live_counts=download(d.counts.get(),worlds_count,stream);unsigned long long controls=0,substeps=0;
                for(const auto& value:live_counts){controls+=value.controls;substeps+=value.substeps;}
                std::ofstream live(root.parent_path()/"status.json");live<<std::setprecision(17)<<"{\"phase\":"<<std::quoted(phase)<<",\"batch\":"<<index
                  <<",\"completed_updates\":"<<d.completed<<",\"tick\":"<<k+1<<",\"ticks_per_episode\":1280,\"worlds\":256,\"unix_seconds\":"<<unix_now()
                  <<",\"controls\":"<<budget.controls+controls-d.collected_controls<<",\"substeps\":"<<budget.substeps+substeps-d.collected_substeps
                  <<",\"optimizer_steps\":"<<(d.optimizer?d.optimizer->steps():0)<<",\"fault\":"<<fault<<"}\n";
                if(fault)break;}
        }
    }catch(...){account(d,budget);c.counts=download(d.counts.get(),worlds_count,stream);
        for(int i=0;i<worlds_count;++i){c.counts[i].controls-=before_count[i].controls;c.counts[i].substeps-=before_count[i].substeps;c.returns[i]=float(c.counts[i].reward_sum/64.);}
        save_batch(root,phase,index,c,d);throw;}
    account(d,budget);fault=download(d.fault.get(),1,stream)[0];c.counts=download(d.counts.get(),worlds_count,stream);
    // Counts above are persistent. Store batch-local attempts while retaining global accounting independently.
    for(int i=0;i<worlds_count;++i){c.counts[i].controls-=before_count[i].controls;c.counts[i].substeps-=before_count[i].substeps;
        c.returns[i]=float(c.counts[i].reward_sum/64.);}
    save_batch(root,phase,index,c,d);
    require(fault==0&&c.ticks==episode_controls,"Whole-episode native physics/data stop before optimizer");
    const auto reward=download(d.reward.get(),trace_rows,stream),metrics=download(d.metrics.get(),trace_rows*6,stream),initial=download(d.initial_metrics.get(),worlds_count*5,stream);
    const auto obs=download(d.observations.get(),trace_rows*41,stream),base=download(d.base.get(),trace_rows*4,stream),joint=download(d.joint_latent.get(),trace_rows*4,stream);
    const auto final_tasks=download(d.tasks.get(),worlds_count,stream);const auto final_worlds=download(d.worlds.get(),worlds_count,stream);
    double params[2]={c.parameters[0],c.parameters[1]};
    for(int i=0;i<worlds_count;++i){require(c.counts[i].controls==episode_controls&&c.counts[i].substeps==episode_controls*10,"Episode attempt count changed");
        require(final_tasks[i].controls==episode_controls&&final_worlds[i].physical.control_steps==episode_controls&&final_worlds[i].physical.substeps==episode_controls*10,
                "Whole episode reset or truncated");
        require(final_tasks[i].hold_anchor_set&&p::norm(final_tasks[i].request.terminal_velocity)==0&&p::norm(final_worlds[i].physical.current)==0,"Wrong holding goal/current/anchor");
        double sum=0;for(int k=0;k<episode_controls;++k){const int row=k*worlds_count+i;sum+=reward[row];
            require(obs[row*41+34]==p::clamp(1.f-float(k)/40,0,1),"Episode clock changed or reset");
            for(int j=3;j<6;++j)require(obs[row*41+j]==0,"Nonholding world in whole-episode batch");
            for(int j=0;j<4;++j)require(joint[row*4+j]==c.gain[i]*base[row*4+j],"Applied command differs from sampled episode policy");
            for(int j=0;j<5;++j)require(std::isfinite(metrics[row*6+j])&&metrics[row*6+j]>=0,"Invalid physical metric");}
        const double error=std::abs(sum-c.counts[i].reward_sum);c.maximum_return_error=std::max(c.maximum_return_error,error);
        require(error<1e-7+1e-7*std::abs(sum)&&c.returns[i]==float(sum/64.),"Whole-episode return mismatch");
        const double lp=math::log_probability(params,double(c.latent[i]));c.maximum_probability_error=std::max(c.maximum_probability_error,std::abs(lp-c.logp[i]));
        require(std::isfinite(c.returns[i])&&std::isfinite(c.logp[i])&&c.gain[i]>=0&&c.gain[i]<=1,"Invalid episode policy sample");
        for(int j=0;j<5;++j)require(std::isfinite(initial[i*5+j]),"Invalid initial sample");
        if(passed(c.counts[i])){++c.quiet;++c.quiet_pose[(i%6)%2];}}
    require(c.maximum_probability_error<1e-4,"Episode scalar latent rescoring mismatch");
    require(d.backbone->get_params()==resume_state->actor.parameters,"Frozen backbone changed");
    return c;
}
struct Case { int pose=0;float offset=0;int controls=episode_controls; };
std::vector<Case> original_cases(){std::vector<Case> out;for(int i=0;i<6;++i)out.push_back({case_pose(i),case_offset(i),episode_controls});return out;}
std::vector<Case> withheld_cases(){std::vector<Case> out;for(float depth:{0.f,-.00025f,.00025f})for(int pose:{1,2})out.push_back({pose,depth,episode_controls});return out;}
std::vector<Case> endurance_cases(){return {{0,0,6000},{3,0,6000}};}
class HoldScore {
    int total_,next_=0;bool valid_=true;float peak_[5]{},forward_window_=0,excursion_=0;double square_speed_=0;
public:
    explicit HoldScore(int total):total_(total){valid_=total>=160;}
    bool record(int control,const m::Metrics& z){
        if(!valid_||control!=next_||control>total_){valid_=false;return false;}
        const float values[5]={p::norm(z.velocity_m_s),p::norm(z.hold_error_m),z.depth_m,z.lateral_m,z.tilt_deg};
        for(int j=0;j<5;++j){if(!std::isfinite(values[j])||values[j]<0){valid_=false;return false;}peak_[j]=std::fmax(peak_[j],values[j]);}
        if(!std::isfinite(z.forward_m)){valid_=false;return false;}
        if(control)square_speed_+=double(values[0])*values[0];
        if(control==total_-160)forward_window_=z.forward_m;
        if(control>=total_-160)excursion_=std::fmax(excursion_,std::fabs(z.forward_m-forward_window_));
        ++next_;return true;
    }
    bool complete()const{return valid_&&next_==total_+1;}
    bool passed()const{return complete()&&peak_[0]<=.0005f&&peak_[1]<=.002f&&peak_[2]<=.001f&&peak_[3]<=.002f&&peak_[4]<=10.f&&excursion_<=.002f;}
    const float* peaks()const{return peak_;}
    float excursion()const{return excursion_;}
    double rms_speed()const{return next_>1?std::sqrt(square_speed_/double(next_-1)):0;}
};
struct HoldResult { int passed=0,completed=0; };
HoldResult evaluate_hold(const std::vector<float>& parameters,float gain,const std::vector<Case>& cases,
                         const fs::path& dir,Budget& budget){
    require(fs::create_directory(dir),"Hold assessment identity exists");nn::CpuMlp<float> actor(spec(4));actor.params()=parameters;
    std::ofstream trace(dir/"controls.csv"),summary(dir/"summary.csv"),poses(dir/"poses.csv"),metadata(dir/"cases.csv");
    trace<<std::setprecision(12);summary<<std::setprecision(12);poses<<std::setprecision(12);metadata<<std::setprecision(12);pose_header(poses);
    trace<<"case,pose,offset_mm,control,time_s,speed_mm_s,forward_velocity_mm_s,forward_mm,depth_mm,lane_mm,tilt_deg,roll_deg,pitch_deg,anchor_error_mm";
    for(int j=0;j<4;++j)trace<<",command"<<j<<",angle"<<j<<",rate"<<j<<",last_torque"<<j;trace<<'\n';
    summary<<"case,pose,offset_mm,passed,max_speed_mm_s,RMS_speed_mm_s,max_anchor_error_mm,max_depth_mm,max_lane_mm,max_tilt_deg,final8_forward_excursion_mm,net_forward_mm,final_speed_mm_s,final_depth_mm,positive_work_J\n";
    metadata<<"case,pose,offset_mm,controls,gain,initial_speed_m_s,initial_anchor_m,initial_depth_m,initial_lane_m,initial_tilt_deg\n";
    HoldResult result;
    for(int id=0;id<int(cases.size());++id){const auto spec_case=cases[id];auto world=m::initial_world(spec_case.pose);world.arena=m::long_swim_arena();m::ForwardTask task;const auto cfg=hold_config(spec_case.controls);
        require(m::begin_forward(world,cfg,task)&&m::offset_episode_start(world,task,cfg,spec_case.offset)&&m::update_velocity_request(world,cfg,task),"Hold assessment reset failed");
        const auto anchor=task.hold_anchor_model,start=task.start_com_model,heading=task.request.heading;HoldScore score(spec_case.controls);
        const auto first=m::measure(task,world);require(score.record(0,first),"Hold initial sample invalid");write_pose(poses,"hold",id,0,world.physical.body,gain);
        metadata<<id<<','<<spec_case.pose<<','<<spec_case.offset*1000<<','<<spec_case.controls<<','<<gain<<','<<p::norm(first.velocity_m_s)<<','<<p::norm(first.hold_error_m)<<','<<first.depth_m<<','<<first.lateral_m<<','<<first.tilt_deg<<'\n';
        for(int k=1;k<=spec_case.controls;++k){
            require(m::update_velocity_request(world,cfg,task)&&p::norm(task.hold_anchor_model-anchor)==0&&p::norm(task.start_com_model-start)==0&&p::norm(task.request.heading-heading)==0,"Hold reference changed");
            float observation[41];require(m::observe_appended(task,world,cfg,observation),"Hold observation failed");actor.forward(observation,1);
            float latent[4];for(int j=0;j<4;++j)latent[j]=gain*actor.output()[j];m::Action action;require(m::action_from_latent(latent,action),"Hold action invalid");
            p::BodyState before=world.physical.body,candidate=before;budget.step(world,action.command,nullptr,&before,&candidate);
            const auto interval=m::record_hold_control(task,world,cfg);require(interval.accepted&&score.record(k,interval.metrics),"Hold interval/score invalid");
            const auto& z=interval.metrics;write_pose(poses,"hold",id,k,world.physical.body,gain);
            trace<<id<<','<<spec_case.pose<<','<<spec_case.offset*1000<<','<<k<<','<<k*.05<<','<<p::norm(z.velocity_m_s)*1000<<','<<p::dot(z.velocity_m_s,heading)*1000<<','<<z.forward_m*1000<<','<<z.signed_depth_m*1000<<','<<z.lateral_m*1000<<','<<z.tilt_deg<<','<<z.roll_deg<<','<<z.pitch_deg<<','<<p::norm(z.hold_error_m)*1000;
            for(int j=0;j<4;++j)trace<<','<<action.command[j]<<','<<world.physical.body.angle[j]<<','<<world.physical.body.velocity[6+j]<<','<<world.physical.last_actual_torques[j];trace<<'\n';
        }
        require(score.complete(),"Hold assessment incomplete");const auto last=m::measure(task,world);const auto peak=score.peaks();
        summary<<id<<','<<spec_case.pose<<','<<spec_case.offset*1000<<','<<score.passed()<<','<<peak[0]*1000<<','<<score.rms_speed()*1000<<','<<peak[1]*1000<<','<<peak[2]*1000<<','<<peak[3]*1000<<','<<peak[4]<<','<<score.excursion()*1000<<','<<last.forward_m*1000<<','<<p::norm(last.velocity_m_s)*1000<<','<<last.signed_depth_m*1000<<','<<world.physical.body.work<<'\n';
        ++result.completed;result.passed+=score.passed();summary.flush();trace.flush();poses.flush();metadata.flush();
    }
    require(bool(trace)&&bool(summary)&&bool(poses)&&bool(metadata),"Hold assessment evidence write failed");return result;
}
using Keys=std::set<std::pair<int,int>>;
Keys long_keys(const fs::path& file){std::ifstream in(file);require(bool(in),"Missing long-swim evidence");std::string line;std::getline(in,line);Keys all,passed;int count=0;
    while(std::getline(in,line)){std::stringstream row(line);std::string item;std::vector<double> v;while(std::getline(row,item,','))v.push_back(std::stod(item));
        require(v.size()==10,"Long-window column count changed");const int pose=int(v[0]),end=int(v[1]);
        require((pose==0||pose==3)&&end>=4&&end<=60&&end%4==0&&all.emplace(pose,end).second,"Invalid long-window key");++count;if(v[9]==1)passed.emplace(pose,end);}
    require(count==30,"Long-window sample count changed");return passed;
}
struct Assessment {int short_pass=0,hold_pass=0,long_pass=0;Keys keys;};
Assessment assess(const std::vector<float>& backbone,float gain,const fs::path& dir,Budget& budget){
    require(fs::create_directory(dir),"Full assessment identity exists");const int before=budget.controls;
    pose_output.open(dir/"all-cpu-poses.csv");pose_output<<std::setprecision(12);pose_header(pose_output);
    pose_group("short",4,1);const auto swim=evaluate(backbone,dir,"frozen",budget);close_pose_group();
    pose_group("hold",6,gain);const auto hold=evaluate_hold(backbone,gain,original_cases(),dir/"rest",budget);close_pose_group();
    require(fs::create_directory(dir/"long"),"Long evidence identity exists");pose_group("long",2,1);const auto longrun=sustained(backbone,dir/"long",budget);close_pose_group();
    pose_output.close();require(longrun.safety_fault.empty()&&hold.completed==6&&budget.controls-before==10720,"Full assessment safety/count mismatch");
    const auto keys=long_keys(dir/"long"/"sustained-windows.csv");require(keys.size()==size_t(longrun.windows_passed),"Long-window count mismatch");
    return {swim.passed,hold.passed,longrun.windows_passed,keys};
}
HoldResult hold_only(const std::vector<float>& backbone,float gain,const std::vector<Case>& cases,const fs::path& dir,Budget& budget,const char* category){
    require(fs::create_directory(dir),"Hold-only identity exists");pose_output.open(dir/"all-cpu-poses.csv");pose_output<<std::setprecision(12);pose_header(pose_output);
    pose_group(category,int(cases.size()),gain);const int before=budget.controls;const auto result=evaluate_hold(backbone,gain,cases,dir/"rest",budget);close_pose_group();pose_output.close();
    int expected=0;for(const auto& c:cases)expected+=c.controls;require(budget.controls-before==expected,"Hold-only physical accounting mismatch");return result;
}

struct Baseline {double scale=0;double means[6]{};int counts[6]{};};
Baseline fit_scale(const std::vector<float>& returns){require(returns.size()==worlds_count,"Calibration episode count differs");Baseline b;
    for(int i=0;i<worlds_count;++i){require(std::isfinite(returns[i]),"Nonfinite calibration return");b.means[i%6]+=returns[i];++b.counts[i%6];}
    for(int k=0;k<6;++k){require(b.counts[k]==(k<4?43:42),"Case coverage changed");b.means[k]/=b.counts[k];}
    double squared=0;for(int i=0;i<worlds_count;++i){const double e=returns[i]-b.means[i%6];squared+=e*e;}
    b.scale=std::sqrt(squared/worlds_count);require(std::isfinite(b.scale)&&b.scale>0,"No independent calibration residual variance");return b;
}
std::vector<float> leave_one_out(const std::vector<float>& returns,double scale){
    require(returns.size()==worlds_count&&std::isfinite(scale)&&scale>0,"Invalid episode advantage input");double sums[6]{};int counts[6]{};
    for(int i=0;i<worlds_count;++i){require(std::isfinite(returns[i]),"Nonfinite episode return");sums[i%6]+=returns[i];++counts[i%6];}
    std::vector<float> result(worlds_count);for(int i=0;i<worlds_count;++i){require(counts[i%6]>1,"LOO context has no peers");
        const double baseline=(sums[i%6]-returns[i])/(counts[i%6]-1);result[i]=float((returns[i]-baseline)/scale);require(std::isfinite(result[i]),"Invalid LOO advantage");}return result;
}
std::vector<uint8_t> baseline_bytes(const Baseline& b,uint64_t protocol){cp::Writer w;w.value(uint32_t{0x31534148});w.value(uint32_t{1});w.value(protocol);w.value(b.scale);
    for(double x:b.means)w.value(x);for(int x:b.counts)w.value(x);return w.data;}
Baseline read_scale(const fs::path& root,uint64_t protocol){const auto bytes=cp::read((root/"fixed-scale.bin").string());cp::Reader r{bytes};
    require(r.value<uint32_t>()==0x31534148&&r.value<uint32_t>()==1&&r.value<uint64_t>()==protocol,"Wrong calibration scale identity");Baseline b;b.scale=r.value<double>();
    for(double& x:b.means)x=r.value<double>();for(int& x:b.counts)x=r.value<int>();r.finish();
    require(std::isfinite(b.scale)&&b.scale>0,"Invalid fixed scale");for(int k=0;k<6;++k)require(std::isfinite(b.means[k])&&b.counts[k]==(k<4?43:42),"Invalid calibration context");return b;
}
void save_policy(Data& d,const fs::path& root,uint64_t protocol,int update){
    const auto file=root/("amplitude-policy-update-"+std::to_string(update)+".bin");require(!fs::exists(file),"Policy checkpoint identity exists");
    const auto parameters=download(d.parameters.get(),2,d.stream->value);cp::Writer w,adam;d.optimizer->save(adam);
    w.value(policy_magic);w.value(uint32_t{1});w.value(protocol);w.value(cp::checksum(parent_checkpoint_payload));
    for(uint32_t x:{6u,6u,3u,4u,41u,4u,336u})w.value(x);w.value(update);w.value(d.optimizer->steps());
    w.array(d.backbone->get_params());w.array(parameters);w.value(math::sigmoid(parameters[0]));w.value(math::standard_deviation(parameters[1]));w.array(adam.data);cp::write_atomic(file.string(),w.data);
    const auto bytes=cp::read(file.string());cp::Reader r{bytes};
    require(r.value<uint32_t>()==policy_magic&&r.value<uint32_t>()==1&&r.value<uint64_t>()==protocol&&r.value<uint64_t>()==cp::checksum(parent_checkpoint_payload),"Policy identity roundtrip failed");
    for(uint32_t x:{6u,6u,3u,4u,41u,4u,336u})require(r.value<uint32_t>()==x,"Policy physical/shape contract differs");
    require(r.value<int>()==update&&r.value<int>()==d.optimizer->steps()&&r.array<float>(2532)==resume_state->actor.parameters&&r.array<float>(2)==parameters&&
            r.value<float>()==math::sigmoid(parameters[0])&&r.value<float>()==math::standard_deviation(parameters[1])&&r.array<uint8_t>()==adam.data,"Policy/Adam roundtrip failed");r.finish();
    cp::Reader a{adam.data};const auto config=a.value<nn::AdamConfig>();const int steps=a.value<int>();const auto first=a.array<float>(2),second=a.array<float>(2);a.finish();
    require(config.lr==.05f&&config.beta1==.9f&&config.beta2==.999f&&config.eps==.00001f&&config.max_grad_norm==.5f&&steps>=0&&steps<=maximum_adam_steps&&first.size()==2&&second.size()==2,"Amplitude Adam contract differs");
    finite(parameters);finite(first);finite(second);for(float x:second)require(x>=0,"Invalid Adam variance");
}
double optimize(Data& d,const Collected& batch,const Baseline& baseline,int update,const fs::path& root){
    const auto advantages=leave_one_out(batch.returns,baseline.scale);const auto stream=d.stream->value;d.advantages.copy_from_host(advantages.data(),worlds_count);
    cp::Writer signal;signal.value(uint32_t{0x31474148});signal.value(uint32_t{1});signal.value(update);signal.value(baseline.scale);signal.array(batch.returns);signal.array(advantages);
    cp::write_atomic((root/("advantages-"+std::to_string(update)+".bin")).string(),signal.data);
    const int before=d.optimizer->steps();double kl=0;
    for(int epoch=0;epoch<maximum_epochs;++epoch){CUDA_CHECK_THROW(cudaMemsetAsync(d.stats.get(),0,5*sizeof(float),stream));
        const auto permutation=rl::make_permutation(worlds_count,0x1c72931u+uint32_t(update*maximum_epochs+epoch));
        for(int first=0;first<worlds_count;first+=minibatch_episodes){wall_guard();
            rl::gather_rows(permutation,first,64,1,d.latent.get(),d.mb_latent.get(),stream);rl::gather_rows(permutation,first,64,1,d.old_logp.get(),d.mb_oldlogp.get(),stream);
            rl::gather_rows(permutation,first,64,1,d.advantages.get(),d.mb_advantage.get(),stream);
            episode_loss<<<1,64,0,stream>>>(d.parameters.get(),d.mb_latent.get(),d.mb_oldlogp.get(),d.mb_advantage.get(),64,d.row_gradient.get(),d.row_stats.get());
            CUDA_CHECK_THROW(cudaMemsetAsync(d.gradient.get(),0,2*sizeof(float),stream));rl::sum_columns(d.row_gradient.get(),64,2,d.gradient.get(),stream);
            rl::sum_columns(d.row_stats.get(),64,5,d.stats.get(),stream);const auto gradient=download(d.gradient.get(),2,stream);finite(gradient);
            d.optimizer->step(d.parameters.get(),d.gradient.get(),stream);const auto parameters=download(d.parameters.get(),2,stream);finite(parameters);
            require(math::standard_deviation(parameters[1])>=.05f&&math::standard_deviation(parameters[1])<=8.f,"Learned sigma outside smooth contract");
        }
        const auto stats=download(d.stats.get(),5,stream);finite(stats);kl=stats[3]/worlds_count;if(kl>.01)break;
    }
    require(d.optimizer->steps()-before>=4&&d.optimizer->steps()-before<=16&&d.optimizer->steps()<=maximum_adam_steps,"Episode PPO optimizer budget exceeded");
    require(d.backbone->get_params()==resume_state->actor.parameters,"Immutable backbone changed during scalar PPO");return kl;
}
__global__ void math_fixture(float* output){
    const int i=threadIdx.x;if(i>=30)return;const float variance[5]={-80,-3,logf(2.975f),3,80};
    const float parameters[2]={i%3==0?-4.59512f:i%3==1?0.f:4.59512f,variance[(i/3)%5]};
    const float sigma=math::standard_deviation(parameters[1]),u=parameters[0]+sigma*(i%2?-.7f:1.2f),adv=i<15?.8f:-.8f;
    float gradient[2];const float old=math::log_probability(parameters,u);const auto row=math::loss(parameters,u,old,adv,1.f,gradient);
    output[i*7]=parameters[0];output[i*7+1]=parameters[1];output[i*7+2]=u;output[i*7+3]=old;
    output[i*7+4]=gradient[0];output[i*7+5]=gradient[1];output[i*7+6]=row.pg_loss;
}
void numerical(const fs::path& dir){
    Stream stream;device_ptr<float> output(30*7);math_fixture<<<1,32,0,stream.value>>>(output.get());
    const auto values=download(output.get(),30*7,stream.value);double max_logp=0,max_gradient=0,max_fd=0;
    for(int i=0;i<30;++i){double parameters[2]={values[i*7],values[i*7+1]},gradient[2];const double u=values[i*7+2],old=values[i*7+3],adv=i<15?.8:-.8;
        const double lp=math::log_probability(parameters,u);max_logp=std::max(max_logp,std::abs(lp-old));
        (void)math::loss(parameters,u,old,adv,1.,gradient);
        for(int j=0;j<2;++j){max_gradient=std::max(max_gradient,std::abs(gradient[j]-values[i*7+4+j]));
            const double saved=parameters[j],epsilon=1e-5;double unused[2];parameters[j]=saved+epsilon;
            const auto plus=math::loss(parameters,u,old,adv,1.,unused);parameters[j]=saved-epsilon;
            const auto minus=math::loss(parameters,u,old,adv,1.,unused);parameters[j]=saved;
            max_fd=std::max(max_fd,std::abs((plus.pg_loss-minus.pg_loss)/(2*epsilon)-gradient[j]));}
        require(math::standard_deviation(parameters[1])>=.05&&math::standard_deviation(parameters[1])<=8.,"Smooth sigma numerical domain failed");
    }
    require(max_logp<2e-5&&max_gradient<2e-4&&max_fd<2e-4,"Whole-episode probability/gradient numerical fixture failed");
    // Prove that an episode's own return does not enter its LOO baseline.
    std::vector<float> returns(worlds_count);for(int i=0;i<worlds_count;++i)returns[i]=float(i%6)+float(i/6)*.1f;
    const auto before=leave_one_out(returns,2);const int selected=17;const double baseline_before=returns[selected]-2*before[selected];
    returns[selected]+=7;const auto after=leave_one_out(returns,2);const double baseline_after=returns[selected]-2*after[selected];
    require(std::abs(baseline_before-baseline_after)<1e-5,"Own return entered the LOO baseline");
    const auto scale=fit_scale(returns);require(scale.scale>0,"Calibration scale numerical fixture failed");
    // Check long scoring at both exact horizons, including reset peaks and N-160.
    for(int n:{1280,6000}){HoldScore score(n);m::Metrics sample;for(int k=0;k<=n;++k){sample.forward_m=k>=n-160?float(k-(n-160))*.000001f:0;
            require(score.record(k,sample),"Generalized quiet score fixture failed");}require(score.passed()&&score.complete(),"Quiet generalized hold did not pass");
        HoldScore failure(n);sample={};sample.depth_m=.0011f;require(failure.record(0,sample),"Initial score fixture rejected sample");sample={};
        for(int k=1;k<=n;++k)require(failure.record(k,sample),"Generalized initial-peak fixture failed");require(!failure.passed(),"Initial violation was lost");}
    auto config=initial_optimizer_config(true);config.lr=.05f;nn::GpuAdam first(2,config),second(2,config);cp::Writer a,b;first.save(a);cp::Reader reader{a.data};second.load(reader);reader.finish();second.save(b);
    require(first.steps()==0&&second.steps()==0&&a.data==b.data,"Fresh two-parameter Adam zero-step roundtrip failed");
    std::ofstream report(dir/"amplitude-math-zero.json");report<<std::setprecision(17)<<"{\"passed\":true,\"actual_CUDA_cases\":30,\"max_logp_error\":"<<max_logp
      <<",\"max_gradient_error\":"<<max_gradient<<",\"max_finite_difference_error\":"<<max_fd
      <<",\"LOO_own_return_excluded\":true,\"generalized_64_and_300s_score\":true,\"fresh_Adam_roundtrip\":true,\"physical_controls\":0,\"optimizer_steps\":0}\n";
}
void zero_fixtures(const fs::path& dir){
    require(fs::create_directory(dir),"Numerical fixture identity exists");numerical(dir);
    Data d(false);const auto stream=d.stream->value;
    initialize_episodes<<<2,128,0,stream>>>(d.worlds.get(),d.tasks.get(),d.counts.get(),d.parameters.get(),d.latent.get(),d.gain.get(),d.old_logp.get(),d.initial_metrics.get(),d.fault.get(),calibration_seed,0);
    observe_episodes<<<2,128,0,stream>>>(d.worlds.get(),d.tasks.get(),d.observations.get(),d.fault.get());
    d.backbone->forward(d.observations.get(),worlds_count,stream);
    const auto world=download(d.worlds.get(),worlds_count,stream);const auto tasks=download(d.tasks.get(),worlds_count,stream);
    const auto inputs=download(d.observations.get(),worlds_count*41,stream),output=download(d.backbone->output(),worlds_count*4,stream);
    const auto u=download(d.latent.get(),worlds_count,stream),g=download(d.gain.get(),worlds_count,stream),oldp=download(d.old_logp.get(),worlds_count,stream);
    const auto policy=initial_parameters();const float sigma=math::standard_deviation(policy[1]);double observation_error=0,migration_error=0,gpu_error=0,sample_error=0,density_error=0;
    nn::CpuMlp<float> old(old_spec(4)),current(spec(4));old.params()=original40.actor.parameters;current.params()=resume_state->actor.parameters;int cases[6]{};
    for(int i=0;i<worlds_count;++i){const int id=i%6;++cases[id];auto expected=m::initial_world(case_pose(id));expected.arena=m::long_swim_arena();m::ForwardTask task;const auto config=hold_config();
        require(m::begin_forward(expected,config,task)&&m::offset_episode_start(expected,task,config,case_offset(id))&&m::update_velocity_request(expected,config,task),"CPU all-hold reset failed");
        float x[41];require(m::observe_appended(task,expected,config,x),"CPU all-hold observation failed");
        require(world[i].attempted_substeps==0&&world[i].physical.control_steps==0&&world[i].physical.substeps==0&&tasks[i].controls==0&&tasks[i].hold_anchor_set&&p::norm(tasks[i].request.terminal_velocity)==0,"Numerical setup advanced physics or retained swimming goal");
        for(int j=0;j<41;++j)observation_error=std::max(observation_error,double(std::abs(x[j]-inputs[i*41+j])));
        old.forward(x,1);current.forward(x,1);for(int j=0;j<4;++j){migration_error=std::max(migration_error,double(std::abs(old.output()[j]-current.output()[j])));gpu_error=std::max(gpu_error,double(std::abs(output[i*4+j]-current.output()[j])));}
        const float sample=policy[0]+sigma*cudarl::rng::normal(calibration_seed,0,uint32_t(i));sample_error=std::max(sample_error,double(std::abs(sample-u[i])));
        const double p0[2]={policy[0],policy[1]};density_error=std::max(density_error,std::abs(math::log_probability(p0,double(u[i]))-oldp[i]));
        require(std::abs(math::sigmoid(u[i])-g[i])<2e-7&&std::isfinite(g[i])&&g[i]>=0&&g[i]<=1,"Sampled gain transform changed");
    }
    for(int k=0;k<6;++k)require(cases[k]==(k<4?43:42),"All-hold case count fixture failed");
    require(download(d.fault.get(),1,stream)[0]==0&&observation_error<1e-6&&migration_error<1e-6&&gpu_error<2e-5&&sample_error<2e-5&&density_error<1e-4,"All-hold CPU/CUDA fixture failed");
    require(resume_state->actor.parameters==rl::append_zero_input_column(original40.actor.parameters)&&migrated41.ppo.actor_adam.steps==21700&&migrated41.actor_column_age==0,"Frozen backbone/old optimizer migration changed");
    std::ofstream report(dir/"all-hold-zero.json");report<<std::setprecision(17)<<"{\"passed\":true,\"worlds\":256,\"all_holding\":true,\"observation_error\":"<<observation_error<<",\"migration_error\":"<<migration_error
      <<",\"CPU_CUDA_actor_error\":"<<gpu_error<<",\"sample_error\":"<<sample_error<<",\"old_logp_error\":"<<density_error<<",\"physical_controls\":0,\"optimizer_steps\":0}\n";
}
struct ReturnBounds {double p99=0,maximum=0;bool pass=false;};
ReturnBounds return_bounds(const std::vector<float>& returns){std::vector<float> values;for(float r:returns){require(std::isfinite(r),"Nonfinite return before actor update");values.push_back(std::abs(r));}
    require(values.size()==worlds_count,"Return distribution shape differs");std::sort(values.begin(),values.end());ReturnBounds b;b.p99=values[size_t(std::ceil(.99*values.size()))-1];b.maximum=values.back();b.pass=b.p99<=5&&b.maximum<=20;return b;}
void result(const fs::path& file,const Budget& budget,const std::string& fault,const std::string& gate,int updates_done,int steps,double prior,bool original_success,bool qualified,int withheld,int endurance){
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();std::ofstream out(file);out<<std::setprecision(17)
      <<"{\"controls\":"<<budget.controls<<",\"substeps\":"<<budget.substeps<<",\"active_seconds\":"<<elapsed<<",\"prior_active_seconds\":"<<prior
      <<",\"PPO_updates\":"<<updates_done<<",\"actor_steps\":"<<steps<<",\"critic_steps\":0,\"critic_forward_calls\":0,\"fault\":"<<std::quoted(fault)<<",\"gate\":"<<std::quoted(gate)
      <<",\"original_success\":"<<(original_success?"true":"false")<<",\"qualified\":"<<(qualified?"true":"false")<<",\"withheld_passes\":"<<withheld<<",\"endurance_passes\":"<<endurance
      <<",\"reserve_consumed\":"<<(original_success?"true":"false")<<",\"closed\":true,\"runtime_adopted\":false,\"retries\":0}\n";out.close();require(bool(out),"Cannot write terminal study result");
    std::ofstream live(file.parent_path()/"status.json");live<<std::setprecision(17)<<"{\"phase\":"<<std::quoted(file.filename()=="calibration-result.json"?"calibration_closed":"training_closed")
      <<",\"completed_updates\":"<<updates_done<<",\"optimizer_steps\":"<<steps<<",\"controls\":"<<budget.controls<<",\"substeps\":"<<budget.substeps
      <<",\"unix_seconds\":"<<unix_now()<<",\"closed\":true,\"qualified\":"<<(qualified?"true":"false")<<",\"fault\":"<<std::quoted(fault)<<"}\n";
}
int prepare(const fs::path& root,const fs::path& protocol){
    require(!fs::exists(root)&&fs::is_regular_file(protocol),"New identity or frozen protocol missing");require(fs::create_directories(root),"Cannot create whole-episode run directory");
    fs::copy_file(protocol,root/"approved-protocol.md");configure(root,false);fs::copy_file(parent_checkpoint_path,root/"parent-checkpoint.bin");fs::copy_file(parent_protocol_path,root/"parent-protocol.bin");
    const auto bytes=protocol_bytes(root);cp::write_atomic((root/"protocol.bin").string(),bytes);cp::write_atomic((root/"migrated-initial.bin").string(),rl::write_appended_hold(migrated41,cp::checksum(bytes)));
    require(fs::create_directory(root/"initial-policy"),"Initial policy identity exists");Data d(true);save_policy(d,root/"initial-policy",cp::checksum(bytes),0);zero_fixtures(root/"prepare-zero");
    std::ofstream info(root/"prepared.json");info<<"{\"prepared\":true,\"worlds\":256,\"controls_per_episode\":1280,\"updates_max\":32,\"controls_cap\":11194560,\"substeps_cap\":111945600,\"physical_controls\":0,\"optimizer_steps\":0}\n";return 0;
}
int preflight(const fs::path& root){configure(root,true);validate_protocol(root);zero_fixtures(root/"verified-zero");std::ofstream marker(root/"preflight-passed.txt");marker<<"Zero-step Gaussian/gradient/LOO/scorer/backbone/all-hold checks passed.\n";return 0;}
int calibrate(const fs::path& root){
    configure(root,true);validate_protocol(root);require(fs::exists(root/"preflight-passed.txt"),"Whole-episode preflight missing");require(fs::create_directory(root/"calibration-started"),"Calibration already attempted; no retry");
    failure_directory=root/"calibration-started";started=std::chrono::steady_clock::now();active_wall_cap=seconds_cap;Budget budget;
    budget.maximum_controls=calibration_controls;budget.maximum_substeps=calibration_controls*10;budget.seconds=seconds_cap;
    std::string fault,gate;
    try{
        const auto original=assess(resume_state->actor.parameters,1,root/"baseline",budget);
        require(original.short_pass==4&&original.hold_pass==0&&original.long_pass==28&&preserved_long_windows(root/"baseline"/"long"),"Original336 physical reference changed");
        const auto parameters=initial_parameters();const float initial_gain=math::sigmoid(parameters[0]);
        const auto initial=hold_only(resume_state->actor.parameters,initial_gain,original_cases(),root/"initial-hold",budget,"hold");
        require(initial.passed==0&&initial.completed==6,"Initial near-parent policy does not reproduce failing holding baseline");
        Data d(false);const auto batch=collect(d,budget,root/"calibration-started","calibration",0,true);
        const auto baseline=fit_scale(batch.returns);const auto bytes=baseline_bytes(baseline,cp::checksum(protocol_bytes(root)));cp::write_atomic((root/"fixed-scale.bin").string(),bytes);
        require(baseline_bytes(read_scale(root,cp::checksum(protocol_bytes(root))),cp::checksum(protocol_bytes(root)))==bytes,"Calibration scale roundtrip failed");
        const auto bounds=return_bounds(batch.returns);const bool pass=batch.quiet>=4&&batch.quiet_pose[0]>0&&batch.quiet_pose[1]>0&&bounds.pass;
        std::ofstream report(root/"calibration-report.json");report<<std::setprecision(17)<<"{\"passed\":"<<(pass?"true":"false")<<",\"episodes\":256,\"quiet_complete_64s\":"<<batch.quiet
          <<",\"quiet_pose0\":"<<batch.quiet_pose[0]<<",\"quiet_pose3\":"<<batch.quiet_pose[1]<<",\"initial_deterministic_gain\":"<<initial_gain<<",\"initial_holds_passed\":0"
          <<",\"fixed_global_residual_RMS\":"<<baseline.scale<<",\"return_abs_p99\":"<<bounds.p99<<",\"return_abs_max\":"<<bounds.maximum
          <<",\"max_old_logp_error\":"<<batch.maximum_probability_error<<",\"max_reward_sum_error\":"<<batch.maximum_return_error
          <<",\"controls\":"<<budget.controls<<",\"substeps\":"<<budget.substeps<<",\"actor_steps\":0,\"critic_steps\":0}\n";report.close();require(bool(report),"Calibration report write failed");
        if(!pass)gate="calibration_whole_episode_coverage_or_return_scale";
        require(budget.controls==calibration_controls&&budget.substeps==calibration_controls*10,"PhaseA whole-episode accounting changed");wall_guard();
    }catch(const std::exception& error){fault=error.what();std::cerr<<fault<<std::endl;}
    result(root/"calibration-result.json",budget,fault,gate,0,0,0,false,false,0,0);
    if(fault.empty()&&gate.empty()){cp::Writer ready;ready.value(ready_magic);ready.value(uint32_t{1});ready.value(cp::checksum(protocol_bytes(root)));
        for(const auto& path:{root/"fixed-scale.bin",root/"calibration-report.json",root/"calibration-result.json",root/"baseline"/"long"/"sustained-windows.csv"})ready.value(file_hash(path));
        ready.value(std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count());cp::write_atomic((root/"calibration-ready.bin").string(),ready.data);}
    return fault.empty()?0:3;
}
double release(const fs::path& root,uint64_t protocol){const auto bytes=cp::read((root/"calibration-ready.bin").string());cp::Reader r{bytes};
    require(r.value<uint32_t>()==ready_magic&&r.value<uint32_t>()==1&&r.value<uint64_t>()==protocol,"Unqualified calibration identity");
    for(const auto& path:{root/"fixed-scale.bin",root/"calibration-report.json",root/"calibration-result.json",root/"baseline"/"long"/"sustained-windows.csv"})require(r.value<uint64_t>()==file_hash(path),"Calibrated evidence changed");
    const double prior=r.value<double>();r.finish();require(std::isfinite(prior)&&prior>0&&prior<seconds_cap,"Invalid prior active time");return prior;}
int train(const fs::path& root){
    configure(root,true);validate_protocol(root);const auto protocol=cp::checksum(protocol_bytes(root));const auto prior=release(root,protocol);const auto baseline=read_scale(root,protocol);
    const auto original=long_keys(root/"baseline"/"long"/"sustained-windows.csv");require(original.size()==28,"Original retained-window set changed");
    require(fs::create_directory(root/"training-started"),"Training already attempted; no retry");failure_directory=root/"training-started";
    started=std::chrono::steady_clock::now();active_wall_cap=seconds_cap-prior;Budget budget;
    budget.maximum_controls=controls_cap;budget.maximum_substeps=controls_cap*10;budget.seconds=active_wall_cap;budget.controls=calibration_controls;budget.substeps=calibration_controls*10;
    std::string fault,gate;int updates_done=0,steps=0,withheld_passes=0,endurance_passes=0;bool original_success=false,qualified=false;
    std::unique_ptr<Data> active;
    std::ofstream progress(root/"updates.jsonl");progress<<std::setprecision(17);
    try{
        active=std::make_unique<Data>(true);auto& d=*active;save_policy(d,root/"training-started",protocol,0);
        for(int update=1;update<=maximum_updates;++update){
            require(budget.controls+trace_rows+10720<=ordinary_controls_cap,"Ordinary lesson budget exhausted");
            const auto batch=collect(d,budget,root/"training-started","training",update,false);
            const auto bounds=return_bounds(batch.returns);
            if(update==1&&(batch.quiet<4||batch.quiet_pose[0]==0||batch.quiet_pose[1]==0||!bounds.pass)){gate="first_fresh_whole_episode_coverage_or_return_scale";break;}
            const double kl=optimize(d,batch,baseline,update,root/"training-started");updates_done=update;d.completed=update;steps=d.optimizer->steps();save_policy(d,root/"training-started",protocol,update);
            const auto parameters=download(d.parameters.get(),2,d.stream->value);const float gain=math::sigmoid(parameters[0]);
            const auto measured=assess(resume_state->actor.parameters,gain,root/"training-started"/("assessment-"+std::to_string(update)),budget);
            const bool retained=measured.short_pass==4&&std::includes(measured.keys.begin(),measured.keys.end(),original.begin(),original.end());
            progress<<"{\"update\":"<<update<<",\"mu\":"<<parameters[0]<<",\"variance_coordinate\":"<<parameters[1]<<",\"sigma\":"<<math::standard_deviation(parameters[1])
              <<",\"deterministic_gain\":"<<gain<<",\"short\":"<<measured.short_pass<<",\"holds\":"<<measured.hold_pass<<",\"long\":"<<measured.long_pass
              <<",\"retained\":"<<(retained?"true":"false")<<",\"quiet_training_episodes\":"<<batch.quiet<<",\"KL\":"<<kl<<",\"actor_steps\":"<<steps<<",\"controls\":"<<budget.controls<<"}\n";
            progress.flush();require(bool(progress),"Whole-episode update report write failed");
            if(!retained){gate="regressed_swimming";break;}
            if(measured.hold_pass==6){
                original_success=true;require(fs::create_directory(root/"reserved-validation"),"Reserve already attempted");
                const auto withheld=hold_only(resume_state->actor.parameters,gain,withheld_cases(),root/"reserved-validation"/"withheld",budget,"hold");withheld_passes=withheld.passed;
                const auto endurance=hold_only(resume_state->actor.parameters,gain,endurance_cases(),root/"reserved-validation"/"endurance",budget,"hold");endurance_passes=endurance.passed;
                qualified=withheld.completed==6&&withheld.passed==6&&endurance.completed==2&&endurance.passed==2;
                gate=qualified?"qualified_after_withheld_endurance_no_adoption":"reserved_validation_failed";break;
            }
        }
        steps=d.optimizer->steps();require(steps<=maximum_adam_steps&&d.backbone->get_params()==resume_state->actor.parameters,"Whole-episode optimizer/backbone final integrity fault");wall_guard();
    }catch(const std::exception& error){fault=error.what();std::cerr<<fault<<std::endl;}
    if(active&&active->optimizer)steps=active->optimizer->steps();
    result(root/"training-result.json",budget,fault,gate,updates_done,steps,prior,original_success,qualified,withheld_passes,endurance_passes);return fault.empty()?0:3;
}
} // namespace amplitude
int main(int argc,char** argv){try{
    require(argc>=3,"Use mode isolated-run-directory [protocol-file]");const std::string mode=argv[1];const auto root=fs::absolute(argv[2]);
    if(mode=="--numerical"){require(argc==3&&!fs::exists(root),"Numerical identity exists");fs::create_directories(root);amplitude::numerical(root);return 0;}
    if(mode=="--prepare"){require(argc==4,"Prepare requires frozen protocol");return amplitude::prepare(root,fs::absolute(argv[3]));}
    require(argc==3,"Unexpected whole-episode study argument");
    if(mode=="--preflight")return amplitude::preflight(root);if(mode=="--calibrate")return amplitude::calibrate(root);if(mode=="--train")return amplitude::train(root);
    throw std::runtime_error("Unknown whole-episode study mode");
}catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 2;}}
