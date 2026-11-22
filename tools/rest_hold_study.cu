// Approved rest-hold curriculum. Shared native physics, PPO, migration and diagnostics.
#define CUDARL_REST_HOLD_STUDY 1
#include "hold_scale_study.cu"
#include "motor/hold_assessment.h"
#include <thread>
namespace {
bool calibration_collect=false;
#ifdef CUDARL_REST_HOLD_RESET_STUDY
bool reset_occupancy_passed=false;
#endif
std::vector<float> verified_adv,verified_returns;
fs::path active_root;
uint32_t rest_action_seed(){
#ifdef CUDARL_REST_HOLD_MEASURED_BASELINE
    return calibration_collect?314159265u:seed+2;
#else
    return calibration_collect?2718281828u:seed+2;
#endif
}
uint32_t rest_action_counter(int update,int t){return calibration_collect?uint32_t(t):430080u+uint32_t(update-336)*320u+uint32_t(t);}
std::vector<uint8_t> rest_protocol(){
    cp::Writer out;
#ifdef CUDARL_REST_HOLD_RESET_STUDY
#ifdef CUDARL_REST_HOLD_MEASURED_BASELINE
    out.value(uint32_t{21});out.value(uint32_t{1});
#elif defined(CUDARL_REST_HOLD_TASK_CRITIC)
    out.value(uint32_t{20});out.value(uint32_t{2});out.value(uint32_t{2466});
#else
    out.value(uint32_t{19});
#endif
    out.value(uint32_t{20});out.value(uint32_t{256});out.value(float{.1f});
#else
    out.value(uint32_t{18});
#endif
    out.value(seed);out.value(cp::checksum(parent_checkpoint_payload));out.value(cp::checksum(parent_protocol_payload));
    for(int n:{41,9,336,8,32,320,256,2,80,4,5120,113664,1136640,430080,120})out.value(n);
#ifdef CUDARL_REST_HOLD_MEASURED_BASELINE
    out.value(uint32_t{314159265u});
#else
    out.value(uint32_t{2718281828u});
#endif
    out.value(uint32_t{3141592653u});
    for(float v:{1.f/1024,.000125f,.001f,.8f,.99f,1.f,.5f,.02f,5.f,20.f,.05f,.75f,.25f,.10f,1.1f,.90f})out.value(v);
#ifdef CUDARL_REST_HOLD_RESET_STUDY
    const std::string text="rest0;centeredholding50%;swim50%originaldepthmix;20-step hold finite terminal;reset physical+AR;10% occupancy;timefeature1at0;all4NN;calibrationworldsplit;80criticsteps;split report before PPO;8PPOmax;noadoption/retry/extension";
#else
    const std::string text="rest0;centeredholding50%;swim50%originaldepthmix;timefeature1at0;all4NN;calibrationworldsplit;80criticsteps;gateandreportbeforePPO;8PPOmax;noadoption/retry/extension";
#endif
    out.array(std::vector<uint8_t>(text.begin(),text.end()));
#ifdef CUDARL_REST_HOLD_MEASURED_BASELINE
    const std::string measured="Monte Carlo raw returns;fit-only task/pose/elapsed means;fixed per-task residual RMS;whole-world held-out occupancy and squared scaled residual share;no critic";
    out.array(std::vector<uint8_t>(measured.begin(),measured.end()));
#elif defined(CUDARL_REST_HOLD_TASK_CRITIC)
    const std::string critic="shared inherited hidden layers;swim initialized raw identity;hold initialized fit mean;fit-world RMS only;raw-return GAE;fresh Adam existing numeric config;strict held-out mean comparator";
    out.array(std::vector<uint8_t>(critic.begin(),critic.end()));
#endif
    return out.data;
}
void configure_rest(const fs::path& root,bool copied){
    configure_append(root,copied);horizon=320;rows=10240;updates=8;training_cap=81920;control_cap=113664;substep_cap=1136640;active_wall_cap=120;
    training_episode.rest_hold=true;training_episode.balanced_hold=true;training_episode.holding_reward_gain=1.f/1024;
    require(training_episode.valid()&&seed==1247037243&&correlation==.8f,"Rest lesson configuration mismatch");
}
m::ForwardConfig rest_config(int maximum=m::RestHoldAssessment::controls){auto c=m::config_for_speed(.008f);c.maximum_controls=maximum;c.observation_schema=4;
    c.target_speed_m_s=0;c.stationary_request=true;c.position_hold=true;return c;}
void save_rest(const fs::path& root,const char* name,int update,nn::GpuMlp& actor,nn::GpuMlp& critic,nn::GpuAdam& ao,nn::GpuAdam& co){
    const auto directory=root/name;require(fs::create_directory(directory),"Checkpoint phase exists; no overwrite");save_append(directory,update,actor,critic,ao,co);
    const auto loaded=rl::read_appended_hold(cp::read((directory/("ppo-append-update-"+std::to_string(update)+".bin")).string()),cp::checksum(rest_protocol()));
    require(loaded.ppo.actor.parameters==actor.get_params()&&loaded.ppo.critic==critic.get_params()&&loaded.ppo.actor_adam.steps==ao.steps()&&loaded.ppo.critic_adam.steps==co.steps()&&
            loaded.actor_column_age==ao.fresh_steps()&&loaded.critic_column_age==co.fresh_steps(),"Rest checkpoint reload/independent ages failed");
}
__global__ void rest_probe(m::TrainingEpisode e,float* output){const int i=threadIdx.x;const int t[4]={0,39,40,319};
    for(int k=0;k<4;++k){const auto c=e.task_config(t[k],i);m::Metrics metrics;metrics.velocity_m_s={.001f,0,0};m::World w;
        const auto costs=m::reward_costs(metrics,w,c,false);output[i*16+k*4]=c.target_speed_m_s;output[i*16+k*4+1]=e.reward_gain(t[k],i);
        output[i*16+k*4+2]=float(c.maximum_controls);output[i*16+k*4+3]=-p::dt*costs.total()*e.reward_gain(t[k],i);}}
#ifdef CUDARL_REST_HOLD_TASK_CRITIC
void task_critic_zero_fixture(const fs::path& root);
#endif
#ifdef CUDARL_REST_HOLD_MEASURED_BASELINE
void measured_zero_fixture(const fs::path& root);
#endif
void rest_zero_checks(const fs::path& root){
#ifdef CUDARL_REST_HOLD_MEASURED_BASELINE
    measured_zero_fixture(root);
#endif
#ifdef CUDARL_REST_HOLD_TASK_CRITIC
    task_critic_zero_fixture(root);
#endif
    Stream stream;Rollout data(stream.value);gpu::initialize<<<1,32,0,stream.value>>>(data.worlds.get(),data.tasks.get(),32,data.fault.get(),training_episode);
    gpu::observe<<<1,32,0,stream.value>>>(data.worlds.get(),data.tasks.get(),data.obs.get(),32,data.fault.get(),training_episode);
    device_ptr<float> probe(512);rest_probe<<<1,32,0,stream.value>>>(training_episode,probe.get());CUDA_CHECK_THROW(cudaGetLastError());
    const auto observed=download(data.obs.get(),32*41,stream.value),probed=download(probe.get(),512,stream.value);
    const auto worlds=download(data.worlds.get(),32,stream.value);
    const auto tasks=download(data.tasks.get(),32,stream.value);require(download(data.fault.get(),1,stream.value)[0]==0,"Zero-step GPU initialization failed");
    nn::CpuMlp<float> oa(old_spec(4)),na(spec(4)),oc(old_spec(1)),nc(spec(1));oa.params()=original40.actor.parameters;na.params()=migrated41.ppo.actor.parameters;oc.params()=original40.critic;nc.params()=migrated41.ppo.critic;
    float obs_error=0,function_error=0;int group[2][2]{};const int times[4]={0,39,40,319};
    for(int i=0;i<32;++i){const bool held=(i/2)%2;++group[i%2][held];m::World w;m::ForwardTask t;const auto c=training_episode.task_config(0,i);
        require(m::begin_training_episode(i,false,c,w,t,training_episode.arena(),training_episode.initial_vertical_offset(i)),"CPU rest initialization failed");
        if(held)require(m::update_velocity_request(w,c,t),"CPU rest anchor failed");float x[41]{},oldx[40]{};require(m::observe_appended(t,w,c,x),"CPU rest observation failed");
        auto legacy=c;legacy.position_hold=false;legacy.observation_schema=2;require(m::observe(t,w,legacy,oldx),"CPU legacy40 observation failed");
        for(int j=0;j<41;++j)obs_error=std::max(obs_error,fabsf(x[j]-observed[i*41+j]));for(int j=0;j<40;++j)require(x[j]==oldx[j],"Original40 meanings changed");
        require(x[34]==1&&x[40]==0&&tasks[i].controls==0&&worlds[i].attempted_substeps==0&&worlds[i].physical.control_steps==0&&tasks[i].start_com_model.y==-2,"Rest init clock/reference/counters wrong");
        require(tasks[i].hold_anchor_set==held,"Rest anchor activation wrong");for(float v:worlds[i].physical.body.velocity)require(v==0,"Initial velocity nonzero");
        for(int j=0;j<4;++j)require(worlds[i].physical.held_targets[j]==worlds[i].physical.body.angle[j]&&worlds[i].reference_rate[j]==0,"Initial filter/pose changed");
        for(int k=0;k<4;++k){const auto cfg=training_episode.task_config(times[k],i);m::Metrics metrics;metrics.velocity_m_s={.001f,0,0};m::World original;
            const auto costs=m::reward_costs(metrics,original,cfg,false);const int at=i*16+k*4;
            require(probed[at]==(held?0.f:.008f)&&probed[at+1]==(held?1.f/1024:1.f)&&probed[at+2]==320&&fabsf(probed[at+3]+p::dt*costs.total()*training_episode.reward_gain(times[k],i))<1e-7f,"CPU/CUDA rest gain/reward/horizon mismatch");}
        oa.forward(oldx,1);na.forward(x,1);oc.forward(oldx,1);nc.forward(x,1);for(int j=0;j<4;++j)function_error=std::max(function_error,fabsf(oa.output()[j]-na.output()[j]));function_error=std::max(function_error,fabsf(oc.output()[0]-nc.output()[0]));
        if(held){for(float displacement:{-.02f,0.f,.02f}){auto scratch=w;scratch.physical.body.position+=t.request.heading*(displacement/p::scale);float z[41]{};require(m::observe_appended(t,scratch,c,z),"Signed rest observation failed");
            require(fabsf(z[40]+tanhf(displacement/.02f))<1e-5f,"Signed hold feature reversed");}}
    }
    for(auto& g:group)for(int count:g)require(count==8,"Rest goal/pose mix unbalanced");require(obs_error<=1e-6f&&function_error<=1e-6f,"Rest zero-step parity failed");
    for(const auto* pair:{&migrated41.ppo.actor_adam,&migrated41.ppo.critic_adam})require(pair->steps==21700,"Inherited optimizer clock changed");
    require(migrated41.ppo.actor.parameters==rl::append_zero_input_column(original40.actor.parameters)&&migrated41.ppo.critic==rl::append_zero_input_column(original40.critic)&&
        migrated41.ppo.actor_adam.first==rl::append_zero_input_column(original40.actor_adam.first)&&migrated41.ppo.actor_adam.second==rl::append_zero_input_column(original40.actor_adam.second)&&
        migrated41.ppo.critic_adam.first==rl::append_zero_input_column(original40.critic_adam.first)&&migrated41.ppo.critic_adam.second==rl::append_zero_input_column(original40.critic_adam.second),"Parameter/moment migration changed");
    auto asymmetric=migrated41;asymmetric.ppo.critic_adam.steps=21780;asymmetric.critic_column_age=80;
    const auto bytes=rl::write_appended_hold(asymmetric,cp::checksum(rest_protocol()));const auto round=rl::read_appended_hold(bytes,cp::checksum(rest_protocol()));
    require(round.ppo.actor_adam.steps==21700&&round.ppo.critic_adam.steps==21780&&round.actor_column_age==0&&round.critic_column_age==80,"Independent age roundtrip failed");
    bool rejected=false;try{(void)rl::read_ppo_training_state(bytes,cp::checksum(rest_protocol()));}catch(const std::exception&){rejected=true;}require(rejected,"Legacy reader accepted new schema");
#ifdef CUDARL_REST_HOLD_RESET_STUDY
    // Synthetic boundary: no simulation or optimizer calls. Make carryover
    // conspicuous, then prove only hold lanes reset and clear AR inputs.
    auto old_worlds=worlds;auto old_tasks=tasks;
    for(int i=0;i<32;++i){
        old_worlds[i].physical.control_steps=20;
        old_worlds[i].physical.substeps=200;
        old_worlds[i].attempted_substeps=200;
        old_worlds[i].physical.body.position.x+=.25f;
        old_worlds[i].physical.body.velocity[0]=.1f;
        old_tasks[i].controls=20;
    }
    data.worlds.copy_from_host(old_worlds.data(),32);data.tasks.copy_from_host(old_tasks.data(),32);
    const std::vector<float> prior_obs(32*41,.3f),prior_action(32*4,.2f);
    data.previous_obs.copy_from_host(prior_obs.data(),32*41);data.previous_action.copy_from_host(prior_action.data(),32*4);
    gpu::rest_hold_reset_boundary<<<1,32,0,stream.value>>>(data.worlds.get(),data.tasks.get(),data.previous_obs.get(),
        data.previous_action.get(),data.history.get(),data.pre_action_speed.get(),data.fault.get(),20,training_episode);
    const auto reset_worlds=download(data.worlds.get(),32,stream.value);
    const auto reset_tasks=download(data.tasks.get(),32,stream.value);
    const auto ar=download(data.history.get(),32,stream.value),speed=download(data.pre_action_speed.get(),32,stream.value);
    const auto px=download(data.previous_obs.get(),32*41,stream.value),pz=download(data.previous_action.get(),32*4,stream.value);
    require(download(data.fault.get(),1,stream.value)[0]==0,"Synthetic boundary GPU fault");
    for(int i=0;i<32;++i){
        const bool held=(i/2)%2;
        const auto& expected=held?worlds[i]:old_worlds[i];
        require(fields(reset_worlds[i])==fields(expected)&&
                reset_worlds[i].attempted_substeps==expected.attempted_substeps&&
                reset_tasks[i].controls==(held?0:20)&&ar[i]==(held?0.f:1.f)&&
                (held?speed[i]==0:speed[i]>0),"Selective boundary failed");
        for(int j=0;j<41;++j)require(px[i*41+j]==(held?0.f:.3f),"Synthetic AR observation did not isolate reset");
        for(int j=0;j<4;++j)require(pz[i*4+j]==(held?0.f:.2f),"Synthetic AR action did not isolate reset");
    }
    std::ofstream reset_check(root/"zero-step-reset-boundaries.json");
    reset_check<<"{\"passed\":true,\"hold_reset\":16,\"swim_preserved\":16,\"AR_history_isolated\":true,\"physical_controls\":0,\"optimizer_steps\":0}\n";
#endif
    std::ofstream out(root/"zero-step-checks.json");out<<std::setprecision(12)<<"{\"passed\":true,\"CPU_CUDA_obs_error\":"<<obs_error<<",\"old40_new41_output_error\":"<<function_error<<",\"worlds\":32,\"goal_pose_each\":8,\"asymmetric_clock_fixture\":true,\"physical_controls\":0,\"optimizer_steps\":0}\n";
}
int prepare_rest(const fs::path& root){require(!fs::exists(root),"Rest study identity exists");fs::create_directories(root);configure_rest(root,false);
    fs::copy_file(parent_checkpoint_path,root/"parent-checkpoint.bin");fs::copy_file(parent_protocol_path,root/"parent-protocol.bin");
    cp::write_atomic((root/"protocol.bin").string(),rest_protocol());cp::write_atomic((root/"migrated-initial.bin").string(),rl::write_appended_hold(migrated41,cp::checksum(rest_protocol())));
    rest_zero_checks(root);std::ofstream out(root/"readiness.json");out<<"{\"ready\":true,\"rest_from_control0\":true,\"controls_max\":113664,\"substeps_max\":1136640,\"PPO_updates_max\":8,\"critic_only_steps\":80,\"actor_steps_max\":640,\"critic_steps_max\":720,\"process_seconds\":120,\"report_pause_before_PPO\":true,\"physical_controls\":0,\"optimizer_steps\":0,\"retries\":0}\n";
    std::cout<<"REST_PREPARED zero physical/optimizer steps"<<std::endl;return 0;}
struct RestAssessment{int passed=0,completed=0;};
RestAssessment evaluate_rest(const std::vector<float>& params,const fs::path& root,Budget& budget){
    require(fs::create_directory(root),"Rest assessment exists");nn::CpuMlp<float> actor(spec(4));actor.params()=params;RestAssessment result;
    std::ofstream trace(root/"controls.csv"),summary(root/"summary.csv");trace<<std::setprecision(12);summary<<std::setprecision(12);
    trace<<"case,pose,offset_mm,control,time_s,speed_mm_s,forward_velocity_mm_s,forward_mm,depth_mm,lane_mm,tilt_deg,roll_deg,pitch_deg,anchor_error_mm";
    for(int j=0;j<4;++j)trace<<",command"<<j<<",angle"<<j<<",rate"<<j<<",last_torque"<<j;trace<<'\n';
    summary<<"case,pose,offset_mm,passed,max_speed_mm_s,RMS_speed_mm_s,max_anchor_error_mm,max_depth_mm,max_lane_mm,max_tilt_deg,final8_forward_excursion_mm,net_forward_mm,final_speed_mm_s,final_depth_mm,positive_work_J\n";
    for(int id=0;id<6;++id){auto w=m::initial_world(id%2?3:0);w.arena=m::long_swim_arena();m::ForwardTask t;const auto c=rest_config();const float offset=id/2==0?0.f:id/2==1?-.0005f:.0005f;
        require(m::begin_forward(w,c,t)&&m::offset_episode_start(w,t,c,offset)&&m::update_velocity_request(w,c,t),"Rest assessment setup failed");
        const auto anchor=t.hold_anchor_model,start=t.start_com_model,heading=t.request.heading;
        m::RestHoldAssessment score;
        require(score.record(0,m::measure(t,w)),"Invalid rest assessment reset sample");
        for(int k=1;k<=m::RestHoldAssessment::controls;++k){require(m::update_velocity_request(w,c,t)&&p::norm(t.hold_anchor_model-anchor)==0&&p::norm(t.start_com_model-start)==0&&p::norm(t.request.heading-heading)==0,"Rest reference moved");
            float x[41];require(m::observe_appended(t,w,c,x),"Rest assessment observation failed");actor.forward(x,1);m::Action action;require(m::action_from_latent(actor.output(),action),"Rest assessment action failed");
            p::BodyState before{},candidate{};try{budget.step(w,action.command,nullptr,&before,&candidate);}catch(...){cp::Writer proof;proof.value(uint32_t{1});proof.value(uint32_t{sizeof(m::World)});proof.value(w);proof.value(before);proof.value(candidate);cp::write_atomic((root/"failure-state.bin").string(),proof.data);throw;}
            const auto interval=m::record_hold_control(t,w,c);require(interval.accepted,"Rest assessment record rejected");const auto& z=interval.metrics;require(score.record(k,z),"Invalid rest assessment control sample");
            trace<<id<<','<<(id%2?3:0)<<','<<offset*1000<<','<<k<<','<<k*.05<<','<<p::norm(z.velocity_m_s)*1000<<','<<p::dot(z.velocity_m_s,heading)*1000<<','<<z.forward_m*1000<<','<<z.signed_depth_m*1000<<','<<z.lateral_m*1000<<','<<z.tilt_deg<<','<<z.roll_deg<<','<<z.pitch_deg<<','<<p::norm(z.hold_error_m)*1000;
            for(int j=0;j<4;++j)trace<<','<<action.command[j]<<','<<w.physical.body.angle[j]<<','<<w.physical.body.velocity[6+j]<<','<<w.physical.last_actual_torques[j];trace<<'\n';}
        require(score.complete(),"Incomplete rest assessment trajectory");result.passed+=score.passed();++result.completed;const auto last=m::measure(t,w);
        summary<<id<<','<<(id%2?3:0)<<','<<offset*1000<<','<<score.passed()<<','<<score.peak_speed_m_s()*1000<<','<<score.rms_speed_m_s()*1000<<','<<score.peak_anchor_m()*1000<<','<<score.peak_depth_m()*1000<<','<<score.peak_lane_m()*1000<<','<<score.peak_tilt_degrees()<<','<<score.final_excursion_m()*1000<<','<<last.forward_m*1000<<','<<p::norm(last.velocity_m_s)*1000<<','<<last.signed_depth_m*1000<<','<<w.physical.body.work<<'\n';summary.flush();trace.flush();}
    return result;
}
void rest_account(Arm& a){account_rollout(a);require(a.budget.controls<=113664&&a.budget.substeps<=1136640,"Rest control cap exceeded");}
void compute_rest_returns(Rollout& d,cudaStream_t s){rl::compute_gae(horizon,32,d.reward.get(),d.value.get(),d.done.get(),d.zeros.get(),d.zeros.get(),d.last.get(),.99f,1.f,d.advantage.get(),d.returns.get(),s);rl::mean_std(d.advantage.get(),rows,d.adv_stats.get(),s);}
void verify_rest_batch(Arm& a,const char* label){auto& d=*a.data;const auto s=a.stream->value;compute_rest_returns(d,s);
    const auto reward=download(d.reward.get(),rows,s),value=download(d.value.get(),rows,s),done=download(d.done.get(),rows,s),cost=download(d.cost_trace.get(),rows*5,s),obs=download(d.obs.get(),rows*41,s),history=download(d.history.get(),rows,s);
    verified_adv=download(d.advantage.get(),rows,s);verified_returns=download(d.returns.get(),rows,s);const auto tasks=download(d.tasks.get(),32,s);const auto worlds=download(d.worlds.get(),32,s);double max_error=0;
#ifdef CUDARL_REST_HOLD_RESET_STUDY
    const auto pre_speed=download(d.pre_action_speed.get(),rows,s),prev=download(d.previous_obs.get(),rows*41,s),
        prev_action=download(d.previous_action.get(),rows*4,s);
    const auto counts=download(d.counts.get(),32,s);
    std::ofstream pre_trace(a.root/(std::string(label)+"-speed-and-boundaries.csv"));
    pre_trace<<std::setprecision(12)<<"slot,world,holding,pre_action_speed_m_s,done,has_AR_history\n";
    int n_quiet=0,n_hold=0,n_reset=0;
#endif
    for(int e=0;e<32;++e){const bool held=(e/2)%2;
#ifdef CUDARL_REST_HOLD_RESET_STUDY
        const int length=held?gpu::rest_hold_reset_controls:320;
        require(counts[e].controls==a.training_controls/32&&counts[e].substeps==a.training_substeps/32,
                "Actual per-world control accounting mismatch");
#else
        const int length=320;
#endif
        require(tasks[e].controls==length&&worlds[e].physical.control_steps==uint64_t(length)&&worlds[e].physical.substeps==uint64_t(length*10),"Actual rest episode length mismatch");
        require(tasks[e].hold_anchor_set==held&&p::norm(tasks[e].request.terminal_velocity-tasks[e].request.heading*(held?0.f:.008f))<1e-8f,"Actual rest goal/anchor mismatch");double target=0;
        for(int t=319;t>=0;--t){const int row=t*32+e;const int episode_t=t%length;
            require(done[row]==(episode_t==length-1?1.f:0.f)&&history[row]==(episode_t?1.f:0.f),"Wrong terminal or AR reset");
            if(episode_t==length-1)target=0;
            target=reward[row]+double(.99f)*target;
#ifdef CUDARL_REST_HOLD_RESET_STUDY
            require(std::isfinite(pre_speed[row]),"Nonfinite pre-action speed");
            pre_trace<<t<<','<<e<<','<<held<<','<<pre_speed[row]<<','<<done[row]<<','<<history[row]<<'\n';
            if(held){++n_hold;n_quiet+=pre_speed[row]<=.0005f;n_reset+=episode_t==0;}
            if(episode_t==0){require(pre_speed[row]==0,"Reset speed is not zero");
                for(int j=0;j<41;++j)require(prev[row*41+j]==0,"Previous observation leaked across reset");
                for(int j=0;j<4;++j)require(prev_action[row*4+j]==0,"Previous action leaked across reset");}
#endif
            const double error=fabs(verified_returns[row]-target);max_error=std::max(max_error,error);require(error<=.002+.0002*fabs(target)&&fabs(verified_adv[row]+value[row]-target)<=.002+.0002*fabs(target),"Rest GAE mismatch");
            float sum=0;for(int j=0;j<5;++j)sum+=cost[row*5+j];require(fabs(reward[row]+p::dt*sum)<=1e-6+1e-5*fabs(reward[row]),"Rest reward components mismatch");
            const auto cfg=training_episode.task_config(t,e);require(cfg.target_speed_m_s==(held?0.f:.008f)&&training_episode.reward_gain(t,e)==(held?1.f/1024:1.f),"Rest gain/time request mismatch");
            const float* x=obs.data()+row*41;require(x[34]==p::clamp(1.f-float(episode_t)/40,0,1),"Elapsed-time meaning changed");if(held)for(int j=3;j<6;++j)require(x[j]==0,"Zero goal missing from actual observation");
            if(episode_t==0){require(x[40]==0&&x[34]==1,"Bad actual rest initial input");for(int j=12;j<18;++j)require(x[j]==0,"Nonzero starting velocity");}}
    }
    const auto rescore=score_batch(*a.actor,d,*a.batch,s),old=download(d.logp.get(),rows,s);double max_logp=0;for(int i=0;i<rows;++i)max_logp=std::max(max_logp,fabs(double(rescore[i])-old[i]));require(max_logp<1e-4,"Actual rest old-policy rescore failed");
#ifdef CUDARL_REST_HOLD_RESET_STUDY
    require(n_hold==5120&&n_reset==256&&bool(pre_trace),"Unexpected balanced hold samples or reset count");
    const double fraction=double(n_quiet)/n_hold;
    reset_occupancy_passed=fraction>=.1;
    std::ofstream cutoff(a.root/(std::string(label)+"-occupancy.json"));
    cutoff<<std::setprecision(12)<<"{\"passed\":"<<(reset_occupancy_passed?"true":"false")
           <<",\"holding_samples\":"<<n_hold<<",\"samples_at_or_below_0p5_mm_s\":"<<n_quiet
           <<",\"hold_fraction\":"<<fraction<<",\"minimum\":0.1,\"hold_resets\":"<<n_reset
           <<",\"hold_episode_controls\":20,\"swim_episode_controls\":320}\n";
#endif
    std::ofstream out(a.root/(std::string(label)+"-integrity.json"));out<<std::setprecision(12)<<"{\"passed\":true,\"rows\":10240,\"holding_rows\":5120,\"max_GAE_error\":"<<max_error<<",\"max_old_logp_error\":"<<max_logp
#ifdef CUDARL_REST_HOLD_RESET_STUDY
        <<",\"per_world_episode_boundaries\":true,\"hold_episode_controls\":20,\"swim_episode_controls\":320,\"zero_goal_gain_from_control0\":true}\n";
#else
        <<",\"terminal_only_at320\":true,\"zero_goal_gain_from_control0\":true}\n";
#endif
}
void rest_verify_minibatch(int update,int epoch,int first,Rollout&,Batches& batch,const rl::Permutation& permutation,cudaStream_t stream){
    const auto adv=download(batch.advantage.get(),256,stream),ret=download(batch.returns.get(),256,stream),grad=download(batch.mean_gradient.get(),512*4,stream);int held=0,nonzero=0;
    for(int i=0;i<256;++i){const int row=int(rl::permute(permutation,first+i));require(adv[i]==verified_adv[row]&&ret[i]==verified_returns[row],"Actual rest minibatch dropped/replaced rows");
        if((row%32/2)%2){++held;float magnitude=0;for(int j=0;j<4;++j)magnitude+=fabsf(grad[i*4+j])+fabsf(grad[(256+i)*4+j]);nonzero+=magnitude>1e-12f;}}
    if(first==0){require(held>0&&nonzero>0,"No actual holding policy gradient");std::ofstream out(active_root/"holding-minibatch-proof.jsonl",std::ios::app);out<<"{\"update\":"<<update<<",\"epoch\":"<<epoch<<",\"held_rows\":"<<held<<",\"nonzero_held_gradient_rows\":"<<nonzero<<"}\n";}
}
struct ScaleStats{double mean=0,stddev=0,rms=0,p99=0,maximum=0;int n=0;bool pass=false;};
ScaleStats scale_stats(const std::vector<float>& returns,bool hold){std::vector<float> absval,selected;double sum2=0;
    for(int row=0;row<rows;++row)if(bool((row%32/2)%2)==hold){const float x=returns[row];require(std::isfinite(x),"Nonfinite return before fitting");selected.push_back(x);absval.push_back(fabsf(x));sum2+=double(x)*x;}
    std::sort(absval.begin(),absval.end());const auto m=rl::diagnostic_moments(selected);ScaleStats s;s.n=int(selected.size());s.mean=m.mean;s.stddev=m.stddev();s.rms=sqrt(sum2/s.n);s.p99=absval[size_t(std::ceil(.99*s.n))-1];s.maximum=absval.back();s.pass=s.p99<=5&&s.maximum<=20;return s;}
void print_scale(std::ostream& out,const ScaleStats& s){out<<"{\"rows\":"<<s.n<<",\"return_mean\":"<<s.mean<<",\"return_std\":"<<s.stddev<<",\"return_RMS\":"<<s.rms<<",\"abs_return_p99\":"<<s.p99<<",\"abs_return_max\":"<<s.maximum<<",\"passed\":"<<(s.pass?"true":"false")<<'}';}
struct ValueStats{double rmse=0,bias=0,rrms=0,ev=0;int n=0;};
ValueStats validation_stats(const std::vector<float>& pred,const std::vector<float>& target,bool hold){ValueStats v;double error2=0,sum=0,rr=0;std::vector<float> p,r;
    for(int row=0;row<rows;++row){const int world=row%32;if(world%8<4||bool((world/2)%2)!=hold)continue;const double e=double(pred[row])-target[row];require(std::isfinite(e),"Nonfinite held-out value");error2+=e*e;sum+=e;rr+=double(target[row])*target[row];p.push_back(pred[row]);r.push_back(target[row]);++v.n;}
    require(v.n==2560,"Validation split/mix wrong");v.rmse=sqrt(error2/v.n);v.bias=sum/v.n;v.rrms=sqrt(rr/v.n);v.ev=rl::diagnostic_value(p,r).explained_variance;return v;}
void print_value(std::ostream& out,const ValueStats& v){out<<"{\"rows\":"<<v.n<<",\"RMSE\":"<<v.rmse<<",\"bias\":"<<v.bias<<",\"return_RMS\":"<<v.rrms<<",\"EV\":"<<v.ev<<'}';}
__global__ void value_gather(const float* input,const float* returns,const int* index,float* obs,float* target){const int i=threadIdx.x;if(i>=256)return;const int row=index[i];for(int j=0;j<41;++j)obs[i*41+j]=input[row*41+j];target[i]=returns[row];}
__global__ void value_gradient(const float* pred,const float* target,float* gradient){const int i=threadIdx.x;if(i<256)gradient[i]=.5f*(pred[i]-target[i])/256.f;}
[[maybe_unused]] void fit_critic(Arm& a){auto& d=*a.data;auto s=a.stream->value;const auto frozen_actor=a.actor->get_params();cp::Writer frozen_adam;a.actor_opt->save(frozen_adam);const int actor_age=a.actor_opt->steps();
    std::vector<int> fitting;for(int row=0;row<rows;++row)if(row%32%8<4)fitting.push_back(row);require(fitting.size()==5120,"Critic fit split wrong");
    std::ofstream split(a.root/"calibration-world-split.csv");split<<"world,pose,hold,fit\n";for(int i=0;i<32;++i)split<<i<<','<<(i%2?3:0)<<','<<(i/2)%2<<','<<(i%8<4)<<'\n';split.close();
    device_ptr<int> indices(256);device_ptr<float> target(256);std::ofstream log(a.root/"critic-calibration.csv");log<<"epoch,minibatch,old_age,new_age\n";
    std::cout<<"CRITIC_CALIBRATION_STARTED unix="<<std::setprecision(17)<<unix_now()<<" exactly80critic_steps; actor_frozen"<<std::endl;
    for(int epoch=0;epoch<4;++epoch){const auto permutation=rl::make_permutation(5120,3141592653u+uint32_t(epoch));std::vector<bool> seen(5120,false);
        for(int first=0;first<5120;first+=256){wall_guard();std::vector<int> idx(256);for(int i=0;i<256;++i){const int fit=int(rl::permute(permutation,first+i));require(!seen[fit],"Critic calibration duplicate row");seen[fit]=true;idx[i]=fitting[fit];require(idx[i]%32%8<4,"Validation row leaked into critic fitting");}
            indices.copy_from_host(idx.data(),256);value_gather<<<1,256,0,s>>>(d.obs.get(),d.returns.get(),indices.get(),a.batch->obs.get(),target.get());a.critic->forward(a.batch->obs.get(),256,s);
            value_gradient<<<1,256,0,s>>>(a.critic->output(),target.get(),a.batch->value_gradient.get());CUDA_CHECK_THROW(cudaGetLastError());
            if(epoch==0&&first==0){const auto pred=download(a.critic->output(),256,s),t=download(target.get(),256,s),g=download(a.batch->value_gradient.get(),256,s);for(int i=0;i<256;++i)require(fabsf(g[i]-.5f*(pred[i]-t[i])/256)<1e-7f,"Critic value-gradient coefficient mismatch");}
            a.critic->zero_grad(s);a.critic->backward(a.batch->value_gradient.get(),256,nullptr,s);a.critic_opt->step(a.critic->params(),a.critic->grads(),s);
            gpu::finite_parameters<<<(2433+255)/256,256,0,s>>>(a.critic->params(),2433,d.fault.get());require(download(d.fault.get(),1,s)[0]==0,"Nonfinite critic calibration");
            log<<epoch<<','<<first/256<<','<<a.critic_opt->steps()<<','<<a.critic_opt->fresh_steps()<<'\n';log.flush();}
        for(bool used:seen)require(used,"Critic calibration missed fit row");}
    cp::Writer after_adam;a.actor_opt->save(after_adam);require(after_adam.data==frozen_adam.data&&a.actor->get_params()==frozen_actor&&a.actor_opt->steps()==actor_age&&a.actor_opt->fresh_steps()==0,"Critic-only fitting changed actor/optimizer");
    require(a.critic_opt->steps()==21780&&a.critic_opt->fresh_steps()==80,"Critic calibration clock mismatch");save_rest(a.root,"critic-calibrated",336,*a.actor,*a.critic,*a.actor_opt,*a.critic_opt);
}
void actual_gpu_migration(Arm& a){check_initial_outputs(a);const auto x=download(a.data->obs.get(),rows*41,a.stream->value);std::vector<float> oldx(size_t(rows)*40);
    for(int row=0;row<rows;++row)std::copy_n(x.data()+row*41,40,oldx.data()+row*40);device_ptr<float> input(oldx.size());input.copy_from_host(oldx.data(),oldx.size());
    nn::GpuMlp olda(old_spec(4),rows,seed),oldc(old_spec(1),rows,seed);olda.set_params(original40.actor.parameters);oldc.set_params(original40.critic);
    olda.forward(input.get(),rows,a.stream->value);oldc.forward(input.get(),rows,a.stream->value);a.actor->forward(a.data->obs.get(),rows,a.stream->value);a.critic->forward(a.data->obs.get(),rows,a.stream->value);
    const auto oa=download(olda.output(),rows*4,a.stream->value),na=download(a.actor->output(),rows*4,a.stream->value),oc=download(oldc.output(),rows,a.stream->value),nc=download(a.critic->output(),rows,a.stream->value);float error=0;
    for(size_t i=0;i<oa.size();++i)error=std::max(error,fabsf(oa[i]-na[i]));for(size_t i=0;i<oc.size();++i)error=std::max(error,fabsf(oc[i]-nc[i]));require(error<=1e-6f,"Actual GPU old40/new41 mismatch before fitting");
    std::ofstream out(a.root/"actual-GPU-migration.json");out<<"{\"passed\":true,\"rows\":10240,\"max_old40_new41_error\":"<<error<<",\"extra_physical_controls\":0}\n";
}
void save_value_rows(Arm& a,const std::vector<float>& before,const std::vector<float>& after){const auto returns=download(a.data->returns.get(),rows,a.stream->value);std::ofstream out(a.root/"calibration-value-rows.csv");out<<std::setprecision(12)<<"row,world,pose,hold,fit,return,value_before,value_after\n";
    for(int row=0;row<rows;++row){const int w=row%32;out<<row<<','<<w<<','<<(w%2?3:0)<<','<<(w/2)%2<<','<<(w%8<4)<<','<<returns[row]<<','<<before[row]<<','<<after[row]<<'\n';}}
#ifdef CUDARL_REST_HOLD_TASK_CRITIC
#include "rest_hold_task_critic.inc"
#endif
[[maybe_unused]] bool first_signal_gate(Arm& a){const auto adv=download(a.data->advantage.get(),rows,a.stream->value),stats=download(a.data->adv_stats.get(),2,a.stream->value);double sum=0,held=0;
    for(int row=0;row<rows;++row){const double z=(adv[row]-stats[0])/(stats[1]+1e-8f);sum+=z*z;if((row%32/2)%2)held+=z*z;}
    const double fraction=sum>0?held/sum:0;const bool pass=std::isfinite(fraction)&&sum>0&&fraction>=.1&&fraction<=.9;
    std::ofstream out(a.root/"first-PPO-signal-gate.json");out<<std::setprecision(12)<<"{\"passed\":"<<(pass?"true":"false")<<",\"holding_fraction_squared_normalized_advantage\":"<<fraction<<",\"minimum\":0.1,\"maximum\":0.9,\"actor_updates_before_gate\":0}\n";return pass;
}
#ifdef CUDARL_REST_HOLD_MEASURED_BASELINE
#include "rest_hold_measured_baseline.inc"
#endif
[[maybe_unused]] int run_rest(const fs::path& root){configure_rest(root,true);require(cp::read((root/"protocol.bin").string())==rest_protocol(),"Rest protocol changed");require(fs::create_directory(root/"run-started"),"Rest trial already attempted; no retry");
    started=std::chrono::steady_clock::now();Arm a;a.root=root;a.budget.seconds=120;active_root=root;
#ifdef CUDARL_REST_HOLD_RESET_STUDY
    a.budget.maximum_controls=21024;a.budget.maximum_substeps=210240;
#else
    a.budget.maximum_controls=113664;a.budget.maximum_substeps=1136640;
#endif
    std::string fault,gate;int critic_only=0,ppo=0;bool candidate=false,baseline_qualified=false;RestAssessment baseline,final;ScaleStats hs,ss;ValueStats hb,ha,sb,sa;const double launch=unix_now();
#ifdef CUDARL_REST_HOLD_TASK_CRITIC
    TaskCriticFit task_fit;ValueStats inherited_hold,inherited_swim,hold_mean_baseline;
    bool old_numeric_pass=false,beats_fit_hold_mean=false;
#endif
    try{
        a.stream=std::make_unique<Stream>();
#ifdef CUDARL_REST_HOLD_RESET_STUDY
        parity(a.budget,a.stream->value);
#else
        Budget axes;axes.seconds=120;parity(axes,a.stream->value);a.budget.controls+=axes.controls;a.budget.substeps+=axes.substeps;
#endif
        status(root,"baseline",a.budget,336);
        a.before=evaluate(initial_actor(),root,"baseline",a.budget);baseline=evaluate_rest(initial_actor(),root/"baseline-rest",a.budget);fs::create_directory(root/"baseline-long");a.long_before=sustained(initial_actor(),root/"baseline-long",a.budget);require(a.long_before.safety_fault.empty(),"Baseline swim physical fault");
#ifdef CUDARL_REST_HOLD_RESET_STUDY
        require(baseline.completed==6&&baseline.passed==0&&a.before.passed==4&&
                preserved_long_windows(root/"baseline-long"),"Frozen baseline differs from reference");
#endif
        baseline_qualified=baseline.passed==6&&a.before.passed==4&&preserved_long_windows(root/"baseline-long");candidate=baseline_qualified;
        std::cout<<"BASELINE_COMPLETED short="<<a.before.passed<<"/4 rest="<<baseline.passed<<"/6 long_windows="<<a.long_before.windows_passed<<"/30 controls="<<a.budget.controls<<std::endl;
        if(!baseline_qualified){
            a.actor=std::make_unique<nn::GpuMlp>(spec(4),rows,seed);a.critic=std::make_unique<nn::GpuMlp>(spec(1),rows,seed+1);
            a.actor_opt=std::make_unique<nn::GpuAdam>(2532,initial_optimizer_config(true));a.critic_opt=std::make_unique<nn::GpuAdam>(2433,initial_optimizer_config(false));restore_training(*resume_state,*a.actor,*a.critic,*a.actor_opt,*a.critic_opt);
            a.actor_opt->set_fresh_column(41,32,0);a.critic_opt->set_fresh_column(41,32,0);save_rest(root,"initial-training-state",336,*a.actor,*a.critic,*a.actor_opt,*a.critic_opt);
            a.data=std::make_unique<Rollout>(a.stream->value);a.batch=std::make_unique<Batches>();fs::create_directory(root/"calibration");calibration_collect=true;
            status(root,"calibration_collect",a.budget,336);std::cout<<"CALIBRATION_ROLLOUT_STARTED unix="<<std::setprecision(17)<<unix_now()<<" seed2718281828; actor_frozen"<<std::endl;
            collect_rollout(336,*a.actor,*a.critic,*a.data,a.stream->value,root/"calibration");rest_account(a);require(a.training_controls==10240,"Calibration physical count mismatch");
            const auto mainroot=a.root;a.root=root/"calibration";cache_batch(a,336);actual_gpu_migration(a);a.root=mainroot;verify_rest_batch(a,"calibration");
            const auto returns=download(a.data->returns.get(),rows,a.stream->value),before=download(a.data->value.get(),rows,a.stream->value);hs=scale_stats(returns,true);ss=scale_stats(returns,false);hb=validation_stats(before,returns,true);sb=validation_stats(before,returns,false);
            const bool scale_pass=hs.pass&&ss.pass;bool validation_pass=false;
            if(!scale_pass)gate="return_scale";
#ifdef CUDARL_REST_HOLD_RESET_STUDY
            else if(!reset_occupancy_passed)gate="calibration_hold_occupancy";
#endif
            else{status(root,"critic_calibration",a.budget,336);
#ifdef CUDARL_REST_HOLD_TASK_CRITIC
                inherited_hold=hb;inherited_swim=sb;
                fit_task_critic(a,returns,before,task_fit);critic_only=task_fit.steps;
                const auto& after=task_fit.final;
                hb=validation_stats(task_fit.initial,returns,true);sb=validation_stats(task_fit.initial,returns,false);
                const std::vector<float> constant(rows,task_fit.scale.mean[1]);
                hold_mean_baseline=validation_stats(constant,returns,true);
                require(returns==download(a.data->returns.get(),rows,a.stream->value),"Normalization changed raw GAE/returns");
                save_value_rows(a,task_fit.initial,after);
#else
                fit_critic(a);critic_only=80;a.critic->forward(a.data->obs.get(),rows,a.stream->value);const auto after=download(a.critic->output(),rows,a.stream->value);save_value_rows(a,before,after);
#endif
                ha=validation_stats(after,returns,true);sa=validation_stats(after,returns,false);
                validation_pass=ha.rmse<=std::max(.05,.75*hb.rmse)&&ha.rmse<=std::max(.05,.5*ha.rrms)&&fabs(ha.bias)<=std::max(.02,.25*ha.rrms)&&sa.rmse<=std::max(.10,1.1*sb.rmse);
#ifdef CUDARL_REST_HOLD_TASK_CRITIC
                old_numeric_pass=validation_pass;
                beats_fit_hold_mean=ha.rmse<hold_mean_baseline.rmse;
                validation_pass=old_numeric_pass&&beats_fit_hold_mean;
                if(validation_pass)save_task_critic(root,task_fit);
                if(!validation_pass)gate=old_numeric_pass?"hold_mean_comparator":"critic_validation";
#else
                if(!validation_pass)gate="critic_validation";
#endif
            }
            {std::ofstream report(root/"pre-PPO-report.json");report<<std::setprecision(12)<<"{\"return_scale_passed\":"<<(scale_pass?"true":"false")<<",\"holding_returns\":";print_scale(report,hs);report<<",\"swimming_returns\":";print_scale(report,ss);
                report<<",\"critic_only_steps\":"<<critic_only<<",\"hold_validation_before\":";print_value(report,hb);report<<",\"swim_validation_before\":";print_value(report,sb);
                report<<",\"hold_validation_after\":";if(critic_only)print_value(report,ha);else report<<"null";report<<",\"swim_validation_after\":";if(critic_only)print_value(report,sa);else report<<"null";
                report<<",\"hold_RMSE_improvement_limit\":"<<std::max(.05,.75*hb.rmse)<<",\"hold_RMSE_scale_limit\":"<<std::max(.05,.5*hb.rrms)<<",\"hold_bias_limit\":"<<std::max(.02,.25*hb.rrms)<<",\"swim_RMSE_limit\":"<<std::max(.10,1.1*sb.rmse)
                    <<",\"critic_validation_passed\":"<<(validation_pass?"true":"false")<<",\"ready_for_PPO\":"<<(scale_pass&&validation_pass?"true":"false")<<",\"gate_failed\":"<<std::quoted(gate)<<",\"actor_updates\":0,\"controls\":"<<a.budget.controls<<",\"substeps\":"<<a.budget.substeps<<"}\n";}
#ifdef CUDARL_REST_HOLD_TASK_CRITIC
            if(critic_only){
                std::ofstream detail(root/"task-critic-validation.json");detail<<std::setprecision(12)
                    <<"{\"fit_only_hold_RMS\":"<<task_fit.scale.rms[1]<<",\"fit_only_swim_RMS\":"<<task_fit.scale.rms[0]
                    <<",\"fit_only_hold_mean\":"<<task_fit.scale.mean[1]<<",\"fit_only_swim_mean\":"<<task_fit.scale.mean[0]
                    <<",\"fit_rows_per_mode\":"<<task_fit.scale.fit_n[0]<<",\"inherited_hold\":";
                print_value(detail,inherited_hold);detail<<",\"inherited_swim\":";print_value(detail,inherited_swim);
                detail<<",\"fit_only_hold_mean_held_out\":";print_value(detail,hold_mean_baseline);
                detail<<",\"old_numeric_gates_passed\":"<<(old_numeric_pass?"true":"false")
                    <<",\"strict_hold_mean_comparator_passed\":"<<(beats_fit_hold_mean?"true":"false")
                    <<",\"max_swim_initial_raw_prediction_error\":"<<task_fit.raw_swim_initial_error
                    <<",\"optimizer_steps\":"<<critic_only<<",\"raw_GAE_unchanged\":true,\"actor_unchanged\":true}\n";
            }
#endif
            std::cout<<"PRE_PPO_REPORT_READY gate="<<gate<<" controls="<<a.budget.controls<<" critic_steps="<<critic_only<<" actor_updates=0"<<std::endl;
            if(gate.empty()){
#ifdef CUDARL_REST_HOLD_RESET_STUDY
                require(a.budget.controls==21024&&a.budget.substeps==210240&&critic_only==80&&
                        a.actor_opt->fresh_steps()==0,"Calibration-only budget/optimizer mismatch");
                status(root,"calibrated_no_PPO",a.budget,336);
#else
                status(root,"awaiting_calibration_report",a.budget,336);std::ofstream marker(root/"awaiting-report-release");marker<<"Calibration report must be sent before PPO; original approved gates only\n";marker.close();
                while(!fs::exists(root/"reported-and-continue")){wall_guard();std::this_thread::sleep_for(std::chrono::milliseconds(25));}
                calibration_collect=false;training_started_unix_seconds=unix_now();std::cout<<"PPO_STARTED after_calibration_report unix="<<training_started_unix_seconds<<std::endl;
                for(int update=336;update<344;++update){status(root,"PPO_collect",a.budget,a.completed);collect_rollout(update,*a.actor,*a.critic,*a.data,a.stream->value,root);rest_account(a);cache_batch(a,update);verify_rest_batch(a,("PPO-"+std::to_string(update)).c_str());
                    if(update==336&&!first_signal_gate(a)){gate="first_PPO_signal_allocation";break;}
                    status(root,"PPO_optimize",a.budget,a.completed);const double kl=optimize_rollout(update,*a.actor,*a.critic,*a.actor_opt,*a.critic_opt,*a.data,*a.batch,a.stream->value,root);
                    require(download(a.data->fault.get(),1,a.stream->value)[0]==0,"Nonfinite PPO parameters");a.completed=update+1;++ppo;
                    require(a.actor_opt->steps()==21700+a.actor_opt->fresh_steps()&&a.critic_opt->steps()==21700+a.critic_opt->fresh_steps()&&a.critic_opt->fresh_steps()==80+a.actor_opt->fresh_steps()&&a.actor_opt->fresh_steps()>=ppo*40&&a.actor_opt->fresh_steps()<=ppo*80,"Asymmetric PPO optimizer clocks wrong");
                    save_rest(root,("update-"+std::to_string(a.completed)).c_str(),a.completed,*a.actor,*a.critic,*a.actor_opt,*a.critic_opt);
                    std::ofstream log(root/"PPO-updates.jsonl",std::ios::app);log<<"{\"completed\":"<<a.completed<<",\"actor_old_age\":"<<a.actor_opt->steps()<<",\"critic_old_age\":"<<a.critic_opt->steps()<<",\"actor_new_age\":"<<a.actor_opt->fresh_steps()<<",\"critic_new_age\":"<<a.critic_opt->fresh_steps()<<",\"KL\":"<<kl<<"}\n";
                    std::cout<<"REST_PPO_UPDATE_COMPLETED "<<a.completed<<"/344 controls="<<a.budget.controls<<std::endl;}
                if(gate.empty()){
                    require(ppo==8,"Wrong PPO update count");status(root,"final_assessments",a.budget,a.completed);a.after=evaluate(a.actor->get_params(),root,"final",a.budget);final=evaluate_rest(a.actor->get_params(),root/"final-rest",a.budget);
                    fs::create_directory(root/"final-long");a.long_after=sustained(a.actor->get_params(),root/"final-long",a.budget);require(a.long_after.safety_fault.empty(),"Final swim physical fault");
                    candidate=final.passed==6&&a.after.passed==4&&preserved_long_windows(root/"final-long");require(a.budget.controls==113664&&a.budget.substeps==1136640&&a.training_controls==92160,"Final rest budget accounting failed");}
#endif
            }
        }
        wall_guard();
    }catch(const std::exception& e){fault=e.what();std::cerr<<"REST_STOPPED "<<fault<<std::endl;
        if(a.data){try{cp::Writer proof;proof.value(uint32_t{1});proof.value(uint32_t{sizeof(m::World)});proof.array(download(a.data->worlds.get(),32,a.stream->value));proof.value(uint32_t{sizeof(m::ForwardTask)});proof.array(download(a.data->tasks.get(),32,a.stream->value));proof.value(uint32_t{sizeof(gpu::Counts)});proof.array(download(a.data->counts.get(),32,a.stream->value));proof.array(download(a.data->fault.get(),1,a.stream->value));cp::write_atomic((root/"GPU-failure-state.bin").string(),proof.data);}catch(const std::exception& save_error){std::ofstream recovery(root/"failure-capture-error.txt");recovery<<save_error.what();}}
    }
#ifdef CUDARL_REST_HOLD_TASK_CRITIC
    critic_only=task_fit.steps;
#else
    if(a.critic_opt)critic_only=std::min(80,a.critic_opt->steps()-21700);
#endif
#ifdef CUDARL_REST_HOLD_TASK_CRITIC
    const int total_critic_steps=critic_only;
#else
    const int total_critic_steps=a.critic_opt?a.critic_opt->steps()-21700:0;
#endif
    std::ofstream out(root/"result.json");out<<std::setprecision(17)<<"{\"fault\":"<<std::quoted(fault)<<",\"gate_failed\":"<<std::quoted(gate)<<",\"launched_unix\":"<<launch<<",\"ended_unix\":"<<unix_now()<<",\"native_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()
        <<",\"controls\":"<<a.budget.controls<<",\"substeps\":"<<a.budget.substeps<<",\"PPO_updates\":"<<ppo<<",\"critic_only_steps\":"<<critic_only<<",\"baseline_short\":"<<a.before.passed<<",\"baseline_rest\":"<<baseline.passed<<",\"baseline_rest_completed\":"<<baseline.completed<<",\"baseline_long_windows\":"<<a.long_before.windows_passed
        <<",\"final_short\":"<<a.after.passed<<",\"final_rest\":"<<final.passed<<",\"final_rest_completed\":"<<final.completed<<",\"final_long_windows\":"<<a.long_after.windows_passed<<",\"baseline_already_qualified\":"<<(baseline_qualified?"true":"false")
        <<",\"candidate_accepted\":"<<(candidate&&fault.empty()?"true":"false")<<",\"actor_PPO_steps\":"<<(a.actor_opt?a.actor_opt->steps()-21700:0)<<",\"critic_optimizer_steps\":"<<total_critic_steps<<",\"runtime_adopted\":false,\"retries\":0,\"closed\":true}\n";
    out.close();status(root,fault.empty()?(gate.empty()?"done":"gate_stopped"):"failed",a.budget,a.completed);std::cout<<"REST_TRIAL_CLOSED PPO="<<ppo<<" critic_only="<<critic_only<<" gate="<<gate<<" fault="<<fault<<std::endl;return fault.empty()?0:3;
}
} // namespace
#ifndef CUDARL_REST_HOLD_MEASURED_PHASE_B
int main(int argc,char** argv){try{
#ifdef CUDARL_REST_HOLD_RESET_STUDY
    require(argc==3,"Use --prepare or --calibrate DIR; no PPO entry point");
    const std::string mode=argv[1];const auto root=fs::absolute(argv[2]);
    if(mode=="--prepare")return prepare_rest(root);
#ifdef CUDARL_REST_HOLD_MEASURED_BASELINE
    if(mode=="--calibrate")return run_measured_phase_a(root);
#else
    if(mode=="--calibrate")return run_rest(root);
#endif
    throw std::runtime_error("Unsupported calibration-only mode; PPO is not in this executable");
#else
    require(argc==3,"Use --prepare/--run DIR");const std::string mode=argv[1];const auto root=fs::absolute(argv[2]);if(mode=="--prepare")return prepare_rest(root);if(mode=="--run")return run_rest(root);throw std::runtime_error("Unknown rest mode");
#endif
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}}
#endif
