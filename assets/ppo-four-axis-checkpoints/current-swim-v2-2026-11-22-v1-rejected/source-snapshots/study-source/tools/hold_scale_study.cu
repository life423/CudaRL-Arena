// Approved paired reward-scale experiment. No simulator or PPO implementation fork.
#define CUDARL_BALANCED_HOLD_STUDY 1
#include "append_hold8.cu"
#include <memory>
namespace {
constexpr int arm_controls = 349184;
double unix_now() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }
void configure_study(const fs::path& root, int arm, bool copied) {
    configure_append(root,copied);
    training_episode.balanced_hold=true;training_episode.holding_reward_gain=arm?1.f/1024:1.f;
    control_cap=arm_controls;substep_cap=10*arm_controls;active_wall_cap=180;
    require(training_episode.valid()&&seed==1247037243&&correlation==.8f,"Study initial configuration mismatch");
}
__global__ void config_probe(m::TrainingEpisode episode,float* result) {
    const int i=threadIdx.x;const int times[4]={0,639,640,1279};
    for(int k=0;k<4;++k){const auto c=episode.task_config(times[k],i);
        result[i*8+k*2]=c.target_speed_m_s;result[i*8+k*2+1]=episode.reward_gain(times[k],i);}
}
void configuration_checks(const fs::path& root) {
    int counts[2][2][3]{};
    Stream stream;device_ptr<float> probe(256);
    config_probe<<<1,32,0,stream.value>>>(training_episode,probe.get());CUDA_CHECK_THROW(cudaGetLastError());
    const auto actual=download(probe.get(),256,stream.value);const int times[4]={0,639,640,1279};
    for(int u=336;u<344;++u){
        training_episode.offset_phase=u%3;int groups[2][2]{};
        for(int i=0;i<32;++i){const int pose=i%2,goal=(i/2)%2;const int depth=(i/2+u%3)%3;
            ++groups[pose][goal];++counts[pose][goal][depth];
            require(training_episode.hold_episode(i)==bool(goal),"Pose/goal confounding retained");
            for(int k=0;k<4;++k){
                require(actual[i*8+k*2]==training_episode.task_config(times[k],i).target_speed_m_s &&
                        actual[i*8+k*2+1]==training_episode.reward_gain(times[k],i),"CPU/CUDA goal/gain mismatch");
                const float gain=goal&&times[k]>=640?training_episode.holding_reward_gain:1.f;
                require(actual[i*8+k*2+1]==gain,"Reward gain switches at wrong boundary");
            }
        }
        for(auto& poses:groups)for(int n:poses)require(n==8,"Unbalanced pose/goal batch");
    }
    std::ofstream out(root/"configuration-checks.json");out<<"{\"passed\":true,\"CPU_CUDA_goal_gain_cases\":128,\"physical_controls\":0,\"PPO_updates\":0,\"depth_counts\":[";
    bool comma=false;for(int pose=0;pose<2;++pose)for(int goal=0;goal<2;++goal)for(int d=0;d<3;++d){
        require(counts[pose][goal][d]>=21&&counts[pose][goal][d]<=22,"Depth mixture imbalance");
        if(comma)out<<',';comma=true;out<<counts[pose][goal][d];}
    out<<"]}\n";
}
void prepare_study(const fs::path& study) {
    require(!fs::exists(study),"Study identity exists; never overwrite");fs::create_directories(study);
    for(int arm=0;arm<2;++arm){const auto root=study/(arm?"B":"A");fs::create_directory(root);configure_study(root,arm,false);
        fs::copy_file(parent_checkpoint_path,root/"parent-checkpoint.bin");fs::copy_file(parent_protocol_path,root/"parent-protocol.bin");
        cp::write_atomic((root/"protocol.bin").string(),append_protocol());
        cp::write_atomic((root/"migrated-initial.bin").string(),rl::write_appended_hold(migrated41,cp::checksum(append_protocol())));
        migration_zero_checks(root);configuration_checks(root);
        const auto round=rl::read_appended_hold(cp::read((root/"migrated-initial.bin").string()),cp::checksum(append_protocol()));
        require(round.ppo.actor.parameters==migrated41.ppo.actor.parameters&&round.actor_column_age==0&&
                round.ppo.actor_adam.steps==21700&&round.ppo.critic_adam.steps==21700,"Initial study checkpoint mismatch");
    }
    std::ofstream out(study/"readiness.json");out<<"{\"ready\":true,\"arms\":2,\"updates_each\":8,\"controls_each\":349184,\"controls_combined\":698368,\"substeps_combined\":6983680,\"native_seconds_each\":180,\"process_seconds_combined\":360,\"retries\":0,\"physical_controls\":0,\"PPO_updates\":0,\"synthetic_Adam_fixture_updates_each_network\":2}\n";
    std::cout<<"STUDY_PREPARED zero physical controls and policy PPO updates"<<std::endl;
}
struct Arm {
    fs::path root;int index=0,completed=336;Budget budget;double elapsed=0,training_start=0,first_collection=0;
    unsigned long long training_controls=0,training_substeps=0;
    Evaluation before,after;StoppingResult stop_before,stop_after;Sustained long_before,long_after;
    std::unique_ptr<Stream> stream;std::unique_ptr<nn::GpuMlp> actor,critic;
    std::unique_ptr<nn::GpuAdam> actor_opt,critic_opt;std::unique_ptr<Rollout> data;std::unique_ptr<Batches> batch;
    std::ofstream log;
};
void arm_status(Arm& a,const char* phase){
    status(a.root,phase,a.budget,a.completed);
    std::ofstream out(a.root/"progress.json");out<<std::setprecision(17)<<"{\"phase\":"<<std::quoted(phase)<<",\"completed_update\":"<<a.completed
        <<",\"controls\":"<<a.budget.controls<<",\"substeps\":"<<a.budget.substeps<<",\"training_controls\":"<<a.training_controls
        <<",\"training_started_unix\":"<<a.training_start<<",\"first_collection_unix\":"<<a.first_collection<<",\"elapsed_native_seconds\":"<<a.elapsed<<"}\n";
}
template<class F> void phase(Arm& a,F body){
    configure_study(a.root,a.index,true);
    require(cp::read((a.root/"protocol.bin").string())==append_protocol(),"Frozen study protocol differs");
    active_wall_cap=180;started=std::chrono::steady_clock::now()-std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(a.elapsed));
    a.budget.seconds=180;a.budget.maximum_controls=arm_controls;a.budget.maximum_substeps=arm_controls*10;
    {std::ofstream active(a.root.parent_path()/"active-arm.json");active<<std::setprecision(17)<<"{\"arm\":"<<std::quoted(a.index?"B":"A")<<",\"elapsed_previous\":"<<a.elapsed<<",\"phase_started_unix\":"<<unix_now()<<"}\n";}
    const auto clock=std::chrono::steady_clock::now();
    try{body();}catch(...){a.elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-clock).count();arm_status(a,"failed");throw;}
    a.elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-clock).count();
    require(a.elapsed<180,"Arm native wall cap reached");
}
void account_rollout(Arm& a){
    const auto counts=download(a.data->counts.get(),32,a.stream->value);unsigned long long controls=0,substeps=0;
    for(const auto& c:counts){controls+=c.controls;substeps+=c.substeps;}
    a.budget.controls+=int(controls-a.training_controls);a.budget.substeps+=int(substeps-a.training_substeps);
    a.training_controls=controls;a.training_substeps=substeps;
    require(download(a.data->fault.get(),1,a.stream->value)[0]==0,"CUDA physical/observation failure before optimizer");
    require(a.budget.controls<=arm_controls&&a.budget.substeps<=arm_controls*10,"Study physical budget exceeded");
}
void cache_batch(Arm& a,int update){
    auto payload=recorded_batch(*a.data,a.stream->value);
    cp::Writer out;out.value(uint32_t{1});out.array(payload);out.array(download(a.data->cost_trace.get(),rows*5,a.stream->value));
    // Native evidence snapshot; ABI sizes and source/executable hashes are frozen.
    out.value(uint32_t{sizeof(m::World)});out.array(download(a.data->worlds.get(),32,a.stream->value));
    out.value(uint32_t{sizeof(m::ForwardTask)});out.array(download(a.data->tasks.get(),32,a.stream->value));
    out.value(uint32_t{sizeof(gpu::Counts)});out.array(download(a.data->counts.get(),32,a.stream->value));
    cp::write_atomic((a.root/("batch-"+std::to_string(update)+".bin")).string(),out.data);
}
void check_initial_outputs(Arm& a){
    const auto x=download(a.data->obs.get(),rows*41,a.stream->value);
    a.actor->forward(a.data->obs.get(),rows,a.stream->value);a.critic->forward(a.data->obs.get(),rows,a.stream->value);
    const auto ga=download(a.actor->output(),rows*4,a.stream->value),gv=download(a.critic->output(),rows,a.stream->value);
    nn::CpuMlp<float> olda(old_spec(4)),olda41(spec(4)),oldc(old_spec(1)),oldc41(spec(1));
    olda.params()=original40.actor.parameters;olda41.params()=migrated41.ppo.actor.parameters;
    oldc.params()=original40.critic;oldc41.params()=migrated41.ppo.critic;
    float cpu_error=0,gpu_error=0;
    for(int i=0;i<rows;++i){const float* row=x.data()+i*41;olda.forward(row,1);olda41.forward(row,1);oldc.forward(row,1);oldc41.forward(row,1);
        for(int j=0;j<4;++j){cpu_error=fmaxf(cpu_error,fabsf(olda.output()[j]-olda41.output()[j]));gpu_error=fmaxf(gpu_error,fabsf(ga[i*4+j]-olda41.output()[j]));}
        cpu_error=fmaxf(cpu_error,fabsf(oldc.output()[0]-oldc41.output()[0]));gpu_error=fmaxf(gpu_error,fabsf(gv[i]-oldc41.output()[0]));}
    require(cpu_error<=1e-6f&&gpu_error<=2e-5f,"Initial old40/new41 CPU/CUDA function mismatch");
    std::ofstream out(a.root/"initial-function-parity.json");out<<std::setprecision(12)<<"{\"passed\":true,\"actual_rollout_rows\":"<<rows<<",\"CPU_old40_new41_error\":"<<cpu_error<<",\"CPU_GPU_error\":"<<gpu_error<<",\"extra_physical_controls\":0}\n";
}
void first_rollout(Arm& a){
    a.stream=std::make_unique<Stream>();auto s=a.stream->value;Budget axes;axes.seconds=180;parity(axes,s);
    a.budget.controls+=axes.controls;a.budget.substeps+=axes.substeps;
    a.before=evaluate(initial_actor(),a.root,"baseline",a.budget);
    a.stop_before=evaluate_stopping(initial_actor(),a.root/"baseline-stop",a.budget);
    fs::create_directory(a.root/"baseline-long");a.long_before=sustained(initial_actor(),a.root/"baseline-long",a.budget);
    require(a.long_before.safety_fault.empty(),"Baseline swimming physical fault");
    a.actor=std::make_unique<nn::GpuMlp>(spec(4),rows,seed);a.critic=std::make_unique<nn::GpuMlp>(spec(1),rows,seed+1);
    a.actor_opt=std::make_unique<nn::GpuAdam>(2532,initial_optimizer_config(true));a.critic_opt=std::make_unique<nn::GpuAdam>(2433,initial_optimizer_config(false));
    restore_training(*resume_state,*a.actor,*a.critic,*a.actor_opt,*a.critic_opt);
    a.actor_opt->set_fresh_column(41,32,0);a.critic_opt->set_fresh_column(41,32,0);
    require(a.actor->get_params()==migrated41.ppo.actor.parameters&&a.critic->get_params()==migrated41.ppo.critic,"Restoration changed network");
    save_append(a.root,336,*a.actor,*a.critic,*a.actor_opt,*a.critic_opt);
    a.data=std::make_unique<Rollout>(s);a.batch=std::make_unique<Batches>();a.first_collection=unix_now();arm_status(a,"first_rollout_before_any_PPO");
    collect_rollout(336,*a.actor,*a.critic,*a.data,s,a.root);account_rollout(a);cache_batch(a,336);check_initial_outputs(a);
    require(a.training_controls==40960&&a.training_substeps==409600,"First rollout accounting mismatch");
}
void paired_check(Arm& a,Arm& b,const fs::path& root){
    auto compare=[&](auto& x,auto& y,int n){const auto vx=download(x.get(),n,a.stream->value),vy=download(y.get(),n,b.stream->value);
        finite(vx);finite(vy);require(vx==vy,"A/B first-rollout state/action/value mismatch");};
    compare(a.data->obs,b.data->obs,rows*41);compare(a.data->previous_obs,b.data->previous_obs,rows*41);
    compare(a.data->action,b.data->action,rows*4);compare(a.data->previous_action,b.data->previous_action,rows*4);
    compare(a.data->history,b.data->history,rows);compare(a.data->logp,b.data->logp,rows);compare(a.data->value,b.data->value,rows);compare(a.data->done,b.data->done,rows);
    const auto wa=download(a.data->worlds.get(),32,a.stream->value),wb=download(b.data->worlds.get(),32,b.stream->value);
    for(int i=0;i<32;++i)require(fields(wa[i])==fields(wb[i]),"A/B final physical state mismatch");
    const auto ra=download(a.data->reward.get(),rows,a.stream->value),rb=download(b.data->reward.get(),rows,b.stream->value),
               ca=download(a.data->cost_trace.get(),rows*5,a.stream->value),cb=download(b.data->cost_trace.get(),rows*5,b.stream->value);
    int held_rows=0;float error=0;
    for(int row=0;row<rows;++row){const bool held=(row/32>=640)&&((row%32)/2)%2;const float gain=held?1.f/1024:1.f;held_rows+=held;
        require(rb[row]==ra[row]*gain,"A/B actual reward gain mismatch");
        for(int j=0;j<5;++j){error=fmaxf(error,fabsf(cb[row*5+j]-ca[row*5+j]*gain));require(cb[row*5+j]==ca[row*5+j]*gain,"A/B component gain mismatch");}}
    require(held_rows==10240&&a.actor_opt->fresh_steps()==0&&b.actor_opt->fresh_steps()==0,"Paired gate ran after PPO or wrong hold mixture");
    std::ofstream out(root/"paired-first-batch.json");out<<"{\"passed\":true,\"rows_each\":40960,\"holding_rows_each\":10240,\"all_observations_actions_values_logp_dones_equal\":true,\"trajectory_observations_and_final_states_equal\":true,\"reward_component_gain_error\":"<<error<<",\"PPO_updates_before_gate\":0,\"extra_physical_controls\":0}\n";
    std::cout<<"PAIRED_FIRST_BATCH_PASSED before any PPO; state/action/value exact match; all5holding costs scaled exactly"<<std::endl;
}
void verify_returns(Arm& a,int update){
    auto& d=*a.data;auto s=a.stream->value;
    rl::compute_gae(horizon,environments,d.reward.get(),d.value.get(),d.done.get(),d.zeros.get(),d.zeros.get(),d.last.get(),.99f,1.f,d.advantage.get(),d.returns.get(),s);
    const auto rewards=download(d.reward.get(),rows,s),values=download(d.value.get(),rows,s),ret=download(d.returns.get(),rows,s),
               adv=download(d.advantage.get(),rows,s),done=download(d.done.get(),rows,s),cost=download(d.cost_trace.get(),rows*5,s);
    const auto tasks=download(d.tasks.get(),32,s);double max_error=0,max_cost=0;int goal_count[2][2]{};
    for(int e=0;e<32;++e){const bool held=(e/2)%2;const auto expected=training_episode.task_config(1279,e);
        require(tasks[e].controls==1280&&p::norm(tasks[e].request.terminal_velocity-tasks[e].request.heading*expected.target_speed_m_s)<1e-9f,"Actual GPU goal/horizon mismatch");++goal_count[e%2][held];
        double target=0;for(int t=1279;t>=0;--t){const int row=t*32+e;require(done[row]==(t==1279?1.f:0.f),"Early terminal/reset");
            target=rewards[row]+double(.99f)*target;const double err=fabs(ret[row]-target);max_error=std::max(max_error,err);
            require(err<=.002+.0002*fabs(target)&&fabs(adv[row]+values[row]-target)<=.002+.0002*fabs(target),"Independent GAE/return mismatch");
            float sum=0;for(int j=0;j<5;++j)sum+=cost[row*5+j];const double ce=fabs(rewards[row]+p::dt*sum);max_cost=std::max(max_cost,ce);
            require(ce<=1e-6+1e-5*fabs(rewards[row]),"Actual reward/component sum mismatch");}}
    for(auto& pose:goal_count)for(int n:pose)require(n==8,"Actual goal/pose mix mismatch");
    // Verify every epoch's permutation covers every actual holding row once.
    for(int epoch=0;epoch<2;++epoch){const auto perm=rl::make_permutation(rows,seed+uint32_t(update*epochs+epoch));std::vector<bool> seen(rows,false);int held=0;
        for(int i=0;i<rows;++i){const int row=int(rl::permute(perm,i));require(row>=0&&row<rows&&!seen[row],"Optimizer permutation drops/duplicates rows");seen[row]=true;held+=row/32>=640&&((row%32)/2)%2;}
        require(held==10240,"Holding optimizer row coverage mismatch");}
    std::ofstream out(a.root/("return-integrity-"+std::to_string(update)+".json"));out<<std::setprecision(12)<<"{\"passed\":true,\"max_GAE_error\":"<<max_error<<",\"max_component_error\":"<<max_cost<<",\"held_rows_per_epoch\":10240,\"actual_pose_goal_counts_each\":8,\"terminal_only_at_horizon\":true}\n";
}
void group_diagnostics(Arm& a,int update){
    auto& d=*a.data;auto s=a.stream->value;const auto ret=download(d.returns.get(),rows,s),before=download(d.value.get(),rows,s),adv=download(d.advantage.get(),rows,s),stats=download(d.adv_stats.get(),2,s),cost=download(d.cost_trace.get(),rows*5,s);
    a.critic->forward(d.obs.get(),rows,s);const auto after=download(a.critic->output(),rows,s);
    std::ofstream out(a.root/"group-diagnostics.jsonl",std::ios::app);out<<std::setprecision(12);
    // Per-pose, per-goal and before/after command, plus pooled active hold.
    for(int group=0;group<9;++group){std::vector<float> r,v0,v1,nadv,residual;double sums[5]{};int positive=0;
        for(int row=0;row<rows;++row){const int e=row%32,t=row/32,pose=e%2,goal=(e/2)%2,post=t>=640;
            if(group==8?!(goal&&post):(pose*4+goal*2+post!=group))continue;
            r.push_back(ret[row]);v0.push_back(before[row]);v1.push_back(after[row]);residual.push_back(after[row]-ret[row]);
            const float na=(adv[row]-stats[0])/(stats[1]+1e-8f);nadv.push_back(na);positive+=na>0;
            for(int j=0;j<5;++j)sums[j]+=.05*cost[row*5+j];}
        const auto rm=rl::diagnostic_moments(r),vm0=rl::diagnostic_moments(v0),vm1=rl::diagnostic_moments(v1),em=rl::diagnostic_moments(residual),am=rl::diagnostic_moments(nadv);
        const auto d0=rl::diagnostic_value(v0,r),d1=rl::diagnostic_value(v1,r);
        out<<"{\"update\":"<<update<<",\"group\":"<<group<<",\"pose\":"<<(group==8?-1:(group/4?3:0))<<",\"goal_hold\":"<<(group==8?1:(group%4)/2)<<",\"post32\":"<<(group==8?1:group%2)
           <<",\"rows\":"<<r.size()<<",\"return_mean\":"<<rm.mean<<",\"return_std\":"<<rm.stddev()<<",\"return_min\":"<<rm.minimum<<",\"return_max\":"<<rm.maximum
           <<",\"critic_mean_before\":"<<vm0.mean<<",\"critic_mean_after\":"<<vm1.mean<<",\"critic_bias_after\":"<<em.mean
           <<",\"half_MSE_before\":"<<d0.half_mse<<",\"half_MSE_after\":"<<d1.half_mse<<",\"EV_before\":"<<d0.explained_variance<<",\"EV_after\":"<<d1.explained_variance
           <<",\"variance_defined\":"<<(rm.variance>1e-8?"true":"false")<<",\"variance_normalized_MSE\":"<<(rm.variance>1e-8?std::to_string(2*d1.half_mse/rm.variance):"null")
           <<",\"normalized_adv_mean\":"<<am.mean<<",\"normalized_adv_std\":"<<am.stddev()<<",\"normalized_adv_positive_fraction\":"<<double(positive)/r.size()<<",\"integrated_costs\":[";
        for(int j=0;j<5;++j)out<<(j?",":"")<<sums[j];out<<"]}\n";
    }
}
void run_training(Arm& a){
    auto s=a.stream->value;a.log.open(a.root/"training.csv");a.log<<"update,controls,old_actor_age,new_column_age,KL,completed_unix\n";
    a.training_start=unix_now();training_started_unix_seconds=a.training_start;
    std::cout<<"TRAINING_STARTED arm="<<(a.index?'B':'A')<<" unix="<<std::setprecision(17)<<a.training_start<<" exactly8updates"<<std::endl;
    for(int update=336;update<344;++update){
        arm_status(a,"training");
        if(update>336){collect_rollout(update,*a.actor,*a.critic,*a.data,s,a.root);account_rollout(a);cache_batch(a,update);}
        require(a.training_controls==uint64_t(update-335)*rows&&a.training_substeps==a.training_controls*10,"Training rollout accounting mismatch");
        verify_returns(a,update);
        const double kl=optimize_rollout(update,*a.actor,*a.critic,*a.actor_opt,*a.critic_opt,*a.data,*a.batch,s,a.root);
        require(download(a.data->fault.get(),1,s)[0]==0,"Nonfinite optimized parameters");
        a.completed=update+1;
        require(a.actor_opt->steps()==21700+a.actor_opt->fresh_steps()&&a.critic_opt->steps()==a.actor_opt->steps()&&
            a.actor_opt->fresh_steps()>=(a.completed-336)*160&&a.actor_opt->fresh_steps()<=(a.completed-336)*320&&a.critic_opt->fresh_steps()==a.actor_opt->fresh_steps(),"Optimizer clocks mismatch");
        save_append(a.root,a.completed,*a.actor,*a.critic,*a.actor_opt,*a.critic_opt);group_diagnostics(a,a.completed);
        a.log<<a.completed<<','<<a.budget.controls<<','<<a.actor_opt->steps()<<','<<a.actor_opt->fresh_steps()<<','<<kl<<','<<unix_now()<<'\n';a.log.flush();
        std::cout<<"PPO_UPDATE_COMPLETED arm="<<(a.index?'B':'A')<<" update="<<a.completed<<"/344"<<std::endl;
    }
    arm_status(a,"final_evaluation");a.after=evaluate(a.actor->get_params(),a.root,"final",a.budget);
    a.stop_after=evaluate_stopping(a.actor->get_params(),a.root/"final-stop",a.budget);
    fs::create_directory(a.root/"final-long");a.long_after=sustained(a.actor->get_params(),a.root/"final-long",a.budget);
    require(a.long_after.safety_fault.empty(),"Final swimming physical fault");
    require(a.completed==344&&a.training_controls==327680&&a.budget.controls==arm_controls&&a.budget.substeps==arm_controls*10,"Final budget mismatch");
    const auto reloaded=rl::read_appended_hold(cp::read((a.root/"ppo-append-update-344.bin").string()),cp::checksum(append_protocol()));
    require(reloaded.ppo.actor.parameters==a.actor->get_params()&&reloaded.ppo.critic==a.critic->get_params()&&
        reloaded.actor_column_age==a.actor_opt->fresh_steps()&&reloaded.critic_column_age==a.critic_opt->fresh_steps()&&
        reloaded.ppo.actor_adam.steps==a.actor_opt->steps(),"Final checkpoint reload mismatch");
    const bool swim=a.after.passed==4&&preserved_long_windows(a.root/"final-long");
    std::ofstream out(a.root/"result.json");out<<std::setprecision(17)<<"{\"fault\":\"\",\"arm\":"<<std::quoted(a.index?"B":"A")<<",\"holding_gain\":"<<training_episode.holding_reward_gain
        <<",\"PPO_updates\":8,\"completed_update\":344,\"training_controls\":"<<a.training_controls<<",\"controls\":"<<a.budget.controls<<",\"substeps\":"<<a.budget.substeps
        <<",\"training_started_unix\":"<<a.training_start<<",\"old_Adam_age\":"<<a.actor_opt->steps()<<",\"new_column_age\":"<<a.actor_opt->fresh_steps()
        <<",\"baseline_short\":"<<a.before.passed<<",\"final_short\":"<<a.after.passed<<",\"baseline_stop\":"<<a.stop_before.passed<<",\"final_stop\":"<<a.stop_after.passed
        <<",\"baseline_long_windows\":"<<a.long_before.windows_passed<<",\"final_long_windows\":"<<a.long_after.windows_passed<<",\"final_formal_long\":"<<a.long_after.passed
        <<",\"swimming_regression_preserved\":"<<(swim?"true":"false")<<",\"candidate_accepted\":"<<(swim&&a.stop_after.passed==6?"true":"false")<<",\"checkpoint_reload_passed\":true,\"runtime_adopted\":false,\"retries\":0}\n";
}
int run_study(const fs::path& root){
    require(fs::exists(root/"readiness.json"),"Study not prepared");require(fs::create_directory(root/"run-started"),"Study already attempted; no retries");
    Arm a,b;a.root=root/"A";a.index=0;b.root=root/"B";b.index=1;
    std::string fault;const double launch=unix_now();
    try{
        phase(a,[&]{first_rollout(a);});phase(b,[&]{first_rollout(b);});
        phase(b,[&]{paired_check(a,b,root);});
        phase(a,[&]{run_training(a);});arm_status(a,"done");
        phase(b,[&]{run_training(b);});arm_status(b,"done");
        require(a.budget.controls+b.budget.controls==698368,"Combined control budget mismatch");
    }catch(const std::exception& e){fault=e.what();std::cerr<<"STUDY_STOPPED "<<fault<<std::endl;}
    std::ofstream out(root/"result.json");out<<std::setprecision(17)<<"{\"fault\":"<<std::quoted(fault)<<",\"launched_unix\":"<<launch<<",\"completed_unix\":"<<unix_now()
        <<",\"A_completed_update\":"<<a.completed<<",\"B_completed_update\":"<<b.completed<<",\"controls\":"<<a.budget.controls+b.budget.controls<<",\"substeps\":"<<a.budget.substeps+b.budget.substeps
        <<",\"A_native_seconds\":"<<a.elapsed<<",\"B_native_seconds\":"<<b.elapsed<<",\"runtime_adopted\":false,\"retries\":0,\"closed\":true}\n";
    return fault.empty()?0:3;
}
}
#ifndef CUDARL_REST_HOLD_STUDY
int main(int argc,char** argv){try{
    require(argc==3,"Use --prepare/--run DIR");const std::string mode=argv[1];const auto root=fs::absolute(argv[2]);
    if(mode=="--prepare"){prepare_study(root);return 0;}
    if(mode=="--run")return run_study(root);
    throw std::runtime_error("Unknown study mode");
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}}

#endif
