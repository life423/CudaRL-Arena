// Isolated41-input research entry; shares the native PPO/evaluation implementation.
#define main historical_motor_entry
#include "ppo_motor.cu"
#undef main
#include "rl/append_hold_checkpoint.h"
#include <cstring>
namespace {
rl::PpoTrainingState original40;
rl::AppendedHoldState migrated41;
constexpr int replay_controls=5120, maximum_replays=2;
constexpr int all_control_cap=359424; //8PPOupdates+two replay caps+axis+baseline/final assessments.
nn::MlpSpec old_spec(int output) {
    return {{40,32,32,output},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}};
}
std::vector<uint8_t> append_protocol() {
    cp::Writer out;out.value(uint32_t{16});out.value(uint32_t{41});out.value(uint32_t{9});
    out.value(seed);out.value(cp::checksum(parent_checkpoint_payload));out.value(cp::checksum(parent_protocol_payload));
    for(int v:{336,8,32,1280,256,2,all_control_cap,all_control_cap*10,maximum_replays})out.value(v);
    for(float v:{.02f,.000125f,.001f,.8f,.99f,1.f,180.f})out.value(v);
    const std::string reward="reward8 unchanged; appended signed forward error tanh(e/.02m); legacy40 schema2 unchanged";
    out.array(std::vector<uint8_t>(reward.begin(),reward.end()));return out.data;
}
void configure_append(const fs::path& root,bool from_copy) {
    horizon=1280;rows=32*horizon;updates=8;training_cap=updates*rows;
    control_cap=all_control_cap;substep_cap=control_cap*10;active_wall_cap=180;
    holding=true;depth_offsets=true;training_episode.appended_hold=true;
    actor_lr_override=.000125f;
    auto base=fs::absolute("assets/ppo-four-axis-checkpoints/update-336-episode64");
    load_resume_source(from_copy?root/"parent-checkpoint.bin":base/"ppo-update-336.bin",
                       from_copy?root/"parent-protocol.bin":base/"protocol.bin",false,false);
    original40=*resume_state;migrated41=rl::migrate_appended_hold(original40);
    resume_state=migrated41.ppo;training_episode.observation_schema=4;
}
void save_append(const fs::path& root,int update,nn::GpuMlp& actor,nn::GpuMlp& critic,
                 nn::GpuAdam& actor_opt,nn::GpuAdam& critic_opt) {
    rl::AppendedHoldState state;state.ppo=*resume_state;
    state.ppo.actor.update=update;state.ppo.actor.parameters=actor.get_params();state.ppo.critic=critic.get_params();
    cp::Writer a,c;actor_opt.save(a);critic_opt.save(c);cp::Reader ar{a.data},cr{c.data};
    state.ppo.actor_adam=rl::read_append_adam(ar,2532);state.ppo.critic_adam=rl::read_append_adam(cr,2433);
    ar.finish();cr.finish();state.actor_column_age=actor_opt.fresh_steps();state.critic_column_age=critic_opt.fresh_steps();
    cp::write_atomic((root/("ppo-append-update-"+std::to_string(update)+".bin")).string(),
                     rl::write_appended_hold(state,cp::checksum(append_protocol())));
}
void migration_zero_checks(const fs::path& root) {
    float signed_feature_error=0,zero_actor_error=0,zero_critic_error=0;
    for(float displacement:{-.02f,-.004f,0.f,.004f,.02f}) {
        auto w=m::initial_world(0);m::ForwardTask task;auto c=m::StopSequence{}.config(640,4);c.position_hold=true;
        require(m::begin_forward(w,c,task),"Signed-feature fixture setup");
        task.controls=640;task.last_world_control=640;w.physical.control_steps=640;
        require(m::update_velocity_request(w,c,task),"Signed-feature anchor setup");
        w.physical.body.position+=task.request.heading*(displacement/p::scale);
        float legacy[40]{},appended[41]{};auto old_c=c;old_c.position_hold=false;old_c.observation_schema=2;
        require(m::observe(task,w,old_c,legacy)&&m::observe_appended(task,w,c,appended),"Signed-feature observation");
        signed_feature_error=std::max(signed_feature_error,fabsf(appended[40]+tanhf(displacement/.02f)));
        for(int j=0;j<40;++j)require(legacy[j]==appended[j],"Old40 changed in signed fixture");
        for(int output:{4,1}) {
            nn::CpuMlp<float> a(old_spec(output)),b(spec(output));
            a.params()=output==4?original40.actor.parameters:original40.critic;
            b.params()=output==4?migrated41.ppo.actor.parameters:migrated41.ppo.critic;
            a.forward(legacy,1);b.forward(appended,1);
            for(int j=0;j<output;++j){float error=fabsf(a.output()[j]-b.output()[j]);
                if(output==4)zero_actor_error=std::max(zero_actor_error,error);else zero_critic_error=std::max(zero_critic_error,error);}
        }
    }
    require(signed_feature_error<1e-5f&&zero_actor_error<1e-6f&&zero_critic_error<1e-6f,"Signed feature/output invariance failed");
    for(int network=0;network<2;++network){
        const auto& old=network?original40.critic:original40.actor.parameters;
        const auto& now=network?migrated41.ppo.critic:migrated41.ppo.actor.parameters;
        const auto& oa=network?original40.critic_adam:original40.actor_adam;
        const auto& na=network?migrated41.ppo.critic_adam:migrated41.ppo.actor_adam;
        require(now==rl::append_zero_input_column(old) && na.first==rl::append_zero_input_column(oa.first) &&
                na.second==rl::append_zero_input_column(oa.second) && na.steps==oa.steps &&
                cp::checksum(rl::adam_payload(oa))!=0,"Migration parameter/moment mapping failed");
        require(std::memcmp(&oa.config,&na.config,sizeof(nn::AdamConfig))==0,"Migration optimizer config changed");
        auto cfg=na.config;cfg.lr=network?.001f:.000125f;
        nn::GpuAdam fixture(static_cast<int>(now.size()),cfg);restore_optimizer(fixture,na);
        fixture.set_learning_rate(cfg.lr);fixture.set_fresh_column(41,32,0);
        device_ptr<float> parameters(now.size()),gradient(now.size());parameters.copy_from_host(now.data(),now.size());
        std::vector<float> g(now.size(),0);g[40]=1;gradient.copy_from_host(g.data(),g.size());
        fixture.step(parameters.get(),gradient.get());CUDA_CHECK_THROW(cudaDeviceSynchronize());
        std::vector<float> actual(now.size());parameters.copy_to_host(actual.data(),actual.size());
        float max_error=0;
        for(size_t i=0;i<now.size();++i){
            const bool fresh=i<1312 && i%41==40;const int age=fresh?1:na.steps+1;
            const float gi=i==40?1.f*std::min(1.f,cfg.max_grad_norm/(1.f+1e-6f)):0.f;
            const float m1=cfg.beta1*na.first[i]+(1-cfg.beta1)*gi;
            const float v1=cfg.beta2*na.second[i]+(1-cfg.beta2)*gi*gi;
            const float bc1=static_cast<float>(1-std::pow(double(cfg.beta1),age));
            const float bc2=static_cast<float>(1-std::pow(double(cfg.beta2),age));
            const float expected=now[i]-cfg.lr*(m1/bc1)/(sqrtf(v1/bc2)+cfg.eps);
            max_error=std::max(max_error,fabsf(expected-actual[i]));
        }
        require(max_error<2e-6f && fixture.steps()==21701 && fixture.fresh_steps()==1,
                "Fresh-column Adam age/reference mismatch");
    }
    const auto bytes=rl::write_appended_hold(migrated41,cp::checksum(append_protocol()));
    const auto loaded=rl::read_appended_hold(bytes,cp::checksum(append_protocol()));
    require(loaded.ppo.actor.parameters==migrated41.ppo.actor.parameters &&
            loaded.ppo.critic_adam.second==migrated41.ppo.critic_adam.second,"Append roundtrip failed");
    bool rejected=false;try{(void)rl::read_ppo_training_state(bytes,cp::checksum(append_protocol()));}
    catch(const std::runtime_error&){rejected=true;}require(rejected,"Old reader accepted41checkpoint");
    std::ofstream out(root/"migration-zero-checks.json");
    out<<"{\"passed\":true,\"actor_parameters\":2532,\"critic_parameters\":2433,\"old_age\":21700,"
          "\"new_column_age\":0,\"synthetic_Adam_fixture_updates_each\":1,\"policy_PPO_updates\":0,\"physical_steps\":0}\n";
}
int prepare_append(const fs::path& root) {
    require(!fs::exists(root),"Append run identity exists");fs::create_directories(root);
    fs::copy_file(parent_checkpoint_path,root/"parent-checkpoint.bin");
    fs::copy_file(parent_protocol_path,root/"parent-protocol.bin");
    cp::write_atomic((root/"protocol.bin").string(),append_protocol());
    cp::write_atomic((root/"migrated-initial.bin").string(),rl::write_appended_hold(migrated41,cp::checksum(append_protocol())));
    migration_zero_checks(root);
    check_initialization();
    std::ofstream out(root/"readiness.json");
    out<<"{\"ready\":true,\"observations\":41,\"checkpoint_schema\":5,\"observation_schema\":4,\"reward\":9,"
          "\"reward_formula_same_as\":8,\"updates_exact\":8,\"starting_update\":336,\"ending_update\":344,"
          "\"training_controls\":327680,\"combined_controls_cap\":359424,\"combined_substeps_cap\":3594240,"
          "\"maximum_replays\":2,\"native_seconds\":180,\"physical_steps\":0,\"PPO_updates\":0,\"runtime_adopted\":false}\n";
    std::cout<<"APPEND_PREPARED zero physical/PPO steps; independent new-column Adam age0"<<std::endl;return 0;
}
int verify_append(const fs::path& root) {
    fs::path attempt;
    for(int i=1;i<=maximum_replays;++i){auto p=root/("verification-"+std::to_string(i));if(!fs::exists(p)){attempt=p;break;}}
    require(!attempt.empty(),"Migration replay budget closed");fs::create_directory(attempt);
    started=std::chrono::steady_clock::now();Budget budget;budget.maximum_controls=replay_controls;
    budget.maximum_substeps=replay_controls*10;budget.seconds=180;
    float max_obs=0,max_actor=0,max_critic=0,max_action=0,max_state=0,max_logp=0;
    std::vector<float> input40,input41,cpu_actor,cpu_value;
    std::ofstream trace(attempt/"replay.csv");
    trace<<std::setprecision(12)<<"pose,control,time_s,new_feature,observation_error,actor_error,critic_error,action_error,state_error,reward6,reward8\n";
    double sum6=0,sum8=0,return6_total=0,return8_total=0;int hold_rows=0;
    std::string fault;
    try{
        for(int case_id=0;case_id<2;++case_id){
            const int pose=case_id?3:0;
            auto a=m::initial_world(pose),b=a;a.arena=b.arena=m::long_swim_arena();
            m::ForwardTask ta,tb;auto initial=m::StopSequence{}.config(0,2);
            require(m::begin_forward(a,initial,ta)&&m::begin_forward(b,initial,tb),"Replay setup");
            nn::CpuMlp<float> old_actor(old_spec(4)),new_actor(spec(4)),old_critic(old_spec(1)),new_critic(spec(1));
            old_actor.params()=original40.actor.parameters;new_actor.params()=migrated41.ppo.actor.parameters;
            old_critic.params()=original40.critic;new_critic.params()=migrated41.ppo.critic;
            float za_previous[4]{},zb_previous[4]{},ma_previous[4]{},mb_previous[4]{};
            std::vector<float> reward6,reward8;
            for(int t=0;t<1280;++t){
                auto c=m::StopSequence{}.config(t,4);c.position_hold=true;
                require(m::update_velocity_request(a,c,ta)&&m::update_velocity_request(b,c,tb),"Replay goal");
                auto legacy=c;legacy.position_hold=false;legacy.observation_schema=2;
                float x[40]{},y[41]{};require(m::observe(ta,a,legacy,x)&&m::observe_appended(tb,b,c,y),"Replay observations");
                for(int j=0;j<40;++j)max_obs=std::max(max_obs,fabsf(x[j]-y[j]));
                old_actor.forward(x,1);new_actor.forward(y,1);old_critic.forward(x,1);new_critic.forward(y,1);
                const auto* ma=old_actor.output();const auto* mb=new_actor.output();
                max_critic=std::max(max_critic,fabsf(old_critic.output()[0]-new_critic.output()[0]));
                float mu_a[4],mu_b[4],ls_a[4],ls_b[4],za[4],zb[4];
                rl::ar_parameters(ma,ma_previous,za_previous,original40.actor.log_sigma.data(),4,correlation,t!=0,mu_a,ls_a);
                rl::ar_parameters(mb,mb_previous,zb_previous,original40.actor.log_sigma.data(),4,correlation,t!=0,mu_b,ls_b);
                for(int j=0;j<4;++j){
                    const float noise=cudarl::rng::normal(seed+2,static_cast<uint32_t>(336*1280+t),case_id*4+j);
                    za[j]=mu_a[j]+expf(ls_a[j])*noise;zb[j]=mu_b[j]+expf(ls_b[j])*noise;
                    max_actor=std::max(max_actor,fabsf(ma[j]-mb[j]));max_action=std::max(max_action,fabsf(za[j]-zb[j]));
                    za_previous[j]=za[j];zb_previous[j]=zb[j];ma_previous[j]=ma[j];mb_previous[j]=mb[j];
                }
                max_logp=std::max(max_logp,fabsf(rl::gaussian_logp(mu_a,ls_a,za,4)-rl::gaussian_logp(mu_b,ls_b,zb,4)));
                input40.insert(input40.end(),x,x+40);input41.insert(input41.end(),y,y+41);
                cpu_actor.insert(cpu_actor.end(),ma,ma+4);cpu_value.push_back(old_critic.output()[0]);
                m::Action aa,ab;require(m::action_from_latent(za,aa)&&m::action_from_latent(zb,ab),"Replay action");
                budget.step(a,aa.command);budget.step(b,ab.command);
                const auto fa=fields(a),fb=fields(b);for(size_t j=0;j<fa.size();++j)max_state=std::max(max_state,fabsf(fa[j]-fb[j]));
                const auto ia=m::record_control(ta,a,legacy),ib=m::record_control(tb,b,c);
                require(ia.accepted&&ib.accepted,"Replay transition");
                if(t>=640){reward6.push_back(ia.reward);reward8.push_back(ib.reward);sum6-=ia.reward;sum8-=ib.reward;++hold_rows;}
                trace<<pose<<','<<t+1<<','<<(t+1)*.05<<','<<y[40]<<','<<max_obs<<','<<max_actor<<','<<max_critic<<','<<max_action<<','<<max_state<<','<<ia.reward<<','<<ib.reward<<'\n';
                require(max_obs<=1e-6f&&max_actor<=1e-6f&&max_critic<=1e-6f&&max_action<=1e-6f&&max_state<=1e-5f&&max_logp<=1e-6f,
                        "Migration replay tolerance failed; fix migration before training");
            }
            double r6=0,r8=0;for(int t=639;t>=0;--t){r6=reward6[t]+.99*r6;r8=reward8[t]+.99*r8;return6_total+=fabs(r6);return8_total+=fabs(r8);}
        }
        const int count=static_cast<int>(cpu_value.size());
        nn::GpuMlp old_actor(old_spec(4),count,seed),new_actor(spec(4),count,seed),old_critic(old_spec(1),count,seed),new_critic(spec(1),count,seed);
        old_actor.set_params(original40.actor.parameters);new_actor.set_params(migrated41.ppo.actor.parameters);
        old_critic.set_params(original40.critic);new_critic.set_params(migrated41.ppo.critic);
        device_ptr<float> x(input40.size()),y(input41.size());x.copy_from_host(input40.data(),input40.size());y.copy_from_host(input41.data(),input41.size());
        old_actor.forward(x.get(),count);new_actor.forward(y.get(),count);old_critic.forward(x.get(),count);new_critic.forward(y.get(),count);
        const auto oa=download(old_actor.output(),count*4,nullptr),na=download(new_actor.output(),count*4,nullptr),
                   oc=download(old_critic.output(),count,nullptr),nc=download(new_critic.output(),count,nullptr);
        float gpu_actor=0,gpu_value=0,cpu_gpu=0;
        for(size_t j=0;j<oa.size();++j){gpu_actor=std::max(gpu_actor,fabsf(oa[j]-na[j]));cpu_gpu=std::max(cpu_gpu,fabsf(na[j]-cpu_actor[j]));}
        for(size_t j=0;j<oc.size();++j){gpu_value=std::max(gpu_value,fabsf(oc[j]-nc[j]));cpu_gpu=std::max(cpu_gpu,fabsf(nc[j]-cpu_value[j]));}
        require(gpu_actor<=1e-6f&&gpu_value<=1e-6f&&cpu_gpu<=2e-5f,"Actual CUDA migrated outputs mismatch");
        std::ofstream gpu(attempt/"gpu-parity.json");gpu<<std::setprecision(12)<<"{\"rows\":"<<count<<",\"actor_error\":"<<gpu_actor<<",\"critic_error\":"<<gpu_value<<",\"CPU_GPU_error\":"<<cpu_gpu<<",\"physical_steps\":0}\n";
        require(budget.controls==5120&&budget.substeps==51200,"Replay accounting");
    }catch(const std::exception& e){fault=e.what();}
    std::ofstream report(attempt/"result.json");
    report<<std::setprecision(12)<<"{\"passed\":"<<(fault.empty()?"true":"false")<<",\"fault\":"<<std::quoted(fault)
          <<",\"seed\":"<<seed<<",\"controls\":"<<budget.controls<<",\"substeps\":"<<budget.substeps
          <<",\"old40_observation_error\":"<<max_obs<<",\"actor_error\":"<<max_actor<<",\"critic_error\":"<<max_critic
          <<",\"sampled_action_error\":"<<max_action<<",\"state_error\":"<<max_state<<",\"AR_logp_error\":"<<max_logp<<",\"PPO_updates\":0}\n";
    std::ofstream scale(attempt/"reward-scale.json");
    scale<<std::setprecision(12)<<"{\"same_physical_trajectory\":true,\"hold_rows\":"<<hold_rows<<",\"reward6_cost_sum\":"<<sum6
         <<",\"reward8_cost_sum\":"<<sum8<<",\"cost_ratio\":"<<(sum6?sum8/sum6:0)<<",\"mean_abs_return6\":"<<return6_total/std::max(1,hold_rows)
         <<",\"mean_abs_return8\":"<<return8_total/std::max(1,hold_rows)<<",\"decision\":\"Keep reward8 formula unchanged for this8-update input-migration comparison; scale remains a separate optimization concern\",\"PPO_updates\":0}\n";
    report.close();scale.close();
    if(fault.empty()){std::ofstream marker(root/"migration-verified");marker<<attempt.filename().string()<<'\n';}
    std::cout<<"MIGRATION_REPLAY "<<(fault.empty()?"PASSED":"FAILED")<<" controls="<<budget.controls<<" max_state="<<max_state<<" reward_cost_ratio="<<(sum6?sum8/sum6:0)<<std::endl;
    return fault.empty()?0:3;
}
int run_append(const fs::path& root) {
    require(fs::exists(root/"migration-verified"),"Migration replay must pass before training");
    require(fs::create_directory(root/"run-started"),"PPO trial already attempted; no retries");
    started=std::chrono::steady_clock::now();Budget budget;budget.seconds=180;
    for(int i=1;i<=2;++i){auto p=root/("verification-"+std::to_string(i))/"result.json";if(fs::exists(p)){
        // Native replay proof is immutable before launch; wrapper verifies its hashes.
        std::ifstream in(p);std::string line((std::istreambuf_iterator<char>(in)),{});
        auto number=[&](const std::string& key){auto at=line.find("\""+key+"\":");require(at!=std::string::npos,"Replay count missing");return std::stoi(line.substr(at+key.size()+3));};
        budget.controls+=number("controls");budget.substeps+=number("substeps");
    }}
    const int replay_spent=budget.controls;
    int completed=336;std::string fault;Evaluation before,after;StoppingResult stop_before,stop_after;Sustained long_before,long_after;
    try{
        Stream stream;Budget axes;axes.seconds=180;parity(axes,stream.value);budget.controls+=axes.controls;budget.substeps+=axes.substeps;
        before=evaluate(initial_actor(),root,"baseline",budget);
        stop_before=evaluate_stopping(initial_actor(),root/"baseline-stop",budget);
        fs::create_directory(root/"baseline-long");long_before=sustained(initial_actor(),root/"baseline-long",budget);
        require(long_before.safety_fault.empty(),"Baseline swimming physical failure");
        nn::GpuMlp actor(spec(4),rows,seed),critic(spec(1),rows,seed+1);
        nn::GpuAdam actor_opt(actor.num_params(),initial_optimizer_config(true)),critic_opt(critic.num_params(),initial_optimizer_config(false));
        restore_training(*resume_state,actor,critic,actor_opt,critic_opt);
        actor_opt.set_fresh_column(41,32,0);critic_opt.set_fresh_column(41,32,0);
        require(actor.get_params()==migrated41.ppo.actor.parameters&&critic.get_params()==migrated41.ppo.critic,"Restoration changed network");
        save_append(root,336,actor,critic,actor_opt,critic_opt);
        Rollout data(stream.value);Batches batch;Timer timer;
        unsigned long long previous_controls=0,previous_substeps=0;
        std::ofstream log(root/"training.csv");log<<"update,controls,old_actor_age,new_column_age,rollout_ms,update_ms,KL\n";
        training_started_unix_seconds=std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::cout<<"TRAINING_STARTED unix_seconds="<<std::setprecision(17)<<training_started_unix_seconds<<" exactly8PPOupdates from336; new-column age0"<<std::endl;
        for(int update=336;update<344;++update){
            status(root,"training",budget,completed);timer.begin(stream.value);
            collect_rollout(update,actor,critic,data,stream.value,root);const double rollout_ms=timer.end(stream.value);
            const auto counts=download(data.counts.get(),32,stream.value);unsigned long long controls=0,substeps=0;
            for(const auto& count:counts){controls+=count.controls;substeps+=count.substeps;}
            budget.controls+=static_cast<int>(controls-previous_controls);budget.substeps+=static_cast<int>(substeps-previous_substeps);
            previous_controls=controls;previous_substeps=substeps;
            require(download(data.fault.get(),1,stream.value)[0]==0,"GPU physical/observation failure before PPO");
            require(controls==static_cast<unsigned long long>((update-335)*rows)&&substeps==controls*10,"Rollout accounting");
            require(budget.controls<=control_cap&&budget.substeps<=substep_cap,"PPO budget exceeded");
            timer.begin(stream.value);const double kl=optimize_rollout(update,actor,critic,actor_opt,critic_opt,data,batch,stream.value,root);
            const double update_ms=timer.end(stream.value);completed=update+1;
            require(download(data.fault.get(),1,stream.value)[0]==0,"Nonfinite optimized parameters");
            require(actor_opt.steps()==21700+(completed-336)*320&&critic_opt.steps()==actor_opt.steps()&&actor_opt.fresh_steps()==(completed-336)*320&&critic_opt.fresh_steps()==actor_opt.fresh_steps(),"Adam age accounting");
            save_append(root,completed,actor,critic,actor_opt,critic_opt);
            log<<completed<<','<<budget.controls<<','<<actor_opt.steps()<<','<<actor_opt.fresh_steps()<<','<<rollout_ms<<','<<update_ms<<','<<kl<<'\n';log.flush();
            if(completed==337){std::ofstream first(root/"first-update.json");first<<std::setprecision(17)<<"{\"update\":337,\"unix_seconds\":"<<std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()<<",\"old_Adam_age\":22020,\"new_column_age\":320}\n";}
            std::cout<<"PPO_UPDATE_COMPLETED "<<completed<<"/344 controls="<<budget.controls<<std::endl;
        }
        status(root,"evaluation",budget,completed);
        after=evaluate(actor.get_params(),root,"final",budget);
        stop_after=evaluate_stopping(actor.get_params(),root/"final-stop",budget);
        fs::create_directory(root/"final-long");long_after=sustained(actor.get_params(),root/"final-long",budget);
        require(long_after.safety_fault.empty(),"Final swimming physical failure");
        const int expected=static_cast<int>(previous_controls)+64+2*10720+replay_spent;
        require(completed==344&&previous_controls==327680&&budget.controls==expected&&budget.substeps==budget.controls*10,"Final accounting mismatch");
    }catch(const std::exception& e){fault=e.what();}
    const bool preserved=fault.empty()&&after.passed==4&&preserved_long_windows(root/"final-long");
    const bool accepted=preserved&&stop_after.passed==6;
    std::ofstream out(root/"result.json");out<<std::setprecision(17)<<"{\"fault\":"<<std::quoted(fault)<<",\"completed_update\":"<<completed<<",\"PPO_updates\":"<<completed-336
        <<",\"training_started_unix_seconds\":"<<training_started_unix_seconds<<",\"controls\":"<<budget.controls<<",\"substeps\":"<<budget.substeps
        <<",\"baseline_short\":"<<before.passed<<",\"final_short\":"<<after.passed<<",\"baseline_stop\":"<<stop_before.passed<<",\"final_stop\":"<<stop_after.passed
        <<",\"stop_total\":6,\"baseline_long_windows\":"<<long_before.windows_passed<<",\"final_long_windows\":"<<long_after.windows_passed<<",\"final_formal_long\":"<<long_after.passed
        <<",\"swimming_regression_preserved\":"<<(preserved?"true":"false")<<",\"candidate_accepted\":"<<(accepted?"true":"false")<<",\"runtime_adopted\":false,\"closed\":true,\"retries\":0}\n";
    out.close();status(root,fault.empty()?"done":"failed",budget,completed);std::ifstream report(root/"result.json");std::cout<<report.rdbuf()<<std::flush;return fault.empty()?0:3;
}
} // namespace
int main(int argc,char** argv){try{
    require(argc==3,"Use --prepare/--verify/--run/--load DIR");const std::string mode=argv[1];const auto root=fs::absolute(argv[2]);
    configure_append(root,mode!="--prepare");
    if(mode=="--prepare")return prepare_append(root);
    require(cp::read((root/"protocol.bin").string())==append_protocol(),"Frozen append protocol differs");
    if(mode=="--verify")return verify_append(root);
    if(mode=="--run")return run_append(root);
    if(mode=="--load"){
        const auto state=rl::read_appended_hold(cp::read((root/"ppo-append-update-344.bin").string()),cp::checksum(append_protocol()));
        nn::GpuMlp actor(spec(4),1,seed),critic(spec(1),1,seed);actor.set_params(state.ppo.actor.parameters);critic.set_params(state.ppo.critic);
        nn::GpuAdam ao(2532,state.ppo.actor_adam.config),co(2433,state.ppo.critic_adam.config);restore_optimizer(ao,state.ppo.actor_adam);restore_optimizer(co,state.ppo.critic_adam);
        ao.set_fresh_column(41,32,state.actor_column_age);co.set_fresh_column(41,32,state.critic_column_age);
        require(state.ppo.actor.update==344&&ao.steps()==24260&&co.steps()==24260&&ao.fresh_steps()==2560&&co.fresh_steps()==2560,"Reload age mismatch");
        std::cout<<"APPEND_RELOAD_PASSED update344 oldAge24260 newAge2560 zero physical/PPO steps"<<std::endl;return 0;
    }
    throw std::runtime_error("Unknown append mode");
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}}
