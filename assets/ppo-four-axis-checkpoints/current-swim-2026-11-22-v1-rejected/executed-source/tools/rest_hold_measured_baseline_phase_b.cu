// Explicitly released actor-only measured-baseline PPO.
#define CUDARL_REST_HOLD_MEASURED_PHASE_B 1
#define CUDARL_REST_HOLD_MEASURED_BASELINE 1
#include "rest_hold_reset_study.cu"
namespace {
constexpr double phase_A_seconds=2.4795972;
constexpr int phase_A_controls=21024,phase_A_substeps=210240;
constexpr char report_receipt[]="Sentinel_2b00f202c5c08191aae9d36814100cec";

MeasuredBaseline load_fixed_measured(const fs::path& root) {
    const auto bytes=cp::read((root/"frozen-fit-baseline-v1.bin").string());
    cp::Reader in{bytes};
    require(in.value<uint32_t>()==0x314d4252&&in.value<uint32_t>()==1&&
            in.value<uint64_t>()==cp::checksum(rest_protocol()),"Fixed baseline metadata mismatch");
    MeasuredBaseline b;
    b.mean=in.array<double>(measured_cells);b.n=in.array<int>(measured_cells);
    require(b.mean.size()==measured_cells&&b.n.size()==measured_cells,"Fixed baseline table shape changed");
    for(int i=0;i<measured_cells;++i)
        require(std::isfinite(b.mean[i])&&b.n[i]==(i<640?4:64),"Invalid fixed baseline cell");
    for(int mode=0;mode<2;++mode){
        b.rms[mode]=in.value<double>();b.fit_n[mode]=in.value<int>();
        require(std::isfinite(b.rms[mode])&&b.rms[mode]>0&&b.fit_n[mode]==2560,"Invalid fixed task normalization");
    }
    in.finish();return b;
}

double measured_actor_update(int update,Arm& a,const MeasuredBaseline& baseline,
                              const std::vector<float>& raw) {
    auto& data=*a.data;auto& batch=*a.batch;auto& actor=*a.actor;auto& opt=*a.actor_opt;
    const auto stream=a.stream->value;
    // d.advantage is overwritten only after the independent raw-return checks.
    std::vector<float> advantage(rows);
    for(int i=0;i<rows;++i){
        const double x=(double(raw[i])-baseline.mean[measured_cell(i)])/baseline.rms[measured_mode(i)];
        advantage[i]=float(x);require(std::isfinite(advantage[i]),"Fixed residual cast invalid");
    }
    const float fixed_identity[2]={0.f,1.f};
    data.advantage.copy_from_host(advantage.data(),advantage.size());
    data.adv_stats.copy_from_host(fixed_identity,2);
    require(download(data.adv_stats.get(),2,stream)==std::vector<float>({0.f,1.f})&&
            download(data.advantage.get(),rows,stream)==advantage,"Frozen actor-signal transfer mismatch");
    const int before=opt.steps();
    double kl=0;
    for(int epoch=0;epoch<epochs;++epoch){
        CUDA_CHECK_THROW(cudaMemsetAsync(batch.stats.get(),0,batch.stats.size_bytes(),stream));
        const auto permutation=rl::make_permutation(rows,seed+static_cast<uint32_t>(update*epochs+epoch));
        for(int first=0;first<rows;first+=minibatch){
            wall_guard();
            rl::gather_rows(permutation,first,minibatch,observations,data.obs.get(),batch.obs.get(),stream);
            rl::gather_rows(permutation,first,minibatch,observations,data.previous_obs.get(),
                            batch.obs.get()+minibatch*observations,stream);
            rl::gather_rows(permutation,first,minibatch,actions,data.previous_action.get(),batch.previous_action.get(),stream);
            rl::gather_rows(permutation,first,minibatch,1,data.history.get(),batch.history.get(),stream);
            rl::gather_rows(permutation,first,minibatch,actions,data.action.get(),batch.action.get(),stream);
            rl::gather_rows(permutation,first,minibatch,1,data.logp.get(),batch.logp.get(),stream);
            rl::gather_rows(permutation,first,minibatch,1,data.advantage.get(),batch.advantage.get(),stream);
            actor.forward(batch.obs.get(),2*minibatch,stream);
            rl::ar_gaussian_ppo_loss_grad(actor.output(),data.sigma.get(),batch.previous_action.get(),
                batch.history.get(),data.zeros.get(),minibatch,actions,correlation,batch.action.get(),
                batch.logp.get(),batch.advantage.get(),data.zeros.get(),data.adv_stats.get(),{.2f,0,0},
                batch.mean_gradient.get(),batch.value_gradient.get(),batch.row_stats.get(),batch.stats.get(),stream);
            if(first==0){
                const auto adv=download(batch.advantage.get(),minibatch,stream);
                const auto critic_grad=download(batch.value_gradient.get(),minibatch,stream);
                for(int i=0;i<minibatch;++i)
                    require(adv[i]==advantage[rl::permute(permutation,i)]&&critic_grad[i]==0,
                            "Minibatch altered fixed signal or produced critic gradient");
            }
            actor.zero_grad(stream);
            actor.backward(batch.mean_gradient.get(),2*minibatch,nullptr,stream);
            opt.step(actor.params(),actor.grads(),stream);
        }
        const auto metrics=download(batch.stats.get(),5,stream);
        finite(metrics);kl=metrics[3]/rows;
        if(kl>.01)break;
    }
    gpu::finite_parameters<<<(actor.num_params()+255)/256,256,0,stream>>>(
        actor.params(),actor.num_params(),data.fault.get());
    require(download(data.fault.get(),1,stream)[0]==0,"Nonfinite actor after measured PPO update");
    require(opt.steps()-before>=40&&opt.steps()-before<=80,"Measured actor update count exceeded");
    return kl;
}

void save_measured_actor(const fs::path& root,const Arm& a){
    const auto path=root/("actor-only-update-"+std::to_string(a.completed)+".bin");
    require(!fs::exists(path),"Actor-only checkpoint identity already exists");
    cp::Writer out,adam;
    a.actor_opt->save(adam);
    out.value(uint32_t{0x31414d42});out.value(uint32_t{1});out.value(cp::checksum(rest_protocol()));
    const int age=a.actor_opt->fresh_steps();
    out.value(a.completed);out.value(age);
    const auto params=a.actor->get_params();
    out.array(params);out.array(adam.data);
    cp::write_atomic(path.string(),out.data);
    const auto stored=cp::read(path.string());cp::Reader in{stored};
    require(in.value<uint32_t>()==0x31414d42&&in.value<uint32_t>()==1&&
            in.value<uint64_t>()==cp::checksum(rest_protocol())&&in.value<int>()==a.completed&&
            in.value<int>()==age&&in.array<float>(2532)==params&&in.array<uint8_t>()==adam.data,
            "Actor-only checkpoint roundtrip failed");
    in.finish();
}

int run_measured_phase_b(const fs::path& root) {
    configure_rest(root,true);
    require(cp::read((root/"protocol.bin").string())==rest_protocol(),"Phase B protocol differs");
    std::ifstream released(root/"verified-user-report-id.txt");std::string line;std::getline(released,line);
    require(line==report_receipt,"Calibrated check report receipt absent or wrong");
    const auto fixed=load_fixed_measured(root);
    require(fs::create_directory(root/"run-started"),"Measured Phase B already attempted");
    started=std::chrono::steady_clock::now();active_wall_cap=120-phase_A_seconds;
    Arm a;a.root=root;a.budget.seconds=active_wall_cap;a.budget.maximum_controls=113664;
    a.budget.maximum_substeps=1136640;a.budget.controls=phase_A_controls;a.budget.substeps=phase_A_substeps;
    std::string fault,gate;RestAssessment final;int updates_done=0;bool qualified=false;
    const double launch=unix_now();
    try{
        a.stream=std::make_unique<Stream>();a.actor=std::make_unique<nn::GpuMlp>(spec(4),rows,seed);
        a.actor->set_params(resume_state->actor.parameters);
        a.actor_opt=std::make_unique<nn::GpuAdam>(2532,initial_optimizer_config(true));
        restore_optimizer(*a.actor_opt,resume_state->actor_adam);
        a.actor_opt->set_learning_rate(*actor_lr_override);
        a.actor_opt->set_fresh_column(41,32,0);
        require(a.actor_opt->steps()==21700&&a.actor_opt->fresh_steps()==0&&
                a.actor->get_params()==resume_state->actor.parameters,"Actor or optimizer handoff mismatch");
        a.data=std::make_unique<Rollout>(a.stream->value);a.batch=std::make_unique<Batches>();
        auto& d=*a.data;const auto stream=a.stream->value;
        calibration_collect=false;
        std::cout<<"ACTOR_ONLY_PPO_FIRST_COLLECTION_STARTED unix="<<std::setprecision(17)<<unix_now()<<std::endl;
        for(int update=336;update<344;++update){
            wall_guard();status(root,"PPO_fresh_collect",a.budget,a.completed);
            collect_rollout_selected_value(update,*a.actor,nullptr,d,stream,root);
            rest_account(a);cache_batch(a,update);verify_rest_batch(a,("PPO-"+std::to_string(update)).c_str());
            require(a.training_controls==10240*(update-335)&&a.training_substeps==102400*(update-335),
                    "Fresh PPO rollout accounting changed");
            const auto raw=download(d.returns.get(),rows,stream);
            const auto value=download(d.value.get(),rows,stream);
            require(raw==verified_returns&&raw==verified_adv&&
                    std::all_of(value.begin(),value.end(),[](float x){return x==0;}),"PPO raw GAE not critic-free");
            const auto sig=measured_signal(raw,fixed,false);
            {
                std::ofstream log(root/"fresh-signal.jsonl",std::ios::app);
                log<<std::setprecision(12)<<"{\"update\":"<<update<<",\"hold_near_rest_occupancy_passed\":"
                   <<(reset_occupancy_passed?"true":"false")<<",\"hold_squared_scaled_share\":"<<sig.hold_share
                   <<",\"hold_raw_residual_RMSE\":"<<sig.rmse[1]<<",\"swim_raw_residual_RMSE\":"<<sig.rmse[0]
                   <<",\"hold_raw_residual_bias\":"<<sig.bias[1]<<",\"swim_raw_residual_bias\":"<<sig.bias[0]
                   <<",\"share_passed\":"<<(sig.passed?"true":"false")
                   <<",\"actor_steps_before\":"<<a.actor_opt->steps()-21700<<"}\n";
            }
            if(update==336&&(!reset_occupancy_passed||!sig.passed)){
                gate=!reset_occupancy_passed?"first_PPO_hold_occupancy":"first_PPO_scaled_signal_share";break;
            }
            status(root,"actor_only_PPO_optimize",a.budget,a.completed);
            const double kl=measured_actor_update(update,a,fixed,raw);
            a.completed=update+1;++updates_done;
            require(a.actor_opt->fresh_steps()==a.actor_opt->steps()-21700&&a.actor_opt->fresh_steps()<=640,
                    "Actor optimizer cap exceeded");
            save_measured_actor(root,a);
            {
                std::ofstream log(root/"ppo-updates.jsonl",std::ios::app);
                log<<std::setprecision(12)<<"{\"completed\":"<<a.completed<<",\"actor_steps\":"
                   <<a.actor_opt->fresh_steps()<<",\"critic_steps\":0,\"KL\":"<<kl<<"}\n";
            }
            std::cout<<"ACTOR_ONLY_PPO_COMPLETED "<<a.completed<<"/344 controls="<<a.budget.controls<<std::endl;
        }
        if(gate.empty()){
            require(updates_done==8,"Missing bounded PPO updates");
            status(root,"final_assessments",a.budget,a.completed);
            const auto params=a.actor->get_params();
            a.after=evaluate(params,root,"final",a.budget);
            final=evaluate_rest(params,root/"final-rest",a.budget);
            fs::create_directory(root/"final-long");
            a.long_after=sustained(params,root/"final-long",a.budget);
            require(a.long_after.safety_fault.empty(),"Frozen final swim reached hard safety fault");
            qualified=final.passed==6&&a.after.passed==4&&preserved_long_windows(root/"final-long");
            require(a.budget.controls==113664&&a.budget.substeps==1136640,"Final total accounting mismatch");
        }
        wall_guard();
    }catch(const std::exception& e){
        fault=e.what();std::cerr<<"PPO_PHASE_B_STOPPED "<<fault<<std::endl;
    }
    const int steps=a.actor_opt?a.actor_opt->steps()-21700:0;
    std::ofstream out(root/"result.json");
    out<<std::setprecision(17)<<"{\"fault\":"<<std::quoted(fault)<<",\"gate_failed\":"<<std::quoted(gate)
       <<",\"launched_unix\":"<<launch<<",\"ended_unix\":"<<unix_now()
       <<",\"phase_B_native_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()
       <<",\"phase_A_active_seconds\":"<<phase_A_seconds<<",\"total_controls\":"<<a.budget.controls
       <<",\"total_substeps\":"<<a.budget.substeps<<",\"PPO_updates\":"<<updates_done<<",\"actor_steps\":"<<steps
       <<",\"critic_steps\":0,\"critic_forward_calls\":0,\"final_short\":"<<a.after.passed
       <<",\"final_rest\":"<<final.passed<<",\"final_rest_completed\":"<<final.completed
       <<",\"final_long_windows\":"<<a.long_after.windows_passed<<",\"candidate_accepted\":"<<(qualified&&fault.empty()?"true":"false")
       <<",\"runtime_adopted\":false,\"retries\":0,\"closed\":true}\n";
    out.close();
    status(root,fault.empty()?(gate.empty()?"done":"gate_stopped"):"failed",a.budget,a.completed);
    std::cout<<"ACTOR_ONLY_PPO_PHASE_B_CLOSED updates="<<updates_done<<" steps="<<steps<<" gate="<<gate<<" fault="<<fault<<std::endl;
    return fault.empty()?0:3;
}
}
int main(int argc,char** argv){
    try{
        require(argc==3,"Use --validate or --run-phase-B DIR");
        const std::string mode=argv[1];const auto root=fs::absolute(argv[2]);
        if(mode=="--validate"){
            configure_rest(root,true);
            require(cp::read((root/"protocol.bin").string())==rest_protocol(),"Validation protocol differs");
            (void)load_fixed_measured(root);
            std::cout<<"PHASE_B_ZERO_STEP_INPUT_CHECK_PASSED"<<std::endl;
            return 0;
        }
        if(mode=="--run-phase-B")return run_measured_phase_b(root);
        throw std::runtime_error("Unsupported Phase B mode");
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}
}
