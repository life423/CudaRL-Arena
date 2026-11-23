/* Readout-neutral, fit-only control variate for actor-only PPO, no critic. */
constexpr int measured_cells=680; // two poses × (320 swim slots + 20 hold slots)
bool measured_fit(int row){return row%32%8<4;}
int measured_mode(int row){return (row%32/2)%2;}
int measured_cell(int row){
    const int mode=measured_mode(row),pose=row%2,t=row/32;
    return mode?640+pose*20+t%20:pose*320+t;
}
struct MeasuredBaseline {
    std::vector<double> mean=std::vector<double>(measured_cells);
    std::vector<int> n=std::vector<int>(measured_cells);
    double rms[2]{};
    int fit_n[2]{};
};
MeasuredBaseline measured_fit_baseline(const std::vector<float>& raw) {
    require(raw.size()==10240,"Measured baseline got invalid row count");
    MeasuredBaseline b;
    for(int row=0;row<int(raw.size());++row){
        require(std::isfinite(raw[row]),"Measured baseline saw nonfinite raw return");
        if(!measured_fit(row))continue;
        const int mode=measured_mode(row),cell=measured_cell(row);
        b.mean[cell]+=raw[row];++b.n[cell];++b.fit_n[mode];
    }
    for(int i=0;i<measured_cells;++i){
        require(b.n[i]==(i<640?4:64),"Unexpected fit-only cell occupancy");
        b.mean[i]/=b.n[i];
    }
    for(int row=0;row<int(raw.size());++row){
        if(!measured_fit(row))continue;
        const double e=double(raw[row])-b.mean[measured_cell(row)];
        b.rms[measured_mode(row)]+=e*e;
    }
    for(int mode=0;mode<2;++mode){
        require(b.fit_n[mode]==2560,"Bad fit-only task balance");
        b.rms[mode]=sqrt(b.rms[mode]/b.fit_n[mode]);
        require(std::isfinite(b.rms[mode])&&b.rms[mode]>0,"Fit-only residual RMS is zero or invalid");
    }
    return b;
}

struct MeasuredSignal {
    double rmse[2]{},bias[2]{},square[2]{},hold_share=0;
    int n[2]{};
    bool passed=false;
};
MeasuredSignal measured_signal(const std::vector<float>& raw,const MeasuredBaseline& b,bool held_out_only) {
    require(raw.size()==10240,"Measured signal got invalid row count");
    MeasuredSignal f;
    for(int row=0;row<int(raw.size());++row){
        if(held_out_only&&measured_fit(row))continue;
        const int mode=measured_mode(row);
        const double e=double(raw[row])-b.mean[measured_cell(row)];
        const double scaled=e/b.rms[mode];
        require(std::isfinite(e)&&std::isfinite(scaled),"Invalid raw or normalized residual");
        ++f.n[mode];f.bias[mode]+=e;f.rmse[mode]+=e*e;f.square[mode]+=scaled*scaled;
    }
    for(int mode=0;mode<2;++mode){
        require(f.n[mode]==(held_out_only?2560:5120),"Measured signal split or task count changed");
        f.bias[mode]/=f.n[mode];
        f.rmse[mode]=sqrt(f.rmse[mode]/f.n[mode]);
    }
    const double sum=f.square[0]+f.square[1];
    require(std::isfinite(sum)&&sum>0,"Measured signal energy is zero or invalid");
    f.hold_share=f.square[1]/sum;
    f.passed=f.hold_share>=.10&&f.hold_share<=.90;
    return f;
}

void measured_zero_fixture(const fs::path& root) {
    std::vector<float> raw(10240);
    for(int row=0;row<10240;++row){
        const int mode=measured_mode(row),cell=measured_cell(row);
        const float residual=(row%32/8)%2?1.f:-1.f;
        raw[row]=float(cell)+(mode?.5f:2.f)*residual;
    }
    const auto b=measured_fit_baseline(raw);
    for(int i=0;i<measured_cells;++i)require(b.mean[i]==i,"Task/pose/time baseline fixture error");
    require(b.rms[0]==2&&b.rms[1]==.5,"Fit residual normalizer fixture error");
    const auto signal=measured_signal(raw,b,true);
    require(signal.passed&&signal.hold_share==.5&&signal.bias[0]==0&&signal.bias[1]==0&&
            signal.rmse[0]==2&&signal.rmse[1]==.5,"Signal share fixture error");
    for(int row=0;row<10240;++row)if(!measured_fit(row))raw[row]+=100.f;
    const auto a=measured_fit_baseline(raw);
    require(a.mean==b.mean&&a.rms[0]==b.rms[0]&&a.rms[1]==b.rms[1],"Held-out rows altered frozen constants");
    std::ofstream out(root/"zero-step-measured-baseline.json");
    out<<"{\"passed\":true,\"mode_pose_elapsed_table\":true,\"fit_residual_rms\":true,\"held_out_exclusion\":true,\"fixed_squared_signal_share\":true,\"optimizer_steps\":0,\"physical_controls\":0}\n";
}

void save_measured_baseline(const fs::path& root,const MeasuredBaseline& b){
    cp::Writer out;
    const auto identity=cp::checksum(rest_protocol());
    out.value(uint32_t{0x314d4252});out.value(uint32_t{1});out.value(identity);
    out.array(b.mean);out.array(b.n);
    for(int mode=0;mode<2;++mode){out.value(b.rms[mode]);out.value(b.fit_n[mode]);}
    const auto path=root/"frozen-fit-baseline-v1.bin";
    require(!fs::exists(path),"Frozen fit baseline already exists");
    cp::write_atomic(path.string(),out.data);
    const auto stored=cp::read(path.string());cp::Reader verify{stored};
    require(verify.value<uint32_t>()==0x314d4252&&verify.value<uint32_t>()==1&&verify.value<uint64_t>()==identity,
            "Measured baseline binary identity error");
    require(verify.array<double>(measured_cells)==b.mean&&verify.array<int>(measured_cells)==b.n,
            "Measured baseline cell roundtrip error");
    for(int mode=0;mode<2;++mode)
        require(verify.value<double>()==b.rms[mode]&&verify.value<int>()==b.fit_n[mode],"Measured baseline scale roundtrip error");
    verify.finish();
}

void measured_actor_parity(Arm& a){
    auto& d=*a.data;auto& actor=*a.actor;const auto s=a.stream->value;
    const auto raw=download(d.obs.get(),rows*41,s);
    std::vector<float> x(size_t(rows)*40);
    for(int row=0;row<rows;++row)std::copy_n(raw.data()+row*41,40,x.data()+row*40);
    device_ptr<float> in(x.size());in.copy_from_host(x.data(),x.size());
    nn::GpuMlp old(old_spec(4),rows,seed);old.set_params(original40.actor.parameters);
    old.forward(in.get(),rows,s);actor.forward(d.obs.get(),rows,s);
    const auto before=download(old.output(),rows*4,s),after=download(actor.output(),rows*4,s);
    double maximum=0;
    for(size_t i=0;i<before.size();++i)maximum=std::max(maximum,fabs(double(before[i])-after[i]));
    require(maximum<=1e-6,"Actual old40/new41 actor parity failure");
    std::ofstream out(a.root/"phase-A-actor-parity.json");
    out<<std::setprecision(12)<<"{\"passed\":true,\"rows\":"<<rows<<",\"maximum_error\":"<<maximum
       <<",\"extra_physical_controls\":0,\"critic_forward_calls\":0}\n";
}

int run_measured_phase_a(const fs::path& root) {
    configure_rest(root,true);
    require(cp::read((root/"protocol.bin").string())==rest_protocol(),"Frozen measured baseline protocol differs");
    require(fs::create_directory(root/"run-started"),"Measured Phase A already attempted");
    started=std::chrono::steady_clock::now();
    Arm a;a.root=root;a.budget.seconds=120;a.budget.maximum_controls=21024;a.budget.maximum_substeps=210240;
    std::string fault,gate;RestAssessment baseline;
    const double launch=unix_now();
    try{
        a.stream=std::make_unique<Stream>();
        parity(a.budget,a.stream->value);status(root,"baseline",a.budget,336);
        a.before=evaluate(initial_actor(),root,"baseline",a.budget);
        baseline=evaluate_rest(initial_actor(),root/"baseline-rest",a.budget);
        fs::create_directory(root/"baseline-long");a.long_before=sustained(initial_actor(),root/"baseline-long",a.budget);
        require(a.long_before.safety_fault.empty()&&baseline.completed==6&&baseline.passed==0&&a.before.passed==4&&
                preserved_long_windows(root/"baseline-long")&&a.budget.controls==10784&&a.budget.substeps==107840,
                "Frozen baseline differs from reference");
        std::cout<<"MEASURED_BASELINE_CHECK_STARTED unix="<<std::setprecision(17)<<unix_now()<<" seed314159265; no optimization"<<std::endl;
        a.actor=std::make_unique<nn::GpuMlp>(spec(4),rows,seed);
        a.actor->set_params(resume_state->actor.parameters);
        a.data=std::make_unique<Rollout>(a.stream->value);a.batch=std::make_unique<Batches>();
        calibration_collect=true;
        auto& d=*a.data;const auto stream=a.stream->value;
        status(root,"measured_baseline_check",a.budget,336);
        collect_rollout_selected_value(336,*a.actor,nullptr,d,stream,root);
        rest_account(a);
        require(a.training_controls==10240&&a.budget.controls==21024&&a.budget.substeps==210240,
               "Measured Phase A physical count changed");
        cache_batch(a,336);
        const auto values=download(d.value.get(),rows,stream);
        require(std::all_of(values.begin(),values.end(),[](float x){return x==0.f;}),
                "Critic value present in measured baseline rollout");
        measured_actor_parity(a);
        verify_rest_batch(a,"check");
        require(a.actor->get_params()==resume_state->actor.parameters,"Baseline check changed actor weights");
        const auto raw=download(d.returns.get(),rows,stream);
        require(raw==verified_returns&&raw==verified_adv,"Critic-free GAE is not the raw measured return");
        const auto hold_returns=scale_stats(raw,true),swim_returns=scale_stats(raw,false);
        const auto fit=measured_fit_baseline(raw);const auto f=measured_signal(raw,fit,true);
        if(!reset_occupancy_passed)gate="check_hold_occupancy";
        else if(!f.passed)gate="check_scaled_signal_share";
        if(gate.empty())save_measured_baseline(root,fit);
        {
            std::ofstream table(root/"fit-baseline.csv");
            table<<std::setprecision(12)<<"mode,pose,elapsed_control,fit_rows,raw_mean\n";
            for(int i=0;i<measured_cells;++i){
                const bool hold=i>=640;const int local=hold?i-640:i,length=hold?20:320;
                table<<(hold?"hold":"swim")<<','<<(local/length?3:0)<<','<<local%length<<','<<fit.n[i]<<','<<fit.mean[i]<<'\n';
            }
            std::ofstream summary(root/"pre-PPO-report.json");
            summary<<std::setprecision(12)<<"{\"hold_raw_returns\":";print_scale(summary,hold_returns);
            summary<<",\"swim_raw_returns\":";print_scale(summary,swim_returns);
            summary<<",\"held_out_rows_per_mode\":"<<f.n[0]<<",\"fit_rows_per_mode\":"<<fit.fit_n[0]
                   <<",\"frozen_hold_residual_RMS\":"<<fit.rms[1]<<",\"frozen_swim_residual_RMS\":"<<fit.rms[0]
                   <<",\"held_out_hold_RMSE\":"<<f.rmse[1]<<",\"held_out_swim_RMSE\":"<<f.rmse[0]
                   <<",\"held_out_hold_bias\":"<<f.bias[1]<<",\"held_out_swim_bias\":"<<f.bias[0]
                   <<",\"held_out_hold_squared_scaled_signal\":"<<f.square[1]
                   <<",\"held_out_swim_squared_scaled_signal\":"<<f.square[0]
                   <<",\"held_out_hold_squared_scaled_signal_share\":"<<f.hold_share
                   <<",\"check_hold_occupancy_passed\":"<<(reset_occupancy_passed?"true":"false")
                   <<",\"check_scaled_signal_passed\":"<<(f.passed?"true":"false")
                   <<",\"ready_for_PPO\":"<<(gate.empty()?"true":"false")
                   <<",\"gate_failed\":"<<std::quoted(gate)<<",\"actor_optimizer_steps\":0,\"critic_optimizer_steps\":0"
                   <<",\"critic_forward_calls\":0,\"controls\":"<<a.budget.controls<<",\"substeps\":"<<a.budget.substeps<<"}\n";
        }
        wall_guard();
    }catch(const std::exception& e){
        fault=e.what();std::cerr<<"MEASURED_PHASE_A_STOPPED "<<fault<<std::endl;
    }
    std::ofstream out(root/"result.json");
    out<<std::setprecision(17)<<"{\"fault\":"<<std::quoted(fault)<<",\"gate_failed\":"<<std::quoted(gate)
       <<",\"launched_unix\":"<<launch<<",\"ended_unix\":"<<unix_now()<<",\"native_seconds\":"
       <<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()
       <<",\"controls\":"<<a.budget.controls<<",\"substeps\":"<<a.budget.substeps
       <<",\"PPO_updates\":0,\"actor_PPO_steps\":0,\"critic_optimizer_steps\":0,\"baseline_short\":"<<a.before.passed
       <<",\"baseline_rest\":"<<baseline.passed<<",\"baseline_rest_completed\":"<<baseline.completed
       <<",\"baseline_long_windows\":"<<a.long_before.windows_passed<<",\"candidate_evaluated\":false,\"runtime_adopted\":false"
       <<",\"retries\":0,\"closed\":true}\n";
    out.close();
    status(root,fault.empty()?(gate.empty()?"check_passed_no_PPO":"gate_stopped"):"failed",a.budget,336);
    std::cout<<"MEASURED_PHASE_A_CLOSED PPO=0 controls="<<a.budget.controls<<" gate="<<gate<<" fault="<<fault<<std::endl;
    return fault.empty()?0:3;
}
