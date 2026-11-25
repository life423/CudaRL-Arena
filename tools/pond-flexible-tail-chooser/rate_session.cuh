#pragma once

// The rate chooser reuses train.cu's CUDA batch collector, categorical PPO,
// optimized MLP/GEMM and GPU Adam. Only its task/runtime and saved schema differ.
struct RateIdentity { uint64_t gait=0, motor=0, starts=0; tail::steering::Kind kind{};
    tail::steering::Config config{}; };

void rate_header(cudarl::checkpoint::Writer& out, const RateIdentity& id) {
    out.value<uint32_t>(0x52435446); out.value<uint32_t>(1); // FTCR
    const std::string task(CUDARL_RATE_TASK_ID);
    out.array(std::vector<char>(task.begin(),task.end()));
    for (int x : {worlds,decisions,channels,actions,batch_size,epochs,macro_controls,max_updates}) out.value(x);
    out.value<uint32_t>(sizeof(Lane)); out.value<uint32_t>(sizeof(Counts));
    out.value(id.gait);out.value(id.motor);out.value(id.starts);out.value(id.kind);out.value(id.config);
    for(float x : {gamma,lambda,max_kl,loss.clip,loss.ent_coef,loss.vf_coef,.0003f,.001f})out.value(x);
    out.value(larger_pond);
}
std::vector<uint8_t> rate_payload(Rollout& roll, nn::GpuMlp& actor, nn::GpuMlp& critic,
    nn::GpuAdam& aa, nn::GpuAdam& ca, const Ledger& l, const RateIdentity& id, cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    cudarl::checkpoint::Writer out;rate_header(out,id);
    out.value(l.completed);out.value(l.actor_steps);out.value(l.critic_steps);
    out.array(actor.get_params());out.array(critic.get_params());aa.save(out);ca.save(out);
    std::vector<Lane> lanes(worlds);std::vector<Counts> counts(worlds);
    roll.lanes.copy_to_host(lanes.data(),lanes.size());roll.counts.copy_to_host(counts.data(),counts.size());
    out.array(lanes);out.array(counts);return std::move(out.data);
}
void rate_restore(const std::vector<uint8_t>& bytes, Rollout& roll, nn::GpuMlp& actor, nn::GpuMlp& critic,
    nn::GpuAdam& aa, nn::GpuAdam& ca, Ledger& l,const RateIdentity& id,cudaStream_t stream) {
    cudarl::checkpoint::Writer header;rate_header(header,id);
    if(bytes.size()<header.data.size() || !std::equal(header.data.begin(),header.data.end(),bytes.begin()))
        throw std::runtime_error("Rate chooser task/input/schema identity mismatch");
    cudarl::checkpoint::Reader in{bytes};in.position=header.data.size();
    const auto completed=in.value<int>(),as=in.value<int>(),cs=in.value<int>();
    if(completed<0||completed>max_updates||as<0||as!=cs)throw std::runtime_error("Invalid rate chooser ages");
    auto a=in.array<float>(actor.num_params()),c=in.array<float>(critic.num_params());
    chooser::validate_weights(a);
    if(c.size()!=size_t(critic.num_params()) || !std::all_of(c.begin(),c.end(),cudarl::physics::finite_number))
        throw std::runtime_error("Invalid rate critic");
    aa.load(in);ca.load(in);
    if(aa.steps()!=as||ca.steps()!=cs||aa.config().lr!=.0003f||ca.config().lr!=.001f)
        throw std::runtime_error("Rate optimizer identity mismatch");
    auto lanes=in.array<Lane>(worlds);auto counts=in.array<Counts>(worlds);in.finish();
    if(lanes.size()!=worlds||counts.size()!=worlds)throw std::runtime_error("Invalid rate worlds");
    for(const auto& lane:lanes)if(!lane.needs_start &&
        (!cudarl::physics::finite_body(lane.state.motor.navigation.motion.body)||lane.state.fault||
        lane.state.motor_kind!=id.kind||!tail::steering::valid_config(lane.state.config)))
        throw std::runtime_error("Invalid restored rate world");
    actor.set_params(a);critic.set_params(c);
    roll.lanes.copy_from_host(lanes.data(),lanes.size());roll.counts.copy_from_host(counts.data(),counts.size());
    l.completed=l.started=completed;l.actor_steps=as;l.critic_steps=cs;count_work(roll,l,stream);
}
void rate_publish(const fs::path& dir, Rollout& roll,nn::GpuMlp& a,nn::GpuMlp& c,
    nn::GpuAdam& aa,nn::GpuAdam& ca,Ledger& l,const RateIdentity& id,cudaStream_t stream) {
    const auto launch=Clock::now();
    auto bytes=rate_payload(roll,a,c,aa,ca,l,id,stream);
    const auto path=dir/"checkpoints"/("session-"+std::to_string(l.completed)+".bin");
    if(fs::exists(path))throw std::runtime_error("Rate checkpoint is immutable");
    cudarl::checkpoint::write_atomic(path.u8string(),bytes);
    cudarl::checkpoint::write_atomic((dir/"latest.checkpoint").u8string(),bytes);
    chooser::write_weights((dir/("actor-"+std::to_string(l.completed)+".bin")).u8string(),a.get_params());
    l.evidence_ms+=elapsed(launch)*1000.;
}
void rate_result(const fs::path& dir,const Ledger& l,bool verification) {
    const auto& c=l.counts;std::ofstream out(dir/"result.json");out.exceptions(std::ios::badbit|std::ios::failbit);
    uint64_t samples=0;for(auto x:c.samples)samples+=x;
    out<<std::setprecision(12)<<"{\"schema\":1,\"task\":\""<<CUDARL_RATE_TASK_ID
       <<"\",\"worlds\":"<<worlds<<",\"updates_completed\":"<<l.completed
       <<",\"samples\":"<<samples<<",\"executed_samples\":"<<l.executed_decisions
       <<",\"attempted_substeps\":"<<c.attempted+c.warmup_attempted
       <<",\"accepted_substeps\":"<<c.accepted+c.warmup_accepted
       <<",\"executed_attempted_substeps\":"<<l.executed_attempted
       <<",\"executed_accepted_substeps\":"<<l.executed_accepted
       <<",\"actor_steps\":"<<l.actor_steps<<",\"critic_steps\":"<<l.critic_steps
       <<",\"executed_actor_steps\":"<<l.executed_actor_steps
       <<",\"executed_critic_steps\":"<<l.executed_critic_steps
       <<",\"clearance_stops\":"<<c.stops<<",\"posture_stops\":"<<c.posture_stops
       <<",\"starts\":"<<c.starts<<",\"gpu_physics_ms\":"<<l.gpu_physics_ms
       <<",\"gpu_inference_ms\":"<<l.gpu_inference_ms<<",\"gpu_ppo_ms\":"<<l.gpu_optimization_ms
       <<",\"transfers_wait_ms\":"<<l.transfers_ms<<",\"evidence_ms\":"<<l.evidence_ms
       <<",\"wall_s\":"<<l.wall_s<<",\"stage\":\""<<l.stage
       <<"\",\"verification\":"<<(verification?"true":"false")<<",\"action_counts\":[";
    for(int i=0;i<actions;++i){if(i)out<<',';out<<c.samples[i];}out<<"]";
#if defined(CUDARL_RELIABLE_CHOOSER)
    out<<",\"curriculum_starts\":["<<c.curriculum_starts[0]<<','<<c.curriculum_starts[1]<<','<<c.curriculum_starts[2]
       <<"],\"curriculum_promotions\":"<<c.promotions<<",\"curriculum_demotions\":"<<c.demotions;
#endif
    out<<"}\n";
}
int rate_main(int argc,char** argv) {
    fs::path output;
    try {
        if(argc!=9)throw std::runtime_error("Usage: rate_chooser new|verify|resume GAIT MOTOR STARTS OUT UPDATES WALL MATERIAL(0|1)");
        const std::string mode=argv[1];const bool verify=mode=="verify",resume=mode=="resume";
        if(mode!="new"&&!verify&&!resume)throw std::runtime_error("Invalid rate session mode");
        output=fs::u8path(argv[5]);const int target=std::stoi(argv[6]);const double wall=std::stod(argv[7]);
        const int material=std::stoi(argv[8]);
        if(target<1||target>max_updates||!std::isfinite(wall)||wall<=0||wall>1200||material<0||material>1)
            throw std::runtime_error("Rate study exceeds finite budget");
        if(!resume){if(!fs::create_directory(output))throw std::runtime_error("Rate output must be unused");fs::create_directory(output/"checkpoints");}
        cudarl::checkpoint::Lease lease((output/"session-owner").u8string());
        auto fitted=tail::read_navigation_weights(argv[2],tail::NavigationMode::Gait);
        auto motor=tail::steering::read_model(argv[3]);auto initial=tail::read_navigation_worlds(argv[4]);
        tail::fast::validate_frozen_gait(fitted);tail::steering::validate_model(motor);
        if(motor.kind<tail::steering::Kind::ProgramScalar || initial.size()!=256 ||
            motor.material!=static_cast<tail::Material>(material))
            throw std::runtime_error("Rate chooser needs declared program motor and training archive");
#if defined(CUDARL_RELIABLE_CHOOSER)
        for(int action=0;action<actions;++action) {
            const float expected[5]{-8,-4,0,4,8};
            if(chooser::action_rate_deg_s(action)!=expected[action%5])throw std::runtime_error("Reliable action mapping differs");
        }
        int checked_starts=0;
        for(int level=0;level<3;++level)for(int lane=0;lane<worlds;++lane) {
            const auto draw=start_episode_seed(lane,0);
            const auto row=start_source_row(draw,int(initial.size()));
            cudarl::physics::V com{};float yaw=0;bank_training::ApproachEvidence proof{};
            if(!bank_training::approach_start(lane,0,initial[row],larger_pond,com,yaw,&proof,level))
                throw std::runtime_error("Reliable curriculum fixture failed whole-body clearance at level "+std::to_string(level)+" lane "+std::to_string(lane));
            ++checked_starts;
        }
        std::ofstream preflight(output/"curriculum-preflight.json");
        preflight<<"{\"pass\":true,\"physics_steps\":0,\"actions\":15,\"geometry_starts\":"<<checked_starts<<"}\n";
#endif
        std::vector<float> packed=motor.weights;packed.insert(packed.end(),motor.frozen_skip.begin(),motor.frozen_skip.end());
        RateIdentity id{cudarl::checkpoint::checksum(cudarl::checkpoint::read(argv[2])),
            cudarl::checkpoint::checksum(cudarl::checkpoint::read(argv[3])),
            cudarl::checkpoint::checksum(cudarl::checkpoint::read(argv[4])),motor.kind,{}};
        id.config.study=tail::steering::Study::Rate;id.config.material=static_cast<tail::Material>(material);
        Stream stream;Rollout roll;Batch batch;Events events;
        CUDA_CHECK_THROW(cudaMemcpyToSymbol(session_motor_kind,&id.kind,sizeof(id.kind)));
        CUDA_CHECK_THROW(cudaMemcpyToSymbol(session_motor_config,&id.config,sizeof(id.config)));
        device_ptr<tail::NavigationWorld> starts(initial.size());device_ptr<float> gait(fitted.size()),offset(packed.size());
        starts.copy_from_host(initial.data(),initial.size());gait.copy_from_host(fitted.data(),fitted.size());
        offset.copy_from_host(packed.data(),packed.size());
        nn::GpuMlp actor(chooser::actor_spec(),batch_size,seed);
        nn::MlpSpec critic_spec{{channels,32,1},{nn::Activation::Tanh,nn::Activation::Identity}};
        nn::GpuMlp critic(critic_spec,batch_size,seed+1);
        actor.set_params(nn::orthogonal_params(actor.spec(),seed,{1.41421356f,0.f}));
        critic.set_params(nn::orthogonal_params(critic_spec,seed+1,{1.41421356f,1.f}));
        nn::AdamConfig ac,cc;ac.lr=.0003f;cc.lr=.001f;ac.max_grad_norm=cc.max_grad_norm=1.f;
        nn::GpuAdam aa(actor.num_params(),ac),ca(critic.num_params(),cc);
        Ledger ledger;ledger.bank_curriculum=true;
        ledger.execution_physical_limit=verify?uint64_t(3)*rows*macro_controls*tail::turn_substeps_per_control*2:max_substeps;
        const int empty=INT_MAX;CUDA_CHECK_THROW(cudaMemcpyAsync(roll.first_fault.get(),&empty,sizeof(empty),cudaMemcpyHostToDevice,stream.value));
        zero(roll.fault_code.get(),roll.fault_code.size_bytes(),stream.value);
        zero(roll.zero.get(),roll.zero.size_bytes(),stream.value);
        zero(roll.lanes.get(),roll.lanes.size_bytes(),stream.value);zero(roll.counts.get(),roll.counts.size_bytes(),stream.value);
        initialize_lanes<<<worlds/lane_threads,lane_threads,0,stream.value>>>(roll.lanes.get());check_launch();
        double prior_used=0;
        if(resume){
            const auto restored=cudarl::checkpoint::read((output/"latest.checkpoint").u8string());
            const auto budget=cudarl::checkpoint::read((output/"budget.checkpoint").u8string());
            cudarl::checkpoint::Reader b{budget};
            if(b.value<uint32_t>()!=1 || b.value<uint64_t>()!=cudarl::checkpoint::checksum(restored) ||
                b.value<int>()!=target || b.value<double>()!=wall)throw std::runtime_error("Rate resume budget differs");
            prior_used=b.value<double>();b.finish();
            if(!std::isfinite(prior_used)||prior_used<0||prior_used>=wall)throw std::runtime_error("Rate wall budget spent");
            rate_restore(restored,roll,actor,critic,aa,ca,ledger,id,stream.value);
        }else rate_publish(output,roll,actor,critic,aa,ca,ledger,id,stream.value);
        auto save_budget=[&](double used){
            cudarl::checkpoint::Writer b;b.value<uint32_t>(1);
            b.value(cudarl::checkpoint::checksum(cudarl::checkpoint::read((output/"latest.checkpoint").u8string())));
            b.value(target);b.value(wall);b.value(used);cudarl::checkpoint::write_atomic((output/"budget.checkpoint").u8string(),b.data);
        };
        if(!resume)save_budget(0);
        const auto launch=Clock::now();std::vector<uint8_t> one,expected;
        bool checking=false;int executed_updates=0;
        std::ofstream curve(output/(resume?"resume-updates.csv":"updates.csv"));
        curve<<"update,reward,kl,policy_loss,value_loss,entropy,explained_variance,stops,samples,wall_s\n";
        try{while(ledger.completed<target && prior_used+elapsed(launch)<wall && !fs::exists(output/"STOP")) {
            const int update=ledger.completed+1;
            if(!collect(roll,actor,critic,starts,gait,offset,ledger,update,stream.value,events,launch))break;
            const float reward=finite_mean(roll,roll.reward.get(),rows,stream.value);
            CUDA_CHECK_THROW(cudaEventRecord(events.start,stream.value));
            const float kl=optimize(roll,batch,actor,critic,aa,ca,ledger,update,stream.value);
            CUDA_CHECK_THROW(cudaEventRecord(events.end,stream.value));CUDA_CHECK_THROW(cudaEventSynchronize(events.end));
            float ms=0;CUDA_CHECK_THROW(cudaEventElapsedTime(&ms,events.start,events.end));ledger.gpu_optimization_ms+=ms;
            ledger.completed=update;++executed_updates;
            if(verify){auto payload=rate_payload(roll,actor,critic,aa,ca,ledger,id,stream.value);
                if(update==1)one=payload;
                else if(!checking){expected=payload;rate_restore(one,roll,actor,critic,aa,ca,ledger,id,stream.value);checking=true;continue;}
                else {if(payload!=expected)throw std::runtime_error("Rate full resume diverged");
                    std::ofstream receipt(output/"verification.json");receipt<<"{\"pass\":true,\"physical_updates_executed\":"<<executed_updates
                        <<",\"sampled_actions_executed\":"<<ledger.executed_decisions<<",\"attempted_substeps\":"<<ledger.executed_attempted
                        <<",\"accepted_substeps\":"<<ledger.executed_accepted<<",\"payload_checksum\":"<<cudarl::checkpoint::checksum(payload)<<"}\n";
                }
            }
            rate_publish(output,roll,actor,critic,aa,ca,ledger,id,stream.value);
            ledger.wall_s=prior_used+elapsed(launch);save_budget(ledger.wall_s);ledger.stage=ledger.completed==target?"complete":"checkpoint";
            uint64_t samples=0;for(auto x:ledger.counts.samples)samples+=x;
            curve<<update<<','<<reward<<','<<kl<<','<<ledger.policy_loss<<','<<ledger.value_loss<<','<<ledger.entropy<<','
                <<ledger.explained_variance<<','<<ledger.counts.stops<<','<<samples<<','<<ledger.wall_s<<'\n';curve.flush();
            rate_result(output,ledger,verify);
        }}catch(...){
            ledger.wall_s=prior_used+elapsed(launch);ledger.stage="fault";
            try{count_work(roll,ledger,stream.value);auto rejected=rate_payload(roll,actor,critic,aa,ca,ledger,id,stream.value);
                cudarl::checkpoint::write_atomic((output/("rejected-update-"+std::to_string(ledger.started)+".bin")).u8string(),rejected);
                rate_result(output,ledger,verify);
            }catch(const std::exception& evidence_error){std::ofstream incomplete(output/"failure-evidence-incomplete.txt");incomplete<<evidence_error.what()<<'\n';}
            throw;
        }
        if(verify&&executed_updates!=3)throw std::runtime_error("Rate resume verification incomplete");
        ledger.wall_s=prior_used+elapsed(launch);rate_result(output,ledger,verify);
        return 0;
    }catch(const std::exception& error){
        if(!output.empty()&&fs::exists(output)){std::ofstream fault(output/"failure.txt");fault<<error.what()<<'\n';}
        std::cerr<<error.what()<<'\n';return 2;
    }
}
