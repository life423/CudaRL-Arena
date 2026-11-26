#pragma once

#if defined(CUDARL_MOVING_STARTS)
#include "moving_placement_audit.cuh"
#endif

// The rate chooser reuses train.cu's CUDA batch collector, categorical PPO,
// optimized MLP/GEMM and GPU Adam. Only its task/runtime and saved schema differ.
struct RateIdentity { uint64_t gait=0, motor=0, starts=0; tail::steering::Kind kind{};
    tail::steering::Config config{}; };
#if defined(CUDARL_EPISODIC_BANK)
inline constexpr bool episodic_session=true;
#else
inline constexpr bool episodic_session=false;
#endif

void rate_header(cudarl::checkpoint::Writer& out, const RateIdentity& id, const char* task_id=CUDARL_RATE_TASK_ID,int input_width=channels,bool episodic=episodic_session) {
    out.value<uint32_t>(0x52435446); out.value<uint32_t>(episodic?4:(input_width==39?3:(input_width==35?2:1))); // FTCR
    const std::string task(task_id);
    out.array(std::vector<char>(task.begin(),task.end()));
    for (int x : {worlds,decisions,input_width,actions,batch_size,epochs,macro_controls,max_updates}) out.value(x);
    out.value<uint32_t>(sizeof(Lane)); out.value<uint32_t>(sizeof(Counts));
    out.value(id.gait);out.value(id.motor);out.value(id.starts);out.value(id.kind);out.value(id.config);
    for(float x : {gamma,lambda,max_kl,loss.clip,loss.ent_coef,loss.vf_coef,.0003f,.001f})out.value(x);
    out.value(larger_pond);
#if defined(CUDARL_EPISODIC_BANK)
    if(episodic){out.value(training_episode_controls);out.value<uint32_t>(sizeof(EpisodeInfo));}
#endif
}
void rate_save_adam(cudarl::checkpoint::Writer& out,const nn::GpuAdam& adam,int width){
    if(width==channels){adam.save(out);return;}
    cudarl::checkpoint::Writer raw;adam.save(raw);cudarl::checkpoint::Reader in{raw.data};
    const auto config=in.value<nn::AdamConfig>();const auto steps=in.value<int>();
    const auto m=in.array<float>(10000),v=in.array<float>(10000);in.finish();
    out.value(config);out.value(steps);
    out.array(chooser::resize_input_parameters(m,channels,width));out.array(chooser::resize_input_parameters(v,channels,width));
}
void rate_load_adam(cudarl::checkpoint::Reader& in,nn::GpuAdam& adam,int width,int count){
    if(width==channels){adam.load(in);return;}
    const auto config=in.value<nn::AdamConfig>();const auto steps=in.value<int>();
    auto m=in.array<float>(count),v=in.array<float>(count);
    if(m.size()!=size_t(count)||v.size()!=size_t(count))throw std::runtime_error("Parent Adam dimensions differ");
    cudarl::checkpoint::Writer expanded;expanded.value(config);expanded.value(steps);
    expanded.array(chooser::resize_input_parameters(m,width,channels));expanded.array(chooser::resize_input_parameters(v,width,channels));
    cudarl::checkpoint::Reader migrated{expanded.data};adam.load(migrated);migrated.finish();
}
std::vector<uint8_t> rate_payload(Rollout& roll, nn::GpuMlp& actor, nn::GpuMlp& critic,
    nn::GpuAdam& aa, nn::GpuAdam& ca, const Ledger& l, const RateIdentity& id, cudaStream_t stream,
    const char* task_id=CUDARL_RATE_TASK_ID,int input_width=channels,bool episodic=episodic_session) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    cudarl::checkpoint::Writer out;rate_header(out,id,task_id,input_width,episodic);
    out.value(l.completed);out.value(l.actor_steps);out.value(l.critic_steps);
    out.array(chooser::resize_input_parameters(actor.get_params(),channels,input_width));
    out.array(chooser::resize_input_parameters(critic.get_params(),channels,input_width));
    rate_save_adam(out,aa,input_width);rate_save_adam(out,ca,input_width);
    if(input_width>=35){out.value(aa.fresh_steps());out.value(ca.fresh_steps());}
    if(input_width==39){out.value(aa.fresh_steps(1));out.value(ca.fresh_steps(1));}
    std::vector<Lane> lanes(worlds);std::vector<Counts> counts(worlds);
    roll.lanes.copy_to_host(lanes.data(),lanes.size());roll.counts.copy_to_host(counts.data(),counts.size());
    out.array(lanes);out.array(counts);
#if defined(CUDARL_EPISODIC_BANK)
    if(episodic){std::vector<EpisodeInfo> episodes(worlds);roll.episodes.copy_to_host(episodes.data(),episodes.size());out.array(episodes);}
#endif
    return std::move(out.data);
}
void rate_restore(const std::vector<uint8_t>& bytes, Rollout& roll, nn::GpuMlp& actor, nn::GpuMlp& critic,
    nn::GpuAdam& aa, nn::GpuAdam& ca, Ledger& l,const RateIdentity& id,cudaStream_t stream,
    const char* task_id=CUDARL_RATE_TASK_ID,int input_width=channels,bool episodic=episodic_session) {
    cudarl::checkpoint::Writer header;rate_header(header,id,task_id,input_width,episodic);
    if(bytes.size()<header.data.size() || !std::equal(header.data.begin(),header.data.end(),bytes.begin()))
        throw std::runtime_error("Rate chooser task/input/schema identity mismatch");
    cudarl::checkpoint::Reader in{bytes};in.position=header.data.size();
    const auto completed=in.value<int>(),as=in.value<int>(),cs=in.value<int>();
    if(completed<0||completed>max_updates||as<0||as!=cs)throw std::runtime_error("Invalid rate chooser ages");
    auto a=in.array<float>(actor.num_params()),c=in.array<float>(critic.num_params());
    a=chooser::resize_input_parameters(a,input_width,channels);c=chooser::resize_input_parameters(c,input_width,channels);
    chooser::validate_weights(a);
    if(c.size()!=size_t(critic.num_params()) || !std::all_of(c.begin(),c.end(),cudarl::physics::finite_number))
        throw std::runtime_error("Invalid rate critic");
    rate_load_adam(in,aa,input_width,actor.num_params()-32*(channels-input_width));
    rate_load_adam(in,ca,input_width,critic.num_params()-32*(channels-input_width));
    if(channels>=35){
        const int actor_age=input_width>=35?in.value<int>():0;
        const int critic_age=input_width>=35?in.value<int>():0;
        aa.set_fresh_column(channels,32,actor_age,3,32);ca.set_fresh_column(channels,32,critic_age,3,32);
        if(channels==39){const int actor_curve_age=input_width==39?in.value<int>():0,critic_curve_age=input_width==39?in.value<int>():0;
            aa.set_secondary_fresh_column(channels,32,actor_curve_age,4,35);ca.set_secondary_fresh_column(channels,32,critic_curve_age,4,35);}
    }
    if(aa.steps()!=as||ca.steps()!=cs||aa.config().lr!=.0003f||ca.config().lr!=.001f)
        throw std::runtime_error("Rate optimizer identity mismatch");
    auto lanes=in.array<Lane>(worlds);auto counts=in.array<Counts>(worlds);
#if defined(CUDARL_EPISODIC_BANK)
    if(episodic){auto episodes=in.array<EpisodeInfo>(worlds);if(episodes.size()!=worlds)throw std::runtime_error("Invalid episode clocks");
        for(int i=0;i<worlds;++i)if(episodes[i].first_decision_control<0||episodes[i].first_decision_control>lanes.at(i).state.motor.navigation.motion.controls)
            throw std::runtime_error("Invalid saved episode age");
        roll.episodes.copy_from_host(episodes.data(),episodes.size());
    }else zero(roll.episodes.get(),roll.episodes.size_bytes(),stream);
#endif
    in.finish();
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
#if defined(CUDARL_THREE_DEPTHS)
// Build independent practice starts from training poses. Translate the complete
// position history and goals together so the first sample has its original velocity.
std::vector<tail::NavigationWorld> depth_practice_starts(const std::vector<tail::NavigationWorld>& originals) {
    std::vector<tail::NavigationWorld> all;
    all.reserve(originals.size()*tail::distant_map::depth_count);
    for (const float y : tail::distant_map::depths) {
        for (auto start : originals) {
            if (start.leg_count<1 || start.active_leg!=0 || start.motion.controls!=0)
                throw std::runtime_error("Expected a fresh training pose");
            const float dy=y-start.legs[0].goal_m.y;
            start.motion.body.position.y+=dy/cudarl::physics::scale;
            for(auto& sample:start.motion.history) sample.y+=dy;
            for(int leg=0;leg<start.leg_count;++leg) start.legs[leg].goal_m.y+=dy;
            tail::prepare_navigation(start);
            if(std::abs(tail::snapshot(start.motion.body).com_m.y-y)>1.e-5f)
                throw std::runtime_error("Translated practice body does not match the requested depth");
            all.push_back(start);
        }
    }
    return all;
}
#endif

void rate_result(const fs::path& dir,const Ledger& l,bool verification) {
    const auto& c=l.counts;std::ofstream out(dir/"result.json");out.exceptions(std::ios::badbit|std::ios::failbit);
    uint64_t samples=0;for(auto x:c.samples)samples+=x;
    out<<std::setprecision(12)<<"{\"schema\":1,\"task\":\""<<CUDARL_RATE_TASK_ID
       <<"\",\"worlds\":"<<worlds<<",\"inputs\":"<<channels<<",\"observation_mode\":"<<chooser::default_observation_mode<<",\"updates_completed\":"<<l.completed
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
#if defined(CUDARL_THREE_DEPTHS)
    out<<",\"depth_samples\":[";
    for(int i=0;i<tail::distant_map::depth_count;++i) {
        if(i)out<<',';
        out<<"{\"mm\":"<<std::lround(-1000*tail::distant_map::depth_at(i))<<",\"transitions\":"<<l.samples_at_depth[i]<<'}';
    }
    out<<']';
#endif
#if defined(CUDARL_MOVING_STARTS)
    out<<",\"decisions_by_depth_and_practice\":[";
    for(int d=0;d<3;++d){
        if(d)out<<',';
        const auto& g=l.samples_by_practice[d];
        out<<"{\"mm\":"<<std::lround(-1000*tail::distant_map::depth_at(d))
           <<",\"short_moving\":"<<g[0]<<",\"medium_moving\":"<<g[1]<<",\"long_moving\":"<<g[2]
           <<",\"tangent\":"<<g[3]<<",\"interior\":"<<g[4]<<'}';
    }
    out<<"],\"moving_limit_s\":48,\"tangent_limit_s\":24,\"interior_time_limit\":false";
#endif
#if defined(CUDARL_EPISODIC_BANK)
    out<<",\"training_episode_s\":"<<training_episode_controls*tail::motor_control_s<<",\"training_time_limits\":"<<l.time_limits<<",\"near_decisions\":"<<l.near_decisions<<",\"episode_decisions\":"<<l.episode_decisions;
#endif
#if defined(CUDARL_RELIABLE_CHOOSER)
    out<<",\"curriculum_starts\":["<<c.curriculum_starts[0]<<','<<c.curriculum_starts[1]<<','<<c.curriculum_starts[2]
       <<"],\"curriculum_promotions\":"<<c.promotions<<",\"curriculum_demotions\":"<<c.demotions;
#endif
    out<<"}\n";
}
int rate_main(int argc,char** argv) {
    fs::path output;
    try {
        if(argc!=9&&argc!=11&&argc!=12)throw std::runtime_error("Usage: rate_chooser new|verify|resume|fork|verify-fork GAIT MOTOR STARTS OUT UPDATES WALL MATERIAL [PARENT_CHECKPOINT PARENT_TASK_ID [PARENT_INPUTS]]");
        const std::string mode=argv[1];const bool verify=mode=="verify"||mode=="verify-fork",resume=mode=="resume";
        const bool inspect=mode=="inspect-fork";
        const bool fork=mode=="fork"||mode=="verify-fork"||inspect;
        if((mode!="new"&&!verify&&!resume&&!fork)||(fork!=(argc>=11)))throw std::runtime_error("Invalid rate session mode/parent arguments");
        const int parent_inputs=argc==12?std::stoi(argv[11]):32;
        if((parent_inputs!=32&&parent_inputs!=35&&parent_inputs!=39)||parent_inputs>channels)throw std::runtime_error("Unsupported explicit parent input width");
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
#if defined(CUDARL_THREE_DEPTHS)
        initial=depth_practice_starts(initial);
#endif
#if defined(CUDARL_RELIABLE_CHOOSER)
        for(int action=0;action<actions;++action) {
            const float expected[5]{-8,-4,0,4,8};
            if(chooser::action_rate_deg_s(action)!=expected[action%5])throw std::runtime_error("Reliable action mapping differs");
        }
        int checked_starts=0;
#if defined(CUDARL_OPEN_WATER)
        int by_depth_cohort[3][3]{};
        double min_start_speed=1.,max_start_speed=0.;
#endif
        for(int level=0;level<3;++level)for(int lane=0;lane<worlds;++lane) {
#if defined(CUDARL_OPEN_WATER)
            const auto draw=start_episode_seed(lane,level);
#else
            const auto draw=start_episode_seed(lane,0);
#endif
#if defined(CUDARL_THREE_DEPTHS)
            const auto row=depth_practice_row(lane,draw,int(initial.size()));
#else
            const auto row=start_source_row(draw,int(initial.size()));
#endif
            cudarl::physics::V com{};float yaw=0;bank_training::ApproachEvidence proof{};
#if defined(CUDARL_OPEN_WATER)
            const auto group=practice_for(lane);
            if((row%256)<12 || int(row/256)!=practice_depth_index(lane))throw std::runtime_error("Invalid open-water training source");
            if(group==Practice::Moving){
                if(!bank_training::moving_start(lane,level,initial[row],larger_pond,com,yaw,proof))throw std::runtime_error("Moving approach preflight failed");
            }else if(group==Practice::Tangent){
                if(!bank_training::close_start(lane,level,initial[row],larger_pond,com,yaw,(lane/8)%2==0))throw std::runtime_error("Tangent approach preflight failed");
            }else place_interior(draw,larger_pond,com,yaw);
            tail::chooser::World placed{};
            tail::chooser::initialize(placed,initial[row],com,yaw);
            const auto speed=cudarl::physics::norm(tail::chooser::horizontal(placed.pose.com_velocity_m_s));
            if(placed.fault || !(speed>.003f) || !(tail::submerged_gap_m(larger_pond,placed.navigation.motion.body,placed.pose)>0.))
                throw std::runtime_error("Invalid moving initial body");
            if(group==Practice::Moving){min_start_speed=std::min(min_start_speed,double(speed));max_start_speed=std::max(max_start_speed,double(speed));}
            ++by_depth_cohort[practice_depth_index(lane)][int(group)];
            ++checked_starts;continue;
#elif defined(CUDARL_NORTH_BANK)
            if(north_practice_lane(lane)){
                if(row%256==8 || row%256==9 ||
                   !bank_training::close_start(lane,level,initial[row],larger_pond,com,yaw,true) ||
                   com.z<=0.f || fabsf(com.x)<.01f)
                    throw std::runtime_error("North practice start entered the held-out sector or failed clearance");
                ++checked_starts;continue;
            }
#endif
            if(!bank_training::approach_start(lane,0,initial[row],larger_pond,com,yaw,&proof,level))
                throw std::runtime_error("Reliable curriculum fixture failed whole-body clearance at level "+std::to_string(level)+" lane "+std::to_string(lane));
            ++checked_starts;
        }
        std::ofstream preflight(output/"curriculum-preflight.json");
        preflight<<"{\"pass\":true,\"physics_steps\":0,\"actions\":15,\"geometry_starts\":"<<checked_starts;
#if defined(CUDARL_OPEN_WATER)
        preflight<<",\"minimum_moving_spawn_COM_mm_s\":"<<min_start_speed*1000<<",\"maximum_moving_spawn_COM_mm_s\":"<<max_start_speed*1000
                 <<",\"first_allowed_training_pose\":12,\"depth_cohort_placements\":[";
        for(int d=0;d<3;++d){if(d)preflight<<',';preflight<<'[';
            for(int g=0;g<3;++g){if(by_depth_cohort[d][g]==0)throw std::runtime_error("Missing training cohort/depth");
                if(g)preflight<<',';preflight<<by_depth_cohort[d][g];}preflight<<']';}
        preflight<<']';
#endif
        preflight<<"}\n";
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
#if defined(CUDARL_MOVING_STARTS)
        if(inspect) verify_moving_geometry(initial,starts.get(),stream.value,output);
#endif
#if defined(CUDARL_CLOSE_SHORE)
        device_ptr<ClosePlacement> device_placements(worlds);std::vector<ClosePlacement> placements(worlds);
        float position_difference=0,yaw_difference=0;int close_checked=0;
        for(int episode=0;episode<4;++episode){
            close_placement_probe<<<worlds/lane_threads,lane_threads,0,stream.value>>>(starts.get(),int(initial.size()),episode,larger_pond,device_placements.get());
            check_launch();CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));device_placements.copy_to_host(placements.data(),placements.size());
            for(int lane=0;lane<worlds;++lane){cudarl::physics::V position{};float yaw=0;
                const auto row=start_source_row(start_episode_seed(lane,episode),int(initial.size()));
                if(row==8||row==9||!bank_training::close_start(lane,episode,initial[row],larger_pond,position,yaw)||!placements[lane].valid)
                    throw std::runtime_error("Close curriculum placement gate failed");
                position_difference=std::max(position_difference,cudarl::physics::norm(position-placements[lane].position));
                yaw_difference=std::max(yaw_difference,std::abs(yaw-placements[lane].yaw));++close_checked;
            }
        }
        if(position_difference>1.e-6f||yaw_difference>1.e-5f)throw std::runtime_error("Close placement CPU/CUDA parity failed");
        std::ofstream close_check(output/"close-placement-check.json");close_check<<std::setprecision(10)
            <<"{\"pass\":true,\"placements\":"<<close_checked<<",\"max_position_difference_m\":"<<position_difference
            <<",\"max_yaw_difference_rad\":"<<yaw_difference<<",\"excluded_rows\":[8,9],\"physics_steps\":0,\"optimizer_updates\":0}\n";
#endif
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
#if defined(CUDARL_EPISODIC_BANK)
        zero(roll.episodes.get(),roll.episodes.size_bytes(),stream.value);
        {
        const std::vector<float> rewards{1,1,1,1000,1,1},values{2,2,2,100,3,3},done{1,1,0,1,1,1},truncated{1,0,0,0,0,0},final_values{4,0,0,0,0,0},last{0,0,0};
        std::vector<float> expected(6),cpu_returns(6),measured(6);
        for(int env=0;env<3;++env)rl::gae_one_env(2,3,env,rewards.data(),values.data(),done.data(),truncated.data(),final_values.data(),last.data(),gamma,lambda,expected.data(),cpu_returns.data());
        device_ptr<float> dr(6),dv(6),dd(6),dt(6),df(6),dl(3),da(6),dret(6);
        dr.copy_from_host(rewards.data(),6);dv.copy_from_host(values.data(),6);dd.copy_from_host(done.data(),6);dt.copy_from_host(truncated.data(),6);df.copy_from_host(final_values.data(),6);dl.copy_from_host(last.data(),3);
        rl::compute_gae(2,3,dr.get(),dv.get(),dd.get(),dt.get(),df.get(),dl.get(),gamma,lambda,da.get(),dret.get(),stream.value);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));da.copy_to_host(measured.data(),6);float difference=0;
        for(int i=0;i<6;++i)difference=std::max(difference,std::abs(expected[i]-measured[i]));
        if(difference>1.e-5f||std::abs(measured[0]-(1+gamma*4-2))>1.e-5f||measured[1]!=-1)throw std::runtime_error("Time-limit bootstrap gate failed");
        std::ofstream bootstrap(output/"time-limit-bootstrap-check.json");bootstrap<<std::setprecision(10)<<"{\"pass\":true,\"CPU_CUDA_max_difference\":"<<difference
            <<",\"truncated_advantage\":"<<measured[0]<<",\"terminal_advantage\":"<<measured[1]<<",\"continuing_advantage\":"<<measured[2]
            <<",\"propagation_across_reset_blocked\":true,\"physics_steps\":0,\"optimizer_updates\":0}\n";
        }
#endif
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
        }else {
            if(fork){
                const std::string parent_task=argv[10];
                if(parent_task.size()!=64||!std::all_of(parent_task.begin(),parent_task.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');}))
                    throw std::runtime_error("Invalid explicit parent task identity");
                const auto parent=cudarl::checkpoint::read(argv[9]);
                cudarl::checkpoint::Reader parent_reader{parent};parent_reader.value<uint32_t>();const bool parent_episodic=parent_reader.value<uint32_t>()==4;
                rate_restore(parent,roll,actor,critic,aa,ca,ledger,id,stream.value,parent_task.c_str(),parent_inputs,parent_episodic);
                const auto roundtrip=rate_payload(roll,actor,critic,aa,ca,ledger,id,stream.value,parent_task.c_str(),parent_inputs,parent_episodic);
                if(parent!=roundtrip) throw std::runtime_error("Parent import did not preserve full state exactly");
#if defined(CUDARL_FRONTIER_REWARD)
                {
                if(parent_inputs!=35)throw std::runtime_error("Frontier continuation requires the qualified appended parent");
                std::vector<Lane> frontier_lanes(worlds);roll.lanes.copy_to_host(frontier_lanes.data(),frontier_lanes.size());
                frontier_probe<<<(worlds+63)/64,64,0,stream.value>>>(roll.lanes.get(),roll.final_value.get());check_launch();
                CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
                std::vector<float> frontier_gpu(worlds);roll.final_value.copy_to_host(frontier_gpu.data(),frontier_gpu.size());
                float frontier_difference=0;
                for(int lane=0;lane<worlds;++lane){const auto& saved=frontier_lanes[lane];
                    const float cpu=saved.needs_start?0.f:tail::frontier::potential(saved.state,saved.state.pose.com_m,
                        saved.state.motor.navigation.motion.request.origin_m.y);
                    frontier_difference=std::max(frontier_difference,std::abs(cpu-frontier_gpu[lane]));
                    if(!std::isfinite(frontier_gpu[lane])||frontier_gpu[lane]<0||frontier_gpu[lane]>2)
                        throw std::runtime_error("Invalid frontier potential");
                }
                chooser::World synthetic{};for(auto& word:synthetic.visited)word=~uint64_t{0};
                int unknown=-1;
                for(int cell=0;cell<tail::chooser::cells_x*tail::chooser::cells_z;++cell)
                    if(tail::distant_map::word_at(0,cell/64)&(uint64_t{1}<<(cell%64))){unknown=cell;break;}
                if(unknown<0)throw std::runtime_error("Empty frontier map");
                synthetic.visited[unknown/64]&=~(uint64_t{1}<<(unknown%64));
                cudarl::physics::V p{(unknown%tail::chooser::cells_x-tail::chooser::cells_x/2+.5f)*tail::chooser::cell_size_m,0,
                    (unknown/tail::chooser::cells_x-tail::chooser::cells_z/2+.5f)*tail::chooser::cell_size_m};
                const float depth=tail::distant_map::depth_at(0);
                const float farther_potential=tail::frontier::potential(synthetic,p+cudarl::physics::V{.10f,0,0},depth);
                const float closer_potential=tail::frontier::potential(synthetic,p+cudarl::physics::V{.05f,0,0},depth);
                const float forward=tail::frontier::shaping(farther_potential,closer_potential,false,gamma),back=tail::frontier::shaping(closer_potential,farther_potential,false,gamma);
                const float cycle=forward+gamma*back;
                if(frontier_difference>1.e-6f||!(forward>0&&back<0&&cycle<=0)||
                    tail::frontier::shaping(closer_potential,closer_potential,false,gamma)>=0||tail::frontier::shaping(closer_potential,2,true,gamma)!=-closer_potential||
                    tail::frontier::potential(synthetic,p,0)!=0)
                    throw std::runtime_error("Frontier shaping algebra/parity failed");
                synthetic.visited[unknown/64]|=uint64_t{1}<<(unknown%64);
                if(tail::frontier::potential(synthetic,p,depth)!=2)throw std::runtime_error("Complete coverage potential differs");
                std::ofstream frontier_check(output/"frontier-check.json");frontier_check<<std::setprecision(10)
                    <<"{\"pass\":true,\"worlds\":"<<worlds<<",\"CPU_CUDA_max_difference\":"<<frontier_difference
                    <<",\"toward_reward\":"<<forward<<",\"away_reward\":"<<back<<",\"discounted_cycle\":"<<cycle
                    <<",\"discount\":"<<gamma<<",\"terminal_potential_zero\":true,\"physics_steps\":0,\"optimizer_updates\":0}\n";
                }
#endif
#if defined(CUDARL_DISTANT_COVERAGE)
                if(!tail::distant::geometry_matches(larger_pond))throw std::runtime_error("Distant map geometry differs");
                std::vector<Lane> probe(worlds);roll.lanes.copy_to_host(probe.data(),probe.size());
#if defined(CUDARL_THREE_DEPTHS)
                // Probe each sensor plane with the parent's real visit patterns;
                // these temporary observations never enter an optimizer batch.
                const auto original_lanes=probe;
                std::array<int,3> depth_probes{};
                for(int i=0;i<worlds;++i) {
                    const int index=practice_depth_index(i);
                    probe[i].state.motor.navigation.motion.request.origin_m.y=tail::distant_map::depth_at(index);
                    depth_probes[index]+=!probe[i].needs_start;
                }
                roll.lanes.copy_from_host(probe.data(),probe.size());
#endif
                observe_lanes<<<worlds/lane_threads,lane_threads,0,stream.value>>>(roll.lanes.get(),larger_pond,roll.obs.get(),roll.first_fault.get(),roll.fault_code.get());
                check_launch();require_clear(roll,ledger,stream.value,true);
                std::vector<float> device_observation(worlds*channels);
                roll.obs.copy_to_host(device_observation.data(),device_observation.size());
                observe_lanes<<<worlds/lane_threads,lane_threads,0,stream.value>>>(roll.lanes.get(),larger_pond,roll.obs.get(),roll.first_fault.get(),roll.fault_code.get(),false);
                check_launch();require_clear(roll,ledger,stream.value,true);
                std::vector<float> legacy_device_observation(worlds*channels);
                roll.obs.copy_to_host(legacy_device_observation.data(),legacy_device_observation.size());
                float maximum_new_field=0,maximum_legacy=0;bool gpu_unchanged=true;
                float max_difference=0;int active_probes=0,worst_index=-1;float worst_cpu=0,worst_gpu=0;
                for(int lane=0;lane<worlds;++lane){
                    float expected[channels]{};
                    if(!probe[lane].needs_start){
                        ++active_probes;chooser::observe(probe[lane].state,larger_pond,expected,chooser::default_observation_mode);
                        float legacy[channels];chooser::observe(probe[lane].state,larger_pond,legacy,false);
                        for(int i=0;i<channels;++i){
                            maximum_legacy=std::max(maximum_legacy,std::abs(legacy[i]-legacy_device_observation[lane*channels+i]));
                            const int first_new=channels>=35?32:21;
                            if(i<first_new||(channels<35&&i>first_new+2)){
                                if(expected[i]!=legacy[i])throw std::runtime_error("Distant feedback changed another CPU input");
                                if(device_observation[lane*channels+i]!=legacy_device_observation[lane*channels+i])gpu_unchanged=false;
                            }
                        }
                    }
                    for(int i=0;i<channels;++i){const float delta=std::abs(expected[i]-device_observation[lane*channels+i]);
                        if(!std::isfinite(delta))throw std::runtime_error("Nonfinite distant observation");
                        const int first_new=channels>=35?32:21;
                        if(i>=first_new&&i<=first_new+2)maximum_new_field=std::max(maximum_new_field,delta);
                        if(delta>max_difference){max_difference=delta;worst_index=lane*channels+i;worst_cpu=expected[i];worst_gpu=device_observation[lane*channels+i];}
                    }
                }
                {std::ofstream report(output/"distant-parity.json");report<<std::setprecision(10)
                    <<"{\"pass\":"<<(maximum_new_field<=1.e-6f&&gpu_unchanged&&active_probes>0?"true":"false")<<",\"max_difference\":"<<max_difference
                    <<",\"worst_lane\":"<<(worst_index<0?-1:worst_index/channels)<<",\"input\":"<<(worst_index<0?-1:worst_index%channels)
                    <<",\"cpu\":"<<worst_cpu<<",\"cuda\":"<<worst_gpu<<",\"new_field_maximum\":"<<maximum_new_field
                    <<",\"legacy_maximum\":"<<maximum_legacy<<",\"other_GPU_inputs_unchanged\":"<<(gpu_unchanged?"true":"false")
                    <<",\"physics_steps\":0,\"optimizer_updates\":0}\n";}
                if(maximum_new_field>1.e-6f||!gpu_unchanged||active_probes==0)throw std::runtime_error("New distant input disagreement or legacy GPU input change");
#if defined(CUDARL_CURVED_BANK)
                {
                    observe_lanes<<<worlds/lane_threads,lane_threads,0,stream.value>>>(roll.lanes.get(),larger_pond,roll.obs.get(),roll.first_fault.get(),roll.fault_code.get(),2);
                    check_launch();CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
                    std::vector<float> old_observation(worlds*channels);roll.obs.copy_to_host(old_observation.data(),old_observation.size());
                    float curved_difference=0;int valid_curves=0;
                    for(int lane=0;lane<worlds;++lane){float current[channels]{},old[channels]{};
                        if(!probe[lane].needs_start){chooser::observe(probe[lane].state,larger_pond,current,3);chooser::observe(probe[lane].state,larger_pond,old,2);}
                        for(int i=0;i<35;++i)if(current[i]!=old[i]||device_observation[lane*channels+i]!=old_observation[lane*channels+i])
                            throw std::runtime_error("Curved sensing changed an inherited input");
                        for(int i=35;i<39;++i){const float delta=std::abs(current[i]-device_observation[lane*channels+i]);
                            if(!std::isfinite(delta))throw std::runtime_error("Invalid curved sensory input");curved_difference=std::max(curved_difference,delta);}
                        valid_curves+=current[38]>0;
                    }
                    int curve_distinctions=0;
                    for(float radius:{.45f,.50f,.55f}){
                        chooser::World sample{};const auto& archived=initial[0];
                        const auto heading=archived.legs[0].heading;const cudarl::physics::V tangent{0,0,-1};
                        const float yaw=atan2f(cudarl::physics::cross(heading,tangent).y,cudarl::physics::dot(heading,tangent));
                        chooser::initialize(sample,archived,{radius,0,0},yaw,id.kind,id.config);
                        const auto body_before=sample.motor.navigation.motion.body;
                        const auto straight=tail::curved_bank::project(body_before,sample.pose,larger_pond,tangent*.012f,0);
                        const auto curve=tail::curved_bank::project(body_before,sample.pose,larger_pond,tangent*.012f,.012f/radius);
                        if(straight.loss_s<24&&curve.loss_s==24&&curve.min_gap_m>0)++curve_distinctions;
                        if(std::memcmp(&body_before,&sample.motor.navigation.motion.body,sizeof(body_before)))throw std::runtime_error("Curve query changed physical body");
                    }
                    if(curved_difference>1.e-4f||valid_curves==0||curve_distinctions==0||(parent_inputs<39&&(aa.fresh_steps(1)!=0||ca.fresh_steps(1)!=0)))
                        throw std::runtime_error("Curved feedback parity/geometry/age gate failed");
                    std::ofstream check(output/"curved-check.json");check<<std::setprecision(10)
                        <<"{\"pass\":true,\"worlds\":"<<worlds<<",\"valid_forecasts\":"<<valid_curves<<",\"max_normalized_CPU_CUDA_difference\":"<<curved_difference
                        <<",\"preserved_inputs\":35,\"safe_curve_unsafe_straight_examples\":"<<curve_distinctions
                        <<",\"inherited_distant_Adam_age\":"<<aa.fresh_steps()<<",\"new_curve_Adam_age\":"<<aa.fresh_steps(1)
                        <<",\"physics_steps\":0,\"optimizer_updates\":0}\n";
                }
#endif
#if defined(CUDARL_APPENDED_COVERAGE)
                if(parent_inputs<channels){
                const auto actor_old=chooser::resize_input_parameters(actor.get_params(),channels,parent_inputs);
                const auto critic_old=chooser::resize_input_parameters(critic.get_params(),channels,parent_inputs);
                const nn::MlpSpec old_actor_spec{{parent_inputs,32,actions},{nn::Activation::Tanh,nn::Activation::Identity}};
                const nn::MlpSpec old_critic_spec{{parent_inputs,32,1},{nn::Activation::Tanh,nn::Activation::Identity}};
                nn::GpuMlp old_actor(old_actor_spec,worlds,seed),old_critic(old_critic_spec,worlds,seed+1);
                old_actor.set_params(actor_old);old_critic.set_params(critic_old);
                std::vector<float> old_input(worlds*parent_inputs);
                for(int lane=0;lane<worlds;++lane)for(int i=0;i<parent_inputs;++i)old_input[lane*parent_inputs+i]=device_observation[lane*channels+i];
                device_ptr<float> legacy_input_gpu(old_input.size());legacy_input_gpu.copy_from_host(old_input.data(),old_input.size());
                roll.obs.copy_from_host(device_observation.data(),device_observation.size());
                old_actor.forward(legacy_input_gpu.get(),worlds,stream.value);old_critic.forward(legacy_input_gpu.get(),worlds,stream.value);
                actor.forward(roll.obs.get(),worlds,stream.value);critic.forward(roll.obs.get(),worlds,stream.value);
                CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
                std::vector<float> old_logits(worlds*actions),new_logits(worlds*actions),old_values(worlds),new_values(worlds);
                CUDA_CHECK_THROW(cudaMemcpy(old_logits.data(),old_actor.output(),old_logits.size()*sizeof(float),cudaMemcpyDeviceToHost));
                CUDA_CHECK_THROW(cudaMemcpy(new_logits.data(),actor.output(),new_logits.size()*sizeof(float),cudaMemcpyDeviceToHost));
                CUDA_CHECK_THROW(cudaMemcpy(old_values.data(),old_critic.output(),old_values.size()*sizeof(float),cudaMemcpyDeviceToHost));
                CUDA_CHECK_THROW(cudaMemcpy(new_values.data(),critic.output(),new_values.size()*sizeof(float),cudaMemcpyDeviceToHost));
                float actor_difference=0,critic_difference=0;
                for(size_t i=0;i<old_logits.size();++i)actor_difference=std::max(actor_difference,std::abs(old_logits[i]-new_logits[i]));
                for(size_t i=0;i<old_values.size();++i)critic_difference=std::max(critic_difference,std::abs(old_values[i]-new_values[i]));
                std::ofstream expansion(output/"input-expansion.json");expansion<<std::setprecision(10)
                    <<"{\"pass\":"<<(actor_difference==0&&critic_difference==0?"true":"false")
                    <<",\"inputs_before\":"<<parent_inputs<<",\"inputs_after\":"<<channels<<",\"worlds\":"<<worlds
                    <<",\"GPU_actor_difference\":"<<actor_difference<<",\"GPU_critic_difference\":"<<critic_difference
                    <<",\"legacy_checkpoint_roundtrip_exact\":true,\"new_actor_critic_weights_and_moments_zero\":true"
                    <<",\"fresh_actor_Adam_age\":"<<aa.fresh_steps()<<",\"fresh_critic_Adam_age\":"<<ca.fresh_steps()
                    <<",\"physics_steps\":0,\"optimizer_updates\":0}\n";
                if(actor_difference!=0||critic_difference!=0)throw std::runtime_error("Zero-column expansion changed GPU inference");
                }
#endif
                for(int map=0;map<tail::distant_map::depth_count;++map){
                auto sensory=probe.front().state;
                for(auto& word:sensory.visited)word=0;
                for(auto& visit:sensory.recent)visit={};
                float fresh[3],known[3],front[3],back[3];
                const float depth=tail::distant_map::depth_at(map);
                tail::distant::observe(sensory,{0,depth,0},{1,0,0},depth,0,fresh);
                for(float x:fresh)if(x!=1.f)throw std::runtime_error("Fresh sector normalization failed");
                for(auto& word:sensory.visited)word=~uint64_t{0};
                tail::distant::observe(sensory,{0,depth,0},{1,0,0},depth,0,known);
                for(float x:known)if(x!=0.f)throw std::runtime_error("Visited water has false novelty");
                int target=-1;
                for(int cell=0;cell<1536;++cell)if((tail::distant_map::word_at(map,cell/64)&(uint64_t{1}<<(cell%64)))&&
                    cell%48>=31&&cell/48==16){target=cell;break;}
                if(target<0)throw std::runtime_error("Missing front sensory fixture");
                sensory.visited[target/64]&=~(uint64_t{1}<<(target%64));
                tail::distant::observe(sensory,{0,depth,0},{1,0,0},depth,0,front);
                tail::distant::observe(sensory,{0,depth,0},{-1,0,0},depth,0,back);
                if(!(front[1]>0&&front[0]==0&&front[2]==0&&back[1]==0&&(back[0]>0||back[2]>0)))
                    throw std::runtime_error("Distant field is not course-relative");
                }
#if defined(CUDARL_THREE_DEPTHS)
                roll.lanes.copy_from_host(original_lanes.data(),original_lanes.size());
                if(*std::min_element(depth_probes.begin(),depth_probes.end())==0)
                    throw std::runtime_error("Missing CPU/CUDA probes at a practice depth");
#endif
                std::ofstream senses(output/"distant-senses.json");
                senses<<"{\"pass\":true,\"active_worlds\":"<<active_probes<<",\"CPU_CUDA_values\":"<<worlds*channels
                    <<",\"max_difference\":"<<maximum_new_field<<",\"inherited_legacy_max_difference\":"<<maximum_legacy
                    <<",\"preserved_local_inputs\":"<<(channels>=35?32:29)<<",\"first_distant_input\":"<<(channels>=35?32:21)
                    <<",\"tested_depths\":"<<tail::distant_map::depth_count
                    <<",\"fresh_known_rotation_checks\":true,\"physics_steps\":0,\"optimizer_updates\":0}\n";
#endif
                std::ofstream imported(output/"parent-import.json");
                imported<<"{\"pass\":true,\"full_state_exact_before_counter_rebase\":true,\"parent_update\":"<<ledger.completed
                    <<",\"parent_checksum\":"<<cudarl::checkpoint::checksum(parent)<<",\"parent_task\":\""<<parent_task
                    <<"\",\"physics_steps\":0,\"optimizer_updates\":0}\n";
                if(inspect){
                    chooser::write_weights((output/"parent-actor.bin").u8string(),actor.get_params());
                    return 0;
                }
                // New study budget and sampling counter; retain learned state.
                ledger.completed=ledger.started=0; ledger.counts={};
                zero(roll.counts.get(),roll.counts.size_bytes(),stream.value);
#if defined(CUDARL_OPEN_WATER)
                zero(roll.episodes.get(),roll.episodes.size_bytes(),stream.value);
#endif
#if defined(CUDARL_THREE_DEPTHS)
                // Begin each cohort at its assigned depth exactly once, at fork.
                // A resume uses the saved live world, including visitation history.
                initialize_lanes<<<worlds/lane_threads,lane_threads,0,stream.value>>>(roll.lanes.get());
                check_launch();
#endif
#if defined(CUDARL_CLOSE_SHORE)
                // Explicit new-study initial conditions, only after the exact
                // parent roundtrip. Never executed on resume or within episodes.
                zero(roll.lanes.get(),roll.lanes.size_bytes(),stream.value);
#if defined(CUDARL_EPISODIC_BANK)
                zero(roll.episodes.get(),roll.episodes.size_bytes(),stream.value);
#endif
                initialize_lanes<<<worlds/lane_threads,lane_threads,0,stream.value>>>(roll.lanes.get());check_launch();
                std::ofstream initial_change(output/"new-study-initial-worlds.json");initial_change
                    <<"{\"new_worlds\":1024,\"close_tangent_worlds\":512,\"older_approach_worlds\":256,\"interior_worlds\":256,\"motor_unchanged\":true,\"actor_critic_Adam_preserved\":true,\"live_resets\":0}\n";
#endif
            }
            rate_publish(output,roll,actor,critic,aa,ca,ledger,id,stream.value);
        }
        auto save_budget=[&](double used){
            cudarl::checkpoint::Writer b;b.value<uint32_t>(1);
            b.value(cudarl::checkpoint::checksum(cudarl::checkpoint::read((output/"latest.checkpoint").u8string())));
            b.value(target);b.value(wall);b.value(used);cudarl::checkpoint::write_atomic((output/"budget.checkpoint").u8string(),b.data);
        };
        if(!resume)save_budget(0);
        const auto launch=Clock::now();std::vector<uint8_t> one,expected;
        bool checking=false;int executed_updates=0;
        std::ofstream curve(output/(resume?"resume-updates.csv":"updates.csv"));
#if defined(CUDARL_REGION_CREDIT)
        std::ofstream regional_log(output/(resume?"resume-regions.csv":"regions.csv"));
        regional_log.exceptions(std::ios::badbit|std::ios::failbit);
        regional_log<<"update,rewarded_transitions,new_regions\n";
        std::vector<int> region_hits(rows);
#endif
        curve<<"update,reward,kl,policy_loss,value_loss,entropy,explained_variance,stops,samples,wall_s\n";
        try{while(ledger.completed<target && prior_used+elapsed(launch)<wall && !fs::exists(output/"STOP")) {
            const int update=ledger.completed+1;
            if(!collect(roll,actor,critic,starts,gait,offset,ledger,update,stream.value,events,launch))break;
            const float reward=finite_mean(roll,roll.reward.get(),rows,stream.value);
#if defined(CUDARL_REGION_CREDIT)
            roll.region_hits.copy_to_host(region_hits.data(),region_hits.size());
            int paid=0,regions=0;
            for(int hits:region_hits){
                if(hits<0||hits>4)throw std::runtime_error("Invalid regional coverage count");
                paid+=hits>0;regions+=hits;
            }
            regional_log<<update<<','<<paid<<','<<regions<<std::endl;
#endif
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
