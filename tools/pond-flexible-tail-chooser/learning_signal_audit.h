#pragma once
namespace learning_signal {
inline constexpr float discount=.994f,trace_lambda=.98f;
std::vector<float> read_critic(const fs::path& path,const Models& models,uint32_t expected_format=3){
    const auto bytes=cudarl::checkpoint::read(path.u8string());cudarl::checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x52435446||in.value<uint32_t>()!=expected_format)throw std::runtime_error("Unexpected 39-input full checkpoint format");
    const auto task=in.array<char>(64);if(task.size()!=64)throw std::runtime_error("Invalid critic task");
    const int expected[]{1024,8,39,15,1024,4,40,128};for(auto n:expected)if(in.value<int>()!=n)throw std::runtime_error("Critic runtime layout differs");
    in.value<uint32_t>();in.value<uint32_t>(); // Recorded Lane/Counts sizes, not deserialized here.
    in.value<uint64_t>();in.value<uint64_t>();in.value<uint64_t>();
    if(in.value<tail::steering::Kind>()!=models.motor.kind)throw std::runtime_error("Critic motor kind differs");
    in.value<tail::steering::Config>();
    const float settings[]{discount,trace_lambda,.05f,.20f,.01f,.50f,.0003f,.001f};
    for(auto n:settings)if(in.value<float>()!=n)throw std::runtime_error("Critic PPO settings differ");
    const auto pond=in.value<cudarl::pond_geometry_v1::Parameters>();
    if(std::memcmp(&pond,&larger_pond,sizeof(pond)))throw std::runtime_error("Critic geometry differs");
    if(expected_format==4 && (in.value<int>()!=3000||in.value<uint32_t>()!=32))
        throw std::runtime_error("Wrong training episode layout");
    in.value<int>();in.value<int>();in.value<int>();
    const auto actor=in.array<float>(10000),critic=in.array<float>(10000);
    if(actor!=models.policy||critic.size()!=size_t(39*32+32+32+1)||!std::all_of(critic.begin(),critic.end(),physics::finite_number))
        throw std::runtime_error("Actor/critic provenance check failed");
    return critic;
}
int run(const Models& models,const fs::path& output,const fs::path& checkpoint,const fs::path& captures,Clock::time_point begin){
    auto critic_weights=read_critic(checkpoint,models);if(!fs::create_directory(output))throw std::runtime_error("Signal output must be unused");
    fs::create_directory(output/"traces");
    cudarl::nn::CpuMlp<float> actor(chooser::actor_spec());actor.params()=models.policy;
    cudarl::nn::MlpSpec spec{{chooser::inputs,32,1},{cudarl::nn::Activation::Tanh,cudarl::nn::Activation::Identity}};
    cudarl::nn::CpuMlp<float> critic(spec);critic.params()=critic_weights;
    const auto value=[&](const chooser::World& w){float input[chooser::inputs];chooser::observe(w,larger_pond,input,models.distant_feedback);critic.forward(input,1);
        const float v=critic.output()[0];if(!std::isfinite(v))throw std::runtime_error("Nonfinite audit value");return v;};
    uint64_t work=0,accepted=0;int cases=0,safe_count=0,ranking_passes=0;
    std::ofstream table(output/"cases.csv");table<<std::setprecision(10)<<"phase,action,hold_controls,pass,fault,seconds,V_initial,discounted_reward,endpoint_bootstrap,return40,advantage40,lambda_advantage8,lambda_target8,mean_speed_mm_s,final_gap_mm\n";
    for(int phase:{8,9}){
        const auto name="phase"+std::to_string(phase)+"-bank1-direction"+std::to_string(phase==8?-1:1)+"-40.bin";
        const auto bytes=cudarl::checkpoint::read((captures/name).u8string());cudarl::checkpoint::Reader in{bytes};
        if(in.value<uint32_t>()!=1)throw std::runtime_error("Invalid exact-state capture");const auto saved=in.value<chooser::World>();in.finish();
        double control_return=0,best_safe=-1.e30;
        for(int branch=-1;branch<30;++branch){
            chooser::World w;std::memcpy(&w,&saved,sizeof(w));const int action=branch<0?-1:branch/2,hold=branch<0?0:(branch%2?400:120);
            const int straight=action<0?0:action/chooser::rate_count*chooser::rate_count+chooser::rate_count/2;
            const float initial_value=value(w);int control=0,steps=0;double power=1,sum=0,gae=0,gae_power=1,path=0;float stall=0,max_stall=0;
            std::ofstream trace(output/"traces"/("phase"+std::to_string(phase)+"-branch"+std::to_string(branch)+".csv"));
            trace<<std::setprecision(10)<<"macro,action,reward_base,reward_curved,V_before,V_after,TD_delta,done,gap_mm\n";
            while(!w.fault&&control<2000){
                if(work>1984000||Clock::now()-begin>=std::chrono::seconds(45))throw std::runtime_error("Signal audit cap reached");
                const int chosen=action<0||control>=hold+400?close_probe::decide(w,actor,models.distant_feedback):(control<hold?action:straight);
                const float pre=value(w);if(!chooser::accept(w,chosen))throw std::runtime_error("Audit action unavailable");
                const auto before=w.pose.com_m;const auto r=chooser::advance(w,models.gait.data(),models.packed_motor.data(),larger_pond,40);
                work+=r.attempted_substeps;accepted+=r.accepted_substeps;control+=r.controls;
                if(r.fault&&r.fault!=chooser::Clearance)throw std::runtime_error("Signal audit native failure");
                const auto sample=tail::fast::feedback_sample(w.motor);const float speed=physics::norm(sample.velocity_m_s);
                const float gap=float(tail::submerged_gap_m(larger_pond,w.motor.navigation.motion.body,w.pose));
                const float curved=r.fault?0:tail::curved_bank::reward(tail::curved_bank::observe(w.motor,larger_pond),gap,speed,r.controls*tail::motor_control_s);
                const float reward=r.reward+curved,next=r.fault?0:value(w),delta=reward+discount*next-pre;
                sum+=power*reward;power*=discount;if(steps<8){gae+=gae_power*delta;gae_power*=discount*trace_lambda;}
                ++steps;path+=physics::norm(horizontal(w.pose.com_m-before));stall=speed<.006f?stall+r.controls*.02f:0;max_stall=std::max(max_stall,stall);
                trace<<steps<<','<<chosen<<','<<r.reward<<','<<curved<<','<<pre<<','<<next<<','<<delta<<','<<(r.fault!=0)<<','<<gap*1000<<'\n';
            }
            const float final_gap=float(tail::submerged_gap_m(larger_pond,w.motor.navigation.motion.body,w.pose));
            const double endpoint=w.fault?0:power*value(w),total=sum+endpoint,seconds=control*.02,mean=seconds>0?path/seconds:0;
            const bool pass=!w.fault&&control==2000&&w.min_gap_m>0&&w.peak_tilt_deg<=10&&w.peak_depth_error_m<=.001f&&mean>=.006&&max_stall<=8.14&&final_gap>=.015;
            if(branch==-1)control_return=total;if(pass)best_safe=std::max(best_safe,total);++cases;safe_count+=pass;
            table<<phase<<','<<action<<','<<hold<<','<<pass<<','<<w.fault<<','<<seconds<<','<<initial_value<<','<<sum<<','<<endpoint<<','<<total<<','<<total-initial_value<<','<<gae<<','<<gae+initial_value<<','<<mean*1000<<','<<final_gap*1000<<'\n';
        }
        ranking_passes+=best_safe>control_return;
    }
    std::ofstream summary(output/"summary.json");summary<<std::setprecision(10)<<"{\"complete\":true,\"cases\":"<<cases<<",\"safe_cases\":"<<safe_count<<",\"states_safe_return_outranks_failed_policy\":"<<ranking_passes<<",\"attempted_substeps\":"<<work<<",\"accepted_substeps\":"<<accepted<<",\"optimizer_updates\":0,\"wall_s\":"<<std::chrono::duration<double>(Clock::now()-begin).count()<<"}\n";
    std::cout<<"Signal ranking: "<<ranking_passes<<"/2, safe cases "<<safe_count<<"/"<<cases<<'\n';return 0;
}
}
