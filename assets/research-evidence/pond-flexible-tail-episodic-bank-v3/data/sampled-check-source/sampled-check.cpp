#define main unused_archived_evaluator_main
#include "../source/tools/pond-flexible-tail-chooser/rate-evaluate.cpp"
#undef main
#include "rl/ppo_math.h"
std::pair<std::vector<float>,std::vector<float>> read_pair(const fs::path& path){
    const auto bytes=cudarl::checkpoint::read(path.u8string());cudarl::checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x52435446||in.value<uint32_t>()!=4)throw std::runtime_error("Expected episodic4");in.array<char>(64);
    for(int i=0;i<8;++i)in.value<int>();in.value<uint32_t>();in.value<uint32_t>();for(int i=0;i<3;++i)in.value<uint64_t>();
    in.value<tail::steering::Kind>();in.value<tail::steering::Config>();for(int i=0;i<8;++i)in.value<float>();in.value<cudarl::pond_geometry_v1::Parameters>();
    if(in.value<int>()!=3000||in.value<uint32_t>()!=32)throw std::runtime_error("Wrong episode layout");for(int i=0;i<3;++i)in.value<int>();
    auto a=in.array<float>(10000),c=in.array<float>(10000);return {a,c};
}
int main(int argc,char** argv){try{
    if(argc!=6)throw std::runtime_error("Usage: sampled_check TRAINING CAPTURES GAIT MOTOR NEW_OUT");
    const auto begin=Clock::now();const fs::path output=argv[5];if(!fs::create_directory(output))throw std::runtime_error("Immutable output exists");fs::create_directory(output/"traces");
    Models m;m.gait=tail::read_navigation_weights(argv[3],tail::NavigationMode::Gait);m.motor=tail::steering::read_model(argv[4]);m.config.study=tail::steering::Study::Rate;
    m.packed_motor=m.motor.weights;m.packed_motor.insert(m.packed_motor.end(),m.motor.frozen_skip.begin(),m.motor.frozen_skip.end());
    cudarl::nn::CpuMlp<float> actor(chooser::actor_spec());cudarl::nn::MlpSpec spec{{39,32,1},{cudarl::nn::Activation::Tanh,cudarl::nn::Activation::Identity}};cudarl::nn::CpuMlp<float> critic(spec);
    uint64_t attempted=0,accepted=0;int cases=0,passes=0;
    std::ofstream table(output/"cases.csv");table<<std::setprecision(10)<<"update,phase,seed,pass,fault,seconds,V_initial,discounted_reward,endpoint_bootstrap,return40,mean_speed_mm_s,final_gap_mm\n";
    for(int u:{0,128}){const auto p=read_pair(fs::path(argv[1])/"checkpoints"/("session-"+std::to_string(u)+".bin"));int mode=0;
        if(chooser::read_weights((fs::path(argv[1])/("actor-"+std::to_string(u)+".bin")).u8string(),&mode)!=p.first||mode!=3)throw std::runtime_error("Wrong actor binding");actor.params()=p.first;critic.params()=p.second;
        for(int phase:{8,9}){const auto name="phase"+std::to_string(phase)+"-bank1-direction"+std::to_string(phase==8?-1:1)+"-40.bin";
            const auto bytes=cudarl::checkpoint::read((fs::path(argv[2])/name).u8string());cudarl::checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=1)throw std::runtime_error("Wrong capture");const auto saved=in.value<chooser::World>();in.finish();
            const auto value=[&](const chooser::World& w){float input[39];chooser::observe(w,larger_pond,input,3);critic.forward(input,1);return critic.output()[0];};
            for(int seed=-1;seed<16;++seed){auto w=saved;const auto initial=value(w);int control=0,macro=0;double power=1,reward_sum=0,path=0;float stall=0,max_stall=0;
                std::ofstream trace(output/"traces"/("u"+std::to_string(u)+"-p"+std::to_string(phase)+"-s"+std::to_string(seed)+".csv"));trace<<std::setprecision(10)<<"macro,action,reward,x_m,y_m,z_m,gap_mm,fault\n";
                while(!w.fault&&control<2000){if(attempted>2176000||Clock::now()-begin>std::chrono::seconds(45))throw std::runtime_error("Sample diagnostic cap");
                    float input[39],logp[15];chooser::observe(w,larger_pond,input,3);actor.forward(input,1);cudarl::rl::log_softmax(actor.output(),15,logp);
                    const float uniform=cudarl::rng::uniform(26112531u,macro,uint32_t(phase*16+std::max(0,seed)));
                    const int action=cudarl::rl::sample_action(logp,15,uniform,seed<0);if(!chooser::accept(w,action))throw std::runtime_error("Invalid sampled request");
                    const auto before=w.pose.com_m;const auto r=chooser::advance(w,m.gait.data(),m.packed_motor.data(),larger_pond,40);attempted+=r.attempted_substeps;accepted+=r.accepted_substeps;control+=r.controls;++macro;
                    if(r.fault&&r.fault!=chooser::Clearance)throw std::runtime_error("Native sampled diagnostic fault");
                    const float speed=physics::norm(tail::fast::feedback_sample(w.motor).velocity_m_s),gap=float(tail::submerged_gap_m(larger_pond,w.motor.navigation.motion.body,w.pose));
                    const float reward=r.reward+(r.fault?0:tail::curved_bank::reward(tail::curved_bank::observe(w.motor,larger_pond),gap,speed,r.controls*.02f));
                    reward_sum+=power*reward;power*=.994f;path+=physics::norm(horizontal(w.pose.com_m-before));stall=speed<.006f?stall+r.controls*.02f:0;max_stall=std::max(max_stall,stall);
                    trace<<macro<<','<<action<<','<<reward<<','<<w.pose.com_m.x<<','<<w.pose.com_m.y<<','<<w.pose.com_m.z<<','<<gap*1000<<','<<w.fault<<'\n';
                }
                const auto gap=tail::submerged_gap_m(larger_pond,w.motor.navigation.motion.body,w.pose);const double seconds=control*.02,mean=seconds>0?path/seconds:0,endpoint=w.fault?0:power*value(w);
                const bool pass=!w.fault&&control==2000&&w.min_gap_m>0&&w.peak_tilt_deg<=10&&w.peak_depth_error_m<=.001f&&mean>=.006&&max_stall<=8.14&&gap>=.015;
                ++cases;passes+=pass;table<<u<<','<<phase<<','<<seed<<','<<pass<<','<<w.fault<<','<<seconds<<','<<initial<<','<<reward_sum<<','<<endpoint<<','<<reward_sum+endpoint<<','<<mean*1000<<','<<gap*1000<<'\n';
            }
        }
    }
    if(cases!=68||attempted>2176000)throw std::runtime_error("Incomplete sampling diagnostic");std::ofstream summary(output/"summary.json");summary<<std::setprecision(10)<<"{\"complete\":true,\"cases\":"<<cases<<",\"passes\":"<<passes<<",\"attempted_substeps\":"<<attempted<<",\"accepted_substeps\":"<<accepted<<",\"optimizer_updates\":0,\"wall_s\":"<<std::chrono::duration<double>(Clock::now()-begin).count()<<"}\n";
    std::cout<<passes<<"/"<<cases<<" sampled/greedy diagnostic passes\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
