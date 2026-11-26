#define main unused_archived_evaluator_main
#include "../source/tools/pond-flexible-tail-chooser/rate-evaluate.cpp"
#undef main
std::pair<std::vector<float>,std::vector<float>> weights(const fs::path& path){
    const auto bytes=cudarl::checkpoint::read(path.u8string());cudarl::checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x52435446)throw std::runtime_error("Invalid full checkpoint");const auto version=in.value<uint32_t>();
    if(version!=4)throw std::runtime_error("Expected episodic4");in.array<char>(64);
    const int expected[]{1024,8,39,15,1024,4,40,128};for(int n:expected)if(in.value<int>()!=n)throw std::runtime_error("Wrong runtime dimensions");
    in.value<uint32_t>();in.value<uint32_t>();for(int i=0;i<3;++i)in.value<uint64_t>();in.value<tail::steering::Kind>();in.value<tail::steering::Config>();
    for(int i=0;i<8;++i)in.value<float>();in.value<cudarl::pond_geometry_v1::Parameters>();
    if(in.value<int>()!=3000||in.value<uint32_t>()!=32)throw std::runtime_error("Wrong episode layout");
    for(int i=0;i<3;++i)in.value<int>();auto a=in.array<float>(10000),c=in.array<float>(10000);return {a,c};
}
int main(int argc,char** argv){try{
    if(argc!=4)throw std::runtime_error("Usage: critic_scan TRAINING CAPTURES NEW_CSV");
    if(fs::exists(argv[3]))throw std::runtime_error("Immutable audit output");
    std::ofstream out(argv[3]);out<<std::setprecision(10)<<"update,phase,V_initial,action,probability,prob_minus4,prob_plus4\n";
    cudarl::nn::CpuMlp<float> actor(chooser::actor_spec());
    cudarl::nn::MlpSpec spec{{39,32,1},{cudarl::nn::Activation::Tanh,cudarl::nn::Activation::Identity}};cudarl::nn::CpuMlp<float> critic(spec);
    std::vector<chooser::World> states;
    for(int phase:{8,9}){const auto name="phase"+std::to_string(phase)+"-bank1-direction"+std::to_string(phase==8?-1:1)+"-40.bin";
        const auto bytes=cudarl::checkpoint::read((fs::path(argv[2])/name).u8string());cudarl::checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=1)throw std::runtime_error("Wrong state");states.push_back(in.value<chooser::World>());in.finish();}
    for(int update=0;update<=128;++update){const auto p=weights(fs::path(argv[1])/"checkpoints"/("session-"+std::to_string(update)+".bin"));
        int mode=0;if(chooser::read_weights((fs::path(argv[1])/("actor-"+std::to_string(update)+".bin")).u8string(),&mode)!=p.first||mode!=3)throw std::runtime_error("Portable actor mismatch");
        actor.params()=p.first;critic.params()=p.second;
        for(int i=0;i<2;++i){float input[39];chooser::observe(states[i],larger_pond,input,3);actor.forward(input,1);critic.forward(input,1);
            const auto logits=actor.output();const int best=int(std::max_element(logits,logits+15)-logits);double sum=0,left=0,right=0;for(int a=0;a<15;++a){const auto mass=std::exp(double(logits[a]-logits[best]));sum+=mass;if(a%5==1)right+=mass;if(a%5==3)left+=mass;}
            out<<update<<','<<i+8<<','<<critic.output()[0]<<','<<best<<','<<1/sum<<','<<right/sum<<','<<left/sum<<'\n';}
    }
    std::cout<<"258 value/action predictions; zero physics, zero optimizer updates\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
