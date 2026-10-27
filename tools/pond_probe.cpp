#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include "demos/tadpole_pond.h"
#include "demos/foraging_pond.h"
namespace nn=cudarl::nn;

int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::invalid_argument("Usage: pond_probe <actor JSON>");
        std::ifstream file(argv[1]);if(!file) throw std::runtime_error("Policy file not found");
        const std::string json((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
        const auto body_field=json.find("\"body_id\"");
        const int body_id=body_field==std::string::npos?6:std::atoi(json.c_str()+json.find(':',body_field)+1);
        const auto foods_field=json.find("\"food_count\"");
        const int food_count=foods_field==std::string::npos?1:std::atoi(json.c_str()+json.find(':',foods_field)+1);
        const auto field=json.find("\"parameters\"");const auto start=json.find('[',field);
        if(field==std::string::npos || start==std::string::npos) throw std::runtime_error("No parameters in policy");
        const char* p=json.c_str()+start+1;std::vector<float> weights;
        while(*p && *p!=']') {
            if(*p==','||*p==' '||*p=='\n'||*p=='\r') {++p;continue;}
            char* end=nullptr;const float value=std::strtof(p,&end);
            if(end==p) throw std::runtime_error("Invalid policy number");
            weights.push_back(value);p=end;
        }
        if(food_count==3) {
            nn::CpuMlp<float> controller(nn::MlpSpec({13,32,32,2},
                {nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}));
            controller.params()=weights;
            int recovered=0;
            for(uint32_t trial=0;trial<64;++trial) {
                auto state=cudarl::foraging::initial(0x71AB93u,trial);
                state.body.angle[0]=(trial&1u)?1.25f:-1.25f;
                state.body.angle[1]=(trial&2u)?1.25f:-1.25f;
                float maximum_distance=0;bool released=false;
                for(int step=0;step<240;++step) {
                    float observation[13];cudarl::foraging::observe(state,observation);
                    controller.forward(observation,1);cudarl::foraging::advance(state,controller.output());
                    maximum_distance=std::max(maximum_distance,std::hypot(state.body.x,state.body.y));
                    released=released || std::fabs(state.body.angle[0])<0.9f || std::fabs(state.body.angle[1])<0.9f;
                }
                if(released && maximum_distance>=0.5f) ++recovered;
            }
            std::cout<<"RECOVERY folded_starts=64 released_and_traveled="<<recovered<<" seconds=12 distance_required=0.5\n";
        }
        for(uint32_t seed=1;seed<=10;++seed) {
            if(food_count==3) {
                cudarl::demos::ForagingPond pond;pond.set_policy(weights);pond.reset(seed);
                int slots[3]{},longest=0;
                for(int t=0;t<6000;++t) {
                    if(pond.step()) ++slots[pond.state().last_eaten];
                    longest=std::max(longest,pond.state().since_food_steps);
                }
                std::cout<<"seed="<<seed<<" food/min="<<pond.state().eaten/5.0<<" longest_wait="<<longest*0.05f
                         <<" slots="<<slots[0]<<','<<slots[1]<<','<<slots[2]<<'\n';
                continue;
            }
            cudarl::demos::TadpolePond pond(cudarl::swimmer::body_preset(body_id));pond.set_policy(weights);pond.reset(seed);
            int longest=0;float extent=0;
            for(int t=0;t<6000;++t) {
                pond.step();const auto& s=pond.state();
                longest=std::max(longest,s.steps);extent=std::max(extent,std::hypot(s.x,s.y));
            }
            std::cout<<"seed="<<seed<<" eaten="<<pond.eaten()<<" longest_wait="<<longest*0.05f<<" extent="<<extent<<'\n';
        }
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
