#include <cstdio>
#include "rl/pond_selector_ppo.h"
#include "nn/mlp_cpu.h"
int main(int argc,char** argv) {
    try {
        if(argc!=3) throw std::invalid_argument("pond_recovery_probe STALL_FILE POLICY");
        const auto bytes=cudarl::checkpoint::read(argv[1]);cudarl::checkpoint::Reader in{bytes};
        if(in.value<uint32_t>()!=0x53544131u) throw std::runtime_error("Invalid stall fixture");
        const auto initial=in.value<cudarl::deep_pond::State>();const int initial_target=in.value<int>();in.finish();
        const auto policy=cudarl::rl::SelectorPolicy::load(argv[2]);
        cudarl::nn::CpuMlp<float> motor(cudarl::rl::swimmer_specification(4,33)),selector(cudarl::rl::selector_spec());motor.params()=policy.motor;selector.params()=policy.selector;
        std::printf("STALL env=%u steps=%d since_food=%.1f selected=%d angles=%.3f,%.3f,%.3f,%.3f\n",initial.environment,initial.steps,initial.since_food*.05,initial_target,initial.body.angle[0],initial.body.angle[1],initial.body.angle[2],initial.body.angle[3]);
        for(int mode=0;mode<8;++mode) {
            auto state=initial;int target=initial_target;float first=-1;
            for(int tick=0;tick<600;++tick) {
                float o[36];cudarl::pond_selector::observe(state,target,o);
                if(mode==0 || (mode>=2 && tick>=20)) {
                    if((state.steps%policy.decision_steps)==0) {
                        float scores[3];for(int f=0;f<3;++f) {float x[32];cudarl::pond_selector::features(o,f,x);selector.forward(x,1);scores[f]=selector.output()[0];}
                        target=0;for(int f=1;f<3;++f) if(scores[f]>scores[target]) target=f;
                    }
                } else if(mode==1) target=cudarl::demos::nearest_pond_food(o);
                else target=(mode-2)%3;
                if(mode>=5) target=(mode-5)%3;
                cudarl::demos::primary_pond_food(o,target);motor.forward(o,1);const auto result=cudarl::deep_pond::advance(state,motor.output());
                if(result.eaten_slot>=0 && first<0) first=(tick+1)*.05f;
            }
            std::printf("RECOVERY mode=%d foods=%d first_food_seconds=%.2f final_since_food=%.2f\n",mode,state.eaten-initial.eaten,first,state.since_food*.05);
        }
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Recovery probe: %s\n",e.what());return 1;}
}
