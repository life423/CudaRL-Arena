#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "rl/swimmer_ppo.h"
#include "demos/foraging_pond.h"
int main(int argc,char** argv) {
    try {
        if(argc<2) throw std::invalid_argument("Usage: foraging_experiment <output directory> [updates=400] [seeds=3]");
        const std::filesystem::path directory=argv[1];std::filesystem::create_directories(directory);
        const int updates=argc>2?std::stoi(argv[2]):400,seeds=argc>3?std::stoi(argv[3]):3;
        std::ofstream csv(directory/"results.csv");
        csv<<"seed,update,training_steps,trials,seconds_per_trial,total_food,food_per_minute,split\n";
        for(int seed=1;seed<=seeds;++seed) {
            cudarl::rl::SwimmerConfig cfg;cfg.seed=static_cast<uint32_t>(seed);cfg.body=cudarl::swimmer::body_preset(8);cfg.evaluation_every=25;
            cudarl::rl::ForagingTrainer trainer(cfg);
            auto record=[&](const char* split) {
                const double rate=static_cast<double>(trainer.evaluation_foods)/trainer.evaluation_count;
                csv<<seed<<','<<trainer.update_id<<','<<static_cast<long long>(trainer.update_id)*trainer.batch<<','
                   <<trainer.evaluation_count<<",60,"<<trainer.evaluation_foods<<','<<rate<<','<<split<<'\n';csv.flush();
                std::cout<<"seed="<<seed<<" update="<<trainer.update_id<<" food/min="<<rate<<" split="<<split<<std::endl;
            };
            record("comparison");
            for(int u=0;u<updates;++u) {trainer.update();if(trainer.update_id%25==0) record("comparison");}
            trainer.evaluate(0xB738019Du);record("held_out");
            const auto weights=trainer.actor_params();
            float learned_std[2]{};trainer.log_std.copy_to_host(learned_std,2);
            std::cout<<"POLICY std="<<std::exp(learned_std[0])<<','<<std::exp(learned_std[1])<<std::endl;
            std::ofstream actor(directory/("policy-seed-"+std::to_string(seed)+".json"));actor<<std::setprecision(9);
            actor<<"{\"inputs\":13,\"hidden\":32,\"outputs\":2,\"body_id\":8,\"food_count\":3,\"food_area\":["
                 <<cudarl::foraging::kFoodRadiusX<<','<<cudarl::foraging::kFoodRadiusY<<"],\"minimum_food_separation\":"
                 <<cudarl::foraging::kFoodSeparation<<",\"update\":"<<updates<<",\"parameters\":[";
            for(size_t i=0;i<weights.size();++i) actor<<(i?",":"")<<weights[i];actor<<"]}\n";
            for(uint32_t pond_seed=101;pond_seed<=105;++pond_seed) {
                cudarl::demos::ForagingPond pond;pond.set_policy(weights);pond.reset(pond_seed);
                int stationary=0,longest_stationary=0;
                for(int t=0;t<cudarl::foraging::kEpisodeSteps;++t) {
                    const auto before=pond.state().body;pond.step();const auto& after=pond.state().body;
                    const float movement=std::hypot(after.x-before.x,after.y-before.y)+std::fabs(after.angle[0]-before.angle[0])+std::fabs(after.angle[1]-before.angle[1]);
                    stationary=movement<1e-5f?stationary+1:0;longest_stationary=std::max(longest_stationary,stationary);
                }
                const auto& body=pond.state().body;
                std::cout<<"CPU seed="<<seed<<" pond="<<pond_seed<<" food/min="<<pond.state().eaten
                         <<" stationary_seconds="<<longest_stationary*0.05f<<" joints="<<body.angle[0]<<','<<body.angle[1]
                         <<" command="<<body.last_action[0]<<','<<body.last_action[1]<<std::endl;
            }
        }
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
