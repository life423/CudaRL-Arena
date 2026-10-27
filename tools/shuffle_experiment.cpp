#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include "rl/swimmer_ppo.h"
namespace rl=cudarl::rl;
std::vector<float> load_reference(const std::filesystem::path& path,int& update) {
    std::ifstream file(path);if(!file) throw std::runtime_error("Reference not found");
    const std::string json((std::istreambuf_iterator<char>(file)),{});
    const auto update_key=json.find("\"update\"");
    if(update_key==std::string::npos) throw std::runtime_error("Missing reference update");
    update=std::stoi(json.substr(json.find(':',update_key)+1));
    const auto start=json.find('[',json.find("\"parameters\"")),end=json.find(']',start);
    if(start==std::string::npos || end==std::string::npos) throw std::runtime_error("Invalid reference weights");
    auto values=json.substr(start+1,end-start-1);std::replace(values.begin(),values.end(),',',' ');
    std::istringstream input(values);std::vector<float> weights((std::istream_iterator<float>(input)),{});
    if(weights.size()!=1570) throw std::runtime_error("Invalid reference architecture");return weights;
}
void save(const std::filesystem::path& path,const rl::EvaluatedPolicy& best) {
    std::ofstream out(path);out<<std::setprecision(9);
    out<<"{\"version\":1,\"protocol\":\"three-food-regional-inertial-v1\",\"inputs\":13,\"hidden\":32,\"outputs\":2,\"body_id\":8,"
       <<"\"inference\":\"deterministic_mean\",\"training_variant\":\""<<(best.shuffled_inputs?"shuffled":"canonical")<<"\","
       <<"\"update\":"<<best.update<<",\"foods\":"<<best.foods<<",\"count\":64,\"steps\":1200,\"evaluation_seed\":"<<best.evaluation_seed
       <<",\"training_seed\":"<<best.training_seed<<",\"food_per_minute\":"<<best.food_per_minute()<<",\"parameters\":[";
    for(size_t i=0;i<best.parameters.size();++i) out<<(i?",":"")<<best.parameters[i];out<<"]}\n";
    if(!out) throw std::runtime_error("Could not save experiment checkpoint");
}
int main(int argc,char** argv) {
    try {
        if(argc<3) throw std::invalid_argument("Usage: shuffle_experiment <directory> <reference.json> [updates=1000] [seeds=2]");
        const std::filesystem::path directory=argv[1];std::filesystem::create_directories(directory);
        int reference_update=0;const auto reference=load_reference(argv[2],reference_update);
        const int updates=argc>3?std::stoi(argv[3]):1000,seeds=argc>4?std::stoi(argv[4]):2;
        if(updates<=0 || seeds<=0) throw std::invalid_argument("Positive budgets required");
        std::ofstream curve(directory/"learning.csv"),scores(directory/"scores.csv");
        curve<<"variant,seed,update,env_steps,comparison_food_per_minute\n";
        scores<<"variant,seed,selected_update,comparison_food_per_minute,heldout_food_per_minute\n";
        for(int seed=1;seed<=seeds;++seed) for(bool shuffled:{false,true}) {
            rl::SwimmerConfig cfg;cfg.seed=static_cast<uint32_t>(seed);cfg.body=cudarl::swimmer::body_preset(8);
            cfg.evaluation_every=25;cfg.shuffle_food_inputs=shuffled;
            rl::ForagingTrainer trainer(cfg);const std::string variant=shuffled?"shuffled":"canonical";
            auto record=[&]() {
                curve<<variant<<','<<seed<<','<<trainer.update_id<<','<<static_cast<long long>(trainer.update_id)*trainer.batch<<','<<trainer.evaluation_foods/64.0<<'\n';curve.flush();
                std::cout<<variant<<" seed="<<seed<<" update="<<trainer.update_id<<" score="<<trainer.evaluation_foods/64.0<<std::endl;
            };
            record();for(int i=0;i<updates;++i) {trainer.update();if(trainer.update_id%25==0) record();}
            const auto best=trainer.best_policy();save(directory/(variant+"-seed-"+std::to_string(seed)+".json"),best);
            trainer.actor.set_params(best.parameters);trainer.evaluate(0x9E11B1A5u);
            scores<<variant<<','<<seed<<','<<best.update<<','<<best.food_per_minute()<<','<<trainer.evaluation_foods/64.0<<'\n';scores.flush();
        }
        rl::SwimmerConfig cfg;cfg.body=cudarl::swimmer::body_preset(8);rl::ForagingTrainer frozen(cfg);
        frozen.actor.set_params(reference);frozen.evaluate();const double comparison=frozen.evaluation_foods/64.0;
        frozen.evaluate(0x9E11B1A5u);scores<<"frozen_reference,1,"<<reference_update<<','<<comparison<<','<<frozen.evaluation_foods/64.0<<'\n';
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
