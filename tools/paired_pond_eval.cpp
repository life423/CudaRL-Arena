#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "demos/foraging_pond.h"

namespace fg=cudarl::foraging;
namespace nn=cudarl::nn;
struct Policy {int update;std::vector<float> weights;};
Policy load(const std::filesystem::path& path) {
    std::ifstream file(path);if(!file) throw std::runtime_error("Cannot open checkpoint");
    const std::string json((std::istreambuf_iterator<char>(file)),{});
    auto integer=[&](const char* name) {
        const auto key=json.find(std::string("\"")+name+"\"");
        if(key==std::string::npos) throw std::runtime_error("Missing checkpoint metadata");
        return std::stoi(json.substr(json.find(':',key)+1));
    };
    if(integer("version")!=1 || integer("inputs")!=13 || integer("hidden")!=32 || integer("outputs")!=2 ||
       integer("body_id")!=8 || integer("count")!=64 || integer("steps")!=1200 ||
       json.find("three-food-regional-inertial-v1")==std::string::npos || json.find("deterministic_mean")==std::string::npos)
        throw std::runtime_error("Incompatible checkpoint protocol");
    const auto start=json.find('[',json.find("\"parameters\"")),end=json.find(']',start);
    if(start==std::string::npos || end==std::string::npos) throw std::runtime_error("Missing weights");
    auto values=json.substr(start+1,end-start-1);std::replace(values.begin(),values.end(),',',' ');
    std::istringstream input(values);Policy policy{integer("update"),std::vector<float>((std::istream_iterator<float>(input)),{})};
    if(policy.weights.size()!=1570) throw std::runtime_error("Wrong actor size");
    for(float value:policy.weights) if(!std::isfinite(value)) throw std::runtime_error("Non-finite actor weights");
    return policy;
}
struct Result {
    int foods=0,first_tick=-1,max_wait=0;
    float x=0,y=0,heading=0;
    double rate(int ticks) const {return foods*1200.0/ticks;}
};
nn::CpuMlp<float> actor(const Policy& policy) {
    nn::CpuMlp<float> model(nn::MlpSpec({13,32,32,2},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}));
    model.params()=policy.weights;return model;
}
Result replay(nn::CpuMlp<float>& model,uint32_t seed,uint32_t world,int ticks) {
    auto state=fg::initial(seed,world);Result result;
    for(int t=0;t<ticks;++t) {
        float observation[13];fg::observe(state,observation);model.forward(observation,1);
        // Exactly the CPU observation -> actor -> physics path shown in Godot.
        const auto outcome=fg::advance(state,model.output());
        result.max_wait=std::max(result.max_wait,state.since_food_steps);
        if(outcome.eaten_slot>=0 && result.first_tick<0) result.first_tick=t+1;
    }
    result.foods=state.eaten;result.x=state.body.x;result.y=state.body.y;result.heading=state.body.heading;
    if(!std::isfinite(result.x) || !std::isfinite(result.y)) throw std::runtime_error("Non-finite physical state");
    return result;
}
bool same(const Result& a,const Result& b) {
    return a.foods==b.foods && a.first_tick==b.first_tick && a.max_wait==b.max_wait && a.x==b.x && a.y==b.y && a.heading==b.heading;
}
struct Paired {Result original,shuffled;};
void summarize(std::ofstream& csv,const std::string& split,int ticks,const std::vector<Paired>& pairs) {
    double original=0,shuffled=0,delta_squared=0;int wins=0,losses=0,ties=0;
    std::array<std::vector<double>,2> first,wait;
    int no_food[2]{};
    for(const auto& pair:pairs) {
        const double a=pair.original.rate(ticks),b=pair.shuffled.rate(ticks),delta=b-a;
        original+=a;shuffled+=b;delta_squared+=delta*delta;
        if(delta>0) ++wins;else if(delta<0) ++losses;else ++ties;
        const Result values[2]={pair.original,pair.shuffled};
        for(int i=0;i<2;++i) {
            if(values[i].first_tick>=0) first[i].push_back(values[i].first_tick*0.05);else ++no_food[i];
            wait[i].push_back(values[i].max_wait*0.05);
        }
    }
    const double n=static_cast<double>(pairs.size()),delta=(shuffled-original)/n;
    const double variance=std::max(0.0,(delta_squared-n*delta*delta)/(n-1));
    const double half=1.96*std::sqrt(variance/n);
    auto median=[](std::vector<double> values) {
        if(values.empty()) return -1.0;std::sort(values.begin(),values.end());
        const size_t mid=values.size()/2;return values.size()%2?values[mid]:(values[mid-1]+values[mid])*0.5;
    };
    auto p95=[](std::vector<double> values) {
        std::sort(values.begin(),values.end());return values[static_cast<size_t>(std::ceil(values.size()*0.95))-1];
    };
    csv<<split<<','<<pairs.size()<<','<<ticks*0.05<<','<<original/n<<','<<shuffled/n<<','<<delta<<','<<delta-half<<','<<delta+half
       <<','<<wins<<','<<ties<<','<<losses<<','<<no_food[0]<<','<<no_food[1]<<','<<median(first[0])<<','<<median(first[1])
       <<','<<p95(wait[0])<<','<<p95(wait[1])<<'\n';csv.flush();
    std::cout<<split<<" worlds="<<pairs.size()<<" seconds="<<ticks*0.05<<" original="<<original/n<<" shuffled="<<shuffled/n
             <<" paired_delta="<<delta<<" approx95CI=["<<delta-half<<','<<delta+half<<"] W/T/L="<<wins<<'/'<<ties<<'/'<<losses<<std::endl;
}
int main(int argc,char** argv) {
    try {
        if(argc<4) throw std::invalid_argument("Usage: paired_pond_eval <original.json> <shuffled.json> <output directory> [worlds per seed=256] [five-minute worlds=64]");
        const auto original=load(argv[1]),shuffled=load(argv[2]);
        const int count=argc>4?std::stoi(argv[4]):256,long_count=argc>5?std::stoi(argv[5]):64;
        if(count<2 || count>4096 || long_count<0 || long_count==1 || long_count>4096) throw std::invalid_argument("Invalid trial counts");
        const std::filesystem::path directory=argv[3];std::filesystem::create_directories(directory);
        std::ofstream cases(directory/"cases.csv"),summary(directory/"summary.csv"),invariants(directory/"invariants.txt");
        if(!cases || !summary || !invariants) throw std::runtime_error("Cannot create result files");
        cases<<std::setprecision(10);summary<<std::setprecision(10);
        cases<<"split,seed,world,seconds,original_foods,shuffled_foods,original_food_per_minute,shuffled_food_per_minute,paired_delta,original_first_seconds,shuffled_first_seconds,original_max_wait_seconds,shuffled_max_wait_seconds\n";
        summary<<"split,worlds,seconds,original_food_per_minute,shuffled_food_per_minute,paired_delta,delta_ci95_low,delta_ci95_high,shuffled_wins,ties,original_wins,original_no_food,shuffled_no_food,original_median_first_seconds,shuffled_median_first_seconds,original_p95_max_wait_seconds,shuffled_p95_max_wait_seconds\n";
        auto a=actor(original),b=actor(shuffled);std::vector<Paired> combined;int repeats=0,playback_checks=0;
        const uint32_t seeds[3]={0xC73B2A91u,0x4F092D67u,0xB5D843E9u};
        for(int block=0;block<3;++block) {
            const int worlds=block==2?long_count:count,ticks=block==2?6000:1200;
            if(worlds==0) continue;
            const std::string split=block==0?"fresh_A":(block==1?"fresh_B":"sustained");
            std::vector<Paired> results;
            for(int world=0;world<worlds;++world) {
                const auto key=static_cast<uint32_t>(world);const Paired pair{replay(a,seeds[block],key,ticks),replay(b,seeds[block],key,ticks)};
                if(world<4) {
                    if(!same(pair.original,replay(a,seeds[block],key,ticks)) || !same(pair.shuffled,replay(b,seeds[block],key,ticks)))
                        throw std::runtime_error("Identity replay failed");repeats+=2;
                }
                if(world==0) {
                    const Policy policies[2]={original,shuffled};const Result outcomes[2]={pair.original,pair.shuffled};
                    for(int i=0;i<2;++i) {
                        cudarl::demos::ForagingPond pond;pond.set_policy(policies[i].weights);pond.reset(seeds[block]);
                        for(int t=0;t<ticks;++t) pond.step();
                        const auto& state=pond.state();
                        if(state.eaten!=outcomes[i].foods || state.body.x!=outcomes[i].x || state.body.y!=outcomes[i].y || state.body.heading!=outcomes[i].heading)
                            throw std::runtime_error("Evaluation differs from visible CPU playback");++playback_checks;
                    }
                }
                if(original.weights==shuffled.weights && !same(pair.original,pair.shuffled)) throw std::runtime_error("Equal actors produced unequal paired outcomes");
                results.push_back(pair);
                cases<<split<<','<<seeds[block]<<','<<world<<','<<ticks*0.05<<','<<pair.original.foods<<','<<pair.shuffled.foods
                     <<','<<pair.original.rate(ticks)<<','<<pair.shuffled.rate(ticks)<<','<<pair.shuffled.rate(ticks)-pair.original.rate(ticks)
                     <<','<<(pair.original.first_tick<0?-1:pair.original.first_tick*0.05)<<','<<(pair.shuffled.first_tick<0?-1:pair.shuffled.first_tick*0.05)
                     <<','<<pair.original.max_wait*0.05<<','<<pair.shuffled.max_wait*0.05<<'\n';
                if((world+1)%64==0) std::cout<<split<<" progress="<<world+1<<'/'<<worlds<<std::endl;
            }
            summarize(summary,split,ticks,results);
            if(block<2) combined.insert(combined.end(),results.begin(),results.end());
        }
        summarize(summary,"fresh_combined",1200,combined);
        if(a.params()!=original.weights || b.params()!=shuffled.weights) throw std::runtime_error("Evaluation changed actor weights");
        invariants<<"original_update="<<original.update<<" shuffled_update="<<shuffled.update<<" identity_repeats="<<repeats
                  <<" visible_cpu_playback_checks="<<playback_checks<<" equal_actor_control="<<(original.weights==shuffled.weights)
                  <<" no_training=true canonical_observations=true unchanged_physics=true actor_weights_unchanged=true\n";
        if(!cases || !summary || !invariants) throw std::runtime_error("Result write failed");
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<std::endl;return 1;}
}
