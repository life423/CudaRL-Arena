#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "envs/swimmer/foraging.h"
#include "nn/mlp_cpu.h"

namespace fg=cudarl::foraging;
namespace nn=cudarl::nn;
using Order=std::array<int,3>;
constexpr std::array<Order,6> orders{{{{0,1,2}},{{0,2,1}},{{1,0,2}},{{1,2,0}},{{2,0,1}},{{2,1,0}}}};

std::string order_name(const Order& order) {
    std::string name;for(int slot:order) name.push_back(static_cast<char>('A'+slot));return name;
}
void observe_order(const fg::State& state,const Order& order,float* observation) {
    float original[fg::kObs];fg::observe(state,original);
    std::copy(original,original+fg::kObs,observation);
    for(int slot=0;slot<3;++slot) {
        observation[slot*2]=original[order[slot]*2];
        observation[slot*2+1]=original[order[slot]*2+1];
    }
}
std::vector<float> load_weights(const std::filesystem::path& path) {
    std::ifstream file(path);if(!file) throw std::runtime_error("Cannot open frozen checkpoint");
    const std::string json((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
    const auto key=json.find("\"parameters\""),start=json.find('[',key),end=json.find(']',start);
    if(key==std::string::npos || start==std::string::npos || end==std::string::npos)
        throw std::runtime_error("Missing checkpoint parameter array");
    auto values=json.substr(start+1,end-start-1);std::replace(values.begin(),values.end(),',',' ');
    std::istringstream input(values);
    std::vector<float> weights((std::istream_iterator<float>(input)),std::istream_iterator<float>());
    if(weights.size()!=1570) throw std::runtime_error("Expected a 13-32-32-2 actor");
    for(float weight:weights) if(!std::isfinite(weight)) throw std::runtime_error("Non-finite checkpoint weight");
    return weights;
}
struct Result {
    int foods=0,first_food=-1,first_input=-1;
    std::array<int,3> physical{},input{};
    double first_seconds=-1,longest_wait=0;
    float final_x=0,final_y=0;
};
Result replay(nn::CpuMlp<float>& actor,uint32_t seed,uint32_t world,const Order& order) {
    // Physical food slots, placement counters and RNG streams are NEVER permuted.
    // Only the six food coordinates presented to the network change order.
    auto state=fg::initial(seed,world);
    Result result;
    for(int tick=0;tick<fg::kEpisodeSteps;++tick) {
        float observation[fg::kObs];observe_order(state,order,observation);
        actor.forward(observation,1);
        const auto outcome=fg::advance(state,actor.output());
        result.longest_wait=std::max(result.longest_wait,static_cast<double>(state.since_food_steps)*0.05);
        if(outcome.eaten_slot<0) continue;
        const int physical=outcome.eaten_slot;
        const int input=static_cast<int>(std::find(order.begin(),order.end(),physical)-order.begin());
        ++result.physical[physical];++result.input[input];
        if(result.first_food<0) {
            result.first_food=physical;result.first_input=input;result.first_seconds=(tick+1)*0.05;
        }
    }
    result.foods=state.eaten;result.final_x=state.body.x;result.final_y=state.body.y;
    return result;
}
struct Aggregate {
    double foods=0,delta=0,delta_squared=0,action_squared=0,max_action_change=0;
    int changed_actions=0,no_food=0;
    std::array<int,3> physical{},input{},first_physical{},first_input{};
};

int main(int argc,char** argv) {
    try {
        if(argc<3) throw std::invalid_argument("Usage: food_order_bias <frozen checkpoint JSON> <output directory> [held-out worlds=256]");
        const int held_out_count=argc>3?std::stoi(argv[3]):256;
        if(held_out_count<2 || held_out_count>4096) throw std::invalid_argument("Held-out worlds must be 2..4096");
        const auto weights=load_weights(argv[1]);
        nn::CpuMlp<float> actor(nn::MlpSpec({13,32,32,2},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}));
        actor.params()=weights;
        const std::filesystem::path directory=argv[2];std::filesystem::create_directories(directory);
        std::ofstream cases(directory/"cases.csv"),summary(directory/"summary.csv"),probes(directory/"action-probes.csv"),invariants(directory/"invariants.txt");
        cases<<"split,world,seed,order,food_per_minute,first_physical,first_input,first_seconds,longest_wait,physical_A,physical_B,physical_C,input_1,input_2,input_3\n";
        probes<<"split,world,order,raw_0,raw_1,clipped_0,clipped_1,max_clipped_delta_from_ABC\n";
        summary<<"split,worlds,order,mean_food_per_minute,paired_delta_from_ABC,delta_ci95_low,delta_ci95_high,action_rms_from_ABC,changed_action_worlds,no_food_worlds,physical_A,physical_B,physical_C,input_1,input_2,input_3,first_input_1,first_input_2,first_input_3\n";
        summary<<std::setprecision(10);cases<<std::setprecision(9);probes<<std::setprecision(9);
        int identity_repeats=0,body_observation_checks=0;
        for(int split=0;split<2;++split) {
            const std::string label=split==0?"comparison":"held_out";
            const uint32_t seed=split==0?0x51A7E123u:0x9E11B1A5u;
            const int count=split==0?64:held_out_count;
            std::array<Aggregate,6> aggregates{};
            int first_choice_changed=0,first_result_changed=0;
            for(int world=0;world<count;++world) {
                const auto initial=fg::initial(seed,static_cast<uint32_t>(world));
                float original[fg::kObs];fg::observe(initial,original);
                std::array<Result,6> results;
                std::array<float,2> identity_action{};
                bool choice_changed=false,result_changed=false;
                for(size_t permutation=0;permutation<orders.size();++permutation) {
                    const auto& order=orders[permutation];auto& totals=aggregates[permutation];
                    float observation[fg::kObs];observe_order(initial,order,observation);
                    for(int j=6;j<fg::kObs;++j) {
                        if(observation[j]!=original[j]) throw std::runtime_error("Body observation changed with food order");
                        ++body_observation_checks;
                    }
                    std::array<bool,3> present{};
                    for(int j=0;j<3;++j) {
                        if(present[order[j]] || observation[j*2]!=original[order[j]*2] || observation[j*2+1]!=original[order[j]*2+1])
                            throw std::runtime_error("Permutation lost or changed food coordinates");
                        present[order[j]]=true;
                    }
                    actor.forward(observation,1);
                    const std::array<float,2> raw{{actor.output()[0],actor.output()[1]}};
                    const std::array<float,2> action{{std::clamp(raw[0],-1.f,1.f),std::clamp(raw[1],-1.f,1.f)}};
                    if(permutation==0) identity_action=action;
                    double maximum_change=0;
                    for(int j=0;j<2;++j) {
                        const double delta=action[j]-identity_action[j];
                        totals.action_squared+=delta*delta;maximum_change=std::max(maximum_change,std::fabs(delta));
                    }
                    totals.max_action_change=std::max(totals.max_action_change,maximum_change);
                    if(maximum_change>0.1) ++totals.changed_actions;
                    probes<<label<<','<<world<<','<<order_name(order)<<','<<raw[0]<<','<<raw[1]<<','<<action[0]<<','<<action[1]<<','<<maximum_change<<'\n';
                    results[permutation]=replay(actor,seed,static_cast<uint32_t>(world),order);
                    const auto& result=results[permutation];
                    if(world<4 && permutation==0) {
                        const auto repeated=replay(actor,seed,static_cast<uint32_t>(world),order);
                        if(repeated.foods!=result.foods || repeated.first_food!=result.first_food || repeated.final_x!=result.final_x || repeated.final_y!=result.final_y)
                            throw std::runtime_error("Identity replay is not deterministic");
                        ++identity_repeats;
                    }
                    totals.foods+=result.foods;
                    const double delta=result.foods-results[0].foods;
                    totals.delta+=delta;totals.delta_squared+=delta*delta;
                    if(result.first_food<0) ++totals.no_food;
                    else {++totals.first_physical[result.first_food];++totals.first_input[result.first_input];}
                    for(int j=0;j<3;++j) {totals.physical[j]+=result.physical[j];totals.input[j]+=result.input[j];}
                    if(permutation>0 && result.first_food!=results[0].first_food) {
                        result_changed=true;
                        if(result.first_food>=0 && results[0].first_food>=0) choice_changed=true;
                    }
                    cases<<label<<','<<world<<','<<seed<<','<<order_name(order)<<','<<result.foods<<','<<result.first_food<<','<<result.first_input<<','<<result.first_seconds<<','<<result.longest_wait;
                    for(int value:result.physical) cases<<','<<value;
                    for(int value:result.input) cases<<','<<value;
                    cases<<'\n';
                }
                first_choice_changed+=choice_changed?1:0;first_result_changed+=result_changed?1:0;
            }
            for(size_t i=0;i<orders.size();++i) {
                const auto& totals=aggregates[i];
                const double delta=totals.delta/count;
                const double variance=std::max(0.0,(totals.delta_squared-count*delta*delta)/(count-1));
                const double half_width=1.96*std::sqrt(variance/count);
                summary<<label<<','<<count<<','<<order_name(orders[i])<<','<<totals.foods/count<<','<<delta<<','<<delta-half_width<<','<<delta+half_width<<','
                       <<std::sqrt(totals.action_squared/(count*2))<<','<<totals.changed_actions<<','<<totals.no_food;
                for(int value:totals.physical) summary<<','<<value;
                for(int value:totals.input) summary<<','<<value;
                for(int value:totals.first_input) summary<<','<<value;
                summary<<'\n';
                std::cout<<label<<" order="<<order_name(orders[i])<<" food/min="<<totals.foods/count<<" paired_delta="<<delta
                         <<" command_changed="<<totals.changed_actions<<'/'<<count<<std::endl;
            }
            invariants<<label<<" worlds="<<count<<" first_physical_choice_changed="<<first_choice_changed
                      <<" first_choice_or_success_changed="<<first_result_changed<<'\n';
        }
        invariants<<"identity_repeats_passed="<<identity_repeats<<" body_observation_scalar_checks="<<body_observation_checks
                  <<" no_training=true no_policy_overwrite=true food_permutation_only=true\n";
        if(actor.params()!=weights) throw std::runtime_error("Evaluation modified policy weights");
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
