#ifndef CUDARL_RL_EVALUATED_POLICY_H
#define CUDARL_RL_EVALUATED_POLICY_H
#include <cstdint>
#include <vector>
namespace cudarl::rl {
// An actor and its measured result captured together, before another update.
// This is a playback checkpoint, not critic/Adam state for resuming training.
struct EvaluatedPolicy {
    int update=-1,foods=0,count=0,steps=0;
    uint32_t evaluation_seed=0,training_seed=0;
    std::vector<float> parameters;
    bool shuffled_inputs=false;
    bool valid() const {return update>=0 && count>0 && steps>0 && !parameters.empty();}
    double food_per_minute() const {return valid()?foods*60.0/(count*steps*0.05):0.0;}
};
}
#endif
