#ifndef CUDARL_DEMOS_FORAGING_POND_H
#define CUDARL_DEMOS_FORAGING_POND_H
#include <cmath>
#include <stdexcept>
#include "envs/swimmer/foraging.h"
#include "nn/mlp_cpu.h"
namespace cudarl::demos {
class ForagingPond {
public:
    explicit ForagingPond(swimmer::Body body=swimmer::body_preset(8)):
        policy_(nn::MlpSpec({foraging::kObs,32,32,2},
            {nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity})),body_(body) {}
    void set_policy(const std::vector<float>& weights) {
        if(weights.size()!=policy_.params().size()) throw std::invalid_argument("Three-food policy architecture mismatch");
        for(float weight:weights) if(!std::isfinite(weight)) throw std::invalid_argument("Non-finite three-food policy");
        policy_.params()=weights;
    }
    void reset(uint32_t seed=1) {state_=foraging::initial(seed,0,0,body_);elapsed_=0;}
    bool step() {
        float observation[foraging::kObs];foraging::observe(state_,observation);
        policy_.forward(observation,1);
        const auto result=foraging::advance(state_,policy_.output());
        elapsed_+=swimmer::kDt;
        return result.eaten_slot>=0;
    }
    const foraging::State& state() const {return state_;}
    float elapsed() const {return elapsed_;}
private:
    nn::CpuMlp<float> policy_;
    swimmer::Body body_;
    foraging::State state_;
    float elapsed_=0;
};
}
#endif
