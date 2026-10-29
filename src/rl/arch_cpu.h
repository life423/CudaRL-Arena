#ifndef CUDARL_ARCH_CPU_H
#define CUDARL_ARCH_CPU_H
#include "envs/swimmer/arch.h"
#include "progress_selector_policy.h"
#include "nn/mlp_cpu.h"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudarl::arch {
// Reference inference uses the same habitat, physical integration, candidate
// scoring and time-limit observation ordering as the CUDA arch batch.
class CpuBatch {
    nn::CpuMlp<float> selector_{rl::progress_selector_spec()};
    static size_t population(int n) {
        if(n<1 || n>2048) throw std::invalid_argument("Invalid CPU arch population");
        return static_cast<size_t>(n);
    }
public:
    std::vector<State> states;
    std::vector<float> observations;
    uint32_t seed;
    CpuBatch(int n,uint32_t key,const std::vector<float>& selector):
        states(population(n)),observations(states.size()*kObs),seed(key) {
        if(selector.size()!=static_cast<size_t>(selector_.spec().num_params()))
            throw std::invalid_argument("Invalid CPU arch selector dimensions");
        selector_.params()=selector;reset();
    }
    void reset() {
        for(size_t i=0;i<states.size();++i) states[i]=initial(seed,static_cast<uint32_t>(i),0,i%4==0);
        prepare();
    }
    void prepare() {
        std::vector<float> input;std::vector<std::pair<size_t,int>> rows;
        for(size_t i=0;i<states.size();++i) if(states[i].pond.decision) {
            const auto& pond=states[i].pond;
            for(int food=0;food<feeding::capacity;++food) if(pond.pellets[food].active) {
                const auto offset=input.size();input.resize(offset+progress_selector::kFeatures);
                feeding::features(pond,food,input.data()+offset);rows.emplace_back(i,food);
            }
        }
        for(auto& state:states) if(state.pond.decision) state.pond.target=-1;
        if(!rows.empty()) {
            selector_.forward(input.data(),static_cast<int>(rows.size()));
            std::vector<float> best(states.size(),-1e30f);
            for(size_t row=0;row<rows.size();++row) {
                const auto [world,food]=rows[row];
                const float score=selector_.output()[row];
                if(score>best[world]) {best[world]=score;states[world].pond.target=food;}
            }
        }
        for(size_t i=0;i<states.size();++i) {
            states[i].pond.decision=false;arch::observe(states[i],observations.data()+i*kObs);
        }
    }
    void step(const float* actions,float* rewards=nullptr,float* done=nullptr,float* final_obs=nullptr,bool autoreset=true) {
        for(size_t i=0;i<states.size();++i) {
            const float reward=arch::advance(states[i],actions+i*kAct);
            const bool timeout=states[i].pond.steps>=kEpisodeSteps;
            if(rewards) rewards[i]=reward;
            if(done) done[i]=timeout?1.f:0.f;
            if(!autoreset || !timeout) arch::replenish(states[i]);
        }
        prepare();
        if(autoreset) {
            bool reset_any=false;
            for(size_t i=0;i<states.size();++i) if(states[i].pond.steps>=kEpisodeSteps) {
                if(final_obs) std::copy_n(observations.data()+i*kObs,kObs,final_obs+i*kObs);
                const uint32_t key=states[i].pond.seed,environment=states[i].pond.environment,episode=states[i].pond.episode;
                const bool retention=states[i].pond.retention;
                states[i]=initial(key,environment,episode+1,retention);
                reset_any=true;
            }
            if(reset_any) prepare();
        }
    }
};
}
#endif
