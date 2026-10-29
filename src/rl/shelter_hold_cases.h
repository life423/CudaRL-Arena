#ifndef CUDARL_SHELTER_HOLD_CASES_H
#define CUDARL_SHELTER_HOLD_CASES_H
#include "shelter_probe.h"

namespace cudarl::rl {
constexpr uint32_t shelter_hold_evaluation_seed=0x48D119A5u;
// Independent calm, perturbed-holding cases. These do not test far approach,
// a learned safety decision, or current compensation. Already concealed starts
// are valid for this lesson and must never be counted as escape successes.
inline std::vector<ShelterProbeCase> shelter_hold_cases(const arch::State& reference,uint32_t seed=shelter_hold_evaluation_seed) {
    std::vector<ShelterProbeCase> cases;const deep::V eye{-8,4,-1.25f};
    for(int goal=0;goal<predator::kCoverCount;++goal) for(int variant=0;variant<6;++variant) {
        auto state=arch::initial(seed,0,static_cast<uint32_t>(goal*6+variant),false,{2});state.habitat=reference.habitat;
        const auto point=predator::cover(goal,eye,state.habitat,{2});
        if(!point.valid) throw std::runtime_error("Calm holding evaluation requires legal cover geometry");
        bool placed=false;
        for(uint32_t attempt=0;attempt<256u && !placed;++attempt) {
            const uint32_t key=static_cast<uint32_t>(goal*6+variant)*4099u+attempt*11u;
            const auto draw=[&](uint32_t offset) {return 2.f*rng::uniform(seed,static_cast<uint32_t>(goal*6+variant),key+offset)-1.f;};
            state.pond.body=predator::cover_pose(point);
            const deep::V error{draw(0)*.12f,draw(1)*.06f,draw(2)*.12f};
            const float yaw_error=draw(3)*.12f;
            if(deep::norm(error)<.025f || fabsf(yaw_error)<.015f) continue;
            state.pond.body.position+=error;
            state.pond.body.orientation=deep::axis_rotation({0,1,0},point.yaw+yaw_error);
            placed=predator::body_clearance(state.pond.body,state.habitat,{2})>=.08f &&
                arch::water(state.pond.body.position,state.habitat,.2f,{2});
        }
        if(!placed) throw std::runtime_error("Could not generate a legal perturbed holding case");
        state.pond.target=-1;state.pond.decision=false;state.pond.anchor=state.pond.body.position;
        shelter::Goal target;target.position=point.position;target.cover=goal;target.valid=true;
        cases.push_back({state,target,eye,{},{2}});
    }
    return cases;
}
}
#endif
