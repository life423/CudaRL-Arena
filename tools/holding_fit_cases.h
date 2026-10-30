#ifndef CUDARL_TOOLS_HOLDING_FIT_CASES_H
#define CUDARL_TOOLS_HOLDING_FIT_CASES_H
#include "envs/swimmer/shelter_motor.h"
#include <stdexcept>
#include <vector>

// Candidate metadata for a supervised representability DIAGNOSIS, not a
// deployed controller or a new production reset/training protocol. Candidate
// generation alone never authorizes a zero-action label: the caller must prove
// initial safety and all 200 unchanged GPU Batch steps before admitting it.
namespace cudarl::diagnostics {
constexpr uint32_t holding_fit_train_seed=0xC07A2026u;
constexpr uint32_t holding_fit_test_seed=0xC07B2026u;
constexpr int holding_fit_candidates_per_cover=32;
struct HoldingFitCase {
    shelter_motor::State world;
    uint32_t seed=0;
    int trajectory=0,cover=0;
    bool training=false;
};
inline bool holding_fit_initial_safe(const HoldingFitCase& item) {
    const auto& world=item.world;const auto& body=world.state.pond.body;
    const float gap=predator::body_clearance(body,world.state.habitat,{2});
    const float exposure=predator::exposure(body,world.request.eye,world.state.habitat,{2});
    return !world.state.pond.invalid && std::isfinite(gap) && std::isfinite(exposure) && gap>=.08f && exposure<=.05f;
}
inline std::vector<HoldingFitCase> holding_fit_candidates(const arch::State& reference,uint32_t seed,
                                                         int count_per_cover,bool training) {
    if(count_per_cover<1 || count_per_cover>holding_fit_candidates_per_cover)
        throw std::invalid_argument("Holding-fit candidate pools are bounded to 1..32 per cover");
    std::vector<HoldingFitCase> result;result.reserve(static_cast<size_t>(predator::kCoverCount*count_per_cover));
    for(int goal=0;goal<predator::kCoverCount;++goal) for(int candidate=0;candidate<count_per_cover;++candidate) {
        HoldingFitCase item;item.seed=seed;item.training=training;item.cover=goal;
        item.trajectory=goal*holding_fit_candidates_per_cover+candidate;
        auto& world=item.world;
        world.state=arch::initial(seed,0,static_cast<uint32_t>(item.trajectory),false,{2});
        world.state.habitat=reference.habitat;
        const auto point=predator::cover(goal,world.request.eye,world.state.habitat,{2});
        if(!point.valid) throw std::runtime_error("Holding fit needs the original valid nominal cover poses");
        world.request.cover=true;world.request.goal.position=point.position;
        world.request.goal.cover=goal;world.request.goal.valid=true;
        world.request.heading=deep::axis_rotation({0,1,0},point.yaw).rotate({1,0,0});
        world.state.pond.body=predator::cover_pose(point); // Zero joints and all ten velocities.
        if(!training || candidate!=0) {
            const auto draw=[&](uint32_t channel) {
                return 2.f*rng::uniform(seed,static_cast<uint32_t>(item.trajectory),channel)-1.f;
            };
            world.state.pond.body.position+=deep::V{draw(0)*.12f,draw(1)*.06f,draw(2)*.12f};
            world.state.pond.body.orientation=deep::axis_rotation({0,1,0},point.yaw+draw(3)*.06f);
        }
        auto& pond=world.state.pond;pond.steps=0;pond.since_food=0;pond.target=-1;pond.decision=false;
        pond.anchor=pond.body.position;world.held_ticks=0;world.peak_hold=0;
        world.exposure=predator::exposure(pond.body,world.request.eye,world.state.habitat,{2});
        // Each candidate gets one seeded perturbation, even if unsafe. Keep
        // rejected candidates in the finite pool; do not rejection-resample or
        // quietly expand the search after inspecting held-out outcomes.
        result.push_back(item);
    }
    return result;
}
}
#endif
