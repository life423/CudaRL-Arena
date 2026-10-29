#include "rl/boulder_source.h"
#include "rl/shelter_probe.h"
#include "envs/swimmer/predator.h"
#include <cstdio>

int main(int argc,char** argv) {
    using namespace cudarl;
    try {
        if(argc!=2) throw std::invalid_argument("Usage: predator_motor_probe <full-boulder-training.bin>");
        const auto source=rl::load_boulder_source(argv[1]);
        // A fixed off-axis predator makes the rooted boulder's real occlusion
        // usable. This is a motor capability diagnostic, not a learned response.
        const deep::V eye{-8,4,-1.25f};
        std::vector<rl::ShelterProbeCase> cases;
        const float yaw[4]={0.f,1.5707963f,3.1415927f,-1.5707963f};
        const deep::V current[3]={{},{.005f,0,0},{.0075f,0,0}};
        for(int location=0;location<2;++location) for(int facing=0;facing<4;++facing) for(auto flow:current) {
            auto state=source.frame;
            // Independent evaluation starts, never a deployed world reset.
            state.pond.body.orientation=deep::axis_rotation({0,1,0},yaw[facing]);
            for(float& v:state.pond.body.velocity) v=0;
            for(float& a:state.pond.body.angle) a=0;
            auto cover=predator::cover(0,eye,state.habitat,{2});
            if(!cover.valid || cover.exposure>.05f) throw std::runtime_error("The fixed motor probe has no valid geometric cover target");
            shelter::Goal goal;goal.position=cover.position;goal.cover=0;goal.valid=true;
            bool found=false;
            // Select the first legal full-body start in each small geometric
            // region. Different headings need different tail clearance. This
            // constructs evaluation cases; it never relocates a live animal.
            for(int trial=0;trial<81 && !found;++trial) {
                const float x=(location==0?3.4f:6.4f)+(trial%9-4)*.2f;
                const float z=(location==0?2.f:-.2f)+(trial/9-4)*.2f;
                state.pond.body.position={x,-1.7f,z};
                found=predator::body_clearance(state.pond.body,state.habitat,{2})>=.08f &&
                    arch::water(state.pond.body.position,state.habitat,.3f,{2}) &&
                    deep::norm(state.pond.body.position-goal.position)>=1.3f;
            }
            if(!found) throw std::runtime_error("No physically valid start in the frozen motor probe region");
            cases.push_back({state,goal,eye,flow,{2}});
        }
        const auto results=rl::probe_shelter_motor(source.motor,cases);
        int arrived=0,held=0;
        for(size_t i=0;i<results.size();++i) {
            const auto& r=results[i];arrived+=r.arrival_step>=0;held+=r.longest_safe_dwell_ticks>=100;
            std::printf("PREDATOR_MOTOR case=%zu current_cm_s=%.2f arrival_s=%.2f min_goal=%.4f final_goal=%.4f mean_exposure=%.4f final_exposure=%.4f safe_dwell_s=%.2f longest_safe_s=%.2f clearance=%.4f contacts=%d finite=%d\n",
                i,deep::norm(cases[i].current)*100,r.arrival_step<0?-1.f:r.arrival_step*deep::dt,r.minimum_goal_distance,r.final_goal_distance,
                r.mean_exposure,r.last_exposure,r.safe_dwell_ticks*deep::dt,r.longest_safe_dwell_ticks*deep::dt,r.minimum_clearance,r.contacts,r.finite?1:0);
        }
        std::printf("PREDATOR_MOTOR_SUMMARY cases=%zu arrived=%d held_full_five_seconds=%d motor_update=%d learner_update=%d source=%llu frozen=1 trained=0\n",
            cases.size(),arrived,held,source.motor_update,source.learner_update,static_cast<unsigned long long>(source.fingerprint));
        return 0;
    } catch(const std::exception& error) {std::fprintf(stderr,"PREDATOR_MOTOR_ERROR %s\n",error.what());return 1;}
}
