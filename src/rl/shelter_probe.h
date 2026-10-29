#ifndef CUDARL_SHELTER_PROBE_H
#define CUDARL_SHELTER_PROBE_H
#include "envs/swimmer/shelter_motor.h"
#include <vector>

namespace cudarl::rl {
struct ShelterProbeCase {
    arch::State state{};
    shelter::Goal goal{};
    deep::V predator_eye{},current{};
    deep::Basin basin{2};
};
struct ShelterProbeResult {
    arch::State final_state{};
    float minimum_goal_distance=1e10f,mean_exposure=0,last_exposure=1;
    float final_goal_distance=1e10f,minimum_clearance=1e10f;
    int arrival_step=-1,safe_dwell_ticks=0,longest_safe_dwell_ticks=0,contacts=0;
    bool finite=true;
};
// Deliberately frozen capability diagnostic, with no PPO, resets or relocation.
std::vector<ShelterProbeResult> probe_shelter_motor(const std::vector<float>& motor,
    const std::vector<ShelterProbeCase>& cases,int approach_ticks=240,int dwell_ticks=100,shelter_motor::ActuatorScale scales={});
}
#endif
