#pragma once
#include <cstdint>
namespace pond_collection {
struct Trace {
    float x=0,z=0,vx=0,vy=0,vz=0,depth=0,tilt=0;
    double minimum_gap=0;
    uint32_t episode=0,control=0,kind=0,occupied_cells=0,occupied_grid[8]{};
    float bank_near_seconds=0,contact_seconds=0,postwarmup_stall_seconds=0;
    float heading_x=0,heading_z=0,requested_angle=0;
    uint32_t request_serial=0,onset_control=0,request_age=0,qualifying_streak=0,events=0,active=0,succeeded=0,long_hold=0;
    uint64_t onsets=0,successes=0,timeouts=0,interrupted=0,early_terminal=0,long_holds=0;
};
static_assert(sizeof(Trace)==192);
}
