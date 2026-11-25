#pragma once
#include "rl/flexible_tail_distant_coverage.h"

namespace cudarl::flexible_tail::frontier {
// A bounded state potential, not a selected destination or action. The actual
// first-visit/revisit awards and physical motion are owned by the existing task.
template<class Memory>
CUDARL_HD inline float potential(const Memory& memory, physics::V position, float depth) {
    int layer=-1;
    for(int i=0;i<distant_map::depth_count;++i)
        if(fabsf(depth-distant_map::depth_at(i))<=.00001f){layer=i;break;}
    if(layer<0)return 0.f;
    float closest2=FLT_MAX;
    for(int word=0;word<distant_map::words;++word){
        const uint64_t unknown=distant_map::word_at(layer,word)&~memory.visited[word];
        if(!unknown)continue;
        for(int bit=0;bit<64;++bit){
            if(!(unknown&(uint64_t{1}<<bit)))continue;
            const int cell=word*64+bit;
            const float dx=(cell%chooser::cells_x-chooser::cells_x/2+.5f)*chooser::cell_size_m-position.x;
            const float dz=(cell/chooser::cells_x-chooser::cells_z/2+.5f)*chooser::cell_size_m-position.z;
            closest2=fminf(closest2,dx*dx+dz*dz);
        }
    }
    // All mapped cells visited is the maximum potential, not a reset of novelty.
    return closest2==FLT_MAX?2.f:2.f*expf(-sqrtf(closest2)/.25f);
}
CUDARL_HD inline float shaping(float before,float after,bool terminal,float discount) {
    return (terminal?0.f:discount*after)-before;
}
}
