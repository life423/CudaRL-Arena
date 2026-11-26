#pragma once
#include "rl/flexible_tail_chooser.h"
#if defined(CUDARL_THREE_DEPTHS)
#include "rl/flexible_tail_three_depth_map.h"
#else
#include "rl/flexible_tail_distant_map.h"
#endif

namespace cudarl::flexible_tail::distant {
inline bool geometry_matches(const pond_geometry_v1::Parameters& pond){
    const auto& g=distant_map::geometry;
    return pond.water_radius_x_m==g.water_radius_x_m&&pond.water_radius_z_m==g.water_radius_z_m&&
        pond.flat_radius_x_m==g.flat_radius_x_m&&pond.flat_radius_z_m==g.flat_radius_z_m&&
        pond.floor_y_m==g.floor_y_m&&pond.reference_scale_m==g.reference_scale_m;
}
// Statistical sensory field, not a destination, route, action mask or steering
// controller. Three 120-degree sectors tile the full measured-course circle.
template<class Memory>
CUDARL_HD inline void observe(const Memory& memory,physics::V position,physics::V course,
    float depth,int control,float (&out)[3]){
    for(auto& x:out)x=0;
    int depth_index=0;float depth_error=FLT_MAX;
    for(int i=0;i<distant_map::depth_count;++i){const float error=fabsf(depth-distant_map::depth_at(i));
        if(error<depth_error){depth_error=error;depth_index=i;}}
    if(depth_error>.00001f)return;
    course.y=0;const float length=physics::norm(course);
    if(length<.0001f)return;course=course/length;
    const physics::V bearings[3]{
        physics::axis_rotation({0,1,0},-2.f*pi/3.f).rotate(course),course,
        physics::axis_rotation({0,1,0},2.f*pi/3.f).rotate(course)};
    int counts[3]{};
    for(int word=0;word<distant_map::words;++word){
        const uint64_t mask=distant_map::word_at(depth_index,word);
        if(!mask)continue;
        for(int bit=0;bit<64;++bit){
            if(!(mask&(uint64_t{1}<<bit)))continue;
            const int cell=word*64+bit;
            const physics::V point{
                (cell%chooser::cells_x-chooser::cells_x/2+.5f)*chooser::cell_size_m,position.y,
                (cell/chooser::cells_x-chooser::cells_z/2+.5f)*chooser::cell_size_m};
            const auto delta=point-position;
            if(physics::dot(delta,delta)<=.24f*.24f)continue;
            int sector=0;float best=physics::dot(bearings[0],delta);
            for(int side=1;side<3;++side){const float score=physics::dot(bearings[side],delta);
                if(score>best){best=score;sector=side;}}
            out[sector]+=chooser::novelty_at(memory,point,control);++counts[sector];
        }
    }
    for(int i=0;i<3;++i)if(counts[i])out[i]/=float(counts[i]);
}
}
