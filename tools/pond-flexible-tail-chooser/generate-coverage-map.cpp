#include "habitat_config.h"
#include "rl/flexible_tail_chooser.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc,char** argv){try{
    if(argc<4)throw std::runtime_error("Usage: generate_coverage_map STARTS HEADER REPORT [EXTRA_DEPTH_MM ...]");
    namespace t=cudarl::flexible_tail;namespace c=t::chooser;
    const auto starts=t::read_navigation_worlds(argv[1]);
    if(starts.empty())throw std::runtime_error("Missing archived body geometry");
    std::vector<float> depths;
    for(const auto& start:starts){const float depth=start.legs[start.active_leg].goal_m.y;
        if(std::none_of(depths.begin(),depths.end(),[&](float d){return std::abs(d-depth)<1.e-6f;}))depths.push_back(depth);}
    for(int i=4;i<argc;++i){
        size_t used=0;
        const std::string argument=argv[i];
        const float mm=std::stof(argument,&used);
        if(used!=argument.size() || !std::isfinite(mm) || mm<=0 || mm>=-1000*larger_pond.floor_y_m)
            throw std::runtime_error("Depth must lie between surface and floor, in millimetres");
        const float y=-mm*.001f;
        if(std::none_of(depths.begin(),depths.end(),[&](float d){return std::abs(d-y)<1.e-6f;}))depths.push_back(y);
    }
    std::sort(depths.begin(),depths.end());
    std::vector<uint64_t> masks;std::vector<int> counts;
    for(float depth:depths){
        auto reference=starts.front();
        const float dy=depth-t::snapshot(reference.motion.body).com_m.y;
        reference.motion.body.position.y+=dy/cudarl::physics::scale;
        for(int leg=0;leg<reference.leg_count;++leg){reference.legs[leg].goal_m.y=depth;}
        constexpr int words=c::cells_x*c::cells_z/64;
        std::vector<uint64_t> mask(words);int count=0;
        for(int cell=0;cell<c::cells_x*c::cells_z;++cell){
            const float x=(cell%c::cells_x-c::cells_x/2+.5f)*c::cell_size_m;
            const float z=(cell/c::cells_x-c::cells_z/2+.5f)*c::cell_size_m;
            for(int yaw=0;yaw<8;++yaw){c::World world{};
                c::initialize(world,reference,{x,0,z},yaw*t::pi/4.f);
                if(!world.fault&&t::submerged_gap_m(larger_pond,world.navigation.motion.body,world.pose)>0){mask[cell/64]|=uint64_t{1}<<(cell%64);++count;break;}
            }
        }
        if(!count)throw std::runtime_error("Empty geometric coverage map");
        masks.insert(masks.end(),mask.begin(),mask.end());counts.push_back(count);
    }
    cudarl::checkpoint::Writer identity;identity.value(larger_pond);identity.array(depths);identity.array(masks);
    const auto id=cudarl::checkpoint::checksum(identity.data);
    std::ofstream out(argv[2]);out.exceptions(std::ios::badbit|std::ios::failbit);
    out<<"#pragma once\n// Generated geometric sensing cache; no physics steps, policy actions or paths.\nnamespace cudarl::flexible_tail::distant_map {\ninline constexpr uint64_t identity="<<id<<"ULL;\ninline constexpr pond_geometry_v1::Parameters geometry={"<<std::setprecision(17)
       <<larger_pond.water_radius_x_m<<','<<larger_pond.water_radius_z_m<<','<<larger_pond.flat_radius_x_m<<','<<larger_pond.flat_radius_z_m<<','<<larger_pond.floor_y_m<<','<<larger_pond.reference_scale_m<<"};\ninline constexpr int depth_count="<<depths.size()<<", words=24;\ninline constexpr float depths[]={"<<std::setprecision(9);
    for(float d:depths)out<<d<<"f,";out<<"};\ninline constexpr uint64_t masks[]={";
    for(auto word:masks)out<<word<<"ULL,";out<<"};\n#if defined(__CUDACC__)\nstatic __device__ __constant__ uint64_t device_masks[]={";
    for(auto word:masks)out<<word<<"ULL,";out<<"};\nstatic __device__ __constant__ float device_depths[]={";
    for(float d:depths)out<<d<<"f,";out<<"};\n#endif\nCUDARL_HD inline float depth_at(int i){\n#ifdef __CUDA_ARCH__\nreturn device_depths[i];\n#else\nreturn depths[i];\n#endif\n}\nCUDARL_HD inline uint64_t word_at(int depth,int word){\n#ifdef __CUDA_ARCH__\nreturn device_masks[depth*words+word];\n#else\nreturn masks[depth*words+word];\n#endif\n}\n}\n";
    std::ofstream report(argv[3]);report<<"{\"pass\":true,\"physics_steps\":0,\"optimizer_updates\":0,\"map_identity\":"<<id<<",\"depths\":[";
    for(size_t i=0;i<depths.size();++i){if(i)report<<',';report<<"{\"depth_y_m\":"<<depths[i]<<",\"cells\":"<<counts[i]<<"}";}report<<"]}\n";
    std::cout<<"Generated "<<depths.size()<<" geometric depth maps\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
