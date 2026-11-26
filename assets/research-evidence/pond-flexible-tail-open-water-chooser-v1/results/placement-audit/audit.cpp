#include "habitat_config.h"
#include "bank_training_starts.h"
#include "rl/flexible_tail_bank_observation.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <array>
#include <stdexcept>

namespace t=cudarl::flexible_tail;
namespace p=cudarl::physics;
namespace r=cudarl::rng;
struct Range {
    double lo=1.e10,hi=-1.e10;
    void add(double v){if(!std::isfinite(v))throw std::runtime_error("nonfinite sample");lo=std::min(lo,v);hi=std::max(hi,v);}
};
struct Stats {int n=0,left=0,right=0;Range speed,slant,runway,gap,angle_error,position_error,outward,certified;};
int main(int argc,char**argv){
    if(argc!=2)return 2;
    auto rows=t::read_navigation_worlds(argv[1]);if(rows.size()!=256)return 3;
    const std::array<float,3> depths={-.0800000131f,-.075000003f,-.0720000044f};
    std::array<Stats,3> totals;
    for(uint32_t episode=0;episode<3;++episode)for(uint32_t lane=0;lane<1024;++lane){
        if((lane/2)%4>=2)continue;
        const int d=(lane/2)%3;
        const uint32_t draw=r::hash_u32(lane^r::hash_u32(episode));
        const auto row=12+r::bits(20261124,draw,3)%(rows.size()-12);
        auto source=rows.at(row);
        const float dy=depths[d]-source.legs[0].goal_m.y;
        source.motion.body.position.y+=dy/p::scale;
        for(auto& sample:source.motion.history)sample.y+=dy;
        for(int i=0;i<source.leg_count;++i)source.legs[i].goal_m.y+=dy;
        t::prepare_navigation(source);
        t::bank_training::ApproachEvidence proof{};
        p::V pos{};float yaw=0;
        if(!t::bank_training::moving_start(lane,episode,source,larger_pond,pos,yaw,proof))return 4;
        t::chooser::World placed{};t::chooser::initialize(placed,source,pos,yaw);
        if(placed.fault)return 5;
        const auto v=t::chooser::horizontal(placed.pose.com_velocity_m_s);
        const auto speed=p::norm(v),ax=float(larger_pond.water_radius_x_m),az=float(larger_pond.water_radius_z_m);
        const auto direction=v/speed,center=t::chooser::horizontal(placed.pose.com_m);
        const auto endpoint=center+direction*proof.straight_runway_m;
        auto normal=p::V{endpoint.x/(ax*ax),0,endpoint.z/(az*az)};
        normal=normal/p::norm(normal);
        const auto tangent=p::cross({0,1,0},normal);
        const double measured=std::atan2(p::dot(direction,normal),std::abs(p::dot(direction,tangent)))*180./t::pi;
        auto& s=totals[d];++s.n;
        if(p::dot(direction,tangent)>0)++s.left;else ++s.right;
        s.slant.add(measured);s.outward.add(p::dot(v,normal)*1000);s.speed.add(speed*1000);
        s.runway.add(proof.straight_runway_m*1000);
        const auto traced=t::bank::transport_range(placed.navigation.motion.body,placed.pose,larger_pond,direction);
        if(!traced.bounded)return 7;
        s.certified.add(1000*traced.distance_m);
        s.gap.add(1000*t::submerged_gap_m(larger_pond,placed.navigation.motion.body,placed.pose));
        s.angle_error.add(std::abs(measured-proof.slant_deg));
        s.position_error.add(p::norm(pos-center));
        if(p::dot(direction,tangent)*proof.inward_side<=0 || p::dot(v,normal)<=0 || s.angle_error.hi>.002)return 6;
    }
    std::cout<<"depth_mm,placements,positive_tangents,negative_tangents,min_actual_COM_mm_s,max_actual_COM_mm_s,min_outward_mm_s,max_outward_mm_s,min_actual_slant_deg,max_actual_slant_deg,max_angle_delta_deg,min_static_runway_mm,max_static_runway_mm,min_initial_certified_gap_mm,max_initial_certified_gap_mm,min_certified_horizontal_mm,max_certified_horizontal_mm\n";
    for(int d:{2,1,0}){const auto& s=totals[d];std::cout<<std::setprecision(9)<<std::lround(-1000*depths[d])<<','<<s.n<<','<<s.left<<','<<s.right<<','<<s.speed.lo<<','<<s.speed.hi<<','<<s.outward.lo<<','<<s.outward.hi<<','<<s.slant.lo<<','<<s.slant.hi<<','<<s.angle_error.hi<<','<<s.runway.lo<<','<<s.runway.hi<<','<<s.gap.lo<<','<<s.gap.hi<<','<<s.certified.lo<<','<<s.certified.hi<<"\n";}
}
