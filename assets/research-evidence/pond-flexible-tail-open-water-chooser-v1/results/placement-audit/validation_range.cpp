#include "habitat_config.h"
#include "bank_training_starts.h"
#include "rl/flexible_tail_bank_observation.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
int main(int argc,char**argv) {
    if(argc!=3)return 2;
    namespace t=cudarl::flexible_tail; namespace p=cudarl::physics;
    const auto rows=t::read_navigation_worlds(argv[1]); std::ifstream input(argv[2]);
    std::string line; std::getline(input,line);int n=0;
    while(std::getline(input,line)){
        std::istringstream s(line);std::vector<std::string> f;
        for(std::string x;std::getline(s,x,',');)f.push_back(x);
        if(f.size()!=9)return 3;
        if(f[0].find("moving_")!=0)continue;
        auto& source=rows.at(std::stoi(f[2]));
        const auto aim=p::axis_rotation({0,1,0},std::stof(f[6])*(t::pi/180.f)).rotate({1,0,0});
        const auto incoming=source.legs[source.active_leg].heading;
        const float yaw=std::atan2(p::cross(incoming,aim).y,p::dot(incoming,aim));
        t::chooser::World w{};
        t::chooser::initialize(w,source,{std::stof(f[3]),0,std::stof(f[5])},yaw);
        const auto v=t::chooser::horizontal(w.pose.com_velocity_m_s);const auto speed=p::norm(v);
        const auto gap=1000*t::submerged_gap_m(larger_pond,w.navigation.motion.body,w.pose);
        if(w.fault||std::abs(gap-std::stof(f[7]))>.001||std::abs(1000*speed-std::stof(f[8]))>.001)return 4;
        const auto range=t::bank::transport_range(w.navigation.motion.body,w.pose,larger_pond,v/speed);
        if(!range.bounded)return 5;
        std::cout<<f[1]<<','<<f[0]<<','<<std::setprecision(10)<<gap<<','<<range.distance_m*1000<<'\n';++n;
    }
    return n==4?0:6;
}
