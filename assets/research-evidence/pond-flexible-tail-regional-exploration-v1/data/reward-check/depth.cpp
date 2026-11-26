#include "habitat_config.h"
#include "rl/flexible_tail_rate_chooser.h"
#include <iostream>
#include <iomanip>
#include <map>
namespace tail=cudarl::flexible_tail;
namespace ch=tail::rate_chooser;
namespace p=cudarl::physics;
int main(int argc,char** argv) {
    if(argc!=4) return 2;
    auto starts=tail::read_navigation_worlds(argv[1]);
    auto motor=tail::steering::read_model(argv[2]);
    tail::steering::Config config{}; config.study=tail::steering::Study::Rate;
    config.material=motor.material;
    struct Case{const char* label; int row; float x,z,deg;};
    std::cout<<"case,request_origin_depth_mm,left,forward,right\n";
    const Case cases[]={{"endurance_godot_rest",0,0,0,0},{"endurance_northeast",11,.2f,-.36f,23},{"endurance_west",6,-.35f,-.06f,128},{"endurance_southwest",7,-.16f,.29f,-78}};
    for (auto test:cases) {
        auto& source=starts.at(test.row);
        auto position=tail::chooser::horizontal(tail::snapshot(source.motion.body).com_m);
        float yaw=0;
        if(test.row) {
            position={test.x*float(larger_pond.water_radius_x_m),0,test.z*float(larger_pond.water_radius_z_m)};
            auto aim=p::axis_rotation({0.f,1.f,0.f},test.deg*tail::pi/180.f).rotate({1.f,0.f,0.f});
            auto incoming=source.legs[source.active_leg].heading;
            yaw=std::atan2(p::cross(incoming,aim).y,p::dot(incoming,aim));
        }
        ch::World w{};
        ch::initialize(w,source,position,yaw,motor.kind,config);
        if(w.fault) return 3;
        float input[ch::inputs]{};
        ch::observe(w,larger_pond,input,2);
        std::cout<<test.label<<','<<std::setprecision(8)<<-1000*w.motor.navigation.motion.request.origin_m.y<<','<<input[32]<<','<<input[33]<<','<<input[34]<<'\n';
    }
    std::map<int,int> training_histogram;
    for(auto& src:tail::read_navigation_worlds(argv[3])) {
        ch::World w{}; ch::initialize(w,src,tail::chooser::horizontal(tail::snapshot(src.motion.body).com_m),0,motor.kind,config);
        if(w.fault)return 4;
        ++training_histogram[std::lround(-1000000*w.motor.navigation.motion.request.origin_m.y)];
    }
    for(auto [microns,count]:training_histogram) std::cerr<<"training archive depth "<<microns<<" micrometres: "<<count<<" starts\n";
}
