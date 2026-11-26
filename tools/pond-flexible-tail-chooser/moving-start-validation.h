// Independent selection fixtures for the moving-start continuation.
namespace moving_start_validation {
struct MovingPlacement {
    const char* name;
    float azimuth, heading_offset, runway_mm;
};
constexpr MovingPlacement moving[] = {
    {"short_a",22,27,45}, {"short_b",-112,-32,60},
    {"medium_a",115,-23,75}, {"medium_b",-38,36,95},
    {"long_a",-164,17,125}, {"long_b",67,-41,150}
};
constexpr open_water_validation::InteriorPlacement interior[3][2] = {
    {{-.16f,.12f,-27},{.31f,.14f,121}},
    {{.13f,-.15f,64},{-.34f,-.12f,-82}},
    {{.11f,.18f,147},{.17f,-.32f,19}}
};

chooser::World placed(const Models& models, const Start& start) {
    const auto& source=models.starts.at(start.archived_row);
    chooser::World world{};
    chooser::initialize(world,source,horizontal_place(start,source),yaw_offset(start,source),models.motor.kind,models.config);
    return world;
}

float horizontal_runway(const chooser::World& world) {
    const auto velocity=horizontal(world.pose.com_velocity_m_s);
    const float speed=physics::norm(velocity);
    if(speed<.006f) throw std::runtime_error("Moving fixture lacks the required initial speed");
    return tail::bank::transport_range(world.motor.navigation.motion.body,world.pose,larger_pond,velocity/speed).distance_m;
}

Start place_moving(const Models& models,int phase,float ray_x,float ray_z,float heading,const MovingPlacement& spec) {
    Start start{spec.name,size_t(phase),0,0,heading};
    float inner=0.f,outer=.98f;
    for(int i=0;i<22;++i) {
        const float radius=.5f*(inner+outer);
        start.x_radius=radius*ray_x;
        start.z_radius=radius*ray_z;
        if(horizontal_runway(placed(models,start))>=spec.runway_mm*.001f) inner=radius;
        else outer=radius;
    }
    start.x_radius=inner*ray_x;
    start.z_radius=inner*ray_z;
    return start;
}

int run(Models models,const fs::path& output,Clock::time_point begin) {
    if(models.starts.size()!=12 || !fs::create_directory(output))
        throw std::runtime_error("Expected 12 reserved poses and a fresh selection directory");
    const int depth=int(std::lround(-1000.*tail::snapshot(models.starts[8].motion.body).com_m.y));
    if(depth!=72 && depth!=75 && depth!=80) throw std::runtime_error("Invalid moving validation depth");
    const int index=depth==72?0:(depth==75?1:2);
    const float target_y=-.001f*depth;
    auto& rest=models.starts[0];
    const float dy=target_y-rest.legs[rest.active_leg].goal_m.y;
    rest.motion.body.position.y+=dy/physics::scale;
    for(auto& point:rest.motion.history) point.y+=dy;
    for(int i=0;i<rest.leg_count;++i) rest.legs[i].goal_m.y+=dy;
    tail::prepare_navigation(rest);

    constexpr float rotations[]={0,7,19};
    std::array<Start,8> close{};
    for(int i=0;i<8;++i) {
        const bool approach=i<6;
        const float azimuth=approach?moving[i].azimuth:(i==6?51.f:-147.f);
        const float angle=radians(azimuth+rotations[index]);
        const float x=std::cos(angle),z=std::sin(angle);
        if(approach) {
            const double outward=std::atan2(-z*larger_pond.water_radius_z_m,x*larger_pond.water_radius_x_m);
            close[i]=place_moving(models,8+i%2,x,z,float(outward*180./tail::pi)+moving[i].heading_offset,moving[i]);
        } else {
            const double tangent=std::atan2(x/larger_pond.water_radius_x_m,z/larger_pond.water_radius_z_m);
            close[i]=place_shore_fixture(models,i==6?9:8,x,z,float(tangent*180./tail::pi)+(i==6?-6:174),
                i==6?.015:.019,i==6?"parallel_a":"parallel_b");
        }
    }
    std::array<Start,2> long_swims{};
    for(int i=0;i<2;++i) {
        const auto& p=interior[index][i];
        long_swims[i]={i?"interior_moving":"interior_rest",size_t(i?9:0),p.x,p.z,p.heading};
    }
    fs::create_directory(output/"traces");
    std::ofstream starts(output/"starts.csv");
    starts.exceptions(std::ios::failbit|std::ios::badbit);
    starts<<std::setprecision(10)<<"case,depth_mm,phase,x_m,y_m,z_m,heading_deg,initial_gap_mm,actual_horizontal_speed_mm_s,measured_runway_mm\n";
    auto record=[&](const Start& s,float desired_runway,float desired_gap=0.f) {
        auto w=placed(models,s);
        const auto com=w.pose.com_m;
        const double gap=1000.*tail::submerged_gap_m(larger_pond,w.motor.navigation.motion.body,w.pose);
        const double speed=1000.*physics::norm(horizontal(w.pose.com_velocity_m_s));
        const double runway=s.archived_row?1000.*horizontal_runway(w):0.;
        if(std::abs(com.y-target_y)>.00001 || !(gap>0.) || (s.archived_row?speed<6.:speed>=.001) ||
            (desired_runway>0 && std::abs(desired_runway-runway)>1.) ||
            (desired_gap>0 && std::abs(desired_gap-gap)>.01))
            throw std::runtime_error(std::string("Invalid selection geometry at ")+s.name);
        starts<<s.name<<','<<depth<<','<<s.archived_row<<','<<com.x<<','<<com.y<<','<<com.z<<','<<s.heading_deg
              <<','<<gap<<','<<speed<<','<<runway<<'\n';
    };
    for(int i=0;i<6;++i) record(close[i],moving[i].runway_mm);
    record(close[6],0,15);
    record(close[7],0,19);
    for(const auto& s:long_swims) record(s,0);
    starts.close();

    const auto cells=reachable_cells(models.starts.front());
    const auto deadline=begin+std::chrono::seconds(55);
    const auto short_cases=run_parallel(close,180,3,models,cells,output,deadline);
    const auto long_cases=run_parallel(long_swims,900,14,models,cells,output,deadline);
    std::ofstream cases(output/"cases.csv");
    cases.exceptions(std::ios::failbit|std::ios::badbit);
    cases<<std::setprecision(10)<<"suite,depth_mm,case,fault,pass,accepted_substeps,min_clearance_mm,max_stall_controls,close_s,unique_cells\n";
    auto record_case=[&](const char* suite,const Case& row) {
        if((row.fault!=0 && row.fault!=chooser::Clearance) ||
            (!row.fault && row.accepted!=row.horizon_s*samples_per_second))
            throw std::runtime_error("Unexpected selection fault or incomplete trajectory");
        cases<<suite<<','<<depth<<','<<row.start.name<<','<<row.fault<<','<<row.pass<<','<<row.accepted<<','<<row.min_gap_m*1000<<','
             <<row.max_stall_controls<<','<<double(row.perimeter_controls)/controls_per_second<<','<<row.visited.size()<<'\n';
    };
    for(size_t i=0;i<short_cases.size();++i) record_case(i<6?"moving":"tangent",short_cases[i]);
    for(const auto& row:long_cases) record_case("long",row);
    return 0;
}
} // namespace moving_start_validation
