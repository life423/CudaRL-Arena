// Included by rate-evaluate.cpp to share production inference, physiology and native metric gates.
namespace open_water_validation {
struct InteriorPlacement { float x, z, heading; };
constexpr InteriorPlacement interior[3][2] = {
    {{-.086f,.063f,37.f},{.26f,-.23f,-71.f}},
    {{.094f,.072f,-51.f},{-.21f,.28f,114.f}},
    {{.073f,-.094f,78.f},{-.27f,-.19f,161.f}}
};

int run(Models models,const fs::path& output,Clock::time_point begin) {
    if(models.starts.size()!=12 || !fs::create_directory(output))
        throw std::runtime_error("Validation needs the reserved poses and an unused output directory");
    const int depth=static_cast<int>(std::lround(-1000.*tail::snapshot(models.starts[8].motion.body).com_m.y));
    if(depth!=72 && depth!=75 && depth!=80)
        throw std::runtime_error("Unexpected validation depth");
    const int index=depth==72?0:(depth==75?1:2);
    const auto cells=reachable_cells(models.starts.front());
    const float target_y=-.001f*depth;
    auto& rest=models.starts[0];
    const float dy=target_y-rest.legs[rest.active_leg].goal_m.y;
    rest.motion.body.position.y+=dy/physics::scale;
    for(auto& point:rest.motion.history) point.y+=dy;
    for(int i=0;i<rest.leg_count;++i) rest.legs[i].goal_m.y+=dy;
    tail::prepare_navigation(rest);
    constexpr const char* names[]={"moving_a","moving_b","moving_c","moving_d","tangent_a","tangent_b","explore_rest","explore_moving"};
    constexpr float rays[]={-63,-133,38,146,83,-102};
    constexpr float gaps_mm[]={38,49,44,36,16,18};
    constexpr float offsets[]={-15,21,-37,46,4,186};
    constexpr int phases[]={8,9,8,9,9,8};
    std::array<Start,6> close{};
    for(int i=0;i<6;++i) {
        const float angle=radians(rays[i]+11.f*index);
        const float x=std::cos(angle),z=std::sin(angle);
        const double heading=i<4?std::atan2(-z*larger_pond.water_radius_z_m,x*larger_pond.water_radius_x_m):
            std::atan2(x/larger_pond.water_radius_x_m,z/larger_pond.water_radius_z_m);
        close[i]=place_shore_fixture(models,phases[i],x,z,float(heading*180./tail::pi)+offsets[i],gaps_mm[i]*.001,names[i]);
    }
    std::array<Start,2> long_swims{};
    for(int i=0;i<2;++i) {
        const auto& p=interior[index][i];
        long_swims[i]={names[6+i],size_t(i?9:0),p.x,p.z,p.heading,false};
    }

    fs::create_directory(output/"traces");
    std::ofstream starts(output/"starts.csv");
    starts.exceptions(std::ios::failbit|std::ios::badbit);
    starts<<std::setprecision(10)<<"case,depth_mm,phase,x_m,y_m,z_m,heading_deg,initial_gap_mm,actual_horizontal_speed_mm_s\n";
    auto record_start=[&](const Start& s) {
        const auto& source=models.starts[s.archived_row];
        chooser::World placed{};
        chooser::initialize(placed,source,horizontal_place(s,source),yaw_offset(s,source),models.motor.kind,models.config);
        const auto& com=placed.pose.com_m;
        const double speed=1000.*physics::norm(horizontal(placed.pose.com_velocity_m_s));
        const double gap=1000.*tail::submerged_gap_m(larger_pond,placed.motor.navigation.motion.body,placed.pose);
        if(std::abs(com.y-target_y)>.00001f || !(gap>0.) ||
            (s.archived_row ? !(speed>=6.) : !(speed<.001)))
            throw std::runtime_error("Validation start depth, clearance or motion differs");
        starts<<s.name<<','<<depth<<','<<s.archived_row<<','<<com.x<<','<<com.y<<','<<com.z<<','<<s.heading_deg<<','<<gap<<','<<speed<<'\n';
        return gap;
    };
    for(int i=0;i<6;++i) {
        if(std::abs(record_start(close[i])-gaps_mm[i])>.01)
            throw std::runtime_error("Validation start misses its specified clearance");
    }
    for(const auto& s:long_swims) record_start(s);
    starts.close();
    const auto deadline=begin+std::chrono::seconds(42);
    const auto short_results=run_parallel(close,180,3,models,cells,output,deadline);
    const auto long_results=run_parallel(long_swims,900,14,models,cells,output,deadline);
    std::ofstream cases(output/"cases.csv");
    cases.exceptions(std::ios::failbit|std::ios::badbit);
    cases<<std::setprecision(10)<<"suite,depth_mm,case,fault,pass,accepted_substeps,min_clearance_mm,max_stall_controls,close_s,unique_cells\n";
    auto write=[&](const char* suite,const Case& row) {
        if((row.fault!=0 && row.fault!=chooser::Clearance) ||
            (!row.fault && row.accepted!=row.horizon_s*samples_per_second))
            throw std::runtime_error("Unexpected validation fault or incomplete trajectory");
        cases<<suite<<','<<depth<<','<<row.start.name<<','<<row.fault<<','<<row.pass<<','<<row.accepted<<','<<row.min_gap_m*1000<<','
             <<row.max_stall_controls<<','<<double(row.perimeter_controls)/controls_per_second<<','<<row.visited.size()<<'\n';
    };
    for(size_t i=0;i<short_results.size();++i) write(i<4?"moving":"tangent",short_results[i]);
    for(const auto& row:long_results) write("long",row);
    return 0;
}
} // namespace open_water_validation
