// Reuse the exact qualified evaluator's loaders, fixture transform and native
// runtime. Its entry point is retained under another name; no copies of physics.
#define main archived_rate_evaluator_main
#include "../pond-flexible-tail-chooser/rate-evaluate.cpp"
#undef main

namespace warning_audit {
constexpr uint64_t work_cap=6912000;
uint64_t attempted=0,accepted=0;
Clock::time_point deadline;
void bounded() {
    if(attempted>work_cap||Clock::now()>=deadline)throw std::runtime_error("Warning audit work/time cap reached");
}
void step(chooser::World& w,const Models& m) {
    bounded();const auto r=chooser::advance(w,m.gait.data(),m.packed_motor.data(),larger_pond,4);
    attempted+=r.attempted_substeps;accepted+=r.accepted_substeps;bounded();
    if(r.fault&&r.fault!=chooser::Clearance)throw std::runtime_error("Warning audit native/nonfinite fault");
}
V measured_velocity(const chooser::World& w) {
    return tail::turn_course(w.motor.navigation.motion,w.pose.com_m,50);
}
struct Reading {float speed=0,range=0,time=0,gap=0;bool valid=false;};
Reading read(const chooser::World& w) {
    const auto sample=tail::fast::feedback_sample(w.motor);float senses[tail::bank::inputs];
    tail::bank::observe(w.motor.navigation.motion.body,w.pose,larger_pond,sample.velocity_m_s,sample.reliable,senses);
    return {senses[tail::bank::HorizontalSpeed]*tail::bank::speed_scale_m_s,
        senses[tail::bank::AheadTransportRange]*tail::bank::maximum_range_m,
        senses[tail::bank::ConstantTransportTime]*tail::bank::time_scale_s,
        float(tail::submerged_gap_m(larger_pond,w.motor.navigation.motion.body,w.pose)),
        senses[tail::bank::TransportTimeValid]>0};
}
int action(int speed_index,int rate) {return speed_index*5+(rate+8)/4;}
V outward(V p) {
    V normal{p.x/float(larger_pond.water_radius_x_m*larger_pond.water_radius_x_m),0,
        p.z/float(larger_pond.water_radius_z_m*larger_pond.water_radius_z_m)};
    const float n=physics::norm(normal);return n>1.e-6f?normal/n:V{1,0,0};
}
struct Capture {bool first=false,found=false;int first_control=0;Reading first_read{};chooser::World state{};};
struct Row {
    int start=0,speed_index=0,heading=0,horizon=0,rate=0,fault=0;
    float cue_s=0,warning_delay=0,cue_speed=0,cue_range=0,cue_ttc=0;
    float elapsed=0,response=-1,turn90=-1,inward=-1,radius=-1,rate_mean=0;
    float max_forward=0,min_gap=FLT_MAX,peak_tilt=0,depth_error=0,signed_angle=0,mean_speed=0;
    uint64_t attempted=0,accepted=0;bool safe=false;
};
void log_state(std::ostream& stream,const char* stage,float t,const chooser::World& w) {
    const auto velocity=measured_velocity(w);const auto sensing=read(w);
    stream<<stage<<','<<t<<','<<w.motor.navigation.motion.controls<<','<<w.pose.com_m.x<<','<<w.pose.com_m.y<<','
        <<w.pose.com_m.z<<','<<velocity.x<<','<<velocity.z<<','<<sensing.speed<<','<<sensing.range<<','<<sensing.time<<','
        <<sensing.valid<<','<<sensing.gap<<','<<w.config.target_speed_m_s<<','
        <<w.config.desired_course_rate_rad_s*180/tail::pi<<','<<w.fault<<'\n';
}
Row branch(int start,int speed_index,int heading,int horizon,int rate,const Capture& capture,const Models& m,const fs::path& root) {
    Row r;r.start=start;r.speed_index=speed_index;r.heading=heading;r.horizon=horizon;r.rate=rate;
    chooser::World w{};std::memcpy(&w,&capture.state,sizeof(w));
    if(std::memcmp(&w,&capture.state,sizeof(w)))throw std::runtime_error("Warning fork differs");
    const auto sensing=read(w);r.cue_s=w.motor.navigation.motion.controls*.02f;
    r.warning_delay=(w.motor.navigation.motion.controls-capture.first_control)*.02f;
    r.cue_speed=1000*sensing.speed;r.cue_range=1000*sensing.range;r.cue_ttc=sensing.time;
    auto v=measured_velocity(w);const float initial_speed=physics::norm(v);
    if(initial_speed<.003f)throw std::runtime_error("Warning cue lacks useful actual course");
    const auto entry_course=v/initial_speed,entry=w.pose.com_m;
    const auto before_attempted=attempted,before_accepted=accepted;
    const int cue_control=w.motor.navigation.motion.controls;
    std::ostringstream name;name<<"start"<<start<<"-speed"<<speed_index<<"-heading"<<heading<<"-h"<<horizon<<"-rate"<<rate;
    std::ofstream trace(root/"traces"/(name.str()+".csv"));trace<<std::setprecision(10)
        <<"stage,elapsed_s,control,x_m,y_m,z_m,COM_vx_m_s,COM_vz_m_s,sensor_speed_m_s,forward_range_m,transport_time_s,transport_valid,certified_gap_m,requested_speed_m_s,requested_rate_deg_s,fault\n";
    log_state(trace,"cue",0,w);
    float previous_angle=0,unwrapped=0,path=0,path_at_response=0,response_angle=0,speed_sum=0;
    int sample_count=0,inward_controls=0,response_controls=0;
    while(w.fault==0&&w.motor.navigation.motion.controls-cue_control<2000) {
        if(chooser::ready(w)&&!chooser::accept(w,action(speed_index,rate)))throw std::runtime_error("Invalid diagnostic turn request");
        const auto before=w.pose.com_m;step(w,m);
        r.elapsed=float(accepted-before_accepted)/800.f;
        path+=physics::norm(horizontal(w.pose.com_m-before));
        const auto velocity=measured_velocity(w);const float speed=physics::norm(velocity);
        speed_sum+=speed;++sample_count;
        const float angle=std::atan2(physics::cross(entry_course,velocity).y,physics::dot(entry_course,velocity));
        unwrapped+=std::remainder(angle-previous_angle,2*tail::pi);previous_angle=angle;
        r.signed_angle=unwrapped*180/tail::pi;
        const float progress=(rate<0?-1.f:1.f)*r.signed_angle;
        response_controls=(rate!=0&&progress>=5&&speed>=.006f)?response_controls+4:0;
        if(r.response<0&&response_controls>=28){r.response=r.elapsed;path_at_response=path;response_angle=progress;}
        if(r.turn90<0&&rate!=0&&progress>=90){r.turn90=r.elapsed;
            if(r.response>=0&&progress>response_angle&&r.turn90>r.response){
                r.radius=1000*(path-path_at_response)/((progress-response_angle)*tail::pi/180.f);
                r.rate_mean=(progress-response_angle)/(r.turn90-r.response);}}
        inward_controls=(physics::dot(velocity,outward(w.pose.com_m))<-.1f*speed&&speed>=.006f)?inward_controls+4:0;
        if(r.inward<0&&inward_controls>=52)r.inward=r.elapsed;
        r.max_forward=fmaxf(r.max_forward,1000*physics::dot(w.pose.com_m-entry,entry_course));
        r.min_gap=fminf(r.min_gap,w.min_gap_m*1000);r.peak_tilt=fmaxf(r.peak_tilt,w.peak_tilt_deg);
        r.depth_error=fmaxf(r.depth_error,w.peak_depth_error_m*1000);
        log_state(trace,w.fault?"stop":"motion",r.elapsed,w);
    }
    r.attempted=attempted-before_attempted;r.accepted=accepted-before_accepted;r.fault=w.fault;
    r.mean_speed=sample_count?1000*speed_sum/sample_count:0;
    r.safe=r.fault==0&&r.elapsed>=39.99f&&r.min_gap>0&&r.peak_tilt<=10&&r.depth_error<=1;
    return r;
}
void write_optional(std::ostream& out,float x){if(x>=0)out<<x;}
int execute(int argc,char** argv) {
    if(argc!=5)throw std::runtime_error("Usage: pond_shoreline_warning GAIT MOTOR STRAIGHT_STARTS NEW_OUT");
    const auto begin=Clock::now();deadline=begin+std::chrono::seconds(50);
    const auto root=fs::u8path(argv[4]);if(!fs::create_directory(root))throw std::runtime_error("Audit output must be unused");
    fs::create_directory(root/"traces");fs::create_directory(root/"captures");
    Models m;m.gait=tail::read_navigation_weights(argv[1],tail::NavigationMode::Gait);m.motor=tail::steering::read_model(argv[2]);
    m.starts=tail::read_navigation_worlds(argv[3]);if(m.starts.size()!=12)throw std::runtime_error("Expected12 native starts");
    m.config.study=tail::steering::Study::Rate;m.config.material=m.motor.material;
    m.packed_motor=m.motor.weights;m.packed_motor.insert(m.packed_motor.end(),m.motor.frozen_skip.begin(),m.motor.frozen_skip.end());
    std::vector<Row> rows;
    std::ofstream cues(root/"cues.csv");cues<<std::setprecision(10)
        <<"start,speed_index,heading_deg,horizon_s,captured,first_warning_s,cue_s,decision_delay_s,first_speed_mm_s,first_range_mm,cue_speed_mm_s,cue_range_mm,cue_ttc_s\n";
    std::ofstream prefixes(root/"prefixes.csv");prefixes<<"start,speed_index,heading_deg,controls,fault,attempted,accepted\n";
    constexpr int horizons[3]{12,20,24};constexpr float speeds[3]{.006f,.008f,.012f};
    int captures=0;
    for(int start:{8,9})for(int speed_index=0;speed_index<3;++speed_index)for(int heading:{0,45}) {
        Start fixture{"audit",size_t(start),0,0,float(heading),false};
        chooser::World w{};m.config.target_speed_m_s=speeds[speed_index];
        chooser::initialize(w,m.starts[start],horizontal_place(fixture,m.starts[start]),yaw_offset(fixture,m.starts[start]),m.motor.kind,m.config);
        Capture capture[3];const auto prefix_attempted=attempted,prefix_accepted=accepted;
        while(w.fault==0&&w.motor.navigation.motion.controls<6000) {
            const auto sensed=read(w);const int control=w.motor.navigation.motion.controls;
            if(control>=300&&sensed.valid)for(int h=0;h<3;++h)if(!capture[h].first&&sensed.time<float(horizons[h])) {
                capture[h].first=true;capture[h].first_control=control;capture[h].first_read=sensed;
            }
            if(chooser::ready(w)) {
                for(int h=0;h<3;++h)if(capture[h].first&&!capture[h].found){capture[h].found=true;capture[h].state=w;++captures;}
                if(capture[0].found&&capture[1].found&&capture[2].found)break;
                if(!chooser::accept(w,action(speed_index,0)))throw std::runtime_error("Prefix request rejected");
            }
            step(w,m);
        }
        prefixes<<start<<','<<speed_index<<','<<heading<<','<<w.motor.navigation.motion.controls<<','<<w.fault<<','<<attempted-prefix_attempted<<','<<accepted-prefix_accepted<<'\n';
        for(int h=0;h<3;++h){const auto& c=capture[h];const auto at=c.found?read(c.state):Reading{};
            cues<<start<<','<<speed_index<<','<<heading<<','<<horizons[h]<<','<<c.found<<','<<c.first_control*.02f<<','
                <<c.state.motor.navigation.motion.controls*.02f<<','<<(c.state.motor.navigation.motion.controls-c.first_control)*.02f<<','
                <<c.first_read.speed*1000<<','<<c.first_read.range*1000<<','<<at.speed*1000<<','<<at.range*1000<<','<<at.time<<'\n';
            if(!c.found)continue;
            std::ostringstream key;key<<"start"<<start<<"-speed"<<speed_index<<"-heading"<<heading<<"-h"<<horizons[h];
            cudarl::checkpoint::Writer saved;saved.value<uint32_t>(1);saved.value(c.state);
            cudarl::checkpoint::write_atomic((root/"captures"/(key.str()+".bin")).u8string(),saved.data);
            for(int rate:{0,-4,4,-8,8})rows.push_back(branch(start,speed_index,heading,horizons[h],rate,c,m,root));
        }
    }
    std::ofstream table(root/"branches.csv");table<<std::setprecision(10)
        <<"start,speed_index,heading_deg,warning_horizon_s,rate_deg_s,cue_s,decision_delay_s,cue_speed_mm_s,cue_range_mm,cue_ttc_s,elapsed_s,response_5deg_s,turn_90deg_s,inward_s,effective_radius_mm,measured_turn_rate_deg_s,max_forward_excursion_mm,min_clearance_mm,peak_tilt_deg,depth_error_mm,achieved_deg,mean_COM_speed_mm_s,fault,safe_40s,attempted_substeps,accepted_substeps\n";
    for(const auto& r:rows){table<<r.start<<','<<r.speed_index<<','<<r.heading<<','<<r.horizon<<','<<r.rate<<','<<r.cue_s<<','<<r.warning_delay<<','
        <<r.cue_speed<<','<<r.cue_range<<','<<r.cue_ttc<<','<<r.elapsed<<',';write_optional(table,r.response);table<<',';write_optional(table,r.turn90);
        table<<',';write_optional(table,r.inward);table<<',';write_optional(table,r.radius);table<<','<<r.rate_mean<<','<<r.max_forward<<','<<r.min_gap<<','
        <<r.peak_tilt<<','<<r.depth_error<<','<<r.signed_angle<<','<<r.mean_speed<<','<<r.fault<<','<<r.safe<<','<<r.attempted<<','<<r.accepted<<'\n';}
    std::ofstream summary(root/"summary.json");summary<<std::setprecision(10)
        <<"{\"complete\":true,\"approach_prefixes\":12,\"captures\":"<<captures<<",\"branches\":"<<rows.size()
        <<",\"attempted_substeps\":"<<attempted<<",\"accepted_substeps\":"<<accepted<<",\"substep_cap\":"<<work_cap
        <<",\"optimizer_updates\":0,\"motor_frozen\":true,\"resets_within_branch\":0,\"wall_s\":"<<std::chrono::duration<double>(Clock::now()-begin).count()<<"}\n";
    std::cout<<"Complete "<<captures<<" captures, "<<rows.size()<<" branches, "<<attempted<<" substeps\n";return 0;
}
}
int main(int argc,char** argv){try{return warning_audit::execute(argc,argv);}catch(const std::exception& e){
    std::cerr<<e.what()<<"; attempted="<<warning_audit::attempted<<", accepted="<<warning_audit::accepted<<'\n';return 2;}}
