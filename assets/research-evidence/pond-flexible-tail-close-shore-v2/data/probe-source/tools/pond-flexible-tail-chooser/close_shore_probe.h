#pragma once

// Offline counterfactual probes only. Never linked into the native Godot bridge.
namespace close_probe {
inline uint64_t attempted=0,accepted=0;
inline Clock::time_point deadline;
void bound(){if(attempted>6000000||Clock::now()>=deadline)throw std::runtime_error("Close-shore diagnostic cap reached");}
chooser::Result step(chooser::World& w,const Models& m){
    bound();auto r=chooser::advance(w,m.gait.data(),m.packed_motor.data(),larger_pond,4);
    attempted+=r.attempted_substeps;accepted+=r.accepted_substeps;bound();
    if(r.fault&&r.fault!=chooser::Clearance)throw std::runtime_error("Close-shore diagnostic native fault");return r;
}
uint64_t hash(const chooser::World& w){cudarl::checkpoint::Writer b;b.value(w);return cudarl::checkpoint::checksum(b.data);}
int decide(const chooser::World& w,cudarl::nn::CpuMlp<float>& actor,int mode){
    float input[chooser::inputs];chooser::observe(w,larger_pond,input,mode);actor.forward(input,1);
    const auto logits=actor.output();if(!std::all_of(logits,logits+chooser::actions,physics::finite_number))throw std::runtime_error("Invalid probe logits");
    return int(std::max_element(logits,logits+chooser::actions)-logits);
}
std::vector<std::string> fields(const std::string& row){std::stringstream s(row);std::string value;std::vector<std::string> out;while(std::getline(s,value,','))out.push_back(value);return out;}
struct Expected {V com{};int action=-1;};
struct Capture {chooser::World world{};int phase=0,control=0;uint64_t checksum=0;};
int run(const Models& models,const fs::path& output,const fs::path& reference,Clock::time_point begin){
    if(!fs::create_directory(output))throw std::runtime_error("Probe output must be unused");
    fs::create_directory(output/"captures");fs::create_directory(output/"traces");deadline=begin+std::chrono::seconds(90);
    cudarl::nn::CpuMlp<float> actor(chooser::actor_spec());actor.params()=models.policy;
    std::vector<Capture> captures;int compared=0,compared_actions=0;float max_error=0;
    for(int phase:{8,9}){
        const int direction=phase==8?-1:1;const std::string name="phase"+std::to_string(phase)+"-bank1-direction"+std::to_string(direction);
        std::ifstream file(reference/"traces"/(name+".csv"));if(!file)throw std::runtime_error("Missing archived failed trace");
        std::string line;std::getline(file,line);std::map<int,Expected> expected;
        while(std::getline(file,line)){auto f=fields(line);if(f.size()<16)throw std::runtime_error("Malformed archived row");const int control=std::stoi(f[0]);
            auto& row=expected[control];row.com={std::stof(f[2]),std::stof(f[3]),std::stof(f[4])};if(f[12]=="decision")row.action=std::stoi(f[13]);}
        const auto fixture=close_shore_fixture(models,phase,1,direction,name.c_str());const auto& saved=models.starts.at(phase);
        chooser::World w{};chooser::initialize(w,saved,horizontal_place(fixture,saved),yaw_offset(fixture,saved),models.motor.kind,models.config);
        int control=0,work=0;
        for(;;){
            if(auto it=expected.find(control);it!=expected.end()){
                const float error=physics::norm(w.pose.com_m-it->second.com);max_error=std::max(max_error,error);++compared;
                if(error>1.e-6f)throw std::runtime_error("Failed-prefix COM mismatch");
            }
            if(w.fault)break;
            if(chooser::ready(w)){
                if(control==40||control==80||control==120){Capture c{w,phase,control,hash(w)};captures.push_back(c);
                    cudarl::checkpoint::Writer b;b.value<uint32_t>(1);b.value(w);cudarl::checkpoint::write_atomic((output/"captures"/(name+"-"+std::to_string(control)+".bin")).u8string(),b.data);}
                const int action=decide(w,actor,models.distant_feedback);
                if(!expected.count(control)||expected.at(control).action!=action)throw std::runtime_error("Failed-prefix action mismatch");++compared_actions;
                if(!chooser::accept(w,action))throw std::runtime_error("Invalid prefix request");
            }
            const auto r=step(w,models);control+=r.controls;work+=r.accepted_substeps;
            if(control>250)throw std::runtime_error("Expected original close failure missing");
        }
        if(w.fault!=chooser::Clearance||work!=(phase==8?3467:3111))throw std::runtime_error("Failed-prefix outcome mismatch");
    }
    if(captures.size()!=6)throw std::runtime_error("Missing exact decision captures");
    std::ofstream table(output/"cases.csv");table<<std::setprecision(10)<<"phase,capture_control,action,hold_controls,pass,fault,seconds,mean_speed_mm_s,min_gap_mm,final_gap_mm,max_stall_s,tilt_deg,depth_mm,state_hash,attempted_substeps,accepted_substeps\n";
    int rows=0,passes=0,first8=0,first9=0;
    for(const auto& c:captures)for(int branch=-1;branch<30;++branch){
        chooser::World w;std::memcpy(&w,&c.world,sizeof(w));if(hash(w)!=c.checksum)throw std::runtime_error("Fork state mismatch");
        const int action=branch<0?-1:branch/2,hold=branch<0?0:(branch%2?400:120),straight=action<0?0:(action/chooser::rate_count)*chooser::rate_count+chooser::rate_count/2;
        int control=0,branch_work=0,branch_accepted=0;float stall=0,max_stall=0;double path=0;const float depth=w.motor.navigation.motion.request.origin_m.y;
        std::ofstream trace(output/"traces"/("p"+std::to_string(c.phase)+"-c"+std::to_string(c.control)+"-b"+std::to_string(branch)+".csv"));
        trace<<std::setprecision(10)<<"control,x_m,y_m,z_m,action,speed_mm_s,gap_mm,fault\n";
        while(!w.fault&&control<2000){
            int chosen=-1;if(chooser::ready(w)){chosen=action<0||control>=hold+400?decide(w,actor,models.distant_feedback):(control<hold?action:straight);
                if(!chooser::accept(w,chosen))throw std::runtime_error("Invalid fork request");}
            const auto before=w.pose.com_m;const auto r=step(w,models);control+=r.controls;branch_work+=r.attempted_substeps;branch_accepted+=r.accepted_substeps;
            path+=physics::norm(horizontal(w.pose.com_m-before));const float speed=physics::norm(tail::fast::feedback_sample(w.motor).velocity_m_s);
            stall=speed<.006f?stall+r.controls*.02f:0;max_stall=std::max(max_stall,stall);
            trace<<control<<','<<w.pose.com_m.x<<','<<w.pose.com_m.y<<','<<w.pose.com_m.z<<','<<chosen<<','<<speed*1000<<','<<tail::submerged_gap_m(larger_pond,w.motor.navigation.motion.body,w.pose)*1000<<','<<w.fault<<'\n';
        }
        const float gap=float(tail::submerged_gap_m(larger_pond,w.motor.navigation.motion.body,w.pose));
        const double seconds=branch_accepted/800.,mean=seconds>0?path/seconds:0;
        const bool pass=!w.fault&&control==2000&&w.min_gap_m>0&&w.peak_tilt_deg<=10&&w.peak_depth_error_m<=.001f&&mean>=.006&&max_stall<=8.14f&&gap>=.015f;
        ++rows;passes+=pass;if(c.control==40&&pass){if(c.phase==8)++first8;else ++first9;}
        table<<c.phase<<','<<c.control<<','<<action<<','<<hold<<','<<pass<<','<<w.fault<<','<<seconds<<','<<mean*1000<<','<<w.min_gap_m*1000<<','<<gap*1000<<','<<max_stall<<','<<w.peak_tilt_deg<<','<<w.peak_depth_error_m*1000<<','<<c.checksum<<','<<branch_work<<','<<branch_accepted<<'\n';
    }
    if(rows!=186)throw std::runtime_error("Incomplete close-shore probe");
    std::ofstream summary(output/"summary.json");summary<<std::setprecision(10)<<"{\"complete\":true,\"replay_samples\":"<<compared<<",\"replay_actions\":"<<compared_actions<<",\"max_COM_error_m\":"<<max_error<<",\"captures\":6,\"branches\":"<<rows<<",\"passes\":"<<passes<<",\"first_phase8_witnesses\":"<<first8<<",\"first_phase9_witnesses\":"<<first9<<",\"attempted_substeps\":"<<attempted<<",\"accepted_substeps\":"<<accepted<<",\"optimizer_updates\":0,\"wall_s\":"<<std::chrono::duration<double>(Clock::now()-begin).count()<<"}\n";
    std::cout<<"Close-shore diagnostic: "<<passes<<"/"<<rows<<", first-state witnesses "<<first8<<" / "<<first9<<'\n';return 0;
}
}
