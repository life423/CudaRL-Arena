// Original retained-skill fixtures and endpoint scorers, evaluated on both
// backends through the corrected mechanics TU. No optimization or new gates.
constexpr const char* retention_names[7]={"short","long60","recovery","long300","original-holds","withheld-holds","endurance-holds"};
Case retention_case_for(int id,int backend){
    Case c;c.id=id;c.fixture=id;c.arm=backend;c.warm=0;c.retention_kind=1;c.maximum=160;
    if(id<4){c.pose=id;c.retention_category=0;}
    else if(id<6){c.pose=id==4?0:3;c.maximum=1200;c.retention_category=1;}
    else if(id<9){c.pose=0;c.maximum=1280;c.offset=id==6?0:id==7?-.0005f:.0005f;c.retention_category=2;}
    else if(id<11){c.pose=id==9?0:3;c.maximum=6000;c.retention_category=3;}
    else if(id<17){const int j=id-11;c.pose=j%2?3:0;c.maximum=1280;c.offset=j/2==0?0:j/2==1?-.0005f:.0005f;c.retention_category=4;c.retention_kind=2;}
    else if(id<23){const int j=id-17;c.pose=j%2?2:1;c.maximum=1280;c.offset=j/2==0?0:j/2==1?-.00025f:.00025f;c.retention_category=5;c.retention_kind=2;}
    else {c.pose=id==23?0:3;c.maximum=6000;c.retention_category=6;c.retention_kind=2;}
    c.gain=c.retention_kind==2?saved_hold_gain:1.f;return c;
}
bool initialize_retention(Case context,Frame& f){
    f={};f.context=context;f.world=old::initial_world(context.pose);
    f.world.arena=context.retention_category==3?cudarl::contact::Arena{3.f,.4f}:old::long_swim_arena();
    const auto c=frame_config(context);
    if(!old::begin_forward(f.world,c,f.task)||!old::offset_episode_start(f.world,f.task,c,context.offset))return false;
    if(context.retention_kind==2&&!old::update_velocity_request(f.world,c,f.task))return false;
    f.initial_heading=f.task.request.heading;f.valid=1;return true;
}
void write_retention_cases(const fs::path& root){
    std::ofstream out(root/"retention-cases.json");out<<std::setprecision(17)<<"{\"schema\":1,\"study\":\"frozen_interface_comparison_v2\",\"study_version\":2,\"arithmetic_profile\":\""<<arithmetic_profile
        <<"\",\"holding_gain\":"<<saved_hold_gain<<",\"cases\":[";
    for(int id=0;id<retention_cases;++id){const auto c=retention_case_for(id,0);if(id)out<<',';
        out<<"{\"case\":"<<id<<",\"category\":\""<<retention_names[c.retention_category]<<"\",\"pose\":"<<c.pose<<",\"controls\":"<<c.maximum
            <<",\"offset_m\":"<<c.offset<<",\"holding\":"<<(c.retention_kind==2?"true":"false")<<",\"arena_half_width_m\":"<<(c.retention_category==3?3.f:.6f)<<",\"arena_half_depth_m\":"<<.4f<<'}';}
    out<<"]}\n";out.close();require(bool(out),"Retention catalog write failed");
}
// Exact variable-duration counterpart of the frozen retention_HoldScore.
class RetentionHoldScore {
    int total_,next_=0;bool valid_=true;float peak_[5]{},forward_window_=0,excursion_=0;
public:
    explicit RetentionHoldScore(int total):total_(total){valid_=total>=160;}
    bool record(int control,const old::Metrics& z){if(!valid_||control!=next_||control>total_){valid_=false;return false;}
        const float v[5]={p::norm(z.velocity_m_s),p::norm(z.hold_error_m),z.depth_m,z.lateral_m,z.tilt_deg};
        for(int j=0;j<5;++j){if(!std::isfinite(v[j])||v[j]<0){valid_=false;return false;}peak_[j]=fmaxf(peak_[j],v[j]);}
        if(!std::isfinite(z.forward_m)){valid_=false;return false;}
        if(control==total_-160)forward_window_=z.forward_m;if(control>=total_-160)excursion_=fmaxf(excursion_,fabsf(z.forward_m-forward_window_));++next_;return true;}
    bool passed()const{return valid_&&next_==total_+1&&peak_[0]<=.0005f&&peak_[1]<=.002f&&peak_[2]<=.001f&&peak_[3]<=.002f&&peak_[4]<=10.f&&excursion_<=.002f;}
    const float* peaks()const{return peak_;}float excursion()const{return excursion_;}
};
struct RetentionSummary {
    int short_passes=0,original_holds=0,long60_windows=0,long60_retained=0,recovery_passes=0,long300_windows=0,long300_retained=0,withheld_holds=0,endurance_holds=0;
    uint64_t controls=0,substeps=0;std::set<std::pair<int,int>> keys60,keys300;
    bool passed()const{return short_passes==4&&original_holds==6&&long60_retained==28&&recovery_passes==3&&long300_retained==148&&withheld_holds==6&&endurance_holds==2;}
};
void retention_json(std::ostream& out,const RetentionSummary& s){
    out<<"{\"full\":true,\"controls\":"<<s.controls<<",\"substeps\":"<<s.substeps
        <<",\"short_passes\":"<<s.short_passes<<",\"original_holds\":"<<s.original_holds<<",\"long60_windows\":"<<s.long60_windows<<",\"long60_retained\":"<<s.long60_retained
        <<",\"recovery_passes\":"<<s.recovery_passes<<",\"long300_windows\":"<<s.long300_windows<<",\"long300_retained\":"<<s.long300_retained
        <<",\"withheld_holds\":"<<s.withheld_holds<<",\"endurance_holds\":"<<s.endurance_holds;
    for(const auto item:{std::pair<const char*,const std::set<std::pair<int,int>>*>{"long60_passing_keys",&s.keys60},{"long300_passing_keys",&s.keys300}}){out<<",\""<<item.first<<"\":[";bool first=true;
        for(const auto& key:*item.second){if(!first)out<<',';first=false;out<<'['<<key.first<<','<<key.second<<']';}out<<']';}out<<'}';
}
bool run_retention(const fs::path& root,std::ofstream& states,std::ofstream& substeps,cp::Writer& raw,Budget& budget,
    nn::CpuMlp<float>& cpu,nn::GpuMlp& gpu,device_ptr<Frame>& d_state,device_ptr<float>& d_input,
    Frame& active,Frame& other,int& active_group,int& active_id,int& active_tick){
    (void)other;const auto folder=root/"retention-started";require(fs::create_directory(folder),"Retention already attempted");
    std::ofstream summary(folder/"summary.csv"),windows(folder/"windows.csv");summary<<std::setprecision(17);windows<<std::setprecision(17);
    summary<<"backend,case,category,pose,offset_m,controls,movement_passed,recovery_passed,hold_passed,steady_m_s,RMSE_m_s,lateral_m,depth_m,tilt_deg,path_deg,work_J_per_m,max_hold_speed_m_s,max_hold_anchor_m,max_hold_depth_m,max_hold_lane_m,max_hold_tilt_deg,final8_excursion_m\n";
    windows<<"backend,case,pose,end_s,steady_m_s,RMSE_m_s,lateral_m,depth_m,tilt_deg,path_deg,work_J_per_m,movement_passed\n";
    RetentionSummary results[2];
    for(int backend=0;backend<2;++backend){auto& result=results[backend];const auto start_controls=budget.controls,start_steps=budget.substeps;
        for(int id=0;id<retention_cases;++id){active_group=2;active_id=id;active_tick=0;const auto context=retention_case_for(id,backend);
            require(initialize_retention(context,active),"Retention fixture initialization failed");const auto c=frame_config(context);
            capture(states,substeps,raw,budget,2,id,backend,0,active);if(backend)d_state.copy_from_host(&active,1);
            RetentionHoldScore holding(context.maximum);if(context.retention_kind==2)require(holding.record(0,old::measure(active.task,active.world)),"Retention hold initial score rejected");
            auto origin=cudarl::motor::motion(active.world.physical.body).com_model;float work0=active.world.physical.body.work;double error2=0;bool recovered=true;
            for(int tick=1;tick<=context.maximum;++tick){active_tick=tick;budget.before();
                if(backend){infer(gpu,d_state,d_input,1);d_state.copy_to_host(&active,1);require(active.valid,"GPU retention observer rejected");}
                else {require(observe(active),"CPU retention observer rejected");cpu.forward(active.observation,1);
                    for(int j=0;j<4;++j){active.base[j]=cpu.output()[j];active.command[j]=tanhf(context.gain*active.base[j]);}}
                ++budget.controls;if(backend){++explicit_cuda_kernel_launches;stepping<<<1,1>>>(d_state.get());sync();d_state.copy_to_host(&active,1);}else advance(active);
                budget.substeps+=active.attempted;budget.accepted+=active.accepted;capture(states,substeps,raw,budget,2,id,backend,tick,active);
                require(active.advanced&&!active.reason,"Retained-skill physical step failed");const auto z=old::measure(active.task,active.world);
                if(context.retention_kind==2)require(holding.record(tick,z),"Retained hold score rejected");
                if(tick>40)error2+=z.speed_error_m_s*z.speed_error_m_s;
                if(tick==40){origin=cudarl::motor::motion(active.world.physical.body).com_model;work0=active.world.physical.body.work;}
                if(context.retention_category==2&&context.offset!=0&&tick>context.maximum-160)recovered=recovered&&z.depth_m<fabsf(context.offset)*.5f;
                if((context.retention_category==1||context.retention_category==3)&&tick%80==0){const auto center=cudarl::motor::motion(active.world.physical.body).com_model;
                    const auto delta=(center-origin)*p::scale;const float forward=p::dot(delta,active.task.request.heading);const auto side=delta-active.task.request.heading*forward;
                    const int samples=tick==80?40:80;const float speed=forward/(samples*p::dt),rmse=sqrtf(float(error2/samples)),
                        path=atan2f(p::norm({side.x,0,side.z}),forward)*180/model::pi,work=(active.world.physical.body.work-work0)/fmaxf(forward,1e-9f);
                    const bool pass=speed>=c.minimum_speed_m_s&&speed<=c.maximum_speed_m_s&&rmse<=c.speed_rmse_limit&&
                        active.task.maximum_lateral<=c.lateral_limit_m&&active.task.maximum_depth<=c.depth_limit_m&&active.task.maximum_tilt<=c.tilt_limit_deg&&path<=c.path_angle_limit_deg&&work<=c.work_per_distance_limit;
                    if(context.retention_category==1){result.long60_windows+=pass;if(pass)result.keys60.emplace(context.pose,tick/20);}
                    else {result.long300_windows+=pass;if(pass)result.keys300.emplace(context.pose,tick/20);}
                    windows<<backend<<','<<id<<','<<context.pose<<','<<tick*.05<<','<<speed<<','<<rmse<<','<<active.task.maximum_lateral<<','<<active.task.maximum_depth<<','<<active.task.maximum_tilt<<','<<path<<','<<work<<','<<pass<<'\n';
                    origin=center;work0=active.world.physical.body.work;error2=0;}
            }
            const bool movement=active.task.outcome==old::Outcome::Success,hold=context.retention_kind==2&&holding.passed();
            if(context.retention_category==0)result.short_passes+=movement;
            if(context.retention_category==2)result.recovery_passes+=movement&&recovered;
            if(context.retention_category==4)result.original_holds+=hold;
            if(context.retention_category==5)result.withheld_holds+=hold;
            if(context.retention_category==6)result.endurance_holds+=hold;
            const auto last=old::measure(active.task,active.world);const float work=(active.world.physical.body.work-active.task.start_work_J)/fmaxf(last.forward_m,1e-9f);
            summary<<backend<<','<<id<<','<<retention_names[context.retention_category]<<','<<context.pose<<','<<context.offset<<','<<context.maximum<<','<<movement<<','<<(movement&&recovered)<<','<<hold
                <<','<<active.task.steady_speed_m_s<<','<<active.task.speed_rmse<<','<<active.task.maximum_lateral<<','<<active.task.maximum_depth<<','<<active.task.maximum_tilt<<','<<active.task.path_angle_deg<<','<<work;
            for(int j=0;j<5;++j)summary<<','<<holding.peaks()[j];summary<<','<<holding.excursion()<<'\n';
        }
        result.controls=budget.controls-start_controls;result.substeps=budget.substeps-start_steps;
        require(result.controls==retention_controls_per_backend&&result.substeps==retention_controls_per_backend*10,"Retention physical accounting differs");
        for(int pose:{0,3}){for(int end=8;end<=60;end+=4)result.long60_retained+=int(result.keys60.count({pose,end}));
            for(int end=8;end<=300;end+=4)result.long300_retained+=int(result.keys300.count({pose,end}));}
    }
    summary.close();windows.close();require(bool(summary)&&bool(windows),"Retention score write failed");
    const bool passed=results[0].passed()&&results[1].passed();std::ofstream report(root/"retention.json");report<<"{\"schema\":1,\"study\":\"frozen_interface_comparison_v2\",\"study_version\":2,\"arithmetic_profile\":\""<<arithmetic_profile<<"\",\"cpu\":";
    retention_json(report,results[0]);report<<",\"cuda\":";retention_json(report,results[1]);report<<",\"both_passed\":"<<(passed?"true":"false")<<"}\n";report.close();require(bool(report),"Retention report write failed");return passed;
}
