#ifndef CUDARL_TURNING_RETENTION_H
#define CUDARL_TURNING_RETENTION_H
// Native CPU retained-skill assessment. The original task and all old passing
// windows are unchanged. Tail-gait fields remain diagnostics only.
#include "motor/gait_task_v1.h"
#include "motor/hold_assessment.h"
namespace turning_study {
void retention_record_pre(const m::World& world,float* values) {
    const auto& b=world.physical.body;
    values[0]=b.position.x;values[1]=b.position.y;values[2]=b.position.z;
    values[3]=b.orientation.x;values[4]=b.orientation.y;values[5]=b.orientation.z;values[6]=b.orientation.w;
    for(int j=0;j<4;++j)values[7+j]=b.angle[j];
    for(int j=0;j<10;++j)values[11+j]=b.velocity[j];
    for(int j=0;j<4;++j){values[21+j]=world.physical.held_targets[j];values[25+j]=world.physical.last_actual_torques[j];values[29+j]=world.reference_rate[j];}
}
struct retention_SwimCase {int pose=0,controls=160;float offset=0;};
struct retention_SwimResult {int movement_passed=0,gait_passed=0,completed=0,window_passed=0,window_total=0,gait_window_passed=0,gait_window_total=0,recovery_passed=0;double mean_cost=0;};
void retention_state_header(std::ostream& out){out<<"case,control";for(int j=0;j<33;++j)out<<",pre"<<j;
    for(int j=0;j<4;++j)out<<",base"<<j;for(int j=0;j<4;++j)out<<",latent"<<j;
    for(int j=0;j<23;++j)out<<",post"<<j;out<<",reward\n";}
void retention_state_row(std::ostream& out,int id,int control,const float* pre,const float* base,const float* latent,const m::World& world,float reward){
    out<<id<<','<<control;for(int j=0;j<33;++j)out<<','<<pre[j];for(int j=0;j<4;++j)out<<','<<base[j];for(int j=0;j<4;++j)out<<','<<latent[j];
    float post[33];retention_record_pre(world,post);for(int j=0;j<21;++j)out<<','<<post[j];out<<','<<world.physical.body.work<<','<<world.physical.mean_squared_torque<<','<<reward<<'\n';}
void retention_gait_columns(std::ostream& out){out<<",mean_pitch1_deg,mean_pitch3_deg,mean_tail_vertical_bend_deg,centered_joint_RMS_deg,gait_passed";}
void retention_gait_values(std::ostream& out,const cudarl::gait_task_v1::Measurements& score){out<<','<<score.mean_pitch(0)<<','<<score.mean_pitch(1)<<','<<score.mean_bend()<<','<<score.centered_joint_rms_degrees()<<','<<score.passed();}
retention_SwimResult retention_assess_swim(const std::vector<float>& means,const std::vector<retention_SwimCase>& cases,const fs::path& directory,Budget& budget,bool long_windows=false,bool recovery=false){
    require(fs::create_directory(directory),"Retained-skill swimming assessment identity exists");nn::CpuMlp<float> body(spec(4));body.params()=resume_state->actor.parameters;
    std::ofstream trace(directory/"controls.csv"),summary(directory/"summary.csv"),windows(directory/"windows.csv"),initial(directory/"initial.csv");
    trace<<std::setprecision(12);summary<<std::setprecision(12);windows<<std::setprecision(12);initial<<std::setprecision(12);retention_state_header(trace);
    summary<<"case,pose,offset_mm,controls,movement_passed,steady_mm_s,RMSE_mm_s,lateral_mm,depth_mm,tilt_deg,path_deg,work_J_per_m,return,recovery_passed";retention_gait_columns(summary);summary<<'\n';
    windows<<"case,pose,end_s,steady_mm_s,RMSE_mm_s,lateral_mm,depth_mm,tilt_deg,path_deg,work_J_per_m,movement_passed,gait_scored";retention_gait_columns(windows);windows<<'\n';
    initial<<"case,pose,offset_mm,controls,arena_half_width_m,arena_half_depth_m,start_com_x,start_com_y,start_com_z,heading_x,heading_y,heading_z\n";
    retention_SwimResult result;
    for(int id=0;id<int(cases.size());++id){const auto c=cases[id];auto world=m::initial_world(c.pose);world.arena=c.controls==6000?cudarl::contact::Arena{3.f,.4f}:m::long_swim_arena();
        auto config=cudarl::gait_task_v1::config(c.controls);m::ForwardTask t;require(m::begin_forward(world,config,t)&&m::offset_episode_start(world,t,config,c.offset),"Retained-skill swimming reset rejected");
        initial<<id<<','<<c.pose<<','<<c.offset*1000<<','<<c.controls<<','<<world.arena.half_width_m<<','<<world.arena.half_depth_m<<','<<t.start_com_model.x<<','<<t.start_com_model.y<<','<<t.start_com_model.z<<','<<t.request.heading.x<<','<<t.request.heading.y<<','<<t.request.heading.z<<'\n';
        cudarl::gait_task_v1::Measurements gait,window_gait;double sum_reward=0,error2=0;auto origin=cudarl::motor::motion(world.physical.body).com_model;float work0=world.physical.body.work;bool recovered=true;
        for(int control=1;control<=c.controls;++control){time_guard();float observation[41],pre[33],raw[4];retention_record_pre(world,pre);
            require(m::observe_appended(t,world,config,observation),"Retained-skill CPU observation rejected");body.forward(observation,1);policy::motor_latents<float>(false,false,means.data(),body.output(),saved_hold_gain,nullptr,raw);
            m::Action action;require(m::action_from_latent(raw,action),"Retained-skill CPU action rejected");budget.step(world,action.command);
            const auto interval=m::record_control(t,world,config);require(interval.accepted&&gait.record(control,world.physical.body)&&window_gait.record(control,world.physical.body),"Retained-skill CPU task/measurement rejected");
            sum_reward+=interval.reward;retention_state_row(trace,id,control,pre,body.output(),raw,world,interval.reward);
            if(control>40)error2+=interval.metrics.speed_error_m_s*interval.metrics.speed_error_m_s;
            if(control==40){origin=cudarl::motor::motion(world.physical.body).com_model;work0=world.physical.body.work;}
            if(recovery&&c.offset!=0&&control>c.controls-160)recovered=recovered&&interval.metrics.depth_m<fabsf(c.offset)*.5f;
            if(long_windows&&control%80==0){const auto center=cudarl::motor::motion(world.physical.body).com_model;
                const auto delta=(center-origin)*p::scale;const float forward=p::dot(delta,t.request.heading);const auto side=delta-t.request.heading*forward;
                const int samples=control==80?40:80;const float speed=forward/(samples*p::dt),rmse=sqrtf(float(error2/samples)),
                  path=atan2f(p::norm({side.x,0,side.z}),forward)*180/cudarl::movement::pi,work=(world.physical.body.work-work0)/fmaxf(forward,1e-9f);
                const bool pass=speed>=config.minimum_speed_m_s&&speed<=config.maximum_speed_m_s&&rmse<=config.speed_rmse_limit&&
                  t.maximum_lateral<=config.lateral_limit_m&&t.maximum_depth<=config.depth_limit_m&&t.maximum_tilt<=config.tilt_limit_deg&&
                  path<=config.path_angle_limit_deg&&work<=config.work_per_distance_limit;
                ++result.window_total;result.window_passed+=pass;
                if(control>=160){++result.gait_window_total;result.gait_window_passed+=window_gait.passed();}
                windows<<id<<','<<c.pose<<','<<control*.05<<','<<speed*1000<<','<<rmse*1000<<','<<t.maximum_lateral*1000<<','<<t.maximum_depth*1000<<','<<t.maximum_tilt<<','<<path<<','<<work<<','<<pass<<','<<(control>=160);retention_gait_values(windows,window_gait);windows<<'\n';
                origin=center;work0=world.physical.body.work;error2=0;window_gait={};}
        }
        const bool movement=t.outcome==m::Outcome::Success;result.movement_passed+=movement;result.gait_passed+=gait.passed();++result.completed;
        result.recovery_passed+=movement&&recovered;const double episode_return=sum_reward/(c.controls*.05)-gait.cost()-(movement?0:cudarl::gait_task_v1::movement_failure_cost);result.mean_cost-=episode_return/cases.size();
        const auto last=m::measure(t,world);const float work=(world.physical.body.work-t.start_work_J)/fmaxf(last.forward_m,1e-9f);
        summary<<id<<','<<c.pose<<','<<c.offset*1000<<','<<c.controls<<','<<movement<<','<<t.steady_speed_m_s*1000<<','<<t.speed_rmse*1000<<','<<t.maximum_lateral*1000<<','<<t.maximum_depth*1000<<','<<t.maximum_tilt<<','<<t.path_angle_deg<<','<<work<<','<<episode_return<<','<<(movement&&recovered);retention_gait_values(summary,gait);summary<<'\n';
    }
    for(auto* out:{&trace,&summary,&windows,&initial}){out->close();require(bool(*out),"Retained-skill CPU swim evidence write failed");}return result;
}
class retention_HoldScore {
    int total_,next_=0;bool valid_=true;float peak_[5]{},forward_window_=0,excursion_=0;double square_speed_=0;
public:
    explicit retention_HoldScore(int total):total_(total){valid_=total>=160;}
    bool record(int control,const m::Metrics& z){if(!valid_||control!=next_||control>total_){valid_=false;return false;}
        const float v[5]={p::norm(z.velocity_m_s),p::norm(z.hold_error_m),z.depth_m,z.lateral_m,z.tilt_deg};
        for(int j=0;j<5;++j){if(!std::isfinite(v[j])||v[j]<0){valid_=false;return false;}peak_[j]=fmaxf(peak_[j],v[j]);}
        if(!std::isfinite(z.forward_m)){valid_=false;return false;}if(control)square_speed_+=double(v[0])*v[0];
        if(control==total_-160)forward_window_=z.forward_m;if(control>=total_-160)excursion_=fmaxf(excursion_,fabsf(z.forward_m-forward_window_));++next_;return true;}
    bool passed()const{return valid_&&next_==total_+1&&peak_[0]<=.0005f&&peak_[1]<=.002f&&peak_[2]<=.001f&&peak_[3]<=.002f&&peak_[4]<=10.f&&excursion_<=.002f;}
    const float* peaks()const{return peak_;}float excursion()const{return excursion_;}
};
int retention_assess_holds(const std::vector<float>& means,const std::vector<retention_SwimCase>& cases,const fs::path& directory,Budget& budget){
    require(fs::create_directory(directory),"Retained-skill hold assessment identity exists");nn::CpuMlp<float> body(spec(4));body.params()=resume_state->actor.parameters;
    std::ofstream trace(directory/"controls.csv"),summary(directory/"summary.csv");trace<<std::setprecision(12);summary<<std::setprecision(12);retention_state_header(trace);
    summary<<"case,pose,offset_mm,controls,passed,max_speed_mm_s,max_anchor_mm,max_depth_mm,max_lane_mm,max_tilt_deg,final8_excursion_mm\n";int passed=0;
    for(int id=0;id<int(cases.size());++id){const auto c=cases[id];auto world=m::initial_world(c.pose);world.arena=m::long_swim_arena();auto config=cudarl::gait_task_v1::config(c.controls);
        config.target_speed_m_s=0;config.stationary_request=true;config.position_hold=true;m::ForwardTask t;
        require(m::begin_forward(world,config,t)&&m::offset_episode_start(world,t,config,c.offset)&&m::update_velocity_request(world,config,t),"Retained-skill hold reset rejected");
        retention_HoldScore score(c.controls);require(score.record(0,m::measure(t,world)),"Retained-skill hold reset scoring rejected");
        for(int control=1;control<=c.controls;++control){time_guard();float observation[41],pre[33],raw[4];retention_record_pre(world,pre);
            require(m::update_velocity_request(world,config,t)&&m::observe_appended(t,world,config,observation),"Retained-skill hold observation rejected");
            body.forward(observation,1);policy::motor_latents<float>(false,true,means.data(),body.output(),saved_hold_gain,nullptr,raw);m::Action action;
            require(m::action_from_latent(raw,action),"Retained-skill hold action rejected");budget.step(world,action.command);
            const auto interval=m::record_hold_control(t,world,config);require(interval.accepted&&score.record(control,interval.metrics),"Retained-skill hold score rejected");
            retention_state_row(trace,id,control,pre,body.output(),raw,world,interval.reward);}
        passed+=score.passed();summary<<id<<','<<c.pose<<','<<c.offset*1000<<','<<c.controls<<','<<score.passed();for(int j=0;j<5;++j)summary<<','<<score.peaks()[j]*(j==4?1:1000);summary<<','<<score.excursion()*1000<<'\n';}
    trace.close();summary.close();require(bool(trace)&&bool(summary),"Retained-skill hold evidence write failed");return passed;
}
std::vector<retention_SwimCase> retention_short_cases(){return {{0,160,0},{1,160,0},{2,160,0},{3,160,0}};}
std::vector<retention_SwimCase> retention_original_holds(){std::vector<retention_SwimCase> out;for(float offset:{0.f,-.0005f,.0005f})for(int pose:{0,3})out.push_back({pose,1280,offset});return out;}
std::vector<retention_SwimCase> retention_withheld_holds(){std::vector<retention_SwimCase> out;for(float offset:{0.f,-.00025f,.00025f})for(int pose:{1,2})out.push_back({pose,1280,offset});return out;}
int retention_retained_windows(const fs::path& csv,int last){std::ifstream input(csv);std::string line;std::getline(input,line);std::set<std::pair<int,int>> passed;
    while(std::getline(input,line)){std::stringstream row(line);std::string value;std::vector<double> values;while(std::getline(row,value,','))values.push_back(std::stod(value));
        require(values.size()==17,"Retained-skill long-window schema differs");if(values[10]==1)passed.emplace(int(values[1]),int(values[2]));}
    int count=0;for(int pose:{0,3})for(int end=8;end<=last;end+=4)count+=int(passed.count({pose,end}));return count;}

struct Retention {
    int short_passes=0,original_holds=0,long60_windows=0,long60_retained=0;
    int recovery_passes=0,long300_windows=0,long300_retained=0,withheld_holds=0,endurance_holds=0;
};
Retention assess_retention(const std::vector<float>& means,const fs::path& directory,Budget& budget,bool full) {
    require(means.size()==16,"Wrong retained motor parameter count");finite(means);
    require(fs::create_directory(directory),"Retained-skill assessment identity exists");
    const int controls_before=budget.controls,substeps_before=budget.substeps;
    Retention result;
    const auto short_score=retention_assess_swim(means,retention_short_cases(),directory/"short",budget);
    const auto long_score=retention_assess_swim(means,{{0,1200,0},{3,1200,0}},directory/"long60",budget,true);
    result.short_passes=short_score.movement_passed;
    result.original_holds=retention_assess_holds(means,retention_original_holds(),directory/"original-holds",budget);
    result.long60_windows=long_score.window_passed;
    result.long60_retained=retention_retained_windows(directory/"long60/windows.csv",60);
    if(full) {
        const auto recovery=retention_assess_swim(means,{{0,1280,0},{0,1280,-.0005f},{0,1280,.0005f}},directory/"recovery",budget,false,true);
        const auto five_minute=retention_assess_swim(means,{{0,6000,0},{3,6000,0}},directory/"long300",budget,true);
        result.recovery_passes=recovery.recovery_passed;
        result.long300_windows=five_minute.window_passed;
        result.long300_retained=retention_retained_windows(directory/"long300/windows.csv",300);
        result.withheld_holds=retention_assess_holds(means,retention_withheld_holds(),directory/"withheld-holds",budget);
        result.endurance_holds=retention_assess_holds(means,{{0,6000,0},{3,6000,0}},directory/"endurance-holds",budget);
    }
    const int expected=full?46240:10720;
    require(budget.controls-controls_before==expected&&budget.substeps-substeps_before==expected*10,"Retained-skill physical accounting differs");
    std::ofstream summary(directory/"retention.json");
    summary<<"{\"full\":"<<(full?"true":"false")<<",\"controls\":"<<expected
      <<",\"substeps\":"<<expected*10<<",\"short_passes\":"<<result.short_passes
      <<",\"original_holds\":"<<result.original_holds<<",\"long60_windows\":"<<result.long60_windows
      <<",\"long60_retained\":"<<result.long60_retained<<",\"recovery_passes\":"<<result.recovery_passes
      <<",\"long300_windows\":"<<result.long300_windows<<",\"long300_retained\":"<<result.long300_retained
      <<",\"withheld_holds\":"<<result.withheld_holds<<",\"endurance_holds\":"<<result.endurance_holds<<"}\n";
    summary.close();require(bool(summary),"Cannot preserve retained-skill result");return result;
}
bool retention_passed(const Retention& result,bool full) {
    return result.short_passes==4&&result.original_holds==6&&result.long60_retained==28&&
      (!full||(result.recovery_passes==3&&result.long300_retained==148&&result.withheld_holds==6&&result.endurance_holds==2));
}
}
#endif
