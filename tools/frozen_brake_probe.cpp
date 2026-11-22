// Frozen CPU model6 diagnostic. No learner, optimizer, CUDA or runtime installation.
#include "motor/straight_task.h"
#include "nn/mlp_cpu.h"
#include "rl/append_hold_checkpoint.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstring>
namespace {
namespace fs=std::filesystem;namespace m=cudarl::straight_motor;namespace p=cudarl::physics;
namespace nn=cudarl::nn;namespace rl=cudarl::rl;namespace cp=cudarl::checkpoint;
constexpr int control_cap=6240,substep_cap=62400;
constexpr float pulse=.2f;
const fs::path source="assets/ppo-four-axis-checkpoints/balanced-hold-scale-336to344/B";
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
nn::MlpSpec actor_spec(){return {{41,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}};}
double unix_now(){return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();}
std::vector<uint8_t> protocol(const std::vector<uint8_t>& checkpoint,const std::vector<uint8_t>& parent){
    cp::Writer out;out.value(uint32_t{1});out.value(cp::checksum(checkpoint));out.value(cp::checksum(parent));
    for(int n:{344,6,640,10,40,10,control_cap,substep_cap,0})out.value(n);
    for(float v:{pulse,.5f,.5f,1.f,2.f,10.f,30.f})out.value(v);
    const std::string text="frozenB344; NNall4;6prefix32s;10matched2sbranches;8or0goal; +/-0.2latent singleaxis first0.5s; zero optimizer; no retry/adoption";
    out.array(std::vector<uint8_t>(text.begin(),text.end()));return out.data;
}
rl::AppendedHoldState load_actor(const fs::path& checkpoint,const fs::path& parent){
    const auto data=rl::read_appended_hold(cp::read(checkpoint.string()),cp::checksum(cp::read(parent.string())));
    require(data.ppo.actor.update==344&&data.ppo.actor_adam.steps==24260&&data.ppo.critic_adam.steps==24260&&
            data.actor_column_age==2560&&data.critic_column_age==2560,"Wrong frozenB344 or optimizer metadata");
    return data;
}
void fixture_check(nn::CpuMlp<float>& actor,const fs::path& fixture,const fs::path& root){
    std::ifstream in(fixture);require(bool(in),"Missing independent numerical fixture");std::string line;int rows=0;float max_error=0;
    while(std::getline(in,line)){if(line.empty())continue;std::stringstream row(line);std::string item;float values[45]{};int n=0;
        while(std::getline(row,item,',')){require(n<45,"Fixture too wide");values[n++]=std::stof(item);}require(n==45,"Fixture shape mismatch");
        actor.forward(values,1);for(int j=0;j<4;++j)max_error=std::max(max_error,fabsf(actor.output()[j]-values[41+j]));++rows;}
    require(rows==16&&max_error<=2e-5f,"Frozen CPU inference fixture failed");
    std::ofstream out(root/"numerical-preflight.json");out<<std::setprecision(12)<<"{\"passed\":true,\"actual_saved_state_rows\":16,\"max_latent_error\":"<<max_error<<",\"physical_controls\":0,\"optimizer_updates\":0}\n";
}
int prepare(const fs::path& root,const fs::path& fixture){
    require(!fs::exists(root),"Diagnostic identity exists; never overwrite");fs::create_directories(root);
    fs::copy_file(source/"ppo-append-update-344.bin",root/"frozen-checkpoint.bin");fs::copy_file(source/"protocol.bin",root/"source-protocol.bin");
    fs::copy_file(fixture,root/"numerical-fixture.csv");
    const auto state=load_actor(root/"frozen-checkpoint.bin",root/"source-protocol.bin");
    nn::CpuMlp<float> actor(actor_spec());actor.params()=state.ppo.actor.parameters;fixture_check(actor,root/"numerical-fixture.csv",root);
    cp::write_atomic((root/"protocol.bin").string(),protocol(cp::read((root/"frozen-checkpoint.bin").string()),cp::read((root/"source-protocol.bin").string())));
    require(actor.params()==state.ppo.actor.parameters,"Preflight changed actor");
    std::ofstream out(root/"readiness.json");out<<"{\"ready\":true,\"backend\":\"CPU original model6 equations\",\"cases\":6,\"prefix_controls\":3840,\"branch_controls\":2400,\"controls_cap\":6240,\"substep_cap\":62400,\"native_seconds\":30,\"process_seconds\":30,\"optimizer_updates\":0,\"physical_preflight_controls\":0,\"retries\":0,\"runtime_installation\":false}\n";
    std::cout<<"FROZEN_BRAKE_PREPARED zero physical/optimizer steps"<<std::endl;return 0;
}
struct Budget {
    int controls=0,substeps=0;std::chrono::steady_clock::time_point started=std::chrono::steady_clock::now();
    int case_id=-1,branch=-1,local_control=0;
    double seconds()const{return std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();}
    void guard()const{require(seconds()<30,"Native30s cap reached");}
    void step(m::World& w,const m::Action& action,const fs::path& root){
        guard();require(controls<control_cap&&substeps+10<=substep_cap,"Diagnostic physical budget exhausted");++controls;
        const auto before=w;const auto attempted=w.attempted_substeps;p::BodyState substep_before{},candidate{};
        const bool ok=m::step_state(w,action.command,nullptr,&substep_before,&candidate);
        substeps+=int(w.attempted_substeps-attempted);
        if(!ok||w.attempted_substeps-attempted!=10){
            cp::Writer proof;proof.value(uint32_t{1});proof.value(uint32_t{sizeof(m::World)});proof.value(before);proof.value(w);proof.value(substep_before);proof.value(candidate);
            cp::write_atomic((root/"failure-state.bin").string(),proof.data);
            std::ofstream f(root/"failure.json");f<<"{\"case\":"<<case_id<<",\"branch\":"<<branch<<",\"local_control\":"<<local_control
              <<",\"failure_code\":"<<w.failure_code<<",\"controls\":"<<controls<<",\"substeps\":"<<substeps<<"}\n";
            throw std::runtime_error("Physics failure; diagnostic closed without retry");}
        for(float torque:w.physical.actual_torques)require(p::finite_number(torque)&&fabsf(torque)<=1,"Torque limit or nonfinite violation");
    }
};
m::ForwardConfig config(int completed,bool continue8=false){auto c=m::StopSequence{}.config(continue8?0:completed,4);c.position_hold=true;return c;}
std::vector<float> semantic_fields(const m::World& w,const m::ForwardTask& t){
    const auto& b=w.physical.body;std::vector<float> v{b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w,b.work,b.travel,
        t.request.position.x,t.request.position.y,t.request.position.z,t.request.terminal_velocity.x,t.request.terminal_velocity.y,t.request.terminal_velocity.z,
        t.request.heading.x,t.request.heading.y,t.request.heading.z,t.start_com_model.x,t.start_com_model.y,t.start_com_model.z,
        t.steady_start_com_model.x,t.steady_start_com_model.y,t.steady_start_com_model.z,t.hold_anchor_model.x,t.hold_anchor_model.y,t.hold_anchor_model.z};
    for(float x:b.angle)v.push_back(x);for(float x:b.velocity)v.push_back(x);
    for(float x:w.physical.held_targets)v.push_back(x);for(float x:w.reference_rate)v.push_back(x);
    for(float x:w.physical.last_actual_torques)v.push_back(x);for(float x:w.physical.raw_action)v.push_back(x);
    return v;
}
struct Metrics {
    float initial_speed=0,initial_forward=0,maximum_depth=0,maximum_lane=0,maximum_tilt=0,maximum_torque=0;
    float initial_work=0,final_work=0,minimum_speed=1e6f,maximum_speed=0,final_speed=0,final_forward=0,maximum_anchor_error=0;
    double last_speed=0;int last_samples=0;
    void include(const m::Metrics& x,const m::World& w,bool late=false){
        const float speed=p::norm(x.velocity_m_s)*1000;minimum_speed=std::min(minimum_speed,speed);maximum_speed=std::max(maximum_speed,speed);final_speed=speed;
        maximum_depth=std::max(maximum_depth,x.depth_m*1000);maximum_lane=std::max(maximum_lane,x.lateral_m*1000);maximum_tilt=std::max(maximum_tilt,x.tilt_deg);
        maximum_anchor_error=std::max(maximum_anchor_error,p::norm(x.hold_error_m)*1000);final_forward=x.forward_m*1000;final_work=w.physical.body.work;
        for(float torque:w.physical.actual_torques)maximum_torque=std::max(maximum_torque,fabsf(torque));
        if(late){last_speed+=speed;++last_samples;}
    }
    bool envelope()const{return maximum_depth<=1&&maximum_lane<=2&&maximum_tilt<=10;}
    double mean_late()const{require(last_samples==10,"Last.5s sample accounting mismatch");return last_speed/10;}
    float displacement()const{return final_forward-initial_forward;}
};
void progress(const fs::path& root,const Budget& b,int completed,const char* phase){
    std::ofstream out(root/"status.json");out<<"{\"phase\":"<<std::quoted(phase)<<",\"controls\":"<<b.controls<<",\"substeps\":"<<b.substeps
       <<",\"completed_branches\":"<<completed<<",\"case\":"<<b.case_id<<",\"branch\":"<<b.branch<<",\"optimizer_updates\":0}\n";
}
void trace_row(std::ofstream& out,const char* phase,int case_id,int branch,int control,const m::World& w,const m::ForwardTask& task,
               const m::ForwardConfig& c,const float* mean,const float* z,const m::Action& action,const Metrics& metrics){
    const auto x=m::measure(task,w);out<<phase<<','<<case_id<<','<<branch<<','<<control<<','<<w.physical.time_seconds<<','<<c.target_speed_m_s*1000
        <<','<<p::norm(x.velocity_m_s)*1000<<','<<p::dot(x.velocity_m_s,task.request.heading)*1000<<','<<x.forward_m*1000<<','<<x.signed_depth_m*1000
        <<','<<x.lateral_m*1000<<','<<x.tilt_deg<<','<<x.roll_deg<<','<<x.pitch_deg<<','<<p::norm(x.hold_error_m)*1000<<','<<metrics.maximum_torque;
    for(int j=0;j<4;++j)out<<','<<mean[j]<<','<<z[j]<<','<<action.command[j]<<','<<w.physical.body.angle[j]<<','<<w.physical.body.velocity[6+j]<<','<<w.physical.last_actual_torques[j];out<<'\n';
}
int run(const fs::path& root){
    Budget budget;require(fs::exists(root/"readiness.json"),"Diagnostic not prepared");require(fs::create_directory(root/"run-started"),"Diagnostic already attempted; no retry");
    const auto checkpoint=cp::read((root/"frozen-checkpoint.bin").string()),parent=cp::read((root/"source-protocol.bin").string());
    require(cp::read((root/"protocol.bin").string())==protocol(checkpoint,parent),"Frozen diagnostic protocol changed");
    const auto state=load_actor(root/"frozen-checkpoint.bin",root/"source-protocol.bin");nn::CpuMlp<float> actor(actor_spec());actor.params()=state.ppo.actor.parameters;
    std::ofstream trace(root/"controls.csv"),summary(root/"branches.csv"),prefix(root/"prefixes.csv");trace<<std::setprecision(12);summary<<std::setprecision(12);prefix<<std::setprecision(12);
    trace<<"phase,case,branch,local_control,absolute_time_s,requested_mm_s,COM_speed_mm_s,forward_velocity_mm_s,forward_mm,depth_mm,lane_mm,tilt_deg,roll_deg,pitch_deg,anchor_error_mm,peak_torque_normalized";
    for(int j=0;j<4;++j)trace<<",mean"<<j<<",latent"<<j<<",command"<<j<<",angle"<<j<<",rate"<<j<<",last_torque"<<j;trace<<'\n';
    summary<<"case,pose,offset_mm,branch,requested_mm_s,pulse_axis,pulse_latent,initial_speed_mm_s,last05_mean_speed_mm_s,final_speed_mm_s,forward_displacement_mm,speed_reduction_vs_zero_mm_s,displacement_reduction_vs_zero_mm,max_depth_mm,max_lane_mm,max_tilt_deg,max_torque_normalized,branch_envelope_pass,braking_threshold_pass,useful_local_braking,min_speed_mm_s,max_speed_mm_s,max_anchor_error_mm,positive_work_J\n";
    prefix<<"case,pose,offset_mm,speed32_mm_s,max_depth_mm,max_lane_mm,max_tilt_deg,prefix_envelope_pass\n";
    std::string fault;int completed=0,useful=0;bool useful_pose[2]{};const double start=unix_now();
    try{
        for(int case_id=0;case_id<6;++case_id){budget.case_id=case_id;budget.branch=-1;
            const int pose=case_id%2?3:0;const float offset=case_id/2==0?0.f:case_id/2==1?-.0005f:.0005f;
            auto world=m::initial_world(pose);world.arena=m::long_swim_arena();m::ForwardTask task;const auto initial=config(0);
            require(m::begin_forward(world,initial,task)&&m::offset_episode_start(world,task,initial,offset),"Prefix initial fixed-depth setup failed");
            Metrics prefix_metrics;prefix_metrics.include(m::measure(task,world),world);
            for(int t=0;t<640;++t){budget.local_control=t+1;const auto c=config(task.controls);require(m::update_velocity_request(world,c,task),"Prefix goal update failed");
                float x[41]{};require(m::observe_appended(task,world,c,x),"Prefix observation failed");actor.forward(x,1);float mean[4];std::copy_n(actor.output(),4,mean);
                m::Action action;require(m::action_from_latent(mean,action),"Prefix action failed");budget.step(world,action,root);require(m::record_control(task,world,c).accepted,"Prefix scoring failed");
                prefix_metrics.include(m::measure(task,world),world);trace_row(trace,"prefix",case_id,-1,t+1,world,task,c,mean,mean,action,prefix_metrics);}
            require(task.controls==640&&world.physical.control_steps==640&&world.physical.substeps==6400&&!task.hold_anchor_set,"Prefix state/counter mismatch");
            prefix<<case_id<<','<<pose<<','<<offset*1000<<','<<prefix_metrics.final_speed<<','<<prefix_metrics.maximum_depth<<','<<prefix_metrics.maximum_lane<<','<<prefix_metrics.maximum_tilt<<','<<prefix_metrics.envelope()<<'\n';prefix.flush();
            cp::Writer captured;captured.value(uint32_t{1});captured.value(uint32_t{sizeof(m::World)});captured.value(world);captured.value(uint32_t{sizeof(m::ForwardTask)});captured.value(task);
            cp::write_atomic((root/("captured-case-"+std::to_string(case_id)+".bin")).string(),captured.data);
            const auto captured_fields=semantic_fields(world,task);Metrics results[10];
            for(int branch=0;branch<10;++branch){budget.branch=branch;progress(root,budget,completed,"branch");auto w=world;auto t=task;
                require(semantic_fields(w,t)==captured_fields&&w.reference_rate[0]==world.reference_rate[0]&&w.attempted_substeps==world.attempted_substeps&&t.last_world_control==task.last_world_control&&t.controls==task.controls,"Branch snapshot clone differs");
                const auto branch_start=m::measure(t,w);auto& metrics=results[branch];metrics.initial_forward=branch_start.forward_m*1000;metrics.initial_speed=p::norm(branch_start.velocity_m_s)*1000;metrics.initial_work=w.physical.body.work;
                metrics.include(branch_start,w);const int axis=branch>=2?(branch-2)/2:-1;const float delta=branch>=2?(branch%2?pulse:-pulse):0.f;
                for(int k=0;k<40;++k){budget.local_control=k+1;const auto c=config(t.controls,branch==0);require(m::update_velocity_request(w,c,t),"Branch goal update failed");
                    require(p::norm(t.start_com_model-task.start_com_model)==0&&p::norm(t.request.heading-task.request.heading)==0,"Branch original reference changed");
                    if(branch!=0)require(t.hold_anchor_set&&t.hold_anchor_model.y==task.start_com_model.y,"Holding anchor/depth setup failed");
                    float x[41]{};require(m::observe_appended(t,w,c,x),"Branch observation failed");actor.forward(x,1);float mean[4],latent[4];std::copy_n(actor.output(),4,mean);std::copy_n(mean,4,latent);
                    if(axis>=0&&k<10)latent[axis]+=delta;
                    for(int j=0;j<4;++j)require(latent[j]==mean[j]+(j==axis&&k<10?delta:0.f),"Diagnostic perturbation leaked to another axis/time");
                    m::Action action;require(m::action_from_latent(latent,action),"Branch action failed");budget.step(w,action,root);require(m::record_control(t,w,c).accepted,"Branch scoring failed");
                    metrics.include(m::measure(t,w),w,k>=30);trace_row(trace,"branch",case_id,branch,k+1,w,t,c,mean,latent,action,metrics);}
                require(w.physical.control_steps==680&&w.physical.substeps==6800&&t.controls==680&&metrics.last_samples==10,"Branch physical/sample accounting mismatch");
                ++completed;trace.flush();
            }
            for(int branch=0;branch<10;++branch){const auto& metrics=results[branch];const double speed_reduction=results[1].mean_late()-metrics.mean_late(),displacement_reduction=results[1].displacement()-metrics.displacement();
                const bool threshold=speed_reduction>=.5&&displacement_reduction>=.5;const bool candidate=branch>=2&&threshold&&metrics.envelope();useful+=candidate;if(candidate)useful_pose[case_id%2]=true;
                const int axis=branch>=2?(branch-2)/2:-1;const float delta=branch>=2?(branch%2?pulse:-pulse):0;
                summary<<case_id<<','<<pose<<','<<offset*1000<<','<<branch<<','<<(branch==0?8:0)<<','<<axis<<','<<delta<<','<<metrics.initial_speed<<','<<metrics.mean_late()<<','<<metrics.final_speed<<','<<metrics.displacement()
                    <<','<<speed_reduction<<','<<displacement_reduction<<','<<metrics.maximum_depth<<','<<metrics.maximum_lane<<','<<metrics.maximum_tilt<<','<<metrics.maximum_torque<<','<<metrics.envelope()<<','<<threshold<<','<<candidate
                    <<','<<metrics.minimum_speed<<','<<metrics.maximum_speed<<','<<metrics.maximum_anchor_error<<','<<metrics.final_work-metrics.initial_work<<'\n';}
            summary.flush();std::cout<<"CASE_COMPLETED "<<case_id<<"/5 controls="<<budget.controls<<" useful_so_far="<<useful<<std::endl;
        }
        require(budget.controls==6240&&budget.substeps==62400&&completed==60,"Final diagnostic accounting mismatch");require(actor.params()==state.ppo.actor.parameters,"Frozen actor changed");budget.guard();
    }catch(const std::exception& e){fault=e.what();}
    trace.flush();summary.flush();prefix.flush();
    std::ofstream out(root/"result.json");out<<std::setprecision(17)<<"{\"fault\":"<<std::quoted(fault)<<",\"started_unix\":"<<start<<",\"ended_unix\":"<<unix_now()<<",\"native_seconds\":"<<budget.seconds()
        <<",\"controls\":"<<budget.controls<<",\"substeps\":"<<budget.substeps<<",\"completed_branches\":"<<completed<<",\"useful_local_responses\":"<<useful
        <<",\"useful_pose0\":"<<(useful_pose[0]?"true":"false")<<",\"useful_pose3\":"<<(useful_pose[1]?"true":"false")<<",\"both_poses_local_braking\":"<<(useful_pose[0]&&useful_pose[1]?"true":"false")
        <<",\"optimizer_updates\":0,\"actor_unchanged\":true,\"runtime_installed\":false,\"retries\":0,\"closed\":true}\n";
    progress(root,budget,completed,fault.empty()?"done":"failed");std::cout<<"FROZEN_BRAKE_CLOSED useful="<<useful<<" fault="<<fault<<std::endl;return fault.empty()?0:3;
}
}
int main(int argc,char** argv){try{
    require(argc>=3,"Use --prepare DIR FIXTURE or --run DIR");const std::string mode=argv[1];const auto root=fs::absolute(argv[2]);
    if(mode=="--prepare"){require(argc==4,"Prepare requires numerical fixture");return prepare(root,fs::absolute(argv[3]));}
    if(mode=="--run"){require(argc==3,"Run takes frozen directory only");return run(root);}
    throw std::runtime_error("Unknown frozen diagnostic mode");
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}}
