#include "mirror.h"
#include "core/checkpoint.h"
#include "nn/mlp_cpu.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace mirror;
namespace fs = std::filesystem;
namespace cp = cudarl::checkpoint;
using Clock = std::chrono::steady_clock;
using Actor = cudarl::nn::CpuMlp<float>;
static void require(bool ok, const char* error) { if (!ok) throw std::runtime_error(error); }
static double seconds(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}

struct Fixture {runtime::World world; task::Task task;};
static Fixture reflected_fixture(Fixture f) {
    auto& world=f.world;
    auto& s=world.motor.physical;
    s.body=reflected(s.body);
    s.current=polar(s.current);
    world.failure_before=reflected(world.failure_before);
    world.failure_candidate=reflected(world.failure_candidate);
    for(int j=0;j<4;++j) {
        const int sign=joint_sign(j);
        world.motor.reference_rate[j]*=sign;
        s.held_targets[j]*=sign;
        s.raw_action[j]*=sign;
        s.last_actual_torques[j]*=sign;
        for(int k=0;k<10;++k) {
            s.actual_torques[k*4+j]*=sign;
            s.requested_torques[k*4+j]*=sign;
        }
    }
    auto& r=f.task.request;
    r.request.position=polar(r.request.position);
    r.request.terminal_velocity=polar(r.request.terminal_velocity);
    r.request.heading=polar(r.request.heading);
    r.start_com_model=polar(r.start_com_model);
    r.hold_anchor_model=polar(r.hold_anchor_model);
    f.task.turn_radians*=-1;
    require(runtime::inspect(world),"Reflected fixture geometry");
    return f;
}
static void issue(Fixture& f,float degrees) {
    f.task.kind=task::Turn;
    f.task.turn_radians=degrees*obs::pi/180.f;
    auto& r=f.task.request;
    const auto rotation=p::axis_rotation({0,1,0},f.task.turn_radians);
    r.request.heading=rotation.rotate(r.request.heading);
    r.request.terminal_velocity=rotation.rotate(r.request.terminal_velocity);
    r.request.position=r.start_com_model+rotation.rotate(r.request.position-r.start_com_model);
    f.task.request_changed=true;
}
static Command inference(Fixture& f,Actor& actor,float* inputs) {
    require(task::observe(f.task,f.world,inputs,false),"Motor observation failed");
    actor.forward(inputs,1);
    Command command;
    for(int j=0;j<4;++j) {
        command[j]=tanhf(actor.output()[j]);
        require(std::isfinite(command[j]),"Nonfinite command");
    }
    return command;
}
static int observation_sign(int i) {
    if(i<15) return i%3==2 ? -1:1;
    if(i<18) return i==17 ? 1:-1;
    if(i<34) return joint_sign((i-18)%4);
    if(i>=36&&i<40) return joint_sign(i-36);
    return 1;
}
static unsigned long long cpu_attempts=0, cpu_steps=0;
static void step(Fixture& f,const Command& command,runtime::SubstepTrace* traces=nullptr) {
    ++cpu_attempts;
    const auto old=f.world.motor.attempted_substeps;
    const bool accepted=runtime::step(f.world,command.data(),.005f,traces);
    cpu_steps+=f.world.motor.attempted_substeps-old;
    require(accepted,"Native runtime rejected diagnostic physics");
    ++f.task.request.controls;
    ++f.task.episode_controls;
}
static double angle(double x,double z) {return atan2(-z,x);}
static double degrees_between(double a,double b) {return atan2(sin(a-b),cos(a-b))*180./obs::pi;}
struct Score {
    double base=0,goal=0,sum_x=0,sum_z=0,depth=0,tilt=0,first=-1;
    int within=0;
    p::V heading{};
    double initial_x[10]{},initial_z[10]{},sum_control_x=0,sum_control_z=0;
    void add(int tick,const obs::RequestState& request,const runtime::SubstepTrace* trace) {
        double vx=0,vz=0;
        for(int k=0;k<10;++k) {
            const auto m=obs::measure(request,trace[k].after);
            depth=std::max(depth,double(m.depth_m));
            tilt=std::max(tilt,double(m.tilt_deg));
            vx+=m.velocity_m_s.x*.1; vz+=m.velocity_m_s.z*.1;
        }
        if(tick>=controls-80) {sum_x+=vx;sum_z+=vz;}
        const int slot=tick%10;
        sum_control_x+=vx-initial_x[slot]; sum_control_z+=vz-initial_z[slot];
        initial_x[slot]=vx; initial_z[slot]=vz;
        const auto latest=obs::measure(request,trace[9].after);
        const bool in=tick>=9&&sum_control_x*heading.x+sum_control_z*heading.z>=.06&&
            fabs(degrees_between(angle(sum_control_x,sum_control_z),goal))<=5&&latest.depth_m<=.001f&&latest.tilt_deg<=10;
        within=in ? within+1:0;
        if(within==20&&first<0) first=(tick+1)*.05;
    }
    void write(std::ostream& out) const {
        const double course=angle(sum_x,sum_z);
        const double error=fabs(degrees_between(course,goal));
        const double projected=(sum_x*heading.x+sum_z*heading.z)/80.;
        out<<','<<degrees_between(course,base)<<','<<error<<','<<projected*1000.<<','
           <<depth*1000.<<','<<tilt<<','<<first<<','<<(error<=5&&projected>=.006&&depth<=.001&&tilt<=10);
    }
};
static Actor load_actor(const std::string& path) {
    const auto bytes=cp::read(path); cp::Reader r{bytes};
    for(int i=0;i<12;++i)(void)r.value<uint32_t>();
    (void)r.value<uint64_t>(); (void)r.array<uint8_t>();
    const auto hashes=r.value<uint32_t>();
    for(uint32_t i=0;i<hashes;++i)(void)r.array<uint8_t>();
    auto weights=r.array<float>(); (void)r.array<float>();
    require(r.value<float>()==.26211655139923096f&&weights.size()==2532,"Invalid motor package");
    (void)r.value<float>(); r.finish();
    using cudarl::nn::Activation;
    Actor a{{{41,32,32,4},{Activation::Tanh,Activation::Tanh,Activation::Identity}}};
    a.params()=std::move(weights);
    return a;
}
static void stats_line(std::ostream& out,const char* backend,int id,const Stats& s) {
    out<<backend<<','<<id<<','<<s.steps<<','<<s.min_gap_m<<','<<s.failure<<','<<s.first_step<<',';
    if(s.first_field>=0)out<<names[s.first_field];
    for(auto v:s.max_error)out<<','<<v;
    out<<'\n';
}
int main(int argc,char** argv) {
    if(argc!=3){std::cerr<<"Use motor-package fresh-output-directory\n"; return 1;}
    const fs::path destination=argv[2];
    try {
        auto actor=load_actor(argv[1]);
        require(!fs::exists(destination)&&fs::create_directories(destination),"Use a fresh output path");
        const auto start=Clock::now();
        std::ofstream parity(destination/"parity.csv"),cases(destination/"cases.csv"),onset(destination/"onset.csv");
        parity<<std::setprecision(17)<<"backend,pair,substeps,min_gap_m,failure,first_breach_substep,first_field";
        for(auto name:names)parity<<','<<name;
        parity<<'\n';
        cases<<std::setprecision(12)<<"pair,warm_ticks,reflected,requested_deg,achieved_deg,abs_error_deg,projected_mm_s,max_depth_error_mm,max_tilt_deg,first_stable_s,passed\n";
        onset<<std::setprecision(12)<<"pair,warm_ticks,reflected,requested_deg,front_yaw_rad,rear_yaw_rad,front_yaw_rate,rear_yaw_rate,obs17,";
        onset<<"front_yaw_command,rear_yaw_command,first_filtered_front_yaw,first_filtered_rear_yaw,first_applied_front_yaw,first_applied_rear_yaw\n";
        Fixture warm{};warm.task.kind=task::Straight;
        auto& s=warm.world.motor.physical;
        require(obs::initialize(0,false,s.body,s.held_targets,warm.world.motor.reference_rate,s.last_actual_torques,warm.task.request),"Start");
        s.paused=false;
        require(runtime::prepare(warm.world),"Prepare");
        std::vector<GpuPair> pairs(num_cases);
        std::vector<Fixture> snapshots;
        for(int t=1;t<=52;++t){
            float inputs[41];step(warm,inference(warm,actor,inputs));
            if(t>=40&&t%4==0)snapshots.push_back(warm);
        }
        int id=0;bool passed=true;double max_input_error=0;
        for(auto& source:snapshots) for(float deg:{-15.f,-10.f,10.f,15.f}) {
            Fixture a=source,b=reflected_fixture(source);
            const int warm_ticks=int(a.task.request.controls);
            Score scores[2];
            const auto base=a.task.request.request.heading;
            scores[0].base=angle(base.x,base.z); scores[1].base=-scores[0].base;
            issue(a,deg);issue(b,-deg);Fixture independent=b;
            pairs[id].worlds[0]=a.world;pairs[id].worlds[1]=b.world;
            scores[0].heading=a.task.request.request.heading;scores[1].heading=b.task.request.request.heading;
            for(auto& score:scores)score.goal=angle(score.heading.x,score.heading.z);
            Stats stats;
            for(int t=0;t<controls;++t){
                require(seconds(start)<60,"60-second diagnostic budget exhausted");
                float input_a[41],input_i[41],input_b[41];
                const auto cmd=inference(a,actor,input_a),cmd_i=inference(independent,actor,input_i);
                require(task::observe(b.task,b.world,input_b,false),"Reflected sensor");
                for(int i=0;i<41;++i) max_input_error=std::max(max_input_error, fabs(double(observation_sign(i))*input_a[i]-input_b[i]));
                Command reflected_cmd{};
                for(int j=0;j<4;++j) {
                    reflected_cmd[j]=joint_sign(j)*cmd[j];
                    if(t<gpu_controls)pairs[id].commands[t][j]=cmd[j];
                }
                runtime::SubstepTrace ta[20],tb[20],ti[20];
                step(a,cmd,ta);step(b,reflected_cmd,tb);step(independent,cmd_i,ti);
                for(int k=0;k<10;++k)stats.compare_step(ta[k],tb[k]);
                scores[0].add(t,a.task.request,ta);scores[1].add(t,independent.task.request,ti);
                if(t==0) for(int r=0;r<2;++r) {
                    const auto& tr=r ? ti[0]:ta[0];
                    const auto& c=r ? cmd_i:cmd;
                    onset<<id<<','<<warm_ticks<<','<<r<<','<<(r?-deg:deg)<<','<<tr.before.angle[0]<<','<<tr.before.angle[2]<<','
                         <<tr.before.velocity[6]<<','<<tr.before.velocity[8]<<','<<(r?input_i[17]:input_a[17])<<','<<c[0]<<','<<c[2]<<','
                         <<tr.held_targets[0]<<','<<tr.held_targets[2]<<','<<tr.applied_torque_normalized[0]<<','<<tr.applied_torque_normalized[2]<<'\n';
                }
            }
            for(int r=0;r<2;++r) {
                cases<<id<<','<<warm_ticks<<','<<r<<','<<(r?-deg:deg);
                scores[r].write(cases);cases<<'\n';
            }
            stats_line(parity,"CPU",id,stats);
            passed=passed&&stats.failure==0&&stats.first_field<0&&stats.steps==controls*10;
            ++id;
        }
        parity.flush();cases.flush();onset.flush();
        const auto cpu_s=seconds(start);const auto gpu_start=Clock::now();
        require(cpu_s<60,"60-second diagnostic budget exhausted");
        replay_cuda(pairs.data());
        const auto gpu_s=seconds(gpu_start);
        unsigned long long gpu_attempts=0,gpu_steps=0;
        for(int i=0;i<num_cases;++i){
            const auto warm_ticks=40+4*(i/4);
            for(const auto& w:pairs[i].worlds) {
                gpu_attempts+=w.attempted_controls-warm_ticks;
                gpu_steps+=w.motor.attempted_substeps-10*warm_ticks;
            }
            const auto& stats=pairs[i].stats;
            stats_line(parity,"CUDA",i,stats);
            passed=passed&&stats.failure==0&&stats.first_field<0&&stats.steps==gpu_controls*10;
        }
        parity.close();cases.close();onset.close();
        require(parity&&cases&&onset,"Evidence writing failed");
        const bool ok=passed&&std::isfinite(max_input_error)&&max_input_error<=tolerance;
        std::ofstream report(destination/"result.json");
        report<<std::setprecision(12)<<"{\n  \"physical_symmetry_passed\": "<<(ok?"true":"false")
              <<",\n  \"matched_pairs\": "<<id<<",\n  \"max_mirrored_observation_error\": "<<max_input_error
              <<",\n  \"cpu_attempted_controls\": "<<cpu_attempts<<",\n  \"cpu_attempted_substeps\": "<<cpu_steps
              <<",\n  \"cuda_attempted_controls\": "<<gpu_attempts<<",\n  \"cuda_attempted_substeps\": "<<gpu_steps
              <<",\n  \"native_cpu_seconds\": "<<cpu_s<<",\n  \"cuda_replay_seconds\": "<<gpu_s
              <<",\n  \"elapsed_seconds\": "<<seconds(start)<<",\n  \"optimizer_steps\": 0\n}\n";
        report.close(); require(bool(report),"Gate write failed");
        std::cout<<(ok?"PASS":"FAIL")<<" cpu controls="<<cpu_attempts<<" cuda controls="<<gpu_attempts<<'\n';
        return ok?0:2;
    } catch(const std::exception& ex) {
        std::cerr<<ex.what()<<", cpu attempted controls="<<cpu_attempts<<", substeps="<<cpu_steps<<'\n';
        return 1;
    }
}
