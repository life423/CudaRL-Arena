#include "front_pad_protocol.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
using namespace pilot;
namespace fs=std::filesystem;
namespace cp=cudarl::checkpoint;
using Clock=std::chrono::steady_clock;
constexpr char Checkpoint[]="assets/ppo-four-axis-checkpoints/update-336-episode64/ppo-update-336.bin";
constexpr char Protocol[]="assets/ppo-four-axis-checkpoints/update-336-episode64/protocol.bin";
constexpr int Population=8, Generations=3, TestTicks=400, TrainTicks=400, Cases=12, SwimTicks=3040;
constexpr int Cap=3*6*1280+2*SwimTicks;
constexpr int Features=6, Params=Outputs*Features;
using X=std::array<float,Inputs>;
using W=std::array<float,Params>;
struct Budget {
    Clock::time_point start=Clock::now(); int controls=0,substeps=0,trial=0;
    double seconds() const {return std::chrono::duration<double>(Clock::now()-start).count();}
    void step(m::World& w,const m::Action& a){
        need(seconds()<120. && controls<Cap && substeps<=10*(Cap-1),"compute cap");
        ++controls;const auto n=w.attempted_substeps;
        bool ok=m::step_state(w,a.command);substeps+=int(w.attempted_substeps-n);
        need(ok&&w.attempted_substeps-n==10,"native transition fault; stop");
    }
};
constexpr float quiet_limit=.035f; // rad/s; allows about 2 degrees/second per axis.
static auto weights(){
    const auto a=cp::read(Checkpoint), b=cp::read(Protocol);
    auto s=rl::read_ppo_training_state(a,cp::checksum(b),1247037243u,cudarl::motor_observation::task_scale);
    need(s.actor.update==336&&s.actor.reward_version==5,"parent checkpoint/reward mismatch");
    return s.actor.parameters;
}
struct Controller{
    nn::CpuMlp<float> a{spec()};
    W w{};
    explicit Controller(const std::vector<float>& parent){a.params()=adapt(parent);}
    std::array<float,4> infer(const X& x,float drive=0.f,float phase=0.f){
        a.forward(x.data(),1);std::array<float,4> z{};std::copy_n(a.output(),4,z.begin());
        if(x[41]==1.f){
            // A learned Hold readout of the shared motor plus physical error inputs.
            // No override of pose, velocity, physics, torques or the target actuator.
            const float dp=std::clamp(10.f*x[40],-2.f,2.f);
            const float v=std::clamp(16.f*x[12],-2.f,2.f);
            for(int j=0;j<4;++j){const float* u=&w[j*Features];z[j]=u[0]+u[1]*z[j]+u[2]*dp+u[3]*v+u[4]*(j==0?std::fabs(drive):drive)*std::sin(phase)+u[5]*(j==0?std::fabs(drive):drive)*std::cos(phase);}
        }
        return z;
    }
};
struct Case{int pose;float distance;float side;};
constexpr Case suite[Cases]={{1,.00195f,0},{2,.00195f,0},{1,.002f,0},{2,.002f,0},
                            {1,.00205f,0},{2,.00205f,0},{1,.002f,.00025f},{2,.002f,-.00025f},
                            {1,.00192f,-.00016f},{2,.00192f,.00016f},{1,.00208f,.00020f},{2,.00208f,-.00020f}};
static float radius(const m::Metrics& z){return std::hypot(z.hold_error_m.x,z.hold_error_m.z);}
static float angular(const m::World& w){
    float q=0;for(int j=3;j<10;++j){q=std::max(q,std::fabs(w.physical.body.velocity[j]));}return q;
}
static float quiet_bonus(const m::Metrics& z,float rate){
    if(!safe(z)||!in_pad(z))return 0.f;
    return 1.f/(1.f+(rate/quiet_limit)*(rate/quiet_limit));
}
struct Result{double score=0.;int first=-1,quiet=0,streak=0,tail=0;bool safe_all=true;float end_dist=0,end_speed=0;};
static Result episode(Controller& actor,Case def,int n,Budget& b,std::ostream& trace,int id,float speed){
    m::World w;auto t=setup_front({def.pose,def.distance},w);auto c=hold();c.maximum_controls=n;
    // Optional lateral offset changes only the reset position. Pad stays fixed thereafter.
    const auto h=t.request.heading; w.physical.body.velocity[0]=h.x*speed;w.physical.body.velocity[2]=h.z*speed;w.external_work=cudarl::movement::kinetic(w.physical.body);
    const p::V side{-h.z,0,h.x};
    w.physical.body.position=w.physical.body.position+side*(def.side/p::scale);
    t.steady_start_com_model=m::old::motion(w.physical.body).com_model;
    const auto anchor=t.hold_anchor_model;Result r{};double sum=0;int consecutive=0;
    auto z=m::measure(t,w);
    for(int i=1;i<=n;++i){
        const float old=radius(z);
        X x{};need(observation(t,w,c,x.data()),"Hold observation");
        const float dx=-p::dot(z.hold_error_m,h)*1000.f;
        const float vx=p::dot(z.velocity_m_s,h)*1000.f;
        const float outer=std::copysign(std::max(0.f,std::fabs(dx)-.55f),dx); const float error=.16f*outer-vx; const float drive=std::copysign(std::clamp((std::fabs(error)-.065f)/.18f,0.f,1.f),error);
        const auto a=actor.infer(x,drive,float(i-1)*p::dt*cudarl::movement::pi);m::Action command;need(m::action_from_latent(a.data(),command),"Hold action");
        b.step(w,command);
        const auto step=m::record_hold_control(t,w,c);
        need(step.accepted && step.done==(i==n) && p::norm(t.hold_anchor_model-anchor)==0,"Hold episode reset/failed");
        z=step.metrics;const float d=radius(z),s=p::norm(z.velocity_m_s),rate=angular(w);
        const bool ok=safe(z), q=ok&&in_pad(z)&&rate<=quiet_limit;
        r.safe_all &= ok;
        if(in_pad(z)&&ok&&r.first<0)r.first=i;
        r.quiet+=q;consecutive=q?consecutive+1:0;r.streak=std::max(r.streak,consecutive);
        // Reuse the food lesson: dense improvement toward a real target. Unlike
        // food, occupancy stays valuable after arrival; swaying fast earns less.
        sum += 3.*std::clamp((old-d)/.001f,-1.f,1.f)+quiet_bonus(z,rate)
               -.35*std::min(5.f,d/.001f)-.30*std::min(5.f,s/.0005f) - (ok?0.:4.);
        trace<<id<<','<<i<<','<<d*1000<<','<<s*1000<<','<<z.tilt_deg<<','<<rate<<','<<ok<<','<<q;
        for(float u:command.command)trace<<','<<u;trace<<'\n';
    }
    r.tail=consecutive;r.score=sum/n+double(r.tail)/n;r.end_dist=radius(z)*1000;r.end_speed=p::norm(z.velocity_m_s)*1000;
    return r;
}
static void header(std::ostream& f){
    need(f.good(),"trace file open");
    f<<std::setprecision(10)<<"case,control,radial_mm,speed_mm_s,tilt_deg,max_angular_rad_s,safe,quiet,raw0,raw1,raw2,raw3\n";
}
static void swim_trace(Controller& actor,Budget& b,const fs::path& path){
    std::ofstream f(path);need(f.good(),"Swim trace open");
    // Compare raw float trajectories for all short and long starts.
    f<<std::setprecision(10);
    for(int run=0;run<6;++run){
        const int pose=run<4?run:(run==4?0:3);
        const int n=run<4?160:1200;auto c=swim(n);auto w=m::initial_world(pose);if(run>=4)w.arena=m::long_swim_arena();
        m::ForwardTask t;need(m::begin_forward(w,c,t),"Swim start");
        for(int i=1;i<=n;++i){X x{};need(observation(t,w,c,x.data())&&x[41]==0.f,"Swim observation");
            auto z=actor.infer(x);m::Action a;need(m::action_from_latent(z.data(),a),"Swim action");b.step(w,a);
            const auto q=m::record_hold_control(t,w,c);need(q.accepted&&q.done==(i==n),"Swim transition");
            f<<run<<','<<i<<','<<q.metrics.forward_m<<','<<q.metrics.speed_error_m_s<<','<<q.metrics.depth_m<<','<<q.metrics.lateral_m<<','<<q.metrics.tilt_deg;
            for(float v:z) f<<','<<v;for(float v:w.physical.body.angle)f<<','<<v;f<<'\n';
        }
    }
    f.flush();need(f.good(),"Swim trace write");
}
static std::string bytes(const fs::path& f){std::ifstream in(f,std::ios::binary);need(in.good(),"fingerprint file open");return {std::istreambuf_iterator<char>(in),{}};}
static void dry(const std::vector<float>& p){
    Controller c(p);nn::CpuMlp<float> orig(spec(40));orig.params()=p;
    for(int j=0;j<Params;++j)c.w[j]=.01f*(j+1);
    for(int i=0;i<4;++i){X x{};m::World w=m::initial_world(i);m::ForwardTask t;auto conf=swim(160);
        need(m::begin_forward(w,conf,t)&&observation(t,w,conf,x.data()),"dry Swim setup");
        auto a=c.infer(x);orig.forward(x.data(),1);
        need(std::equal(a.begin(),a.end(),orig.output()),"Swim raw outputs differ");
        need(w.attempted_substeps==0,"dry physics");
    }
    m::Metrics z{};need(quiet_bonus(z,0)==1.f&&quiet_bonus(z,quiet_limit)==.5f,"quiet scaling");
    z.hold_error_m.x=.00101f;need(quiet_bonus(z,0)==0.f,"bonus outside pad");
    z.hold_error_m.x=0;z.velocity_m_s.x=.00051f;need(quiet_bonus(z,0)==0.f,"bonus during fast crossing");
    z.velocity_m_s.x=0;z.tilt_deg=10.1f;need(quiet_bonus(z,0)==0.f,"bonus in unsafe posture");
    std::cout<<"DRY PASS: gates and exact 336 Swim logits; 0 controls.\n";
}

static void load_candidate(Controller& c){
    const auto a=cp::read("build/quiet-gait-v2/recorded-run/candidate-research-only.bin");cp::Reader cp_in{a};
    need(cp_in.value<uint32_t>()==0x32475148 && cp_in.value<uint32_t>()==1 && cp_in.value<uint32_t>()==336,"candidate format mismatch");
    const auto w=cp_in.array<float>(Params);cp_in.finish();need(w.size()==Params,"candidate shape mismatch");
    for(float v:w)need(std::isfinite(v),"nonfinite candidate coefficient");
    std::copy(w.begin(),w.end(),c.w.begin());
}
static int eval(const std::vector<float>& p){
    const fs::path out="build/quiet-gait-endurance-v1/recorded-run";
    need(fs::create_directory(out),"fresh eval folder required"); Budget b;int tiers=0;
    auto status=[&](const std::string& result){std::ofstream f(out/"status.json");f<<std::setprecision(10)<<"{\"result\":"<<std::quoted(result)<<",\"tiers_completed\":"<<tiers<<",\"cases\":"<<b.trial<<",\"controls\":"<<b.controls<<",\"substeps\":"<<b.substeps<<",\"seconds\":"<<b.seconds()<<"}\n";};
    try{
        status("running"); Controller c(p);
        swim_trace(c,b,out/"swim-reference.csv");load_candidate(c);
        swim_trace(c,b,out/"swim-candidate.csv");need(bytes(out/"swim-reference.csv")==bytes(out/"swim-candidate.csv"),"Swim changed");
        std::ofstream f(out/"endurance.csv"),trace(out/"trace.csv");header(trace);need(f.good(),"endurance summary open");
        f<<std::setprecision(10)<<"case,initial_speed_mm_s,pose,initial_dist_mm,side_mm,first_settled_s,best_quiet_s,final_quiet_s,final_radial_mm,final_speed_mm_s,all_safe,pass\n";
        for(float s:{.0005f,.00055f,.0006f}){
            int count=0;
            for(int pose:{1,2})for(int k=0;k<3;++k){
                const float dist=.00196f+k*.00004f, side=k==0?.00014f:k==1?-.00021f:.00028f;
                const Case d{pose,dist,side};auto r=episode(c,d,1280,b,trace,b.trial,s);
                const bool pass=r.safe_all&&r.tail>=321;count+=pass;
                f<<b.trial++<<','<<s*1000<<','<<pose<<','<<dist*1000<<','<<side*1000<<','<<r.first*.05f<<','
                 <<(std::max(0,r.streak-1)*.05f)<<','<<(std::max(0,r.tail-1)*.05f)<<','<<r.end_dist<<','<<r.end_speed<<','<<r.safe_all<<','<<pass<<'\n';
            }
            ++tiers;f.flush();trace.flush();need(f.good()&&trace.good(),"evidence write");
            if(count!=6){status("tier_"+std::to_string(tiers)+"_held_"+std::to_string(count)+"_of_6_no_faster_trials");return 0;}
        }
        need(b.controls==Cap&&b.substeps==10*Cap,"evaluation budget mismatch");status("all_three_tiers_pass");
    }catch(const std::exception& e){status(std::string("fault: ")+e.what());throw;}
    return 0;
}
int main(int argc,char** argv){
    try{
        auto p=weights();
        if(argc==2&&std::string(argv[1])=="--dry-run"){dry(p);Controller c(p);load_candidate(c);return 0;}
        need(argc==2&&std::string(argv[1])=="--run-once","isolated finite evaluation only");return eval(p);
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
