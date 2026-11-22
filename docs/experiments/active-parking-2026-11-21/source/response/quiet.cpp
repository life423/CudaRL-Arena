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
constexpr int Population=8, Generations=3, TestTicks=160, TrainTicks=80, Cases=8, SwimTicks=3040;
constexpr int ticks=240, variants=9, Cap=2*variants*ticks;
constexpr int Features=4, Params=Outputs*Features;
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

static int run(){
    const fs::path out="build/slow-stroke-response-v1/recorded-run";
    need(fs::create_directory(out),"run output already exists");
    Budget b;std::ofstream csv(out/"traces.csv"),overview(out/"summary.csv");
    need(csv.good()&&overview.good(),"trace write open");
    csv<<std::setprecision(10)<<"pose,variant,control,signed_mm,radial_mm,velocity_forward_mm_s,speed_mm_s,tilt_deg,max_angular_rad_s,safe,j0,j1,j2,j3\n";
    overview<<std::setprecision(10)<<"pose,variant,amp_rad,frequency_hz,lag_sign,x_2s_mm,x_4s_mm,x_6s_mm,x_12s_mm,v_6s_mm_s,tilt_max_deg,all_safe,quiet_final_s\n";
    auto status=[&](const std::string& result){std::ofstream f(out/"status.json");f<<std::setprecision(10)<<"{\"result\":"<<std::quoted(result)<<",\"cases\":"<<b.trial<<",\"controls\":"<<b.controls<<",\"substeps\":"<<b.substeps<<",\"seconds\":"<<b.seconds()<<"}\n";};
    try{
        status("running");
        for(int pose:{0,3})for(int k=0;k<variants;++k){
            const float A=k?(k<=4?.25f:.5f):0.f;
            const float freq=k&&((k-1)%4)<2?.25f:.5f;
            const float sign=k&&k%2==0?-1.f:1.f;
            m::World w;auto t=setup_front({pose,.002f},w);auto c=hold();c.maximum_controls=ticks;
            const auto h=t.request.heading;
            bool allsafe=true;float tilt=0, v6=0;std::array<float,4> dx{};int tail=0;m::Action a{};
            for(int i=0;i<=ticks;++i){
                const auto z=m::measure(t,w);const auto d=p::dot(z.hold_error_m,h)*1000.f;
                const auto v=p::dot(z.velocity_m_s,h)*1000.f;
                float angular=0;for(int j=3;j<10;++j)angular=std::max(angular,std::fabs(w.physical.body.velocity[j]));
                tilt=std::max(tilt,z.tilt_deg);allsafe &= safe(z);
                if(i>0)tail=safe(z)&&in_pad(z)&&angular<=.035f?tail+1:0;
                if(i==40)dx[0]=d;if(i==80)dx[1]=d;if(i==120){dx[2]=d;v6=v;}if(i==ticks)dx[3]=d;
                csv<<pose<<','<<k<<','<<i<<','<<d<<','<<std::hypot(z.hold_error_m.x,z.hold_error_m.z)*1000.f<<','<<v<<','<<p::norm(z.velocity_m_s)*1000.f<<','<<z.tilt_deg<<','<<angular<<','<<safe(z);
                for(float u:a.command)csv<<','<<u;csv<<'\n';
                if(i==ticks)break;
                const float time=i*p::dt,pi=cudarl::movement::pi;
                const float taper=time<1.f?.5f-.5f*std::cos(pi*time):time<5.f?1.f:time<6.f?.5f+.5f*std::cos(pi*(time-5.f)):0.f;
                a.command[0]=A/p::limit*taper*std::sin(2.f*pi*freq*time);
                a.command[2]=A/p::limit*taper*std::sin(2.f*pi*freq*time-sign*pi/2.f);
                b.step(w,a);const auto row=m::record_hold_control(t,w,c);
                need(row.accepted&&row.done==(i+1==ticks),"diagnostic transition rejected");
            }
            overview<<pose<<','<<k<<','<<A<<','<<freq<<','<<sign;for(float v:dx)overview<<','<<v;
            overview<<','<<v6<<','<<tilt<<','<<allsafe<<','<<tail*.05f<<'\n';++b.trial;
        }
        overview.flush();csv.flush();need(overview.good()&&csv.good()&&b.controls==Cap&&b.substeps==10*Cap,"trace/cap failure");
        status("closed_diagnostic_only");
    }catch(const std::exception& e){status(std::string("fault: ")+e.what());throw;}
    return 0;
}
int main(int argc,char** argv){
    try{
        static_assert(Cap==4320);
        if(argc==2&&std::string(argv[1])=="--dry-run"){
            auto x=weights();m::World w;auto t=setup_front({0,.002f},w);
            need(w.attempted_substeps==0 && p::norm(m::measure(t,w).velocity_m_s)<.000501f,"bad dry start");
            std::cout<<"DRY PASS, zero physical controls\n";return 0;
        }
        need(argc==2&&std::string(argv[1])=="--run-once","diagnostic flag required");return run();
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
