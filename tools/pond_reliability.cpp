#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include "demos/pond_reliability.h"
#include "demos/pond_current_schedule.h"
#include "demos/pond_bank_monitor.h"
namespace dm=cudarl::demos;
int main(int argc,char** argv) {
    try {
        std::string path,output,module,backend="cuda";bool progress=false,food_drift=false;int minutes=30,switch_seconds=-1;std::vector<uint32_t> seeds;float current_cm_s=0;double rotation_seconds=0;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];if(arg=="--progress") {progress=true;continue;}
            if(arg=="--drifting-food") {food_drift=true;continue;}
            if(arg=="--help") {std::puts("pond_reliability --policy FILE [--progress] --output NEW_DIR --backend cpu|cuda [--module DLL] [--minutes 30] [--seed N (repeatable)] [--current-cm-s 0.5] [--switch-seconds 60] [--drifting-food] [--rotation-seconds 120|-120]\nActual app playback, frozen saved weights. Optional signed X current, at most 1 cm/s. Rotation preserves speed and uses simulation time; positive turns +X toward +Z, negative toward -Z. Rotation and delayed switching are mutually exclusive. Reports immobility, food gaps, sampled bank visits and diagnostic traces. No training or rescue. Quality failures remain in summary/cases; infrastructure errors return nonzero.");return 0;}
            if(i+1>=argc) throw std::invalid_argument("Missing reliability argument");const std::string value=argv[++i];
            if(arg=="--policy") path=value;else if(arg=="--output") output=value;else if(arg=="--module") module=value;else if(arg=="--backend") backend=value;
            else if(arg=="--minutes") minutes=std::stoi(value);else if(arg=="--seed") seeds.push_back(static_cast<uint32_t>(std::stoul(value,nullptr,0)));
            else if(arg=="--current-cm-s") current_cm_s=std::stof(value);else if(arg=="--switch-seconds") switch_seconds=std::stoi(value);
            else if(arg=="--rotation-seconds") rotation_seconds=std::stod(value);else throw std::invalid_argument("Unknown reliability argument");
        }
        if(path.empty() || output.empty() || std::filesystem::exists(output) || (backend!="cpu" && backend!="cuda") || minutes<1 || minutes>60 || seeds.size()>16) throw std::invalid_argument("Use a policy, new output directory, strict CPU/CUDA backend, 1..60 minutes and at most 16 seeds");
        const cudarl::deep::V current{current_cm_s*.01f,0,0};dm::validate_pond_current(current);
        const dm::PondCurrentSchedule schedule(current,rotation_seconds);
        if(switch_seconds < -1 || switch_seconds>=minutes*60) throw std::invalid_argument("Current switch must occur within the trial");
        if(rotation_seconds!=0 && switch_seconds!=-1) throw std::invalid_argument("Choose continuous rotation or a delayed fixed-current switch");
        if(seeds.empty()) seeds={7,0xC0DE7101u,0xC0DE7102u,0xC0DE7103u};
        const auto original=cudarl::checkpoint::read(path);
        cudarl::rl::SelectorPolicy legacy;cudarl::rl::ProgressPolicy candidate;
        if(progress) candidate=cudarl::rl::ProgressPolicy::load(path);else legacy=cudarl::rl::SelectorPolicy::load(path);
        std::filesystem::create_directories(output);std::ofstream csv(std::filesystem::path(output)/"cases.csv"),events(std::filesystem::path(output)/"events.csv");
        std::ofstream bank_events(std::filesystem::path(output)/"bank-events.csv"),trace(std::filesystem::path(output)/"trace.csv");
        csv<<std::setprecision(10);bank_events<<std::setprecision(10);trace<<std::setprecision(10);
        csv<<"seed,environment,backend,minutes,foods,food_per_minute,max_food_gap_s,immobility_events,max_immobility_s,invalid_states,invalid_foods,nonsequential_steps,passes,current_x_cm_s,switch_seconds,drifting_food,rotation_seconds,bank_visits,sustained_bank_visits,sustained_bank_cleared,sustained_bank_open,bank_contact_s,max_bank_contact_s,sustained_bank_food20_eligible,sustained_bank_food20,bank_food_censored,bank_slow_contact_s,max_bank_clear_s\n";
        bank_events<<"seed,start_s,last_observed_s,clear_s,contact_s,slow_contact_s,first_food_after_entry_s,sustained,clear_observed,food_followup_s\n";
        trace<<"seed,time_s,current_x_m_s,current_z_m_s,foods,since_food_s,speed_cm_s,near_bank,target,x,y,z,qx,qy,qz,qw,food0_x,food0_y,food0_z,food1_x,food1_y,food1_z,food2_x,food2_y,food2_z,angle0,angle1,angle2,angle3,action0,action1,action2,action3\n";
        events<<"seed,start_s,end_s,duration_s\n";bool passes=true;int total_events=0;
        for(uint32_t seed:seeds) {
            auto player=progress?dm::create_pond_playback(candidate,backend,module,seed):dm::create_pond_playback(legacy,backend,module,seed);
            if(std::string(player->backend())!=(backend=="cpu"?"CPU":"CUDA")) throw std::runtime_error("Requested backend was not selected");
            player->set_food_drift(food_drift);
            if(switch_seconds<=0) player->set_current(schedule.at_step(0));
            dm::PondReliability monitor;monitor.observe(player->frame());dm::PondBankMonitor bank;bank.observe(player->frame().state);int snapshot=0;
            for(int tick=1;tick<=minutes*1200;++tick) {
                if(switch_seconds>0 && tick==switch_seconds*20+1) player->set_current(current);
                if(rotation_seconds!=0) player->set_current(schedule.at_step(tick-1));
                const auto& frame=player->step();
                const bool near=bank.observe(frame.state);
                if(tick%100==0 || (tick%20==0 && (near || frame.state.since_food>=200))) {
                    const auto& state=frame.state;const auto& p=state.body.position;const auto& q=state.body.orientation;
                    const float speed=cudarl::deep::norm({state.body.velocity[0],state.body.velocity[1],state.body.velocity[2]});
                    trace<<seed<<','<<tick*.05<<','<<frame.current.x<<','<<frame.current.z<<','<<state.eaten<<','<<state.since_food*.05<<','<<speed*100<<','<<near<<','<<frame.selected<<','<<p.x<<','<<p.y<<','<<p.z<<','<<q.x<<','<<q.y<<','<<q.z<<','<<q.w;
                    for(auto food:state.food) trace<<','<<food.x<<','<<food.y<<','<<food.z;
                    for(float angle:state.body.angle) trace<<','<<angle;
                    for(float action:state.last_action) trace<<','<<action;trace<<'\n';
                }
                if(monitor.observe(frame)) {
                    cudarl::checkpoint::Writer out;out.value(uint32_t{0x52535031});out.value(uint32_t{3});out.value(frame);
                    cudarl::checkpoint::write_atomic((std::filesystem::path(output)/("stall-"+std::to_string(seed)+"-"+std::to_string(snapshot++)+".bin")).string(),out.data);
                }
            }
            monitor.finish();const int foods=player->frame().state.eaten;passes=passes && monitor.passes();total_events+=monitor.confirmed_events();
            int sustained=0,cleared=0,food_eligible=0,food20=0,slow=0,max_contact=0,max_clear=0;
            for(const auto& visit:bank.visits) {
                slow+=visit.slow_steps;max_contact=std::max(max_contact,visit.contact_steps);
                const int food_delay=visit.first_food<0?-1:visit.first_food-visit.start;
                if(visit.contact_steps>=20) {
                    ++sustained;if(visit.clear>=0) {++cleared;max_clear=std::max(max_clear,visit.clear-visit.start);}
                    const bool success=food_delay>=0 && food_delay<=400;
                    if(success || minutes*1200-visit.start>=400) {++food_eligible;if(success) ++food20;}
                }
                bank_events<<seed<<','<<visit.start*.05<<','<<visit.last*.05<<','<<(visit.clear<0?-1:visit.clear*.05)<<','<<visit.contact_steps*.05<<','<<visit.slow_steps*.05<<','<<(food_delay<0?-1:food_delay*.05)<<','<<(visit.contact_steps>=20)<<','<<(visit.clear>=0)<<','<<(minutes*1200-visit.start)*.05<<'\n';
            }
            csv<<seed<<",0,"<<player->backend()<<','<<minutes<<','<<foods<<','<<double(foods)/minutes<<','<<monitor.max_food_gap*.05f<<','<<monitor.confirmed_events()<<','<<monitor.max_immobility_seconds<<','<<monitor.invalid_states<<','<<monitor.invalid_foods<<','<<monitor.nonsequential_steps<<','<<(monitor.passes()?1:0)<<','<<current_cm_s<<','<<switch_seconds<<','<<food_drift<<','<<rotation_seconds<<','<<bank.visits.size()<<','<<sustained<<','<<cleared<<','<<(sustained-cleared)<<','<<bank.contact_steps*.05<<','<<max_contact*.05<<','<<food_eligible<<','<<food20<<','<<(sustained-food_eligible)<<','<<slow*.05<<','<<max_clear*.05<<'\n';csv.flush();
            for(const auto& e:monitor.events) events<<seed<<','<<e.start*.05f<<','<<e.end*.05f<<','<<(e.end-e.start)*.05f<<'\n';
            std::printf("RELIABILITY kind=%s backend=%s seed=%u minutes=%d rotation=%.0f foods=%d immobility_events=%d food_gap=%.2f sustained_bank=%d passes=%s\n",progress?"progress":"legacy",player->backend(),seed,minutes,rotation_seconds,foods,monitor.confirmed_events(),monitor.max_food_gap*.05f,sustained,monitor.passes()?"true":"false");std::fflush(stdout);
        }
        if(original!=cudarl::checkpoint::read(path)) throw std::runtime_error("Reliability check changed source policy");
        std::ofstream summary(std::filesystem::path(output)/"summary.json");summary<<"{\"policy_kind\":\""<<(progress?"progress":"legacy")<<"\",\"backend\":\""<<backend<<"\",\"cases\":"<<seeds.size()<<",\"minutes_per_case\":"<<minutes<<",\"immobility_events\":"<<total_events<<",\"passes_provisional_gate\":"<<(passes?"true":"false")<<",\"source_unchanged\":true,\"training\":false,\"current_m_s\":"<<current_cm_s*.01f<<",\"switch_seconds\":"<<switch_seconds<<"}\n";
        std::ofstream mode(std::filesystem::path(output)/"food-mode.json");mode<<"{\"drifting_food\":"<<(food_drift?"true":"false")<<",\"boundary\":\"no-flux feeding clearance envelope\"}\n";
        std::ofstream rotation(std::filesystem::path(output)/"current-mode.json");rotation<<"{\"initial_m_s\":["<<current.x<<",0,"<<current.z<<"],\"signed_rotation_seconds\":"<<rotation_seconds<<",\"clock\":\"completed physics steps\",\"bank_proxy\":\"prior bank-study sampled gap below 1 mm, radius squared above 0.35\"}\n";
        if(!csv || !events || !summary || !mode || !rotation || !bank_events || !trace) throw std::runtime_error("Failed writing reliability report");return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Pond reliability: %s\n",e.what());return 1;}
}
