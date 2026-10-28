#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include "demos/pond_reliability.h"
namespace dm=cudarl::demos;
int main(int argc,char** argv) {
    try {
        std::string path,output,module,backend="cuda";bool progress=false,food_drift=false;int minutes=30,switch_seconds=-1;std::vector<uint32_t> seeds;float current_cm_s=0;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];if(arg=="--progress") {progress=true;continue;}
            if(arg=="--drifting-food") {food_drift=true;continue;}
            if(arg=="--help") {std::puts("pond_reliability --policy FILE [--progress] --output NEW_DIR --backend cpu|cuda [--module DLL] [--minutes 30] [--seed N (repeatable)] [--current-cm-s 0.5] [--switch-seconds 60] [--drifting-food]\nActual app playback, frozen saved weights. Optional signed X current, at most 1 cm/s. Default current starts immediately; --switch-seconds changes it during continuous playback. Reports 3s small-motion windows and fails a provisional gate at 5s. No training or rescue. A failed quality gate is reported in summary.json; infrastructure errors return nonzero.");return 0;}
            if(i+1>=argc) throw std::invalid_argument("Missing reliability argument");const std::string value=argv[++i];
            if(arg=="--policy") path=value;else if(arg=="--output") output=value;else if(arg=="--module") module=value;else if(arg=="--backend") backend=value;
            else if(arg=="--minutes") minutes=std::stoi(value);else if(arg=="--seed") seeds.push_back(static_cast<uint32_t>(std::stoul(value,nullptr,0)));
            else if(arg=="--current-cm-s") current_cm_s=std::stof(value);else if(arg=="--switch-seconds") switch_seconds=std::stoi(value);else throw std::invalid_argument("Unknown reliability argument");
        }
        if(path.empty() || output.empty() || std::filesystem::exists(output) || (backend!="cpu" && backend!="cuda") || minutes<1 || minutes>60 || seeds.size()>16) throw std::invalid_argument("Use a policy, new output directory, strict CPU/CUDA backend, 1..60 minutes and at most 16 seeds");
        const cudarl::deep::V current{current_cm_s*.01f,0,0};dm::validate_pond_current(current);
        if(switch_seconds < -1 || switch_seconds>=minutes*60) throw std::invalid_argument("Current switch must occur within the trial");
        if(seeds.empty()) seeds={7,0xC0DE7101u,0xC0DE7102u,0xC0DE7103u};
        const auto original=cudarl::checkpoint::read(path);
        cudarl::rl::SelectorPolicy legacy;cudarl::rl::ProgressPolicy candidate;
        if(progress) candidate=cudarl::rl::ProgressPolicy::load(path);else legacy=cudarl::rl::SelectorPolicy::load(path);
        std::filesystem::create_directories(output);std::ofstream csv(std::filesystem::path(output)/"cases.csv"),events(std::filesystem::path(output)/"events.csv");
        csv<<"seed,environment,backend,minutes,foods,food_per_minute,max_food_gap_s,immobility_events,max_immobility_s,invalid_states,invalid_foods,nonsequential_steps,passes,current_x_cm_s,switch_seconds,drifting_food\n";
        events<<"seed,start_s,end_s,duration_s\n";bool passes=true;int total_events=0;
        for(uint32_t seed:seeds) {
            auto player=progress?dm::create_pond_playback(candidate,backend,module,seed):dm::create_pond_playback(legacy,backend,module,seed);
            if(std::string(player->backend())!=(backend=="cpu"?"CPU":"CUDA")) throw std::runtime_error("Requested backend was not selected");
            player->set_food_drift(food_drift);
            if(switch_seconds<=0) player->set_current(current);
            dm::PondReliability monitor;monitor.observe(player->frame());int snapshot=0;
            for(int tick=1;tick<=minutes*1200;++tick) {
                if(switch_seconds>0 && tick==switch_seconds*20+1) player->set_current(current);
                const auto& frame=player->step();
                if(monitor.observe(frame)) {
                    cudarl::checkpoint::Writer out;out.value(uint32_t{0x52535031});out.value(uint32_t{3});out.value(frame);
                    cudarl::checkpoint::write_atomic((std::filesystem::path(output)/("stall-"+std::to_string(seed)+"-"+std::to_string(snapshot++)+".bin")).string(),out.data);
                }
            }
            monitor.finish();const int foods=player->frame().state.eaten;passes=passes && monitor.passes();total_events+=monitor.confirmed_events();
            csv<<seed<<",0,"<<player->backend()<<','<<minutes<<','<<foods<<','<<double(foods)/minutes<<','<<monitor.max_food_gap*.05f<<','<<monitor.confirmed_events()<<','<<monitor.max_immobility_seconds<<','<<monitor.invalid_states<<','<<monitor.invalid_foods<<','<<monitor.nonsequential_steps<<','<<(monitor.passes()?1:0)<<','<<current_cm_s<<','<<switch_seconds<<','<<food_drift<<'\n';csv.flush();
            for(const auto& e:monitor.events) events<<seed<<','<<e.start*.05f<<','<<e.end*.05f<<','<<(e.end-e.start)*.05f<<'\n';
            std::printf("RELIABILITY kind=%s backend=%s seed=%u minutes=%d foods=%d immobility_events=%d max_immobility=%.2f food_gap=%.2f passes=%s\n",progress?"progress":"legacy",player->backend(),seed,minutes,foods,monitor.confirmed_events(),monitor.max_immobility_seconds,monitor.max_food_gap*.05f,monitor.passes()?"true":"false");std::fflush(stdout);
        }
        if(original!=cudarl::checkpoint::read(path)) throw std::runtime_error("Reliability check changed source policy");
        std::ofstream summary(std::filesystem::path(output)/"summary.json");summary<<"{\"policy_kind\":\""<<(progress?"progress":"legacy")<<"\",\"backend\":\""<<backend<<"\",\"cases\":"<<seeds.size()<<",\"minutes_per_case\":"<<minutes<<",\"immobility_events\":"<<total_events<<",\"passes_provisional_gate\":"<<(passes?"true":"false")<<",\"source_unchanged\":true,\"training\":false,\"current_m_s\":"<<current_cm_s*.01f<<",\"switch_seconds\":"<<switch_seconds<<"}\n";
        std::ofstream mode(std::filesystem::path(output)/"food-mode.json");mode<<"{\"drifting_food\":"<<(food_drift?"true":"false")<<",\"boundary\":\"no-flux feeding clearance envelope\"}\n";
        if(!csv || !events || !summary || !mode) throw std::runtime_error("Failed writing reliability report");return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Pond reliability: %s\n",e.what());return 1;}
}
