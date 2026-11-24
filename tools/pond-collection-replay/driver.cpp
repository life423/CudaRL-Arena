#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#include "data.h"
#include <atomic>
#include "core/checkpoint.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace fs=std::filesystem;
namespace cp=cudarl::checkpoint;
using namespace pond_collection;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void require(bool yes,const std::string& why){if(!yes)throw std::runtime_error(why);}
std::string digest(const fs::path& path){
    std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"Missing input: "+path.string());
    const auto bytes=static_cast<size_t>(f.tellg());require(bytes<=64*1024*1024,"Oversize saved rollout");
    std::vector<unsigned char> data(bytes);f.seekg(0);f.read(reinterpret_cast<char*>(data.data()),bytes);require(bool(f),"Incomplete input");
    BCRYPT_ALG_HANDLE provider=nullptr;require(BCryptOpenAlgorithmProvider(&provider,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"SHA provider");
    std::array<unsigned char,32> hash{};
    const auto rc=BCryptHash(provider,nullptr,0,data.data(),static_cast<ULONG>(data.size()),hash.data(),32);
    BCryptCloseAlgorithmProvider(provider,0);require(rc>=0,"SHA-256");
    std::ostringstream s;s<<std::hex<<std::setfill('0');for(auto v:hash)s<<std::setw(2)<<unsigned(v);return s.str();
}

namespace {
constexpr int N=512,rows=N*64;
constexpr char batch_sha[]="5994cec04228de4a50c71ad782149194804905d33146356b4232d1922cb6b8da";
constexpr char package_sha[]="4b6400c4db65873bd346c3e0c4fed384e3a77f9e5174d0dcea5d74073b94616d";
std::atomic<int> issued{0},completed{0};
struct Deadline {
    fs::path file;std::mutex mutex;std::condition_variable wake;bool done=false;std::thread watch;
    Deadline(const fs::path& out,Clock::time_point start):file(out/"timeout.json"),watch([this,start]{
        std::unique_lock<std::mutex> l(mutex);
        if(!wake.wait_until(l,start+std::chrono::seconds(4),[this]{return done;})){
            std::ofstream f(file);f<<"{\"passed\":false,\"reason\":\"four_second_wall_cap\",\"optimizer_calls\":0,\"issued_controls_upper_bound\":"
              <<issued.load()*N<<",\"completed_controls\":"<<completed.load()*N<<",\"completed_substeps\":"<<completed.load()*N*10<<"}\n";f.close();std::_Exit(3);
        }
    }){}
    ~Deadline(){{std::lock_guard<std::mutex> l(mutex);done=true;}wake.notify_one();watch.join();}
};
template<class T>std::vector<T> array(cp::Reader& r,size_t n){auto v=r.array<T>(n);require(v.size()==n,"Frozen batch shape mismatch");return v;}
void read(const fs::path& root,Inputs& inputs,Samples& ref){
    const auto p=root/"build/turn38-residual/updates-9-16/batch-12.bin";const auto policy=root/"build/turn34/inputs/original-amplitude-package.bin";
    require(digest(p)==batch_sha&&digest(policy)==package_sha,"Frozen input SHA mismatch");inputs.package=policy.string();
    const auto bytes=cp::read(p.string());cp::Reader r{bytes};require(r.value<uint32_t>()==0x31425252&&r.value<uint32_t>()==1&&r.value<int>()==12&&r.value<int>()==64,"Saved recording ABI");
    inputs.worlds=array<uint8_t>(r,N*1384);inputs.tasks=array<uint8_t>(r,N*368);inputs.residual=array<float>(r,722);inputs.critic=array<float>(r,2433);
    ref.obs=array<float>(r,rows*42);ref.residual=array<float>(r,rows*2);ref.mask=array<float>(r,rows);ref.base=array<float>(r,rows*4);ref.latent=array<float>(r,rows*4);
    ref.logp=array<float>(r,rows);ref.value=array<float>(r,rows);ref.reward=array<float>(r,rows);ref.done=array<float>(r,rows);ref.truncated=array<float>(r,rows);
    ref.final_value=array<float>(r,rows);ref.last_value=array<float>(r,N);ref.attempted=array<uint64_t>(r,rows);ref.substeps=array<uint64_t>(r,rows);
    ref.reason=array<int>(r,rows);require(r.value<uint32_t>()==sizeof(Trace),"Trace ABI mismatch");const auto trace=array<uint8_t>(r,rows*sizeof(Trace));
    ref.trace.resize(rows);std::memcpy(ref.trace.data(),trace.data(),trace.size());r.finish();
}
struct Delta {
    std::string first;uint64_t failing=0,changed=0;
    double max_obs=0,max_action=0,max_logp=0,max_value=0,max_reward=0,max_motion=0,max_speed=0,max_precise_gap=0,max_excess=0;
    void check(bool ok,const char* why){if(!ok){++failing;if(first.empty())first=why;}}
    void value(double a,double b,double limit,double& peak,const char* label){const auto e=std::abs(a-b);peak=std::max(peak,e);changed+=a!=b;check(std::isfinite(a)&&std::isfinite(b)&&e<=limit,label);}
    void vector(const std::vector<float>& a,const std::vector<float>& b,double limit,double& peak,const char* name){
        require(a.size()==b.size(),name);if(std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0)return;
        for(size_t i=0;i<a.size();++i)value(a[i],b[i],limit,peak,name);
    }
    void stats(const Trace& a,const Trace& b){
        value(a.x,b.x,2e-5,max_motion,"com_x");value(a.z,b.z,2e-5,max_motion,"com_z");
        value(a.vx,b.vx,2e-7,max_speed,"com_vx");value(a.vy,b.vy,2e-7,max_speed,"com_vy");value(a.vz,b.vz,2e-7,max_speed,"com_vz");
        value(a.depth,b.depth,2e-5,max_motion,"depth");value(a.tilt,b.tilt,2e-5,max_motion,"tilt");
        check(a.episode==b.episode&&a.control==b.control&&a.kind==b.kind&&a.occupied_cells==b.occupied_cells&&
              std::equal(a.occupied_grid,a.occupied_grid+8,b.occupied_grid),"episode_or_coverage");
        check(a.bank_near_seconds==b.bank_near_seconds&&a.contact_seconds==b.contact_seconds&&a.postwarmup_stall_seconds==b.postwarmup_stall_seconds,"bank_contact_stall");
        check(a.heading_x==b.heading_x&&a.heading_z==b.heading_z&&a.requested_angle==b.requested_angle&&a.request_serial==b.request_serial&&
              a.onset_control==b.onset_control&&a.request_age==b.request_age&&a.qualifying_streak==b.qualifying_streak&&a.events==b.events&&
              a.active==b.active&&a.succeeded==b.succeeded&&a.long_hold==b.long_hold&&a.onsets==b.onsets&&a.successes==b.successes&&
              a.timeouts==b.timeouts&&a.interrupted==b.interrupted&&a.early_terminal==b.early_terminal&&a.long_holds==b.long_holds,"request_events");
    }
};
}
int main(int argc,char** argv){
    const auto started=Clock::now();fs::path out;bool owns=false;
    try{
        const std::string binding=std::string("{\"protocol\":\"pond-collection-replay-v1\",\"source_sha256\":\"")+source()
          +"\",\"batch_sha256\":\""+batch_sha+"\",\"original_package_sha256\":\""+package_sha+"\",\"worlds\":512,\"ticks\":64,\"max_controls\":32768,\"max_substeps\":327680,\"max_wall_seconds\":4,\"optimizer_calls\":0}";
        if(argc==2&&std::string(argv[1])=="--describe"){std::cout<<binding<<'\n';return 0;}
        const bool pure=argc==3&&std::string(argv[1])=="--preflight";
        require(pure||(argc==5&&std::string(argv[1])=="--run"),"Expected --describe, --preflight <root>, or --run <root> <seal> <SHA256>");
        const auto root=fs::canonical(argv[2]);out=root/"assets/research-evidence/pond-collection-replay-v1";
        std::unique_ptr<Deadline> deadline;
        if(!pure){
            require(digest(argv[3])==argv[4]&&fs::file_size(argv[3])>80,"Seal mismatch");
            require(!fs::exists(out)&&fs::create_directories(out),"The four-second replay has already been attempted");owns=true;deadline=std::make_unique<Deadline>(out,started);
        }
        Inputs inputs;Samples ref;read(root,inputs,ref);const double read_ms=ms(started);
        if(pure){std::cout<<"{\"archive_rows\":32768,\"physical_controls\":0,\"optimizer_calls\":0}\n";return 0;}
        Samples d;Timing time;std::vector<Gap> gaps;
        collect_once(inputs,d,gaps,time,[](int n){issued=n;},[](int n){completed=n;});
        const auto cpu_begin=Clock::now();Delta m;
        m.vector(d.obs,ref.obs,2e-5,m.max_obs,"policy_observations");m.vector(d.residual,ref.residual,2e-5,m.max_action,"residual_actions");
        m.vector(d.base,ref.base,2e-5,m.max_action,"frozen_base_actions");m.vector(d.latent,ref.latent,2e-5,m.max_action,"combined_actions");
        m.vector(d.logp,ref.logp,1e-4,m.max_logp,"logp");m.vector(d.value,ref.value,2e-5,m.max_value,"values");
        m.vector(d.reward,ref.reward,2e-5,m.max_reward,"rewards");m.vector(d.final_value,ref.final_value,2e-5,m.max_value,"final_values");
        m.vector(d.last_value,ref.last_value,2e-5,m.max_value,"bootstrap_values");
        m.check(d.mask==ref.mask&&d.done==ref.done&&d.truncated==ref.truncated&&d.reason==ref.reason&&d.attempted==ref.attempted&&d.substeps==ref.substeps,"exact_flags_and_counts");
        require(d.trace.size()==rows&&gaps.size()==rows,"Incomplete physical trace");
        uint64_t controls=0,substeps=0,free=0,exact=0,queries=0,precise=0,terminal=0,truncated=0;
        for(int i=0;i<rows;++i){
            const auto& c=gaps[i];const double old=ref.trace[i].minimum_gap,new_min=d.trace[i].minimum_gap;
            controls+=d.attempted[i];substeps+=d.substeps[i];free+=c.free_steps;exact+=c.exact_steps;queries+=c.full_queries;precise+=c.precise;
            terminal+=d.done[i]==1;truncated+=d.truncated[i]==1;m.stats(d.trace[i],ref.trace[i]);
            m.check(std::isfinite(c.bound)&&std::isfinite(old)&&c.free_steps+c.exact_steps==10,"invalid_classification");
            if(c.precise)m.value(old,new_min,1e-5,m.max_precise_gap,"exact_minimum");
            else{m.check(std::isnan(new_min)&&c.bound>.01&&old>.01,"swept_threshold_class");m.max_excess=std::max(m.max_excess,c.bound-old);
                m.check(c.bound<=old+1e-5,"swept_lower_bound");}
        }
        m.check(controls==rows&&substeps==rows*10&&issued==64&&completed==64,"physical_budget");
        const double cpu_ms=ms(cpu_begin);const auto evidence_start=Clock::now();
        // Store the validity tag and bound for every row; measured numeric minima
        // are intentionally NaN for threshold-only results. The source and fixed
        // archive recover all other physical, actuator, reward and request fields.
        cp::Writer w;w.value(uint32_t{0x31435053});w.value(uint32_t{1});w.value(uint32_t{rows});
        for(int i=0;i<rows;++i){w.value(gaps[i].precise);w.value(gaps[i].bound);w.value(d.trace[i].minimum_gap);w.value(gaps[i].free_steps);w.value(gaps[i].full_queries);}
        cp::write_atomic((out/"clearance-tags.bin").string(),w.data);
        const double evidence_ms=ms(evidence_start);const bool passed=m.failing==0;
        std::ofstream o(out/"result.json");o<<std::setprecision(17)<<"{\"passed\":"<<(passed?"true":"false")<<",\"seal_sha256\":\""<<argv[4]<<"\",\"bindings\":"<<binding
          <<",\"completed_updates\":0,\"physical_controls\":"<<controls<<",\"physical_substeps\":"<<substeps<<",\"free_substeps\":"<<free<<",\"exact_substeps\":"<<exact
          <<",\"exact_queries\":"<<queries<<",\"precise_rows\":"<<precise<<",\"terminal_transitions\":"<<terminal<<",\"time_limit_truncations\":"<<truncated
          <<",\"failed_comparisons\":"<<m.failing<<",\"first_failure\":\""<<m.first<<"\",\"changed_numeric_values\":"<<m.changed<<",\"max_observation_delta\":"<<m.max_obs
          <<",\"max_action_delta\":"<<m.max_action<<",\"max_logp_delta\":"<<m.max_logp<<",\"max_reward_delta\":"<<m.max_reward<<",\"max_value_delta\":"<<m.max_value
          <<",\"max_motion_delta\":"<<m.max_motion<<",\"max_velocity_delta_m_s\":"<<m.max_speed<<",\"max_precise_gap_delta_m\":"<<m.max_precise_gap<<",\"max_swept_bound_minus_reference_m\":"<<m.max_excess
          <<",\"input_verification_ms\":"<<read_ms<<",\"setup_ms\":"<<time.setup_ms<<",\"upload_wall_ms\":"<<time.input_transfer_ms<<",\"gpu_physics_and_task_ms\":"<<time.physics_ms
          <<",\"gpu_inference_sampling_composition_ms\":"<<time.inference_ms<<",\"collection_wall_ms\":"<<time.collection_wall_ms<<",\"collection_transitions_per_second\":"<<1000*controls/time.collection_wall_ms
          <<",\"readback_wall_ms\":"<<time.readback_ms<<",\"cpu_comparison_ms\":"<<cpu_ms<<",\"typed_evidence_write_ms\":"<<evidence_ms<<",\"wall_ms\":"<<ms(started)<<"}\n";
        o.close();require(bool(o),"Could not close result file");return passed?0:2;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';if(owns){std::ofstream f(out/"error.json");f<<"{\"passed\":false,\"issued_controls_upper_bound\":"<<N*issued<<",\"completed_controls\":"<<N*completed<<",\"completed_substeps\":"<<N*completed*10<<"}\n";std::ofstream(out/"error.txt")<<e.what();}return 2;}
}
