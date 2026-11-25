#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "data.h"
#include "../pond-ppo/admission.h"
#include "collection_replay_identity.h"
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
#include <iterator>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace fs=std::filesystem;
namespace cp=cudarl::checkpoint;
namespace ph=cudarl::rl::pond_ppo_host;
using namespace pond_collection;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void require(bool yes,const std::string& why){if(!yes)throw std::runtime_error(why);}
std::vector<uint8_t> read_bytes(const fs::path& path, size_t limit){
    std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"Missing input: "+path.string());
    const auto length=f.tellg();require(length>=0&&static_cast<uint64_t>(length)<=limit,"Invalid input length");
    std::vector<uint8_t> data(static_cast<size_t>(length));
    f.seekg(0);f.read(reinterpret_cast<char*>(data.data()),data.size());require(bool(f),"Incomplete input");return data;
}

namespace {
constexpr int N=512,rows=N*64;
#if defined(POND_COLLECTION_FOUR_OUTPUT)
#if defined(POND_POSTURE_LESSON)
constexpr char protocol[]="pond-residual-four-posture-replay-v1";
#else
constexpr char protocol[]="pond-residual-four-collection-replay-v1";
#endif
constexpr char batch_path[]="build/turn39-residual-four/updates-9-16/batch-12.bin";
constexpr char batch_sha[]="c4ea4ba3196142013ec7d322214047136ec76264c78a3d6989df7873b4d50457";
constexpr int archive_version=2,residual_parameters=756,residual_outputs=4;
#else
constexpr char protocol[]="pond-collection-replay-v3";
constexpr char batch_path[]="build/turn38-residual/updates-9-16/batch-12.bin";
constexpr char batch_sha[]="5994cec04228de4a50c71ad782149194804905d33146356b4232d1922cb6b8da";
constexpr int archive_version=1,residual_parameters=722,residual_outputs=2;
#endif
constexpr char package_sha[]="4b6400c4db65873bd346c3e0c4fed384e3a77f9e5174d0dcea5d74073b94616d";
std::atomic<int> issued{0},completed{0};
constexpr const char* phases[]={"seal_verification","archive_read","archive_verification","archive_decode",
    "motor_load","cuda_context","collection_buffers","inference_networks","input_upload",
    "first_control_preparation","observation_kernel_load","physics_kernel_load","reset_kernel_load","initialize_kernel_load",
    "physics_collection","readback","cuda_cleanup","compare","evidence_write"};
Clock::time_point process_start;
std::array<std::atomic<int64_t>,std::size(phases)> phase_start{};
void mark(const char* phase){
    for(size_t i=0;i<std::size(phases);++i)if(std::strcmp(phase,phases[i])==0){
        int64_t empty=0;
        phase_start[i].compare_exchange_strong(empty,1+std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-process_start).count());
        return;
    }
    throw std::runtime_error("Unknown replay phase");
}
void write_phases(std::ostream& out){
    out<<",\"phase_started_ms\":{";
    bool first=true;
    for(size_t i=0;i<std::size(phases);++i){
        const auto stamp=phase_start[i].load();
        if(stamp==0)continue;
        if(!first)out<<',';
        out<<'"'<<phases[i]<<"\":"<<double(stamp-1)/1000;
        first=false;
    }
    out<<'}';
}
struct Deadline {
    fs::path file;std::mutex mutex;std::condition_variable wake;bool done=false;std::thread watch;
    Deadline(const fs::path& out,Clock::time_point start):file(out/"timeout.json"),watch([this,start]{
        std::unique_lock<std::mutex> l(mutex);
        if(!wake.wait_until(l,start+std::chrono::seconds(4),[this]{return done;})){
            std::ofstream f(file);f<<"{\"passed\":false,\"reason\":\"four_second_wall_cap\",\"optimizer_calls\":0,\"issued_controls_upper_bound\":"
              <<issued.load()*N<<",\"completed_controls\":"<<completed.load()*N<<",\"completed_substeps\":"<<completed.load()*N*10;
            write_phases(f);f<<"}\n";f.close();std::_Exit(3);
        }
    }){}
    ~Deadline(){{std::lock_guard<std::mutex> l(mutex);done=true;}wake.notify_one();watch.join();}
};
template<class T>std::vector<T> array(cp::Reader& r,size_t n){auto v=r.array<T>(n);require(v.size()==n,"Frozen batch shape mismatch");return v;}
void read(const fs::path& root,Inputs& inputs,Samples& ref){
    const auto p=root/batch_path;const auto policy=root/"build/turn34/inputs/original-amplitude-package.bin";
    mark("archive_read");
    auto bytes=read_bytes(p,64*1024*1024);
    mark("archive_verification");
    require(ph::sha256(bytes)==batch_sha&&ph::sha256(read_bytes(policy,1024*1024))==package_sha,"Frozen input SHA mismatch");
    inputs.package=policy.string();
    cp::Reader header{bytes};
    require(header.value<uint64_t>()==0x31504B434C524443ull,"Frozen checkpoint envelope");
    require(header.value<uint64_t>()==bytes.size()-24,"Frozen checkpoint length");
    const auto expected=header.value<uint64_t>();
    bytes.erase(bytes.begin(),bytes.begin()+24);
    require(cp::checksum(bytes)==expected,"Frozen checkpoint payload checksum");
    mark("archive_decode");
    cp::Reader r{bytes};require(r.value<uint32_t>()==0x31425252&&r.value<uint32_t>()==archive_version&&r.value<int>()==12&&r.value<int>()==64,"Saved recording ABI");
    inputs.worlds=array<uint8_t>(r,N*1384);inputs.tasks=array<uint8_t>(r,N*368);inputs.residual=array<float>(r,residual_parameters);inputs.critic=array<float>(r,2433);
    ref.obs=array<float>(r,rows*42);ref.residual=array<float>(r,rows*residual_outputs);ref.mask=array<float>(r,rows);ref.base=array<float>(r,rows*4);ref.latent=array<float>(r,rows*4);
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
    const auto started=Clock::now();process_start=started;fs::path out;bool owns=false;
    std::unique_ptr<Deadline> deadline;
    try{
        const std::string binding=std::string("{\"protocol\":\"")+protocol+"\",\"source_sha256\":\""+source()
          +"\",\"replay_source_sha256\":\""+collection_replay_source_sha256
          +"\",\"batch_sha256\":\""+batch_sha+"\",\"original_package_sha256\":\""+package_sha+"\",\"worlds\":512,\"ticks\":64,\"max_controls\":32768,\"max_substeps\":327680,\"max_wall_seconds\":4,\"optimizer_calls\":0}";
        if(argc==2&&std::string(argv[1])=="--describe"){std::cout<<binding<<'\n';return 0;}
        const bool pure=argc==3&&std::string(argv[1])=="--preflight";
        const bool module_only=argc==5&&std::string(argv[1])=="--probe-module";
        require(pure||module_only||(argc==5&&std::string(argv[1])=="--run"),
                "Expected --describe, --preflight <root>, or --run/--probe-module <root> <seal> <SHA256>");
        const auto root=fs::canonical(argv[2]);out=root/"assets/research-evidence"/protocol;
        if(module_only)out=root/"assets/research-evidence"/(std::string(protocol)+"-module");
        if(!pure){
            mark("seal_verification");
            const auto sealed=read_bytes(argv[3],64*1024);
            require(ph::sha256(sealed)==argv[4],"Seal mismatch");
            const std::string document(sealed.begin(),sealed.end());
            require(ph::scalar(document,"bindings")==binding&&ph::scalar(document,"maximum_attempts")=="1",
                    "Seal does not authorize this exact replay");
            require(ph::scalar(document,"operation")== (module_only?"module_probe":"collection"),
                    "Seal operation differs");
            require(!fs::exists(out)&&fs::create_directories(out),"The four-second replay has already been attempted");owns=true;deadline=std::make_unique<Deadline>(out,started);
        }
        if(module_only){
            probe_module(mark);
            mark("evidence_write");
            const double wall=ms(started);
            const bool passed=wall<4000;
            std::ofstream o(out/"result.json");
            o<<std::setprecision(17)<<"{\"passed\":"<<(passed?"true":"false")
             <<",\"operation\":\"module_probe\",\"seal_sha256\":\""<<argv[4]<<"\",\"bindings\":"<<binding
             <<",\"physical_controls\":0,\"optimizer_calls\":0,\"wall_ms\":"<<wall;
            write_phases(o);o<<"}\n";o.close();
            require(bool(o),"Could not close module timing result");
            return passed?0:2;
        }
        Inputs inputs;Samples ref;read(root,inputs,ref);const double read_ms=ms(started);
        if(pure){std::cout<<"{\"archive_rows\":32768,\"physical_controls\":0,\"optimizer_calls\":0}\n";return 0;}
        Samples d;Timing time;std::vector<Gap> gaps;
        collect_once(inputs,d,gaps,time,[](int n){issued=n;if(n==1)mark("physics_collection");},[](int n){completed=n;},mark);
        mark("compare");
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
        const double cpu_ms=ms(cpu_begin);mark("evidence_write");const auto evidence_start=Clock::now();
        // Store the validity tag and bound for every row; measured numeric minima
        // are intentionally NaN for threshold-only results. The source and fixed
        // archive recover all other physical, actuator, reward and request fields.
        cp::Writer w;w.value(uint32_t{0x31435053});w.value(uint32_t{1});w.value(uint32_t{rows});
        for(int i=0;i<rows;++i){w.value(gaps[i].precise);w.value(gaps[i].bound);w.value(d.trace[i].minimum_gap);w.value(gaps[i].free_steps);w.value(gaps[i].full_queries);}
        cp::write_atomic((out/"clearance-tags.bin").string(),w.data);
        const double evidence_ms=ms(evidence_start);
        const bool passed=m.failing==0&&ms(started)<4000;
        std::ofstream o(out/"result.json");o<<std::setprecision(17)<<"{\"passed\":"<<(passed?"true":"false")<<",\"seal_sha256\":\""<<argv[4]<<"\",\"bindings\":"<<binding
          <<",\"completed_updates\":0,\"physical_controls\":"<<controls<<",\"physical_substeps\":"<<substeps<<",\"free_substeps\":"<<free<<",\"exact_substeps\":"<<exact
          <<",\"exact_queries\":"<<queries<<",\"precise_rows\":"<<precise<<",\"terminal_transitions\":"<<terminal<<",\"time_limit_truncations\":"<<truncated
          <<",\"failed_comparisons\":"<<m.failing<<",\"first_failure\":\""<<m.first<<"\",\"changed_numeric_values\":"<<m.changed<<",\"max_observation_delta\":"<<m.max_obs
          <<",\"max_action_delta\":"<<m.max_action<<",\"max_logp_delta\":"<<m.max_logp<<",\"max_reward_delta\":"<<m.max_reward<<",\"max_value_delta\":"<<m.max_value
          <<",\"max_motion_delta\":"<<m.max_motion<<",\"max_velocity_delta_m_s\":"<<m.max_speed<<",\"max_precise_gap_delta_m\":"<<m.max_precise_gap<<",\"max_swept_bound_minus_reference_m\":"<<m.max_excess
          <<",\"input_verification_ms\":"<<read_ms<<",\"setup_ms\":"<<time.setup_ms<<",\"upload_wall_ms\":"<<time.input_transfer_ms<<",\"gpu_physics_and_task_ms\":"<<time.physics_ms
          <<",\"gpu_inference_sampling_composition_ms\":"<<time.inference_ms<<",\"collection_wall_ms\":"<<time.collection_wall_ms<<",\"collection_transitions_per_second\":"<<1000*controls/time.collection_wall_ms
          <<",\"readback_wall_ms\":"<<time.readback_ms<<",\"cpu_comparison_ms\":"<<cpu_ms<<",\"typed_evidence_write_ms\":"<<evidence_ms<<",\"wall_ms\":"<<ms(started);
        write_phases(o);o<<"}\n";
        o.close();require(bool(o),"Could not close result file");return passed?0:2;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';if(owns){std::ofstream f(out/"error.json");f<<"{\"passed\":false,\"issued_controls_upper_bound\":"<<N*issued<<",\"completed_controls\":"<<N*completed<<",\"completed_substeps\":"<<N*completed*10;write_phases(f);f<<"}\n";std::ofstream(out/"error.txt")<<e.what();}return 2;}
}
