#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#include "queries.h"
#include "bindings.h"
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
using namespace query_profile;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void require(bool yes,const std::string& why){if(!yes)throw std::runtime_error(why);}
void cuda(cudaError_t v){if(v!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(v));}
struct Deadline{
    fs::path out;std::mutex mutex;std::condition_variable cv;bool done=false;std::thread watchdog;
    Deadline(const fs::path& path,Clock::time_point start):out(path),watchdog([this,start]{
        std::unique_lock<std::mutex> lock(mutex);
        if(!cv.wait_until(lock,start+std::chrono::seconds(30),[this]{return done;})){
            std::ofstream(out/"timeout.json")<<"{\"passed\":false,\"reason\":\"30_second_wall_cap\"}\n";
            std::_Exit(3);
        }
    }){}
    ~Deadline(){{std::lock_guard<std::mutex> l(mutex);done=true;}cv.notify_one();watchdog.join();}
};
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
std::vector<uint8_t> load(const fs::path& path,const char* expected,int update){
    require(digest(path)==expected,"Frozen input SHA-256 changed: "+path.string());
    const auto payload=cp::read(path.string());cp::Reader r{payload};
    require(r.value<uint32_t>()==0x31425252&&r.value<uint32_t>()==1&&r.value<int>()==update&&r.value<int>()==64,"Residual rollout header");
    auto data=r.array<uint8_t>(worlds*world_bytes);require(data.size()==worlds*world_bytes,"Native world ABI");return data;
}
struct Device{
    cudaStream_t s{};cudaEvent_t a{},b{};void* w=nullptr;Summary* q[2]{};void* upload=nullptr;Summary* returned[2]{};
    Device(){
        cuda(cudaSetDevice(0));cuda(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));cuda(cudaEventCreate(&a));cuda(cudaEventCreate(&b));
        cuda(cudaMalloc(&w,worlds*world_bytes));cuda(cudaMallocHost(&upload,worlds*world_bytes));
        for(int v=0;v<2;++v){cuda(cudaMalloc(&q[v],worlds*sizeof(Summary)));cuda(cudaMallocHost(&returned[v],worlds*sizeof(Summary)));}
    }
    ~Device(){for(int v=0;v<2;++v){cudaFreeHost(returned[v]);cudaFree(q[v]);}cudaFreeHost(upload);cudaFree(w);cudaEventDestroy(a);cudaEventDestroy(b);cudaStreamDestroy(s);}
    template<class F> std::array<double,3> time(F issue){
        const auto begin=Clock::now();cuda(cudaEventRecord(a,s));issue();cuda(cudaGetLastError());cuda(cudaEventRecord(b,s));
        const auto wait=Clock::now();cuda(cudaEventSynchronize(b));const double wait_ms=ms(wait);
        float gpu=0;cuda(cudaEventElapsedTime(&gpu,a,b));return {gpu,ms(begin),wait_ms};
    }
};
double median(std::vector<double> values){
    std::sort(values.begin(),values.end());const auto n=values.size();
    return .5*(values[n/2]+values[(n-1)/2]);
}
struct Stage{
    std::string label;int count=0;std::vector<double> timings[2];double warm[2]{},wall[2]{},wait[2]{},cpu_ms=0,h2d=0,d2h=0,transfer_wall=0,transfer_wait=0;
    uint64_t visits[3]{};uint32_t maximum[3]{},nonflat[3]{};int failed=0;double max_gap=0,max_point=0,max_gradient=0;
};
bool interval(double a,double b,double c,double d){
    return std::isfinite(a)&&std::isfinite(b)&&std::isfinite(c)&&std::isfinite(d)&&a<=b&&c<=d&&
           b-a<=tolerance&&d-c<=tolerance&&a<=d&&c<=b&&std::abs(a-c)<=tolerance&&std::abs(b-d)<=tolerance;
}
bool matches(const Summary& ref,const Summary& q,Stage& stage){
    bool ok=ref.valid&&q.valid&&ref.certified&&q.certified&&ref.failure==0&&q.failure==0&&
            interval(ref.low,ref.high,q.low,q.high)&&std::abs(ref.max_y-q.max_y)<=tolerance;
    for(int i=0;i<3;++i){const auto& a=ref.link[i];const auto& b=q.link[i];ok=ok&&a.certified&&b.certified&&interval(a.low,a.high,b.low,b.high);
        stage.max_gap=std::max({stage.max_gap,std::abs(a.low-b.low),std::abs(a.high-b.high)});
        for(int k=0;k<3;++k)stage.max_point=std::max(stage.max_point,std::abs(a.point[k]-b.point[k]));
        for(int k=0;k<2;++k)stage.max_gradient=std::max(stage.max_gradient,std::abs(a.gradient[k]-b.gradient[k]));
    }
    return ok;
}
Stage measure(const std::string& name,const std::vector<uint8_t>& raw,bool timed,Device& d,std::ostream& csv,std::ostream& calls){
    Stage m;m.label=name;m.count=int(raw.size()/world_bytes);const int n=m.count;
    std::memcpy(d.upload,raw.data(),raw.size());
    auto transfer=d.time([&]{cuda(cudaMemcpyAsync(d.w,d.upload,raw.size(),cudaMemcpyHostToDevice,d.s));});
    m.h2d=transfer[0];m.transfer_wall+=transfer[1];m.transfer_wait+=transfer[2];
    const Launch launch[2]={baseline::launch,candidate::launch};
    const int passes=timed?repeats+1:1;
    for(int pass=0;pass<passes;++pass)for(int slot=0;slot<2;++slot){
        // Warm independently. Timed orders AB,BA,AB,BA counterbalance ordering.
        const int v=slot^((pass>0)?((pass-1)%2):0);
        const auto record=d.time([&]{launch[v](d.w,d.q[v],n,d.s);});
        calls<<name<<','<<pass<<','<<slot<<','<<(v?"candidate":"baseline")<<','<<n<<','<<record[0]<<','<<record[1]<<','<<record[2]<<'\n';calls.flush();
        if(pass==0)m.warm[v]=record[0];
        else{m.timings[v].push_back(record[0]);m.wall[v]+=record[1];m.wait[v]+=record[2];}
    }
    for(int v=0;v<2;++v){
        transfer=d.time([&]{cuda(cudaMemcpyAsync(d.returned[v],d.q[v],n*sizeof(Summary),cudaMemcpyDeviceToHost,d.s));});
        m.d2h+=transfer[0];m.transfer_wall+=transfer[1];m.transfer_wait+=transfer[2];
    }
    std::vector<Summary> refs(n);const auto t=Clock::now();
    for(int i=0;i<n;++i)refs[i]=baseline::cpu(raw.data()+i*world_bytes);
    m.cpu_ms=ms(t);
    for(int i=0;i<n;++i){
        const auto& old=d.returned[0][i];const auto& next=d.returned[1][i];const auto& ref=refs[i];
        const bool cpu_old=matches(ref,old,m),cpu_new=matches(ref,next,m),old_new=matches(old,next,m);
        const bool independent=timed|| (i==0?(ref.low<=.0736 && ref.high>=.0736 && next.low<=.0736 && next.high>=.0736):next.link[1].high<0);
        const bool ok=cpu_old&&cpu_new&&old_new&&independent;
        if(!ok)++m.failed;
        int j=0;for(const Summary* item:{&ref,&old,&next}){m.visits[j]+=item->visits;m.maximum[j]=std::max(m.maximum[j],item->visits);m.nonflat[j]+=item->visits>4;++j;}
        csv<<name<<','<<i<<','<<ref.visits<<','<<old.visits<<','<<next.visits<<','<<ref.low<<','<<ref.high<<','<<old.low<<','<<old.high<<','<<next.low<<','<<next.high<<','
           <<ref.failure<<','<<old.failure<<','<<next.failure<<','<<next.link[1].high<<','<<cpu_old<<','<<cpu_new<<','<<old_new<<','<<independent<<','<<ok<<'\n';
    }
    csv.flush();return m;
}
void report(std::ostream& o,const Stage& m){
    o<<"{\"stage\":\""<<m.label<<"\",\"unique_worlds\":"<<m.count<<",\"failed_worlds\":"<<m.failed<<",\"cpu_reference_ms\":"<<m.cpu_ms
     <<",\"cpu_reference_visits\":"<<m.visits[0]<<",\"h2d_device_ms\":"<<m.h2d<<",\"d2h_both_device_ms\":"<<m.d2h<<",\"transfer_wall_ms\":"<<m.transfer_wall
     <<",\"transfer_wait_ms\":"<<m.transfer_wait<<",\"max_link_bound_difference_m\":"<<m.max_gap<<",\"max_witness_component_difference_m\":"<<m.max_point
     <<",\"max_witness_gradient_component_difference\":"<<m.max_gradient<<",\"arms\":[";
    for(int v=0;v<2;++v){if(v)o<<',';o<<"{\"arm\":\""<<(v?"candidate":"baseline")<<"\",\"warm_or_fixture_gpu_ms\":"<<m.warm[v]
        <<",\"visits_per_complete_batch\":"<<m.visits[v+1]<<",\"maximum_world_visits\":"<<m.maximum[v+1]<<",\"worlds_over_four_visits\":"<<m.nonflat[v+1]
        <<",\"timed_kernel_wall_total_ms\":"<<m.wall[v]<<",\"timed_kernel_wait_total_ms\":"<<m.wait[v];
        if(!m.timings[v].empty())o<<",\"median_gpu_ms\":"<<median(m.timings[v]);o<<'}';}
    o<<']';if(!m.timings[0].empty())o<<",\"baseline_over_candidate_median\":"<<median(m.timings[0])/median(m.timings[1]);o<<'}';
}
const char early_hash[]="ebdd0441397e2a9f26e47dea2832322e0a4fa152030e29372215cdd6bc9483ca";
const char late_hash[]="5994cec04228de4a50c71ad782149194804905d33146356b4232d1922cb6b8da";
int main(int argc,char** argv){
    const auto started=Clock::now();fs::path output;bool owns=false;
    try{
        if(argc==2&&std::string(argv[1])=="--describe"){std::cout<<"{\"baseline_sha256\":\""<<old_sha<<"\",\"candidate_sha256\":\""<<new_sha
            <<"\",\"shared_dependency_hash\":\""<<common_sha<<"\",\"worlds_per_batch\":512,\"timed_passes\":4,\"warmups_per_stage_and_arm\":1,\"extra_correctness_fixtures\":2,\"wall_cap_seconds\":30,\"optimizer_calls\":0}\n";return 0;}
        require(argc==3&&std::string(argv[1])=="--run","Use --describe or --run <worktree>");
        const auto root=fs::canonical(argv[2]);output=root/"assets/research-evidence/pond-clearance-profile-v1";
        require(!fs::exists(output)&&fs::create_directories(output),"This finite profile has already been attempted");owns=true;Deadline deadline(output,started);
        require(std::string(old_sha)!=new_sha,"Compiled candidate matches historical implementation");
        const auto t=Clock::now();
        const auto early=load(root/"build/turn38-residual/updates-1-8/batch-8.bin",early_hash,8),late=load(root/"build/turn38-residual/updates-9-16/batch-12.bin",late_hash,12);
        const double read_ms=ms(t);std::vector<uint8_t> fixtures(2*world_bytes);const double corner=baseline::make_fixtures(fixtures.data());
        require(corner>0,"Curved-bank counterexample failed corner check");
        std::ofstream csv(output/"worlds.csv"),calls(output/"launches.csv");csv<<std::setprecision(17);calls<<std::setprecision(17);
        csv<<"stage,world,cpu_ref_visits,baseline_visits,candidate_visits,cpu_lower_m,cpu_upper_m,baseline_lower_m,baseline_upper_m,candidate_lower_m,candidate_upper_m,cpu_reason,baseline_reason,candidate_reason,candidate_rod_upper_m,cpu_vs_baseline,cpu_vs_candidate,baseline_vs_candidate,independent_fixture_pass,passed\n";
        calls<<"stage,pass_zero_is_warm_or_fixture,order,arm,worlds,gpu_ms,wall_ms,event_wait_ms\n";require(bool(csv)&&bool(calls),"Evidence streams");
        std::vector<Stage> stages;double setup=0;
        {const auto setup_start=Clock::now();Device d;setup=ms(setup_start);
        stages.push_back(measure("early_8",early,true,d,csv,calls));stages.push_back(measure("late_12",late,true,d,csv,calls));
        stages.push_back(measure("analytic_flat_and_curved_rod",fixtures,false,d,csv,calls));}
        const bool passed=std::all_of(stages.begin(),stages.end(),[](const Stage& m){return m.failed==0;});
        std::ofstream o(output/"result.json");o<<std::setprecision(17)<<"{\"passed\":"<<(passed?"true":"false")<<",\"baseline_sha256\":\""<<old_sha<<"\",\"candidate_sha256\":\""<<new_sha<<"\",\"common_dependency_sha256\":\""<<common_sha
            <<"\",\"early_batch_sha256\":\""<<early_hash<<"\",\"late_batch_sha256\":\""<<late_hash<<"\",\"block_threads\":8,\"wall_cap_seconds\":30,\"tolerance_m\":"<<tolerance
            <<",\"cpu_body_queries\":1026,\"baseline_gpu_queries\":5122,\"candidate_gpu_queries\":5122,\"optimizer_calls\":0,\"physical_controls\":0,\"physical_substeps\":0"
            <<",\"fixture_read_and_checksum_ms\":"<<read_ms<<",\"cuda_setup_ms\":"<<setup<<",\"curved_rod_endpoint_corner_min_m\":"<<corner<<",\"stages\":[";
        for(size_t i=0;i<stages.size();++i){if(i)o<<',';report(o,stages[i]);}o<<"],\"wall_ms\":"<<ms(started)<<"}\n";o.close();csv.close();calls.close();
        require(bool(o)&&bool(csv)&&bool(calls),"Evidence finalization");return passed?0:2;
    }catch(const std::exception& err){std::cerr<<err.what()<<'\n';if(owns)std::ofstream(output/"error.txt")<<err.what()<<'\n';return 2;}
}
