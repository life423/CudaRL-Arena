#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#include "replay.h"
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
#include <memory>
#include <sstream>
#include <thread>
#include <vector>

namespace fs=std::filesystem;
namespace cp=cudarl::checkpoint;
using namespace replay;
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

const char early_hash[]="ebdd0441397e2a9f26e47dea2832322e0a4fa152030e29372215cdd6bc9483ca";
const char late_hash[]="5994cec04228de4a50c71ad782149194804905d33146356b4232d1922cb6b8da";
const char bank_from_hash[]="3d0f9f11a51fc0b779cdd835a5b28cd93891a44db6a5da8ead078e255aab8c3b";
const char bank_to_hash[]="03e9fa2783db42c2932c4155903cae3c93a1e26cc737de382c2b0c1b65f8ab5e";
struct Group{std::string name;std::vector<uint8_t> states;std::vector<Request> requests;std::vector<int> cpu_ids;};
Group rollout(const fs::path& p,const char* hash,int update,std::vector<int> ids){
    require(digest(p)==hash,"Frozen residual batch SHA");const auto bytes=cp::read(p.string());cp::Reader r{bytes};
    require(r.value<uint32_t>()==0x31425252&&r.value<uint32_t>()==1&&r.value<int>()==update&&r.value<int>()==64,"Residual rollout header");
    Group g{"batch_"+std::to_string(update),r.array<uint8_t>(worlds*world_bytes),std::vector<Request>(worlds),ids};
    const auto tasks=r.array<uint8_t>(worlds*task_bytes);require(tasks.size()==worlds*task_bytes&&g.states.size()==worlds*world_bytes,"World/task stride");
    const int rows=worlds*64;
    for(const int count:{722,2433,rows*42,rows*2,rows,rows*4}){
        require(r.value<uint64_t>()==count&&count*sizeof(float)<=bytes.size()-r.position,"Malformed raw batch");r.position+=count*sizeof(float);
    }
    const auto latent=r.array<float>(rows*4);require(latent.size()==rows*4,"Latent dimension");
    for(int i=0;i<worlds;++i){
        auto& c=g.requests[i];c.holding=candidate::holding(tasks.data()+i*task_bytes);c.direct=0;
        for(int j=0;j<4;++j){c.value[j]=latent[4*i+j];require(std::isfinite(c.value[j]),"Nonfinite saved action");}
    }
    return g;
}
Group contacts(const fs::path& root){
    Group g{"bank_control_2",std::vector<uint8_t>(4*world_bytes),std::vector<Request>(4),{0,1,2,3}};
    for(int t=1;t<=2;++t){
        const auto p=root/"build/pw31/gpu/frames"/(t==1?"step-0001.bin":"step-0002.bin");
        require(digest(p)==(t==1?bank_from_hash:bank_to_hash),"Frozen bank frame SHA");
        const auto bytes=cp::read(p.string());cp::Reader r{bytes};
        require(r.value<uint32_t>()==0x32474350&&r.value<uint32_t>()==1&&r.value<int>()==t&&r.value<uint32_t>()==12448&&r.value<uint32_t>()==8&&bytes.size()==20+16*12448,"Bank frame ABI");
        for(int i=0;i<4;++i){
            const auto* frame=bytes.data()+20+(8+4+i)*12448;
            if(t==1)std::memcpy(g.states.data()+i*world_bytes,frame,world_bytes);
            else{auto& c=g.requests[i];c.direct=1;std::memcpy(c.value,frame+world_bytes,4*sizeof(float));for(float z:c.value)require(z==0,"Frozen bank request");}
        }
    }
    return g;
}
struct Differences{
    double body=0,velocity=0,motor=0,requested=0,energy=0,force=0,gap=0;
    int failures=0,contact_steps=0,solver_disagreements=0;std::string first;
    void test(bool ok,const char* label){if(!ok){++failures;if(first.empty())first=label;}}
    void compare(double a,double b,double tolerance,double& high,const char* label){
        const double diff=std::abs(a-b);high=std::max(high,diff);test(std::isfinite(a)&&std::isfinite(b)&&diff<=tolerance,label);
    }
    void relative(double a,double b,double absolute,double scale,double& high,const char* label){
        compare(a,b,absolute+scale*std::max(std::abs(a),std::abs(b)),high,label);
    }
    void assess(const Output& a,const Output& b){
        test(a.success&&b.success&&a.controls==1&&b.controls==1&&a.substeps==10&&b.substeps==10,"control_counts");
        test(std::memcmp(a.command,b.command,sizeof(a.command))==0,"command_identity");
        for(int k=0;k<ticks;++k){
            const auto& x=a.steps[k];const auto& y=b.steps[k];
            test(x.valid&&y.valid&&x.failed==0&&y.failed==0&&x.attempts==y.attempts&&x.accepted==y.accepted,"substep_status");
            solver_disagreements+=x.iterations!=y.iterations;
            for(int t=0;t<3;++t)for(int j=0;j<21;++j){
                const bool linear=j>=11&&j<14;compare(x.body[t][j],y.body[t][j],linear?2e-7:2e-5,linear?velocity:body,"body");
            }
            for(int j=0;j<4;++j){
                compare(x.held[j],y.held[j],2e-5,motor,"held");compare(x.rate[j],y.rate[j],2e-5,motor,"filter");
                compare(x.actual[j],y.actual[j],2e-5,motor,"actual_torque");compare(x.applied[j],y.applied[j],2e-5,motor,"applied_torque");
                compare(x.requested[j],y.requested[j],1e30,requested,"finite_requested_torque_Nm");
            }
            for(int j=0;j<16;++j)relative(x.energy[j],y.energy[j],1e-12,1e-4,energy,"energy");
            compare(x.low,y.low,1e-5,gap,"lower_gap");compare(x.high,y.high,1e-5,gap,"upper_gap");
            bool active=false;
            for(int l=0;l<3;++l){
                test(x.mode[l]==y.mode[l],"contact_mode");active|=x.mode[l]!=0;
                for(int c=0;c<3;++c)relative(x.force[l][c],y.force[l][c],1e-12,2e-5,force,"force_N");
            }
            contact_steps+=active;
        }
    }
    void print(std::ostream& o) const{
        o<<"{\"failed_components\":"<<failures<<",\"first_failure\":\""<<first<<"\",\"max_body\":"<<body<<",\"max_velocity_m_s\":"<<velocity
         <<",\"max_actuator\":"<<motor<<",\"max_requested_torque_Nm\":"<<requested<<",\"max_contact_force_N\":"<<force<<",\"max_energy_J\":"<<energy
         <<",\"max_gap_m\":"<<gap<<",\"nonzero_contact_substeps\":"<<contact_steps<<",\"solver_iteration_count_differences\":"<<solver_disagreements<<'}';
    }
};

struct Device{
    cudaStream_t stream{};cudaEvent_t begin{},end{};void* state=nullptr;void* scratch=nullptr;Request* r=nullptr;Output* out=nullptr;
    Device(){
        cuda(cudaSetDevice(0));cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));cuda(cudaEventCreate(&begin));cuda(cudaEventCreate(&end));
        require(baseline::scratch()==candidate::scratch(),"Trace ABI drift");cuda(cudaMalloc(&scratch,worlds*baseline::scratch()));
        cuda(cudaMalloc(&state,worlds*world_bytes));cuda(cudaMalloc(&r,worlds*sizeof(Request)));cuda(cudaMalloc(&out,worlds*sizeof(Output)));
        baseline::load();candidate::load();
    }
    ~Device(){cudaFree(out);cudaFree(r);cudaFree(scratch);cudaFree(state);cudaEventDestroy(end);cudaEventDestroy(begin);cudaStreamDestroy(stream);}
    double run(Issue launch,const Group& g,std::vector<Output>& results,double& transfer){
        const int n=int(g.requests.size());const auto a=Clock::now();
        cuda(cudaMemcpyAsync(state,g.states.data(),n*world_bytes,cudaMemcpyHostToDevice,stream));
        cuda(cudaMemcpyAsync(r,g.requests.data(),n*sizeof(Request),cudaMemcpyHostToDevice,stream));
        cuda(cudaStreamSynchronize(stream));transfer+=ms(a);
        cuda(cudaEventRecord(begin,stream));launch(state,r,scratch,out,n,stream);cuda(cudaGetLastError());cuda(cudaEventRecord(end,stream));cuda(cudaEventSynchronize(end));
        float amount=0;cuda(cudaEventElapsedTime(&amount,begin,end));
        const auto b=Clock::now();cuda(cudaMemcpyAsync(results.data(),out,n*sizeof(Output),cudaMemcpyDeviceToHost,stream));cuda(cudaStreamSynchronize(stream));transfer+=ms(b);return amount;
    }
};
void count(std::ofstream& ledger,const Group& g,const char* arm,unsigned long long controls,unsigned long long steps,const char* state){
    ledger<<g.name<<','<<arm<<','<<controls<<','<<steps<<','<<state<<'\n';ledger.flush();require(bool(ledger),"Accounting ledger");
}
int main(int argc,char** argv){
    const auto started=Clock::now();fs::path output;bool owns=false;
    try{
        const std::string binding=std::string("{\"protocol\":\"pond-clearance-dynamics-v1\",\"baseline_sha256\":\"")+old_sha+"\",\"candidate_sha256\":\""+new_sha
            +"\",\"shared_dependency_sha256\":\""+common_sha+"\",\"maximum_gpu_controls\":2056,\"maximum_cpu_controls\":8,\"maximum_physical_substeps\":20640,\"wall_cap_seconds\":30,\"optimizer_calls\":0}";
        if(argc==2&&std::string(argv[1])=="--describe"){std::cout<<binding<<'\n';return 0;}
        const bool preflight=argc==3&&std::string(argv[1])=="--preflight";
        require(preflight||(argc==5&&std::string(argv[1])=="--run"),"Use --describe, --preflight <root>, or --run <root> <seal path> <seal SHA256>");
        const auto root=fs::canonical(argv[2]);output=root/"assets/research-evidence/pond-clearance-dynamics-v1";
        std::unique_ptr<Deadline> deadline;
        if(!preflight){
            require(digest(argv[3])==argv[4],"Physics seal SHA-256");const auto seal_data=fs::file_size(argv[3]);require(seal_data>80,"Empty seal");
            require(!fs::exists(output)&&fs::create_directories(output),"This one-shot physical comparison was already attempted");
            owns=true;deadline=std::make_unique<Deadline>(output,started);
        }
        std::vector<Group> groups;
        groups.push_back(rollout(root/"build/turn38-residual/updates-1-8/batch-8.bin",early_hash,8,{0,1}));
        groups.push_back(rollout(root/"build/turn38-residual/updates-9-16/batch-12.bin",late_hash,12,{55,315}));
        groups.push_back(contacts(root));
        if(preflight){std::cout<<"{\"fixture_groups\":3,\"verified_states\":1028,\"saved_command_sets\":1028,\"physical_controls\":0}\n";return 0;}
        const double start_ms=ms(started);const auto setup=Clock::now();Device d;const double setup_ms=ms(setup);
        std::ofstream csv(output/"worlds.csv"),log(output/"ledger.csv"),bank(output/"bank-forces.csv");
        csv<<"stage,world,backend_pair,failed_component_count,first_failure,contact_substeps,max_force_diff_N,max_position_rotation_joint_diff,max_linear_velocity_diff_m_s,max_actuator_diff,max_energy_diff_J\n";
        bank<<"bank_id,backend,substep,link,contact_mode,force_x_N,force_y_N,force_z_N\n";
        log<<"stage,arm,controls,substeps,status\n";csv<<std::setprecision(17);bank<<std::setprecision(17);
        unsigned long long gpu_controls=0,gpu_steps=0,cpu_controls=0,cpu_steps=0;
        bool passed=true;int index=0;std::ostringstream stages;stages<<std::setprecision(17);
        for(const auto& g:groups){
            const int n=int(g.requests.size());const bool is_bank=g.name=="bank_control_2";
            std::vector<Output> left(n),right(n);std::vector<Output>* sets[2]={&left,&right};const Issue launches[2]={baseline::launch,candidate::launch};
            double transfers=0,times[2]{},cpu_ms=0;
            for(int v=0;v<2;++v){
                const char* label=v?"candidate_cuda":"frozen_cuda";count(log,g,label,n,0,"issued");
                times[v]=d.run(launches[v],g,*sets[v],transfers);
                unsigned long long controls=0,substeps=0;for(const auto& o:*sets[v]){controls+=o.controls;substeps+=o.substeps;}
                gpu_controls+=controls;gpu_steps+=substeps;count(log,g,label,controls,substeps,"actual");
            }
            auto row=[&](int id,const char* compare,const Differences& delta){
                csv<<g.name<<','<<(is_bank?id+4:id)<<','<<compare<<','<<delta.failures<<','<<delta.first<<','<<delta.contact_steps<<','<<delta.force<<','<<delta.body<<','<<delta.velocity<<','<<delta.motor<<','<<delta.energy<<'\n';
            };
            auto save_bank=[&](int i,const Output& o,const char* label){
                for(int k=0;k<ticks;++k)for(int l=0;l<3;++l){
                    const auto& b=o.steps[k];bank<<i+4<<','<<label<<','<<k+1<<','<<l<<','<<b.mode[l]<<','<<b.force[l][0]<<','<<b.force[l][1]<<','<<b.force[l][2]<<'\n';
                }
            };
            Differences gpu_delta,cpu_delta;
            for(int i=0;i<n;++i){
                Differences entry;entry.assess(left[i],right[i]);if(is_bank)entry.test(entry.contact_steps>0,"no_bank_force");
                row(i,"frozen_cuda_vs_candidate_cuda",entry);gpu_delta.assess(left[i],right[i]);passed&=entry.failures==0;
                if(is_bank){save_bank(i,left[i],"frozen_cuda");save_bank(i,right[i],"candidate_cuda");}
            }
            for(int i:g.cpu_ids){
                const auto t=Clock::now();const auto ref=candidate::cpu(g.states.data()+i*world_bytes,right[i].command);cpu_ms+=ms(t);
                count(log,g,"candidate_cpu",ref.controls,ref.substeps,"actual");cpu_controls+=ref.controls;cpu_steps+=ref.substeps;
                Differences entry;entry.assess(ref,right[i]);if(is_bank)entry.test(entry.contact_steps>0,"no_bank_force");
                row(i,"candidate_cpu_vs_candidate_cuda",entry);cpu_delta.assess(ref,right[i]);passed&=entry.failures==0;
                if(is_bank)save_bank(i,ref,"candidate_cpu");
            }
            if(index++)stages<<',';stages<<"{\"stage\":\""<<g.name<<"\",\"worlds\":"<<n<<",\"old_gpu_ms\":"<<times[0]<<",\"new_gpu_ms\":"<<times[1]
                <<",\"transfer_wall_ms\":"<<transfers<<",\"cpu_reference_ms\":"<<cpu_ms<<",\"gpu_comparison\":";gpu_delta.print(stages);stages<<",\"cpu_comparison\":";cpu_delta.print(stages);stages<<'}';
            csv.flush();bank.flush();if(!passed)break;
        }
        passed&=gpu_controls==2056&&gpu_steps==20560&&cpu_controls==8&&cpu_steps==80;
        std::ofstream o(output/"result.json");o<<std::setprecision(17)<<"{\"passed\":"<<(passed?"true":"false")<<",\"protocol_seal_sha256\":\""<<argv[4]
            <<"\",\"bindings\":"<<binding<<",\"gpu_controls\":"<<gpu_controls<<",\"gpu_substeps\":"<<gpu_steps<<",\"cpu_controls\":"<<cpu_controls<<",\"cpu_substeps\":"<<cpu_steps
            <<",\"fixture_read_ms\":"<<start_ms<<",\"cuda_setup_load_ms\":"<<setup_ms<<",\"stages\":["<<stages.str()<<"],\"wall_ms\":"<<ms(started)<<"}\n";
        o.close();csv.close();bank.close();log.close();require(bool(o)&&bool(csv)&&bool(bank)&&bool(log),"Evidence finalization");return passed?0:2;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';if(owns)std::ofstream(output/"error.txt")<<e.what()<<'\n';return 2;}
}
