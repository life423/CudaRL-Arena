#include "../pond-ppo/admission.h"
#include "../pond-mirror-check/mirror.h"
#include "core/checkpoint.h"
#include "nn/adam.h"
#include "nn/loss.h"
#include "nn/mlp_gpu.h"
#include "rl/pond_ppo_contract.h"
#include "distill_identity.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <tuple>

namespace fs=std::filesystem;
namespace cp=cudarl::checkpoint;
namespace pp=cudarl::rl::pond_ppo;
namespace ph=cudarl::rl::pond_ppo_host;
namespace nn=cudarl::nn;
using Clock=std::chrono::steady_clock;
constexpr int width=42, actions=4, worlds=512, steps=64, rows=worlds*steps, mb=4096, updates=64;
constexpr float lr=.0001f;
constexpr char study[]="pond-motor-mirror-distill-v1-2026-11-24";
constexpr char parent_sha[]="9b68920b345b4be6699f8b979729e07d9f9d7a451664e4ca99f34487f2ca4ded";
constexpr char source_plan_sha[]="ee7d102f0f13222f1128aa603bcd7bfa36d5082293556f81f57fb788ed2c7eae";
static void require(bool ok,const std::string& reason){if(!ok)throw std::runtime_error(reason);}
static double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}

// Same polar vectors, axial rotation and servo-axis signs verified in the native mirrored replay.
// The appended yaw error is a signed angle and reverses under sagittal reflection.
static int reflect_input(int i){
    if(i<15)return i%3==2?-1:1;
    if(i<18)return i==17?1:-1;
    if(i<34)return mirror::joint_sign((i-18)%4);
    if(i>=36&&i<40)return mirror::joint_sign(i-36);
    return i==41?-1:1;
}
static void verify_reflection(){
    const std::set<int> negated={2,5,8,11,14,15,16,18,20,22,24,26,28,30,32,36,38,41};
    for(int i=0;i<width;++i)require(reflect_input(i)==(negated.count(i)?-1:1)&&reflect_input(i)*reflect_input(i)==1,"Input mirror convention");
    for(int j=0;j<actions;++j)require(mirror::joint_sign(j)==(j%2?1:-1),"Action mirror convention");
}
using Key=std::tuple<int,uint32_t,uint32_t>; // environment, episode, request serial
struct Metadata{uint32_t episode=0,kind=0,serial=0,age=0,events=0;float angle=0;};
template<class T> static T field(const uint8_t* p,size_t offset){T v;std::memcpy(&v,p+offset,sizeof(v));return v;}
struct Batch{std::vector<float> actor,obs,means;std::vector<Metadata> trace;};
static Batch read_batch(const fs::path& path,const std::string& hash,int update,uint64_t source){
    require(ph::sha256(ph::whole_file(path))==hash,"Batch SHA mismatch: "+path.string());
    const auto bytes=cp::read(path.string());cp::Reader in{bytes};
    require(in.value<uint32_t>()==0x31425450&&in.value<uint32_t>()==3&&in.value<uint32_t>()==uint32_t(update)&&in.value<uint32_t>()==steps,"Batch schema");
    require(in.value<uint64_t>()==uint64_t((update-1)*steps)&&in.value<uint64_t>()==source,"Batch serial/source");
    auto skip=[&](size_t n,size_t stride){
        require(in.value<uint64_t>()==n&&n*stride<=in.data.size()-in.position,"Batch array shape");
        in.position+=n*stride;
    };
    skip(1384*worlds,1);skip(368*worlds,1);
    Batch b;b.actor=in.array<float>(2564);skip(2433,sizeof(float));b.obs=in.array<float>(rows*width);
    require(b.actor.size()==2564&&b.obs.size()==rows*width,"Teacher topology");
    skip(rows*actions,4);for(int i=0;i<6;++i)skip(rows,4);skip(worlds,4);
    const auto attempted=in.array<uint64_t>(rows),substeps=in.array<uint64_t>(rows);skip(rows,4);
    require(attempted.size()==rows&&substeps.size()==rows&&std::all_of(attempted.begin(),attempted.end(),[](auto v){return v==1;})&&
            std::all_of(substeps.begin(),substeps.end(),[](auto v){return v==10;}),"Incomplete recorded physics");
    require(in.value<uint32_t>()==192&&in.value<uint64_t>()==rows*192&&in.data.size()-in.position==rows*192,"Expected complete v6 trace");
    b.trace.reserve(rows);
    for(int r=0;r<rows;++r){
        const auto ptr=in.data.data()+in.position+r*192;
        b.trace.push_back({field<uint32_t>(ptr,40),field<uint32_t>(ptr,48),field<uint32_t>(ptr,112),
            field<uint32_t>(ptr,120),field<uint32_t>(ptr,128),field<float>(ptr,108)});
    }
    return b;
}
struct Row{int batch=0,row=0;};
using Pools=std::array<std::vector<Row>,4>; // mirrored successful right, original successful right, straight, rest
static Pools select(const std::vector<Batch>& batches,size_t& successes){
    std::map<Key,uint32_t> completed;
    for(const auto& batch:batches)for(int r=0;r<rows;++r){
        const auto& t=batch.trace[r];
        if((t.events&2u)==0)continue;
        require(t.kind==2&&t.angle<0&&t.serial>0&&t.age>=120&&t.age<=400&&
                completed.emplace(Key{r%worlds,t.episode,t.serial},t.age).second,"Unexpected native completion");
    }
    successes=completed.size();require(successes==5,"Expected all five successful source requests");
    std::map<Key,uint32_t> ages;Pools pools;
    for(size_t i=0;i<batches.size();++i)for(int r=0;r<rows;++r){
        const auto& t=batches[i].trace[r];const Row row{static_cast<int>(i),r};
        if(t.kind==0){pools[3].push_back(row);continue;}
        if(t.kind==1){pools[2].push_back(row);continue;}
        const Key key{r%worlds,t.episode,t.serial};const auto found=completed.find(key);
        if(found==completed.end()||t.age>found->second)continue;
        require(t.age==++ages[key]&&t.angle<0&&((t.events&1u)!=0)==(t.age==1),"Discontinuous successful source request");
        pools[0].push_back(row);pools[1].push_back(row);
    }
    require(ages==completed,"Missed successful request rows");
    for(const auto& pool:pools)require(!pool.empty(),"Empty training stratum");
    return pools;
}
using Mlp=nn::GpuMlp;
static nn::MlpSpec spec(){return {{width,32,32,actions},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}};}
struct Device{
    device_ptr<float> observation{mb*width},target{mb*actions},gradient{mb*actions};
    Mlp teacher{spec(),mb,11},student{spec(),mb,11};
};
static void compute_teachers(std::vector<Batch>& batches,Device& d){
    for(auto& batch:batches){
        d.teacher.set_params(batch.actor);batch.means.resize(rows*actions);
        for(int r=0;r<rows;r+=mb){
            d.observation.copy_from_host(batch.obs.data()+r*width,mb*width);
            d.teacher.forward(d.observation.get(),mb);
            CUDA_CHECK_THROW(cudaMemcpy(batch.means.data()+r*actions,d.teacher.output(),mb*actions*sizeof(float),cudaMemcpyDeviceToHost));
        }
    }
}
struct Mini{
    std::vector<float> obs=std::vector<float>(mb*width),target=std::vector<float>(mb*actions);
    std::array<int,mb> group{};
    void fill(const std::vector<Batch>& batches,const Pools& pools,std::mt19937& rng,bool balanced,int update){
        for(int i=0;i<mb;++i){
            const int group_id=balanced?i%4:(i<mb/2?0:1+(i-mb/2+update)%3);
            group[i]=group_id;const auto& pool=pools[group_id];
            const Row row=pool[std::uniform_int_distribution<size_t>(0,pool.size()-1)(rng)];
            const auto& b=batches[row.batch];
            for(int j=0;j<width;++j)obs[i*width+j]=b.obs[row.row*width+j]*(group_id==0?reflect_input(j):1);
            for(int j=0;j<actions;++j)target[i*actions+j]=b.means[row.row*actions+j]*(group_id==0?mirror::joint_sign(j):1);
        }
    }
    void upload(Device& d)const{d.observation.copy_from_host(obs.data(),obs.size());d.target.copy_from_host(target.data(),target.size());}
};
static std::array<double,4> fit_error(const Mini& probes,Device& d){
    probes.upload(d);d.student.forward(d.observation.get(),mb);
    std::vector<float> prediction(mb*actions);
    CUDA_CHECK_THROW(cudaMemcpy(prediction.data(),d.student.output(),prediction.size()*sizeof(float),cudaMemcpyDeviceToHost));
    std::array<double,4> out{};
    for(int i=0;i<mb;++i)for(int j=0;j<actions;++j){
        const auto delta=double(prediction[i*actions+j])-probes.target[i*actions+j];
        require(std::isfinite(delta),"Nonfinite distillation prediction");out[probes.group[i]]+=delta*delta;
    }
    for(double& v:out)v/=mb*actions/4;
    return out;
}
static void array(std::ostream& s,const std::array<double,4>& values){
    s<<'[';for(size_t i=0;i<values.size();++i)s<<(i?",":"")<<values[i];s<<']';
}
static void run(const fs::path& plan_path,const fs::path& output){
    const auto started=Clock::now();const auto plan_bytes=ph::whole_file(plan_path);
    const std::string plan(plan_bytes.begin(),plan_bytes.end());
    auto value=[&](const std::string& key){return ph::scalar(plan,key);};
    require(value("protocol_id")==study&&value("distill_source_sha256")==source_hash&&value("max_supervised_Adam_calls")=="64"&&
            value("minibatch")=="4096"&&std::stod(value("learning_rate"))==.0001&&
            value("max_new_physical_controls")=="0"&&value("max_PPO_updates")=="0","Distillation allocation/identity mismatch");
    const fs::path source_plan=value("v6_protocol_path_absolute"),source_candidate=value("source_candidate_path_absolute"),batch_root=value("raw_batches_root_absolute");
    require(source_plan.is_absolute()&&source_candidate.is_absolute()&&batch_root.is_absolute()&&fs::absolute(output).lexically_normal()==
            fs::path(value("output_path_absolute")).lexically_normal(),"Unbound experiment paths");
    const auto prior=ph::whole_file(source_plan);
    require(value("v6_protocol_sha256")==source_plan_sha&&ph::sha256(prior)==source_plan_sha&&
            value("source_candidate_sha256")==parent_sha&&ph::sha256(ph::whole_file(source_candidate))==parent_sha,"Protected source differs");
    const std::string prior_text(prior.begin(),prior.end());
    const auto parent=std::stoull(ph::scalar(prior_text,"parent_payload_fnv1a64")),source=std::stoull(ph::scalar(prior_text,"source_identity_fnv1a64"));
    const auto candidate_payload=cp::read(source_candidate.string());
    const auto initial=pp::read_candidate(candidate_payload,parent,cp::checksum(prior),source,1384,368,width);
    require(initial.counters.completed_updates==8,"Require saved v6 update 8");verify_reflection();
    std::vector<Batch> batches;for(int i=1;i<=8;++i)
        batches.push_back(read_batch(batch_root/("batch-"+std::to_string(i)+".bin"),value("batch_"+std::to_string(i)+"_sha256"),i,source));
    size_t successes=0;const Pools pools=select(batches,successes);
    require(!fs::exists(output)&&fs::create_directories(output),"Distillation output exists; one attempt only");
    int calls=0;
    try{
        CUDA_CHECK_THROW(cudaSetDevice(0));Device d;compute_teachers(batches,d);d.student.set_params(initial.actor);
        std::mt19937 probe_rng(93481),train_rng(81239);
        Mini probes,mini;probes.fill(batches,pools,probe_rng,true,0);const auto before=fit_error(probes,d);
        nn::AdamConfig cfg;cfg.lr=lr;cfg.max_grad_norm=1;
        nn::GpuAdam adam(2564,cfg);
        const auto fit_start=Clock::now();
        for(int i=0;i<updates;++i){
            mini.fill(batches,pools,train_rng,false,i);mini.upload(d);d.student.forward(d.observation.get(),mb);
            nn::mse_loss_grad(d.student.output(),d.target.get(),d.gradient.get(),mb*actions);
            d.student.zero_grad();d.student.backward(d.gradient.get(),mb,nullptr);
            const auto gradients=d.student.get_grads();
            require(std::all_of(gradients.begin(),gradients.end(),[](float g){return std::isfinite(g);}),"Nonfinite distillation gradients");
            adam.step(d.student.params(),d.student.grads());++calls;
        }
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        const double fit_s=seconds(fit_start);const auto after=fit_error(probes,d);
        const auto weights=d.student.get_params();require(std::all_of(weights.begin(),weights.end(),[](float x){return std::isfinite(x);}),"Nonfinite student weights");
        require(weights!=initial.actor&&calls==updates,"Supervised actor did not change");
        cp::Writer actor;for(uint32_t v:{0x31445350u,1u,42u,4u,2564u,uint32_t(calls)})actor.value(v);
        actor.value(cp::checksum(candidate_payload));actor.value(cp::checksum(plan_bytes));actor.array(weights);pp::seal(actor);
        cp::write_atomic((output/"distilled-actor.bin").string(),actor.data);
        std::ofstream result(output/"result.json");result<<std::setprecision(17)<<"{\"completed\":true,\"protocol_id\":\""<<study
            <<"\",\"successful_right_requests\":"<<successes<<",\"successful_right_rows\":"<<pools[0].size()
            <<",\"straight_anchor_pool\":"<<pools[2].size()<<",\"rest_anchor_pool\":"<<pools[3].size()
            <<",\"minibatch\":4096,\"supervised_Adam_calls\":"<<calls<<",\"PPO_updates\":0,\"physics_controls\":0,\"critic_updates\":0"
            <<",\"strata\":[\"mirrored_right_as_left\",\"original_right\",\"straight\",\"rest\"],\"training_distribution_probe_MSE_before\":";
        array(result,before);result<<",\"training_distribution_probe_MSE_after\":";array(result,after);
        result<<",\"fit_wall_seconds\":"<<fit_s<<",\"total_wall_seconds\":"<<seconds(started)
              <<",\"actor_sha256\":\""<<ph::sha256(ph::whole_file(output/"distilled-actor.bin"))<<"\"}\n";result.close();
        require(bool(result),"Result write failed");
    }catch(...){std::ofstream failed(output/"failed-attempt.json");failed<<"{\"completed\":false,\"supervised_Adam_calls\":"<<calls<<",\"PPO_updates\":0,\"physics_controls\":0}\n";throw;}
}
int main(int argc,char** argv){
    try{
        verify_reflection();
        if(argc==2&&std::string(argv[1])=="--describe"){
            std::cout<<"{\"protocol_id\":\""<<study<<"\",\"source_sha256\":\""<<source_hash
                <<"\",\"actor_inputs\":42,\"actor_weights\":2564,\"max_supervised_Adam_calls\":64,\"minibatch\":4096,\"learning_rate\":0.0001,\"mirror_signs_checked\":46,\"physics_controls\":0,\"PPO_updates\":0}\n";return 0;}
        require(argc==6&&std::string(argv[1])=="--run"&&std::string(argv[2])=="--protocol"&&std::string(argv[4])=="--out","Expected --describe or --run --protocol <plan> --out <fresh output>");
        run(argv[3],argv[5]);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
