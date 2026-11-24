// Read-only inference on preserved motor 336 and its first archived rollout.
// This target links no optimizer, CUDA learner or physical stepping code.
#include "core/checkpoint.h"
#include "nn/mlp_cpu.h"
#include "rl/append_hold_checkpoint.h"
#include "rl/pond_ppo_task.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace cp=cudarl::checkpoint;
namespace nn=cudarl::nn;
namespace p=cudarl::physics;
namespace task=cudarl::rl::pond_ppo_task;
namespace fs=std::filesystem;
constexpr int D=41,H=32,A=4,N=512,T=64,R=N*T;
constexpr double pi=3.14159265358979323846;
using Input=std::array<float,D>;
using Output=std::array<float,A>;
using Net=nn::CpuMlp<float>;
void need(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
template<class V> std::vector<V> array(cp::Reader& in,size_t n){
    auto values=in.array<V>(n);need(values.size()==n,"Wrong archived array size");return values;
}
template<class V> V field(const uint8_t* bytes,int offset){
    V v;std::memcpy(&v,bytes+offset,sizeof(V));return v;
}
Net model(){return Net({{D,H,H,A},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}});}
struct Moment{
    size_t n=0;double sum=0,squares=0,lo=1e30,hi=-1e30;
    void add(double v){need(std::isfinite(v),"Nonfinite read-only result");++n;sum+=v;squares+=v*v;lo=std::min(lo,v);hi=std::max(hi,v);}
    double mean()const{return sum/double(n);}
    double rms()const{return std::sqrt(squares/double(n));}
};
std::vector<Output> commands(Net& actor,const std::vector<Input>& rows){
    need(!rows.empty(),"No audit states");std::vector<float> inputs;
    inputs.reserve(rows.size()*D);for(const auto& row:rows)inputs.insert(inputs.end(),row.begin(),row.end());
    actor.forward(inputs.data(),static_cast<int>(rows.size()));
    std::vector<Output> outputs(rows.size());
    for(size_t i=0;i<rows.size();++i)for(int j=0;j<A;++j)outputs[i][j]=std::tanh(actor.output()[i*A+j]);
    return outputs;
}
void values(std::ostream& out,const std::array<Moment,A>& rows){
    for(const auto& row:rows)out<<','<<row.mean()<<','<<row.rms()<<','<<row.lo<<','<<row.hi;
}
int reflection(int i){
    if(i<15)return i%3==2?-1:1;
    if(i<18)return i==17?1:-1;
    if(i<34)return (i-18)%2?1:-1;
    if(i>=36&&i<40)return (i-36)%2?1:-1;
    return 1;
}
std::vector<float> read_actor(const fs::path& root){
    const auto old=cp::read((root/"assets/ppo-four-axis-checkpoints/update-336-episode64/ppo-update-336.bin").string());
    need(cp::checksum(old)==2039539270511780368ull,"Wrong historical motor 336");
    const auto original=cudarl::rl::read_ppo_actor(old,14874979637633386686ull,1247037243u,2u);
    need(original.update==336&&original.parameters.size()==2500,"Wrong historical actor");
    auto migrated=cudarl::rl::append_zero_input_column(original.parameters);
    const auto bytes=cp::read((root/"live/research_data/learned-amplitude-v1.bin").string());
    need(cp::checksum(bytes)==4517507952728352494ull,"Wrong live motor package");
    cp::Reader in{bytes};
    for(uint32_t expected:{0x31564c48u,2u,6u,6u,3u,4u,41u,4u,336u,1u,7u,1u})
        need(in.value<uint32_t>()==expected,"Wrong package topology");
    in.value<uint64_t>();array<uint8_t>(in,64);const auto hashes=in.value<uint32_t>();
    for(uint32_t j=0;j<hashes;++j)array<uint8_t>(in,64);
    need(array<float>(in,2532)==migrated,"Live actor differs from original motor");
    for(int j=0;j<H;++j)need(migrated[j*D+40]==0,"Appended input has nonzero weights");
    return migrated;
}
struct Dataset{std::vector<Input> all,swim;std::array<int,4> pose_rows{};std::set<int> worlds;size_t long_rows=0,replayed_commands=0;double replay_error=0;};
Dataset read_states(const fs::path& root,const std::vector<float>& weights){
    const auto bytes=cp::read((root/"build/turn35/updates-1-8/batch-1.bin").string());
    need(cp::checksum(bytes)==0x3859bb0d0051cf03ull,"Wrong original-policy rollout");
    cp::Reader in{bytes};
    for(uint32_t v:{0x31425450u,2u,1u,64u})need(in.value<uint32_t>()==v,"Wrong archived rollout header");
    need(in.value<uint64_t>()==0&&in.value<uint64_t>()==14100886521839488064ull,"Wrong original rollout provenance");
    array<uint8_t>(in,1384*N);array<uint8_t>(in,360*N);
    need(array<float>(in,2532)==weights,"Collected actor differs from protected motor");
    array<float>(in,2433);const auto observations=array<float>(in,R*D);
    array<float>(in,R*A);for(int j=0;j<6;++j)array<float>(in,R);array<float>(in,N);
    const auto attempted=array<uint64_t>(in,R),substeps=array<uint64_t>(in,R);array<int>(in,R);
    need(in.value<uint32_t>()==192,"Wrong trace stride");const auto traces=array<uint8_t>(in,R*192);in.finish();
    Dataset data;
    for(int i=0;i<R;++i){
        need(attempted[i]==1&&substeps[i]==10,"Rejected physical source row");
        Input row;std::copy_n(observations.data()+i*D,D,row.begin());data.all.push_back(row);
        const auto* trace=traces.data()+i*192;
        const auto kind=field<uint32_t>(trace,48),control=field<uint32_t>(trace,44);
        if(kind==1&&control>=41){
            const int env=i%N;const auto sample=task::sample_request(env,0,0x504f4e44u,task::half_second_course_v1);
            need(sample.kind==1&&sample.start<4,"Source task sampling mismatch");
            data.swim.push_back(row);data.worlds.insert(env);++data.pose_rows[sample.start];
        }
    }
    need(data.swim.size()==2592&&data.worlds.size()==108,"Expected straight-swimming sample coverage");
    return data;
}
void add_long_swims(const fs::path& root,Dataset& data,Net& actor){
    std::ifstream input(root/"build/turn34/baseline0/controls.csv");
    need(bool(input),"Missing original long-swim evidence");
    const auto fields=[](const std::string& line){
        std::vector<std::string> out;std::stringstream row(line);std::string value;
        while(std::getline(row,value,','))out.push_back(value);return out;
    };
    std::string line;std::getline(input,line);const auto columns=fields(line);std::map<std::string,size_t> index;
    for(size_t i=0;i<columns.size();++i)index[columns[i]]=i;
    int previous_case=-1,previous_control=-1;Output previous_command{};
    while(std::getline(input,line)){
        const auto row=fields(line);if(row.empty()||row[0]!="long")continue;
        const auto value=[&](const std::string& name){return std::stof(row.at(index.at(name)));};
        const int pose=std::stoi(row.at(index.at("case"))),control=std::stoi(row.at(index.at("control")));
        if(previous_case==pose&&control==previous_control+1){
            for(int j=0;j<A;++j)data.replay_error=std::max(data.replay_error,std::abs(double(value("command"+std::to_string(j)))-previous_command[j]));
            ++data.replayed_commands;
        }
        p::BodyState body{};float held[4]{},rate[4]{},torque[4]{};cudarl::pond_motor_observation::RequestState request;
        need(cudarl::pond_motor_observation::initialize(pose,false,body,held,rate,torque,request),"Bad archived long fixture");
        body.position={value("px_model"),value("py_model"),value("pz_model")};
        body.orientation.x=value("qx");body.orientation.y=value("qy");body.orientation.z=value("qz");body.orientation.w=value("qw");
        for(int j=0;j<4;++j){body.angle[j]=value("joint"+std::to_string(j));held[j]=value("held"+std::to_string(j));rate[j]=value("reference_rate"+std::to_string(j));torque[j]=value("last_actual"+std::to_string(j));}
        for(int j=0;j<10;++j)body.velocity[j]=value("velocity"+std::to_string(j));
        request.controls=control;Input observation;
        need(cudarl::pond_motor_observation::observe(request,body,held,rate,torque,observation.data()),"Invalid reconstructed observation");
        actor.forward(observation.data(),1);for(int j=0;j<A;++j)previous_command[j]=std::tanh(actor.output()[j]);
        previous_case=pose;previous_control=control;
        if(control>=40&&control%4==0){data.swim.push_back(observation);data.all.push_back(observation);++data.long_rows;}
    }
    need(data.long_rows==582&&data.replayed_commands==2398&&data.replay_error<2e-6,"Reconstructed long-swim inputs fail saved action replay");
}
void header(std::ostream& out,const char* first){
    out<<first;
    for(const char* axis:{"front_yaw","front_pitch","rear_yaw","rear_pitch"})
        for(const char* stat:{"mean","rms","min","max"})out<<','<<axis<<'_'<<stat;
    out<<'\n'<<std::setprecision(10);
}
void saturation(const std::vector<float>& weights,const std::vector<Input>& inputs,const fs::path& dir){
    std::array<std::array<Moment,H>,2> hidden;
    std::array<std::array<size_t,H>,2> sat95{},sat99{};
    std::array<Moment,A> latent,command;std::array<size_t,A> output95{},output99{};
    auto net=model();net.params()=weights;
    for(const auto& input:inputs){
        std::vector<float> previous(input.begin(),input.end());
        for(int layer=0;layer<3;++layer){
            const int width=layer==2?A:H;std::vector<float> next(width);
            const auto& spec=net.spec();
            for(int j=0;j<width;++j){
                float z=weights[spec.bias_offset(layer)+j];
                for(size_t k=0;k<previous.size();++k)z+=previous[k]*weights[spec.weight_offset(layer)+j*previous.size()+k];
                next[j]=layer<2?std::tanh(z):z;
                if(layer<2){
                    hidden[layer][j].add(next[j]);sat95[layer][j]+=std::abs(next[j])>.95f;sat99[layer][j]+=std::abs(next[j])>.99f;
                }else{
                    latent[j].add(z);const float bounded=std::tanh(z);command[j].add(bounded);
                    output95[j]+=std::abs(bounded)>.95f;output99[j]+=std::abs(bounded)>.99f;
                }
            }
            previous=std::move(next);
        }
    }
    std::ofstream out(dir/"activations.csv");out<<"layer,neuron,mean,rms,min,max,fraction_abs_gt_0_95,fraction_abs_gt_0_99\n"<<std::setprecision(10);
    for(int l=0;l<2;++l)for(int j=0;j<H;++j){const auto& m=hidden[l][j];out<<l+1<<','<<j<<','<<m.mean()<<','<<m.rms()<<','<<m.lo<<','<<m.hi<<','<<double(sat95[l][j])/inputs.size()<<','<<double(sat99[l][j])/inputs.size()<<'\n';}
    for(int j=0;j<A;++j){
        const auto& m=command[j];out<<"command,"<<j<<','<<m.mean()<<','<<m.rms()<<','<<m.lo<<','<<m.hi<<','<<double(output95[j])/inputs.size()<<','<<double(output99[j])/inputs.size()<<'\n';
        const auto& z=latent[j];out<<"latent,"<<j<<','<<z.mean()<<','<<z.rms()<<','<<z.lo<<','<<z.hi<<",,\n";
    }
}
int main(int argc,char** argv)try{
    need(argc==3,"Use repository fresh-output-directory");
    const auto start=std::chrono::steady_clock::now();const fs::path root=fs::absolute(argv[1]),out=fs::absolute(argv[2]);
    const auto weights=read_actor(root);auto data=read_states(root,weights);
    auto actor=model();actor.params()=weights;add_long_swims(root,data,actor);
    need(!fs::exists(out)&&fs::create_directories(out),"Audit output already exists");
    std::array<Moment,D> ranges;for(const auto& row:data.all)for(int i=0;i<D;++i)ranges[i].add(row[i]);
    std::ofstream range(out/"input-ranges.csv");range<<"index,min,max,mean,first_layer_column_L2\n"<<std::setprecision(10);
    for(int i=0;i<D;++i){double norm=0;for(int h=0;h<H;++h)norm+=double(weights[h*D+i])*weights[h*D+i];range<<i<<','<<ranges[i].lo<<','<<ranges[i].hi<<','<<ranges[i].mean()<<','<<std::sqrt(norm)<<'\n';}
    const auto original=commands(actor,data.swim);
    std::ofstream sensitivity(out/"input-sensitivity.csv");header(sensitivity,"index,normalized_half_step");
    for(int i=0;i<D;++i){
        const float delta=i<3?.5f:.05f;auto positive=data.swim,negative=data.swim;
        for(size_t n=0;n<positive.size();++n){positive[n][i]+=delta;negative[n][i]-=delta;}
        const auto plus=commands(actor,positive),minus=commands(actor,negative);std::array<Moment,A> response;
        for(size_t n=0;n<plus.size();++n)for(int j=0;j<A;++j)response[j].add(plus[n][j]-minus[n][j]);
        sensitivity<<i<<','<<delta;values(sensitivity,response);sensitivity<<'\n';
    }
    std::ofstream heading(out/"heading-sensitivity.csv");header(heading,"variant,angle_degrees,sign");
    std::ofstream pairs(out/"heading-paired.csv");header(pairs,"variant,angle_degrees");
    for(const char* name:{"heading_only","velocity_only","coherent_request"})for(int degrees:{5,15,30,60,90}){
        std::array<std::vector<Output>,2> by_sign;
        for(int s:{-1,1}){
            auto changed=data.swim;
            for(auto& row:changed){
                const auto rotation=p::axis_rotation({row[9],row[10],row[11]},float(s*degrees*pi/180));
                const bool h=std::string(name)!="velocity_only",v=std::string(name)!="heading_only";
                if(h){const auto q=rotation.rotate({row[6],row[7],row[8]});row[6]=q.x;row[7]=q.y;row[8]=q.z;}
                if(v){const auto q=rotation.rotate({row[3],row[4],row[5]});row[3]=q.x;row[4]=q.y;row[5]=q.z;}
            }
            auto result=commands(actor,changed);std::array<Moment,A> response;
            for(size_t n=0;n<result.size();++n)for(int j=0;j<A;++j)response[j].add(result[n][j]-original[n][j]);
            heading<<name<<','<<degrees<<','<<s;values(heading,response);heading<<'\n';by_sign[s>0]=std::move(result);
        }
        std::array<Moment,A> response;
        for(size_t n=0;n<data.swim.size();++n)for(int j=0;j<A;++j)response[j].add(by_sign[1][n][j]-by_sign[0][n][j]);
        pairs<<name<<','<<degrees;values(pairs,response);pairs<<'\n';
    }
    auto reflected=data.swim;for(auto& row:reflected)for(int i=0;i<D;++i)row[i]*=reflection(i);
    const auto mirrored=commands(actor,reflected);std::array<Moment,A> mirror_error;
    for(size_t n=0;n<original.size();++n)for(int j=0;j<A;++j)mirror_error[j].add(mirrored[n][j]-(j%2?1:-1)*original[n][j]);
    std::ofstream mirror(out/"mirror-sensitivity.csv");header(mirror,"states");mirror<<data.swim.size();values(mirror,mirror_error);mirror<<'\n';
    saturation(weights,data.swim,out);
    std::ofstream receipt(out/"audit.json");receipt<<std::setprecision(12)<<"{\"historical_motor_update\":336,\"historical_parameters\":2500,\"live_actor_parameters\":2532,\"actor_inputs\":41,\"historical_actor_and_live_zero_column_equal\":true,\"checkpoint_payload_checksum\":\"2039539270511780368\",\"package_payload_checksum\":\"4517507952728352494\",\"raw_source_rows\":"<<data.all.size()<<",\"sensitivity_states\":"<<data.swim.size()<<",\"distinct_straight_worlds\":"<<data.worlds.size()<<",\"preaction_controls\":[41,64],\"state_time_seconds\":[2,3.15],\"pose_rows\":["<<data.pose_rows[0]<<','<<data.pose_rows[1]<<','<<data.pose_rows[2]<<','<<data.pose_rows[3]<<"],\"measured_axes\":[\"front_yaw\",\"front_pitch\",\"rear_yaw\",\"rear_pitch\"],\"source_policy\":\"original_motor_336_before_any_optimizer_update\",\"measured_output\":\"tanh_of_actor_latent_mean_before_filter_and_servo\",\"new_physics_controls\":0,\"optimizer_calls\":0,\"elapsed_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
    std::ofstream provenance(out/"long-swim-replay.json");provenance<<std::setprecision(12)<<"{\"additional_long_states\":"<<data.long_rows<<",\"saved_commands_replayed\":"<<data.replayed_commands<<",\"maximum_absolute_normalized_command_error\":"<<data.replay_error<<",\"additional_physics_steps\":0,\"optimizer_updates\":0}\n";
    std::cout<<"Audited "<<data.swim.size()<<" preserved swimming states and all 41 inputs, no simulation or optimization.\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
