// Read-only PPO surrogate and command probes on the final archived v6 rollout.
#include "core/checkpoint.h"
#include "nn/mlp_cpu.h"
#include "physics/dynamics.h"
#include "rl/gaussian_math.h"
#include "rl/pond_ppo_contract.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace cp=cudarl::checkpoint;
namespace nn=cudarl::nn;
namespace rl=cudarl::rl;
namespace pp=rl::pond_ppo;
namespace p=cudarl::physics;
namespace fs=std::filesystem;
using Activation=nn::Activation;
using Net=nn::CpuMlp<float>;
constexpr int N=pp::worlds,T=pp::controls_per_world,R=N*T,D=42,H=32,A=4;
static void check(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
template <class T> std::vector<T> array(cp::Reader& r,int n){
    auto v=r.array<T>();check(v.size()==size_t(n),"Unexpected array dimensions");return v;
}
template <class T> T field(const uint8_t* row,int offset) {
    T v;std::memcpy(&v,row+offset,sizeof(v));return v;
}
struct Numbers {
    double sum=0,ss=0,peak=0;
    int n=0,sat=0;
    void add(double x) {
        check(std::isfinite(x),"Nonfinite statistic");
        ++n;sum+=x;ss+=x*x;peak=std::max(peak,std::abs(x));sat+=std::abs(x)>.95;
    }
    void print(bool bounded=false)const {
        std::cout<<"{\"mean\":"<<sum/n<<",\"rms\":"<<sqrt(ss/n)<<",\"max_abs\":"<<peak;
        if(bounded)std::cout<<",\"abs_gt_095_fraction\":"<<double(sat)/n;
        std::cout<<'}';
    }
};
template<size_t K> void output(const char* key,const std::array<Numbers,K>& a,bool bounded=false) {
    std::cout<<",\""<<key<<"\":[";
    for(size_t j=0;j<K;++j){if(j)std::cout<<',';a[j].print(bounded);}std::cout<<']';
}
struct Group {
    int rows=0, clipped=0, all_zero=0;
    std::array<Numbers,2> hidden;
    std::array<Numbers,A> mean,command,score_derivative,heading,mirror;
    std::array<std::array<double,A>,N> world_derivative{};
    std::array<int,N> world_count{};
    std::array<double,32> feature_grad{};
    std::array<std::array<double,H>,A> head_grad{};
    std::vector<double> network_grad;
    void print(const char* name)const {
        std::cout<<'"'<<name<<"\":{\"rows\":"<<rows<<",\"ratio_outside_clip\":"<<clipped<<",\"zero_policy_gradient_rows\":"<<all_zero;
        output("hidden_tanh",hidden,true);output("latent_mean",mean);output("physical_mean_command",command,true);
        output("PPO_dloss_dmean_per_row",score_derivative);
        if(heading[0].n){output("plus_minus15deg_command_difference",heading);output("mirrored_command_error",mirror);}
        const auto squared=[](auto& x){return std::inner_product(x.begin(),x.end(),x.begin(),0.);};
        std::cout<<",\"net_weight_gradient_L2\":"<<sqrt(squared(network_grad))/rows;
        std::cout<<",\"signed_course_input_column_gradient_L2\":"<<sqrt(squared(feature_grad))/rows<<",\"output_row_weight_gradient_L2\":[";
        for(int j=0;j<A;++j){if(j)std::cout<<',';std::cout<<sqrt(squared(head_grad[j]))/rows;}
        std::cout<<"],\"world_mean_dloss_dmean_SE\":[";
        for(int j=0;j<A;++j){
            double sum=0,ss=0;int n=0;
            for(int e=0;e<N;++e)if(world_count[e]){
                const double v=world_derivative[e][j]/world_count[e];sum+=v;ss+=v*v;++n;
            }
            check(n>1,"Insufficient world clusters");if(j)std::cout<<',';
            std::cout<<sqrt(std::max(0.,ss-sum*sum/n)/(n*(n-1.)));
        }
        std::cout<<"]}";
    }
};
static int parity(int i){
    if(i<15)return i%3==2?-1:1;
    if(i<18)return i==17?1:-1;
    if(i<34)return (i-18)%2?1:-1;
    if(i>=36&&i<40)return (i-36)%2?1:-1;
    return i==41?-1:1;
}
static void intent_probes(Net& actor,const float* x,const float* original,Group& g) {
    std::array<float,D> changed{};
    float targets[2][4];
    for(int j=0;j<D;++j)changed[j]=parity(j)*x[j];
    actor.forward(changed.data(),1);
    for(int j=0;j<A;++j)g.mirror[j].add(tanhf(actor.output()[j])-(j%2?1:-1)*tanhf(original[j]));
    for(int s:{-1,1}) {
        std::copy_n(x,D,changed.begin());
        const p::V up{x[9],x[10],x[11]};
        const auto rotation=p::axis_rotation(up,s*.261799388f);
        for(int k:{3,6}) {
            const auto value=rotation.rotate({x[k],x[k+1],x[k+2]});
            changed[k]=value.x;changed[k+1]=value.y;changed[k+2]=value.z;
        }
        // Keep the measured COM trajectory and change the persistent request.
        changed[41]=std::remainder(x[41]+s/12.f,2.f);
        actor.forward(changed.data(),1);
        for(int j=0;j<A;++j)targets[s>0][j]=tanhf(actor.output()[j]);
    }
    for(int j=0;j<A;++j)g.heading[j].add(targets[1][j]-targets[0][j]);
}
int main(int argc,char** argv)try {
    check(argc==2,"Use archived v6 study directory");
    const auto start=std::chrono::steady_clock::now();
    const fs::path root=argv[1];
    const auto bytes=cp::read((root/"updates-1-8/batch-8.bin").string());
    check(cp::checksum(bytes)==0x366965649ff2c4aeull,"Wrong v6 batch");
    auto data=cp::read((root/"updates-1-8/candidate-8.bin").string());
    check(cp::checksum(data)==0xcf80b83efa6e6a91ull,"Wrong candidate");
    constexpr uint64_t source=370545787127571638ull;
    auto candidate=pp::read_candidate(data,4517507952728352494ull,16817206019680038892ull,source,1384,368,D);
    cp::Reader r{bytes};
    for(uint32_t expected:{0x31425450u,3u,8u,64u})check(r.value<uint32_t>()==expected,"Invalid header");
    r.value<uint64_t>();check(r.value<uint64_t>()==source,"Invalid source");
    array<uint8_t>(r,1384*N);array<uint8_t>(r,368*N);
    const auto old_weights=array<float>(r,pp::actor_parameter_count(D));array<float>(r,pp::critic_weights);
    const auto x=array<float>(r,R*D),latent=array<float>(r,R*A),logp=array<float>(r,R),v=array<float>(r,R);
    const auto reward=array<float>(r,R),done=array<float>(r,R),trunc=array<float>(r,R),end=array<float>(r,R),last=array<float>(r,N);
    array<uint64_t>(r,R);array<uint64_t>(r,R);array<int>(r,R);
    check(r.value<uint32_t>()==192,"Trace ABI");auto traces=array<uint8_t>(r,R*192);r.finish();
    std::vector<float> adv(R),ret(R);
    for(int e=0;e<N;++e)rl::gae_one_env(T,N,e,reward.data(),v.data(),done.data(),trunc.data(),end.data(),last.data(),pp::gamma,pp::lambda,adv.data(),ret.data());
    const double mean=std::accumulate(adv.begin(),adv.end(),0.)/R;
    double variance=0;for(auto value:adv)variance+=(value-mean)*(value-mean);const float stddev=float(sqrt(variance/R));
    const nn::MlpSpec spec{{D,H,H,A},{Activation::Tanh,Activation::Tanh,Activation::Identity}};
    Net actor(spec),old(spec),first({{D,H},{Activation::Tanh}}),second({{D,H,H},{Activation::Tanh,Activation::Tanh}});
    actor.params()=candidate.actor;old.params()=old_weights;
    std::copy_n(actor.params().begin(),first.params().size(),first.params().begin());
    std::copy_n(actor.params().begin(),second.params().size(),second.params().begin());
    std::array<Group,4> groups;
    for(auto& g:groups)g.network_grad.resize(actor.params().size());
    double max_logp_error=0;
    for(int row=0;row<R;++row) {
        const uint8_t* tr=traces.data()+192*row;const auto kind=field<uint32_t>(tr,48);
        check(kind<=2,"Unknown task");
        const auto i=kind<2?kind:(field<float>(tr,108)>0?2u:3u);
        auto& g=groups[i];++g.rows;++g.world_count[row%N];
        const auto* input=x.data()+row*D;
        const float* sample=latent.data()+row*A;
        old.forward(input,1);
        max_logp_error=std::max(max_logp_error,fabs(double(rl::gaussian_logp(old.output(),candidate.log_std.data(),sample,A))-logp[row]));
        actor.forward(input,1);first.forward(input,1);second.forward(input,1);
        float mu[A],dy[A],dstd[A],dv=0;
        std::copy_n(actor.output(),A,mu);
        const auto terms=rl::ppo_gaussian_row(mu,candidate.log_std.data(),A,sample,logp[row],
            float((adv[row]-mean)/(stddev+1e-8)),v[row],ret[row],{pp::clip,0,.5f},1.f,dy,dstd,&dv);
        g.clipped+=int(terms.clipped);g.all_zero+=std::all_of(dy,dy+A,[](float y){return y==0;});
        actor.zero_grad();actor.backward(dy,1);const auto& weights=actor.grads();
        for(size_t j=0;j<weights.size();++j)g.network_grad[j]+=weights[j];
        for(int h=0;h<H;++h) {
            g.hidden[0].add(first.output()[h]);g.hidden[1].add(second.output()[h]);
            g.feature_grad[h]+=weights[h*D+D-1];
        }
        const float gain=kind==0?candidate.frozen_gain:1;
        for(int j=0;j<A;++j) {
            g.mean[j].add(mu[j]);g.command[j].add(tanhf(gain*mu[j]));
            g.score_derivative[j].add(dy[j]);g.world_derivative[row%N][j]+=dy[j];
            for(int h=0;h<H;++h)g.head_grad[j][h]+=weights[spec.weight_offset(2)+j*H+h];
        }
        if(kind==2&&input[41]!=0)intent_probes(actor,input,mu,g);
    }
    check(max_logp_error<.001,"CPU reference likelihood drift");
    std::cout<<std::setprecision(9)<<"{\"candidate_sha256\":\"9b68920b345b4be6699f8b979729e07d9f9d7a451664e4ca99f34487f2ca4ded\",\"batch_sha256\":\"2b5555a86e2a2fbcf56a0d476a5341f30c1de7eb36b8b9a07de3dde5c315c413\",\"checkpoint\":8,\"collection_checkpoint\":7,\"rows\":"<<R
        <<",\"axis_order\":[\"front_yaw\",\"front_pitch\",\"rear_yaw\",\"rear_pitch\"],\"latent_sigma\":"<<pp::sigma
        <<",\"global_raw_advantage_mean\":"<<mean<<",\"global_raw_advantage_std\":"<<stddev
        <<",\"old_cpu_replay_max_abs_logp_error\":"<<max_logp_error
        <<",\"method\":\"Full clipped PPO surrogate gradient at checkpoint 8 on archived batch 8. Native GAE, recorded collection likelihood, global population advantage normalization, no optimizer. Per-row output derivatives use unit sample scale. Weight norms are gradients of each task mean loss. SE uses independent world clusters. Heading probe rotates heading and velocity by +/-15 degrees around world up at one measured gait state and updates signed error. Mirror error compares reflected input output with reflected original output. Saturation means absolute value greater than 0.95.\",\"tasks\":{";
    const char* names[]={"rest","straight","left_turn","right_turn"};
    for(int i=0;i<4;++i){if(i)std::cout<<',';groups[i].print(names[i]);}
    std::cout<<"},\"analysis_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
             <<",\"physics_steps\":0,\"optimizer_steps\":0}\n";
    return 0;
}catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}
