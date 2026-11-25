#include "rl/ppo_kernels.h"
#include "cuda/cuda_memory.h"
#undef CUDARL_RL_PPO_MATH_H
#define rl rl_capacity8_preserved
#include "preserved_ppo_math.h"
#undef rl

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace r=cudarl::rl;
namespace old=cudarl::rl_capacity8_preserved;
constexpr int rows=4096;
constexpr float sentinel=12345.25f;
struct Result {int k=0,mismatched_samples=0;double softmax=0,logp=0,gradient=0,value=0,gradient_sum=0,distribution=0;bool canaries=true,legacy_exact=true;};
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class T>std::vector<T> host(const device_ptr<T>& data,size_t size){std::vector<T> values(size);data.copy_to_host(values.data(),size);return values;}
__global__ void full_softmax(const float* logits,int n,int k,float* logp,int* legacy_differences){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    r::log_softmax(logits+i*k,k,logp+i*k);
    if(k==3){float previous[8],a[16],b[8],va,vb;old::log_softmax(logits+i*k,k,previous);
        for(int j=0;j<k;++j)if(__float_as_uint(previous[j])!=__float_as_uint(logp[i*k+j]))atomicAdd(legacy_differences,1);
        r::ppo_row(logits+i*k,k,1,previous[1],.75f,.2f,.8f,r::PpoCoefficients{.2f,.01f,.5f},1.f,a,&va);
        old::ppo_row(logits+i*k,k,1,previous[1],.75f,.2f,.8f,old::PpoCoefficients{.2f,.01f,.5f},1.f,b,&vb);
        for(int j=0;j<k;++j)if(__float_as_uint(a[j])!=__float_as_uint(b[j]))atomicAdd(legacy_differences,1);
        if(__float_as_uint(va)!=__float_as_uint(vb))atomicAdd(legacy_differences,1);}
}
Result exercise(int k,bool uniform){Result result;result.k=k;
    std::vector<float> logits(rows*k+2,sentinel),single(rows+2,sentinel),all(rows*k+2,sentinel),rowstats(rows*r::kPpoStats+2,sentinel),stats(r::kPpoStats+2,sentinel);
    std::vector<int> chosen(rows+2,12345);std::vector<float> values(rows,.2f),advantages(rows,.75f),returns(rows,.8f);
    for(int i=0;i<rows;++i)for(int j=0;j<k;++j)logits[1+i*k+j]=uniform?0.f:float(j-k/2)*.2f+float(i%7)*.03f;
    for(int j=0;j<r::kPpoStats;++j)stats[1+j]=0;
    device_ptr<float> dl(logits.size()),dp(single.size()),full(all.size()),gradient(all.size()),dv(single.size()),drs(rowstats.size()),ds(stats.size());
    device_ptr<float> dvalues(rows),dadv(rows),dreturns(rows),advstats(2);device_ptr<int> da(chosen.size()),dlegacy(1);
    dl.copy_from_host(logits.data(),logits.size());dp.copy_from_host(single.data(),single.size());full.copy_from_host(all.data(),all.size());gradient.copy_from_host(all.data(),all.size());dv.copy_from_host(single.data(),single.size());drs.copy_from_host(rowstats.data(),rowstats.size());ds.copy_from_host(stats.data(),stats.size());da.copy_from_host(chosen.data(),chosen.size());
    dvalues.copy_from_host(values.data(),rows);dadv.copy_from_host(advantages.data(),rows);dreturns.copy_from_host(returns.data(),rows);const float normalize[]{0.f,1.f};advstats.copy_from_host(normalize,2);check(cudaMemset(dlegacy.get(),0,sizeof(int)));
    full_softmax<<<(rows+127)/128,128>>>(dl.get()+1,rows,k,full.get()+1,dlegacy.get());check(cudaGetLastError());
    r::sample_actions(dl.get()+1,rows,k,20261124,99,false,da.get()+1,dp.get()+1,nullptr);
    r::ppo_loss_grad(dl.get()+1,dvalues.get(),rows,k,da.get()+1,dp.get()+1,dadv.get(),dreturns.get(),advstats.get(),{.2f,.01f,.5f},gradient.get()+1,dv.get()+1,drs.get()+1,ds.get()+1,nullptr);
    check(cudaDeviceSynchronize());const auto gfull=host(full,all.size()),glogp=host(dp,single.size()),ggradient=host(gradient,all.size()),gvalue=host(dv,single.size()),grs=host(drs,rowstats.size()),gs=host(ds,stats.size());const auto action=host(da,chosen.size());const auto previous=host(dlegacy,1);
    result.legacy_exact=previous[0]==0;std::vector<int> histogram(k,0);
    for(const auto* a:{&gfull,&glogp,&ggradient,&gvalue,&grs,&gs})result.canaries=result.canaries&&a->front()==sentinel&&a->back()==sentinel;
    result.canaries=result.canaries&&action.front()==12345&&action.back()==12345;
    for(int i=0;i<rows;++i){float lp[16],grad[16],value; r::log_softmax(logits.data()+1+i*k,k,lp);
        const int expected=r::sample_action(lp,k,cudarl::rng::uniform(20261124,99,i),false);
        const int actual=action[1+i];require(actual>=0&&actual<k,"Sample outside supported action range");++histogram[actual];result.mismatched_samples+=actual!=expected;
        r::ppo_row(logits.data()+1+i*k,k,actual,glogp[1+i],.75f,.2f,.8f,{.2f,.01f,.5f},1.f/rows,grad,&value);
        result.logp=std::max(result.logp,std::abs(double(glogp[1+i])-lp[actual]));result.value=std::max(result.value,std::abs(double(gvalue[1+i])-value));double sum=0,probability=0;
        for(int j=0;j<k;++j){const int index=1+i*k+j;require(std::isfinite(gfull[index])&&std::isfinite(ggradient[index]),"Nonfinite probability or gradient");result.softmax=std::max(result.softmax,std::abs(double(gfull[index])-lp[j]));result.gradient=std::max(result.gradient,std::abs(double(ggradient[index])-grad[j]));sum+=ggradient[index];probability+=std::exp(gfull[index]);}
        result.gradient_sum=std::max(result.gradient_sum,std::abs(sum));require(std::abs(probability-1)<2e-6,"Probabilities do not sum to one");
        if(uniform)require(std::abs(double(glogp[1+i])+std::log(double(k)))<1e-6,"Uniform categorical log probability is wrong");
    }
    float lp[16];r::log_softmax(logits.data()+1,k,lp);for(int j=0;j<k;++j)result.distribution=std::max(result.distribution,std::abs(double(histogram[j])/rows-std::exp(lp[j])));
    require(result.softmax<=1e-6&&result.logp<=1e-6&&result.gradient<=1e-7&&result.value<=1e-7&&result.gradient_sum<=1e-7,"CPU/CUDA categorical math disagreement");
    require(result.mismatched_samples==0&&result.distribution<.035&&result.canaries&&result.legacy_exact,"Sampling, bounds canaries or old arithmetic failed");return result;
}
int main(int argc,char** argv){const auto start=std::chrono::steady_clock::now();bool pass=true,rejected_sample=false,rejected_loss=false;std::string failure;std::vector<Result> results;
    try{require(argc==2,"Usage: pond_ppo_capacity_check NEW_REPORT_JSON");for(int k:{3,8,9,15,16}){results.push_back(exercise(k,false));results.push_back(exercise(k,true));}
        try{r::sample_actions(nullptr,1,17,0,0,false,nullptr,nullptr,nullptr);}catch(const std::invalid_argument&){rejected_sample=true;}
        try{r::ppo_loss_grad(nullptr,nullptr,1,17,nullptr,nullptr,nullptr,nullptr,nullptr,{},nullptr,nullptr,nullptr,nullptr,nullptr);}catch(const std::invalid_argument&){rejected_loss=true;}
        require(rejected_sample&&rejected_loss,"Unsupported17-action shapes were not rejected before CUDA launch");
    }catch(const std::exception& error){pass=false;failure=error.what();}
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();if(argc>1){std::ofstream out(argv[1]);out<<std::setprecision(12)<<"{\"pass\":"<<(pass?"true":"false")<<",\"physics_steps\":0,\"optimizer_updates\":0,\"capacity\":"<<r::kMaxActions<<",\"rejected17_sample\":"<<(rejected_sample?"true":"false")<<",\"rejected17_loss\":"<<(rejected_loss?"true":"false")<<",\"failure\":\""<<failure<<"\",\"wall_s\":"<<seconds<<",\"cases\":[";
        for(size_t i=0;i<results.size();++i){const auto& r=results[i];if(i)out<<',';out<<"{\"actions\":"<<r.k<<",\"uniform\":"<<(i%2?"true":"false")<<",\"rows\":"<<rows<<",\"softmax_error\":"<<r.softmax<<",\"sample_logp_error\":"<<r.logp<<",\"gradient_error\":"<<r.gradient<<",\"gradient_sum_error\":"<<r.gradient_sum<<",\"sample_mismatches\":"<<r.mismatched_samples<<",\"distribution_error\":"<<r.distribution<<",\"canaries\":"<<(r.canaries?"true":"false")<<",\"old3_arithmetic_exact\":"<<(r.legacy_exact?"true":"false")<<"}";}out<<"]}\n";}
    std::cout<<(pass?"PASS":"FAIL")<<" capacity="<<r::kMaxActions<<" cases="<<results.size()<<" wall_s="<<seconds<<' '<<failure<<'\n';return pass?0:1;
}
