#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include "cuda/device.h"
#include "cuda/cuda_memory.h"
#include "rl/pond_selector_policy.h"
#include "rl/ppo_kernels.h"
#include "nn/mlp_gpu.h"
namespace d=cudarl::deep;
namespace p=cudarl::deep_pond;
namespace ps=cudarl::pond_selector;
namespace rl=cudarl::rl;
struct Condition {std::string name;d::V current;};
struct Case {
    int foods=0,slots[3]{},low_ground=0,low_relative=0,no_food_10=0,max_gap=0,invalid=0;
    double ground_speed=0,relative_speed=0;
};
struct Trial {uint32_t seed;int environment;Case result;};
__global__ void initialize(p::State* states,float* obs,int* targets,int n,uint32_t seed) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    states[i]=p::initial(seed,static_cast<uint32_t>(i));targets[i]=-1;ps::observe(states[i],-1,obs+i*36);
}
__global__ void selector_inputs(const p::State* states,const int* targets,float* obs,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) ps::observe(states[i],targets[i],obs+i*36);
}
__global__ void motor_inputs(const p::State* states,const int* targets,float* obs,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    p::observe(states[i],obs+i*33);cudarl::demos::primary_pond_food(obs+i*33,targets[i]);
}
__global__ void advance_current(p::State* states,const float* actions,Case* cases,int n,d::V current) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    auto s=states[i];auto c=cases[i];const int gap=s.since_food+1;
    const auto result=p::advance(s,actions+i*4,current);
    if(result.eaten_slot>=0) {++c.foods;++c.slots[result.eaten_slot];}
    const d::V velocity{s.body.velocity[0],s.body.velocity[1],s.body.velocity[2]};
    const float speed=d::norm(velocity),relative=d::norm(velocity-current);
    c.ground_speed+=speed;c.relative_speed+=relative;
    if(speed<.005f) ++c.low_ground;if(relative<.005f) ++c.low_relative;
    if(s.since_food>=200) ++c.no_food_10;c.max_gap=gap>c.max_gap?gap:c.max_gap;
    bool valid=isfinite(s.body.position.x) && isfinite(s.body.position.y) && isfinite(s.body.position.z) &&
        isfinite(s.body.orientation.x) && isfinite(s.body.orientation.y) && isfinite(s.body.orientation.z) && isfinite(s.body.orientation.w);
    for(int j=0;j<d::dofs;++j) valid=valid && isfinite(s.body.velocity[j]);
    for(int j=0;j<4;++j) valid=valid && isfinite(s.body.angle[j]);
    if(!valid) ++c.invalid;states[i]=s;cases[i]=c;
}
std::vector<Case> evaluate(const rl::SelectorPolicy& policy,int n,uint32_t seed,int ticks,d::V current) {
    cudarl::nn::GpuMlp motor(rl::swimmer_specification(4,33),n,7),selector(rl::selector_spec(),n*3,7);
    motor.set_params(policy.motor);selector.set_params(policy.selector);
    device_ptr<p::State> states(n);device_ptr<Case> metrics(n);device_ptr<int> targets(n);
    device_ptr<float> obs(n*36),features(n*3*32),motor_obs(n*33),logp(n);
    CUDA_CHECK_THROW(cudaMemset(metrics.get(),0,metrics.size_bytes()));
    initialize<<<(n+63)/64,64>>>(states.get(),obs.get(),targets.get(),n,seed);CUDA_CHECK_THROW(cudaGetLastError());
    for(int tick=0;tick<ticks;++tick) {
        if(tick%policy.decision_steps==0) {
            selector_inputs<<<(n+63)/64,64>>>(states.get(),targets.get(),obs.get(),n);CUDA_CHECK_THROW(cudaGetLastError());
            ps::make_features(obs.get(),features.get(),n,nullptr);selector.forward(features.get(),n*3,nullptr);
            rl::sample_actions(selector.output(),n,3,0,0,true,targets.get(),logp.get(),nullptr);
        }
        motor_inputs<<<(n+63)/64,64>>>(states.get(),targets.get(),motor_obs.get(),n);CUDA_CHECK_THROW(cudaGetLastError());
        motor.forward(motor_obs.get(),n,nullptr);
        advance_current<<<(n+63)/64,64>>>(states.get(),motor.output(),metrics.get(),n,current);CUDA_CHECK_THROW(cudaGetLastError());
    }
    CUDA_CHECK_THROW(cudaDeviceSynchronize());std::vector<Case> result(n);metrics.copy_to_host(result.data(),n);
    if(motor.get_params()!=policy.motor || selector.get_params()!=policy.selector) throw std::runtime_error("Evaluation changed neural weights");
    for(const auto& c:result) if(c.invalid) throw std::runtime_error("Non-finite physical state in current evaluation");
    return result;
}
std::vector<Condition> conditions() {
    std::vector<Condition> out{{"calm",{}}};
    const char* names[8]={"px","nx","pz","nz","px_pz","px_nz","nx_pz","nx_nz"};
    const d::V directions[8]={{1,0,0},{-1,0,0},{0,0,1},{0,0,-1},{1,0,1},{1,0,-1},{-1,0,1},{-1,0,-1}};
    for(int strength=0;strength<2;++strength) for(int j=0;j<8;++j) {
        const float speed=strength==0?.005f:.01f;
        out.push_back({std::string(strength==0?"cm05_":"cm10_")+names[j],directions[j]*(speed/d::norm(directions[j]))});
    }
    return out;
}
#include "pond_current_diagnose.h"
#include "pond_progress_eval.h"
int main(int argc,char** argv) {
    try {
        std::string path,output,selection,progress_path,fixture_path;int n=64,seconds=60;std::vector<uint32_t> seeds;bool diagnose=false;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--diagnose") {diagnose=true;continue;}
            if(arg=="--help") {std::puts("pond_current_eval --policy FROZEN_BUNDLE --output DIR [--seed N (repeatable)] [--trials 64] [--seconds 60] [--conditions calm,cm05_px,...]\nRead-only GPU evaluation; no training, exploration, new observations or episode resets. Default: 2 seeds, calm + 0.5/1.0 cm/s in 8 directions.");return 0;}
            if(i+1>=argc) throw std::invalid_argument("Missing argument value");const std::string v=argv[++i];
            if(arg=="--policy") path=v;else if(arg=="--output") output=v;else if(arg=="--seed") seeds.push_back(static_cast<uint32_t>(std::stoul(v,nullptr,0)));
            else if(arg=="--progress") progress_path=v;else if(arg=="--fixture") fixture_path=v;
            else if(arg=="--trials") n=std::stoi(v);else if(arg=="--seconds") seconds=std::stoi(v);else if(arg=="--conditions") selection=v;
            else throw std::invalid_argument("Unknown argument: "+arg);
        }
        if((path.empty() && progress_path.empty()) || output.empty() || n<2 || n>4096 || seconds<1 || seconds>3600) throw std::invalid_argument("Supply policy/output, 2..4096 trials per seed and 1..3600 seconds");
        if(diagnose) return current_diagnosis::run_current_diagnosis(path,output);
        if(seeds.empty()) seeds={0xC071A101u,0xC071A102u};
        auto suite=conditions();
        if(!selection.empty()) {
            std::vector<std::string> names;std::stringstream parts(selection);std::string item;
            while(std::getline(parts,item,',')) names.push_back(item);
            for(const auto& name:names) if(std::none_of(suite.begin(),suite.end(),[&](const auto& c){return c.name==name;})) throw std::invalid_argument("Unknown current condition: "+name);
            suite.erase(std::remove_if(suite.begin(),suite.end(),[&](const auto& c){return c.name!="calm" && std::find(names.begin(),names.end(),c.name)==names.end();}),suite.end());
        }
        const auto device=cudarl::probe_device();if(!device.available) throw std::runtime_error(device.error);
        if(!progress_path.empty()) return progress_evaluation::evaluate(progress_path,output,n,seconds,seeds,suite,fixture_path);
        const auto original=cudarl::checkpoint::read(path);const auto policy=rl::SelectorPolicy::load(path);
        const int ticks=seconds*20;const double rate=60./seconds;
        std::filesystem::create_directories(output);std::vector<std::vector<Trial>> results(suite.size());
        for(const uint32_t seed:seeds) for(size_t condition=0;condition<suite.size();++condition) {
            const auto cases=evaluate(policy,n,seed,ticks,suite[condition].current);int foods=0;
            for(int i=0;i<n;++i) {results[condition].push_back({seed,i,cases[i]});foods+=cases[i].foods;}
            std::printf("CURRENT seed=%u condition=%s cm_s=%.2f trials=%d seconds=%d fpm=%.6f\n",seed,suite[condition].name.c_str(),d::norm(suite[condition].current)*100,n,seconds,foods*rate/n);std::fflush(stdout);
        }
        std::ofstream rows(std::filesystem::path(output)/"cases.csv"),summary(std::filesystem::path(output)/"summary.csv"),manifest(std::filesystem::path(output)/"protocol.txt");
        if(!rows || !summary || !manifest) throw std::runtime_error("Could not write evaluation outputs");
        rows<<"condition,current_cm_s,current_x_m_s,current_z_m_s,seed,environment,seconds,foods,slot0,slot1,slot2,low_ground_seconds,low_water_relative_seconds,no_food_10_seconds,max_food_gap_seconds,mean_ground_cm_s,mean_water_relative_cm_s\n";
        summary<<"condition,current_cm_s,trials,seconds,food_per_minute,delta_vs_calm,paired_ci_low,paired_ci_high,win,tie,loss,mean_low_ground_seconds,mean_low_water_relative_seconds,mean_no_food_10_seconds,mean_max_food_gap_seconds,max_food_gap_seconds,worlds_gap_over_20s,zero_food_worlds,mean_ground_cm_s,mean_water_relative_cm_s\n";
        for(size_t condition=0;condition<suite.size();++condition) {
            const auto& c=suite[condition];const auto& trials=results[condition];const size_t count=trials.size();
            double food=0,delta=0,variance=0,low=0,relative_low=0,no_food=0,gap=0,max_gap=0,speed=0,relative_speed=0;
            int win=0,tie=0,loss=0,stalled=0,zero=0;
            for(size_t i=0;i<count;++i) {
                const auto& t=trials[i];const auto& r=t.result;const int difference=r.foods-results[0][i].result.foods;
                food+=r.foods;delta+=difference;if(difference>0) ++win;else if(difference<0) ++loss;else ++tie;
                low+=r.low_ground*.05;relative_low+=r.low_relative*.05;no_food+=r.no_food_10*.05;gap+=r.max_gap*.05;
                max_gap=std::max(max_gap,r.max_gap*.05);if(r.max_gap>400) ++stalled;if(r.foods==0) ++zero;
                speed+=r.ground_speed/ticks*100;relative_speed+=r.relative_speed/ticks*100;
                rows<<std::setprecision(10)<<c.name<<','<<d::norm(c.current)*100<<','<<c.current.x<<','<<c.current.z<<','<<t.seed<<','<<t.environment<<','<<seconds<<','<<r.foods;
                for(int slot:r.slots) rows<<','<<slot;
                rows<<','<<r.low_ground*.05<<','<<r.low_relative*.05<<','<<r.no_food_10*.05<<','<<r.max_gap*.05<<','<<r.ground_speed/ticks*100<<','<<r.relative_speed/ticks*100<<'\n';
            }
            delta/=count;
            for(size_t i=0;i<count;++i) {const double error=trials[i].result.foods-results[0][i].result.foods-delta;variance+=error*error;}
            const double ci=rate*1.96*std::sqrt(variance/(count-1)/count);delta*=rate;
            summary<<std::setprecision(10)<<c.name<<','<<d::norm(c.current)*100<<','<<count<<','<<seconds<<','<<food*rate/count<<','<<delta<<','<<delta-ci<<','<<delta+ci<<','<<win<<','<<tie<<','<<loss<<','<<low/count<<','<<relative_low/count<<','<<no_food/count<<','<<gap/count<<','<<max_gap<<','<<stalled<<','<<zero<<','<<speed/count<<','<<relative_speed/count<<'\n';
            std::printf("SUMMARY %s fpm=%.6f paired_delta=%.6f ci=[%.6f,%.6f] gap_over20=%d/%zu\n",c.name.c_str(),food*rate/count,delta,delta-ci,delta+ci,stalled,count);
        }
        if(original!=cudarl::checkpoint::read(path)) throw std::runtime_error("The frozen policy file changed");
        manifest<<"device="<<device.name<<"\nselector_update="<<policy.update<<"\nmotor_update="<<policy.motor_update<<"\nseconds="<<seconds<<"\ntrials_per_seed="<<n<<"\nseeds=";
        for(uint32_t seed:seeds) manifest<<seed<<',';
        manifest<<"\nSame initial layouts for each seed/environment. Subsequent food placements depend on collection history.\n"
            <<"Uniform imposed horizontal water velocity acts through relative distributed drag. Food stays fixed until collection.\n"
            <<"Reduced-order fluid model, not CFD or a mass-conserving closed-pond circulation solver.\n"
            <<"Observations, model dimensions, weights, decision cadence, geometry, food rules and 5ms substeps unchanged.\n"
            <<"No gradients, optimizer, training save access, autoreset, rescue or scripted propulsion.\n"
            <<"Paired approximate 95% intervals are per condition, not corrected for testing multiple directions.\n"
            <<"Ground speed and water-relative speed are separate: passive drift is not learned propulsion.\n";
        if(!rows || !summary || !manifest) throw std::runtime_error("Evaluation output write failed");
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Current evaluation: %s\n",e.what());return 1;}
}
