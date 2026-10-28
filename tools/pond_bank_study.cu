#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include "rl/pond_recovery_pool.h"
#include "nn/mlp_gpu.h"
#include "rl/ppo_kernels.h"
namespace rl=cudarl::rl;
namespace p=cudarl::deep_pond;
namespace d=cudarl::deep;
namespace pg=cudarl::progress_selector;
namespace {
struct Result {int foods=0,first=0,low=0,max_low=0,invalid=0;};
__device__ bool near_bank(const p::State& state) {
    if(state.body.position.y>=-.2f) return false;
    const auto geometry=d::shape(state.body);
    for(int b=0;b<3;++b) {
        const auto& l=geometry.link[b];
        for(int sample=0;sample<3;++sample) {
            const auto point=l.center+l.q.rotate({(sample-1)*l.length/3.f,0,0});const float x=point.x/d::scale,z=point.z/d::scale;
            const float r2=x*x/(4.6f*4.6f)+z*z/(3.5f*3.5f);
            const float gx=r2<1?27.f*x*r2*r2/(4.6f*4.6f):0,gz=r2<1?27.f*z*r2*r2/(3.5f*3.5f):0;
            const float gap=(point.y-d::pond_floor(x,z)*d::scale)/sqrtf(1+gx*gx+gz*gz)-(b==0?.0064f:.002f);
            if(r2>.35f && gap<.001f) return true;
        }
    }
    return false;
}
__global__ void initialize(pg::RecoveryStart* worlds,int n,uint32_t seed) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    pg::RecoveryStart r;r.state=p::initial(seed,static_cast<uint32_t>(i));r.history=pg::initial_history(r.state);
    const float mode=cudarl::rng::uniform(seed^0xB4730191u,static_cast<uint32_t>(i),0);
    if(mode>=.25f) {
        const float speed=mode<.625f?.005f:.01f;
        const float angle=cudarl::rng::uniform(seed^0xB4730191u,static_cast<uint32_t>(i),1)*6.2831853f;r.current={speed*cosf(angle),0,speed*sinf(angle)};
    }
    worlds[i]=r;
}
__global__ void selector_inputs(const pg::RecoveryStart* worlds,float* obs,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) pg::observe(worlds[i].state,worlds[i].target,worlds[i].history,obs+i*40);
}
__global__ void choose(pg::RecoveryStart* worlds,const int* targets,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) worlds[i].target=targets[i];
}
__global__ void motor_inputs(const pg::RecoveryStart* worlds,float* obs,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;p::observe(worlds[i].state,obs+i*33);cudarl::demos::primary_pond_food(obs+i*33,worlds[i].target);
}
__global__ void step(pg::RecoveryStart* worlds,const float* actions,Result* results,pg::RecoveryStart* captured,int* flags,int n,int tick,int period) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    auto r=worlds[i];auto m=results[i];const auto outcome=p::advance(r.state,actions+i*4,r.current);
    const float speed=d::norm({r.state.body.velocity[0],r.state.body.velocity[1],r.state.body.velocity[2]});
    if(outcome.eaten_slot>=0) {++m.foods;if(!m.first) m.first=tick;}
    m.low=speed<.005f?m.low+1:0;if(m.low>m.max_low) m.max_low=m.low;
    if(!isfinite(r.state.body.position.x) || !isfinite(r.state.body.position.y) || !isfinite(r.state.body.position.z) || !isfinite(speed)) ++m.invalid;
    if(tick%period==0) {
        pg::update_history(r.state,r.history,period);
        if(captured && !flags[i] && r.state.since_food>=400 && speed<.005f && near_bank(r.state)) {
            bool water=true;for(auto food:r.state.food) water=water && p::water(food);
            if(water) {captured[i]=r;flags[i]=1;}
        }
    }
    worlds[i]=r;results[i]=m;
}
struct Run {
    int n;const rl::SelectorPolicy& base;bool augmented;
    cudarl::nn::GpuMlp motor,selector;device_ptr<pg::RecoveryStart> worlds;
    device_ptr<float> obs,features,motor_obs,logp;device_ptr<int> targets;device_ptr<Result> result;
    Run(const rl::SelectorPolicy& policy,int count,const std::vector<float>* actor=nullptr):n(count),base(policy),augmented(actor!=nullptr),
        motor(rl::swimmer_specification(4,33),n,7),selector(augmented?rl::progress_selector_spec():rl::selector_spec(),n*3,7),worlds(n),
        obs(static_cast<size_t>(n)*40),features(static_cast<size_t>(n)*3*34),motor_obs(static_cast<size_t>(n)*33),logp(n),targets(n),result(n) {
        motor.set_params(base.motor);selector.set_params(actor?*actor:base.selector);CUDA_CHECK_THROW(cudaMemset(result.get(),0,result.size_bytes()));
    }
    void tick(int tick,pg::RecoveryStart* captured=nullptr,int* flags=nullptr) {
        if((tick-1)%base.decision_steps==0) {
            selector_inputs<<<(n+127)/128,128>>>(worlds.get(),obs.get(),n);CUDA_CHECK_THROW(cudaGetLastError());
            if(augmented) pg::make_features(obs.get(),features.get(),n,nullptr);
            else {
                // The original feature kernel has a 36-float row stride.
                // Pack it explicitly from the common 40-float observations.
                pack_original();
            }
            selector.forward(features.get(),n*3);rl::sample_actions(selector.output(),n,3,0,0,true,targets.get(),logp.get(),nullptr);
            choose<<<(n+127)/128,128>>>(worlds.get(),targets.get(),n);CUDA_CHECK_THROW(cudaGetLastError());
        }
        motor_inputs<<<(n+127)/128,128>>>(worlds.get(),motor_obs.get(),n);CUDA_CHECK_THROW(cudaGetLastError());motor.forward(motor_obs.get(),n);
        step<<<(n+63)/64,64>>>(worlds.get(),motor.output(),result.get(),captured,flags,n,tick,base.decision_steps);CUDA_CHECK_THROW(cudaGetLastError());
    }
    void pack_original();
};
__global__ void original_features(const float* obs,float* out,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*3) cudarl::pond_selector::features(obs+(i/3)*40,i%3,out+i*32);
}
void Run::pack_original() {original_features<<<(n*3+127)/128,128>>>(obs.get(),features.get(),n);CUDA_CHECK_THROW(cudaGetLastError());}
float median(std::vector<float> v) {if(v.empty()) return -1;std::sort(v.begin(),v.end());const size_t i=v.size()/2;return v.size()%2?v[i]:(v[i-1]+v[i])*.5f;}
void vector_json(std::ostream& out,d::V value) {out<<'['<<value.x<<','<<value.y<<','<<value.z<<']';}
void quaternion_json(std::ostream& out,d::Q value) {out<<'['<<value.x<<','<<value.y<<','<<value.z<<','<<value.w<<']';}
void frame_json(std::ostream& out,const pg::RecoveryStart& frame,int tick,int initial_foods) {
    const auto& s=frame.state;const auto geometry=d::shape(s.body);
    out<<"{\"time\":"<<tick*.05<<",\"eaten\":"<<s.eaten-initial_foods<<",\"since_food\":"<<s.since_food*.05<<",\"target\":"<<frame.target<<",\"position\":";
    vector_json(out,s.body.position);out<<",\"orientation\":";quaternion_json(out,s.body.orientation);
    out<<",\"speed_cm_s\":"<<d::norm({s.body.velocity[0],s.body.velocity[1],s.body.velocity[2]})*100<<",\"foods\":[";
    for(int f=0;f<3;++f) {if(f) out<<',';vector_json(out,s.food[f]);}out<<"],\"links\":[";
    for(int b=0;b<3;++b) {if(b) out<<',';out<<"{\"position\":";vector_json(out,geometry.link[b].center/d::scale);out<<",\"orientation\":";quaternion_json(out,geometry.link[b].q);out<<'}';}
    out<<"],\"angles\":[";for(int j=0;j<4;++j) {if(j) out<<',';out<<s.body.angle[j];}out<<"]}";
}
}
int main(int argc,char** argv) {
    try {
        std::string policy_path,pool_path,progress_path,output;uint32_t seed=0;int n=8192,seconds=300,record_environment=-1;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];if(arg=="--help") {std::puts("pond_bank_study --policy FROZEN_BUNDLE --output NEW_DIR [--collect SEED --envs 8192 --seconds 300] | [--pool FILE --progress CANDIDATE --seconds 60]\nFrozen collection/evaluation only. No optimizer, resets or scripted recovery. All food stays underwater.");return 0;}
            if(i+1>=argc) throw std::invalid_argument("Missing bank-study argument");const std::string value=argv[++i];
            if(arg=="--policy") policy_path=value;else if(arg=="--output") output=value;else if(arg=="--collect") seed=static_cast<uint32_t>(std::stoul(value,nullptr,0));
            else if(arg=="--pool") pool_path=value;else if(arg=="--progress") progress_path=value;else if(arg=="--record") record_environment=std::stoi(value);
            else if(arg=="--envs") n=std::stoi(value);else if(arg=="--seconds") seconds=std::stoi(value);else throw std::invalid_argument("Unknown bank-study argument");
        }
        if(policy_path.empty() || output.empty() || std::filesystem::exists(output) || n<16 || n>32768 || seconds<1 || seconds>600 || (!seed && pool_path.empty())) throw std::invalid_argument("Use frozen policy and a new output directory, plus collection seed or pool");
        const auto bytes=cudarl::checkpoint::read(policy_path);const auto policy=rl::SelectorPolicy::load(policy_path);std::filesystem::create_directories(output);
        if(seed) {
            Run run(policy,n);device_ptr<pg::RecoveryStart> captured(n);device_ptr<int> flags(n);CUDA_CHECK_THROW(cudaMemset(flags.get(),0,flags.size_bytes()));
            initialize<<<(n+127)/128,128>>>(run.worlds.get(),n,seed);CUDA_CHECK_THROW(cudaGetLastError());
            for(int tick=1;tick<=seconds*20;++tick) run.tick(tick,captured.get(),flags.get());CUDA_CHECK_THROW(cudaDeviceSynchronize());
            std::vector<pg::RecoveryStart> states(n);std::vector<int> chosen(n);std::vector<Result> metrics(n);captured.copy_to_host(states.data(),n);flags.copy_to_host(chosen.data(),n);run.result.copy_to_host(metrics.data(),n);
            rl::RecoveryPool pool;pool.baseline=policy;std::ofstream csv(std::filesystem::path(output)/"sources.csv");csv<<"seed,environment,tick,since_food_s,current_cm_s,prior_target,foods,body_x,body_y,body_z\n";
            for(int i=0;i<n;++i) {
                if(metrics[i].invalid) throw std::runtime_error("Non-finite bank collection world");
                if(!chosen[i]) continue;rl::validate_recovery(states[i]);const auto& r=states[i];pool.starts.push_back(r);
                csv<<seed<<','<<i<<','<<r.state.steps<<','<<r.state.since_food*.05f<<','<<d::norm(r.current)*100<<','<<r.target<<','<<r.state.eaten<<','<<r.state.body.position.x<<','<<r.state.body.position.y<<','<<r.state.body.position.z<<'\n';
            }
            if(pool.starts.size()>4096) throw std::runtime_error("Recovery pool exceeds checkpoint bound");
            pool.save((std::filesystem::path(output)/"pool.bin").string());if(!csv) throw std::runtime_error("Failed pool manifest");
            std::printf("BANK_COLLECT seed=%u worlds=%d seconds=%d bank_states=%zu all_food_underwater=true\n",seed,n,seconds,pool.starts.size());
            if(run.motor.get_params()!=policy.motor || run.selector.get_params()!=policy.selector) throw std::runtime_error("Collection changed frozen weights");
        } else {
            const auto pool=rl::RecoveryPool::load(pool_path);if(pool.baseline.motor!=policy.motor || pool.baseline.selector!=policy.selector) throw std::runtime_error("Pool baseline differs");
            std::vector<float> actor;std::vector<uint8_t> candidate_bytes;int selector_update=policy.update;
            if(!progress_path.empty()) {candidate_bytes=cudarl::checkpoint::read(progress_path);const auto candidate=rl::ProgressPolicy::load(progress_path);if(candidate.baseline.motor!=policy.motor) throw std::runtime_error("Candidate changed swimming controller");actor=candidate.actor;selector_update=candidate.update;}
            const int count=static_cast<int>(pool.starts.size());Run run(policy,count,actor.empty()?nullptr:&actor);run.worlds.copy_from_host(pool.starts.data(),pool.starts.size());
            std::ofstream replay;int record_index=-1;
            if(record_environment>=0) {
                for(int i=0;i<count;++i) if(pool.starts[i].state.environment==static_cast<uint32_t>(record_environment)) record_index=i;
                if(record_index<0) throw std::invalid_argument("Recording environment is not in this pool");
                replay.open(std::filesystem::path(output)/"replay.json");replay<<std::setprecision(9)<<"{\"version\":1,\"cases\":[{\"name\":\""<<(actor.empty()?"Original selector":"Bank-trained selector")<<"\",\"selector_update\":"<<selector_update<<",\"motor_update\":"<<policy.motor_update<<",\"current_cm_s\":"<<d::norm(pool.starts[record_index].current)*100<<",\"frames\":[";
                frame_json(replay,pool.starts[record_index],0,pool.starts[record_index].state.eaten);
            }
            for(int tick=1;tick<=seconds*20;++tick) {
                run.tick(tick);
                if(record_index>=0) {pg::RecoveryStart frame;CUDA_CHECK_THROW(cudaMemcpy(&frame,run.worlds.get()+record_index,sizeof(frame),cudaMemcpyDeviceToHost));replay<<',';frame_json(replay,frame,tick,pool.starts[record_index].state.eaten);}
            }
            if(record_index>=0) {replay<<"]}]}";if(!replay) throw std::runtime_error("Failed recording bank replay");}
            CUDA_CHECK_THROW(cudaDeviceSynchronize());std::vector<Result> metrics(count);run.result.copy_to_host(metrics.data(),count);
            std::ofstream csv(std::filesystem::path(output)/"cases.csv");csv<<"seed,environment,first_food_s,foods,max_low_speed_s\n";
            int within20=0,within60=0,foods=0,low20=0;std::vector<float> times;
            for(int i=0;i<count;++i) {const auto& m=metrics[i];if(m.invalid) throw std::runtime_error("Non-finite recovery evaluation");within20+=m.first && m.first<=400;within60+=m.first && m.first<=1200;foods+=m.foods;low20+=m.max_low>400;if(m.first) times.push_back(m.first*.05f);
                csv<<pool.starts[i].state.seed<<','<<pool.starts[i].state.environment<<','<<(m.first?m.first*.05f:-1)<<','<<m.foods<<','<<m.max_low*.05f<<'\n';}
            std::ofstream summary(std::filesystem::path(output)/"summary.csv");summary<<"trials,recovered_20s,recovered_60s,median_first_success_s,mean_foods,low_speed_over_20s\n"<<count<<','<<within20<<','<<within60<<','<<median(times)<<','<<double(foods)/count<<','<<low20<<'\n';
            if(!csv || !summary) throw std::runtime_error("Failed recovery report");std::printf("BANK_EVAL cases=%d recovered20=%d recovered60=%d median=%.2f mean_foods=%.4f low20=%d\n",count,within20,within60,median(times),double(foods)/count,low20);
            if(run.motor.get_params()!=policy.motor || run.selector.get_params()!=(actor.empty()?policy.selector:actor)) throw std::runtime_error("Recovery evaluation changed weights");
            if(!progress_path.empty() && candidate_bytes!=cudarl::checkpoint::read(progress_path)) throw std::runtime_error("Candidate file changed");
        }
        if(bytes!=cudarl::checkpoint::read(policy_path)) throw std::runtime_error("Original policy changed");return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Bank study: %s\n",e.what());return 1;}
}
