// Small, isolated learnability experiment. Reuses the project's CUDA PPO math,
// MLP, Gaussian gradients and Adam; does not modify the existing demos.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "cuda/device.h"
#include "envs/swimmer/swimmer.h"
#include "nn/adam.h"
#include "nn/mlp_gpu.h"
#include "nn/mlp_cpu.h"
#include "rl/gaussian_kernels.h"
#include "rl/ppo_kernels.h"
#include "rl/swimmer_ppo.h"

using namespace cudarl;
namespace sw=cudarl::swimmer;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
void publish(const std::filesystem::path& source,const std::filesystem::path& destination) {
    std::error_code error;
    for(int attempt=0;attempt<20;++attempt) {
        std::filesystem::remove(destination,error);
        if(!error) {
            std::filesystem::rename(source,destination,error);
            if(!error) return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    throw std::runtime_error("Could not publish swimmer preview: "+error.message());
}
constexpr int hidden=32,rollout=64;
using Stream=rl::SwimmerStream;
using Trainer=rl::SwimmerTrainer;
nn::MlpSpec specification(int outputs) { return rl::swimmer_specification(outputs); }

struct Evaluation {
    int successes=0,count=0; double mean_seconds=0,mean_distance=0;
    double median_seconds=-1,distance_per_stroke=0,work_per_distance=0,turn_rate=0,median_turn_seconds=-1,capped_seconds=0;
    std::vector<std::vector<sw::State>> trace;
    double rate() const { return static_cast<double>(successes)/count; }
};
double median(std::vector<double> values) {
    if(values.empty()) return -1;
    std::sort(values.begin(),values.end());const auto n=values.size();
    return 0.5*(values[(n-1)/2]+values[n/2]);
}
Evaluation evaluate(const std::vector<float>& weights,uint32_t seed,int count,bool traces,bool stochastic=false,sw::Body body={}) {
    Stream stream; sw::Batch env(count,seed,body);nn::GpuMlp policy(specification(sw::kAct),count,1);
    policy.set_params(weights);CUDA_CHECK_THROW(cudaDeviceSynchronize());
    device_ptr<float> noise_std(2),sampled(static_cast<size_t>(count)*2),logp(count);
    const float initial_std[2]={-0.5f,-0.5f};noise_std.copy_from_host(initial_std,2);
    Evaluation e;e.count=count;
    std::vector<sw::State> states(count);
    std::vector<bool> behind(count,false);std::vector<double> turns(count,-1),food_times;
    if(traces) e.trace.resize(6);
    for(int t=0;t<=sw::kMaxSteps;++t) {
        if(t==sw::kMaxSteps || t%4==0) {
            CUDA_CHECK_THROW(cudaMemcpyAsync(states.data(),env.states(),states.size()*sizeof(sw::State),cudaMemcpyDeviceToHost,stream.value));
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
            if(traces) for(size_t i=0;i<e.trace.size();++i) e.trace[i].push_back(states[i]);
            for(int i=0;i<count;++i) {
                const auto& state=states[i];
                const float error=atan2f(state.food_y-state.y,state.food_x-state.x)-state.heading;
                const float bearing=std::fabs(atan2f(sinf(error),cosf(error)));
                if(t==0) behind[i]=bearing>sw::kPi*0.5f;
                if(behind[i] && turns[i]<0 && bearing<sw::kPi/12.f) turns[i]=t*sw::kDt;
            }
        }
        if(t==sw::kMaxSteps) break;
        policy.forward(env.obs(),count,stream.value);
        const float* actions=policy.output();
        if(stochastic) {
            rl::sample_gaussian(actions,noise_std.get(),count,2,seed+19,static_cast<uint32_t>(t),sampled.get(),logp.get(),stream.value);
            actions=sampled.get();
        }
        env.step(actions,{},stream.value,false);
    }
    double work=0,travel=0,strokes=0;std::vector<double> turn_times;int turning_attempts=0;
    for(int i=0;i<count;++i) {
        const auto& state=states[i];
        const float distance=sw::distance(state);e.mean_distance+=distance/count;
        if(distance<=sw::kFoodRadius) {++e.successes;e.mean_seconds+=state.steps*sw::kDt;food_times.push_back(state.steps*sw::kDt);}
        e.capped_seconds+=state.steps*sw::kDt/count;
        work+=state.work;travel+=state.travel;strokes+=state.strokes;
        if(behind[i]) {++turning_attempts;if(turns[i]>=0) turn_times.push_back(turns[i]);}
    }
    if(e.successes) e.mean_seconds/=e.successes;
    e.median_seconds=median(food_times);e.median_turn_seconds=median(turn_times);
    e.distance_per_stroke=strokes>0?travel/strokes:0;e.work_per_distance=travel>0?work/travel:0;
    e.turn_rate=turning_attempts>0?static_cast<double>(turn_times.size())/turning_attempts:0;
    return e;
}

void physical_checks() {
    sw::State stationary;stationary.food_x=1;const float zero[2]{};
    for(int t=0;t<100;++t) sw::integrate(stationary,zero);
    if(stationary.x!=0 || stationary.y!=0 || stationary.heading!=0) throw std::runtime_error("Passive swimmer generated propulsion");
    sw::State stroke;stroke.food_x=1;
    for(int t=0;t<sw::kMaxSteps;++t) {
        const float time=static_cast<float>(t)*sw::kDt;
        const float action[2]={cosf(4.f*time),cosf(4.f*time+sw::kPi*0.5f)};
        sw::integrate(stroke,action);
    }
    const float displacement=std::hypot(stroke.x,stroke.y);
    std::cout<<"PHYSICS_CHECK zero motors: no motion; cyclic gait displacement="<<displacement<<" in 40 simulated seconds\n";
    if(!std::isfinite(displacement)||displacement<0.1f) throw std::runtime_error("Cyclic joint movement failed to propel swimmer");
    constexpr int n=32;sw::Batch gpu(n,7321);std::vector<sw::State> cpu(n),actual(n);
    device_ptr<float> actions(n*2);std::vector<float> tape(n*2);std::vector<uint32_t> episodes(n,0);
    for(int i=0;i<n;++i) cpu[i]=sw::initial(7321,static_cast<uint32_t>(i));
    for(int t=0;t<200;++t) {
        for(int i=0;i<n;++i) {
            tape[i*2]=sinf(static_cast<float>(t+i)*0.17f);tape[i*2+1]=cosf(static_cast<float>(t+i)*0.11f);
            const auto r=sw::advance(cpu[i],tape.data()+i*2);
            if(r.success||r.timeout) cpu[i]=sw::initial(7321,static_cast<uint32_t>(i),++episodes[i]);
        }
        actions.copy_from_host(tape.data(),tape.size());gpu.step(actions.get(),{},nullptr);
    }
    CUDA_CHECK_THROW(cudaDeviceSynchronize());CUDA_CHECK_THROW(cudaMemcpy(actual.data(),gpu.states(),sizeof(sw::State)*n,cudaMemcpyDeviceToHost));
    float maximum=0;
    for(int i=0;i<n;++i) {
        maximum=std::max(maximum,std::fabs(actual[i].x-cpu[i].x));maximum=std::max(maximum,std::fabs(actual[i].y-cpu[i].y));
        maximum=std::max(maximum,std::fabs(actual[i].heading-cpu[i].heading));
        for(int j=0;j<2;++j) maximum=std::max(maximum,std::fabs(actual[i].angle[j]-cpu[i].angle[j]));
        if(actual[i].steps!=cpu[i].steps) throw std::runtime_error("CPU/GPU episode mismatch");
    }
    std::cout<<"PHYSICS_CHECK CPU/GPU max state error="<<maximum<<" over 32 environments x 200 steps\n";
    if(!std::isfinite(maximum)||maximum>0.002f) throw std::runtime_error("CPU/GPU swimmer disagreement");
}

void write_live(const std::filesystem::path& directory,const Evaluation& e,int seed,int update,double wall,double train,bool finished) {
    const auto temporary=directory/"live.tmp";std::ofstream f(temporary);f.exceptions(std::ios::failbit|std::ios::badbit);
    f<<std::setprecision(8)<<"{\"seed\":"<<seed<<",\"update\":"<<update<<",\"wall_seconds\":"<<wall
     <<",\"training_seconds\":"<<train<<",\"success_rate\":"<<e.rate()<<",\"mean_food_seconds\":"<<e.mean_seconds
     <<",\"median_food_seconds\":"<<e.median_seconds<<",\"distance_per_stroke\":"<<e.distance_per_stroke
     <<",\"work_per_distance\":"<<e.work_per_distance<<",\"turn_success_rate\":"<<e.turn_rate<<",\"median_turn_seconds\":"<<e.median_turn_seconds
     <<",\"finished\":"<<(finished?"true":"false")<<",\"frame_dt\":0.2,\"traces\":[";
    for(size_t i=0;i<e.trace.size();++i) {
        if(i) f<<',';f<<'[';
        for(size_t t=0;t<e.trace[i].size();++t) {
            if(t) f<<',';const auto& s=e.trace[i][t];
            f<<'['<<s.x<<','<<s.y<<','<<s.heading<<','<<s.angle[0]<<','<<s.angle[1]<<','<<s.food_x<<','<<s.food_y<<','<<s.steps<<','<<s.body.head<<','<<s.body.proximal<<','<<s.body.distal<<','<<s.body.fin_normal<<','<<(s.body.tapered?1:0)<<','<<(s.body.eel?1:0)<<']';
        }
        f<<']';
    }
    f<<"]}\n";f.close();
    // Reader tolerates the brief absent-file window, preserving its last frame.
    const auto destination=directory/"live.json";
    publish(temporary,destination);
    std::filesystem::copy_file(destination,directory/("seed-"+std::to_string(seed)+"-update-"+std::to_string(update)+".json"),std::filesystem::copy_options::overwrite_existing);
}

void write_preview(const std::filesystem::path& directory,const std::vector<sw::State>& states,int update) {
    const auto temporary=directory/"preview.tmp";std::ofstream f(temporary);
    f.exceptions(std::ios::failbit|std::ios::badbit);
    f<<std::setprecision(8)<<"{\"update\":"<<update<<",\"states\":[";
    for(size_t i=0;i<states.size();++i) {
        if(i) f<<',';const auto& s=states[i];
        f<<'['<<s.x<<','<<s.y<<','<<s.heading<<','<<s.angle[0]<<','<<s.angle[1]<<','<<s.food_x<<','<<s.food_y<<','<<s.steps<<','<<s.body.head<<','<<s.body.proximal<<','<<s.body.distal<<','<<s.body.fin_normal<<','<<(s.body.tapered?1:0)<<','<<(s.body.eel?1:0)<<']';
    }
    f<<"]}\n";f.close();const auto destination=directory/"preview.json";
    publish(temporary,destination);
}

int main(int argc,char** argv) {
    try {
        std::cout<<std::unitbuf;
        const std::filesystem::path directory=argc>1?argv[1]:"build/swimmer-experiment";
        const int updates=argc>2?std::stoi(argv[2]):600,seeds=argc>3?std::stoi(argv[3]):3,n=argc>4?std::stoi(argv[4]):1024;
        const int watch_delay=argc>5?std::stoi(argv[5]):0;
        const int body_id=argc>6?std::stoi(argv[6]):0;
        if(body_id<0||body_id>8||watch_delay<0) throw std::invalid_argument("Invalid body or viewing delay");
        const auto body=sw::body_preset(body_id);
        if(updates<0||seeds<1||n<8||n>32768) throw std::invalid_argument("Invalid experiment counts");
        std::filesystem::create_directories(directory);
        const auto device=probe_device();if(!device.available) throw std::runtime_error(device.error);
        std::cout<<"DEVICE "<<device.name<<" | envs="<<n<<" | body="<<body_id<<" | 9->32->32->2 | untrained PPO | 40 simulated seconds per attempt\n";
        physical_checks();
        std::ofstream metadata(directory/"config.json");
        metadata<<"{\"body_id\":"<<body_id<<",\"head_length\":"<<body.head<<",\"proximal_length\":"<<body.proximal
            <<",\"distal_length\":"<<body.distal<<",\"fin_normal_drag\":"<<body.fin_normal<<",\"motor_speed\":"<<body.motor_speed
            <<",\"tapered\":"<<(body.tapered?"true":"false")<<",\"eel\":"<<(body.eel?"true":"false")<<",\"environments\":"<<n<<",\"updates\":"<<updates
            <<",\"inertial\":"<<(body.inertial?"true":"false")
            <<",\"seeds\":"<<seeds<<",\"watch_delay_ms\":"<<watch_delay<<",\"rollout_steps\":64,\"hidden\":32,\"epochs\":4,\"minibatches\":8,\"learning_rate\":0.001,\"validation_seed\":90017,\"test_seed\":99173}\n";
        metadata.close();
        std::ofstream csv(directory/"results.csv");csv.exceptions(std::ios::failbit|std::ios::badbit);
        csv<<"seed,update,env_steps,wall_seconds,training_seconds,evaluation_count,successes,mean_food_seconds,mean_final_distance,median_food_seconds,distance_per_stroke,work_per_distance,turn_success_rate,median_turn_seconds,capped_mean_seconds,split\n";
        const auto record=[&](int seed,int update,double wall,double train,const Evaluation& e,const char* split) {
            csv<<seed<<','<<update<<','<<static_cast<long long>(update)*n*rollout<<','<<wall<<','<<train<<','<<e.count<<','<<e.successes<<','<<e.mean_seconds<<','<<e.mean_distance<<','<<e.median_seconds<<','<<e.distance_per_stroke<<','<<e.work_per_distance<<','<<e.turn_rate<<','<<e.median_turn_seconds<<','<<e.capped_seconds<<','<<split<<'\n';csv.flush();
            std::cout<<"SEED "<<seed<<" UPDATE "<<update<<" | wall "<<wall<<"s | train "<<train<<"s | "<<split<<" food "<<e.successes<<'/'<<e.count<<" | mean time "<<e.mean_seconds<<"s | distance "<<e.mean_distance<<'\n';
        };
        for(int key=1;key<=seeds;++key) {
            const auto start=Clock::now();Trainer trainer(n,static_cast<uint32_t>(key),body);double train=0;
            auto weights=trainer.actor_params();
            const auto baseline=evaluate(weights,99173,256,false,false,body);record(key,0,seconds(start),train,baseline,"test");
            const auto random_baseline=evaluate(weights,99173,256,false,true,body);record(key,0,seconds(start),train,random_baseline,"untrained_sampled_test");
            auto evaluation=evaluate(weights,90017,128,true,false,body);record(key,0,seconds(start),train,evaluation,"validation");
            write_live(directory,evaluation,key,0,seconds(start),train,updates==0);
            nn::CpuMlp<float> preview_policy(specification(sw::kAct));
            std::vector<sw::State> preview(6);std::vector<uint32_t> preview_episodes(6,0);
            for(int i=0;i<6;++i) preview[i]=sw::initial(44571,static_cast<uint32_t>(i),0,body);
            if(watch_delay>0) {write_preview(directory,preview,0);std::this_thread::sleep_for(std::chrono::seconds(2));}
            auto preview_time=Clock::now();double accumulator=0;
            int streak=0;double first_80=-1;
            for(int update=1;update<=updates;++update) {
                train+=trainer.update().update_ms*0.001;
                if(watch_delay>0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(watch_delay));
                    accumulator+=seconds(preview_time);preview_time=Clock::now();
                    preview_policy.params()=trainer.actor_params();
                    while(accumulator>=sw::kDt) {
                        for(int i=0;i<6;++i) {
                            float observation[sw::kObs];sw::observe(preview[i],observation);preview_policy.forward(observation,1);
                            const auto r=sw::advance(preview[i],preview_policy.output());
                            if(r.success||r.timeout) preview[i]=sw::initial(44571,static_cast<uint32_t>(i),++preview_episodes[i],body);
                        }
                        accumulator-=sw::kDt;
                    }
                    write_preview(directory,preview,update);
                }
                if(update%25!=0 && update!=updates) continue;
                weights=trainer.actor_params();evaluation=evaluate(weights,90017,128,true,false,body);
                record(key,update,seconds(start),train,evaluation,"validation");
                write_live(directory,evaluation,key,update,seconds(start),train,update==updates && key==seeds);
                streak=evaluation.rate()>=0.8?streak+1:0;
                if(streak==2 && first_80<0) {first_80=seconds(start);std::cout<<"TWO_CONSECUTIVE_80_PERCENT seed="<<key<<" wall_seconds="<<first_80<<'\n';}
            }
            weights=trainer.actor_params();const auto test=evaluate(weights,99173,256,false,false,body);
            record(key,updates,seconds(start),train,test,"test");
            std::ofstream policy(directory/("policy-seed-"+std::to_string(key)+".json"));
            policy<<"{\"inputs\":9,\"hidden\":32,\"outputs\":2,\"body_id\":"<<body_id<<",\"parameters\":["<<std::setprecision(9);
            for(size_t i=0;i<weights.size();++i) {if(i) policy<<',';policy<<weights[i];}policy<<"]}\n";
        }
        std::cout<<"EXPERIMENT_COMPLETE "<<directory.string()<<'\n';return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
