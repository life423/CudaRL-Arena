#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <string>
#include "demos/pond_target.h"
#include "rl/swimmer_ppo.h"
#include "rl/pond_selector_ppo.h"
#include "nn/mlp_cpu.h"
namespace p=cudarl::deep_pond;
namespace d=cudarl::deep;
namespace rl=cudarl::rl;
struct Case {
    int foods=0,slots[3]{},switches=0,retargets=0,ascending=0,descending=0;
    double low_speed=0,no_food=0,max_food_gap=0,low_run=0,max_low_run=0;
};
std::vector<Case> evaluate(rl::DeepPondTrainer& trainer,const std::vector<float>& weights,
                          int n,uint32_t seed,int mode,const rl::SelectorPolicy* candidate,int ticks,const std::string& output) {
    p::Batch world(n,seed);trainer.actor.set_params(weights);CUDA_CHECK_THROW(cudaDeviceSynchronize());
    device_ptr<float> input(static_cast<size_t>(n)*p::kObs);
    std::vector<float> observations(static_cast<size_t>(n)*p::kObs);
    std::vector<p::State> before(n),after(n);std::vector<int> target(n,-1);
    std::vector<float> food_origin(n);std::vector<Case> cases(n);
    bool captured=false;
    cudarl::nn::CpuMlp<float> selector(rl::selector_spec());if(candidate) selector.params()=candidate->selector;
    CUDA_CHECK_THROW(cudaMemcpy(before.data(),world.states(),n*sizeof(p::State),cudaMemcpyDeviceToHost));
    for(int i=0;i<n;++i) food_origin[i]=before[i].body.position.y;
    for(int tick=0;tick<ticks;++tick) {
        CUDA_CHECK_THROW(cudaMemcpyAsync(observations.data(),world.obs(),input.size_bytes(),cudaMemcpyDeviceToHost,trainer.stream.value));
        CUDA_CHECK_THROW(cudaStreamSynchronize(trainer.stream.value));
        for(int i=0;i<n;++i) {
            auto* o=observations.data()+i*p::kObs;int next=target[i];
            const bool decision=mode<2 || tick%candidate->decision_steps==0;
            if(decision) {
                if(mode==0) next=0;
                else if(mode<3) next=cudarl::demos::nearest_pond_food(o);
                else {
                    float x[36];std::copy(o,o+33,x);for(int j=0;j<3;++j) x[33+j]=j==target[i]?1.f:0.f;
                    float scores[3];for(int j=0;j<3;++j) {float f[32];cudarl::pond_selector::features(x,j,f);selector.forward(f,1);scores[j]=selector.output()[0];}
                    next=0;for(int j=1;j<3;++j) if(scores[j]>scores[next]) next=j;
                }
            }
            if(target[i]>=0 && target[i]!=next) {
                ++cases[i].retargets;
                if(before[i].last_eaten!=target[i]) ++cases[i].switches;
            }
            target[i]=next;cudarl::demos::primary_pond_food(o,next);
        }
        CUDA_CHECK_THROW(cudaMemcpyAsync(input.get(),observations.data(),input.size_bytes(),cudaMemcpyHostToDevice,trainer.stream.value));
        trainer.actor.forward(input.get(),n,trainer.stream.value);
        world.step(trainer.actor.output(),{},trainer.stream.value,false,ticks);
        CUDA_CHECK_THROW(cudaMemcpyAsync(after.data(),world.states(),n*sizeof(p::State),cudaMemcpyDeviceToHost,trainer.stream.value));
        CUDA_CHECK_THROW(cudaStreamSynchronize(trainer.stream.value));
        for(int i=0;i<n;++i) {
            auto& c=cases[i];const auto& s=after[i];
            const d::V velocity{s.body.velocity[0],s.body.velocity[1],s.body.velocity[2]};
            if(!std::isfinite(d::norm(s.body.position)) || !std::isfinite(d::norm(velocity))) throw std::runtime_error("Non-finite swimmer trajectory");
            if(d::norm(velocity)<.005f) {c.low_speed+=.05;c.low_run+=.05;} else c.low_run=0;
            c.max_low_run=std::max(c.max_low_run,c.low_run);c.max_food_gap=std::max(c.max_food_gap,s.since_food*.05);
            if(s.since_food>=200) c.no_food+=.05; // Time after >=10 s without collection.
            if(!captured && s.since_food>=400 && d::norm(velocity)<.005f) {
                cudarl::checkpoint::Writer saved;saved.value(uint32_t{0x53544131});saved.value(s);saved.value(target[i]);
                cudarl::checkpoint::write_atomic((std::filesystem::path(output)/("stall-mode"+std::to_string(mode)+".bin")).string(),saved.data);captured=true;
            }
            if(s.last_eaten>=0) {
                ++c.foods;++c.slots[s.last_eaten];
                if(s.body.position.y>food_origin[i]) ++c.ascending;else ++c.descending;
                food_origin[i]=s.body.position.y;
            }
        }
        before.swap(after);
    }
    if(trainer.actor_params()!=weights) throw std::runtime_error("Evaluation changed motor weights");
    return cases;
}
int main(int argc,char** argv) {
    try {
        std::string checkpoint,output,selector_path,motor_policy;uint32_t seed=0;int n=64,seconds=60;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--help") {std::puts("pond_choice_eval (--checkpoint COPY | --motor-policy BUNDLE) --output DIR --seed N [--trials 64] [--seconds 60] [--selector POLICY]\nRead-only canonical, nearest-food and optional learned-selector benchmark of the saved best motor policy. --motor-policy uses packaged weights without a private training checkpoint.");return 0;}
            if(i+1>=argc) throw std::invalid_argument("Missing argument value");const std::string v=argv[++i];
            if(arg=="--checkpoint") checkpoint=v;else if(arg=="--output") output=v;else if(arg=="--selector") selector_path=v;
            else if(arg=="--motor-policy") motor_policy=v;
            else if(arg=="--seed") seed=static_cast<uint32_t>(std::stoul(v));else if(arg=="--trials") n=std::stoi(v);
            else if(arg=="--seconds") seconds=std::stoi(v);
            else throw std::invalid_argument("Unknown argument: "+arg);
        }
        if((checkpoint.empty() && motor_policy.empty()) || output.empty() || n<2 || n>4096 || seconds<1 || seconds>3600) throw std::invalid_argument("Supply a snapshot or motor bundle, output folder, 2..4096 trials and 1..3600 seconds");
        std::vector<uint8_t> original;if(!checkpoint.empty()) original=cudarl::checkpoint::read(checkpoint);
        rl::SwimmerConfig cfg;cfg.num_envs=1024;cfg.seed=7;cfg.evaluation_every=checkpoint.empty()?0:25;cfg.training_checkpoint_path=checkpoint;
        rl::DeepPondTrainer trainer(cfg);std::vector<float> weights;int motor_update=0,checkpoint_update=-1;
        if(!motor_policy.empty()) {const auto bundled=rl::SelectorPolicy::load(motor_policy);weights=bundled.motor;motor_update=bundled.motor_update;}
        else {
            if(!trainer.checkpoint_status().resumed || !trainer.best_policy().valid()) throw std::runtime_error("An existing best-policy checkpoint is required");
            weights=trainer.best_policy().parameters;motor_update=trainer.best_policy().update;checkpoint_update=trainer.current_stats().update;
        }
        const int ticks=seconds*20;const double rate=60./seconds;
        const auto canonical=evaluate(trainer,weights,n,seed,0,nullptr,ticks,output),nearest=evaluate(trainer,weights,n,seed,1,nullptr,ticks,output);
        std::vector<std::vector<Case>> comparisons{canonical,nearest};std::vector<std::string> labels{"canonical","nearest"};
        rl::SelectorPolicy candidate;
        if(!selector_path.empty()) {
            candidate=rl::SelectorPolicy::load(selector_path);if(candidate.motor!=weights) throw std::runtime_error("Candidate must use exactly the same frozen best motor");
            comparisons.push_back(evaluate(trainer,weights,n,seed,2,&candidate,ticks,output));labels.push_back("held_nearest");
            comparisons.push_back(evaluate(trainer,weights,n,seed,3,&candidate,ticks,output));labels.push_back("learned_selector");
        }
        if(!checkpoint.empty() && original!=cudarl::checkpoint::read(checkpoint)) throw std::runtime_error("Evaluation changed the checkpoint");
        std::filesystem::create_directories(output);
        std::ofstream rows(std::filesystem::path(output)/"cases.csv"),summary(std::filesystem::path(output)/"summary.csv");
        if(!rows || !summary) throw std::runtime_error("Could not create evaluation outputs");
        rows<<"mode,seed,environment,seconds,foods,slot0,slot1,slot2,low_speed_seconds,no_food_10_seconds,uncollected_target_switches,all_target_changes,ascending,descending,max_food_gap_seconds,max_low_speed_run_seconds\n";
        summary<<"mode,seed,trials,seconds,food_per_minute,mean_low_speed_seconds,mean_no_food_10_seconds,mean_uncollected_target_switches,slot0,slot1,slot2,max_food_gap_seconds,max_low_speed_run_seconds\n";
        for(size_t mode=0;mode<comparisons.size();++mode) {
            const auto& cases=comparisons[mode];const char* label=labels[mode].c_str();Case total;
            for(int i=0;i<n;++i) {
                const auto& c=cases[i];rows<<std::setprecision(9)<<label<<','<<seed<<','<<i<<','<<seconds<<','<<c.foods;
                for(int x:c.slots) rows<<','<<x;
                rows<<','<<c.low_speed<<','<<c.no_food<<','<<c.switches<<','<<c.retargets<<','<<c.ascending<<','<<c.descending<<','<<c.max_food_gap<<','<<c.max_low_run<<'\n';
                total.foods+=c.foods;total.low_speed+=c.low_speed;total.no_food+=c.no_food;total.switches+=c.switches;
                total.max_food_gap=std::max(total.max_food_gap,c.max_food_gap);total.max_low_run=std::max(total.max_low_run,c.max_low_run);
                for(int j=0;j<3;++j) total.slots[j]+=c.slots[j];
            }
            summary<<std::setprecision(9)<<label<<','<<seed<<','<<n<<','<<seconds<<','<<double(total.foods)*rate/n<<','<<total.low_speed/n<<','<<total.no_food/n<<','<<double(total.switches)/n;
            for(int x:total.slots) summary<<','<<x;summary<<','<<total.max_food_gap<<','<<total.max_low_run<<'\n';
            std::printf("POND_CHOICE mode=%s seed=%u n=%d fpm=%.6f low_speed=%.3f no_food_10=%.3f switches=%.3f slots=%d,%d,%d max_food_gap=%.2f max_low_run=%.2f\n",
                label,seed,n,double(total.foods)*rate/n,total.low_speed/n,total.no_food/n,double(total.switches)/n,total.slots[0],total.slots[1],total.slots[2],total.max_food_gap,total.max_low_run);
        }
        double mean=0,variance=0;int wins=0,ties=0;
        for(int i=0;i<n;++i) {const int difference=nearest[i].foods-canonical[i].foods;mean+=difference;if(difference>0) ++wins;else if(difference==0) ++ties;}
        mean/=n;for(int i=0;i<n;++i) {const double error=(nearest[i].foods-canonical[i].foods)-mean;variance+=error*error;}
        mean*=rate;const double ci=rate*1.96*std::sqrt(variance/(n-1)/n);
        std::ofstream manifest(std::filesystem::path(output)/"result.txt");
        manifest<<"best_update="<<motor_update<<" checkpoint_update="<<checkpoint_update<<" seed="<<seed<<" trials="<<n<<" seconds="<<seconds<<'\n'
            <<"paired_delta_fpm="<<mean<<" approximate_95_ci="<<mean-ci<<","<<mean+ci<<" wins="<<wins<<" ties="<<ties<<" losses="<<n-wins-ties<<"\n"
            <<"motor_weights_unchanged=true checkpoint_unchanged=true physics_unchanged=true gradients=false\n"
            <<"Same initial layouts and environment IDs. Subsequent placements depend on physical collection history; they need not match across modes.\n"
            <<"Nearest is a programmed per-control-step input-order benchmark, not a learned policy. Wall time includes CPU instrumentation and is not a throughput benchmark.\n";
        if(!rows || !summary || !manifest) throw std::runtime_error("Could not write evaluation outputs");
        std::printf("PAIRED_DELTA %.6f approximate95=[%.6f,%.6f] wins=%d ties=%d losses=%d best_update=%d\n",mean,mean-ci,mean+ci,wins,ties,n-wins-ties,motor_update);
        if(!selector_path.empty()) {
            manifest<<"selector_update="<<candidate.update<<" decision_steps="<<candidate.decision_steps<<" candidate_fixed_fpm="<<candidate.food_per_minute()<<'\n';
            for(size_t baseline=0;baseline<3;++baseline) {
                double delta=0,error_sum=0;for(int i=0;i<n;++i) delta+=comparisons[3][i].foods-comparisons[baseline][i].foods;delta/=n;
                for(int i=0;i<n;++i) {const double e=(comparisons[3][i].foods-comparisons[baseline][i].foods)-delta;error_sum+=e*e;}
                delta*=rate;const double width=rate*1.96*std::sqrt(error_sum/(n-1)/n);
                manifest<<"selector_vs_"<<labels[baseline]<<" delta="<<delta<<" approximate_95_ci="<<delta-width<<","<<delta+width<<'\n';
                std::printf("SELECTOR_VS_%s delta=%.6f approximate95=[%.6f,%.6f]\n",labels[baseline].c_str(),delta,delta-width,delta+width);
            }
        }
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Pond choice evaluation: %s\n",e.what());return 1;}
}
