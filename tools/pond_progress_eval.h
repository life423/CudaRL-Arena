// Included after the original current-evaluation kernels. Both policies use the
// exact same compiled physical step; only selector observations/weights differ.
#include "rl/progress_selector_policy.h"
namespace progress_evaluation {
namespace pg=cudarl::progress_selector;
__global__ void history_reset(const p::State* states,pg::History* history,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) history[i]=pg::initial_history(states[i]);
}
__global__ void progress_inputs(const p::State* states,const int* targets,const pg::History* history,float* obs,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) pg::observe(states[i],targets[i],history[i],obs+i*pg::kObs);
}
__global__ void history_update(const p::State* states,pg::History* history,int n,int period) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) pg::update_history(states[i],history[i],period);
}
std::vector<Case> run(const rl::ProgressPolicy& candidate,bool augmented,int n,uint32_t seed,int ticks,d::V current,
                      const p::State* fixture=nullptr,int fixture_target=-1,float* first=nullptr) {
    const auto& parent=candidate.baseline;
    cudarl::nn::GpuMlp motor(rl::swimmer_specification(4,33),n,7),selector(augmented?rl::progress_selector_spec():rl::selector_spec(),n*3,7);
    motor.set_params(parent.motor);selector.set_params(augmented?candidate.actor:parent.selector);
    device_ptr<p::State> states(n);device_ptr<Case> metrics(n);device_ptr<int> targets(n);device_ptr<pg::History> history(n);
    device_ptr<float> obs(n*40),features(n*3*34),motor_obs(n*33),logp(n);
    CUDA_CHECK_THROW(cudaMemset(metrics.get(),0,metrics.size_bytes()));
    initialize<<<(n+63)/64,64>>>(states.get(),obs.get(),targets.get(),n,seed);CUDA_CHECK_THROW(cudaGetLastError());
    if(fixture) {
        if(n<=24) throw std::invalid_argument("The recorded fixture requires at least 25 worlds");
        CUDA_CHECK_THROW(cudaMemcpy(states.get()+24,fixture,sizeof(*fixture),cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(targets.get()+24,&fixture_target,sizeof(int),cudaMemcpyHostToDevice));
    }
    history_reset<<<(n+63)/64,64>>>(states.get(),history.get(),n);CUDA_CHECK_THROW(cudaGetLastError());
    if(first) *first=-1;
    for(int tick=0;tick<ticks;++tick) {
        if(tick%parent.decision_steps==0) {
            if(augmented) {
                progress_inputs<<<(n+63)/64,64>>>(states.get(),targets.get(),history.get(),obs.get(),n);CUDA_CHECK_THROW(cudaGetLastError());
                pg::make_features(obs.get(),features.get(),n,nullptr);
            } else {
                selector_inputs<<<(n+63)/64,64>>>(states.get(),targets.get(),obs.get(),n);CUDA_CHECK_THROW(cudaGetLastError());
                ps::make_features(obs.get(),features.get(),n,nullptr);
            }
            selector.forward(features.get(),n*3,nullptr);rl::sample_actions(selector.output(),n,3,0,0,true,targets.get(),logp.get(),nullptr);
        }
        motor_inputs<<<(n+63)/64,64>>>(states.get(),targets.get(),motor_obs.get(),n);CUDA_CHECK_THROW(cudaGetLastError());motor.forward(motor_obs.get(),n,nullptr);
        advance_current<<<(n+63)/64,64>>>(states.get(),motor.output(),metrics.get(),n,current);CUDA_CHECK_THROW(cudaGetLastError());
        if((tick+1)%parent.decision_steps==0) {
            history_update<<<(n+63)/64,64>>>(states.get(),history.get(),n,parent.decision_steps);CUDA_CHECK_THROW(cudaGetLastError());
            if(fixture && first && *first<0) {p::State state;CUDA_CHECK_THROW(cudaMemcpy(&state,states.get()+24,sizeof(state),cudaMemcpyDeviceToHost));if(state.eaten>fixture->eaten) *first=(tick+1)*.05f;}
        }
    }
    CUDA_CHECK_THROW(cudaDeviceSynchronize());std::vector<Case> result(n);metrics.copy_to_host(result.data(),n);
    if(motor.get_params()!=parent.motor || selector.get_params()!=(augmented?candidate.actor:parent.selector)) throw std::runtime_error("Evaluation changed weights");
    for(const auto& c:result) if(c.invalid) throw std::runtime_error("Non-finite current evaluation state");return result;
}
int evaluate(const std::string& candidate_path,const std::string& output,int n,int seconds,const std::vector<uint32_t>& seeds,
             const std::vector<Condition>& suite,const std::string& fixture_path) {
    const auto source=cudarl::checkpoint::read(candidate_path);const auto candidate=rl::ProgressPolicy::load(candidate_path);
    std::filesystem::create_directories(output);
    if(!fixture_path.empty()) {
        const auto bytes=cudarl::checkpoint::read(fixture_path);cudarl::checkpoint::Reader reader{bytes};
        if(reader.value<uint32_t>()!=0x43535431u) throw std::runtime_error("Invalid current fixture");
        const auto state=reader.value<p::State>();const int target=reader.value<int>();const auto current=reader.value<d::V>();reader.finish();
        std::ofstream result(std::filesystem::path(output)/"fixture.csv");result<<"mode,foods_in_60s,first_food_upper_seconds\n";
        for(bool augmented:{false,true}) {
            float first;const auto cases=run(candidate,augmented,64,state.seed,1200,current,&state,target,&first);
            result<<(augmented?"candidate":"original")<<','<<cases[24].foods<<','<<first<<'\n';
            std::printf("FIXTURE mode=%s foods=%d first_food_upper=%.2f\n",augmented?"candidate":"original",cases[24].foods,first);
        }
        return 0;
    }
    std::ofstream rows(std::filesystem::path(output)/"cases.csv"),summary(std::filesystem::path(output)/"summary.csv");
    rows<<"condition,seed,environment,seconds,original_foods,candidate_foods,original_gap_seconds,candidate_gap_seconds,original_low_seconds,candidate_low_seconds\n";
    summary<<"condition,current_cm_s,trials,seconds,original_fpm,candidate_fpm,delta,paired_ci_low,paired_ci_high,original_gaps_over_20s,candidate_gaps_over_20s,original_max_gap,candidate_max_gap\n";
    for(const auto& condition:suite) {
        std::vector<Case> before,after;
        for(uint32_t seed:seeds) {
            const auto a=run(candidate,false,n,seed,seconds*20,condition.current),b=run(candidate,true,n,seed,seconds*20,condition.current);
            for(int i=0;i<n;++i) {before.push_back(a[i]);after.push_back(b[i]);rows<<condition.name<<','<<seed<<','<<i<<','<<seconds<<','<<a[i].foods<<','<<b[i].foods<<','<<a[i].max_gap*.05<<','<<b[i].max_gap*.05<<','<<a[i].low_ground*.05<<','<<b[i].low_ground*.05<<'\n';}
        }
        double mean=0,variance=0,old_food=0,new_food=0,old_max=0,new_max=0;int old_gaps=0,new_gaps=0;const auto count=before.size();
        for(size_t i=0;i<count;++i) {mean+=after[i].foods-before[i].foods;old_food+=before[i].foods;new_food+=after[i].foods;old_gaps+=before[i].max_gap>400;new_gaps+=after[i].max_gap>400;old_max=std::max(old_max,before[i].max_gap*.05);new_max=std::max(new_max,after[i].max_gap*.05);}
        mean/=count;for(size_t i=0;i<count;++i) {const double error=after[i].foods-before[i].foods-mean;variance+=error*error;}
        const double rate=60./seconds,ci=rate*1.96*std::sqrt(variance/(count-1)/count);mean*=rate;
        summary<<std::setprecision(10)<<condition.name<<','<<d::norm(condition.current)*100<<','<<count<<','<<seconds<<','<<old_food*rate/count<<','<<new_food*rate/count<<','<<mean<<','<<mean-ci<<','<<mean+ci<<','<<old_gaps<<','<<new_gaps<<','<<old_max<<','<<new_max<<'\n';
        std::printf("PAIRED %s original=%.4f candidate=%.4f delta=%.4f ci=[%.4f,%.4f] gaps=%d/%d\n",condition.name.c_str(),old_food*rate/count,new_food*rate/count,mean,mean-ci,mean+ci,old_gaps,new_gaps);std::fflush(stdout);
    }
    if(source!=cudarl::checkpoint::read(candidate_path)) throw std::runtime_error("Candidate file changed");
    if(!rows || !summary) throw std::runtime_error("Could not write progress evaluation");return 0;
}
}
