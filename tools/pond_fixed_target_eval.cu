#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include "pond_fixed_target.h"
#include "cuda/device.h"
#include "cuda/cuda_memory.h"
#include "rl/pond_selector_policy.h"
#include "nn/mlp_gpu.h"
namespace ft=cudarl::fixed_target;
namespace rl=cudarl::rl;
namespace {
constexpr const char* names[9]={"calm","cm05_upstream","cm05_downstream","cm05_cross_left","cm05_cross_right",
    "cm10_upstream","cm10_downstream","cm10_cross_left","cm10_cross_right"};
struct Tag {uint32_t seed;int environment,folded,condition,passive;};
__global__ void observations(const ft::Trial* trials,float* obs,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) ft::observe(trials[i],obs+i*33);
}
__global__ void advance(ft::Trial* trials,ft::Metrics* metrics,const float* actions,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) ft::step(trials[i],metrics[i],actions+i*4);
}
float median(std::vector<float> values) {
    if(values.empty()) return -1;std::sort(values.begin(),values.end());const size_t mid=values.size()/2;
    return values.size()%2?values[mid]:(values[mid-1]+values[mid])*.5f;
}
void record(const std::filesystem::path& output,int seconds,const std::vector<Tag>& tags,
            const std::vector<ft::Trial>& initial,const std::vector<ft::Trial>& end,const std::vector<ft::Metrics>& metrics) {
    std::ofstream cases(output/("cases-"+std::to_string(seconds)+"s.csv"));
    cases<<"seed,environment,pose,controller,condition,target,arrival_s,elapsed_s,initial_distance_cm,min_distance_cm,remaining_cm,max_low_speed_s,max_no_progress_s,incidental_foods,invalid,mean_ground_cm_s,mean_water_relative_cm_s,mean_toward_cm_s,current_x_m_s,current_z_m_s,initial_x,initial_y,initial_z,goal_x,goal_y,goal_z,final_x,final_y,final_z\n";
    cases<<std::setprecision(7);
    for(size_t i=0;i<tags.size();++i) {
        const auto& t=tags[i];const auto& m=metrics[i];const auto& a=initial[i];const auto& b=end[i];
        cases<<t.seed<<','<<t.environment<<','<<(t.folded?"folded":"normal")<<','<<(t.passive?"passive":"learned")<<','<<names[t.condition]<<','<<a.target<<','
            <<(m.arrival?m.arrival*.05f:-1)<<','<<m.ticks*.05f<<','<<m.initial_distance*4<<','<<m.min_distance*4<<','<<m.remaining*4<<','
            <<m.max_low*.05f<<','<<m.max_no_progress*.05f<<','<<m.incidental<<','<<m.invalid<<','
            <<m.ground/std::max(1,m.ticks)*100<<','<<m.relative/std::max(1,m.ticks)*100<<','<<m.toward/std::max(1,m.ticks)*100<<','
            <<a.current.x<<','<<a.current.z<<','<<a.state.body.position.x<<','<<a.state.body.position.y<<','<<a.state.body.position.z<<','
            <<a.goal.x<<','<<a.goal.y<<','<<a.goal.z<<','<<b.state.body.position.x<<','<<b.state.body.position.y<<','<<b.state.body.position.z<<'\n';
    }
    std::ofstream summary(output/("summary-"+std::to_string(seconds)+"s.csv"));
    summary<<"pose,controller,condition,trials,arrived,success_percent,median_success_s,capped_mean_arrival_s,low_speed_10s,no_progress_10s,failed,failed_low_speed_10s,failed_no_progress_10s,mean_incidental_foods\n";
    for(int folded=0;folded<2;++folded) for(int passive=0;passive<2;++passive) for(int condition=0;condition<9;++condition) {
        int count=0,arrived=0,low=0,stalled=0,failed_low=0,failed_stalled=0,incidental=0;double capped=0;std::vector<float> times;
        for(size_t i=0;i<tags.size();++i) {
            const auto& tag=tags[i];if(tag.folded!=folded || tag.passive!=passive || tag.condition!=condition) continue;
            const auto& m=metrics[i];++count;if(m.arrival) {++arrived;times.push_back(m.arrival*.05f);}
            capped+=m.arrival?m.arrival*.05:seconds;low+=m.max_low>=200;stalled+=m.max_no_progress>=200;incidental+=m.incidental;
            if(!m.arrival) {failed_low+=m.max_low>=200;failed_stalled+=m.max_no_progress>=200;}
        }
        summary<<(folded?"folded":"normal")<<','<<(passive?"passive":"learned")<<','<<names[condition]<<','<<count<<','<<arrived<<','<<100.*arrived/count<<','
            <<median(times)<<','<<capped/count<<','<<low<<','<<stalled<<','<<count-arrived<<','<<failed_low<<','<<failed_stalled<<','<<double(incidental)/count<<'\n';
        if(!passive) std::printf("FIXED_TARGET seconds=%d pose=%s condition=%s arrived=%d/%d median=%.2f failed_low=%d failed_no_progress=%d\n",
            seconds,folded?"folded":"normal",names[condition],arrived,count,median(times),failed_low,failed_stalled);
    }
    if(!cases || !summary) throw std::runtime_error("Failed writing fixed-target report");std::fflush(stdout);
}
}
int main(int argc,char** argv) {
    try {
        std::string policy_path,output_path;int count=128,seconds=300;std::vector<uint32_t> seeds;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];if(arg=="--help") {std::puts("pond_fixed_target_eval --policy FROZEN_POLICY --output NEW_DIR [--trials 128] [--seconds 300] [--seed N (repeatable)]\nRead-only CUDA motor diagnostic. Fixed target, normal/folded poses, nine currents, learned/passive controls. Stops at target arrival; no training or episode reset.");return 0;}
            if(i+1>=argc) throw std::invalid_argument("Missing argument value");const std::string value=argv[++i];
            if(arg=="--policy") policy_path=value;else if(arg=="--output") output_path=value;else if(arg=="--trials") count=std::stoi(value);
            else if(arg=="--seconds") seconds=std::stoi(value);else if(arg=="--seed") seeds.push_back(static_cast<uint32_t>(std::stoul(value,nullptr,0)));
            else throw std::invalid_argument("Unknown fixed-target argument");
        }
        if(policy_path.empty() || output_path.empty() || count<16 || count>256 || seconds<60 || seconds>300) throw std::invalid_argument("Supply policy, new output directory, 16..256 trials and 60..300 seconds");
        if(seeds.empty()) seeds={0xF17ED101u,0xF17ED102u};if(seeds.size()>4) throw std::invalid_argument("At most four seeds");
        const std::filesystem::path output=std::filesystem::absolute(output_path);
        if(std::filesystem::exists(output)) throw std::invalid_argument("Use a new output directory to preserve previous evidence");
        const auto original=cudarl::checkpoint::read(policy_path);const auto policy=rl::SelectorPolicy::load(policy_path);
        const auto device=cudarl::probe_device();if(!device.available) throw std::runtime_error(device.error);
        std::vector<ft::Trial> initial;std::vector<Tag> tags;
        for(auto seed:seeds) for(int folded=0;folded<2;++folded) for(int condition=0;condition<9;++condition) for(int passive=0;passive<2;++passive) for(int i=0;i<count;++i) {
            initial.push_back(ft::initial(seed,i,folded,condition,passive));tags.push_back({seed,i,folded,condition,passive});
        }
        const int n=static_cast<int>(initial.size());std::vector<ft::Metrics> metrics;for(const auto& t:initial) metrics.push_back(ft::metrics(t));
        device_ptr<ft::Trial> trials(n);device_ptr<ft::Metrics> results(n);device_ptr<float> obs(static_cast<size_t>(n)*33);
        trials.copy_from_host(initial.data(),n);results.copy_from_host(metrics.data(),n);
        cudarl::nn::GpuMlp motor(rl::swimmer_specification(4,33),n,7);motor.set_params(policy.motor);
        std::filesystem::create_directories(output);cudarl::checkpoint::Writer start;start.value(uint32_t{0x46544731});start.value(uint32_t{1});start.array(initial);start.array(tags);
        cudarl::checkpoint::write_atomic((output/"initial.bin").string(),start.data);
        std::printf("FIXED_TARGET device=%s cases=%d seconds=%d motor_update=%d no_training=true\n",device.name.c_str(),n,seconds,policy.motor_update);std::fflush(stdout);
        std::vector<ft::Trial> end(n);
        for(int tick=1;tick<=seconds*20;++tick) {
            observations<<<(n+127)/128,128>>>(trials.get(),obs.get(),n);CUDA_CHECK_THROW(cudaGetLastError());motor.forward(obs.get(),n);
            advance<<<(n+63)/64,64>>>(trials.get(),results.get(),motor.output(),n);CUDA_CHECK_THROW(cudaGetLastError());
            if(tick==1200 || tick==seconds*20) {
                CUDA_CHECK_THROW(cudaDeviceSynchronize());results.copy_to_host(metrics.data(),n);trials.copy_to_host(end.data(),n);
                for(const auto& m:metrics) if(m.invalid) throw std::runtime_error("Non-finite fixed-target trajectory");
                record(output,tick/20,tags,initial,end,metrics);
            }
        }
        cudarl::checkpoint::Writer finish;finish.value(uint32_t{0x46544731});finish.value(uint32_t{1});finish.array(end);finish.array(metrics);
        cudarl::checkpoint::write_atomic((output/"final.bin").string(),finish.data);
        if(motor.get_params()!=policy.motor || original!=cudarl::checkpoint::read(policy_path)) throw std::runtime_error("Fixed-target evaluation changed protected weights");
        std::ofstream verification(output/"verification.json");verification<<"{\"cases\":"<<n<<",\"seconds\":"<<seconds<<",\"motor_unchanged\":true,\"source_unchanged\":true,\"training\":false,\"selector_executed\":false,\"autoreset\":false}\n";
        if(!verification) throw std::runtime_error("Could not write fixed-target verification");std::puts("FIXED_TARGET_COMPLETE motor_unchanged=true source_unchanged=true no_training=true");return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Fixed-target diagnostic: %s\n",e.what());return 1;}
}
