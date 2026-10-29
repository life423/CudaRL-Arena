#include "envs/swimmer/feeding_batch.h"
#include "rl/feeding_policy.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fd=cudarl::feeding;
namespace d=cudarl::deep;
namespace rl=cudarl::rl;
namespace cp=cudarl::checkpoint;
constexpr int worlds=16,seconds=120,steps=2400,window=100;
constexpr std::array<uint32_t,8> seeds={0x631A4B01u,0x7D92E653u,0x438BC7A9u,0x95C1234Du,0x2B6F84D7u,0xC17A93B5u,0x5E209F6Bu,0xA83D7C19u};
struct Snapshot {
    rl::ProgressPolicy source;
    std::vector<float> actor;
    int learner_update=0,validated_update=0;
};
Snapshot read_snapshot(const std::string& path) {
    const auto bytes=cp::read(path);cp::Reader in{bytes};
    if(in.value<uint32_t>()!=0x46454544u || in.value<uint32_t>()!=2u || in.value<uint32_t>()!=sizeof(fd::State))
        throw std::runtime_error("Expected compatible full CUDA feeding snapshot");
    const int n=in.value<int>();if(n<16 || n>2048 || n%8) throw std::runtime_error("Invalid saved population");
    Snapshot out;out.source=rl::ProgressPolicy::read(in);
    (void)in.value<fd::State>();const auto context=in.value<fd::Context>();
    const auto parameters=static_cast<size_t>(rl::swimmer_specification(4,fd::kObs).num_params());
    out.actor=in.array<float>(parameters);out.validated_update=in.value<int>();
    const auto training=in.array<uint8_t>(128000000);in.finish();cp::Reader learner{training};out.learner_update=learner.value<int>();
    if(context.count<0 || context.count>32 || out.actor.size()!=parameters || out.validated_update<0 || out.learner_update<out.validated_update)
        throw std::runtime_error("Invalid saved actor metadata");
    for(float weight:out.actor) if(!std::isfinite(weight)) throw std::runtime_error("Non-finite saved actor");
    return out;
}
std::string fingerprint(const std::vector<float>& actor) {
    uint64_t hash=14695981039346656037ull;const auto* p=reinterpret_cast<const uint8_t*>(actor.data());
    for(size_t i=0;i<actor.size()*sizeof(float);++i) {hash^=p[i];hash*=1099511628211ull;}
    std::ostringstream out;out<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;return out.str();
}
void export_actor(const Snapshot& snapshot,const std::filesystem::path& path) {
    cp::Writer out;out.value(uint32_t{0x46485031});out.value(uint32_t{1});snapshot.source.write(out);
    out.value(snapshot.validated_update);out.array(snapshot.actor);cp::write_atomic(path.u8string(),out.data);
}
struct Metrics {
    int foods=0,first_food_tick=-1,max_gap=0,available_gap=0,max_available_gap=0;
    int immobility_events=0,qualified_ticks=0,max_immobile_ticks=0,immobile_active=0;
    int empty_ticks=0,final_gap=0,final_food_count=0,created_foods=0,cap_ticks=0;
    int invalid_body_ticks=0,invalid_food_ticks=0,previous_foods=0,available_streak=0;
    d::V history[window+1]{};
};
__device__ bool finite_vector(d::V v) {return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);}
__global__ void begin_observation(const fd::State* states,Metrics* metrics,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    Metrics value{};for(auto& point:value.history) point=states[i].body.position;metrics[i]=value;
}
__global__ void observe(const fd::State* states,Metrics* metrics,int n,int tick) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;const auto& s=states[i];auto& m=metrics[i];
    const int active=fd::count(s),collected=s.eaten-m.previous_foods;
    m.foods=s.eaten;m.previous_foods=s.eaten;
    if(collected && m.first_food_tick<0) m.first_food_tick=tick;
    m.max_gap=max(m.max_gap,collected?m.final_gap+1:s.since_food);m.final_gap=s.since_food;
    if(active==0) ++m.empty_ticks;
    const int available_period=collected || active?m.available_gap+1:0;
    m.max_available_gap=max(m.max_available_gap,available_period);
    m.available_gap=active && !collected?available_period:0;
    m.available_streak=active && !collected?m.available_streak+1:0;
    m.history[tick%(window+1)]=s.body.position;
    bool immobile=tick>=window && m.available_streak>=window;
    if(immobile) for(const auto point:m.history) if(d::norm(point-s.body.position)>.1f) {immobile=false;break;}
    if(immobile) {
        if(!m.immobile_active) ++m.immobility_events;
        ++m.qualified_ticks;m.max_immobile_ticks=max(m.max_immobile_ticks,m.qualified_ticks+window-1);
    } else m.qualified_ticks=0;
    m.immobile_active=immobile?1:0;
    bool valid=!s.invalid && finite_vector(s.body.position) && isfinite(s.body.orientation.x) && isfinite(s.body.orientation.y) && isfinite(s.body.orientation.z) && isfinite(s.body.orientation.w);
    for(float v:s.body.velocity) valid=valid && isfinite(v);
    for(float angle:s.body.angle) valid=valid && isfinite(angle);
    if(!valid) ++m.invalid_body_ticks;
    bool food_valid=true;
    for(const auto& p:s.pellets) if(p.active) {
        const float r2=fd::radial(p.position),gx=27.f*p.position.x*r2*r2/(4.6f*4.6f),gz=27.f*p.position.z*r2*r2/(3.5f*3.5f);
        const float normal_length=sqrtf(1+gx*gx+gz*gz);
        const float gap=(p.position.y-d::pond_floor(p.position.x,p.position.z))*d::scale/normal_length-fd::radius;
        food_valid=food_valid && finite_vector(p.position) && finite_vector(p.velocity) && p.position.y<=.01f && r2<=.9401f*.9401f && gap>=-.0001f;
    }
    if(!food_valid) ++m.invalid_food_ticks;
    if(active>=16) ++m.cap_ticks;
    m.final_food_count=active;m.created_foods=static_cast<int>(s.next_id)-1;
}
#include "feeding_heldout_monitor_selfcheck.cuh"

std::vector<Metrics> evaluate(const Snapshot& snapshot,uint32_t seed,int duration_steps) {
    int low=0,high=0;CUDA_CHECK_THROW(cudaDeviceGetStreamPriorityRange(&low,&high));
    cudaStream_t stream{};CUDA_CHECK_THROW(cudaStreamCreateWithPriority(&stream,cudaStreamNonBlocking,low));
    try {
        fd::Context context;context.current={.005f,0,0}; // Common held-out context; no saved player positions.
        fd::Batch batch(worlds,seed);batch.configure(snapshot.source.actor,context,stream,true);
        cudarl::nn::GpuMlp actor(rl::swimmer_specification(4,fd::kObs),worlds,7);actor.set_params(snapshot.actor);
        device_ptr<Metrics> metrics(worlds);begin_observation<<<1,32,0,stream>>>(batch.states(),metrics.get(),worlds);CUDA_CHECK_THROW(cudaGetLastError());
        for(int tick=1;tick<=duration_steps;++tick) {
            actor.forward(batch.obs(),worlds,stream);batch.step(actor.output(),{},stream,false,true);
            observe<<<1,32,0,stream>>>(batch.states(),metrics.get(),worlds,tick);CUDA_CHECK_THROW(cudaGetLastError());
            if(tick%20==0) CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        }
        std::vector<Metrics> result(worlds);CUDA_CHECK_THROW(cudaMemcpyAsync(result.data(),metrics.get(),result.size()*sizeof(Metrics),cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        if(actor.get_params()!=snapshot.actor) throw std::runtime_error("Frozen actor changed during evaluation");
        CUDA_CHECK_THROW(cudaStreamDestroy(stream));return result;
    } catch(...) {cudaStreamDestroy(stream);throw;}
}
const char* group(int world) {constexpr const char* names[]={"retention","calm","steady","fluctuating"};return names[world%4];}
void rows(std::ofstream& out,const char* policy,uint32_t seed,const std::vector<Metrics>& result) {
    for(int i=0;i<worlds;++i) {
        const auto& m=result[static_cast<size_t>(i)];
        out<<policy<<','<<seed<<','<<i<<','<<group(i)<<','<<m.foods<<','<<m.foods*60./seconds<<','
           <<(m.first_food_tick<0?-1.:m.first_food_tick*.05)<<','<<(m.first_food_tick<0?1:0)<<','
           <<m.max_gap*.05<<','<<m.max_available_gap*.05<<','<<m.immobility_events<<','<<m.max_immobile_ticks*.05<<','<<m.immobile_active<<','
           <<m.empty_ticks*.05<<','<<m.final_gap*.05<<','<<m.final_food_count<<','<<m.created_foods<<','<<m.cap_ticks*.05<<','
           <<m.invalid_body_ticks<<','<<m.invalid_food_ticks<<'\n';
    }
}
void metadata(const std::filesystem::path& output,const Snapshot& before,const Snapshot& after) {
    std::ofstream out(output/"metadata.json");
    out<<"{\n  \"protocol\":\"autonomous-heldout-v1\",\n  \"seconds\":"<<seconds<<",\n  \"worlds\":"<<worlds<<",\n  \"seeds\":[";
    for(size_t i=0;i<seeds.size();++i) out<<(i?",":"")<<seeds[i];
    out<<"],\n  \"before\":{\"learner_update\":"<<before.learner_update<<",\"validated_update\":"<<before.validated_update<<",\"actor_fnv64\":\""<<fingerprint(before.actor)<<"\"},\n"
       <<"  \"after\":{\"learner_update\":"<<after.learner_update<<",\"validated_update\":"<<after.validated_update<<",\"actor_fnv64\":\""<<fingerprint(after.actor)<<"\"},\n"
       <<"  \"self_check\":true,\n  \"actor_weights_unchanged\":true,\n  \"source_reference_unchanged\":true\n}\n";
    if(!out) throw std::runtime_error("Could not write metadata");
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--monitor-self-check") {passive_monitor_self_check();return 0;}
        if(argc!=4) throw std::runtime_error("Usage: feeding_heldout_eval before-full-save after-full-save output-folder");
        const auto before=read_snapshot(argv[1]),after=read_snapshot(argv[2]);
        const auto output=std::filesystem::u8path(argv[3]);
        for(const auto* file:{"cases.csv","metadata.json","before-policy.bin","after-policy.bin"})
            if(std::filesystem::exists(output/file)) throw std::runtime_error("Use a fresh output folder; refusing to overwrite evaluation evidence");
        if(before.source.actor!=after.source.actor || before.source.baseline.motor!=after.source.baseline.motor)
            throw std::runtime_error("Snapshots do not share inherited selector and swimming provenance");
        const auto before_hash=fingerprint(before.actor),after_hash=fingerprint(after.actor);
        std::printf("FROZEN_ACTORS before_validated=%d before_learner=%d after_validated=%d after_learner=%d identical=%d\n",before.validated_update,before.learner_update,after.validated_update,after.learner_update,before.actor==after.actor?1:0);std::fflush(stdout);
        // Separate development seed: verify identical-policy runs before touching the held-out seeds.
        const auto first=evaluate(before,0x11D019E3u,200),second=evaluate(before,0x11D019E3u,200);
        if(first.size()!=second.size() || std::memcmp(first.data(),second.data(),first.size()*sizeof(Metrics)))
            throw std::runtime_error("Identical-policy evaluation is not bit-identical");
        std::puts("IDENTICAL_POLICY_SELF_CHECK PASS");std::fflush(stdout);
        std::filesystem::create_directories(output);
        export_actor(before,output/"before-policy.bin");export_actor(after,output/"after-policy.bin");
        std::ofstream out(output/"cases.csv");out<<std::fixed<<std::setprecision(6)
            <<"policy,seed,world,group,foods,food_per_minute,first_food_seconds,first_food_censored,max_gap_seconds,max_available_gap_seconds,immobility_events,max_immobility_seconds,terminal_immobile,empty_seconds,final_gap_seconds,final_food_count,created_foods,cap_seconds,invalid_body_ticks,invalid_food_ticks\n";
        for(const auto seed:seeds) {
            const auto a=evaluate(before,seed,steps),b=evaluate(after,seed,steps);rows(out,"before",seed,a);rows(out,"after",seed,b);out.flush();
            if(!out) throw std::runtime_error("Could not write raw evaluation cases");
            std::printf("HELDOUT_SEED %u complete (%d frozen trials)\n",seed,worlds*2);std::fflush(stdout);
        }
        if(fingerprint(before.actor)!=before_hash || fingerprint(after.actor)!=after_hash) throw std::runtime_error("Frozen policy changed");
        metadata(output,before,after);std::puts("FROZEN_HELDOUT COMPLETE: no training or policy replacement performed");return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"FROZEN_HELDOUT FAILED: %s\n",e.what());return 1;}
}
