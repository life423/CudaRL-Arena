// Same resident collection entry as the separate candidate motor trainer. This
// executable constructs only forward networks and uses the archived weights/RNG.
#define POND_RESIDUAL_COLLECTOR_ONLY
#include "../pond-residual-ppo/train.cu"
#include "data.h"
namespace pond_collection {
namespace rt=residual_training;
static_assert(sizeof(Trace)==192 && Rows==32768, "Frozen collection layout");
std::string source(){return residual_source_hash;}
struct Inference {
    nn::GpuMlp base{actor_spec(D),MB,default_seed},residual{rt::residual_spec(),MB,default_seed+7},critic{critic_spec(),MB,default_seed+1};
};
struct Fixed {Package motor;};
void collect_once(const Inputs& input,Samples& out,std::vector<Gap>& gaps,Timing& timing,void(*issued)(int),void(*complete)(int)){
    const auto start=rt::Clock::now();Fixed fixed{load_package(input.package)};
    CUDA_CHECK_THROW(cudaSetDevice(0));Stream stream;const auto s=stream.value;
    Buffer b(true);rt::Extra extra;Inference net;device_ptr<pr::ControlClearance> classifications{Rows};
    timing.setup_ms=1000*rt::elapsed(start);
    const auto upload_start=rt::Clock::now();
    net.base.set_params(fixed.motor.weights);net.residual.set_params(input.residual);net.critic.set_params(input.critic);
    std::vector<float> logs(A,pp::fixed_log_std());b.log_std.copy_from_host(logs.data(),logs.size());
    restore(b.worlds,input.worlds);restore_tasks(b,input.tasks,rt::program);
    rt::clear_batch(b,extra,s);CUDA_CHECK_THROW(cudaStreamSynchronize(s));timing.input_transfer_ms=1000*rt::elapsed(upload_start);
    rt::TimeBudget budget;rt::StageAccounting accounting{};accounting.update=12;
    rt::CollectionTiming costs;costs.on_issue=issued;costs.on_complete=complete;
    const auto collect_start=rt::Clock::now();
    timing.physics_ms=rt::collect(b,extra,net,fixed,accounting,s,budget,classifications.get(),&costs);
    timing.collection_wall_ms=1000*rt::elapsed(collect_start);timing.inference_ms=costs.inference_ms;
    const auto read_start=rt::Clock::now();auto h=batch(b,s,64);
    out.obs=std::move(h.observation);out.latent=std::move(h.latent);out.logp=std::move(h.logp);out.value=std::move(h.value);
    out.reward=std::move(h.reward);out.done=std::move(h.done);out.truncated=std::move(h.truncated);
    out.final_value=std::move(h.final_value);out.last_value=std::move(h.last_value);
    out.attempted=std::move(h.attempted);out.substeps=std::move(h.substeps);out.reason=std::move(h.reason);out.trace=std::move(h.trace);
    out.residual=download(extra.action,Rows*rt::Y,s);out.mask=download(extra.mask,Rows,s);out.base=download(extra.base,Rows*A,s);
    const auto raw=download(classifications,Rows,s);timing.readback_ms=1000*rt::elapsed(read_start);
    gaps.reserve(Rows);for(const auto& c:raw)gaps.push_back({c.lower_m,uint32_t(c.precise),c.rejected_contacts,c.exact_steps,c.exact_queries});
}
}
