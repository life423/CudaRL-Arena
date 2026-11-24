#define POND_RESIDUAL_COLLECTOR_ONLY
#include "../pond-residual-four-ppo/train.cu"
#include "../pond-collection-replay/data.h"

namespace pond_collection {
namespace rt=residual_training;
constexpr int archived_update=12;
static_assert(N==512 && T==64 && Rows==32768 && sizeof(Trace)==192);
static_assert(rt::Y==4 && rt::W==42 && rt::rp::parameters==756);
static_assert(sizeof(pr::World)==1384 && sizeof(ps::SignedCourseTask)==368);

std::string source(){return residual_source_hash;}

void probe_module(void(*phase)(const char*)){
    phase("cuda_context");
    CUDA_CHECK_THROW(cudaSetDevice(0));
    cudaFuncAttributes attributes{};
    phase("observation_kernel_load");
    CUDA_CHECK_THROW(cudaFuncGetAttributes(&attributes,observe_kernel));
    phase("physics_kernel_load");
    CUDA_CHECK_THROW(cudaFuncGetAttributes(&attributes,step_kernel));
    phase("reset_kernel_load");
    CUDA_CHECK_THROW(cudaFuncGetAttributes(&attributes,reset_terminal_kernel));
    phase("initialize_kernel_load");
    CUDA_CHECK_THROW(cudaFuncGetAttributes(&attributes,initialize_kernel));
}

struct Inference {
    nn::GpuMlp base,residual,critic;
    Inference(const Package& motor,const Inputs& input,cudaStream_t stream)
        :base(nn::GpuMlp::for_inference(actor_spec(D),N,motor.weights,stream)),
         residual(nn::GpuMlp::for_inference(rt::residual_spec(),N,input.residual,stream)),
         critic(nn::GpuMlp::for_inference(critic_spec(),N,input.critic,stream)){}
};

void collect_once(const Inputs& input,Samples& out,std::vector<Gap>& gaps,Timing& timing,
                  void(*issued)(int),void(*complete)(int),void(*phase)(const char*)){
    const auto start=rt::Clock::now();
    phase("motor_load");
    struct Fixed {Package motor;};
    const Fixed fixed{load_package(input.package)};
    phase("cuda_context");
    CUDA_CHECK_THROW(cudaSetDevice(0));
    Stream stream;
    const auto s=stream.value;
    phase("collection_buffers");
    Buffer b(true,BufferUse::Collection);
    rt::Extra extra(BufferUse::Collection);
    device_ptr<pr::ControlClearance> classifications{Rows};
    phase("inference_networks");
    Inference net(fixed.motor,input,s);
    timing.setup_ms=1000*rt::elapsed(start);

    const auto upload=rt::Clock::now();
    phase("input_upload");
    const std::vector<float> logs(A,pp::fixed_log_std());
    b.log_std.copy_from_host(logs.data(),logs.size());
    restore(b.worlds,input.worlds);
    restore_tasks(b,input.tasks,rt::program);
    rt::clear_batch(b,extra,s);
    CUDA_CHECK_THROW(cudaStreamSynchronize(s));
    timing.input_transfer_ms=1000*rt::elapsed(upload);

    rt::TimeBudget budget;
    rt::StageAccounting accounting{};
    accounting.update=archived_update;
    rt::CollectionTiming costs;
    costs.on_issue=issued;
    costs.on_complete=complete;
    const auto collect_start=rt::Clock::now();
    phase("first_control_preparation");
    // The four-output collector derives both sampler streams from the update
    // and tick, including the original independent pitch salt.
    timing.physics_ms=rt::collect(b,extra,net,fixed,accounting,s,budget,classifications.get(),&costs);
    timing.collection_wall_ms=1000*rt::elapsed(collect_start);
    timing.inference_ms=costs.inference_ms;

    phase("readback");
    const auto read_start=rt::Clock::now();
    auto host=batch(b,s,T);
    out.obs=std::move(host.observation);
    out.latent=std::move(host.latent);
    out.logp=std::move(host.logp);
    out.value=std::move(host.value);
    out.reward=std::move(host.reward);
    out.done=std::move(host.done);
    out.truncated=std::move(host.truncated);
    out.final_value=std::move(host.final_value);
    out.last_value=std::move(host.last_value);
    out.attempted=std::move(host.attempted);
    out.substeps=std::move(host.substeps);
    out.reason=std::move(host.reason);
    out.trace=std::move(host.trace);
    out.residual=download(extra.action,Rows*rt::Y,s);
    out.mask=download(extra.mask,Rows,s);
    out.base=download(extra.base,Rows*A,s);
    const auto raw=download(classifications,Rows,s);
    timing.readback_ms=1000*rt::elapsed(read_start);
    gaps.reserve(Rows);
    for(const auto& c:raw)gaps.push_back({c.lower_m,uint32_t(c.precise),c.rejected_contacts,c.exact_steps,c.exact_queries});
    phase("cuda_cleanup");
}
} // namespace pond_collection
