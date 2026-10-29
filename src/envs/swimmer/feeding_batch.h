#ifndef CUDARL_FEEDING_BATCH_H
#define CUDARL_FEEDING_BATCH_H
#include "feeding.h"
#include "nn/mlp_gpu.h"
#include "rl/progress_selector_policy.h"
namespace cudarl::feeding {
class Batch {
    int n_,ticks_=0;uint32_t seed_;Context context_;
    device_ptr<State> states_;device_ptr<float> observations_,features_;
    nn::GpuMlp selector_;
    void prepare(cudaStream_t stream);
public:
    Batch(int n,uint32_t seed,swimmer::Body unused={});
    void configure(const std::vector<float>& selector,const Context& context,cudaStream_t stream,bool reset=false);
    void step(const float* actions,swimmer::Outputs out,cudaStream_t stream,bool autoreset=true);
    const float* obs() const {return observations_.get();}
    const State* states() const {return states_.get();}
    const Context& context() const {return context_;}
    State state(cudaStream_t stream) const;
    void set_state(const State& state,cudaStream_t stream);
    void save(checkpoint::Writer& out) const;
    void load(checkpoint::Reader& in);
};
}
#endif
