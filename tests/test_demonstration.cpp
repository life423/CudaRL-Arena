#include "check.h"
#include "motor/demonstration.h"
#include <limits>
namespace demo=cudarl::demonstration;namespace task=cudarl::straight_motor;
TEST_CASE("demonstration: feedback depends on requested and actual speed with finite physical commands") {
    auto world=task::initial_world(0);auto config=task::config_for_speed(.004f);task::ForwardTask request;
    CHECK(task::begin_forward(world,config,request));float observation[42]{};CHECK(task::observe(request,world,config,observation));
    demo::Label stopped,cruising,fast;CHECK(demo::label(observation,stopped));
    observation[12]=.004f/.02f;CHECK(demo::label(observation,cruising));
    observation[12]=.012f/.02f;CHECK(demo::label(observation,fast));
    CHECK(stopped.amplitude>cruising.amplitude&&cruising.amplitude>fast.amplitude);
    for(const auto& x:{stopped,cruising,fast}){CHECK(x.amplitude>=demo::kMinimumAmplitude&&x.amplitude<=demo::kMaximumAmplitude);
        CHECK(std::isfinite(x.action.latent[0])&&std::isfinite(x.action.latent[1]));
        CHECK(std::fabs(.35f*(x.action.command[0]+1)-x.amplitude)<1e-7f);
        CHECK(std::fabs(.25f+.625f*(x.action.command[1]+1)-1)<1e-7f);}
    CHECK(world.attempted_substeps==0);
}
TEST_CASE("demonstration: no hidden teacher state and body-frame rotations preserve labels") {
    float a[42]{},b[42]{};a[3]=.006f/.02f;a[6]=1;a[12]=.003f/.02f;
    b[5]=a[3];b[8]=1;b[14]=a[12];b[34]=.5f;b[36]=1;b[38]=.4f;
    demo::Label first,second;CHECK(demo::label(a,first));CHECK(demo::label(b,second));
    CHECK(std::fabs(first.amplitude-second.amplitude)<1e-7f);
    CHECK(first.action.latent[0]==second.action.latent[0]);
    a[3]=.004f/.02f;CHECK(demo::label(a,first));a[3]=.008f/.02f;CHECK(demo::label(a,second));
    CHECK(second.amplitude>first.amplitude);
}
TEST_CASE("demonstration: invalid observations never produce a usable label") {
    demo::Label label;float observation[42]{};CHECK(!demo::label(nullptr,label));CHECK(!demo::label(observation,label));
    observation[3]=.006f/.02f;observation[6]=1;CHECK(demo::label(observation,label));
    observation[3]=.012f/.02f;CHECK(!demo::label(observation,label));observation[3]=.006f/.02f;
    observation[36]=std::numeric_limits<float>::quiet_NaN();CHECK(!demo::label(observation,label));
}
#if CUDARL_WITH_CUDA
#include "gpu_util.h"
#include "nn/loss.h"
GPU_TEST_CASE("demonstration: CUDA supervised gradient matches latent-action MSE without training") {
    using cudarl::test::to_device;using cudarl::test::to_host;
    float observation[42]{};observation[3]=.006f/.02f;observation[6]=1;demo::Label label;CHECK(demo::label(observation,label));
    const std::vector<float> targets{label.action.latent[0],label.action.latent[1]};
    const std::vector<float> predictions{targets[0]+.3f,targets[1]-.2f};
    auto y=to_device(predictions),target=to_device(targets);device_ptr<float> gradient(2);
    cudarl::nn::mse_loss_grad(y.get(),target.get(),gradient.get(),2);const auto actual=to_host(gradient,2);
    CHECK(std::fabs(actual[0]-.3f)<1e-7f&&std::fabs(actual[1]+.2f)<1e-7f);
}
#endif

#include "motor/demonstration_data.h"
namespace {
std::vector<uint8_t> accepted_fixture(bool bad_label=false,bool missing_lineage=false){
    std::vector<float> obs(demo::kAcceptedRows*42),target(demo::kAcceptedRows*2);
    for(int row=0;row<demo::kAcceptedRows;++row){float* sample=obs.data()+row*42;sample[3]=(.004f+.002f*float((row%6)/2))/.02f;sample[6]=1;sample[12]=.003f/.02f;demo::Label label;CHECK(demo::label(sample,label));
        target[row*2]=label.action.latent[0];target[row*2+1]=label.action.latent[1];}
    if(bad_label)target[1]+=.1f;
    cudarl::checkpoint::Writer out;out.value(uint32_t{0x4344454D});out.value(uint32_t{1});out.value(demo::kVersion);out.value(uint32_t{demo::kAcceptedRows});out.value(uint64_t{missing_lineage?0u:123u});out.value(uint64_t{456});out.array(obs);out.array(target);return out.data;
}
}
TEST_CASE("demonstration_data: exact accepted slice preserves goal order and label dimensions") {
    const auto samples=demo::accepted_examples(accepted_fixture());CHECK(samples.count()==960);
    CHECK(samples.observation.size()==960*42&&samples.target.size()==960*2);
    CHECK(samples.source_dataset_checksum==123&&samples.source_gate_checksum==456);
}
TEST_CASE("demonstration_data: missing lineage or altered labels fail before any physical work") {
    for(const auto& invalid:{accepted_fixture(true),accepted_fixture(false,true)}){bool rejected=false;
        try{(void)demo::accepted_examples(invalid);}catch(const std::runtime_error&){rejected=true;}CHECK(rejected);}
}
