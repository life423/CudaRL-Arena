#include "check.h"
#include "motor/demonstration.h"
#include <limits>
namespace demo=cudarl::demonstration;namespace task=cudarl::straight_motor;
TEST_CASE("demonstration_braking: reverse authority tapers continuously into forward regulation") {
    float observation[42]{};observation[3]=.004f/.02f;observation[6]=1;
    float previous=-2;
    for(float velocity:{.008f,.0048f,.0046f,.0044f,.004f,.003f}){
        observation[12]=velocity/.02f;demo::BrakingLabel out;
        CHECK(demo::feedback_brake(observation,out));CHECK(out.cadence>=previous-1e-5f);
        CHECK(out.amplitude>=.05f&&out.amplitude<=.65f&&out.cadence>=-1&&out.cadence<=1);
        if(velocity>=.0048f)CHECK(std::fabs(out.cadence+1)<1e-5f&&std::fabs(out.amplitude-.6f)<1e-5f);
        if(velocity<=.0044f)CHECK(std::fabs(out.cadence-1)<1e-5f);
        previous=out.cadence;
    }
    observation[12]=std::numeric_limits<float>::quiet_NaN();demo::BrakingLabel rejected;
    CHECK(!demo::feedback_brake(observation,rejected));
}
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

#include "motor/transition_examples.h"
#include "nn/mlp_cpu.h"
namespace {
demo::TransitionExamples transition_fixture(){
    auto data=demo::reuse_compatible(demo::accepted_examples(accepted_fixture()));
    float obs[42]{};obs[3]=.004f/.02f;obs[6]=1;
    for(int row=0;row<demo::kTransitionRows;++row){obs[12]=(row%2?.004f:.008f)/.02f;demo::SignedReference action;CHECK(demo::signed_teacher(obs,action));demo::append_example(data,obs,action);++data.new_rows;}
    data.accepted_segments=demo::kAcceptedTransitionMask;return data;
}
}
TEST_CASE("transition_examples: signed mapping supports reversal and leaves forward source commands intact") {
    float latent[2]{0,atanhf(-1.f/1.5f)};demo::SignedReference reverse;CHECK(demo::signed_action(latent,reverse));CHECK(std::fabs(reverse.cadence+1)<1e-6f&&reverse.amplitude==.35f);
    const auto converted=demo::reuse_compatible(demo::accepted_examples(accepted_fixture()));CHECK(converted.reused_rows==960);
    CHECK(std::fabs(converted.executed_commands[1]-.2f)<1e-6f);CHECK(std::fabs(tanhf(converted.data.target[1])-1.f/1.5f)<1e-6f);
    CHECK(converted.source_action_mapping[0]==1);
}
TEST_CASE("transition_examples: braking label is observable without a hidden phase or hysteresis state") {
    float a[42]{},b[42]{};a[3]=.004f/.02f;a[6]=1;a[12]=.008f/.02f;b[5]=a[3];b[8]=1;b[14]=a[12];b[34]=.7f;b[36]=1;b[39]=-1;
    demo::SignedReference x,y;CHECK(demo::signed_teacher(a,x)&&demo::signed_teacher(b,y));CHECK(x.action.latent[0]==y.action.latent[0]&&x.action.latent[1]==y.action.latent[1]);
}
TEST_CASE("transition_examples: complete gated rows roundtrip with explicit source action mappings") {
    const auto data=transition_fixture();const auto restored=demo::decode_transitions(demo::encode_transitions(data,123),123);
    CHECK(restored.data.observation==data.data.observation&&restored.data.target==data.data.target&&restored.executed_commands==data.executed_commands);
    CHECK(restored.new_rows==480&&restored.source_action_mapping.back()==2);
    const auto training=demo::balanced_training(restored);int negative=0;for(int row=0;row<training.count();++row)if(training.target[row*2+1]<0)++negative;
    CHECK(training.count()>=256&&negative*3==training.count());
}
TEST_CASE("transition_examples: missing gates changed actions and wrong protocol fail closed") {
    for(int fault=0;fault<5;++fault){auto data=transition_fixture();if(fault==0)data.accepted_segments=7;if(fault==1)data.executed_commands.back()=0;if(fault==2)data.data.target.back()+=.1f;if(fault==3)data.source_action_mapping.back()=1;
        bool rejected=false;try{(void)demo::decode_transitions(demo::encode_transitions(data,123),fault==4?124:123);}catch(const std::runtime_error&){rejected=true;}CHECK(rejected);}
}
TEST_CASE("transition_examples: signed initialization preserves strength and fixes cadence at one Hz") {
    const cudarl::nn::MlpSpec spec{{42,32,32,2},{cudarl::nn::Activation::Tanh,cudarl::nn::Activation::Tanh,cudarl::nn::Activation::Identity}};
    const auto original=cudarl::nn::init_params(spec,123);const auto initialized=demo::initialize_signed(original,spec);
    cudarl::nn::CpuMlp<float> old_actor(spec),new_actor(spec);old_actor.params()=original;new_actor.params()=initialized;float obs[42]{};
    for(int sample=0;sample<3;++sample){obs[12]=float(sample);old_actor.forward(obs,1);new_actor.forward(obs,1);demo::SignedReference action;CHECK(demo::signed_action(new_actor.output(),action));
        CHECK(new_actor.output()[0]==old_actor.output()[0]);CHECK(std::fabs(action.cadence-1)<1e-6f);}
}
