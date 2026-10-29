#include "check.h"
#include "envs/swimmer/arch_batch.h"
#include "rl/arch_cpu.h"
#include "rl/arch_policy.h"
#include "rl/swimmer_ppo.h"
#include "nn/mlp_cpu.h"

namespace ar=cudarl::arch;
namespace dp=cudarl::deep;
TEST_CASE("arch navigation: transfer preserves inherited actions with arbitrary obstacle readings") {
    const auto oldspec=cudarl::rl::swimmer_specification(4,37);
    const auto weights=cudarl::nn::init_params(oldspec,923);
    cudarl::nn::CpuMlp<float> old(oldspec),expanded(cudarl::rl::swimmer_specification(4,ar::kObs));
    old.params()=weights;expanded.params()=cudarl::rl::arch_motor_parameters(weights);
    for(uint32_t seed=0;seed<16;++seed) {
        float observation[ar::kObs];for(int j=0;j<ar::kObs;++j) observation[j]=cudarl::rng::uniform(seed,0,static_cast<uint32_t>(j))*2-1;
        old.forward(observation,1);expanded.forward(observation,1);
        for(int j=0;j<4;++j) CHECK(fabsf(old.output()[j]-expanded.output()[j])<2e-7f);
    }
}
TEST_CASE("arch navigation: sensing distinguishes solid rock from the clear opening in the body frame") {
    auto state=ar::initial(7,0,0,false);state.pond.body.position=state.habitat.center;
    const float solid=ar::proximity(state.habitat.center,{1,0,0},state.habitat);
    CHECK(solid>.7f && solid<.73f);CHECK(ar::proximity(state.habitat.center,{0,0,1},state.habitat)==0);
    float first[ar::kObs],turned[ar::kObs];state.pond.body.orientation={};ar::observe(state,first);
    state.pond.body.orientation=dp::axis_rotation({0,1,0},-1.57079632679f);ar::observe(state,turned);
    CHECK(first[37]>.7f && turned[37]==0);
    state.pond.retention=true;state.habitat={};ar::observe(state,first);
    for(int j=37;j<ar::kObs;++j) CHECK(first[j]==0);
}
TEST_CASE("arch navigation: automatic foods remain clear and replenish across the rock") {
    for(uint32_t seed=1;seed<8;++seed) for(uint32_t world=0;world<16;++world) {
        auto state=ar::initial(seed,world,3,false);CHECK(cudarl::feeding::count(state.pond)==2);
        CHECK(dp::arch_body_clearance(state.pond.body,state.habitat)>0);
        for(int slot=0;slot<2;++slot) {
            CHECK(ar::water(state.pond.pellets[slot].position,state.habitat));
            CHECK(!state.pond.pellets[slot].settling);
            CHECK(dp::norm(state.pond.pellets[slot].position-state.pond.body.position)>=.79f);
            const auto before=dp::arch_to_local(state.pond.pellets[slot].position,state.habitat);
            state.pond.pellets[slot].active=false;ar::replenish(state);
            const auto after=dp::arch_to_local(state.pond.pellets[slot].position,state.habitat);
            CHECK(before.z*after.z<0);CHECK(ar::water(state.pond.pellets[slot].position,state.habitat));
        }
    }
}
TEST_CASE("arch navigation: passage observer requires the full body and rejects routes around rock") {
    auto run=[](float x,bool touch) {
        auto s=ar::initial(7,0,0,false);s.pond.body.orientation=dp::axis_rotation({0,1,0},-1.57079632679f);
        for(int i=0;i<=120;++i) {s.pond.body.position=s.habitat.center+dp::V{x,0,-3.f+i*.05f};ar::record_passage(s,touch && i==60);}
        return s;
    };
    const auto clean=run(0,false),touch=run(0,true),around=run(2.5f,false);
    CHECK(clean.passages==1 && clean.clean_passages==1);
    CHECK(touch.passages==1 && touch.clean_passages==0);
    CHECK(around.passages==0 && around.clean_passages==0);
    auto partial=ar::initial(7,0,0,false);partial.pond.body.orientation=dp::axis_rotation({0,1,0},-1.57079632679f);
    for(int i=0;i<65;++i) {partial.pond.body.position=partial.habitat.center+dp::V{0,0,-3.f+i*.05f};ar::record_passage(partial,false);}
    CHECK(partial.passages==0); // Head crossed, tail has not cleared the rock.
}
TEST_CASE("arch navigation: retention follows the old calm task exactly") {
    auto state=ar::initial(321,0,0,true);auto old=cudarl::feeding::initial(321,0,0,{},true);
    const float action[4]={.3f,-.2f,.1f,-.4f};
    for(int tick=0;tick<120;++tick) {
        const float a=ar::advance(state,action),b=cudarl::feeding::advance(old,action,{});
        ar::replenish(state);cudarl::feeding::replenish(old,{});
        CHECK(a==b);CHECK(dp::norm(state.pond.body.position-old.body.position)==0);
        CHECK(state.pond.eaten==old.eaten && state.contacts==0 && state.passages==0);
    }
}
GPU_TEST_CASE("arch navigation: CUDA and CPU share contacts sensing goals and resumable worlds") {
    const auto selector=cudarl::nn::init_params(cudarl::rl::progress_selector_spec(),7);
    cudarl::rl::SwimmerStream stream;ar::Batch gpu(1,7);gpu.configure(selector,stream.value);
    ar::CpuBatch cpu(1,7,selector);auto state=ar::initial(7,0,0,false);
    state.pond.body.position=state.habitat.center+dp::V{state.habitat.major,0,state.habitat.minor+.08f};
    gpu.set_state(state,stream.value);cpu.states[0]=state;cpu.prepare();
    const float action[4]={.1f,-.2f,.2f,.1f};device_ptr<float> device_action(4);device_action.copy_from_host(action,4);
    for(int tick=0;tick<30;++tick) {
        gpu.step(device_action.get(),{},stream.value,false);cpu.step(action,nullptr,nullptr,nullptr,false);
        const auto actual=gpu.state(stream.value);const auto& expected=cpu.states[0];
        CHECK(dp::norm(actual.pond.body.position-expected.pond.body.position)<.0005f);
        CHECK(actual.pond.target==expected.pond.target && actual.pond.eaten==expected.pond.eaten);
        float observations[ar::kObs];CUDA_CHECK_THROW(cudaMemcpyAsync(observations,gpu.obs(),sizeof(observations),cudaMemcpyDeviceToHost,stream.value));
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
        for(int j=0;j<ar::kObs;++j) CHECK(fabsf(observations[j]-cpu.observations[j])<.003f);
    }
    CHECK(gpu.state(stream.value).contacts>0);
    // Save/load a normal practice batch, then compare actual continuation.
    ar::Batch a(8,99),b(8,99);a.configure(selector,stream.value);b.configure(selector,stream.value);
    std::vector<float> actions(32,.1f);device_ptr<float> packed(32);packed.copy_from_host(actions.data(),actions.size());
    for(int i=0;i<5;++i) a.step(packed.get(),{},stream.value);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));cudarl::checkpoint::Writer saved;a.save(saved);
    cudarl::checkpoint::Reader reader{saved.data};b.load(reader);reader.finish();
    a.step(packed.get(),{},stream.value);b.step(packed.get(),{},stream.value);CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
    std::vector<ar::State> one(8),two(8);CUDA_CHECK_THROW(cudaMemcpy(one.data(),a.states(),one.size()*sizeof(ar::State),cudaMemcpyDeviceToHost));
    CUDA_CHECK_THROW(cudaMemcpy(two.data(),b.states(),two.size()*sizeof(ar::State),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<one.size();++i) {CHECK(dp::norm(one[i].pond.body.position-two[i].pond.body.position)==0);CHECK(one[i].pond.steps==two[i].pond.steps);}
}
