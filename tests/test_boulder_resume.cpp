#include "check.h"
#include "envs/swimmer/arch_batch.h"
#include "rl/arch_cpu.h"
#include "rl/swimmer_ppo.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>

namespace {
void CUDART_CB delay_arch_restore_producer(void*) {
    // Test-only pending producer. CUDA calls are forbidden inside this callback.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}
std::vector<cudarl::arch::State> resume_worlds(const cudarl::arch::Batch& batch,int count) {
    std::vector<cudarl::arch::State> states(static_cast<size_t>(count));
    CUDA_CHECK_THROW(cudaMemcpy(states.data(),batch.states(),states.size()*sizeof(states[0]),cudaMemcpyDeviceToHost));
    return states;
}
void pending_arch_restore_producer(cudarl::arch::Batch& batch,int count,cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaLaunchHostFunc(stream,delay_arch_restore_producer,nullptr));
    // This deliberately substitutes a delayed producer for an unfinished
    // initialization kernel. It is never part of runtime training or recovery.
    CUDA_CHECK_THROW(cudaMemsetAsync(const_cast<cudarl::arch::State*>(batch.states()),0,
        static_cast<size_t>(count)*sizeof(cudarl::arch::State),stream));
}
}

GPU_TEST_CASE("boulder resume: nonblocking initialization cannot overwrite restored episode clocks") {
    namespace ar=cudarl::arch;namespace cp=cudarl::checkpoint;
    constexpr int count=1024,parent_steps=1024;
    auto selector=cudarl::nn::init_params(cudarl::rl::progress_selector_spec(),7);
    std::fill(selector.begin(),selector.end(),0.f);
    ar::Context inherited;inherited.current={.005f,0,0};inherited.version=3;inherited.basin.layout=1;
    auto desired=inherited;desired.version=4;desired.basin.layout=2;
    std::vector<ar::State> original(static_cast<size_t>(count));
    for(int world=0;world<count;++world) {
        const auto basin=world%4<2?cudarl::deep::Basin{}:inherited.basin;
        original[world]=ar::initial(71,static_cast<uint32_t>(world),0,world%4==0,basin);
        original[world].pond.steps=parent_steps;original[world].pond.target=-1;original[world].pond.decision=false;
    }
    cp::Writer source;source.array(original);source.value(inherited);
    cudarl::rl::SwimmerStream stream;
    ar::Batch batch(count,71);batch.configure(selector,desired,stream.value,true);
    pending_arch_restore_producer(batch,count,stream.value);
    cp::Reader restore{source.data};batch.load(restore);restore.finish();
    auto states=resume_worlds(batch,count);
    CHECK(std::memcmp(states.data(),original.data(),states.size()*sizeof(states[0]))==0);
    CHECK(batch.context().basin.layout==1 && batch.context().version==3);
    for(const auto& state:states) CHECK(state.pond.steps==parent_steps);

    // The full source population remains intact. The new habitat is requested
    // at a rollout boundary without restarting those inherited episode clocks.
    batch.configure(selector,desired,stream.value,false);
    ar::CpuBatch cpu(4,71,selector,desired);
    for(int world=0;world<4;++world) cpu.states[world]=original[world];
    cpu.prepare();
    std::vector<float> actions(static_cast<size_t>(count)*ar::kAct,0.f);
    device_ptr<float> packed(actions.size());packed.copy_from_host(actions.data(),actions.size());
    for(int tick=0;tick<ar::kEpisodeSteps-parent_steps;++tick) {
        batch.step(packed.get(),{},stream.value);
        cpu.step(actions.data());
    }
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));states=resume_worlds(batch,count);
    for(const auto& state:states) CHECK(state.pond.steps==0 && state.pond.episode==1);
    for(const auto& state:cpu.states) CHECK(state.pond.steps==0 && state.pond.episode==1);
    for(int tick=0;tick<17;++tick) {batch.step(packed.get(),{},stream.value);cpu.step(actions.data());}
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));states=resume_worlds(batch,count);
    for(const auto& state:states) CHECK(state.pond.steps==17 && state.pond.episode==1);
    for(const auto& state:cpu.states) CHECK(state.pond.steps==17 && state.pond.episode==1);

    cp::Writer saved;batch.save(saved);
    ar::Batch resumed(count,71);resumed.configure(selector,desired,stream.value,true);
    pending_arch_restore_producer(resumed,count,stream.value);
    cp::Reader reload{saved.data};resumed.load(reload);reload.finish();
    const auto loaded=resume_worlds(resumed,count);
    CHECK(std::memcmp(states.data(),loaded.data(),states.size()*sizeof(states[0]))==0);
    for(int tick=0;tick<3;++tick) {
        batch.step(packed.get(),{},stream.value);resumed.step(packed.get(),{},stream.value);cpu.step(actions.data());
    }
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));states=resume_worlds(batch,count);
    const auto continued=resume_worlds(resumed,count);
    CHECK(std::memcmp(states.data(),continued.data(),states.size()*sizeof(states[0]))==0);
    for(const auto& state:states) CHECK(state.pond.steps==20);
    for(const auto& state:cpu.states) CHECK(state.pond.steps==20);

    // A genuinely mixed-clock file remains invalid; this is not a migration
    // that silently repairs or relaxes inconsistent saved practice state.
    states[8].pond.steps=21;cp::Writer corrupt;corrupt.array(states);corrupt.value(desired);
    bool rejected=false;
    try {cp::Reader bad{corrupt.data};resumed.load(bad);}
    catch(const std::runtime_error& error) {rejected=std::string(error.what()).find("world 8")!=std::string::npos;}
    CHECK(rejected);
}
