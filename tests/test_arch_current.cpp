#include "check.h"
#include "envs/swimmer/arch.h"
#include "envs/swimmer/arch_batch.h"
#include "rl/arch_cpu.h"
#include "rl/swimmer_ppo.h"
#include <array>

namespace ac=cudarl::arch;
namespace cf=cudarl::deep;
namespace {
bool finite_flow(cf::V value) {return std::isfinite(value.x)&&std::isfinite(value.y)&&std::isfinite(value.z);}
bool fluid_particle(cf::V p,const cf::Habitat& h) {
    constexpr float radius=cudarl::feeding::radius/cf::scale;
    const auto slope=cf::habitat_gradient(p.x,p.z,h);
    const float length=sqrtf(1+slope.x*slope.x+slope.z*slope.z);
    return finite_flow(p) && cf::arch_distance(p,h)>=radius-2e-4f &&
        (p.y-cf::habitat_floor(p.x,p.z,h))/length>=radius-2e-4f &&
        p.y<=-radius+2e-4f && ac::radial(p,h)<=.94f*.94f+1e-4f;
}
}

TEST_CASE("arch current: local field preserves uniform legacy flow and vanishes in solids") {
    cf::Habitat legacy;const cf::V imposed{.005f,.001f,-.003f};
    for(const auto p:{cf::V{0,-2,0},cf::V{10,10,10},cf::V{-4,-8,1}})
        CHECK(cf::norm(cf::habitat_flow(p,legacy,imposed)-imposed)==0);
    const auto h=ac::geometry();
    CHECK(cf::norm(cf::habitat_flow(h.center,h,imposed)-imposed)==0); // Open central aperture.
    CHECK(cf::norm(cf::habitat_flow(h.center+cf::V{h.major,0,0},h,imposed))==0);
    CHECK(cf::norm(cf::habitat_flow(h.center+cf::V{h.major,0,h.minor},h,imposed))<1e-9f);
    CHECK(cf::norm(cf::habitat_flow({4,-4,0},h,imposed))==0);
    CHECK(cf::norm(cf::habitat_flow({0,0,0},h,imposed))==0);
    CHECK(cf::norm(cf::habitat_flow({0,.5f,0},h,imposed))==0);
    CHECK(cf::norm(cf::habitat_flow({8,-2,0},h,imposed))==0);
    CHECK(cf::norm(cf::habitat_flow({-3,-2,2},h,imposed)-imposed)==0);
}

TEST_CASE("arch current: boundary field is finite bounded and turns normal flow near rock") {
    auto h=ac::geometry();h.yaw=.31f;h.center.x=.2f;
    const cf::V imposed{.0075f,0,0};
    for(int x=-14;x<=14;++x) for(int y=-9;y<=1;++y) for(int z=-11;z<=11;++z) {
        const auto velocity=cf::habitat_flow({x*.5f,y*.5f,z*.5f},h,imposed);
        CHECK(finite_flow(velocity));CHECK(cf::norm(velocity)<=cf::norm(imposed)+1e-8f);
    }
    const auto point=cf::arch_to_world({h.major+h.minor+.1f,0,0},h);
    cf::V normal;cf::arch_distance(point,h,&normal);
    const auto toward=normal*(-.0075f),local=cf::habitat_flow(point,h,toward);
    CHECK(fabsf(cf::dot(local,normal))<fabsf(cf::dot(toward,normal))*.1f);
    const auto tangent=cf::arch_to_world({0,0,1},h)-h.center;
    CHECK(cf::norm(cf::habitat_flow(point,h,tangent*.005f))>.0049f);
    for(const auto p:{point,h.center,cf::V{5,-1,0}}) CHECK(cf::norm(cf::habitat_flow(p,h,{}))==0);
}

TEST_CASE("arch current: passive spheres cross the hole and stay outside rock bed and surface") {
    const auto h=ac::geometry();auto swimmer_food=h.center+cf::V{0,0,-2};
    for(int tick=0;tick<400;++tick) swimmer_food=ac::advect(swimmer_food,{0,0,.005f},h,cf::dt);
    CHECK(fabsf((swimmer_food-h.center).z-.5f)<.001f);CHECK(fluid_particle(swimmer_food,h));
    const std::array<cf::V,8> starts={cf::V{-3,-2,-1},cf::V{3,-2,1},cf::V{0,-1,-2},cf::V{0,-3,2},
        cf::V{5,-1.5f,0},cf::V{-5,-1.5f,0},h.center+cf::V{h.major+h.minor+.08f,0,0},h.center};
    for(size_t sample=0;sample<starts.size();++sample) {
        auto p=starts[sample];const float angle=static_cast<float>(sample)*.79f;
        const cf::V imposed{.0075f*cosf(angle),0,.0075f*sinf(angle)};
        CHECK(fluid_particle(p,h));
        for(int tick=0;tick<800;++tick) {
            const auto before=p;p=ac::advect(p,imposed,h,cf::dt);
            CHECK(fluid_particle(p,h));CHECK(cf::norm(p-before)<=cf::norm(imposed)*cf::dt/cf::scale+3e-4f);
        }
        const auto stopped=ac::advect(p,{},h,.2f);CHECK(cf::norm(stopped-p)==0);
        CHECK(cf::norm(ac::advect(p,imposed,h,0)-p)==0);
    }
}

TEST_CASE("arch current: practice schedules preserve calm worlds and clocks and cover bounded variation") {
    ac::Context context;context.current={.0075f,0,0};context.version=3;
    auto retention=ac::initial(7,0,0,true),calm=ac::initial(7,1,0,false),steady=ac::initial(7,2,0,false),varying=ac::initial(7,3,0,false);
    CHECK(cf::norm(ac::practice_current(retention,context))==0);CHECK(cf::norm(ac::practice_current(calm,context))==0);
    const auto unchanged=steady;const auto fixed=ac::practice_current(steady,context);
    CHECK(fixed.x>=0 && fixed.x<=.0075f && fixed.y==0 && fixed.z==0);
    bool saw_calm=false,saw_flow=false;
    for(int tick=0;tick<5000;++tick) {
        varying.pond.steps=tick;steady.pond.steps=tick;
        const auto velocity=ac::practice_current(varying,context);
        CHECK(velocity.x>=0 && velocity.x<=.00750001f && velocity.y==0 && velocity.z==0);
        saw_calm=saw_calm || velocity.x==0;saw_flow=saw_flow || velocity.x>.0001f;
        CHECK(cf::norm(ac::practice_current(steady,context)-fixed)==0);
    }
    CHECK(saw_calm && saw_flow);CHECK(cf::norm(steady.pond.body.position-unchanged.pond.body.position)==0);
    context.current={};CHECK(cf::norm(ac::practice_current(steady,context))==0);CHECK(cf::norm(ac::practice_current(varying,context))==0);
}

TEST_CASE("arch current: zero flow keeps calm physics and food unchanged and switch-off clears velocity") {
    auto calm=ac::initial(7,0,0,false),flowing=calm;const float action[4]={.1f,-.1f,.2f,-.2f};
    auto expected=calm.pond.body;cf::Physics physics;physics.pond=true;physics.habitat=calm.habitat;
    cf::advance_physics(expected,action,physics);const auto first_food=calm.pond.pellets[0].position;
    ac::advance(calm,action,{});
    CHECK(cf::norm(calm.pond.body.position-expected.position)==0);
    for(int i=0;i<cf::dofs;++i) CHECK(calm.pond.body.velocity[i]==expected.velocity[i]);
    CHECK(cf::norm(calm.pond.pellets[0].position-first_food)==0);CHECK(cf::norm(calm.pond.pellets[0].velocity)==0);
    ac::advance(flowing,action,{.005f,0,0});
    CHECK(cf::norm(flowing.pond.pellets[0].position-first_food)>0);
    CHECK(cf::norm(flowing.pond.pellets[0].velocity)>0);
    CHECK(cf::norm(flowing.pond.body.position-calm.pond.body.position)>0);
    const auto moved=flowing.pond.pellets[0].position;ac::advance(flowing,action,{});
    CHECK(cf::norm(flowing.pond.pellets[0].position-moved)==0);CHECK(cf::norm(flowing.pond.pellets[0].velocity)==0);
    ac::Context context;context.current={.0075f,0,0};auto retained=ac::initial(7,0,0,true),reference=retained;
    CHECK(ac::advance(retained,action,context.current)==ac::advance(reference,action,{}));
    CHECK(cf::norm(retained.pond.body.position-reference.pond.body.position)==0);
}

GPU_TEST_CASE("arch current: CUDA matches CPU local drag food drift and full current-context continuation") {
    const auto selector=cudarl::nn::init_params(cudarl::rl::progress_selector_spec(),7);
    ac::Context context;context.current={.0075f,0,0};context.version=9;
    cudarl::rl::SwimmerStream stream;ac::Batch gpu(1,7);gpu.configure(selector,context,stream.value);
    ac::CpuBatch cpu(1,7,selector,context);auto state=ac::initial(7,0,0,false);
    state.pond.body.position=state.habitat.center+cf::V{state.habitat.major,0,state.habitat.minor+.25f};
    state.pond.body.orientation={};gpu.set_state(state,stream.value);cpu.states[0]=state;cpu.prepare();
    const float action[4]={.1f,-.2f,.2f,.1f};device_ptr<float> device_action(4);device_action.copy_from_host(action,4);
    const auto original_food=state.pond.pellets[0].position;
    for(int tick=0;tick<40;++tick) {
        gpu.step(device_action.get(),{},stream.value,false,false);cpu.step(action,nullptr,nullptr,nullptr,false,false);
        const auto actual=gpu.state(stream.value);const auto& expected=cpu.states[0];
        CHECK(cf::norm(actual.pond.body.position-expected.pond.body.position)<.001f);
        CHECK(actual.pond.target==expected.pond.target && actual.pond.eaten==expected.pond.eaten);
        for(int slot=0;slot<2;++slot) {
            CHECK(cf::norm(actual.pond.pellets[slot].position-expected.pond.pellets[slot].position)<.0001f);
            CHECK(cf::norm(actual.pond.pellets[slot].velocity-expected.pond.pellets[slot].velocity)<.0001f);
        }
    }
    CHECK(cf::norm(gpu.state(stream.value).pond.pellets[0].position-original_food)>.05f);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));cudarl::checkpoint::Writer saved;gpu.save(saved);
    ac::Batch restored(1,7);restored.configure(selector,stream.value);
    cudarl::checkpoint::Reader reader{saved.data};restored.load(reader);reader.finish();
    CHECK(restored.context().version==9 && restored.context().current.x==.0075f);
    gpu.step(device_action.get(),{},stream.value,false,false);restored.step(device_action.get(),{},stream.value,false,false);
    const auto first=gpu.state(stream.value),second=restored.state(stream.value);
    CHECK(cf::norm(first.pond.body.position-second.pond.body.position)==0);
    CHECK(cf::norm(first.pond.pellets[0].position-second.pond.pellets[0].position)==0);
    context.current={};++context.version;const auto before=first.pond;
    gpu.configure(selector,context,stream.value);const auto unchanged=gpu.state(stream.value).pond;
    CHECK(unchanged.steps==before.steps && cf::norm(unchanged.body.position-before.body.position)==0);
    gpu.step(device_action.get(),{},stream.value,false,false);
    const auto stopped=gpu.state(stream.value);
    CHECK(cf::norm(stopped.pond.pellets[0].position-before.pellets[0].position)==0);
    CHECK(cf::norm(stopped.pond.pellets[0].velocity)==0);
}
