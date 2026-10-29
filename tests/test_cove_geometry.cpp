#include "check.h"
#include "envs/swimmer/arch.h"
#include "envs/swimmer/arch_batch.h"
#include "rl/arch_cpu.h"
#include "rl/swimmer_ppo.h"
#include <type_traits>

namespace cv=cudarl::deep;
namespace ca=cudarl::arch;
namespace {
bool cove_particle(cv::V p,const cv::Habitat& h,cv::Basin basin) {
    constexpr float radius=cudarl::feeding::radius/cv::scale;
    const auto slope=cv::habitat_gradient(p.x,p.z,h,basin);
    return std::isfinite(cv::norm(p)) && cv::arch_distance(p,h)>=radius-3e-4f &&
        (p.y-cv::habitat_floor(p.x,p.z,h,basin))/sqrtf(1+slope.x*slope.x+slope.z*slope.z)>=radius-3e-4f &&
        p.y<=-radius+3e-4f;
}
struct LegacyHabitat {float scale=1;bool arch=false;cv::V center{0,-2.25f,0};float major=1.35f,minor=.5f,yaw=0;};
struct LegacyState {
    cudarl::feeding::State pond;LegacyHabitat habitat;
    int contacts=0,passages=0,clean_passages=0,pending_side=0,last_full_side=0;
    bool contacted_in_passage=false,crossed_opening=false,passage_valid=true;
};
}

TEST_CASE("cove habitat: versioned basin stays outside old raw world layout") {
    static_assert(sizeof(cv::Basin)==sizeof(int),"Basin layout identity must stay one integer");
    static_assert(sizeof(cv::Habitat)==sizeof(LegacyHabitat),"Do not change the raw inherited Habitat layout");
    static_assert(sizeof(ca::State)==sizeof(LegacyState),"Do not change the raw inherited arch State layout");
    static_assert(sizeof(ca::Context)==sizeof(cv::V)+2*sizeof(int),"Current context gains only the external basin identity");
    static_assert(std::is_trivially_copyable<ca::State>::value,"Arch worlds must remain trivially serializable");
    CHECK(ca::geometry().scale==1.5f && ca::geometry().arch);
}

TEST_CASE("cove habitat: smooth union extends water without reducing the old pool or central depths") {
    const auto h=ca::geometry();const cv::Basin cove{1};
    for(int xi=-20;xi<=24;++xi) for(int zi=-15;zi<=15;++zi) {
        const float x=xi*.4f,z=zi*.4f,old=cv::habitat_floor(x,z,h),next=cv::habitat_floor(x,z,h,cove);
        CHECK(next<=old+2e-6f);
        if(old<-3.35f) CHECK(next==old);
        CHECK(cv::habitat_floor(x,z,h,cv::Basin{})==old);
        CHECK(cv::norm(cv::habitat_gradient(x,z,h,cv::Basin{})-cv::habitat_gradient(x,z,h))==0);
    }
    CHECK(cv::habitat_floor(0,0,h,cove)==-4.f);
    CHECK(cv::habitat_floor(7,-1.5f,h)>0);CHECK(cv::habitat_floor(7,-1.5f,h,cove)<-2.9f);
    CHECK(ca::water({7,-1.5f,-1.5f},h,.4f,cove));CHECK(!ca::water({7,-1.5f,-1.5f},h));
    CHECK(cv::norm(cv::cove_center(h)-cv::V{5.8f,0,-1.5f})==0);
    CHECK(cv::norm(cv::cove_radii(h)-cv::V{3.2f,0,2.8f})==0);
}

TEST_CASE("cove habitat: analytic join gradients and continuous open shoreline agree") {
    const auto h=ca::geometry();const cv::Basin cove{1};constexpr float epsilon=.001f;
    for(int xi=0;xi<34;++xi) for(int zi=0;zi<25;++zi) {
        const float x=3.1f+xi*.14f,z=-3.4f+zi*.14f;
        // The old clamped dry plateau has a deliberate normal discontinuity;
        // compare the smooth underwater and connection portions.
        if(cv::habitat_floor(x,z,h,cove)>-.1f) continue;
        const auto gradient=cv::habitat_gradient(x,z,h,cove);
        const float gx=(cv::habitat_floor(x+epsilon,z,h,cove)-cv::habitat_floor(x-epsilon,z,h,cove))/(2*epsilon);
        const float gz=(cv::habitat_floor(x,z+epsilon,h,cove)-cv::habitat_floor(x,z-epsilon,h,cove))/(2*epsilon);
        CHECK(fabsf(gradient.x-gx)<.003f);CHECK(fabsf(gradient.z-gz)<.003f);
    }
    for(int ray=0;ray<180;++ray) {
        const float angle=ray*6.28318530718f/180;
        const auto boundary=cv::basin_boundary(angle,h,cove);
        CHECK(fabsf(cv::habitat_floor(boundary.x,boundary.z,h,cove))<2e-5f);
        CHECK(fabsf(cv::basin_radius(boundary,h,cove)-1)<2e-6f);
        for(int sample=1;sample<40;++sample) {
            const auto inside=boundary*(sample/40.f);
            CHECK(cv::habitat_floor(inside.x,inside.z,h,cove)<0);
        }
        const auto outside=boundary*1.01f;CHECK(cv::habitat_floor(outside.x,outside.z,h,cove)>0);
    }
    for(int sample=0;sample<=40;++sample) {
        const float x=2.5f+sample*.12f;
        CHECK(cv::habitat_floor(x,-1.5f,h,cove)<-2.9f); // A broad submerged connection, not an isolated pocket.
    }
}

TEST_CASE("cove habitat: automatic food covers both original sides and the reachable new region") {
    const cv::Basin cove{1};int front=0,back=0,new_area=0;
    for(uint32_t seed=1;seed<13;++seed) for(uint32_t env=0;env<8;++env) {
        auto s=ca::initial(seed,env,0,false,cove);CHECK(cudarl::feeding::count(s.pond)==2);
        for(int replacement=0;replacement<12;++replacement) {
            const int slot=replacement%2;s.pond.pellets[slot].active=false;ca::replenish(s,cove);
            CHECK(cudarl::feeding::count(s.pond)==2);
            const auto point=s.pond.pellets[slot].position;
            CHECK(ca::water(point,s.habitat,.4f,cove));
            CHECK(cv::norm(point-s.pond.body.position)>=.8f);
            if(cv::habitat_floor(point.x,point.z,s.habitat)>=0) ++new_area;
            else if(cv::arch_to_local(point,s.habitat).z<0) ++front;else ++back;
        }
    }
    CHECK(front>200 && back>200 && new_area>200);
}

TEST_CASE("cove habitat: particles and imposed flow use the extended basin without old ellipse clipping") {
    const auto h=ca::geometry();const cv::Basin cove{1};const cv::V original{7,-1.5f,-1.5f},current{.005f,0,0};
    CHECK(cv::norm(cv::habitat_flow(original,h,current))==0);
    CHECK(cv::norm(cv::habitat_flow(original,h,current,cove))>.0049f);
    auto position=ca::advect(original,current,h,cv::dt,cove);
    CHECK(position.x>original.x);CHECK(cv::norm(position-original)<.007f);
    for(int tick=0;tick<800;++tick) {
        const auto before=position;position=ca::advect(position,current,h,cv::dt,cove);
        CHECK(cove_particle(position,h,cove));CHECK(cv::norm(position-before)<.007f);
    }
    CHECK(cv::norm(ca::advect(position,{},h,.2f,cove)-position)==0);
    CHECK(cv::norm(cv::habitat_flow({9.5f,-1.5f,-1.5f},h,current,cove))==0);
    cv::BodyState body;body.position=original;cv::Physics legacy;legacy.pond=true;legacy.habitat=h;
    auto expanded=legacy;expanded.basin=cove;
    CHECK(cv::assemble(body,legacy).force[1]>cv::assemble(body,expanded).force[1]+.01f);
    const float action[4]{};
    for(int tick=0;tick<40;++tick) cv::advance_physics(body,action,expanded);
    CHECK(cv::norm(body.position-original)<.001f);
}

GPU_TEST_CASE("cove habitat: CUDA and CPU agree on extended contact flow food and saved layout") {
    const auto selector=cudarl::nn::init_params(cudarl::rl::progress_selector_spec(),7);
    ca::Context context;context.current={.005f,0,0};context.version=11;context.basin.layout=1;
    cudarl::rl::SwimmerStream stream;ca::Batch gpu(1,7);gpu.configure(selector,context,stream.value);
    ca::CpuBatch cpu(1,7,selector,context);auto state=ca::initial(7,0,0,false,context.basin);
    state.pond.body.position={7,-1.5f,-1.5f};state.pond.body.orientation={};
    gpu.set_state(state,stream.value);cpu.states[0]=state;cpu.prepare();
    const float action[4]={.1f,-.1f,.1f,-.1f};device_ptr<float> packed(4);packed.copy_from_host(action,4);
    for(int tick=0;tick<40;++tick) {
        gpu.step(packed.get(),{},stream.value,false,false);cpu.step(action,nullptr,nullptr,nullptr,false,false);
        const auto actual=gpu.state(stream.value);const auto& expected=cpu.states[0];
        CHECK(cv::norm(actual.pond.body.position-expected.pond.body.position)<.001f);
        CHECK(actual.pond.target==expected.pond.target && actual.pond.eaten==expected.pond.eaten);
        for(int food=0;food<2;++food) CHECK(cv::norm(actual.pond.pellets[food].position-expected.pond.pellets[food].position)<.0002f);
    }
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));cudarl::checkpoint::Writer saved;gpu.save(saved);
    ca::Batch resumed(1,7);resumed.configure(selector,stream.value);cudarl::checkpoint::Reader reader{saved.data};resumed.load(reader);reader.finish();
    CHECK(resumed.context().basin.layout==1 && resumed.context().version==11);
    gpu.step(packed.get(),{},stream.value,false,false);resumed.step(packed.get(),{},stream.value,false,false);
    const auto a=gpu.state(stream.value),b=resumed.state(stream.value);CHECK(cv::norm(a.pond.body.position-b.pond.body.position)==0);
    CHECK(cv::norm(a.pond.pellets[0].position-b.pond.pellets[0].position)==0);
}
