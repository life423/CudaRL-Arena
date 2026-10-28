#include "check.h"
#include "envs/swimmer/deep_pond.h"
#include <cmath>
#include <vector>
namespace d=cudarl::deep;
namespace p=cudarl::deep_pond;
TEST_CASE("pond currents: drag uses relative water velocity and advects a passive body") {
    d::BodyState a,b;a.velocity[0]=.002f;a.velocity[2]=-.003f;b=a;
    const d::V boost{.005f,0,.004f};b.velocity[0]+=boost.x;b.velocity[2]+=boost.z;
    d::Physics still,flow;still.gravity=0;still.boundaries=false;flow=still;flow.water_velocity=boost;
    const auto x=d::assemble(a,still),y=d::assemble(b,flow);
    for(int i=0;i<d::dofs;++i) {
        float fx=x.force[i],fy=y.force[i];
        for(int j=0;j<d::dofs;++j) {fx-=x.R[i][j]*a.velocity[j];fy-=y.R[i][j]*b.velocity[j];}
        CHECK(std::fabs(fx-fy)<1e-9f);
    }
    d::BodyState passive;const float zero[4]{};flow.water_velocity={.005f,0,0};
    for(int tick=0;tick<100;++tick) d::advance_physics(passive,zero,flow);
    std::printf("  PASSIVE_CURRENT x_model=%.6f speed_m_s=%.6f\n",passive.position.x,passive.velocity[0]);
    CHECK(passive.position.x>0);CHECK(passive.velocity[0]>0 && passive.velocity[0]<.0055f);
    d::BodyState reverse;flow.water_velocity={-.005f,0,0};
    for(int tick=0;tick<100;++tick) d::advance_physics(reverse,zero,flow);
    CHECK(reverse.position.x<0);CHECK(std::fabs(passive.position.x+reverse.position.x)<1e-6f);
    CHECK(std::isfinite(d::energy(passive)));
}
TEST_CASE("pond currents: explicit calm water preserves the default trajectory") {
    auto a=p::initial(7,0),b=a;
    for(int tick=0;tick<80;++tick) {
        const float action[4]={std::sin(tick*.13f),std::cos(tick*.17f),std::sin(tick*.19f),std::cos(tick*.23f)};
        p::advance(a,action);p::advance(b,action,{});
        CHECK(d::norm(a.body.position-b.body.position)==0);CHECK(a.eaten==b.eaten);
    }
}
GPU_TEST_CASE("pond currents: CPU CUDA share current forces and physical food rules") {
    constexpr int n=16;p::Batch world(n,7);std::vector<p::State> cpu(n),gpu(n);
    std::vector<float> actions(n*4);device_ptr<float> device_action(n*4);
    const d::V current{.005f,0,-.005f};
    for(int i=0;i<n;++i) cpu[i]=p::initial(7,static_cast<uint32_t>(i));
    for(int tick=0;tick<20;++tick) {
        for(int i=0;i<n*4;++i) actions[i]=.4f*std::sin(tick*.13f+i*.41f);
        device_action.copy_from_host(actions.data(),actions.size());
        world.step(device_action.get(),{},nullptr,false,1200,current);
        for(int i=0;i<n;++i) p::advance(cpu[i],actions.data()+i*4,current);
    }
    CUDA_CHECK_THROW(cudaDeviceSynchronize());CUDA_CHECK_THROW(cudaMemcpy(gpu.data(),world.states(),n*sizeof(p::State),cudaMemcpyDeviceToHost));
    for(int i=0;i<n;++i) {CHECK(d::norm(cpu[i].body.position-gpu[i].body.position)<.002f);CHECK(cpu[i].eaten==gpu[i].eaten);}
}
