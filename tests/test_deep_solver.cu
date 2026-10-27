#include "check.h"
#include "fixtures/deep_solver_reference.h"
#include "cuda/cuda_memory.h"
#include "core/rng.h"
#include <cstring>
namespace {
namespace d=cudarl::deep;
struct Agreement {int differing_steps=0,invalid=0;float max_absolute=0;};
CUDARL_HD unsigned bits(float x) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(x);
#else
    unsigned result;std::memcpy(&result,&x,sizeof(x));return result;
#endif
}
CUDARL_HD void compare(float a,float b,bool& differs,Agreement& out) {
    differs=differs || bits(a)!=bits(b);
    if(!(fabsf(a)<1e30f && fabsf(b)<1e30f)) ++out.invalid;
    out.max_absolute=fmaxf(out.max_absolute,fabsf(a-b));
}
CUDARL_HD void compare(const d::BodyState& a,const d::BodyState& b,Agreement& out) {
    bool differs=false;
    compare(a.position.x,b.position.x,differs,out);compare(a.position.y,b.position.y,differs,out);compare(a.position.z,b.position.z,differs,out);
    compare(a.orientation.x,b.orientation.x,differs,out);compare(a.orientation.y,b.orientation.y,differs,out);
    compare(a.orientation.z,b.orientation.z,differs,out);compare(a.orientation.w,b.orientation.w,differs,out);
    for(int j=0;j<4;++j) compare(a.angle[j],b.angle[j],differs,out);
    for(int j=0;j<d::dofs;++j) compare(a.velocity[j],b.velocity[j],differs,out);
    compare(a.work,b.work,differs,out);compare(a.travel,b.travel,differs,out);
    if(differs) ++out.differing_steps;
}
CUDARL_HD Agreement run_case(int index,int ticks) {
    auto random=[index](unsigned tick) {return cudarl::rng::uniform(8731u,static_cast<unsigned>(index),tick)*2-1;};
    d::BodyState a;a.position={random(0)*2,-2+random(1),random(2)};
    a.orientation=d::axis_rotation({0,1,0},random(3)*3.14159265f)*d::axis_rotation({1,0,0},random(4)*.3f);
    for(int j=0;j<4;++j) a.angle[j]=random(5+j)*.98f;
    d::Physics physics;physics.pond=index%2==0;
    if(index%4!=0) physics.water_velocity={random(9)*.007f,0,random(10)*.007f};
    // Explicit floor, bank, surface and box contacts supplement free swimming.
    if(index%8==0) a.position.y=d::pond_floor(a.position.x,a.position.z)+.02f;
    if(index%8==1) a.position.x=4.45f;
    if(index%8==2) a.position.y=-.02f;
    if(index%8==3) {physics.boundaries=false;physics.gravity=0;physics.density=0;physics.viscosity=0;}
    d::BodyState b=a;Agreement out;
    // Vacuum is a short internal-torque check. Indefinite undamped actuation
    // exceeds this semi-implicit integrator's stable regime in BOTH solvers.
    const int duration=index%8==3 && ticks>100?100:ticks;
    for(int tick=0;tick<duration;++tick) {
        const float phase=tick*.05f;
        const float action[4]={sinf(phase*7+index),.5f*sinf(phase*5+index),sinf(phase*7-1.2f+index),.5f*cosf(phase*5+index)};
        for(int sub=0;sub<10;++sub) {
            d::reference::substep(a,action,.005f,physics);d::substep(b,action,.005f,physics);
            compare(a,b,out);
        }
    }
    return out;
}
__global__ void check_solver(Agreement* result,int n,int ticks) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) result[i]=run_case(i,ticks);
}
void require_agreement(const std::vector<Agreement>& cases,const char* backend,int ticks) {
    int differences=0,invalid=0;float maximum=0;
    for(size_t i=0;i<cases.size();++i) {const auto& c=cases[i];differences+=c.differing_steps;invalid+=c.invalid;maximum=fmaxf(maximum,c.max_absolute);
        if(c.invalid || c.differing_steps) std::printf("    case=%zu differences=%d invalid=%d\n",i,c.differing_steps,c.invalid);}
    std::printf("  SOLVER_REFERENCE backend=%s worlds=%zu ticks=%d differing_substeps=%d invalid=%d max_abs=%.9g\n",backend,cases.size(),ticks,differences,invalid,maximum);
    CHECK(differences==0);CHECK(invalid==0);CHECK(maximum==0);
}
}
TEST_CASE("deep solver: reused factor workspace matches dense reference on CPU") {
    std::vector<Agreement> cases(32);for(int i=0;i<32;++i) cases[i]=run_case(i,100);
    require_agreement(cases,"CPU",100);
}
GPU_TEST_CASE("deep solver: reused workspace preserves five-minute CUDA trajectories") {
    constexpr int n=64,ticks=6000;device_ptr<Agreement> result(n);
    check_solver<<<(n+63)/64,64>>>(result.get(),n,ticks);CUDA_CHECK_THROW(cudaGetLastError());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());std::vector<Agreement> cases(n);result.copy_to_host(cases.data(),n);
    require_agreement(cases,"CUDA",ticks);
}
