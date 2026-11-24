#include "queries.h"
#include <cstring>
// Separate compiled namespaces prevent linker folding/ODR collisions between
// the two copies of inline model, terrain and geometry definitions.
#define cudarl MODEL_NAMESPACE
#include "cuda/pond_math_profile.h"
#undef cudarl

namespace PROFILE_NAMESPACE {
namespace pr=MODEL_NAMESPACE::pond_runtime;
namespace bg=MODEL_NAMESPACE::pond_body_geometry;
namespace pg=MODEL_NAMESPACE::pond_geometry_v1;
namespace p=MODEL_NAMESPACE::physics;
using namespace query_profile;
static_assert(sizeof(pr::World)==world_bytes && pr::query_tolerance_m==tolerance && pr::query_visit_limit==131072);
CUDARL_HD Summary query(const pr::World& w){
    const auto hit=bg::inspect(w.motor.physical.body,w.terrain,pr::query_tolerance_m,pr::query_visit_limit);
    Summary result{hit.lower_gap_m,hit.upper_gap_m,hit.max_body_y_m,hit.visits,hit.valid,hit.certified,hit.failure_reason,{}};
    for(int i=0;i<3;++i){const auto& v=hit.link[i];
        result.link[i]={v.lower_gap_m,v.upper_gap_m,{v.exact_point_m.x,v.exact_point_m.y,v.exact_point_m.z},
                        {v.gradient_x,v.gradient_z},v.certified};}
    return result;
}
__global__ void query_batch(const pr::World* w,Summary* out,int count){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;if(index<count)out[index]=query(w[index]);
}
void launch(const void* d,Summary* output,int count,cudaStream_t stream){query_batch<<<(count+7)/8,8,0,stream>>>(static_cast<const pr::World*>(d),output,count);}
#ifdef PROFILE_BASELINE
Summary cpu(const void* data){pr::World w;std::memcpy(&w,data,sizeof(w));return query(w);}
double make_fixtures(void* dest){
    // Reproduce the analytic flat-head test and the established curved-bank
    // tangent-rod counterexample, independently of either search traversal.
    pr::World w; auto* bytes=static_cast<unsigned char*>(dest); std::memcpy(bytes,&w,sizeof(w));
    auto& state=w.motor.physical.body;state=p::BodyState{};const double x=.419;
    const auto surface=pg::sample(w.terrain,x,0);
    const double hx=-surface.normal.x/surface.normal.y,hz=-surface.normal.z/surface.normal.y,slope=hypot(hx,hz),angle=atan(slope),yaw=atan2(-hz,hx);
    state.orientation=p::axis_rotation({0,1,0},float(yaw))*p::axis_rotation({0,0,1},float(angle));
    state.position.x=float((x+.023*cos(angle)*cos(yaw))/p::scale);
    state.position.z=float((-.023*cos(angle)*sin(yaw))/p::scale);
    state.position.y=float((surface.height_m+.0015*sqrt(1+slope*slope)-.00008+.023*sin(angle))/p::scale);
    std::memcpy(bytes+sizeof(w),&w,sizeof(w));
    const auto links=MODEL_NAMESPACE::movement::shape(state);const auto rod=bg::detail::shape(links.link[1],1);double corner=1e30;
    for(int a:{-1,1})for(int b:{-1,1})for(int c:{-1,1}){
        const auto v=bg::detail::transform(rod,{a*rod.a,b*rod.b,c*rod.c});
        corner=fmin(corner,v.y-pg::sample(w.terrain,v.x,v.z).height_m);
    }
    return corner;
}
#endif
}
