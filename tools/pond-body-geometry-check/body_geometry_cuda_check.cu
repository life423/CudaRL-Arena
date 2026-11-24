#include <cuda_runtime.h>
#include "environment/pond_body_geometry.h"
#include <cstdio>
#include <limits>
namespace bg=cudarl::pond_body_geometry;
namespace pg=cudarl::pond_geometry_v1;
namespace p=cudarl::physics;
constexpr int cases=16;
constexpr uint32_t visits_per_case=131072;
// Predeclared BEFORE execution: matching scope/reason/certification; certified
// intervals must overlap and sampled-minimum upper gaps differ by <=1micron.
constexpr double maximum_upper_gap_difference_m=1e-6;
__global__ void query(const p::BodyState* states,bg::Inspection* results) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<cases)results[index]=bg::inspect(states[index],pg::current_visual_parameters(),1e-5,visits_per_case);
}
bool cuda_ok(cudaError_t error,const char* operation) {
    if(error==cudaSuccess)return true;
    std::fprintf(stderr,"CUDA %s: %s\n",operation,cudaGetErrorString(error));return false;
}
int main() {
    p::BodyState states[cases];
    states[1].angle[1]=.2f;states[1].angle[3]=-.2f;
    states[2].orientation=p::axis_rotation({1,0,0},.4f);
    states[3].orientation=p::axis_rotation({1,0,0},1.57079632679f);
    states[4].position.x=.25f/p::scale;states[5].position.x=-.25f/p::scale;
    states[6].position.z=.18f/p::scale;states[7].position.z=-.18f/p::scale;
    states[8].position.x=.35f/p::scale;states[9].position.x=-.35f/p::scale;
    states[10].position.x=.37f/p::scale;states[10].position.z=.03f/p::scale;
    states[10].orientation=p::axis_rotation({0,0,1},.13f);
    states[11].position.x=-.37f/p::scale;states[11].position.z=-.03f/p::scale;
    states[11].orientation=p::axis_rotation({1,0,0},.33f);
    const double x=.419;const auto surface=pg::sample(pg::current_visual_parameters(),x,0);
    const double hx=-surface.normal.x/surface.normal.y,hz=-surface.normal.z/surface.normal.y,slope=hypot(hx,hz),angle=atan(slope),yaw=atan2(-hz,hx);
    states[12].orientation=p::axis_rotation({0,1,0},float(yaw))*p::axis_rotation({0,0,1},float(angle));
    states[12].position.x=float((x+.023*cos(angle)*cos(yaw))/p::scale);
    states[12].position.z=float((-.023*cos(angle)*sin(yaw))/p::scale);
    states[12].position.y=float((surface.height_m+.0015*sqrt(1+slope*slope)-.00008+.023*sin(angle))/p::scale);
    states[13].position.y=-.004f/p::scale;
    states[14].position.x=.6f/p::scale;
    states[15].position.x=std::numeric_limits<float>::quiet_NaN();
    bg::Inspection host[cases],device[cases];
    uint64_t cpu_visits=0,cuda_visits=0;
    for(int index=0;index<cases;++index){host[index]=bg::inspect(states[index],pg::current_visual_parameters(),1e-5,visits_per_case);cpu_visits+=host[index].visits;}
    p::BodyState* device_states=nullptr;bg::Inspection* device_results=nullptr;
    if(!cuda_ok(cudaMalloc(&device_states,sizeof states),"state allocation")||!cuda_ok(cudaMalloc(&device_results,sizeof device),"result allocation"))return 2;
    if(!cuda_ok(cudaMemcpy(device_states,states,sizeof states,cudaMemcpyHostToDevice),"identical input upload"))return 2;
    query<<<1,32>>>(device_states,device_results);
    if(!cuda_ok(cudaGetLastError(),"query launch")||!cuda_ok(cudaDeviceSynchronize(),"query completion"))return 2;
    if(!cuda_ok(cudaMemcpy(device,device_results,sizeof device,cudaMemcpyDeviceToHost),"query download"))return 2;
    bool passed=true;double maximum_difference=0;uint32_t certified=0;
    for(int index=0;index<cases;++index) {
        const auto& a=host[index];const auto& b=device[index];cuda_visits+=b.visits;
        bool same=a.valid==b.valid&&a.certified==b.certified&&a.failure_reason==b.failure_reason;
        double error=0;
        if(a.certified&&b.certified) {
            ++certified;
            same=same&&a.lower_gap_m<=b.upper_gap_m&&b.lower_gap_m<=a.upper_gap_m;
            for(int link=0;link<3;++link) {
                const double difference=fabs(a.link[link].upper_gap_m-b.link[link].upper_gap_m);
                if(difference>error)error=difference;
                same=same&&a.link[link].lower_gap_m<=b.link[link].upper_gap_m&&b.link[link].lower_gap_m<=a.link[link].upper_gap_m&&difference<=maximum_upper_gap_difference_m;
            }
        }
        if(error>maximum_difference)maximum_difference=error;
        passed=passed&&same;
        std::printf("case %d passed %d CPU(valid%d certified%d reason%d visits%u) CUDA(valid%d certified%d reason%d visits%u) upper_difference_m %.17g\n",index,same,a.valid,a.certified,a.failure_reason,a.visits,b.valid,b.certified,b.failure_reason,b.visits,error);
    }
    cudaFree(device_results);cudaFree(device_states);
    std::printf("{\"passed\":%s,\"cases\":16,\"CPU_body_queries\":16,\"CUDA_body_queries\":16,\"certified_cases\":%u,\"CPU_query_visits\":%llu,\"CUDA_query_visits\":%llu,\"maximum_sampled_upper_difference_m\":%.17g,\"upper_difference_limit_m\":1e-6,\"CUDA_geometry_kernels\":1,\"physical_controls\":0,\"physical_substeps\":0,\"optimizer_steps\":0,\"cuda_profile\":\"fmad=false\"}\n",passed?"true":"false",certified,static_cast<unsigned long long>(cpu_visits),static_cast<unsigned long long>(cuda_visits),maximum_difference);
    return passed?0:1;
}
