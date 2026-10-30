#include "envs/swimmer/predator.h"
#include "cuda/cuda_memory.h"
#include <stdexcept>
#include <vector>

namespace cudarl::test {
namespace {
constexpr int probe_fields=200;
// Fixed diagnostic layout: SDF/normal0..3, clearance4, exposure5..8,
// particle9..11, reward12, advanced body13..29, native force30..39,
// damping40..139, motor observations140..194, counters195..198,
// advanced whole-body clearance199. Both paths execute these same equations.
CUDARL_HD void probe(arch::State state,deep::V point,deep::V eye,float* out) {
    constexpr deep::Basin basin{3};deep::V normal;
    out[0]=deep::obstacle_distance(point,state.habitat,basin,&normal);
    out[1]=normal.x;out[2]=normal.y;out[3]=normal.z;
    out[4]=predator::body_clearance(state.pond.body,state.habitat,basin);
    const auto exposure=predator::exposure_detail(state.pond.body,eye,state.habitat,basin);
    out[5]=exposure.visible;out[6]=exposure.head;out[7]=exposure.proximal_tail;out[8]=exposure.distal_tail;
    const auto particle=arch::particle_contact(point,state.habitat,basin);
    out[9]=particle.x;out[10]=particle.y;out[11]=particle.z;
    deep::Physics physics;physics.pond=true;physics.habitat=state.habitat;
    physics.basin=basin;physics.water_velocity={.005f,0,0};
    const auto system=deep::assemble(state.pond.body,physics);
    for(int i=0;i<deep::dofs;++i) out[30+i]=system.force[i];
    for(int i=0;i<deep::dofs;++i) for(int j=0;j<deep::dofs;++j) out[40+i*deep::dofs+j]=system.R[i][j];
    arch::observe(state,out+140,basin);out[195]=system.arch_touched?1.f:0.f;
    const float action[arch::kAct]={.1f,-.1f,.08f,-.08f};
    out[12]=arch::advance(state,action,physics.water_velocity,basin);
    const auto& body=state.pond.body;out[13]=body.position.x;out[14]=body.position.y;out[15]=body.position.z;
    for(int i=0;i<4;++i) out[16+i]=body.angle[i];
    for(int i=0;i<deep::dofs;++i) out[20+i]=body.velocity[i];
    out[196]=static_cast<float>(state.pond.eaten);out[197]=static_cast<float>(state.contacts);
    out[198]=static_cast<float>(state.passages);out[199]=predator::body_clearance(body,state.habitat,basin);
}
__global__ void probes(const arch::State* states,const deep::V* points,deep::V eye,int count,float* out) {
    const int i=static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);
    if(i<count) probe(states[i],points[i],eye,out+i*probe_fields);
}
}
int overhang_cpu_gpu_probe(const std::vector<arch::State>& states,const std::vector<deep::V>& points,
                           deep::V eye,std::vector<float>& cpu,std::vector<float>& gpu) {
    if(states.empty() || states.size()!=points.size()) throw std::invalid_argument("Invalid overhang parity probes");
    cpu.resize(states.size()*probe_fields);gpu.resize(cpu.size());
    for(size_t i=0;i<states.size();++i) probe(states[i],points[i],eye,cpu.data()+i*probe_fields);
    device_ptr<arch::State> native_states(states.size());native_states.copy_from_host(states.data(),states.size());
    device_ptr<deep::V> native_points(points.size());native_points.copy_from_host(points.data(),points.size());
    device_ptr<float> results(gpu.size());
    probes<<<(static_cast<int>(states.size())+31)/32,32>>>(native_states.get(),native_points.get(),eye,static_cast<int>(states.size()),results.get());
    CUDA_CHECK_THROW(cudaGetLastError());results.copy_to_host(gpu.data(),gpu.size());
    return probe_fields;
}
}
