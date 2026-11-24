#include "replay.h"
#include <cstring>
#define cudarl MODEL_NAMESPACE
#include "cuda/pond_math_profile.h"
#include "rl/pond_ppo_schedule.h"
#undef cudarl
namespace PROFILE_NAMESPACE {
namespace pr=MODEL_NAMESPACE::pond_runtime;
namespace p=MODEL_NAMESPACE::physics;
namespace ps=MODEL_NAMESPACE::rl::pond_ppo_schedule;
using namespace replay;
static_assert(sizeof(pr::World)==world_bytes && sizeof(ps::SignedCourseTask)==task_bytes);
static_assert(offsetof(ps::SignedCourseTask,task)==0);
constexpr float hold_gain=.26211655139923096f;
CUDARL_HD void body(const p::BodyState& v,float* out){
    out[0]=v.position.x;out[1]=v.position.y;out[2]=v.position.z;out[3]=v.orientation.x;
    out[4]=v.orientation.y;out[5]=v.orientation.z;out[6]=v.orientation.w;
    for(int i=0;i<4;++i)out[7+i]=v.angle[i];
    for(int i=0;i<10;++i)out[11+i]=v.velocity[i];out[21]=v.work;out[22]=v.travel;
}
CUDARL_HD void control(pr::World& w,const float* command,pr::SubstepTrace* trace,Output& out){
    const auto c=w.attempted_controls,s=w.motor.attempted_substeps;
    out.success=pr::step(w,command,.005f,trace);
    out.controls=w.attempted_controls-c;out.substeps=w.motor.attempted_substeps-s;
    for(int j=0;j<4;++j)out.command[j]=command[j];
    for(int tick=0;tick<ticks;++tick){
        const auto& a=trace[tick];auto& b=out.steps[tick];
        body(a.before,b.body[0]);body(a.candidate,b.body[1]);body(a.after,b.body[2]);
        for(int j=0;j<4;++j){
            b.held[j]=a.held_targets[j];b.rate[j]=a.reference_rate[j];b.actual[j]=a.last_actual_torques[j];
            b.applied[j]=a.applied_torque_normalized[j];b.requested[j]=a.requested_torque_Nm[j];
        }
        for(int j=0;j<12;++j)b.energy[j]=a.energy[j];
        b.energy[12]=a.step_spring_work;b.energy[13]=a.step_damping_loss;b.energy[14]=a.step_potential_before;b.energy[15]=a.step_potential_after;
        b.low=a.gap_lower_m;b.high=a.gap_upper_m;
        for(int link=0;link<3;++link){
            b.mode[link]=a.observed_mode[link];
            for(int k=0;k<3;++k)b.force[link][k]=a.observed_force[link][k];
        }
        b.valid=a.recorded&&a.attempted&&a.accepted;b.failed=a.failure;b.attempts=a.attempted_substeps;
        b.accepted=a.accepted_substeps;b.iterations=a.contact_iterations;
    }
}
__global__ void advance(pr::World* w,const Request* requests,pr::SubstepTrace* trace,Output* out,int count){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    float command[4];const auto& r=requests[i];
    for(int j=0;j<4;++j)command[j]=r.direct?r.value[j]:tanhf((r.holding?hold_gain:1.f)*r.value[j]);
    control(w[i],command,trace+i*pr::substep_trace_capacity,out[i]);
}
void load(){
    cudaFuncAttributes attr{};const auto rc=cudaFuncGetAttributes(&attr,advance);
    if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));
}
size_t scratch(){return sizeof(pr::SubstepTrace)*pr::substep_trace_capacity;}
void launch(void* w,const Request* r,void* trace,Output* out,int n,cudaStream_t stream){
    advance<<<(n+7)/8,8,0,stream>>>(static_cast<pr::World*>(w),r,static_cast<pr::SubstepTrace*>(trace),out,n);
}
#ifndef PROFILE_BASELINE
Output cpu(const void* source,const float* command){
    pr::World w;std::memcpy(&w,source,sizeof(w));pr::SubstepTrace trace[pr::substep_trace_capacity];
    Output out{};control(w,command,trace,out);return out;
}
int holding(const void* bytes){ps::SignedCourseTask t;std::memcpy(&t,bytes,sizeof(t));return t.task.request.holding?1:0;}
#endif
}
