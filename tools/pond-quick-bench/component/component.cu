// Pure read-only component evaluations, not physical substeps or training.
#include "cuda/pond_math_profile.h"
#include "rl/pond_ppo_task.h"
#include "rl/pond_ppo_contract.h"
#include <chrono>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace cp=cudarl::checkpoint;namespace pr=cudarl::pond_runtime;namespace pt=cudarl::rl::pond_ppo_task;
namespace p=cudarl::physics;namespace m=cudarl::movement;namespace bg=cudarl::pond_body_geometry;
namespace fs=std::filesystem;using Clock=std::chrono::steady_clock;
constexpr int N=512;
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
void cuda(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
struct Input {std::vector<pr::World>w{N};std::vector<pt::Task>t{N};std::vector<float>latent;uint32_t update;};
Input fixture(cp::Reader&r){Input f;f.update=r.value<uint32_t>();(void)r.array<uint8_t>();const auto w=r.array<uint8_t>(),t=r.array<uint8_t>();require(w.size()==N*sizeof(pr::World)&&t.size()==N*sizeof(pt::Task),"Wrong ABI");std::memcpy(f.w.data(),w.data(),w.size());std::memcpy(f.t.data(),t.data(),t.size());f.latent=r.array<float>();for(int i=0;i<3;++i)(void)r.array<float>();for(int i=0;i<2;++i)(void)r.array<uint64_t>();(void)r.array<int>();(void)r.array<uint8_t>();require(f.latent.size()==N*4,"Wrong latent shape");return f;}
struct Algebra {double velocity[10]{};float command[4]{},reference[12]{};uint32_t system_evaluations=0;int failure=0;bool solved=false;};
// This reproduces the exact first-substep force/contact solve algebra using a
// newly queried initial-body inspection and local filter values. No candidate
// body is integrated, and no World/Task or saved filter state is modified.
__global__ void algebra(const pr::World*w,const pt::Task*t,const float*latent,const bg::Inspection*q,Algebra*out){const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=N)return;
 auto& result=out[id];const auto& body=w[id].motor.physical.body;float held[4]{},rate[4]{};for(int j=0;j<4;++j){held[j]=w[id].motor.physical.held_targets[j];rate[j]=w[id].motor.reference_rate[j];result.command[j]=tanhf((t[id].request.holding?cudarl::rl::pond_ppo::frozen_holding_gain:1.f)*latent[id*4+j]);}
 m::Reference reference; // filter is a pure map on local copies
 if(!cudarl::joint_motor::filter_targets(held,rate,result.command,reference)){result.failure=10;return;}
 for(int j=0;j<4;++j){result.reference[j]=reference.angle[j];result.reference[4+j]=reference.rate[j];result.reference[8+j]=reference.acceleration[j];}
 const auto system=m::assemble(body,{},true);const auto links=m::shape(body);pr::ContactPoint points[3];
 for(int b=0;b<3;++b){const auto& hit=q[id].link[b];const auto& link=links.link[b];const auto offset=hit.point_m-link.center;points[b].gap_m=hit.upper_gap_m;points[b].gradient={float(-hit.gradient_x),1,float(-hit.gradient_z)};points[b].damping_N_s_m=2*sqrtf(pr::stiffness_N_m*link.mass);for(int j=0;j<10;++j)points[b].jacobian[j]=link.linear[j]+p::cross(link.angular[j],offset);}
 pr::ContactSolution solution;m::Telemetry telemetry;result.solved=pr::solve_contacts(body,system,reference,.005f,points,solution,telemetry,result.failure);result.system_evaluations=solution.iterations;for(int j=0;j<10;++j)result.velocity[j]=solution.velocity[j];
}
__global__ void query(const pr::World*w,bg::Inspection*q){const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id<N)q[id]=bg::inspect(w[id].motor.physical.body,w[id].terrain,pr::query_tolerance_m,pr::query_visit_limit);}
template<class F>float timed(cudaEvent_t a,cudaEvent_t b,F f){cuda(cudaEventRecord(a));f();cuda(cudaGetLastError());cuda(cudaEventRecord(b));cuda(cudaEventSynchronize(b));float duration=0;cuda(cudaEventElapsedTime(&duration,a,b));return duration;}
int main(int argc,char**argv){try{require(argc==3,"Use component_bench fixtures.bin unique-out-directory");const auto started=Clock::now();const fs::path out=argv[2];require(!fs::exists(out)&&fs::create_directories(out),"Unique out required");const auto raw=cp::read(argv[1]);cp::Reader r{raw};require(r.value<uint32_t>()==0x31585150&&r.value<uint32_t>()==1&&r.value<uint64_t>()==16308029809216107606ull,"Wrong frozen fixture");(void)r.array<uint8_t>();(void)r.array<uint8_t>();const auto early=fixture(r),late=fixture(r);r.finish();const double read_ms=ms(started);
 pr::World*dw=nullptr;pt::Task*dt=nullptr;float*dl=nullptr;bg::Inspection*dq=nullptr;Algebra*da=nullptr;cuda(cudaMalloc(&dw,N*sizeof(pr::World)));cuda(cudaMalloc(&dt,N*sizeof(pt::Task)));cuda(cudaMalloc(&dl,N*4*sizeof(float)));cuda(cudaMalloc(&dq,N*sizeof(bg::Inspection)));cuda(cudaMalloc(&da,N*sizeof(Algebra)));cudaEvent_t a,b;cuda(cudaEventCreate(&a));cuda(cudaEventCreate(&b));std::ostringstream rows;rows<<std::setprecision(17);std::ofstream csv(out/"queries.csv");csv<<"batch,world,cpu_visits,gpu_visits,certified,interval_overlap,gap_lower,gap_upper,solver_evaluations,solver_solved\n";bool okay=true;uint64_t total_visits=0;int case_id=0;
 for(const auto* f:{&early,&late}){cuda(cudaMemcpy(dw,f->w.data(),N*sizeof(pr::World),cudaMemcpyHostToDevice));cuda(cudaMemcpy(dt,f->t.data(),N*sizeof(pt::Task),cudaMemcpyHostToDevice));cuda(cudaMemcpy(dl,f->latent.data(),N*4*sizeof(float),cudaMemcpyHostToDevice));cuda(cudaMemset(dq,0,N*sizeof(bg::Inspection)));cuda(cudaMemset(da,0,N*sizeof(Algebra)));
 const float query_ms=timed(a,b,[&]{query<<<64,8>>>(dw,dq);});const float algebra_ms=timed(a,b,[&]{algebra<<<64,8>>>(dw,dt,dl,dq,da);});std::vector<bg::Inspection>q(N);std::vector<Algebra>values(N);cuda(cudaMemcpy(q.data(),dq,N*sizeof(bg::Inspection),cudaMemcpyDeviceToHost));cuda(cudaMemcpy(values.data(),da,N*sizeof(Algebra),cudaMemcpyDeviceToHost));const auto cpu_start=Clock::now();uint64_t cpu_visits=0,gpu_visits=0,solver_evaluations=0;uint32_t max_visits=0,flat=0,certified=0,solved=0;double max_gap_delta=0;
 for(int id=0;id<N;++id){const auto reference=bg::inspect(f->w[id].motor.physical.body,f->w[id].terrain,pr::query_tolerance_m,pr::query_visit_limit);const auto&v=q[id];bool same=v.valid==reference.valid&&v.certified==reference.certified&&v.failure_reason==reference.failure_reason&&v.visits==reference.visits;const bool overlap=v.lower_gap_m<=reference.upper_gap_m&&reference.lower_gap_m<=v.upper_gap_m;max_gap_delta=fmax(max_gap_delta,fabs(v.upper_gap_m-reference.upper_gap_m));same=same&&overlap&&max_gap_delta<=1e-6;okay=okay&&same&&v.certified&&values[id].solved&&values[id].failure==0;cpu_visits+=reference.visits;gpu_visits+=v.visits;max_visits=std::max(max_visits,v.visits);flat+=v.visits==0;certified+=v.certified;solved+=values[id].solved;solver_evaluations+=values[id].system_evaluations;csv<<f->update<<','<<id<<','<<reference.visits<<','<<v.visits<<','<<v.certified<<','<<overlap<<','<<std::setprecision(17)<<v.lower_gap_m<<','<<v.upper_gap_m<<','<<values[id].system_evaluations<<','<<values[id].solved<<'\n';}
 if(case_id++)rows<<',';rows<<"{\"batch\":"<<f->update<<",\"worlds\":512,\"query_GPU_ms\":"<<query_ms<<",\"force_contact_solver_GPU_ms\":"<<algebra_ms<<",\"CPU_queries_ms\":"<<ms(cpu_start)<<",\"CPU_visits\":"<<cpu_visits<<",\"GPU_visits\":"<<gpu_visits<<",\"maximum_visits\":"<<max_visits<<",\"zero_visit_flat_worlds\":"<<flat<<",\"certified\":"<<certified<<",\"solved\":"<<solved<<",\"contact_solver_system_evaluations\":"<<solver_evaluations<<",\"maximum_CPU_GPU_upper_gap_delta_m\":"<<max_gap_delta<<'}';total_visits+=gpu_visits;}
 cudaEventDestroy(a);cudaEventDestroy(b);cudaFree(da);cudaFree(dq);cudaFree(dl);cudaFree(dt);cudaFree(dw);std::ofstream result(out/"result.json");result<<std::setprecision(17)<<"{\"passed\":"<<(okay?"true":"false")<<",\"scope\":\"Pure512initial-bodyqueries and512first-substepforce/contactalgebra evaluations perstage; no integration. Separate compiled componentprofile, not an exact timing decomposition of original kernel.\",\"compiler_profile\":\"CUDAReleaseoptimized;fmad=false;sm120\",\"fixture_read_ms\":"<<read_ms<<",\"wall_ms\":"<<ms(started)<<",\"CPU_body_queries\":1024,\"CUDA_body_queries\":1024,\"CUDA_force_solver_evaluations\":1024,\"CUDA_query_visits\":"<<total_visits<<",\"physical_controls\":0,\"physical_substeps\":0,\"optimizer_calls\":0,\"cases\":["<<rows.str()<<"]}\n";result.close();std::ifstream shown(out/"result.json");std::cout<<shown.rdbuf();return okay?0:2;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
