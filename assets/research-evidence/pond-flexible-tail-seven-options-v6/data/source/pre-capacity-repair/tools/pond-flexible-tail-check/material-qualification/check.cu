#include <cuda_runtime.h>
#define flexible_tail flexible_tail_preserved
#include "legacy_flexible_tail.h"
#undef flexible_tail
#include "physics/flexible_tail.h"
#include "rl/flexible_tail_bank_observation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ft=cudarl::flexible_tail;
namespace old=cudarl::flexible_tail_preserved;
namespace p=cudarl::physics;
using Clock=std::chrono::steady_clock;

namespace {
constexpr int fixtures=8, short_steps=64, release_steps=1600, neutral_steps=400;
constexpr uint64_t maximum_substeps=64000;
constexpr float dt=.00125f;
struct Row {p::BodyState body{};ft::StepInfo info{};ft::Snapshot pose{};bool valid=false;};
struct Fixture {p::BodyState initial{};float commands[short_steps][2]{};ft::Material material{};};
struct Release {p::BodyState initial{};ft::Material material{};bool neutral=false;};
struct ReleaseResult {
    p::BodyState body{};uint64_t attempted=0,accepted=0;
    double energy_ratio=0,maximum_energy_fraction=0,maximum_linear=0,maximum_angular=0;
    float displacement_m=0,remaining_bend=0;bool valid=true;
};
struct Counts {uint64_t cpu=0,cuda=0,legacy=0;};
Counts counts;
Clock::time_point deadline;
std::string failure;
void require(bool pass,const char* message){if(!pass)throw std::runtime_error(message);}
void budget(){require(counts.cpu+counts.cuda+counts.legacy<=maximum_substeps,"Physical work cap exceeded");require(Clock::now()<deadline,"Qualification wall deadline reached");}
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}

CUDARL_HD p::BodyState reflect(p::BodyState body) {
    body.position.z=-body.position.z;body.orientation.x=-body.orientation.x;body.orientation.y=-body.orientation.y;
    body.velocity[2]=-body.velocity[2];body.velocity[3]=-body.velocity[3];body.velocity[4]=-body.velocity[4];
    body.angle[0]=-body.angle[0];body.angle[2]=-body.angle[2];body.velocity[6]=-body.velocity[6];body.velocity[8]=-body.velocity[8];return body;
}
Fixture fixture(int index) {
    Fixture out;out.material=index/4==0?ft::Material::Baseline:ft::Material::YawHalfStiffness;
    const int pose=(index%4)/2;const bool mirrored=index%2;
    out.initial.position={.2f,-2.f,.3f};
    out.initial.orientation=p::axis_rotation({0,1,0},.16f)*p::axis_rotation({0,0,1},pose?-.03f:0.f);
    const float velocity[]={.008f,-.0001f,-.0013f,.04f,-.11f,.06f,.21f,-.14f,.13f,-.18f};
    const float angle[]={.07f,-.04f,.16f,-.08f};
    std::copy(std::begin(velocity),std::end(velocity),out.initial.velocity);
    std::copy(std::begin(angle),std::end(angle),out.initial.angle);
    for(int tick=0;tick<short_steps;++tick){const double phase=2.*ft::pi*2.5*(tick+.5)*dt;
        out.commands[tick][0]=float(.5*std::sin(phase)+.1);out.commands[tick][1]=float(.13*std::sin(phase+.8));
        if(mirrored)out.commands[tick][0]=-out.commands[tick][0];}
    if(mirrored)out.initial=reflect(out.initial);return out;
}
CUDARL_HD Row advance(p::BodyState& body,const float* targets,ft::Material material,float h=dt) {
    Row row;row.valid=ft::step(body,targets,h,row.info,{}, {},material);row.body=body;row.pose=ft::snapshot(body);return row;
}
__global__ void parity_kernel(const Fixture* inputs,Row* rows) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=fixtures)return;
    auto body=inputs[id].initial;
    for(int tick=0;tick<short_steps;++tick){rows[id*short_steps+tick]=advance(body,inputs[id].commands[tick],inputs[id].material);if(!rows[id*short_steps+tick].valid)break;}
}
CUDARL_HD ReleaseResult release_run(const Release& input) {
    ReleaseResult out;auto body=input.initial;const auto initial=ft::snapshot(body);
    double previous=ft::elastic_energy(body,input.material)+ft::kinetic_energy(body),initial_energy=previous;
    const int steps=input.neutral?neutral_steps:release_steps;
    for(int tick=0;tick<steps;++tick){ft::StepInfo info;++out.attempted;
        if(!ft::step(body,nullptr,dt,info,input.neutral?p::Parameters{}:p::Parameters{0,0,0},{},input.material)){out.valid=false;break;}
        ++out.accepted;const auto pose=ft::snapshot(body);const double energy=info.elastic_energy_J+info.kinetic_energy_J;
        const double fraction=(energy-previous)/(1e-12+1e-4*fmax(fabs(previous),fabs(energy)));
        out.maximum_energy_fraction=fmax(out.maximum_energy_fraction,fraction);previous=energy;
        out.maximum_linear=fmax(out.maximum_linear,double(p::norm(pose.linear_momentum_kg_m_s)));
        out.maximum_angular=fmax(out.maximum_angular,double(p::norm(pose.angular_momentum_about_com_kg_m2_s)));
        if(info.root_work_J!=0||info.torque_Nm[0]!=0||info.torque_Nm[1]!=0)out.valid=false;
    }
    out.body=body;out.energy_ratio=initial_energy>0?previous/initial_energy:0;
    out.displacement_m=p::norm(ft::snapshot(body).com_m-initial.com_m);
    out.remaining_bend=fmaxf(fabsf(body.angle[2]),fabsf(body.angle[3]));return out;
}
__global__ void release_kernel(const Release* input,ReleaseResult* result,int size){const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<size)result[i]=release_run(input[i]);}

double difference(p::V a,p::V b){return std::max({std::abs(double(a.x)-b.x),std::abs(double(a.y)-b.y),std::abs(double(a.z)-b.z)});}
struct Errors {
    double position=0,orientation=0,joint=0,linear=0,angular=0,com=0,com_velocity=0,torque=0,energy=0;
    bool pass()const{return position<=2e-5&&orientation<=2e-5&&joint<=2e-5&&linear<=2e-7&&angular<=2e-5&&com<=2e-5*p::scale&&com_velocity<=2e-7&&torque<=2e-5*p::torque&&energy<=1;}
};
void compare(const Row& a,const Row& b,Errors& e){require(a.valid&&b.valid,"Invalid physical step");
    e.position=std::max(e.position,difference(a.body.position,b.body.position));
    const float qa[]={a.body.orientation.x,a.body.orientation.y,a.body.orientation.z,a.body.orientation.w};
    const float qb[]={b.body.orientation.x,b.body.orientation.y,b.body.orientation.z,b.body.orientation.w};
    for(int i=0;i<4;++i){e.orientation=std::max(e.orientation,std::abs(double(qa[i])-qb[i]));e.joint=std::max(e.joint,std::abs(double(a.body.angle[i])-b.body.angle[i]));}
    for(int i=0;i<10;++i){auto& error=i<3?e.linear:e.angular;error=std::max(error,std::abs(double(a.body.velocity[i])-b.body.velocity[i]));}
    e.com=std::max(e.com,difference(a.pose.com_m,b.pose.com_m));e.com_velocity=std::max(e.com_velocity,difference(a.pose.com_velocity_m_s,b.pose.com_velocity_m_s));
    for(int i=0;i<2;++i){e.torque=std::max(e.torque,std::abs(double(a.info.torque_Nm[i])-b.info.torque_Nm[i]));require(std::abs(a.info.torque_Nm[i])<=p::torque&&std::abs(b.info.torque_Nm[i])<=p::torque,"Torque cap exceeded");}
    const double ea[]={a.info.root_work_J,a.info.elastic_energy_J,a.info.drag_loss_J,a.info.bend_loss_J,a.info.kinetic_energy_J};
    const double eb[]={b.info.root_work_J,b.info.elastic_energy_J,b.info.drag_loss_J,b.info.bend_loss_J,b.info.kinetic_energy_J};
    for(int i=0;i<5;++i)e.energy=std::max(e.energy,std::abs(ea[i]-eb[i])/(1e-12+1e-4*std::max(std::abs(ea[i]),std::abs(eb[i]))));
}
bool body_exact(const p::BodyState& a,const p::BodyState& b){static_assert(sizeof(p::BodyState)==23*sizeof(float));return std::memcmp(&a,&b,sizeof(a))==0;}
}

int main(int argc,char** argv){const auto started=Clock::now();Errors parity,mirror;double timestep_com_m=0,timestep_joint_rad=0,maximum_release_energy_fraction=0,maximum_release_ratio=0,maximum_release_linear=0,maximum_release_angular=0;bool baseline_exact=true,bank_checks=false;int matched=0;
 try{require(argc==3,"Usage: pond_flexible_tail_material_check NEW_REPORT_JSON WALL_SECONDS");const double wall=std::stod(argv[2]);require(wall>0&&wall<=20,"Qualification requires a finite <=20 s execution cap");deadline=started+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(wall));require(!std::filesystem::exists(argv[1]),"Report must not already exist");
    std::array<Fixture,fixtures> inputs;for(int i=0;i<fixtures;++i)inputs[i]=fixture(i);
    std::vector<Row> gpu(fixtures*short_steps),cpu(gpu.size());Fixture* device_inputs=nullptr;Row* device_rows=nullptr;
    check(cudaMalloc(&device_inputs,sizeof(inputs)));check(cudaMalloc(&device_rows,gpu.size()*sizeof(Row)));check(cudaMemcpy(device_inputs,inputs.data(),sizeof(inputs),cudaMemcpyHostToDevice));check(cudaMemset(device_rows,0,gpu.size()*sizeof(Row)));
    parity_kernel<<<1,32>>>(device_inputs,device_rows);check(cudaGetLastError());
    for(int i=0;i<fixtures;++i){auto body=inputs[i].initial,legacy=body;
        for(int tick=0;tick<short_steps;++tick){++counts.cpu;cpu[i*short_steps+tick]=advance(body,inputs[i].commands[tick],inputs[i].material);
            if(inputs[i].material==ft::Material::Baseline){old::StepInfo info;++counts.legacy;require(old::step(legacy,inputs[i].commands[tick],dt,info),"Preserved physical step invalid");baseline_exact=baseline_exact&&body_exact(body,legacy)&&std::memcmp(&info,&cpu[i*short_steps+tick].info,sizeof(info))==0;}
            budget();}}
    check(cudaMemcpy(gpu.data(),device_rows,gpu.size()*sizeof(Row),cudaMemcpyDeviceToHost));check(cudaFree(device_rows));check(cudaFree(device_inputs));counts.cuda+=gpu.size();
    for(size_t i=0;i<gpu.size();++i){compare(cpu[i],gpu[i],parity);++matched;}
    require(baseline_exact,"Default material differs from preserved body arithmetic");require(parity.pass(),"CPU/CUDA parity exceeds existing tolerance");
    for(int i=0;i<fixtures;i+=2)for(int tick=0;tick<short_steps;++tick){auto reflected=cpu[i*short_steps+tick];reflected.body=reflect(reflected.body);reflected.pose=ft::snapshot(reflected.body);reflected.info.torque_Nm[0]=-reflected.info.torque_Nm[0];compare(reflected,cpu[(i+1)*short_steps+tick],mirror);}
    require(mirror.pass(),"Mirrored physics exceeds existing parity tolerance");
    for(int i=0;i<fixtures;++i){auto coarse=inputs[i].initial,fine=coarse;
        for(int tick=0;tick<800;++tick){const double time=(tick/2+.5)*dt;float targets[]={float(.3*std::sin(2.*ft::pi*2.5*time)+.06),float(.05*std::sin(2.*ft::pi*2.5*time+.8))};if(i%2)targets[0]=-targets[0];
            auto row=advance(fine,targets,inputs[i].material,.5f*dt);++counts.cpu;require(row.valid,"Fine timestep rejected");
            if(tick%2==0){auto c=advance(coarse,targets,inputs[i].material,dt);++counts.cpu;require(c.valid,"Coarse timestep rejected");}budget();}
        timestep_com_m=std::max(timestep_com_m,double(p::norm(ft::snapshot(coarse).com_m-ft::snapshot(fine).com_m)));for(int j=0;j<4;++j)timestep_joint_rad=std::max(timestep_joint_rad,std::abs(double(coarse.angle[j])-fine.angle[j]));}
    require(timestep_com_m<=.00025&&timestep_joint_rad<=.02,"Timestep convergence exceeds fixed 0.25 mm/0.02 rad limits");
    std::array<Release,10> releases;int ri=0;for(int material=0;material<2;++material){for(int axis=0;axis<2;++axis)for(int sign:{-1,1}){auto& r=releases[ri++];r.material=static_cast<ft::Material>(material);r.initial.angle[2+axis]=sign*.4f;}auto& r=releases[ri++];r.material=static_cast<ft::Material>(material);r.neutral=true;}
    Release* device_release=nullptr;ReleaseResult* device_results=nullptr;check(cudaMalloc(&device_release,sizeof(releases)));check(cudaMalloc(&device_results,10*sizeof(ReleaseResult)));check(cudaMemcpy(device_release,releases.data(),sizeof(releases),cudaMemcpyHostToDevice));
    release_kernel<<<1,32>>>(device_release,device_results,10);check(cudaGetLastError());std::array<ReleaseResult,10> cpu_release{},gpu_release{};
    for(int i=0;i<10;++i){budget();cpu_release[i]=release_run(releases[i]);counts.cpu+=cpu_release[i].attempted;}
    check(cudaMemcpy(gpu_release.data(),device_results,sizeof(gpu_release),cudaMemcpyDeviceToHost));check(cudaFree(device_release));check(cudaFree(device_results));
    for(int i=0;i<10;++i){counts.cuda+=gpu_release[i].attempted;for(const auto& r:{cpu_release[i],gpu_release[i]}){require(r.valid&&r.accepted==r.attempted,"Disconnected material step rejected");if(releases[i].neutral)require(r.displacement_m<=.0005f,"Neutral body drift exceeds existing 0.5 mm gate");else{maximum_release_energy_fraction=std::max(maximum_release_energy_fraction,r.maximum_energy_fraction);maximum_release_ratio=std::max(maximum_release_ratio,r.energy_ratio);maximum_release_linear=std::max(maximum_release_linear,r.maximum_linear);maximum_release_angular=std::max(maximum_release_angular,r.maximum_angular);require(r.maximum_energy_fraction<=1&&r.energy_ratio<.1&&r.remaining_bend<=.04f&&r.maximum_linear<5e-7&&r.maximum_angular<5e-9,"Passive energy/momentum gate failed");}}}budget();
    {p::BodyState body;const auto pose=ft::snapshot(body);const auto original=body;float values[ft::bank::inputs];const auto pond=cudarl::pond_geometry_v1::current_visual_parameters();ft::bank::observe(body,pose,pond,{.008f,0,0},true,values);require(body_exact(body,original),"Bank observer mutated physical state");for(float value:values)require(std::isfinite(value),"Nonfinite bank feature");require(values[ft::bank::CourseValid]==1&&values[ft::bank::TerrainValid]==1,"Bank validity flags missing");ft::bank::observe(body,pose,pond,{},false,values);require(values[ft::bank::CourseValid]==0&&values[ft::bank::TransportTimeValid]==0,"Stationary observer manufactured TTC");bank_checks=true;}
 }catch(const std::exception& error){failure=error.what();}
 const bool pass=failure.empty();const double wall=std::chrono::duration<double>(Clock::now()-started).count();
 if(argc>1){std::ofstream out(argv[1]);out<<std::setprecision(12)<<"{\"schema\":1,\"pass\":"<<(pass?"true":"false")<<",\"failure\":\""<<failure<<"\",\"cpu_substeps\":"<<counts.cpu<<",\"cuda_substeps\":"<<counts.cuda<<",\"preserved_baseline_substeps\":"<<counts.legacy<<",\"total_substeps\":"<<counts.cpu+counts.cuda+counts.legacy<<",\"maximum_substeps\":"<<maximum_substeps<<",\"optimizer_updates\":0,\"default_baseline_bit_exact\":"<<(baseline_exact?"true":"false")<<",\"matched_CPU_CUDA_rows\":"<<matched<<",\"CPU_CUDA_pass\":"<<(parity.pass()?"true":"false")<<",\"mirror_pass\":"<<(mirror.pass()?"true":"false")<<",\"maximum_COM_position_error_m\":"<<parity.com<<",\"maximum_COM_velocity_error_m_s\":"<<parity.com_velocity<<",\"maximum_joint_error_rad\":"<<parity.joint<<",\"maximum_angular_rate_error_rad_s\":"<<parity.angular<<",\"maximum_torque_error_Nm\":"<<parity.torque<<",\"maximum_energy_tolerance_fraction\":"<<parity.energy<<",\"timestep_COM_difference_m\":"<<timestep_com_m<<",\"timestep_joint_difference_rad\":"<<timestep_joint_rad<<",\"passive_maximum_energy_increase_fraction\":"<<maximum_release_energy_fraction<<",\"passive_final_energy_ratio\":"<<maximum_release_ratio<<",\"passive_linear_momentum_max\":"<<maximum_release_linear<<",\"passive_angular_momentum_max\":"<<maximum_release_angular<<",\"bank_observation_checks\":"<<(bank_checks?"true":"false")<<",\"wall_s\":"<<wall<<"}\n";}
 std::cout<<(pass?"PASS":"FAIL")<<" substeps="<<counts.cpu+counts.cuda+counts.legacy<<" wall_s="<<wall<<" "<<failure<<'\n';return pass?0:1;
}
