// Fixed-input CPU/CUDA native physics gate. No optimizer or training path.
#include "cuda/pond_math_profile.h"
#include "motor/pond_motor_observation.h"
#include "motor/saved_motor_contract.h"
#include "rl/learned_amplitude_math.h"
#include "nn/mlp_cpu.h"
#include "core/checkpoint.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
namespace pr=cudarl::pond_runtime;
namespace po=cudarl::pond_motor_observation;
namespace pg=cudarl::pond_geometry_v1;
namespace bg=cudarl::pond_body_geometry;
namespace mp=cudarl::pond_math_profile;
namespace p=cudarl::physics;
namespace m=cudarl::movement;
namespace nn=cudarl::nn;
namespace cp=cudarl::checkpoint;
namespace {
constexpr int cases=8,controls_per_case=128;
constexpr uint64_t maximum_controls=2048,maximum_substeps=20480,evidence_cap=512ull*1024*1024;
constexpr double wall_cap=180;
constexpr const char* protocol_id="shoreline-product-v4-cuda-2026-11-23";
struct Options {bool preflight=false,run=false;fs::path policy,out,protocol,cpu_gate;};
struct Package {std::vector<float> weights;float gain=0;uint64_t checksum=0;};
struct Frame {pr::World world{};float command[4]{};pr::SubstepTrace trace[20]{};bool advanced=false;};
struct Algebra {p::System system{};bg::Inspection inspection{};float radii[3]{};};
struct Budget {
    uint64_t cpu_controls=0,gpu_controls=0,cpu_substeps=0,gpu_substeps=0,kernels=0,pure_cpu_bank_queries=0,bytes=0;
    bool gpu_substep_readback_complete=true;
    double cpu_reference_ms=0,gpu_kernel_ms=0,transfer_wall_ms=0,evidence_wall_ms=0;
    const std::chrono::steady_clock::time_point began=std::chrono::steady_clock::now();
    void guard() const {
        if(cpu_controls+gpu_controls+16>maximum_controls||cpu_substeps+gpu_substeps+160>maximum_substeps)
            throw std::runtime_error("Fixed CPU/CUDA gate budget exhausted");
        if(std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count()>=wall_cap)
            throw std::runtime_error("Fixed CPU/CUDA gate wall cap reached");
    }
    void charge(uint64_t count) {if(bytes+count>evidence_cap)throw std::runtime_error("CUDA gate evidence cap reached");bytes+=count;}
};
void require(bool yes,const std::string& why) {if(!yes)throw std::runtime_error(why);}
void cuda_require(cudaError_t status,const char* operation) {if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));}
std::string escape(const std::string& input) {
    std::string result;
    for(char c:input){if(c=='"'||c=='\\')result+='\\';if(c=='\n')result+="\\n";else if(c=='\r')result+="\\r";else result+=c;}
    return result;
}
std::vector<uint8_t> whole_file(const fs::path& path) {
    std::ifstream file(path,std::ios::binary);require(bool(file),"Missing binding file: "+path.string());
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)),{});
}
Package package(const fs::path& path) {
    const auto bytes=cp::read(path.string());cp::Reader r{bytes};Package out;out.checksum=cp::checksum(bytes);
    require(r.value<uint32_t>()==0x31564c48&&r.value<uint32_t>()==2,"Wrong saved package format");
    for(uint32_t expected:{6u,6u,3u,4u,41u,4u,336u,1u,7u,1u})require(r.value<uint32_t>()==expected,"Saved motor contract mismatch");
    require(r.value<uint64_t>()!=0,"Saved protocol missing");const auto parent=r.array<uint8_t>(64);
    require(std::string(parent.begin(),parent.end())=="810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0","Saved motor parent mismatch");
    const auto& sources=cudarl::motor::saved_motor_source_contract_hashes();require(r.value<uint32_t>()==sources.size(),"Saved source count mismatch");
    for(const auto expected:sources){const auto hash=r.array<uint8_t>(64);require(std::string(hash.begin(),hash.end())==expected,"Saved motor source provenance mismatch");}
    out.weights=r.array<float>(2532);const auto head=r.array<float>(2);const float gain=r.value<float>(),sigma=r.value<float>();r.finish();
    require(out.weights.size()==2532&&head.size()==2&&std::isfinite(head[0])&&std::isfinite(head[1]),"Saved motor parameter shape");
    for(float value:out.weights)require(std::isfinite(value),"Saved motor nonfinite parameter");
    out.gain=cudarl::learned_amplitude::sigmoid(head[0]);
    require(std::isfinite(gain)&&std::isfinite(sigma)&&std::abs(gain-out.gain)<=2e-7f&&
        std::abs(sigma-cudarl::learned_amplitude::standard_deviation(head[1]))<=2e-6f,"Saved amplitude mapping mismatch");
    return out;
}
nn::CpuMlp<float> actor(const Package& saved) {
    nn::CpuMlp<float> net({{41,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}});
    net.params()=saved.weights;return net;
}
Frame initialize(int id,po::RequestState& request,Budget& count) {
    Frame out;auto& state=out.world.motor.physical;
    if(id<4) {
        require(po::initialize(id,false,state.body,state.held_targets,out.world.motor.reference_rate,state.last_actual_torques,request),"Straight fixture initialization failed");
        state.paused=false;
    } else {
        const float angle=float(id-4)*po::pi*.5f;const p::V direction{cosf(angle),0,sinf(angle)};
        state.body.orientation=p::axis_rotation({0,1,0},-angle);state.paused=false;
        const auto query=[&](double radius) {
            auto body=state.body;body.position.x=direction.x*static_cast<float>(radius/p::scale);body.position.z=direction.z*static_cast<float>(radius/p::scale);
            ++count.pure_cpu_bank_queries;return bg::inspect(body,out.world.terrain,pr::query_tolerance_m,pr::query_visit_limit);
        };
        const double eangle=atan2(double(direction.z)/out.world.terrain.water_radius_z_m,double(direction.x)/out.world.terrain.water_radius_x_m);
        const double shore=pg::edge(eangle)/hypot(double(direction.x)/out.world.terrain.water_radius_x_m,double(direction.z)/out.world.terrain.water_radius_z_m);
        double low=0,high=shore*.8;const auto lower=query(low),upper=query(high);
        require(lower.valid&&lower.certified&&upper.valid&&upper.certified&&lower.lower_gap_m>.0001&&upper.lower_gap_m<.0001,"Bank fixture bracket failed");
        for(int i=0;i<48;++i){const double center=(low+high)*.5;const auto q=query(center);require(q.valid&&q.certified,"Bank fixture certificate failed");if(q.lower_gap_m>=.0001)low=center;else high=center;}
        state.body.position.x=direction.x*static_cast<float>(low/p::scale);state.body.position.z=direction.z*static_cast<float>(low/p::scale);
        state.body.velocity[0]=direction.x*.005f;state.body.velocity[2]=direction.z*.005f;
        request.start_com_model=po::motion(state.body).com_model;request.request.heading=direction;
        request.request.position=request.start_com_model+direction*(.072f/p::scale);request.request.terminal_velocity=direction*.008f;
    }
    require(pr::prepare(out.world),"Pond fixture prepare failed");return out;
}
__global__ void static_algebra(const Frame* input,Algebra* output) {
    const int index=blockIdx.x;if(threadIdx.x!=0||index>=24)return;
    const int id=index%cases,density=index/cases;const float densities[3]={975,1000,1025};
    p::Parameters fluid;fluid.density=densities[density];output[index].system=m::assemble(input[id].world.motor.physical.body,fluid,true);
    if(density==0) {
        output[index].inspection=bg::inspect(input[id].world.motor.physical.body,input[id].world.terrain,pr::query_tolerance_m,pr::query_visit_limit);
        const auto shape=p::shape(input[id].world.motor.physical.body);
        for(int body=0;body<3;++body)output[index].radii[body]=mp::radius(3*shape.link[body].volume/(4*m::pi));
    }
}
__global__ void advance(Frame* frames) {
    const int id=blockIdx.x;if(threadIdx.x!=0||id>=cases)return;
    frames[id].advanced=pr::step(frames[id].world,frames[id].command,.005f,frames[id].trace);
}
__global__ void inspect_fixtures(const Frame* input,bg::Inspection* output) {
    const int id=blockIdx.x;if(threadIdx.x!=0||id>=cases)return;
    const auto& w=input[id].world;
    output[id]=bg::inspect(w.motor.physical.body,w.terrain,pr::query_tolerance_m,pr::query_visit_limit);
}
std::array<float,23> body_values(const p::BodyState& body) {
    std::array<float,23> out{body.position.x,body.position.y,body.position.z,body.orientation.x,body.orientation.y,body.orientation.z,body.orientation.w};
    for(int j=0;j<4;++j)out[static_cast<size_t>(7+j)]=body.angle[j];
    for(int j=0;j<10;++j)out[static_cast<size_t>(11+j)]=body.velocity[j];
    out[21]=body.work;out[22]=body.travel;return out;
}
struct Differences {double position=0,orientation=0,joint=0,linear_velocity=0,angular_velocity=0,motor=0,energy=0,requested_torque=0;};
void finite_difference(double a,double b,double tolerance,double& maximum,const std::string& stage) {
    const double error=std::abs(a-b);maximum=std::max(maximum,error);
    if(!std::isfinite(a)||!std::isfinite(b)||error>tolerance) {
        std::ostringstream why;why<<std::setprecision(17)<<stage<<" CPU="<<a<<" CUDA="<<b<<" difference="<<error<<" limit="<<tolerance;
        throw std::runtime_error(why.str());
    }
}
void compare_body(const p::BodyState& a,const p::BodyState& b,Differences& delta,const std::string& stage) {
    const auto left=body_values(a),right=body_values(b);
    for(int j=0;j<21;++j) {
        double* maximum=j<3?&delta.position:j<7?&delta.orientation:j<11?&delta.joint:j<14?&delta.linear_velocity:&delta.angular_velocity;
        finite_difference(left[static_cast<size_t>(j)],right[static_cast<size_t>(j)],j>=11&&j<14?2e-7:2e-5,*maximum,stage+" body["+std::to_string(j)+"]");
    }
}
void compare_trace(const pr::SubstepTrace& a,const pr::SubstepTrace& b,Differences& delta,const std::string& stage) {
    require(a.recorded==b.recorded&&a.attempted==b.attempted&&a.accepted==b.accepted&&a.failure==b.failure&&
        a.attempted_substeps==b.attempted_substeps&&a.accepted_substeps==b.accepted_substeps,stage+" acceptance/counter difference");
    if(!a.recorded)return;
    compare_body(a.before,b.before,delta,stage+" PRE");compare_body(a.candidate,b.candidate,delta,stage+" candidate");compare_body(a.after,b.after,delta,stage+" POST");
    for(int j=0;j<4;++j) {
        finite_difference(a.held_targets[j],b.held_targets[j],2e-5,delta.motor,stage+" held_target");
        finite_difference(a.reference_rate[j],b.reference_rate[j],2e-5,delta.motor,stage+" reference_rate");
        finite_difference(a.last_actual_torques[j],b.last_actual_torques[j],2e-5,delta.motor,stage+" last_actual_torque");
        finite_difference(a.applied_torque_normalized[j],b.applied_torque_normalized[j],2e-5,delta.motor,stage+" actual_substep_torque");
        delta.requested_torque=std::max(delta.requested_torque,std::abs(double(a.requested_torque_Nm[j])-b.requested_torque_Nm[j]));
    }
    for(uint32_t j=0;j<pr::trace_energy_fields;++j)
        finite_difference(a.energy[j],b.energy[j],1e-12+1e-4*std::max(std::abs(a.energy[j]),std::abs(b.energy[j])),delta.energy,stage+" energy["+std::to_string(j)+"]");
}
void compare_algebra(const p::System& a,const p::System& b,double& maximum) {
    for(int row=0;row<10;++row) {
        finite_difference(a.force[row],b.force[row],1e-12+2e-5*std::max(std::abs(a.force[row]),std::abs(b.force[row])),maximum,"pure force");
        finite_difference(a.C[row],b.C[row],1e-12+2e-5*std::max(std::abs(a.C[row]),std::abs(b.C[row])),maximum,"pure bias");
        for(int column=0;column<10;++column) {
            finite_difference(a.M[row][column],b.M[row][column],1e-12+2e-5*std::max(std::abs(a.M[row][column]),std::abs(b.M[row][column])),maximum,"pure mass");
            finite_difference(a.R[row][column],b.R[row][column],1e-12+2e-5*std::max(std::abs(a.R[row][column]),std::abs(b.R[row][column])),maximum,"pure resistance");
        }
    }
}
template<class T>struct Device {
    T* pointer=nullptr;
    explicit Device(size_t count){cuda_require(cudaMalloc(&pointer,count*sizeof(T)),"CUDA allocation");}
    ~Device(){if(pointer)cudaFree(pointer);}
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
};
void synchronize() {cuda_require(cudaGetLastError(),"CUDA kernel launch");cuda_require(cudaDeviceSynchronize(),"CUDA kernel completion");}
struct KernelTimer {
    cudaEvent_t begin{},end{};
    KernelTimer() {
        cuda_require(cudaEventCreate(&begin),"Create GPU start event");
        cuda_require(cudaEventCreate(&end),"Create GPU end event");
    }
    ~KernelTimer(){cudaEventDestroy(end);cudaEventDestroy(begin);}
    void start(){cuda_require(cudaEventRecord(begin),"Record GPU start");}
    float stop() {
        cuda_require(cudaGetLastError(),"Launch physics kernel");
        cuda_require(cudaEventRecord(end),"Record GPU end");
        cuda_require(cudaEventSynchronize(end),"Complete physics kernel");
        float ms=0;cuda_require(cudaEventElapsedTime(&ms,begin,end),"Time physics kernel");return ms;
    }
};
double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
void layout(const fs::path& out) {
    std::ofstream file(out/"layout.json");file<<"{\"schema\":1,\"frame_bytes\":"<<sizeof(Frame)<<",\"world_bytes\":"<<sizeof(pr::World)
        <<",\"trace_bytes\":"<<sizeof(pr::SubstepTrace)<<",\"frame_world_offset\":"<<offsetof(Frame,world)
        <<",\"frame_command_offset\":"<<offsetof(Frame,command)<<",\"frame_trace_offset\":"<<offsetof(Frame,trace)
        <<",\"frame_advanced_offset\":"<<offsetof(Frame,advanced)<<",\"trace_before_offset\":"<<offsetof(pr::SubstepTrace,before)
        <<",\"trace_candidate_offset\":"<<offsetof(pr::SubstepTrace,candidate)<<",\"trace_after_offset\":"<<offsetof(pr::SubstepTrace,after)
        <<",\"trace_energy_offset\":"<<offsetof(pr::SubstepTrace,energy)<<",\"trace_energy_fields\":"<<pr::trace_energy_fields
        <<",\"body_bytes\":"<<sizeof(p::BodyState)<<",\"raw_payload_order\":\"magic_u32,version_u32,tick_i32,frame_bytes_u32,case_count_u32,CPU_frames[8],CUDA_frames[8]\"}\n";
    require(bool(file),"Layout evidence write failed");
}
void raw_frames(const fs::path& out,const char* name,int tick,const std::array<Frame,cases>& cpu,const std::array<Frame,cases>& gpu,Budget& budget) {
    cp::Writer bytes;bytes.value(uint32_t{0x32474350});bytes.value(uint32_t{1});bytes.value(tick);
    bytes.value(static_cast<uint32_t>(sizeof(Frame)));bytes.value(uint32_t{cases});
    for(const auto& frame:cpu)bytes.value(frame);for(const auto& frame:gpu)bytes.value(frame);
    budget.charge(bytes.data.size()+24);cp::write_atomic((out/name).string(),bytes.data);
}
void trace_header(std::ostream& stream) {
    stream<<"case,backend,control,substep,recorded,attempted,accepted,failure,attempted_substeps,accepted_substeps,contact_system_evaluations,contact_iterations";
    for(const char* phase:{"before","candidate","after"})for(int j=0;j<23;++j)stream<<','<<phase<<j;
    for(const char* name:{"held","reference_rate","last_actual","requested_Nm","actual_normalized"})for(int j=0;j<4;++j)stream<<','<<name<<j;
    for(uint32_t j=0;j<pr::trace_energy_fields;++j)stream<<",energy"<<j;
    stream<<",minimum_gap_m,gap_lower_m,gap_upper_m,step_spring_work_J,step_damping_loss_J,step_potential_before_J,step_potential_after_J\n";
}
void trace_row(std::ofstream& file,int id,const char* backend,int control,int substep,const pr::SubstepTrace& trace,Budget& budget) {
    std::ostringstream row;row<<std::setprecision(17)<<id<<','<<backend<<','<<control<<','<<substep+1<<','<<trace.recorded<<','<<trace.attempted<<','<<trace.accepted
        <<','<<trace.failure<<','<<trace.attempted_substeps<<','<<trace.accepted_substeps<<','<<trace.contact_system_evaluations<<','<<trace.contact_iterations;
    for(const auto* body:{&trace.before,&trace.candidate,&trace.after})for(float value:body_values(*body))row<<','<<value;
    for(const auto* values:{trace.held_targets,trace.reference_rate,trace.last_actual_torques,trace.requested_torque_Nm,trace.applied_torque_normalized})
        for(int j=0;j<4;++j)row<<','<<values[j];
    for(double value:trace.energy)row<<','<<value;
    row<<','<<trace.minimum_gap_m<<','<<trace.gap_lower_m<<','<<trace.gap_upper_m<<','<<trace.step_spring_work<<','<<trace.step_damping_loss
        <<','<<trace.step_potential_before<<','<<trace.step_potential_after<<'\n';
    const auto line=row.str();budget.charge(line.size());file<<line;require(bool(file),"Substep evidence write failed");
}
void result(const Options& options,const Package& saved,const Budget& count,bool pass,const std::string& reason,
            const Differences& differences,int controls_completed,uint64_t protocol_hash=0,uint64_t cpu_gate_hash=0) {
    std::ofstream out(options.out/(options.preflight?"preflight.json":"result.json"));
    out<<std::setprecision(17)<<"{\"schema\":1,\"protocol_id\":\""<<protocol_id<<"\",\"runtime_version\":"<<pr::version
        <<",\"arithmetic_profile\":\"cpu_compatible_no_implicit_fma_v1\",\"canonical_radius_version\":1,\"preflight_only\":"<<(options.preflight?"true":"false")
        <<",\"passed\":"<<(pass?"true":"false")<<",\"reason\":\""<<escape(reason)<<"\",\"saved_payload_checksum_fnv1a64\":"<<saved.checksum
        <<",\"protocol_checksum_fnv1a64\":"<<protocol_hash<<",\"cpu_gate_checksum_fnv1a64\":"<<cpu_gate_hash
        <<",\"CPU_attempted_controls\":"<<count.cpu_controls<<",\"CUDA_attempted_controls\":"<<count.gpu_controls
        <<",\"CPU_attempted_substeps\":"<<count.cpu_substeps<<",\"CUDA_attempted_substeps\":";
    if(count.gpu_substep_readback_complete)out<<count.gpu_substeps;else out<<"null";
    out<<",\"CUDA_attempted_substeps_upper_bound\":"<<count.gpu_substeps+(count.gpu_substep_readback_complete?0:cases*10)
        <<",\"CUDA_substep_readback_complete\":"<<(count.gpu_substep_readback_complete?"true":"false")
        <<",\"CUDA_explicit_kernels\":"<<count.kernels<<",\"pure_CPU_bank_placement_queries\":"<<count.pure_cpu_bank_queries
        <<",\"controls_completed_per_case\":"<<controls_completed<<",\"maximum_position_difference_model\":"<<differences.position
        <<",\"maximum_quaternion_difference\":"<<differences.orientation<<",\"maximum_joint_angle_difference_rad\":"<<differences.joint
        <<",\"maximum_linear_velocity_difference_m_s\":"<<differences.linear_velocity<<",\"maximum_angular_joint_rate_difference_rad_s\":"<<differences.angular_velocity
        <<",\"maximum_motor_difference\":"<<differences.motor<<",\"maximum_energy_difference_J\":"<<differences.energy
        <<",\"maximum_requested_torque_difference_Nm_diagnostic\":"<<differences.requested_torque
        <<",\"query_arithmetic_version\":"<<bg::query_arithmetic_version
        <<",\"cpu_reference_wall_ms\":"<<count.cpu_reference_ms<<",\"gpu_physics_event_ms\":"<<count.gpu_kernel_ms
        <<",\"transfer_wall_ms\":"<<count.transfer_wall_ms<<",\"evidence_wall_ms\":"<<count.evidence_wall_ms
        <<",\"evidence_generated_bytes\":"<<count.bytes<<",\"optimizer_steps\":0,\"PPO_updates\":0,\"GPU_NN_forward_calls\":0"
        <<",\"autonomous_navigation_qualified\":false,\"wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-count.began).count()<<"}\n";
    out.flush();require(bool(out),"Result write failed");
}
Options options(int argc,char** argv) {
    Options out;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];if(arg=="--preflight")out.preflight=true;else if(arg=="--run")out.run=true;
        else if(arg=="--policy"&&i+1<argc)out.policy=argv[++i];else if(arg=="--out"&&i+1<argc)out.out=argv[++i];
        else if(arg=="--protocol"&&i+1<argc)out.protocol=argv[++i];else if(arg=="--cpu-gate"&&i+1<argc)out.cpu_gate=argv[++i];
        else throw std::runtime_error("Unknown or incomplete argument");
    }
    require(out.preflight!=out.run&&!out.policy.empty()&&!out.out.empty(),"Require one mode, --policy and fresh --out");
    if(out.run)require(!out.protocol.empty()&&!out.cpu_gate.empty(),"Dynamic parity requires frozen --protocol and passed --cpu-gate");
    return out;
}
void validate_cpu_gate(const std::vector<uint8_t>& bytes) {
    const std::string text(bytes.begin(),bytes.end());
    require(std::regex_search(text,std::regex("\"passed\"\\s*:\\s*true")),"CPU bank/retention qualification has not passed");
    require(std::regex_search(text,std::regex("\"protocol_id\"\\s*:\\s*\"shoreline-product-v4-2026-11-23\"")),"CPU shoreline proof has a different protocol");
    require(std::regex_search(text,std::regex("\"preflight_only\"\\s*:\\s*false")),"CPU proof is only zero-integration preflight");
}
int preflight(const Options& options,const Package& saved,std::array<Frame,cases>& frames,Budget& budget) {
    Device<Frame> input(cases);Device<Algebra> output(24);std::array<Algebra,24> gpu;
    cuda_require(cudaMemcpy(input.pointer,frames.data(),sizeof(frames),cudaMemcpyHostToDevice),"Static frame upload");
    ++budget.kernels;static_algebra<<<24,32>>>(input.pointer,output.pointer);synchronize();
    cuda_require(cudaMemcpy(gpu.data(),output.pointer,sizeof(gpu),cudaMemcpyDeviceToHost),"Static algebra download");
    double maximum=0;cp::Writer raw;raw.value(uint32_t{0x31414350});raw.value(uint32_t{1});
    for(int i=0;i<24;++i) {
        p::Parameters fluid;fluid.density=i/cases==0?975.f:i/cases==1?1000.f:1025.f;
        const auto host=m::assemble(frames[static_cast<size_t>(i%cases)].world.motor.physical.body,fluid,true);
        compare_algebra(host,gpu[static_cast<size_t>(i)].system,maximum);raw.value(host);raw.value(gpu[static_cast<size_t>(i)]);
        if(i<cases) {
            const auto& h=frames[static_cast<size_t>(i)].world.inspection;const auto& g=gpu[static_cast<size_t>(i)].inspection;
            require(h.valid==g.valid&&h.certified==g.certified&&h.failure_reason==g.failure_reason,"Static geometry scope mismatch");
            require(h.lower_gap_m<=g.upper_gap_m&&g.lower_gap_m<=h.upper_gap_m&&std::abs(h.upper_gap_m-g.upper_gap_m)<=1e-6,"Static geometry intervals differ");
            for(int body=0;body<3;++body)require(mp::bits(gpu[static_cast<size_t>(i)].radii[body])==mp::original_cpu_result_bits[body],"Canonical CUDA radius bits differ");
        }
    }
    budget.charge(raw.data.size()+24);cp::write_atomic((options.out/"static-algebra.bin").string(),raw.data);
    Device<bg::Inspection> query_output(cases);KernelTimer timer;
    std::ofstream timing(options.out/"fixture-query-timing.csv");
    timing<<"repetition,eight_world_kernel_ms,case,visited_points,lower_gap_m,upper_gap_m\n";
    for(int rep=0;rep<2;++rep) {
        timer.start();++budget.kernels;inspect_fixtures<<<cases,32>>>(input.pointer,query_output.pointer);
        const float ms=timer.stop();std::array<bg::Inspection,cases> observations;
        cuda_require(cudaMemcpy(observations.data(),query_output.pointer,sizeof(observations),cudaMemcpyDeviceToHost),"Read fixed geometry timings");
        for(int id=0;id<cases;++id) {
            const auto& q=observations[static_cast<size_t>(id)];
            const auto& original=gpu[static_cast<size_t>(id)].inspection;
            require(q.valid&&q.certified&&q.visits==original.visits&&q.lower_gap_m==original.lower_gap_m&&q.upper_gap_m==original.upper_gap_m,"Repeated fixture query differs");
            timing<<std::setprecision(17)<<rep<<','<<ms<<','<<id<<','<<q.visits<<','<<q.lower_gap_m<<','<<q.upper_gap_m<<'\n';
        }
    }
    std::ofstream algebra(options.out/"static-algebra.json");algebra<<std::setprecision(17)<<"{\"passed\":true,\"CPU_assemble_evaluations\":24,\"CUDA_assemble_evaluations\":24,\"CPU_geometry_prepares\":8,\"CUDA_geometry_queries\":8,\"maximum_scalar_difference\":"<<maximum<<",\"physical_controls\":0,\"physical_substeps\":0,\"optimizer_steps\":0}\n";
    result(options,saved,budget,true,"zero_integration_static_profile_passed",Differences{},0);return 0;
}
int run(const Options& options,const Package& saved,std::array<Frame,cases>& cpu,std::array<po::RequestState,cases>& requests,Budget& budget) {
    const auto document=whole_file(options.protocol),proof=whole_file(options.cpu_gate);validate_cpu_gate(proof);
    const uint64_t document_hash=cp::checksum(document),proof_hash=cp::checksum(proof);
    auto network=actor(saved);std::array<Frame,cases> gpu=cpu;
    KernelTimer timer;
    Device<Frame> device(cases);cuda_require(cudaMemcpy(device.pointer,gpu.data(),sizeof(gpu),cudaMemcpyHostToDevice),"Initial identical world upload");
    fs::create_directory(options.out/"frames");std::ofstream traces(options.out/"substeps.csv"),commands(options.out/"commands.csv"),case_summary(options.out/"cases.csv");
    trace_header(traces);commands<<std::setprecision(17)<<"control,case,command0,command1,command2,command3\n";
    case_summary<<"case,category,planned_controls,completed_controls,status\n";
    std::ofstream start(options.out/"run-started.json");start<<"{\"protocol_id\":\""<<protocol_id<<"\",\"attempts_allowed\":1,\"physical_controls_cap\":2048,\"physical_substeps_cap\":20480,\"optimizer_steps\":0}\n";start.close();
    Differences differences;int complete=0;std::string fault;
    try {
        for(int tick=1;tick<=controls_per_case;++tick) {
            budget.guard();
            for(int id=0;id<cases;++id) {
                auto& frame=cpu[static_cast<size_t>(id)];std::fill(frame.command,frame.command+4,0.f);
                if(id<4) {
                    float observation[41]{};require(po::observe(requests[static_cast<size_t>(id)],frame.world.motor.physical.body,frame.world.motor.physical.held_targets,
                        frame.world.motor.reference_rate,frame.world.motor.physical.last_actual_torques,observation),"CPU fixed-input command observation failed");
                    network.forward(observation,1);for(int j=0;j<4;++j)frame.command[j]=tanhf(network.output()[j]);
                }
                commands<<tick<<','<<id;for(float value:frame.command)commands<<','<<value;commands<<'\n';
                std::copy(frame.command,frame.command+4,gpu[static_cast<size_t>(id)].command);
            }
            // Upload the commands with their independently continued GPU worlds.
            auto phase=std::chrono::steady_clock::now();
            cuda_require(cudaMemcpy(device.pointer,gpu.data(),sizeof(gpu),cudaMemcpyHostToDevice),"Identical commands and continued GPU state upload");
            budget.transfer_wall_ms+=elapsed_ms(phase);
            std::array<uint64_t,cases> gpu_before{};
            for(int id=0;id<cases;++id)gpu_before[static_cast<size_t>(id)]=gpu[static_cast<size_t>(id)].world.motor.attempted_substeps;
            phase=std::chrono::steady_clock::now();
            for(int id=0;id<cases;++id) {
                auto& frame=cpu[static_cast<size_t>(id)];const auto before=frame.world.motor.attempted_substeps;
                ++budget.cpu_controls;frame.advanced=pr::step(frame.world,frame.command,.005f,frame.trace);
                budget.cpu_substeps+=frame.world.motor.attempted_substeps-before;
                if(frame.advanced)++requests[static_cast<size_t>(id)].controls;
                require(frame.advanced,"CPU pond gate physical failure before CUDA comparison");
            }
            budget.cpu_reference_ms+=elapsed_ms(phase);
            timer.start();budget.gpu_controls+=cases;budget.gpu_substep_readback_complete=false;
            ++budget.kernels;advance<<<cases,32>>>(device.pointer);budget.gpu_kernel_ms+=timer.stop();
            phase=std::chrono::steady_clock::now();
            cuda_require(cudaMemcpy(gpu.data(),device.pointer,sizeof(gpu),cudaMemcpyDeviceToHost),"Continued GPU state and trace download");
            budget.transfer_wall_ms+=elapsed_ms(phase);
            for(int id=0;id<cases;++id)budget.gpu_substeps+=gpu[static_cast<size_t>(id)].world.motor.attempted_substeps-gpu_before[static_cast<size_t>(id)];
            budget.gpu_substep_readback_complete=true;
            phase=std::chrono::steady_clock::now();
            std::ostringstream filename;filename<<"frames/step-"<<std::setw(4)<<std::setfill('0')<<tick<<".bin";
            raw_frames(options.out,filename.str().c_str(),tick,cpu,gpu,budget);
            for(int substep=0;substep<10;++substep)for(int id=0;id<cases;++id) {
                trace_row(traces,id,"CPU",tick,substep,cpu[static_cast<size_t>(id)].trace[substep],budget);
                trace_row(traces,id,"CUDA",tick,substep,gpu[static_cast<size_t>(id)].trace[substep],budget);
            }
            traces.flush();commands.flush();require(bool(traces)&&bool(commands),"Gate trace flush failed");
            budget.evidence_wall_ms+=elapsed_ms(phase);
            for(int substep=0;substep<10;++substep)for(int id=0;id<cases;++id)
                compare_trace(cpu[static_cast<size_t>(id)].trace[substep],gpu[static_cast<size_t>(id)].trace[substep],differences,
                    "case"+std::to_string(id)+" control"+std::to_string(tick)+" substep"+std::to_string(substep+1));
            for(int id=0;id<cases;++id)require(gpu[static_cast<size_t>(id)].advanced,"CUDA pond runtime physical failure");
            complete=tick;
        }
        require(budget.cpu_controls+budget.gpu_controls==maximum_controls&&budget.cpu_substeps+budget.gpu_substeps==maximum_substeps,"Completed CPU/CUDA gate accounting differs");
        require(cp::checksum(cp::read(options.policy.string()))==saved.checksum&&cp::checksum(whole_file(options.protocol))==document_hash&&cp::checksum(whole_file(options.cpu_gate))==proof_hash,"Gate inputs changed");
    }catch(const std::exception& error){
        fault=error.what();
        raw_frames(options.out,"terminal-last-known-frames.bin",complete+1,cpu,gpu,budget);
    }
    for(int id=0;id<cases;++id)case_summary<<id<<','<<(id<4?"straight_saved_motor":"bank_impact_zero_commands")<<','<<controls_per_case<<','<<complete<<','<<(fault.empty()?"completed":"stopped_shared_gate_failure")<<'\n';
    if(!fault.empty()){std::ofstream note(options.out/"first-failure.json");note<<"{\"reason\":\""<<escape(fault)<<"\",\"last_complete_control\":"<<complete<<"}\n";}
    result(options,saved,budget,fault.empty(),fault.empty()?"matched_fixed_input_dynamics_passed":fault,differences,complete,document_hash,proof_hash);
    return fault.empty()?0:2;
}
}
int main(int argc,char** argv) {
    Options config;Budget budget;Package saved;bool owns_output=false;
    try {
        config=options(argc,argv);require(!fs::exists(config.out)&&fs::create_directories(config.out),"Refusing an existing evidence directory");owns_output=true;
        require(pr::version==4&&bg::query_arithmetic_version==2,"CUDA gate requires the corrected shoreline gap gradient");mp::validate_host_geometry();saved=package(config.policy);
        std::array<Frame,cases> frames;std::array<po::RequestState,cases> requests;
        for(int id=0;id<cases;++id)frames[static_cast<size_t>(id)]=initialize(id,requests[static_cast<size_t>(id)],budget);
        layout(config.out);
        if(config.preflight)return preflight(config,saved,frames,budget);
        return run(config,saved,frames,requests,budget);
    }catch(const std::exception& error) {
        if(owns_output)result(config,saved,budget,false,error.what(),Differences{},0);
        std::cerr<<error.what()<<'\n';return 2;
    }
}
