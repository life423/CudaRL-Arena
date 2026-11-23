// Zero-time algebra evaluated on historical fixed states. This program never
// calls a substep, advances a body, samples an actor, or invokes an optimizer.
#include "motor/joint_motor.h"
#include "cuda/cuda_memory.h"
#include "core/checkpoint.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace arithmetic_v2 {
namespace p=cudarl::physics;namespace m=cudarl::movement;namespace j=cudarl::joint_motor;
namespace cp=cudarl::checkpoint;namespace fs=std::filesystem;
constexpr int values=638;
constexpr unsigned profile_id=CUDARL_INTERFACE_UNFUSED?1u:0u;
constexpr const char* profile=CUDARL_INTERFACE_UNFUSED?"cpu_compatible_no_implicit_fma_v1":"historical_implicit_fma_reference_v1";
struct Input {p::BodyState body;float held[4]{},rate[4]{},command[4]{};p::Parameters fluid{};int original_control=0,source_backend=0;};
struct Output {double value[values]{};int filter_valid=0,algebra_valid=0;};
static_assert(sizeof(Input)==160&&sizeof(Output)==5112,"Arithmetic trace schema changed");
CUDARL_HD inline void put(Output& out,int& at,double value){out.value[at++]=value;}
CUDARL_HD inline void vector(Output& out,int& at,p::V value){put(out,at,value.x);put(out,at,value.y);put(out,at,value.z);}
CUDARL_HD inline void evaluate(const Input& input,Output& out){
    out={};int at=0;float angle[4],rate[4];for(int k=0;k<4;++k){angle[k]=input.held[k];rate[k]=input.rate[k];}
    m::Reference reference;out.filter_valid=j::filter_targets(angle,rate,input.command,reference)?1:0;
    for(float value:reference.acceleration)put(out,at,value);for(float value:reference.rate)put(out,at,value);for(float value:reference.angle)put(out,at,value);
    const auto links=m::shape(input.body);
    for(const auto& link:links.link){vector(out,at,link.center);put(out,at,link.q.x);put(out,at,link.q.y);put(out,at,link.q.z);put(out,at,link.q.w);
        vector(out,at,link.omega);vector(out,at,link.acceleration);vector(out,at,link.alpha);vector(out,at,link.inertia);put(out,at,link.mass);put(out,at,link.volume);
        for(const auto value:link.linear)vector(out,at,value);for(const auto value:link.angular)vector(out,at,value);}
    for(const auto& link:links.link)put(out,at,(input.fluid.density*link.volume-link.mass)*input.fluid.gravity);
    const auto motion=cudarl::motor::motion(input.body);vector(out,at,motion.com_model);vector(out,at,motion.velocity_m_s);
    const auto system=m::assemble(input.body,input.fluid,true);
    for(const auto& row:system.M)for(float value:row)put(out,at,value);for(const auto& row:system.R)for(float value:row)put(out,at,value);
    for(float value:system.C)put(out,at,value);for(float value:system.force)put(out,at,value);
    const float h=.005f;double A[10][10]{},rhs[10]{},root_rhs[10]{},root_v[10]{},desired_v[4]{};
    m::discrete_system(system,input.body,h,A,rhs);
    for(const auto& row:A)for(double value:row)put(out,at,value);for(double value:rhs)put(out,at,value);
    const double omega=2*static_cast<double>(m::pi)*3,zeta=.9;
    for(int k=0;k<4;++k){const double acceleration=reference.acceleration[k]+omega*omega*(reference.angle[k]-input.body.angle[k])+2*zeta*omega*(reference.rate[k]-input.body.velocity[6+k]);
        desired_v[k]=input.body.velocity[6+k]+h*acceleration;put(out,at,desired_v[k]);}
    for(int i=0;i<6;++i){root_rhs[i]=rhs[i];for(int k=0;k<4;++k)root_rhs[i]-=A[i][6+k]*desired_v[k];put(out,at,root_rhs[i]);}
    const bool root_ok=m::solve(A,root_rhs,root_v,6);for(int i=0;i<6;++i)put(out,at,root_v[i]);
    float requested[4]{},torque[4]{};
    for(int k=0;k<4;++k){double impulse=-rhs[6+k];for(int i=0;i<6;++i)impulse+=A[6+k][i]*root_v[i];for(int i=0;i<4;++i)impulse+=A[6+k][6+i]*desired_v[i];
        requested[k]=static_cast<float>(impulse/h);torque[k]=p::clamp(requested[k],-p::torque,p::torque);rhs[6+k]+=h*torque[k];}
    for(float value:requested)put(out,at,value);for(float value:torque)put(out,at,value);for(double value:rhs)put(out,at,value);
    double next[10]{};const bool next_ok=m::solve(A,rhs,next,10);for(double value:next)put(out,at,value);
    out.algebra_valid=at==values&&root_ok&&next_ok?1:0;
}
__global__ void fixed_state_evaluation(const Input* input,Output* output,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)evaluate(input[i],output[i]);}
struct Stage {const char* name;int count;const char* precision;};
const Stage stages[]={
    {"filter_acceleration",4,"float32"},{"filter_rate",4,"float32"},{"filter_angle",4,"float32"},
    {"link0",81,"float32"},{"link1",81,"float32"},{"link2",81,"float32"},
    {"buoyancy_force",3,"float32"},{"COM_model",3,"float32"},{"COM_velocity_m_s",3,"float32"},
    {"system_M",100,"float32"},{"system_R",100,"float32"},{"system_C",10,"float32"},{"system_force",10,"float32"},
    {"discrete_A",100,"float64"},{"discrete_rhs",10,"float64"},{"desired_joint_velocity",4,"float64"},
    {"root_rhs",6,"float64"},{"root_velocity",6,"float64"},{"requested_torque",4,"float32"},{"applied_torque",4,"float32"},
    {"actuated_rhs",10,"float64"},{"prospective_velocity",10,"float64"}};
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int run(const fs::path& source,const fs::path& destination){
    require(!fs::exists(destination),"Never overwrite an arithmetic evaluation");
    const auto payload=cp::read(source.string());cp::Reader in{payload};
    require(in.value<uint32_t>()==0x32414946&&in.value<uint32_t>()==2,"Wrong arithmetic input schema");
    const auto count=in.value<uint32_t>();require(count>0&&count<=16384&&in.value<uint32_t>()==sizeof(Input),"Wrong arithmetic input shape");
    const auto inputs=in.array<Input>(16384);in.finish();require(inputs.size()==count,"Arithmetic input count differs");
    std::vector<Output> cpu(count),gpu(count);for(size_t i=0;i<inputs.size();++i)evaluate(inputs[i],cpu[i]);
    device_ptr<Input> d_input(count);device_ptr<Output> d_output(count);d_input.copy_from_host(inputs.data(),count);
    fixed_state_evaluation<<<(count+63)/64,64>>>(d_input.get(),d_output.get(),static_cast<int>(count));
    CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());d_output.copy_to_host(gpu.data(),count);
    std::vector<Input> unchanged(count);d_input.copy_to_host(unchanged.data(),count);require(std::memcmp(inputs.data(),unchanged.data(),inputs.size()*sizeof(Input))==0,"Arithmetic fixture mutated input state");
    bool finite=true,neutral_exact=true,filter_exact=true;int neutral_count=0;double peak=0;
    for(size_t i=0;i<inputs.size();++i){finite=finite&&cpu[i].filter_valid&&cpu[i].algebra_valid&&gpu[i].filter_valid&&gpu[i].algebra_valid;
        for(int k=0;k<values;++k){finite=finite&&std::isfinite(cpu[i].value[k])&&std::isfinite(gpu[i].value[k]);peak=std::max(peak,std::abs(cpu[i].value[k]-gpu[i].value[k]));}
        filter_exact=filter_exact&&std::memcmp(cpu[i].value,gpu[i].value,12*sizeof(double))==0;
        if(inputs[i].fluid.density==1000){++neutral_count;for(int k=255;k<258;++k)neutral_exact=neutral_exact&&cpu[i].value[k]==0&&gpu[i].value[k]==0;}}
    require(fs::create_directories(destination),"Cannot create arithmetic evidence directory");
    cp::Writer out;out.value(uint32_t{0x324F4946});out.value(uint32_t{2});out.value(count);out.value(uint32_t{sizeof(Output)});out.value(profile_id);out.value(cp::checksum(payload));
    out.array(inputs);out.array(cpu);out.array(gpu);cp::write_atomic((destination/"trace.bin").string(),out.data);
    std::ofstream layout(destination/"trace-layout.json");layout<<"{\"schema\":1,\"input_bytes\":160,\"output_bytes\":5112,\"values\":638,\"stages\":[";int offset=0;
    for(const auto& s:stages){if(offset)layout<<',';layout<<"{\"name\":\""<<s.name<<"\",\"offset\":"<<offset<<",\"count\":"<<s.count<<",\"precision\":\""<<s.precision<<"\"}";offset+=s.count;}
    layout<<"],\"link_field_order\":[\"center3\",\"quaternion4\",\"omega3\",\"acceleration3\",\"alpha3\",\"inertia3\",\"mass1\",\"volume1\",\"linear_J30\",\"angular_J30\"],\"integration_performed\":false}\n";layout.close();require(bool(layout)&&offset==values,"Arithmetic trace layout write failed");
    std::ofstream summary(destination/"summary.json");summary<<std::setprecision(17)<<"{\"schema\":1,\"study\":\"frozen_interface_arithmetic_v2\",\"study_version\":2,\"arithmetic_profile\":\""<<profile
        <<"\",\"profile_id\":"<<profile_id<<",\"rows\":"<<count<<",\"CPU_zero_time_algebra_evaluations\":"<<count<<",\"CUDA_zero_time_algebra_evaluations\":"<<count
        <<",\"finite_and_algebra_valid\":"<<(finite?"true":"false")<<",\"neutral_density_rows\":"<<neutral_count<<",\"neutral_force_exact_zero\":"<<(neutral_exact?"true":"false")
        <<",\"filter_bits_exact\":"<<(filter_exact?"true":"false")<<",\"maximum_absolute_field_difference_mixed_units\":"<<peak
        <<",\"body_inputs_unchanged\":true,\"explicit_CUDA_kernel_launches\":1,\"native_runs\":1,\"physical_controls\":0,\"physical_substeps\":0,\"integration_calls\":0,\"PPO_updates\":0,\"optimizer_steps\":0}\n";summary.close();require(bool(summary),"Arithmetic report write failed");
    require(finite,"Fixed-state algebra failed");if(profile_id)require(neutral_exact&&filter_exact,"Unfused scalar/filter profile did not match");
    std::cout<<"Fixed-state arithmetic rows="<<count<<" neutral_exact="<<neutral_exact<<" filter_exact="<<filter_exact<<" physical_controls=0\n";return 0;
}
}
int main(int argc,char** argv){try{if(argc!=3)throw std::runtime_error("Use INPUT_BIN NEW_OUTPUT_DIRECTORY; zero integration only");
    return arithmetic_v2::run(std::filesystem::absolute(argv[1]),std::filesystem::absolute(argv[2]));
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
