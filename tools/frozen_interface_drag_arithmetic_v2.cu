// One immutable-input drag-arithmetic evaluation. No body integration.
#include "motor/joint_motor.h"
#include "cuda/cuda_memory.h"
#include "core/checkpoint.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace drag_trace_v2 {
namespace p=cudarl::physics;namespace m=cudarl::movement;namespace cp=cudarl::checkpoint;namespace fs=std::filesystem;
struct Input {p::BodyState body;float held[4]{},rate[4]{},command[4]{};p::Parameters fluid{};int original_control=0,source_backend=0;};
constexpr int capacity=2048;
struct Output {double value[capacity]{};int count=0;};
CUDARL_HD inline void put(Output& out,float value){out.value[out.count++]=value;}
CUDARL_HD inline void vec(Output& out,p::V value){put(out,value.x);put(out,value.y);put(out,value.z);}
CUDARL_HD inline void add_resistance(Output& out,p::System& system,const p::V* J,p::V axis,float coefficient){
    float projection[10];for(int i=0;i<10;++i){projection[i]=p::dot(J[i],axis);put(out,projection[i]);}
    const float first20=coefficient*projection[2],add20=first20*projection[0];
    const float first02=coefficient*projection[0],add02=first02*projection[2];
    put(out,first20);put(out,add20);put(out,system.R[2][0]);put(out,system.R[2][0]+add20);
    put(out,first02);put(out,add02);put(out,system.R[0][2]);put(out,system.R[0][2]+add02);
    p::resist(system,J,axis,coefficient);
}
CUDARL_HD inline void evaluate(const Input& input,Output& out){
    out={};const auto links=m::shape(input.body);p::System resistance;
    for(int b=0;b<3;++b){const auto& l=links.link[b];const auto g=m::body_geometry(b);
        const float numerator=3*l.volume,denominator=4*m::pi,argument=numerator/denominator,radius=cbrtf(argument);
        const float linear0=6*m::pi,linear1=linear0*input.fluid.viscosity,linear2=linear1*radius;
        const float angular0=8*m::pi,angular1=angular0*input.fluid.viscosity,angular2=angular1*radius,
            angular3=angular2*radius,angular4=angular3*radius;
        const float scalars[]={l.volume,numerator,denominator,argument,radius,linear0,linear1,linear2,angular0,angular1,angular2,angular3,angular4};for(float value:scalars)put(out,value);
        const p::V axis[3]={l.q.rotate({1,0,0}),l.q.rotate({0,1,0}),l.q.rotate({0,0,1})};
        for(auto value:axis)vec(out,value);
        const float projected[3]={b==0?m::pi*g.half_height*g.half_width:m::pi*g.half_width*g.half_width+(b==2?m::Geometry::fin_extension*m::Geometry::fin_thickness:0),
            b==0?m::pi*g.length*.5f*g.half_width:l.length*2*g.half_width,b==0?m::pi*g.length*.5f*g.half_height:l.length*2*g.half_height};
        const float Cd[3]={b==0?.3f:.15f,b==0?.8f:1.f,b==0?.8f:1.f};
        for(float value:projected)put(out,value);for(float value:Cd)put(out,value);
        const float node[3]={-.3872983346207417f,0,.3872983346207417f},weight[3]={5.f/18,4.f/9,5.f/18};
        for(int sample=0;sample<3;++sample){const auto r=axis[0]*(node[sample]*l.length);p::V J[10];vec(out,r);
            for(int i=0;i<10;++i){J[i]=l.linear[i]+p::cross(l.angular[i],r);vec(out,J[i]);}const auto velocity=p::multiply(J,input.body.velocity);vec(out,velocity);
            for(int direction=0;direction<3;++direction){const float v=p::dot(velocity,axis[direction]),pressure0=.5f*input.fluid.density,
                pressure1=pressure0*Cd[direction],pressure2=pressure1*projected[direction],pressure3=pressure2*fabsf(v),
                total=linear2+pressure3,coefficient=weight[sample]*total;
                const float scalars_direction[]={v,pressure0,pressure1,pressure2,pressure3,total,coefficient};for(float value:scalars_direction)put(out,value);
                add_resistance(out,resistance,J,axis[direction],coefficient);}
        }
        for(int direction=0;direction<3;++direction){put(out,angular4);add_resistance(out,resistance,l.angular,axis[direction],angular4);}
    }
    put(out,resistance.R[2][0]);put(out,resistance.R[0][2]);
}
__global__ void trace(const Input* input,Output* output,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)evaluate(input[i],output[i]);}
void require(bool condition,const char* why){if(!condition)throw std::runtime_error(why);}
void labels(std::ostream& out){bool first=true;auto label=[&](const std::string& value){if(!first)out<<',';first=false;out<<'"'<<value<<'"';};
    const char* scalar[]={"volume","radius_numerator","radius_denominator","radius_argument","radius_cbrtf","linear_6pi","linear_viscosity","linear_radius","angular_8pi","angular_viscosity","angular_radius1","angular_radius2","angular_radius3"};
    auto resistance=[&](const std::string& prefix){for(int i=0;i<10;++i)label(prefix+"projection"+std::to_string(i));for(const char* name:{"first20","add20","before20","after20","first02","add02","before02","after02"})label(prefix+name);};
    for(int b=0;b<3;++b){const std::string link="link"+std::to_string(b)+".";for(const char* name:scalar)label(link+name);
        for(int d=0;d<3;++d)for(int k=0;k<3;++k)label(link+"axis"+std::to_string(d)+"."+std::to_string(k));
        for(const char* name:{"projected","Cd"})for(int d=0;d<3;++d)label(link+name+std::to_string(d));
        for(int sample=0;sample<3;++sample){const auto s=link+"sample"+std::to_string(sample)+".";
            for(int k=0;k<3;++k)label(s+"r"+std::to_string(k));for(int i=0;i<10;++i)for(int k=0;k<3;++k)label(s+"J"+std::to_string(i)+"."+std::to_string(k));for(int k=0;k<3;++k)label(s+"velocity"+std::to_string(k));
            for(int d=0;d<3;++d){const auto prefix=s+"direction"+std::to_string(d)+".";for(const char* name:{"velocity_projection","pressure_density","pressure_Cd","pressure_projected","pressure_abs_velocity","total","coefficient"})label(prefix+name);resistance(prefix);}}
        for(int d=0;d<3;++d){const auto prefix=link+"angular"+std::to_string(d)+".";label(prefix+"coefficient");resistance(prefix);}}
    label("final_R20");label("final_R02");}
int run(const fs::path& source,const fs::path& destination){require(!fs::exists(destination),"Never overwrite drag trace evidence");const auto payload=cp::read(source.string());cp::Reader in{payload};
    require(in.value<uint32_t>()==0x32414946&&in.value<uint32_t>()==2,"Wrong input version");const auto count=in.value<uint32_t>();require(count==642&&in.value<uint32_t>()==sizeof(Input),"Wrong fixed input shape");const auto inputs=in.array<Input>(642);in.finish();require(inputs.size()==count,"Wrong input count");
    std::vector<Output> cpu(count),gpu(count);for(size_t i=0;i<inputs.size();++i)evaluate(inputs[i],cpu[i]);device_ptr<Input> d_input(count);device_ptr<Output> d_output(count);d_input.copy_from_host(inputs.data(),count);
    trace<<<(count+63)/64,64>>>(d_input.get(),d_output.get(),static_cast<int>(count));CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());d_output.copy_to_host(gpu.data(),count);
    std::vector<Input> unchanged(count);d_input.copy_to_host(unchanged.data(),count);require(std::memcmp(inputs.data(),unchanged.data(),count*sizeof(Input))==0,"Input mutated");
    require(fs::create_directories(destination),"Cannot create trace directory");cp::Writer output;output.value(uint32_t{0x32444946});output.value(uint32_t{2});output.value(count);output.value(uint32_t{sizeof(Output)});output.value(cp::checksum(payload));output.array(inputs);output.array(cpu);output.array(gpu);cp::write_atomic((destination/"trace.bin").string(),output.data);
    require(cpu[0].count<=capacity,"Trace overflow");for(size_t i=0;i<inputs.size();++i)require(cpu[i].count==cpu[0].count&&gpu[i].count==cpu[0].count,"Trace field count differs");
    std::ofstream layout(destination/"layout.json");layout<<"{\"schema\":1,\"input_bytes\":160,\"output_bytes\":"<<sizeof(Output)<<",\"capacity\":2048,\"values\":"<<cpu[0].count<<",\"labels\":[";labels(layout);layout<<"]}\n";layout.close();require(bool(layout),"Layout write failed");
    std::ofstream summary(destination/"summary.json");summary<<"{\"schema\":1,\"study\":\"frozen_interface_drag_arithmetic_v2\",\"rows\":"<<count<<",\"CPU_zero_time_drag_evaluations\":"<<count<<",\"CUDA_zero_time_drag_evaluations\":"<<count<<",\"explicit_CUDA_kernel_launches\":1,\"native_runs\":1,\"body_inputs_unchanged\":true,\"physical_controls\":0,\"physical_substeps\":0,\"integration_calls\":0,\"PPO_updates\":0,\"optimizer_steps\":0}\n";summary.close();require(bool(summary),"Summary write failed");return 0;
}
}
int main(int argc,char** argv){try{if(argc!=3)throw std::runtime_error("Use INPUT_BIN NEW_OUTPUT_DIRECTORY");return drag_trace_v2::run(std::filesystem::absolute(argv[1]),std::filesystem::absolute(argv[2]));}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
