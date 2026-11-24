// Host-only diagnostic of EXACT extracted collector kernels, never a trainer.
#include "cuda/pond_math_profile.h"
#include "rl/pond_ppo_task.h"
#include "rl/pond_ppo_contract.h"
#include "motor/saved_motor_contract.h"
#include "rl/learned_amplitude_math.h"
#include "nn/mlp_gpu.h"
#include "admission.h"
#include "collector_bench_identity.h"
#include <cuda.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs=std::filesystem;namespace pr=cudarl::pond_runtime;namespace pt=cudarl::rl::pond_ppo_task;
namespace pp=cudarl::rl::pond_ppo;namespace po=cudarl::pond_motor_observation;namespace p=cudarl::physics;
namespace nn=cudarl::nn;namespace cp=cudarl::checkpoint;namespace host=cudarl::rl::pond_ppo_host;
namespace {
constexpr int N=512,D=41,A=4,Capacity=4096;constexpr double wall_cap=120,launch_reserve=10;
constexpr uint64_t evidence_cap=64ull*1024*1024;
using Clock=std::chrono::steady_clock;
void require(bool value,const std::string& reason){if(!value)throw std::runtime_error(reason);}
void rt(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
void drv(CUresult value){if(value!=CUDA_SUCCESS){const char* message=nullptr;cuGetErrorString(value,&message);throw std::runtime_error(message?message:"CUDA driver failure");}}
double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
std::string escape(const std::string& text){std::string result;for(char c:text){if(c=='"'||c=='\\')result+='\\';if(c=='\n')result+="\\n";else if(c=='\r')result+="\\r";else result+=c;}return result;}
struct Frame {pr::World world{};float command[4]{};pr::SubstepTrace trace[20]{};bool advanced=false;};
struct TransitionTrace {float x=0,z=0,vx=0,vy=0,vz=0,depth=0,tilt=0;double minimum_gap=0;
    uint32_t episode=0,control=0,kind=0,occupied_cells=0,occupied_grid[8]{};float bank_near_seconds=0,contact_seconds=0,postwarmup_stall_seconds=0;};
static_assert(sizeof(pr::World)==1384&&sizeof(pt::Task)==184&&sizeof(TransitionTrace)==104);
struct Inputs {std::string protocol;fs::path module,trainer,recording,layout,commands,parent,core;
    std::string observe_symbol,step_symbol;std::array<Frame,8> recorded{};std::vector<float> actor;float gain=0;};
struct Options {bool run=false,preflight=false;fs::path protocol,out;};
Options options(int argc,char** argv){Options out;for(int i=1;i<argc;++i){const std::string arg=argv[i];
    if(arg=="--run")out.run=true;else if(arg=="--preflight")out.preflight=true;else if(arg=="--protocol"&&i+1<argc)out.protocol=argv[++i];
    else if(arg=="--out"&&i+1<argc)out.out=argv[++i];else throw std::runtime_error("Unknown/incomplete argument");}
    require(out.run!=out.preflight&&!out.protocol.empty()&&!out.out.empty(),"Require one --run/--preflight, --protocol and unique --out");return out;}
Inputs load(const Options& o){Inputs in;in.protocol=host::text_file(o.protocol);const auto scalar=[&](const char* key){return host::scalar(in.protocol,key);};
    require(scalar("protocol_id")=="pond-collector-replay-v1-2026-11-23","Wrong nontraining replay protocol");
    require(scalar("worlds_per_cohort")=="512"&&scalar("cohorts")=="2"&&scalar("controls_per_world")=="1"&&scalar("max_controls")=="1024"&&scalar("max_substeps")=="10240"&&scalar("max_wall_seconds")=="120"&&scalar("max_evidence_bytes")=="67108864","Replay allocation changed");
    require(scalar("bench_source_sha256")==collector_bench_sha256&&scalar("trainer_source_identity_fnv1a64")==std::to_string(host::source_identity()),"Source identity differs");
    const auto bound=[&](const char* key){const fs::path path=scalar((std::string(key)+"_path_absolute").c_str());require(path.is_absolute(),"Absolute bound input required");require(host::sha256(host::whole_file(path))==scalar((std::string(key)+"_sha256").c_str()),"Input SHA256 mismatch: "+std::string(key));return path;};
    in.module=bound("module");in.trainer=bound("trainer");in.recording=bound("recording");in.layout=bound("layout");in.commands=bound("commands");in.parent=bound("parent");in.core=bound("native_core");require(scalar("native_core_sha256")==collector_bench_native_core_sha256,"Linked native MLP library differs from protocol");
    in.observe_symbol=scalar("observe_symbol");in.step_symbol=scalar("step_symbol");require(in.observe_symbol.find("observe_kernel")!=std::string::npos&&in.step_symbol.find("step_kernel")!=std::string::npos,"Wrong extracted collector symbols");
    const auto layout=host::text_file(in.layout);require(host::scalar(layout,"frame_bytes")==std::to_string(sizeof(Frame))&&host::scalar(layout,"world_bytes")==std::to_string(sizeof(pr::World))&&host::scalar(layout,"frame_world_offset")=="0","Recorded ABI differs");
    const auto bytes=cp::read(in.recording.string());cp::Reader reader{bytes};require(reader.value<uint32_t>()==0x32474350&&reader.value<uint32_t>()==1&&reader.value<int>()==1&&reader.value<uint32_t>()==sizeof(Frame)&&reader.value<uint32_t>()==8,"Require exact recorded post-control1 fixtures");
    for(int i=0;i<8;++i)(void)reader.value<Frame>();for(auto& frame:in.recorded)frame=reader.value<Frame>();reader.finish();
    for(const auto& frame:in.recorded)require(frame.advanced&&frame.world.failure==0&&frame.world.motor.physical.control_steps==1&&frame.world.motor.attempted_substeps==10,"Recorded world is not accepted control1 state");
    std::ifstream commands(in.commands);std::string line;std::getline(commands,line);uint32_t mask=0;
    while(std::getline(commands,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);int control=0,id=0;float value[4]{};row>>control>>id>>value[0]>>value[1]>>value[2]>>value[3];
        if(control==2&&id>=4&&id<8){require(bool(row)&&value[0]==0&&value[1]==0&&value[2]==0&&value[3]==0,"Recorded common command is not exact zero");mask|=1u<<(id-4);}}
    require(mask==15,"Missing four recorded zero command rows");require(host::sha256(host::whole_file(in.parent))==pp::parent_package_sha256,"Original saved motor changed");
    const auto payload=cp::read(in.parent.string());cp::Reader model{payload};for(uint32_t expected:{0x31564c48u,2u,6u,6u,3u,4u,41u,4u,336u,1u,7u,1u})require(model.value<uint32_t>()==expected,"Original package contract changed");
    require(model.value<uint64_t>()!=0,"Missing original protocol");const auto parent=model.array<uint8_t>(64);require(std::string(parent.begin(),parent.end())=="810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0","Wrong original parent");
    const auto& hashes=cudarl::motor::saved_motor_source_contract_hashes();require(model.value<uint32_t>()==hashes.size(),"Wrong original source count");for(const auto expected:hashes){const auto hash=model.array<uint8_t>(64);require(std::string(hash.begin(),hash.end())==expected,"Wrong original provenance");}
    in.actor=model.array<float>(2532);const auto head=model.array<float>(2);const float gain=model.value<float>(),sigma=model.value<float>();model.finish();require(in.actor.size()==2532&&head.size()==2,"Wrong original weights");in.gain=cudarl::learned_amplitude::sigmoid(head[0]);
    require(in.gain==pp::frozen_holding_gain&&std::abs(gain-in.gain)<=2e-7f&&std::abs(sigma-cudarl::learned_amplitude::standard_deviation(head[1]))<=2e-6f,"Original gain/sigma changed");for(float value:in.actor)require(std::isfinite(value),"Nonfinite original weight");return in;}
struct HostState {std::vector<pr::World> worlds{N};std::vector<pt::Task> tasks{N};std::vector<float> latent=std::vector<float>(N*A,0.f);};
HostState cohort(const Inputs& in,int kind){HostState out;for(int i=0;i<N;++i){const int id=kind*4+i%4;const auto& frame=in.recorded[static_cast<size_t>(id)];out.worlds[static_cast<size_t>(i)]=frame.world;auto& task=out.tasks[static_cast<size_t>(i)];
    task.kind=pt::Straight;task.episode_controls=1;task.request.controls=1;task.request.start_com_model=po::motion(frame.trace[0].before).com_model;
    auto heading=frame.trace[0].before.orientation.rotate({1,0,0});heading=heading/p::norm(heading);task.request.request={task.request.start_com_model+heading*(.072f/p::scale),heading*.008f,heading};}
    return out;}
struct Stream {cudaStream_t value=nullptr;Stream(){rt(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}~Stream(){if(value)cudaStreamDestroy(value);}};
struct Events {cudaEvent_t first=nullptr,last=nullptr;Events(){rt(cudaEventCreate(&first));rt(cudaEventCreate(&last));}~Events(){if(first)cudaEventDestroy(first);if(last)cudaEventDestroy(last);}};
struct Span {bool confirmed=false;double host_ms=0,enqueue_host_ms=0,wait_host_ms=0;float event_span_ms=0;};
template<class Function> Span stage(Events& events,cudaStream_t stream,Function function){Span result;const auto start=Clock::now();rt(cudaEventRecord(events.first,stream));function();rt(cudaEventRecord(events.last,stream));result.enqueue_host_ms=ms(start);const auto waiting=Clock::now();rt(cudaEventSynchronize(events.last));result.wait_host_ms=ms(waiting);rt(cudaEventElapsedTime(&result.event_span_ms,events.first,events.last));result.host_ms=ms(start);result.confirmed=true;return result;}
template<class T> struct Device {T* value=nullptr;explicit Device(size_t count){rt(cudaMalloc(reinterpret_cast<void**>(&value),count*sizeof(T)));}~Device(){if(value)cudaFree(value);}Device(const Device&)=delete;Device& operator=(const Device&)=delete;};
struct Buffers {Device<pr::World> worlds{N};Device<pt::Task> tasks{N};Device<float> latent{N*A},observation{N*D},final_observation{N*D},reward{N},done{N},truncated{N};Device<uint64_t> attempted{N},substeps{N};Device<int> reason{N},fault{1};Device<TransitionTrace> trace{N};};
struct Row {int kind=0;bool scheduled=false,readback=false;uint64_t attempted=0,substeps=0;int fault=INT_MAX;
    Span upload,observe,fault_readback,inference,physics,readback_span;double fixture_copy_ms=0,payload_ms=0,checksum_write_ms=0;std::vector<uint64_t> attempts,steps;std::vector<int> reasons;};
void span_json(std::ostream& out,const Span& span){out<<"{\"confirmed\":"<<(span.confirmed?"true":"false");if(span.confirmed)out<<",\"event_span_ms\":"<<span.event_span_ms<<",\"host_wall_ms\":"<<span.host_ms<<",\"enqueue_host_ms\":"<<span.enqueue_host_ms<<",\"wait_host_ms\":"<<span.wait_host_ms;else out<<",\"event_span_ms\":null,\"host_wall_ms\":null,\"enqueue_host_ms\":null,\"wait_host_ms\":null";out<<'}';}
struct Module {CUmodule value=nullptr;explicit Module(const fs::path& path){drv(cuModuleLoad(&value,path.string().c_str()));}~Module(){if(value)cuModuleUnload(value);}CUfunction function(const std::string& name){CUfunction result=nullptr;drv(cuModuleGetFunction(&result,value,name.c_str()));return result;}};
struct Budget {Clock::time_point start=Clock::now();uint64_t evidence=0;double seconds()const{return ms(start)/1000.;}void before_launch()const{require(seconds()<wall_cap-launch_reserve,"Wall reserve stops new launches");}void charge(uint64_t bytes){require(bytes<=evidence_cap-evidence,"Replay evidence cap");evidence+=bytes;}};
void write_binary(const fs::path& path,const std::vector<uint8_t>& bytes,Budget& budget){budget.charge(bytes.size()+24);require(!fs::exists(path),"Evidence already exists");cp::write_atomic(path.string(),bytes);}
int run(const Options& options,const Inputs& input){Budget budget;require(!fs::exists(options.out)&&fs::create_directories(options.out),"Unique benchmark output required");
    const fs::path ledger=host::scalar(input.protocol,"run_ledger_path_absolute");require(ledger.is_absolute(),"Absolute one-attempt ledger required");cp::Lease lease(ledger.string()+".lease");require(!fs::exists(ledger),"Replay allocation already attempted; no repeat");
    cp::Writer began;began.value(uint32_t{0x31424350});began.value(uint32_t{1});began.value(cp::checksum(host::whole_file(options.protocol)));began.value(uint32_t{1});write_binary(ledger,began.data,budget);
    std::vector<Row> rows;std::string failure;double initialization_ms=0;int actor_calls=0,critic_calls=0,step_registers=0,step_local_bytes=0;
    const auto init=Clock::now();try{drv(cuInit(0));rt(cudaSetDevice(0));Stream stream;Events events;Buffers buffers;Module module(input.module);const auto observe=module.function(input.observe_symbol),step=module.function(input.step_symbol);
        drv(cuFuncGetAttribute(&step_registers,CU_FUNC_ATTRIBUTE_NUM_REGS,step));drv(cuFuncGetAttribute(&step_local_bytes,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,step));
        nn::GpuMlp actor({{41,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}},Capacity,0x504f4e44u),critic({{41,32,32,1},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}},Capacity,0x504f4e45u);actor.set_params(input.actor);rt(cudaDeviceSynchronize());initialization_ms=ms(init);
        for(int kind=0;kind<2;++kind){budget.before_launch();rows.emplace_back();auto& row=rows.back();row.kind=kind;const auto copied=Clock::now();auto state=cohort(input,kind);row.fixture_copy_ms=ms(copied);const int good=INT_MAX;
            row.upload=stage(events,stream.value,[&]{rt(cudaMemcpyAsync(buffers.worlds.value,state.worlds.data(),N*sizeof(pr::World),cudaMemcpyHostToDevice,stream.value));rt(cudaMemcpyAsync(buffers.tasks.value,state.tasks.data(),N*sizeof(pt::Task),cudaMemcpyHostToDevice,stream.value));rt(cudaMemcpyAsync(buffers.latent.value,state.latent.data(),N*A*sizeof(float),cudaMemcpyHostToDevice,stream.value));
                rt(cudaMemcpyAsync(buffers.fault.value,&good,sizeof(good),cudaMemcpyHostToDevice,stream.value));for(auto* value:{buffers.reward.value,buffers.done.value,buffers.truncated.value})rt(cudaMemsetAsync(value,0,N*sizeof(float),stream.value));rt(cudaMemsetAsync(buffers.attempted.value,0,N*sizeof(uint64_t),stream.value));rt(cudaMemsetAsync(buffers.substeps.value,0,N*sizeof(uint64_t),stream.value));rt(cudaMemsetAsync(buffers.reason.value,0,N*sizeof(int),stream.value));});
            row.observe=stage(events,stream.value,[&]{void* args[]={&buffers.worlds.value,&buffers.tasks.value,&buffers.observation.value,&buffers.fault.value};drv(cuLaunchKernel(observe,16,1,1,32,1,1,0,reinterpret_cast<CUstream>(stream.value),args,nullptr));});
            row.fault_readback=stage(events,stream.value,[&]{rt(cudaMemcpyAsync(&row.fault,buffers.fault.value,sizeof(int),cudaMemcpyDeviceToHost,stream.value));});require(row.fault==INT_MAX,"Recorded observation failed before physical launch");
            row.inference=stage(events,stream.value,[&]{actor.forward(buffers.observation.value,N,stream.value);++actor_calls;critic.forward(buffers.observation.value,N,stream.value);++critic_calls;});
            budget.before_launch();row.scheduled=true;auto& saved=row;float gain=input.gain;
            saved.physics=stage(events,stream.value,[&]{void* args[]={&buffers.worlds.value,&buffers.tasks.value,&buffers.latent.value,&gain,&buffers.reward.value,&buffers.done.value,&buffers.truncated.value,&buffers.final_observation.value,&buffers.attempted.value,&buffers.substeps.value,&buffers.reason.value,&buffers.trace.value,&buffers.fault.value};drv(cuLaunchKernel(step,16,1,1,32,1,1,0,reinterpret_cast<CUstream>(stream.value),args,nullptr));});
            std::vector<TransitionTrace> trace(N);std::vector<float> rewards(N);saved.attempts.resize(N);saved.steps.resize(N);saved.reasons.resize(N);
            saved.readback_span=stage(events,stream.value,[&]{rt(cudaMemcpyAsync(state.worlds.data(),buffers.worlds.value,N*sizeof(pr::World),cudaMemcpyDeviceToHost,stream.value));rt(cudaMemcpyAsync(state.tasks.data(),buffers.tasks.value,N*sizeof(pt::Task),cudaMemcpyDeviceToHost,stream.value));rt(cudaMemcpyAsync(saved.attempts.data(),buffers.attempted.value,N*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream.value));rt(cudaMemcpyAsync(saved.steps.data(),buffers.substeps.value,N*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream.value));rt(cudaMemcpyAsync(saved.reasons.data(),buffers.reason.value,N*sizeof(int),cudaMemcpyDeviceToHost,stream.value));rt(cudaMemcpyAsync(trace.data(),buffers.trace.value,N*sizeof(TransitionTrace),cudaMemcpyDeviceToHost,stream.value));rt(cudaMemcpyAsync(rewards.data(),buffers.reward.value,N*sizeof(float),cudaMemcpyDeviceToHost,stream.value));rt(cudaMemcpyAsync(&saved.fault,buffers.fault.value,sizeof(int),cudaMemcpyDeviceToHost,stream.value));});
            saved.readback=true;for(int i=0;i<N;++i){saved.attempted+=saved.attempts[static_cast<size_t>(i)];saved.substeps+=saved.steps[static_cast<size_t>(i)];}
            const auto payload_start=Clock::now();cp::Writer output;output.value(uint32_t{0x31524350});output.value(uint32_t{1});output.value(kind);output.value(uint32_t{sizeof(pr::World)});output.value(uint32_t{sizeof(pt::Task)});output.value(uint32_t{sizeof(TransitionTrace)});for(const auto& world:state.worlds)output.value(world);for(const auto& task:state.tasks)output.value(task);output.array(saved.attempts);output.array(saved.steps);output.array(saved.reasons);output.array(rewards);for(const auto& value:trace)output.value(value);saved.payload_ms=ms(payload_start);
            const auto write_start=Clock::now();write_binary(options.out/(kind==0?"flat-after.bin":"bank-after.bin"),output.data,budget);saved.checksum_write_ms=ms(write_start);
            require(saved.fault==INT_MAX&&saved.attempted==512&&saved.substeps==5120,"Physical replay failed; stop without repeating");for(int i=0;i<N;++i)require(saved.attempts[static_cast<size_t>(i)]==1&&saved.steps[static_cast<size_t>(i)]==10&&state.worlds[static_cast<size_t>(i)].motor.physical.control_steps==2,"Physical/control counter mismatch");
            if(budget.seconds()>=wall_cap)throw std::runtime_error("Wall cap reached including in-flight completion");
        }
    }catch(const std::exception& error){failure=error.what();if(initialization_ms==0)initialization_ms=ms(init);}
    uint64_t attempted=0,substeps=0,upper=0;bool known=true;for(const auto& row:rows){attempted+=row.attempted;substeps+=row.substeps;upper+=row.scheduled?512:0;known=known&&(!row.scheduled||row.readback);}
    std::ostringstream json;json<<std::setprecision(17)<<"{\"protocol_id\":\"pond-collector-replay-v1-2026-11-23\",\"benchmark_complete\":"<<(failure.empty()&&rows.size()==2?"true":"false")<<",\"reason\":\""<<escape(failure)<<"\",\"bench_source_sha256\":\""<<collector_bench_sha256<<"\",\"trainer_source_identity_fnv1a64\":\""<<host::source_identity()<<"\",\"physical_accounting_complete\":"<<(known?"true":"false")<<",\"attempted_controls_confirmed\":"<<attempted<<",\"attempted_substeps_confirmed\":"<<substeps<<",\"attempted_controls_upper_bound\":"<<upper<<",\"attempted_substeps_upper_bound\":"<<upper*10<<",\"actual_wall_seconds\":"<<budget.seconds()<<",\"wall_cap_seconds\":120,\"wall_cap_overrun_seconds\":"<<std::max(0.,budget.seconds()-wall_cap)<<",\"initialization_host_ms\":"<<initialization_ms<<",\"actor_forward_calls\":"<<actor_calls<<",\"critic_forward_calls\":"<<critic_calls<<",\"step_registers_per_thread\":"<<step_registers<<",\"step_local_bytes_per_thread\":"<<step_local_bytes<<",\"optimizer_calls\":0,\"candidate_checkpoint_created\":false,\"training_admission\":false,\"on_policy_rollout\":false,\"inference_drives_commands\":false,\"integration_warmups\":0,\"repeats\":0,\"timings_overlap_do_not_add\":true,\"cohorts\":[";
    for(size_t i=0;i<rows.size();++i){if(i)json<<',';const auto& row=rows[i];json<<"{\"kind\":\""<<(row.kind==0?"flat_recorded":"bank_recorded")<<"\",\"attempted\":"<<row.attempted<<",\"substeps\":"<<row.substeps<<",\"readback_complete\":"<<(row.readback?"true":"false")<<",\"fixture_copy_host_ms\":"<<row.fixture_copy_ms<<",\"H2D_and_clear\":";span_json(json,row.upload);json<<",\"observe\":";span_json(json,row.observe);json<<",\"fault_D2H\":";span_json(json,row.fault_readback);json<<",\"actor_critic_forward\":";span_json(json,row.inference);json<<",\"physical_step\":";span_json(json,row.physics);json<<",\"D2H\":";span_json(json,row.readback_span);json<<",\"payload_build_host_ms\":"<<row.payload_ms<<",\"checksum_atomic_write_host_ms\":"<<row.checksum_write_ms<<'}';}
    json<<"]}\n";const auto text=json.str();budget.charge(text.size());std::ofstream result(options.out/"result.json");result<<text;result.flush();require(bool(result),"Result write failed");return failure.empty()?0:2;}
}
int main(int argc,char** argv){try{const auto options=::options(argc,argv);const auto input=load(options);if(options.preflight){require(!fs::exists(options.out)&&fs::create_directories(options.out),"Unique preflight output required");const auto flat=cohort(input,0),bank=cohort(input,1);(void)flat;(void)bank;std::ofstream result(options.out/"preflight.json");result<<"{\"passed\":true,\"bench_source_sha256\":\""<<collector_bench_sha256<<"\",\"world_stride\":"<<sizeof(pr::World)<<",\"task_stride\":"<<sizeof(pt::Task)<<",\"trace_stride\":"<<sizeof(TransitionTrace)<<",\"prepared_recorded_copies\":1024,\"physical_controls\":0,\"optimizer_calls\":0,\"CUDA_calls\":0}\n";return 0;}return run(options,input);}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
