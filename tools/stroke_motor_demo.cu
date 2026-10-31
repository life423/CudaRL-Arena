// One fixed, bounded native pilot. Not a general trainer/session framework.
#include "motor/frozen_controller.h"
#include "nn/mlp_cpu.h"
#include "nn/mlp_gpu.h"
#include "nn/adam.h"
#include "rl/gaussian_kernels.h"
#include "nn/loss.h"
#include "motor/demonstration_data.h"
#include "core/checkpoint.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <sstream>
#include <map>

namespace {
namespace fs=std::filesystem;namespace p=cudarl::physics;namespace s=cudarl::simulation;
namespace m=cudarl::straight_motor;namespace nn=cudarl::nn;namespace rl=cudarl::rl;namespace cp=cudarl::checkpoint;
constexpr int minibatch=256;
int fit_steps=600;float wall_limit=20;
fs::path warm_start,reference_csv,example_file;
constexpr float train_speeds[]={.004f,.006f,.008f};
constexpr float test_speeds[]={.004f,.005f,.006f,.007f,.008f};
constexpr int evaluation_controls=1600;
uint32_t seed=0x53574D32u;
namespace demo=cudarl::demonstration;
struct Stream {cudaStream_t value=nullptr;Stream(){CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}~Stream(){if(value){cudaStreamSynchronize(value);cudaStreamDestroy(value);}}};
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void finite(const std::vector<float>& values){for(float x:values)require(std::isfinite(x),"Nonfinite numerical buffer; stop before further work");}
std::vector<float> download(const float* pointer,size_t count,cudaStream_t stream){std::vector<float> out(count);CUDA_CHECK_THROW(cudaMemcpyAsync(out.data(),pointer,count*sizeof(float),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));finite(out);return out;}
void upload(float* pointer,const std::vector<float>& values,cudaStream_t stream){finite(values);CUDA_CHECK_THROW(cudaMemcpyAsync(pointer,values.data(),values.size()*sizeof(float),cudaMemcpyHostToDevice,stream));}
nn::MlpSpec specification(int outputs){return {{m::kObservations,32,32,outputs},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}};}
m::World initial(int id){return m::initial_world(id);}
m::ForwardConfig task_config(float speed=.008f){return m::config_for_speed(speed);}
std::vector<uint8_t> binding(){cp::Writer out;out.value(uint32_t{0x53574D32});out.value(uint32_t{7});out.value(demo::kVersion);out.value(seed);out.value(m::kTaskVersion);out.value(m::kRewardVersion);out.value(cudarl::movement::kVersion);
    for(int x:{m::kActions,m::kObservations,fit_steps,minibatch,32,demo::kAcceptedRows,evaluation_controls,1632,16320,80,40})out.value(x);
    for(float x:{.001f,.00001f,.5f,demo::kFeedforward,demo::kFeedback,demo::kFrequency,demo::kMinimumAmplitude,demo::kMaximumAmplitude})out.value(x);
    for(float speed:train_speeds)out.value(speed);for(float speed:test_speeds)out.value(speed);out.value(cp::checksum(cp::read(warm_start.string())));out.value(cp::checksum(cp::read(example_file.string())));
    const auto c=task_config();for(float x:{c.clearance_m,c.speed_rmse_limit,c.lateral_limit_m,c.depth_limit_m,c.path_angle_limit_deg,c.tilt_limit_deg,c.work_per_distance_limit,wall_limit})out.value(x);
    for(int id=0;id<4;++id){const auto w=initial(id).physical;for(float x:{w.body.position.x,w.body.position.y,w.body.position.z,w.body.orientation.x,w.body.orientation.y,w.body.orientation.z,w.body.orientation.w,w.fluid.density,w.fluid.viscosity,w.fluid.gravity})out.value(x);for(float x:w.body.angle)out.value(x);for(float x:w.body.velocity)out.value(x);}
    return out.data;}
void status(const fs::path& root,const char* phase,int completed_updates=0,int controls=0,int physical_substeps=0){std::ofstream out(root/"status.json");out<<"{\"phase\":"<<std::quoted(phase)<<",\"task\":5,\"reward_version\":2,\"model\":6,\"supervised_steps_cap\":"<<fit_steps<<",\"completed_updates\":"<<completed_updates<<",\"controls\":"<<controls<<",\"physical_substeps\":"<<physical_substeps<<",\"control_cap\":"<<1632<<",\"wall_seconds_cap\":"<<wall_limit<<"}\n";out.flush();}
void prepare(const fs::path& root){fs::create_directories(root);require(!fs::exists(root/"protocol.bin")&&!fs::exists(root/"run-started"),"Protocol already prepared/attempted");cp::write_atomic((root/"protocol.bin").string(),binding());
 (void)demo::accepted_examples(cp::read(example_file.string()));
 std::ofstream out(root/"protocol.json");out<<"{\"task\":5,\"model\":6,\"algorithm\":\"constant-speed behavior cloning from accepted recorded profiles\",\"source_teacher_trial_accepted\":false,\"accepted_profiles\":[0,1,2,3,4,5],\"teacher_rerun\":false,\"switch_evaluated\":false,\"previous_switch_gate_remains_unpassed\":true,\"supervised_steps_cap\":"<<fit_steps<<",\"batch_size\":256,\"accepted_examples\":960,\"controls_max\":1632,\"native_substeps_max\":16320,\"native_wall_seconds_max\":"<<wall_limit<<",\"retries\":0,\"train_targets_mm_s\":[4,6,8],\"withheld_targets_mm_s\":[5,7],\"frozen_evaluation\":\"unchanged4s cases0/3, five constant requested speeds from rest\"}\n";status(root,"registered");}

auto wall_start=std::chrono::steady_clock::now();
void wall_guard(){require(std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count()<wall_limit,"Native 20s wall cap reached");}
struct Budget {int charged[2]{},actual_substeps=0;void reserve(int stage,int n){wall_guard();const int cap[2]{32,evaluation_controls};require(n>0&&charged[stage]+n<=cap[stage],"Stage control budget exhausted");charged[stage]+=n;}int total()const{return charged[0]+charged[1];}};
struct Command {float value[4]{};};
__global__ void step_kernel(m::World* states,const Command* commands,int n){const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&!m::step_state(states[i],commands[i].value))states[i].physical.invalid=true;}
void advance(std::vector<m::World>& state,const std::vector<std::array<float,4>>& command,int stage,bool gpu,cudaStream_t stream,Budget& budget){
    budget.reserve(stage,static_cast<int>(state.size()));std::vector<uint64_t> before;for(const auto& w:state)before.push_back(w.attempted_substeps);
    if(gpu){device_ptr<m::World> worlds_device(state.size());device_ptr<Command> command_device(command.size());std::vector<Command> native_command(command.size());for(size_t i=0;i<command.size();++i)for(int j=0;j<4;++j)native_command[i].value[j]=command[i][j];
        CUDA_CHECK_THROW(cudaMemcpyAsync(worlds_device.get(),state.data(),state.size()*sizeof(m::World),cudaMemcpyHostToDevice,stream));CUDA_CHECK_THROW(cudaMemcpyAsync(command_device.get(),native_command.data(),native_command.size()*sizeof(Command),cudaMemcpyHostToDevice,stream));
        step_kernel<<<1,32,0,stream>>>(worlds_device.get(),command_device.get(),static_cast<int>(state.size()));CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaMemcpyAsync(state.data(),worlds_device.get(),state.size()*sizeof(m::World),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    }else for(size_t i=0;i<state.size();++i)if(!m::step_state(state[i],command[i].data()))state[i].physical.invalid=true;
    for(size_t i=0;i<state.size();++i)budget.actual_substeps+=static_cast<int>(state[i].attempted_substeps-before[i]);
    for(size_t i=0;i<state.size();++i){const auto& w=state[i].physical;if(w.invalid)throw std::runtime_error("Movement failure code "+std::to_string(state[i].failure_code)+" at world "+std::to_string(i));require(state[i].attempted_substeps-before[i]==10&&w.telemetry.executed_substeps==10,"Incomplete movement interval");
        for(float x:w.actual_torques)require(std::isfinite(x)&&std::fabs(x)<=1,"Invalid applied torque");require(m::in_domain(state[i],task_config()),"Physical/domain failure; reject before supervised fitting");}
}
std::vector<float> fields(const m::World& input){const auto& w=input.physical;std::vector<float> out{w.body.position.x,w.body.position.y,w.body.position.z,w.body.orientation.x,w.body.orientation.y,w.body.orientation.z,w.body.orientation.w,w.body.work,w.body.travel,w.mean_squared_torque};
    for(float x:w.body.angle)out.push_back(x);for(float x:w.body.velocity)out.push_back(x);for(float x:w.last_actual_torques)out.push_back(x);for(float x:w.held_targets)out.push_back(x);for(float x:w.actual_torques)out.push_back(x);
    for(float x:w.requested_torques)out.push_back(x);for(float x:w.raw_action)out.push_back(x);
    const auto& t=w.telemetry;const auto& i=t.inspection;
    for(float x:{w.current.x,w.current.y,w.current.z,w.fluid.density,w.fluid.viscosity,w.fluid.gravity,i.floor_gap,i.min_other_gap,i.min_gap,i.max_floor_gap,t.minimum_floor_gap,t.minimum_other_gap,t.floor_spring_force_max,t.total_spring_force_max})out.push_back(x);
    for(float x:w.gains.kp)out.push_back(x);for(float x:w.gains.kd)out.push_back(x);out.push_back(input.phase);for(float x:input.parameter)out.push_back(x);for(float x:input.rate)out.push_back(x);return out;}
void readiness(Stream& stream,Budget& budget){std::vector<m::World> cpu,gpu;for(int i=0;i<4;++i)cpu.push_back(initial(i));gpu=cpu;
    for(int tick=0;tick<4;++tick){std::vector<std::array<float,4>> commands(4);for(auto& command:commands)command={tick%2?.1f:-.1f,0,tick%2?-.1f:.1f,0};
        advance(cpu,commands,0,false,stream.value,budget);advance(gpu,commands,0,true,stream.value,budget);
        for(int i=0;i<4;++i){const auto a=fields(cpu[i]),b=fields(gpu[i]);finite(a);finite(b);for(size_t j=0;j<a.size();++j)require(std::fabs(a[j]-b[j])<=2e-4f+2e-3f*std::fabs(a[j]),"Readiness CPU/CUDA physical mismatch");
            for(const auto pair:{std::array<float,2>{cpu[i].physical.telemetry.minimum_floor_gap,gpu[i].physical.telemetry.minimum_floor_gap},std::array<float,2>{cpu[i].physical.telemetry.minimum_other_gap,gpu[i].physical.telemetry.minimum_other_gap}})
                require(std::fabs(pair[0]-pair[1])<=2e-6f+2e-3f*std::fabs(pair[0]),"Readiness CPU/CUDA clearance mismatch");
            require(cpu[i].physical.substeps==gpu[i].physical.substeps&&cpu[i].physical.control_steps==gpu[i].physical.control_steps&&cpu[i].physical.time_seconds==gpu[i].physical.time_seconds&&cpu[i].physical.unsafe==gpu[i].physical.unsafe&&cpu[i].physical.invalid==gpu[i].physical.invalid,"Readiness CPU/CUDA flags/count mismatch");}
    }
}
struct Evaluation {int successes=0;float mean_forward=0,mean_return=0,mean_work=0,mean_speed=0,mean_speed_rmse=0,max_lateral=0,max_depth=0,max_head_yaw=0,max_tilt=0,max_path_angle=0;};
struct Report {Evaluation speed[5]{};int successes=0;float normalized_rmse=0,mean_return=0;};
void json_eval(std::ostream& out,const Evaluation& e,float target){out<<"{\"target_mm_s\":"<<target*1000<<",\"passed\":"<<e.successes<<",\"speed_mm_s\":"<<e.mean_speed*1000<<",\"RMSE_mm_s\":"<<e.mean_speed_rmse*1000<<",\"work_J\":"<<e.mean_work<<",\"work_J_per_m\":"<<e.mean_work/fmaxf(e.mean_forward,1e-9f)<<",\"lateral_mm\":"<<e.max_lateral*1000<<",\"depth_mm\":"<<e.max_depth*1000<<",\"path_angle_deg\":"<<e.max_path_angle<<",\"head_yaw_deg\":"<<e.max_head_yaw<<",\"tilt_deg\":"<<e.max_tilt<<",\"return\":"<<e.mean_return<<"}";}
Report evaluate(const std::vector<float>& parameters,const fs::path& root,const char* name,Budget& budget,Stream& stream){
    nn::CpuMlp<float> actor(specification(2));actor.params()=parameters;Report report;
    std::ofstream trace(root/(std::string(name)+".csv"));trace.exceptions(std::ios::badbit|std::ios::failbit);
    trace<<std::setprecision(9)<<"target_mm_s,scenario,segment,case,tick,forward_m,depth_m,lateral_m,speed_error_m_s,reward,success,head_x,head_y,head_z,qx,qy,qz,qw,j0,j1,j2,j3,phase,amplitude,frequency,positive_work_J\n";
    auto segment=[&](std::vector<m::World>& state,m::ForwardTask& task,const m::ForwardConfig& config,int id,int group,const char* scenario){
        Evaluation e;m::Transition transition;const float work_start=state[0].physical.body.work;
        for(int tick=0;tick<80;++tick){float obs[m::kObservations]{};require(m::observe(task,state[0],config,obs),"Invalid evaluation observation");actor.forward(obs,1);m::Action action;require(m::action_from_latent(actor.output(),action),"Invalid deterministic action");
            std::vector<std::array<float,4>> command(1);for(int j=0;j<4;++j)command[0][j]=action.command[j];advance(state,command,1,false,stream.value,budget);transition=m::record_control(task,state[0],config);require(transition.accepted,"Rejected evaluation sample");
            const auto& b=state[0].physical.body;trace<<config.target_speed_m_s*1000<<','<<scenario<<','<<group<<','<<id<<','<<tick+1<<','<<transition.metrics.forward_m<<','<<transition.metrics.depth_m<<','<<transition.metrics.lateral_m<<','<<transition.metrics.speed_error_m_s<<','<<transition.reward<<','<<(transition.outcome==m::Outcome::Success)<<','<<b.position.x<<','<<b.position.y<<','<<b.position.z<<','<<b.orientation.x<<','<<b.orientation.y<<','<<b.orientation.z<<','<<b.orientation.w;for(float x:b.angle)trace<<','<<x;trace<<','<<state[0].phase;for(float x:state[0].parameter)trace<<','<<x;trace<<','<<b.work<<'\n';
            e.mean_return+=transition.reward;
        }
        e.successes=task.outcome==m::Outcome::Success?1:0;e.mean_forward=transition.metrics.forward_m;e.mean_work=state[0].physical.body.work-work_start;e.mean_speed=task.steady_speed_m_s;e.mean_speed_rmse=task.speed_rmse;e.max_lateral=task.maximum_lateral;e.max_depth=task.maximum_depth;e.max_head_yaw=task.maximum_head_yaw;e.max_tilt=task.maximum_tilt;e.max_path_angle=task.path_angle_deg;return e;
    };
    for(int speed=0;speed<5;++speed){auto& total=report.speed[speed];for(int id:{0,3}){std::vector<m::World> state{initial(id)};auto config=task_config(test_speeds[speed]);m::ForwardTask task;require(m::begin_forward(state[0],config,task),"Invalid evaluation start");const auto e=segment(state,task,config,id,0,"constant");
        total.successes+=e.successes;total.mean_forward+=e.mean_forward*.5f;total.mean_return+=e.mean_return*.5f;total.mean_work+=e.mean_work*.5f;total.mean_speed+=e.mean_speed*.5f;total.mean_speed_rmse+=e.mean_speed_rmse*.5f;total.max_lateral=fmaxf(total.max_lateral,e.max_lateral);total.max_depth=fmaxf(total.max_depth,e.max_depth);total.max_head_yaw=fmaxf(total.max_head_yaw,e.max_head_yaw);total.max_tilt=fmaxf(total.max_tilt,e.max_tilt);total.max_path_angle=fmaxf(total.max_path_angle,e.max_path_angle);
    }report.successes+=total.successes;report.normalized_rmse+=total.mean_speed_rmse/test_speeds[speed]/5;report.mean_return+=total.mean_return/5;}
    return report;
}
std::vector<std::string> csv_fields(const std::string& line){std::stringstream in(line);std::vector<std::string> out;std::string value;while(std::getline(in,value,','))out.push_back(value);return out;}
void verify_baseline(const fs::path& actual,const fs::path& expected){
    using Row=std::map<std::string,std::string>;std::map<std::pair<int,int>,Row> reference;
    auto read=[&](const fs::path& file,auto accept){std::ifstream in(file);require(bool(in),"Missing frozen refactor reference");std::string line;require(bool(std::getline(in,line)),"Empty reference CSV");const auto header=csv_fields(line);
        while(std::getline(in,line)){const auto values=csv_fields(line);require(values.size()==header.size(),"Malformed reference CSV");Row row;for(size_t i=0;i<header.size();++i)row[header[i]]=values[i];accept(row);}};
    read(expected,[&](const Row& row){reference[{std::stoi(row.at("case")),std::stoi(row.at("tick"))}]=row;});int rows_checked=0;float maximum=0;
    read(actual,[&](const Row& row){if(row.at("scenario")!="constant"||std::stof(row.at("target_mm_s"))!=8)return;
        const auto& ref=reference.at({std::stoi(row.at("case")),std::stoi(row.at("tick"))});
        for(const char* field:{"head_x","head_y","head_z","qx","qy","qz","qw","j0","j1","j2","j3","phase","amplitude","frequency","positive_work_J","forward_m","lateral_m","depth_m","speed_error_m_s"}){
            const float a=std::stof(row.at(field)),b=std::stof(ref.at(field));const float score=std::fabs(a-b)/(1e-5f+1e-5f*std::fabs(b));require(std::isfinite(score)&&score<=1,"Refactor changed the saved8mm/s trajectory; stop before supervised fitting");maximum=std::fmax(maximum,score);}
        ++rows_checked;});require(rows_checked==160,"Incomplete saved8mm/s reference coverage");std::cout<<"REFACTOR_REFERENCE_PASSED rows="<<rows_checked<<" max_score="<<maximum<<std::endl;
}
void snapshot(const fs::path& root,int step,nn::GpuMlp& actor,nn::GpuAdam& optimizer){
    cp::Writer out;out.value(uint32_t{0x46575331});out.value(uint32_t{2});out.value(uint32_t{sizeof(m::World)});out.value(uint32_t{sizeof(m::ForwardTask)});out.value(step);out.array(binding());out.array(actor.get_params());optimizer.save(out);
    const auto path=root/("update-"+std::to_string(step)+".bin");require(!fs::exists(path),"Snapshot exists");cp::write_atomic(path.string(),out.data);require(cp::read(path.string())==out.data,"Snapshot roundtrip failed");
}
int run(const fs::path& root){require(cp::read((root/"protocol.bin").string())==binding(),"Frozen numeric protocol differs");require(fs::create_directory(root/"run-started"),"Trial already attempted; no retry");
    wall_start=std::chrono::steady_clock::now();Budget budget;Report before,after;demo::Dataset data;int completed=0;std::string fault;bool readiness_passed=false,baseline_evaluated=false,candidate_evaluated=false,fit_started=false;
    try{Stream stream;int devices=0;CUDA_CHECK_THROW(cudaGetDeviceCount(&devices));require(devices>0,"Real CUDA required");cudarl::motor::FrozenController warm;require(warm.load(warm_start.string()),"Selected warm actor rejected");
        status(root,"checking");readiness(stream,budget);readiness_passed=true;
        status(root,"validating_examples",0,budget.total(),budget.actual_substeps);data=demo::accepted_examples(cp::read(example_file.string()));std::cout<<"ACCEPTED_EXAMPLES_VALIDATED rows="<<data.count()<<"; only six passed constant profiles; no teacher stepping"<<std::endl;
        status(root,"evaluating_baseline",0,budget.total(),budget.actual_substeps);before=evaluate(warm.actor_parameters(),root,"before",budget,stream);baseline_evaluated=true;verify_baseline(root/"before.csv",reference_csv);
        nn::GpuMlp actor(specification(2),minibatch,seed);actor.set_params(warm.actor_parameters());nn::AdamConfig config;config.lr=.001f;config.eps=.00001f;config.max_grad_norm=.5f;nn::GpuAdam optimizer(actor.num_params(),config);
        device_ptr<float> observations(data.observation.size()),targets(data.target.size()),batch_observation(minibatch*42),batch_target(minibatch*2),gradient(minibatch*2);upload(observations.get(),data.observation,stream.value);upload(targets.get(),data.target,stream.value);
        cp::Writer initial_policy;initial_policy.array(binding());initial_policy.array(warm.actor_parameters());cp::write_atomic((root/"initial-policy.bin").string(),initial_policy.data);
        std::ofstream progress(root/"training.csv");progress<<"step,latent_MSE,Adam_steps,physical_controls\n";
        fit_started=true;status(root,"supervised_training",0,budget.total(),budget.actual_substeps);std::cout<<"TRAINING_STARTED constant-speed behavior cloning; reused960accepted examples; no teacher or switches during frozen evaluation"<<std::endl;
        for(int step=0;step<fit_steps;++step){wall_guard();const auto permutation=rl::make_permutation(static_cast<uint32_t>(data.count()),seed+static_cast<uint32_t>(step));
            rl::gather_rows(permutation,0,minibatch,42,observations.get(),batch_observation.get(),stream.value);rl::gather_rows(permutation,0,minibatch,2,targets.get(),batch_target.get(),stream.value);actor.forward(batch_observation.get(),minibatch,stream.value);
            const auto predicted=download(actor.output(),minibatch*2,stream.value),expected=download(batch_target.get(),minibatch*2,stream.value);double loss=0;for(size_t j=0;j<predicted.size();++j)loss+=(predicted[j]-expected[j])*(predicted[j]-expected[j])/predicted.size();require(std::isfinite(loss),"Nonfinite supervised loss");
            nn::mse_loss_grad(actor.output(),batch_target.get(),gradient.get(),minibatch*2,stream.value);actor.zero_grad(stream.value);actor.backward(gradient.get(),minibatch,nullptr,stream.value);finite(download(actor.grads(),actor.num_params(),stream.value));optimizer.step(actor.params(),actor.grads(),stream.value);++completed;
            require(optimizer.steps()==completed,"Supervised optimizer count mismatch");progress<<completed<<','<<loss<<','<<optimizer.steps()<<','<<budget.total()<<'\n';if(completed%100==0||completed==fit_steps){progress.flush();snapshot(root,completed,actor,optimizer);status(root,"supervised_training",completed,budget.total(),budget.actual_substeps);}
        }
        finite(actor.get_params());status(root,"evaluating_candidate",completed,budget.total(),budget.actual_substeps);after=evaluate(actor.get_params(),root,"after",budget,stream);candidate_evaluated=true;
    }catch(const std::exception& error){fault=error.what();}
    const auto& b=before.speed[4];const auto& a=after.speed[4];const bool improved=fault.empty()&&data.count()==demo::kAcceptedRows&&after.successes==10&&after.normalized_rmse<before.normalized_rmse&&a.mean_speed_rmse<=b.mean_speed_rmse&&a.mean_work/fmaxf(a.mean_forward,1e-9f)<=b.mean_work/fmaxf(b.mean_forward,1e-9f);
    std::ofstream out(root/"summary.json");out<<std::setprecision(9)<<"{\"algorithm\":\"behavior_cloning\",\"model\":6,\"task\":5,\"teacher_version\":1,\"source_teacher_trial_accepted\":false,\"accepted_profile_count\":6,\"switch_evaluated\":false,\"previous_switch_gate_remains_unpassed\":true,\"baseline_evaluated\":"<<(baseline_evaluated?"true":"false")<<",\"candidate_evaluated\":"<<(candidate_evaluated?"true":"false")<<",\"fit_started\":"<<(fit_started?"true":"false")<<",\"teacher_examples\":"<<data.count()<<",\"fault\":"<<std::quoted(fault)<<",\"readiness_passed\":"<<(readiness_passed?"true":"false")<<",\"supervised_steps\":"<<completed<<",\"charged_controls\":"<<budget.total()<<",\"actual_native_substeps\":"<<budget.actual_substeps<<",\"stage_controls\":["<<budget.charged[0]<<','<<budget.charged[1]<<"],\"hard_control_cap\":1632,\"before_passed\":"<<before.successes<<",\"after_passed\":"<<after.successes<<",\"selected_candidate\":"<<(improved?"true":"false")<<",\"before\":[";
    for(int j=0;j<5;++j){if(j)out<<',';json_eval(out,before.speed[j],test_speeds[j]);}out<<"],\"after\":[";for(int j=0;j<5;++j){if(j)out<<',';json_eval(out,after.speed[j],test_speeds[j]);}out<<"],\"wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count()<<",\"closed\":true,\"retries\":0}\n";
    out.close();status(root,fault.empty()?"completed":"stopped",completed,budget.total(),budget.actual_substeps);std::ifstream immediate(root/"summary.json");std::cout<<immediate.rdbuf()<<std::flush;return fault.empty()?0:3;
}
}
int main(int argc,char** argv){try{require(argc>=3,"Use --prepare/--run DIR --actor ASSET --reference CSV --examples ACCEPTED_DATA [--fit-steps N --seed N --wall-seconds N]");for(int i=3;i<argc;i+=2){require(i+1<argc,"Missing argument value");const std::string flag=argv[i];if(flag=="--examples")example_file=fs::absolute(argv[i+1]);else if(flag=="--fit-steps")fit_steps=std::stoi(argv[i+1]);else if(flag=="--reference")reference_csv=fs::absolute(argv[i+1]);else if(flag=="--actor")warm_start=fs::absolute(argv[i+1]);else if(flag=="--seed")seed=static_cast<uint32_t>(std::stoul(argv[i+1],nullptr,0));else if(flag=="--wall-seconds")wall_limit=std::stof(argv[i+1]);else require(false,"Unknown configuration argument");}
    require(fit_steps>=1&&fit_steps<=600&&std::isfinite(wall_limit)&&wall_limit>0&&wall_limit<=20&&fs::is_regular_file(warm_start)&&fs::is_regular_file(reference_csv)&&fs::is_regular_file(example_file),"Configuration exceeds bounded teacher/fit limits");const fs::path root=fs::absolute(argv[2]);if(std::string(argv[1])=="--prepare"){prepare(root);return 0;}require(std::string(argv[1])=="--run","Unknown mode");return run(root);
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
