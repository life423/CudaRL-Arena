// One fixed, bounded native pilot. Not a general trainer/session framework.
#include "motor/straight_task.h"
#include "nn/mlp_cpu.h"
#include "nn/mlp_gpu.h"
#include "nn/adam.h"
#include "rl/gaussian_kernels.h"
#include "rl/ppo_kernels.h"
#include "core/checkpoint.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>

namespace {
namespace fs=std::filesystem;namespace p=cudarl::physics;namespace s=cudarl::simulation;
namespace m=cudarl::straight_motor;namespace nn=cudarl::nn;namespace rl=cudarl::rl;namespace cp=cudarl::checkpoint;
constexpr int worlds=16,minibatch=256,epochs=2;
int rollout=32,rows=worlds*rollout;float gae_lambda=.95f;
int updates=64;float requested_speed=.008f,wall_limit=20;
uint32_t seed=0x53574D32u;
constexpr float log_sigma=-2.f;
struct Stream {cudaStream_t value=nullptr;Stream(){CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}~Stream(){if(value){cudaStreamSynchronize(value);cudaStreamDestroy(value);}}};
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void finite(const std::vector<float>& values){for(float x:values)require(std::isfinite(x),"Nonfinite numerical buffer; stop before further work");}
std::vector<float> download(const float* pointer,size_t count,cudaStream_t stream){std::vector<float> out(count);CUDA_CHECK_THROW(cudaMemcpyAsync(out.data(),pointer,count*sizeof(float),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));finite(out);return out;}
void upload(float* pointer,const std::vector<float>& values,cudaStream_t stream){finite(values);CUDA_CHECK_THROW(cudaMemcpyAsync(pointer,values.data(),values.size()*sizeof(float),cudaMemcpyHostToDevice,stream));}
nn::MlpSpec specification(int outputs){return {{m::kObservations,32,32,outputs},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}};}
m::World initial(int id){m::World w;w.physical.paused=false;
    w.physical.body.orientation=p::axis_rotation({0,1,0},(float(id%4)-1.5f)*.05f);
    w.physical.body.position.x=(float(id%4)-1.5f)*.02f;
    w.physical.body.angle[0]=(id%2?.02f:-.02f);w.physical.body.angle[2]=-w.physical.body.angle[0];return w;}
m::ForwardConfig task_config(){m::ForwardConfig c;c.target_speed_m_s=requested_speed;c.minimum_speed_m_s=.9f*requested_speed;c.maximum_speed_m_s=1.1f*requested_speed;return c;}
std::vector<uint8_t> binding(){cp::Writer out;out.value(uint32_t{0x53574D32});out.value(uint32_t{3});out.value(seed);out.value(m::kTaskVersion);out.value(m::kRewardVersion);out.value(cudarl::movement::kVersion);
    for(int x:{m::kActions,m::kObservations,worlds,rollout,updates,minibatch,epochs,32,32768,640,33440,334400})out.value(x);
    for(float x:{log_sigma,.001f,.99f,gae_lambda,.2f,.5f,.00001f,.5f,.01f})out.value(x);
    const auto c=task_config();
    for(float x:{c.target_speed_m_s,c.clearance_m,c.minimum_speed_m_s,c.maximum_speed_m_s,c.speed_rmse_limit,c.lateral_limit_m,c.depth_limit_m,c.path_angle_limit_deg,c.tilt_limit_deg,c.work_per_distance_limit,wall_limit})out.value(x);
    out.value(c.maximum_controls);out.value(c.measure_start);
    // Canonical fields, never compiler-dependent struct padding, bind fixtures.
    for(int id=0;id<4;++id){const auto w=initial(id).physical;
        for(float x:{w.body.position.x,w.body.position.y,w.body.position.z,w.body.orientation.x,w.body.orientation.y,w.body.orientation.z,w.body.orientation.w,w.fluid.density,w.fluid.viscosity,w.fluid.gravity})out.value(x);
        for(float x:w.body.angle)out.value(x);for(float x:w.body.velocity)out.value(x);
        for(float x:w.gains.kp)out.value(x);for(float x:w.gains.kd)out.value(x);
        out.value(static_cast<uint32_t>(w.actuation));
    }return out.data;}
void status(const fs::path& root,const char* phase){std::ofstream out(root/"status.json");out<<"{\"phase\":"<<std::quoted(phase)<<",\"task\":4,\"reward_version\":2,\"model\":6,\"updates\":"<<updates<<",\"control_cap\":"<<32+updates*rows+640<<",\"wall_seconds_cap\":"<<wall_limit<<"}\n";out.flush();}
void prepare(const fs::path& root){fs::create_directories(root);require(!fs::exists(root/"protocol.bin")&&!fs::exists(root/"run-started"),"Protocol already prepared/attempted");cp::write_atomic((root/"protocol.bin").string(),binding());
 std::ofstream out(root/"protocol.json");out<<"{\"task\":4,\"reward_version\":2,\"model\":6,\"scope\":\"steady2-4s tracking reward, effort all4s; supplied stroke; no arrival test\",\"updates\":"<<updates<<",\"rollout_controls\":"<<rollout<<",\"GAE_lambda\":"<<gae_lambda<<",\"seed\":"<<seed<<",\"target_speed_m_s\":"<<requested_speed<<",\"controls_max\":"<<32+updates*rows+640<<",\"native_substeps_max\":"<<10*(32+updates*rows+640)<<",\"wall_seconds_max\":"<<wall_limit<<",\"heading\":\"COM path direction gate; instantaneous head yaw reported separately\",\"retries\":0}\n";status(root,"registered");}
auto wall_start=std::chrono::steady_clock::now();
void wall_guard(){require(std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count()<wall_limit,"Native 20s wall cap reached");}
struct Budget {int charged[3]{},actual_substeps=0;void reserve(int stage,int n){wall_guard();const int cap[3]{32,updates*rows,640};require(n>0&&charged[stage]+n<=cap[stage],"Stage control budget exhausted");charged[stage]+=n;}int total()const{return charged[0]+charged[1]+charged[2];}};
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
        for(float x:w.actual_torques)require(std::isfinite(x)&&std::fabs(x)<=1,"Invalid applied torque");require(m::in_domain(state[i],task_config()),"Physical/domain failure; reject before PPO");}
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
struct Evaluation {int successes=0;float mean_forward=0,mean_depth=0,mean_return=0,mean_controls=0,mean_work=0,mean_speed=0,mean_speed_rmse=0,max_lateral=0,max_depth=0,max_head_yaw=0,max_tilt=0,max_path_angle=0;};
Evaluation evaluate(const std::vector<float>& parameters,const fs::path& root,const char* name,Budget& budget,Stream& stream){
    nn::CpuMlp<float> actor(specification(2));actor.params()=parameters;const auto config=task_config();Evaluation result;
    std::ofstream trace(root/(std::string(name)+".csv"));trace.exceptions(std::ios::badbit|std::ios::failbit);trace<<std::setprecision(9)<<"case,tick,forward_m,depth_m,lateral_m,speed_error_m_s,reward,success,head_x,head_y,head_z,qx,qy,qz,qw,j0,j1,j2,j3,phase,amplitude,frequency,positive_work_J\n";
    for(int id=0;id<4;++id){std::vector<m::World> state{initial(id)};m::ForwardTask task;require(m::begin_forward(state[0],config,task),"Invalid evaluation start");m::Transition transition;
        for(int tick=0;tick<config.maximum_controls;++tick){float obs[m::kObservations]{};require(m::observe(task,state[0],config,obs),"Invalid evaluation observation");actor.forward(obs,1);m::Action action;require(m::action_from_latent(actor.output(),action),"Invalid deterministic action");
            std::vector<std::array<float,4>> command(1);for(int j=0;j<4;++j)command[0][j]=action.command[j];advance(state,command,2,false,stream.value,budget);transition=m::record_control(task,state[0],config);require(transition.accepted,"Rejected evaluation sample");
            const auto& b=state[0].physical.body;trace<<id<<','<<tick+1<<','<<transition.metrics.forward_m<<','<<transition.metrics.depth_m<<','<<transition.metrics.lateral_m<<','<<transition.metrics.speed_error_m_s<<','<<transition.reward<<','<<(transition.outcome==m::Outcome::Success)<<','<<b.position.x<<','<<b.position.y<<','<<b.position.z<<','<<b.orientation.x<<','<<b.orientation.y<<','<<b.orientation.z<<','<<b.orientation.w;for(float x:b.angle)trace<<','<<x;trace<<','<<state[0].phase;for(float x:state[0].parameter)trace<<','<<x;trace<<','<<b.work<<'\n';
            result.mean_return+=transition.reward/4;result.mean_controls+=.25f;
            if(transition.done)break;
        }result.successes+=transition.outcome==m::Outcome::Success?1:0;result.mean_forward+=transition.metrics.forward_m/4;result.mean_depth+=transition.metrics.depth_m/4;result.mean_work+=state[0].physical.body.work/4;result.mean_speed+=task.steady_speed_m_s/4;result.mean_speed_rmse+=task.speed_rmse/4;result.max_lateral=fmaxf(result.max_lateral,task.maximum_lateral);result.max_depth=fmaxf(result.max_depth,task.maximum_depth);result.max_head_yaw=fmaxf(result.max_head_yaw,task.maximum_head_yaw);result.max_tilt=fmaxf(result.max_tilt,task.maximum_tilt);result.max_path_angle=fmaxf(result.max_path_angle,task.path_angle_deg);
    }return result;
}
void snapshot(const fs::path& root,int update,nn::GpuMlp& actor,nn::GpuMlp& critic,nn::GpuAdam& actor_opt,nn::GpuAdam& critic_opt,const std::vector<m::World>& state,const std::vector<m::ForwardTask>& task,const std::vector<int>& episodes){
    cp::Writer out;out.value(uint32_t{0x46575331});out.value(uint32_t{1});out.value(uint32_t{sizeof(m::World)});out.value(uint32_t{sizeof(m::ForwardTask)});out.value(update);out.array(binding());out.array(actor.get_params());out.array(critic.get_params());actor_opt.save(out);critic_opt.save(out);out.array(state);out.array(task);out.array(episodes);
    const auto path=root/("update-"+std::to_string(update)+".bin");require(!fs::exists(path),"Snapshot already exists");cp::write_atomic(path.string(),out.data);require(cp::read(path.string())==out.data,"Snapshot checksum/roundtrip failure");
}
int run(const fs::path& root){require(cp::read((root/"protocol.bin").string())==binding(),"Frozen numeric protocol differs");require(fs::create_directory(root/"run-started"),"Pilot already attempted; no retry");
    wall_start=std::chrono::steady_clock::now();Budget budget;Evaluation before,after;int completed=0;std::string fault;bool readiness_passed=false;const auto start=std::chrono::steady_clock::now();status(root,"running");
    try{Stream stream;int devices=0;CUDA_CHECK_THROW(cudaGetDeviceCount(&devices));require(devices>0,"Real CUDA required");readiness(stream,budget);readiness_passed=true;
        const auto actor_spec=specification(2),critic_spec=specification(1);nn::GpuMlp actor(actor_spec,rows,seed),critic(critic_spec,rows,seed+1);
        auto initial_parameters=nn::orthogonal_params(actor_spec,seed,{1.414213562f,1.414213562f,.01f});initial_parameters[actor_spec.bias_offset(2)]=.8958797346f;initial_parameters[actor_spec.bias_offset(2)+1]=.2027325541f;actor.set_params(initial_parameters);critic.set_params(nn::orthogonal_params(critic_spec,seed+1,{1.414213562f,1.414213562f,1.f}));
        nn::AdamConfig optimizer;optimizer.lr=.001f;optimizer.eps=.00001f;optimizer.max_grad_norm=.5f;nn::GpuAdam actor_opt(actor.num_params(),optimizer),critic_opt(critic.num_params(),optimizer);
        cp::Writer initial_policy;initial_policy.array(binding());initial_policy.array(actor.get_params());cp::write_atomic((root/"initial-policy.bin").string(),initial_policy.data);
        before=evaluate(actor.get_params(),root,"before",budget,stream);
        std::vector<m::World> state;std::vector<m::ForwardTask> task(worlds);std::vector<int> episodes(worlds);const auto config=task_config();
        for(int i=0;i<worlds;++i){state.push_back(initial(i));require(m::begin_forward(state.back(),config,task[i]),"Invalid training start");}
        device_ptr<float> step_obs(worlds*m::kObservations),latent(worlds*2),logp(worlds),sigma(2);
        sigma.copy_from_host(std::vector<float>(2,log_sigma).data(),2);
        device_ptr<float> d_obs(rows*m::kObservations),d_action(rows*2),d_logp(rows),d_value(rows),d_reward(rows),d_done(rows),d_zero(rows),d_last(worlds),d_adv(rows),d_return(rows),adv_stats(2);
        CUDA_CHECK_THROW(cudaMemsetAsync(d_zero.get(),0,d_zero.size_bytes(),stream.value));
        device_ptr<float> mb_obs(minibatch*m::kObservations),mb_action(minibatch*2),mb_logp(minibatch),mb_adv(minibatch),mb_return(minibatch),d_mean(minibatch*2),d_std(minibatch*2),d_v(minibatch),row_stats(minibatch*5),stats(5);
        std::ofstream progress(root/"training.csv");progress.exceptions(std::ios::badbit|std::ios::failbit);progress<<"update,controls,mean_reward,actor_Adam_steps,critic_Adam_steps\n";
        for(int update=0;update<updates;++update){std::vector<float> observations(rows*m::kObservations),actions(rows*2),old_logp(rows),values(rows),rewards(rows),done(rows);
            for(int tick=0;tick<rollout;++tick){std::vector<float> obs(worlds*m::kObservations);for(int i=0;i<worlds;++i)require(m::observe(task[i],state[i],config,obs.data()+i*m::kObservations),"Invalid rollout observation");
                upload(step_obs.get(),obs,stream.value);actor.forward(step_obs.get(),worlds,stream.value);critic.forward(step_obs.get(),worlds,stream.value);rl::sample_gaussian(actor.output(),sigma.get(),worlds,2,seed+2,static_cast<uint32_t>(update*rollout+tick),latent.get(),logp.get(),stream.value);
                const auto z=download(latent.get(),worlds*2,stream.value),lp=download(logp.get(),worlds,stream.value),v=download(critic.output(),worlds,stream.value);
                std::copy(obs.begin(),obs.end(),observations.begin()+tick*worlds*m::kObservations);std::copy(z.begin(),z.end(),actions.begin()+tick*worlds*2);std::copy(lp.begin(),lp.end(),old_logp.begin()+tick*worlds);std::copy(v.begin(),v.end(),values.begin()+tick*worlds);
                std::vector<std::array<float,4>> command(worlds);for(int i=0;i<worlds;++i){m::Action action;require(m::action_from_latent(z.data()+i*2,action),"Invalid sampled latent");for(int j=0;j<4;++j)command[i][j]=action.command[j];}
                advance(state,command,1,true,stream.value,budget);
                for(int i=0;i<worlds;++i){const auto t=m::record_control(task[i],state[i],config);require(t.accepted,"Rejected rollout; stop BEFORE GAE/Adam");rewards[tick*worlds+i]=t.reward;done[tick*worlds+i]=t.done?1.f:0.f;
                    if(t.done){++episodes[i];state[i]=initial(i+episodes[i]);require(m::begin_forward(state[i],config,task[i]),"Invalid explicit training reset");}}
            }
            std::vector<float> next_obs(worlds*m::kObservations);for(int i=0;i<worlds;++i)require(m::observe(task[i],state[i],config,next_obs.data()+i*m::kObservations),"Invalid bootstrap observation");upload(step_obs.get(),next_obs,stream.value);critic.forward(step_obs.get(),worlds,stream.value);const auto last=download(critic.output(),worlds,stream.value);
            upload(d_obs.get(),observations,stream.value);upload(d_action.get(),actions,stream.value);upload(d_logp.get(),old_logp,stream.value);upload(d_value.get(),values,stream.value);upload(d_reward.get(),rewards,stream.value);upload(d_done.get(),done,stream.value);upload(d_last.get(),last,stream.value);
            rl::compute_gae(rollout,worlds,d_reward.get(),d_value.get(),d_done.get(),d_zero.get(),d_zero.get(),d_last.get(),.99f,gae_lambda,d_adv.get(),d_return.get(),stream.value);rl::mean_std(d_adv.get(),rows,adv_stats.get(),stream.value);
            finite(download(d_adv.get(),rows,stream.value));finite(download(d_return.get(),rows,stream.value));
            for(int epoch=0;epoch<epochs;++epoch){wall_guard();const auto perm=rl::make_permutation(rows,seed+static_cast<uint32_t>(update*epochs+epoch));for(int first=0;first<rows;first+=minibatch){
                rl::gather_rows(perm,first,minibatch,m::kObservations,d_obs.get(),mb_obs.get(),stream.value);rl::gather_rows(perm,first,minibatch,2,d_action.get(),mb_action.get(),stream.value);rl::gather_rows(perm,first,minibatch,1,d_logp.get(),mb_logp.get(),stream.value);rl::gather_rows(perm,first,minibatch,1,d_adv.get(),mb_adv.get(),stream.value);rl::gather_rows(perm,first,minibatch,1,d_return.get(),mb_return.get(),stream.value);
                actor.forward(mb_obs.get(),minibatch,stream.value);critic.forward(mb_obs.get(),minibatch,stream.value);CUDA_CHECK_THROW(cudaMemsetAsync(stats.get(),0,stats.size_bytes(),stream.value));
                rl::gaussian_ppo_loss_grad(actor.output(),sigma.get(),critic.output(),minibatch,2,mb_action.get(),mb_logp.get(),mb_adv.get(),mb_return.get(),adv_stats.get(),{.2f,0.f,.5f},d_mean.get(),d_std.get(),d_v.get(),row_stats.get(),stats.get(),stream.value);
                actor.zero_grad(stream.value);critic.zero_grad(stream.value);actor.backward(d_mean.get(),minibatch,nullptr,stream.value);critic.backward(d_v.get(),minibatch,nullptr,stream.value);
                finite(download(actor.grads(),actor.num_params(),stream.value));finite(download(critic.grads(),critic.num_params(),stream.value));actor_opt.step(actor.params(),actor.grads(),stream.value);critic_opt.step(critic.params(),critic.grads(),stream.value);
            }}
            finite(actor.get_params());finite(critic.get_params());++completed;require(actor_opt.steps()==completed*epochs*(rows/minibatch)&&critic_opt.steps()==completed*epochs*(rows/minibatch),"Optimizer count differs");snapshot(root,completed,actor,critic,actor_opt,critic_opt,state,task,episodes);
            float mean_reward=0;for(float reward:rewards)mean_reward+=reward/rows;progress<<completed<<','<<budget.charged[1]<<','<<mean_reward<<','<<actor_opt.steps()<<','<<critic_opt.steps()<<'\n';progress.flush();std::cout<<"PPO update "<<completed<<"/"<<updates<<"; training controls "<<budget.charged[1]<<"/"<<updates*rows<<"\n";
        }
        after=evaluate(actor.get_params(),root,"after",budget,stream);
    }catch(const std::exception& error){fault=error.what();}
    std::ofstream out(root/"summary.json");out<<std::setprecision(9)<<"{\"scope\":\"straight-speed task4/reward2; supplied oscillator, no runtime adoption\",\"fault\":\""<<fault<<"\",\"reward_version\":2,\"readiness_passed\":"<<(readiness_passed?"true":"false")<<",\"PPO_updates\":"<<completed<<",\"rollout_controls\":"<<rollout<<",\"GAE_lambda\":"<<gae_lambda<<",\"charged_controls\":"<<budget.total()<<",\"charged_native_slots\":"<<budget.total()*10<<",\"actual_native_substeps\":"<<budget.actual_substeps<<",\"stage_controls\":["<<budget.charged[0]<<','<<budget.charged[1]<<','<<budget.charged[2]<<"],\"hard_control_cap\":33440,\"planned_control_cap\":"<<32+updates*rows+640<<",\"before_successes\":"<<before.successes<<",\"after_successes\":"<<after.successes<<",\"before_mean_forward_m\":"<<before.mean_forward<<",\"after_mean_forward_m\":"<<after.mean_forward<<",\"before_mean_depth_m\":"<<before.mean_depth<<",\"after_mean_depth_m\":"<<after.mean_depth<<",\"before_mean_speed_m_s\":"<<before.mean_speed<<",\"after_mean_speed_m_s\":"<<after.mean_speed<<",\"before_speed_RMSE_m_s\":"<<before.mean_speed_rmse<<",\"after_speed_RMSE_m_s\":"<<after.mean_speed_rmse<<",\"before_max_lateral_m\":"<<before.max_lateral<<",\"after_max_lateral_m\":"<<after.max_lateral<<",\"before_max_depth_m\":"<<before.max_depth<<",\"after_max_depth_m\":"<<after.max_depth<<",\"before_max_head_yaw_deg\":"<<before.max_head_yaw<<",\"after_max_head_yaw_deg\":"<<after.max_head_yaw<<",\"before_max_tilt_deg\":"<<before.max_tilt<<",\"after_max_tilt_deg\":"<<after.max_tilt<<",\"before_max_path_angle_deg\":"<<before.max_path_angle<<",\"after_max_path_angle_deg\":"<<after.max_path_angle<<",\"before_mean_return\":"<<before.mean_return<<",\"after_mean_return\":"<<after.mean_return<<",\"before_mean_controls\":"<<before.mean_controls<<",\"after_mean_controls\":"<<after.mean_controls<<",\"before_mean_work_J\":"<<before.mean_work<<",\"after_mean_work_J\":"<<after.mean_work<<",\"wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"closed\":true,\"retries\":0}\n";
    std::cout<<"Pilot closed: updates="<<completed<<", controls="<<budget.total()<<"/33440, success="<<after.successes<<"/4, fault="<<fault<<'\n';out.close();status(root,fault.empty()?"completed":"stopped");std::ifstream immediate(root/"summary.json");std::cout<<immediate.rdbuf()<<std::flush;return fault.empty()?0:3;
}
}
int main(int argc,char** argv){try{require(argc>=3,"Use --prepare/--run DIR [--updates N --seed N --speed-mm-s N --wall-seconds N]");for(int i=3;i<argc;i+=2){require(i+1<argc,"Missing argument value");const std::string flag=argv[i];if(flag=="--updates")updates=std::stoi(argv[i+1]);else if(flag=="--rollout-controls")rollout=std::stoi(argv[i+1]);else if(flag=="--lambda")gae_lambda=std::stof(argv[i+1]);else if(flag=="--seed")seed=static_cast<uint32_t>(std::stoul(argv[i+1],nullptr,0));else if(flag=="--speed-mm-s")requested_speed=std::stof(argv[i+1])*.001f;else if(flag=="--wall-seconds")wall_limit=std::stof(argv[i+1]);else require(false,"Unknown configuration argument");}
 rows=worlds*rollout;require((rollout==32||rollout==80)&&updates>=1&&updates<=64&&updates*rows<=32768&&std::isfinite(gae_lambda)&&gae_lambda>=0&&gae_lambda<=1&&std::isfinite(requested_speed)&&requested_speed>=.002f&&requested_speed<=.012f&&std::isfinite(wall_limit)&&wall_limit>0&&wall_limit<=20,"Configuration exceeds bounded pilot limits");const fs::path root=fs::absolute(argv[2]);if(std::string(argv[1])=="--prepare"){prepare(root);return 0;}require(std::string(argv[1])=="--run","Unknown mode");return run(root);}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
