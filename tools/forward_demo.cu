// One fixed, bounded native pilot. Not a general trainer/session framework.
#include "motor/forward_task.h"
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
namespace m=cudarl::motor;namespace nn=cudarl::nn;namespace rl=cudarl::rl;namespace cp=cudarl::checkpoint;
constexpr int worlds=16,rollout=32,updates=256,rows=worlds*rollout,minibatch=256,epochs=2;
constexpr uint32_t seed=0x46574431u;
constexpr float log_sigma=-1.5f;
struct Stream {cudaStream_t value=nullptr;Stream(){CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}~Stream(){if(value){cudaStreamSynchronize(value);cudaStreamDestroy(value);}}};
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void finite(const std::vector<float>& values){for(float x:values)require(std::isfinite(x),"Nonfinite numerical buffer; stop before further work");}
std::vector<float> download(const float* pointer,size_t count,cudaStream_t stream){std::vector<float> out(count);CUDA_CHECK_THROW(cudaMemcpyAsync(out.data(),pointer,count*sizeof(float),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));finite(out);return out;}
void upload(float* pointer,const std::vector<float>& values,cudaStream_t stream){finite(values);CUDA_CHECK_THROW(cudaMemcpyAsync(pointer,values.data(),values.size()*sizeof(float),cudaMemcpyHostToDevice,stream));}
nn::MlpSpec specification(int outputs){return {{m::kObservations,32,32,outputs},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}};}
s::WorldState initial(int id){s::WorldState w;w.paused=false;w.actuation=s::Actuation::JointTarget;
    w.body.orientation=p::axis_rotation({0,1,0},(float(id%4)-1.5f)*.05f);
    w.body.position.x=(float(id%4)-1.5f)*.02f;w.body.angle[0]=(id%2?.02f:-.02f);w.body.angle[2]=-w.body.angle[0];return w;}
std::vector<uint8_t> binding(){cp::Writer out;out.value(uint32_t{0x46574431});out.value(uint32_t{1});out.value(seed);out.value(m::kTaskVersion);out.value(cudarl::contact::kPhysicsVersion);
    for(int x:{worlds,rollout,updates,minibatch,epochs,32,131072,640,131744,1317440})out.value(x);
    for(float x:{log_sigma,.001f,.99f,.95f,.2f,.5f,.00001f,.5f,.01f})out.value(x);
    const m::ForwardConfig c;
    for(float x:{c.distance_m,c.terminal_speed_m_s,c.clearance_m,c.lateral_tolerance_m,c.speed_tolerance_m_s,c.heading_cos})out.value(x);
    out.value(c.maximum_controls);out.value(c.success_controls);
    // Canonical fields, never compiler-dependent struct padding, bind fixtures.
    for(int id=0;id<4;++id){const auto w=initial(id);
        for(float x:{w.body.position.x,w.body.position.y,w.body.position.z,w.body.orientation.x,w.body.orientation.y,w.body.orientation.z,w.body.orientation.w,w.fluid.density,w.fluid.viscosity,w.fluid.gravity})out.value(x);
        for(float x:w.body.angle)out.value(x);for(float x:w.body.velocity)out.value(x);
        for(float x:w.gains.kp)out.value(x);for(float x:w.gains.kd)out.value(x);
        out.value(static_cast<uint32_t>(w.actuation));
    }return out.data;}
void prepare(const fs::path& root){fs::create_directories(root);require(!fs::exists(root/"protocol.bin")&&!fs::exists(root/"run-started"),"Protocol already prepared/attempted");cp::write_atomic((root/"protocol.bin").string(),binding());
    std::ofstream out(root/"protocol.json");out.exceptions(std::ios::badbit|std::ios::failbit);
    out<<"{\"version\":1,\"scope\":\"one bounded native forward pilot, no promised swimming\",\"physics\":2,\"interface\":\"fixed-gain joint targets\",\"current\":0,\"clearance_m\":0.01,\"readiness_controls\":32,\"training_controls_max\":131072,\"evaluation_controls_max\":640,\"total_controls_max\":131744,\"native_5ms_slots_max\":1317440,\"worlds\":16,\"rollout\":32,\"PPO_updates_max\":256,\"epochs\":2,\"minibatches\":2,\"Adam_steps_max_each\":1024,\"fixed_log_sigma\":-1.5,\"retries\":0,\"training_resets\":\"explicit finite episode boundaries only\",\"evaluation\":\"four fixed starts, 80 controls each, before and after, deterministic latent means\",\"stop\":\"any physical domain, numerical, integrity or parity failure stops the pilot; no extension\"}\n";
}
struct Budget {int charged[3]{},actual_substeps=0;void reserve(int stage,int n){const int cap[3]{32,131072,640};require(n>0&&charged[stage]+n<=cap[stage],"Stage control budget exhausted");charged[stage]+=n;}int total()const{return charged[0]+charged[1]+charged[2];}};
void advance(std::vector<s::WorldState>& state,const std::vector<std::array<float,4>>& command,int stage,bool gpu,cudaStream_t stream,Budget& budget){
    budget.reserve(stage,static_cast<int>(state.size()));std::vector<uint64_t> before;for(const auto& w:state)before.push_back(w.substeps);
    if(gpu)s::cuda_step(state,command,stream);else for(size_t i=0;i<state.size();++i)require(s::step_state(state[i],command[i].data()),"CPU step refused");
    const m::ForwardConfig config;
    for(size_t i=0;i<state.size();++i)budget.actual_substeps+=static_cast<int>(state[i].substeps-before[i]);
    for(size_t i=0;i<state.size();++i){const auto& w=state[i];
        require(w.substeps-before[i]==10&&w.telemetry.executed_substeps==10,"Incomplete native interval");
        require(w.telemetry.contact_samples==0&&w.telemetry.total_spring_force_max==0,"Unexpected contact in free-water pilot");
        for(float x:w.actual_torques)require(std::isfinite(x)&&std::fabs(x)<=1,"Invalid applied substep torque");
        for(float x:w.requested_torques)require(std::isfinite(x),"Invalid requested substep torque");
        require(m::in_domain(w,config),"Physical/domain failure: no contact/clearance rescue or update allowed");}
}
std::vector<float> fields(const s::WorldState& w){std::vector<float> out{w.body.position.x,w.body.position.y,w.body.position.z,w.body.orientation.x,w.body.orientation.y,w.body.orientation.z,w.body.orientation.w,w.body.work,w.body.travel,w.mean_squared_torque};
    for(float x:w.body.angle)out.push_back(x);for(float x:w.body.velocity)out.push_back(x);for(float x:w.last_actual_torques)out.push_back(x);for(float x:w.held_targets)out.push_back(x);for(float x:w.actual_torques)out.push_back(x);
    for(float x:w.requested_torques)out.push_back(x);for(float x:w.raw_action)out.push_back(x);
    const auto& t=w.telemetry;const auto& i=t.inspection;
    for(float x:{w.current.x,w.current.y,w.current.z,w.fluid.density,w.fluid.viscosity,w.fluid.gravity,i.floor_gap,i.min_other_gap,i.min_gap,i.max_floor_gap,t.minimum_floor_gap,t.minimum_other_gap,t.floor_spring_force_max,t.total_spring_force_max})out.push_back(x);
    for(float x:w.gains.kp)out.push_back(x);for(float x:w.gains.kd)out.push_back(x);return out;}
void readiness(Stream& stream,Budget& budget){std::vector<s::WorldState> cpu,gpu;for(int i=0;i<4;++i)cpu.push_back(initial(i));gpu=cpu;
    for(int tick=0;tick<4;++tick){std::vector<std::array<float,4>> commands(4);for(auto& command:commands)command={tick%2?.1f:-.1f,0,tick%2?-.1f:.1f,0};
        advance(cpu,commands,0,false,stream.value,budget);advance(gpu,commands,0,true,stream.value,budget);
        for(int i=0;i<4;++i){const auto a=fields(cpu[i]),b=fields(gpu[i]);finite(a);finite(b);for(size_t j=0;j<a.size();++j)require(std::fabs(a[j]-b[j])<=2e-4f+2e-3f*std::fabs(a[j]),"Readiness CPU/CUDA physical mismatch");
            for(const auto pair:{std::array<float,2>{cpu[i].telemetry.minimum_floor_gap,gpu[i].telemetry.minimum_floor_gap},std::array<float,2>{cpu[i].telemetry.minimum_other_gap,gpu[i].telemetry.minimum_other_gap}})
                require(std::fabs(pair[0]-pair[1])<=2e-6f+2e-3f*std::fabs(pair[0]),"Readiness CPU/CUDA clearance mismatch");
            require(cpu[i].substeps==gpu[i].substeps&&cpu[i].control_steps==gpu[i].control_steps&&cpu[i].time_seconds==gpu[i].time_seconds&&cpu[i].unsafe==gpu[i].unsafe&&cpu[i].invalid==gpu[i].invalid,"Readiness CPU/CUDA flags/count mismatch");}
    }
}
struct Evaluation {int successes=0;float mean_forward=0,mean_error=0;};
Evaluation evaluate(const std::vector<float>& parameters,const fs::path& root,const char* name,Budget& budget,Stream& stream){
    nn::CpuMlp<float> actor(specification(4));actor.params()=parameters;const m::ForwardConfig config;Evaluation result;
    std::ofstream trace(root/(std::string(name)+".csv"));trace.exceptions(std::ios::badbit|std::ios::failbit);trace<<std::setprecision(9)<<"case,tick,forward_m,error_m,lateral_m,speed_error_m_s,reward,success,head_x,head_y,head_z,qx,qy,qz,qw,j0,j1,j2,j3\n";
    for(int id=0;id<4;++id){std::vector<s::WorldState> state{initial(id)};m::ForwardTask task;require(m::begin_forward(state[0],config,task),"Invalid evaluation start");m::Transition transition;
        for(int tick=0;tick<config.maximum_controls;++tick){float obs[m::kObservations]{};require(m::observe(task,state[0],config,obs),"Invalid evaluation observation");actor.forward(obs,1);m::Action action;require(m::action_from_latent(actor.output(),action),"Invalid deterministic action");
            std::vector<std::array<float,4>> command(1);for(int j=0;j<4;++j)command[0][j]=action.command[j];advance(state,command,2,false,stream.value,budget);transition=m::record_control(task,state[0],config);require(transition.accepted,"Rejected evaluation sample");
            const auto& b=state[0].body;trace<<id<<','<<tick+1<<','<<transition.metrics.forward_m<<','<<transition.metrics.error_m<<','<<transition.metrics.lateral_m<<','<<transition.metrics.speed_error_m_s<<','<<transition.reward<<','<<(transition.outcome==m::Outcome::Success)<<','<<b.position.x<<','<<b.position.y<<','<<b.position.z<<','<<b.orientation.x<<','<<b.orientation.y<<','<<b.orientation.z<<','<<b.orientation.w;for(float x:b.angle)trace<<','<<x;trace<<'\n';
            if(transition.done)break;
        }result.successes+=transition.outcome==m::Outcome::Success?1:0;result.mean_forward+=transition.metrics.forward_m/4;result.mean_error+=transition.metrics.error_m/4;
    }return result;
}
void snapshot(const fs::path& root,int update,nn::GpuMlp& actor,nn::GpuMlp& critic,nn::GpuAdam& actor_opt,nn::GpuAdam& critic_opt,const std::vector<s::WorldState>& state,const std::vector<m::ForwardTask>& task,const std::vector<int>& episodes){
    cp::Writer out;out.value(uint32_t{0x46575331});out.value(uint32_t{1});out.value(uint32_t{sizeof(s::WorldState)});out.value(uint32_t{sizeof(m::ForwardTask)});out.value(update);out.array(binding());out.array(actor.get_params());out.array(critic.get_params());actor_opt.save(out);critic_opt.save(out);out.array(state);out.array(task);out.array(episodes);
    const auto path=root/("update-"+std::to_string(update)+".bin");require(!fs::exists(path),"Snapshot already exists");cp::write_atomic(path.string(),out.data);require(cp::read(path.string())==out.data,"Snapshot checksum/roundtrip failure");
}
int run(const fs::path& root){require(cp::read((root/"protocol.bin").string())==binding(),"Frozen numeric protocol differs");require(fs::create_directory(root/"run-started"),"Pilot already attempted; no retry");
    Budget budget;Evaluation before,after;int completed=0;std::string fault;bool readiness_passed=false;const auto start=std::chrono::steady_clock::now();
    try{Stream stream;int devices=0;CUDA_CHECK_THROW(cudaGetDeviceCount(&devices));require(devices>0,"Real CUDA required");readiness(stream,budget);readiness_passed=true;
        const auto actor_spec=specification(4),critic_spec=specification(1);nn::GpuMlp actor(actor_spec,rows,seed),critic(critic_spec,rows,seed+1);
        actor.set_params(nn::orthogonal_params(actor_spec,seed,{1.414213562f,1.414213562f,.01f}));critic.set_params(nn::orthogonal_params(critic_spec,seed+1,{1.414213562f,1.414213562f,1.f}));
        nn::AdamConfig optimizer;optimizer.lr=.001f;optimizer.eps=.00001f;optimizer.max_grad_norm=.5f;nn::GpuAdam actor_opt(actor.num_params(),optimizer),critic_opt(critic.num_params(),optimizer);
        cp::Writer initial_policy;initial_policy.array(binding());initial_policy.array(actor.get_params());cp::write_atomic((root/"initial-policy.bin").string(),initial_policy.data);
        before=evaluate(actor.get_params(),root,"before",budget,stream);
        std::vector<s::WorldState> state;std::vector<m::ForwardTask> task(worlds);std::vector<int> episodes(worlds);const m::ForwardConfig config;
        for(int i=0;i<worlds;++i){state.push_back(initial(i));require(m::begin_forward(state.back(),config,task[i]),"Invalid training start");}
        device_ptr<float> step_obs(worlds*m::kObservations),latent(worlds*4),logp(worlds),sigma(4);
        sigma.copy_from_host(std::vector<float>(4,log_sigma).data(),4);
        device_ptr<float> d_obs(rows*m::kObservations),d_action(rows*4),d_logp(rows),d_value(rows),d_reward(rows),d_done(rows),d_zero(rows),d_last(worlds),d_adv(rows),d_return(rows),adv_stats(2);
        CUDA_CHECK_THROW(cudaMemsetAsync(d_zero.get(),0,d_zero.size_bytes(),stream.value));
        device_ptr<float> mb_obs(minibatch*m::kObservations),mb_action(minibatch*4),mb_logp(minibatch),mb_adv(minibatch),mb_return(minibatch),d_mean(minibatch*4),d_std(minibatch*4),d_v(minibatch),row_stats(minibatch*5),stats(5);
        std::ofstream progress(root/"training.csv");progress.exceptions(std::ios::badbit|std::ios::failbit);progress<<"update,controls,mean_reward,actor_Adam_steps,critic_Adam_steps\n";
        for(int update=0;update<updates;++update){std::vector<float> observations(rows*m::kObservations),actions(rows*4),old_logp(rows),values(rows),rewards(rows),done(rows);
            for(int tick=0;tick<rollout;++tick){std::vector<float> obs(worlds*m::kObservations);for(int i=0;i<worlds;++i)require(m::observe(task[i],state[i],config,obs.data()+i*m::kObservations),"Invalid rollout observation");
                upload(step_obs.get(),obs,stream.value);actor.forward(step_obs.get(),worlds,stream.value);critic.forward(step_obs.get(),worlds,stream.value);rl::sample_gaussian(actor.output(),sigma.get(),worlds,4,seed+2,static_cast<uint32_t>(update*rollout+tick),latent.get(),logp.get(),stream.value);
                const auto z=download(latent.get(),worlds*4,stream.value),lp=download(logp.get(),worlds,stream.value),v=download(critic.output(),worlds,stream.value);
                std::copy(obs.begin(),obs.end(),observations.begin()+tick*worlds*m::kObservations);std::copy(z.begin(),z.end(),actions.begin()+tick*worlds*4);std::copy(lp.begin(),lp.end(),old_logp.begin()+tick*worlds);std::copy(v.begin(),v.end(),values.begin()+tick*worlds);
                std::vector<std::array<float,4>> command(worlds);for(int i=0;i<worlds;++i){m::Action action;require(m::action_from_latent(z.data()+i*4,action),"Invalid sampled latent");for(int j=0;j<4;++j)command[i][j]=action.command[j];}
                advance(state,command,1,true,stream.value,budget);
                for(int i=0;i<worlds;++i){const auto t=m::record_control(task[i],state[i],config);require(t.accepted,"Rejected rollout; stop BEFORE GAE/Adam");rewards[tick*worlds+i]=t.reward;done[tick*worlds+i]=t.done?1.f:0.f;
                    if(t.done){++episodes[i];state[i]=initial(i+episodes[i]);require(m::begin_forward(state[i],config,task[i]),"Invalid explicit training reset");}}
            }
            std::vector<float> next_obs(worlds*m::kObservations);for(int i=0;i<worlds;++i)require(m::observe(task[i],state[i],config,next_obs.data()+i*m::kObservations),"Invalid bootstrap observation");upload(step_obs.get(),next_obs,stream.value);critic.forward(step_obs.get(),worlds,stream.value);const auto last=download(critic.output(),worlds,stream.value);
            upload(d_obs.get(),observations,stream.value);upload(d_action.get(),actions,stream.value);upload(d_logp.get(),old_logp,stream.value);upload(d_value.get(),values,stream.value);upload(d_reward.get(),rewards,stream.value);upload(d_done.get(),done,stream.value);upload(d_last.get(),last,stream.value);
            rl::compute_gae(rollout,worlds,d_reward.get(),d_value.get(),d_done.get(),d_zero.get(),d_zero.get(),d_last.get(),.99f,.95f,d_adv.get(),d_return.get(),stream.value);rl::mean_std(d_adv.get(),rows,adv_stats.get(),stream.value);
            finite(download(d_adv.get(),rows,stream.value));finite(download(d_return.get(),rows,stream.value));
            for(int epoch=0;epoch<epochs;++epoch){const auto perm=rl::make_permutation(rows,seed+static_cast<uint32_t>(update*epochs+epoch));for(int first=0;first<rows;first+=minibatch){
                rl::gather_rows(perm,first,minibatch,m::kObservations,d_obs.get(),mb_obs.get(),stream.value);rl::gather_rows(perm,first,minibatch,4,d_action.get(),mb_action.get(),stream.value);rl::gather_rows(perm,first,minibatch,1,d_logp.get(),mb_logp.get(),stream.value);rl::gather_rows(perm,first,minibatch,1,d_adv.get(),mb_adv.get(),stream.value);rl::gather_rows(perm,first,minibatch,1,d_return.get(),mb_return.get(),stream.value);
                actor.forward(mb_obs.get(),minibatch,stream.value);critic.forward(mb_obs.get(),minibatch,stream.value);CUDA_CHECK_THROW(cudaMemsetAsync(stats.get(),0,stats.size_bytes(),stream.value));
                rl::gaussian_ppo_loss_grad(actor.output(),sigma.get(),critic.output(),minibatch,4,mb_action.get(),mb_logp.get(),mb_adv.get(),mb_return.get(),adv_stats.get(),{.2f,0.f,.5f},d_mean.get(),d_std.get(),d_v.get(),row_stats.get(),stats.get(),stream.value);
                actor.zero_grad(stream.value);critic.zero_grad(stream.value);actor.backward(d_mean.get(),minibatch,nullptr,stream.value);critic.backward(d_v.get(),minibatch,nullptr,stream.value);
                finite(download(actor.grads(),actor.num_params(),stream.value));finite(download(critic.grads(),critic.num_params(),stream.value));actor_opt.step(actor.params(),actor.grads(),stream.value);critic_opt.step(critic.params(),critic.grads(),stream.value);
            }}
            finite(actor.get_params());finite(critic.get_params());++completed;require(actor_opt.steps()==completed*4&&critic_opt.steps()==completed*4,"Optimizer count differs");snapshot(root,completed,actor,critic,actor_opt,critic_opt,state,task,episodes);
            float mean_reward=0;for(float reward:rewards)mean_reward+=reward/rows;progress<<completed<<','<<budget.charged[1]<<','<<mean_reward<<','<<actor_opt.steps()<<','<<critic_opt.steps()<<'\n';progress.flush();std::cout<<"PPO update "<<completed<<"/256; training controls "<<budget.charged[1]<<"/131072\n";
        }
        after=evaluate(actor.get_params(),root,"after",budget,stream);
    }catch(const std::exception& error){fault=error.what();}
    std::ofstream out(root/"summary.json");out<<std::setprecision(9)<<"{\"scope\":\"bounded forward pilot; no runtime adoption\",\"fault\":\""<<fault<<"\",\"readiness_passed\":"<<(readiness_passed?"true":"false")<<",\"PPO_updates\":"<<completed<<",\"charged_controls\":"<<budget.total()<<",\"charged_native_slots\":"<<budget.total()*10<<",\"actual_native_substeps\":"<<budget.actual_substeps<<",\"stage_controls\":["<<budget.charged[0]<<','<<budget.charged[1]<<','<<budget.charged[2]<<"],\"control_cap\":131744,\"before_successes\":"<<before.successes<<",\"after_successes\":"<<after.successes<<",\"before_mean_forward_m\":"<<before.mean_forward<<",\"after_mean_forward_m\":"<<after.mean_forward<<",\"before_mean_error_m\":"<<before.mean_error<<",\"after_mean_error_m\":"<<after.mean_error<<",\"wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"closed\":true,\"retries\":0}\n";
    std::cout<<"Pilot closed: updates="<<completed<<", controls="<<budget.total()<<"/131744, success="<<after.successes<<"/4, fault="<<fault<<'\n';return fault.empty()?0:3;
}
// Fixed prescribed diagnostic. It never constructs a policy or optimizer.
std::vector<uint8_t> stroke_binding(){cp::Writer out;out.value(uint32_t{0x53545231});out.value(uint32_t{1});out.value(cudarl::contact::kPhysicsVersion);
    for(int x:{160,2,2,640,6400})out.value(x);
    for(float x:{.05f,.005f,.5f,.5f,2.f,8.f,.01f,.0001f,5.f,2e-4f,2e-3f,2e-6f})out.value(x);return out.data;}
void prepare_stroke(const fs::path& root){fs::create_directories(root);require(!fs::exists(root/"protocol.bin")&&!fs::exists(root/"run-started"),"Diagnostic already prepared/attempted");cp::write_atomic((root/"protocol.bin").string(),stroke_binding());
    std::ofstream out(root/"protocol.json");out.exceptions(std::ios::badbit|std::ios::failbit);
    out<<"{\"scope\":\"fixed prescribed stroke versus zero-target rest, NOT learned\",\"physics\":2,\"controls_max\":640,\"native_slots_max\":6400,\"seconds_each\":8,\"frequency_Hz\":0.5,\"amplitude_rad\":0.5,\"tail_phase_rad\":-1.5707963267948966,\"ramp_seconds_each_end\":2,\"sample\":\"control interval start, held 50ms\",\"start\":\"default centered straight stationary BodyState, y=-2 model units\",\"backends\":\"CPU and CUDA, each stroke and rest\",\"clearance_m\":0.01,\"measurable_net_forward_m\":0.0001,\"backend_margin_factor\":5,\"positive_interior_cycles_required\":2,\"retries\":0,\"PPO_updates\":0}\n";}
struct StrokeStats {float lo[4]{},hi[4]{},tracking_sum[4]{},peak_microNm=0,requested_peak_microNm=0;};
int run_stroke(const fs::path& root){require(cp::read((root/"protocol.bin").string())==stroke_binding(),"Frozen stroke protocol differs");require(fs::create_directory(root/"run-started"),"Diagnostic already attempted; no retry");
    Budget budget;std::string fault;const auto start=std::chrono::steady_clock::now();int completed=0;float maximum_parity=0;
    std::vector<s::WorldState> cpu(2),gpu(2);for(auto& w:cpu)w.paused=false;gpu=cpu;
    const auto origin=m::motion(cpu[0].body).com_model;StrokeStats stats[2][2];p::V net[2][5]{};
    try{Stream stream;int devices=0;CUDA_CHECK_THROW(cudaGetDeviceCount(&devices));require(devices>0,"Real CUDA required");
        std::ofstream trace(root/"stroke.csv");trace.exceptions(std::ios::badbit|std::ios::failbit);trace<<std::setprecision(10)<<"backend,condition,tick,time_s,com_x_m,com_y_m,com_z_m,target0,target2,j0,j1,j2,j3,actual_peak_microNm,positive_work_J,min_floor_m,min_other_m\n";
        for(int tick=0;tick<160;++tick){const float t=float(tick)*p::dt,pi=3.14159265358979323846f;
            float envelope=1;if(t<2)envelope=sinf(pi*t/4)*sinf(pi*t/4);else if(t>6)envelope=sinf(pi*(8-t)/4)*sinf(pi*(8-t)/4);
            std::vector<std::array<float,4>> command(2);command[0]={.5f*envelope*sinf(pi*t),0,.5f*envelope*sinf(pi*t-pi/2),0};
            advance(cpu,command,2,false,stream.value,budget);advance(gpu,command,2,true,stream.value,budget);
            for(int condition=0;condition<2;++condition){const auto a=fields(cpu[condition]),b=fields(gpu[condition]);finite(a);finite(b);require(a.size()==b.size(),"Defined field size mismatch");
                for(size_t j=0;j<a.size();++j){const float score=fabsf(a[j]-b[j])/(2e-4f+2e-3f*fabsf(a[j]));maximum_parity=fmaxf(maximum_parity,score);require(score<=1,"Stroke CPU/CUDA mismatch");}
                const auto& x=cpu[condition];const auto& y=gpu[condition];
                require(x.substeps==y.substeps&&x.control_steps==y.control_steps&&x.time_seconds==y.time_seconds&&x.invalid==y.invalid&&x.unsafe==y.unsafe,"Stroke backend flags/count mismatch");
                for(const auto pair:{std::array<float,2>{x.telemetry.minimum_floor_gap,y.telemetry.minimum_floor_gap},std::array<float,2>{x.telemetry.minimum_other_gap,y.telemetry.minimum_other_gap}})
                    require(fabsf(pair[0]-pair[1])<=2e-6f+2e-3f*fabsf(pair[0]),"Stroke backend clearance mismatch");
            }
            for(int backend=0;backend<2;++backend){const auto& state=backend?gpu:cpu;
                for(int condition=0;condition<2;++condition){const auto& w=state[condition];auto& v=stats[backend][condition];float local_peak=0;
                    for(float x:w.actual_torques)local_peak=fmaxf(local_peak,fabsf(x)*10);v.peak_microNm=fmaxf(v.peak_microNm,local_peak);
                    for(float x:w.requested_torques)v.requested_peak_microNm=fmaxf(v.requested_peak_microNm,fabsf(x)*1e6f);
                    for(int j=0;j<4;++j){v.lo[j]=fminf(v.lo[j],w.body.angle[j]);v.hi[j]=fmaxf(v.hi[j],w.body.angle[j]);const float error=w.held_targets[j]-w.body.angle[j];v.tracking_sum[j]+=error*error;}
                    const auto com=m::motion(w.body).com_model*p::scale;
                    trace<<(backend?"CUDA":"CPU")<<','<<(condition?"rest":"stroke")<<','<<tick+1<<','<<w.time_seconds<<','<<com.x<<','<<com.y<<','<<com.z<<','<<w.held_targets[0]<<','<<w.held_targets[2];for(float x:w.body.angle)trace<<','<<x;
                    trace<<','<<local_peak<<','<<w.body.work<<','<<w.telemetry.minimum_floor_gap<<','<<w.telemetry.minimum_other_gap<<'\n';
                }
                if((tick+1)%40==0){const auto stroke=m::motion(state[0].body).com_model,rest=m::motion(state[1].body).com_model;net[backend][(tick+1)/40]=(stroke-rest)*p::scale;}
            }++completed;
        }
    }catch(const std::exception& error){fault=error.what();}
    const float backend_gap=fabsf(net[0][4].x-net[1][4].x);bool measurable=fault.empty();
    for(int backend=0;backend<2;++backend)measurable=measurable&&net[backend][4].x>.0001f&&net[backend][4].x>5*backend_gap&&net[backend][2].x>net[backend][1].x&&net[backend][3].x>net[backend][2].x;
    std::ofstream out(root/"summary.json");out<<std::setprecision(10)<<"{\"scope\":\"prescribed diagnostic, NOT learned\",\"fault\":"<<std::quoted(fault)<<",\"completed_ticks\":"<<completed<<",\"controls\":"<<budget.total()<<",\"charged_native_slots\":"<<budget.total()*10<<",\"actual_native_substeps\":"<<budget.actual_substeps<<",\"control_cap\":640,\"max_backend_score\":"<<maximum_parity<<",\"net_forward_backend_difference_m\":"<<backend_gap<<",\"measurable_forward\":"<<(measurable?"true":"false")<<",\"states\":[";
    for(int backend=0;backend<2;++backend){const auto& state=backend?gpu:cpu;for(int condition=0;condition<2;++condition){if(backend||condition)out<<',';const auto& w=state[condition];const auto delta=(m::motion(w.body).com_model-origin)*p::scale;const auto& v=stats[backend][condition];
        out<<"{\"backend\":"<<std::quoted(backend?"CUDA":"CPU")<<",\"condition\":"<<std::quoted(condition?"rest":"stroke")<<",\"com_displacement_m\":["<<delta.x<<','<<delta.y<<','<<delta.z<<"],\"peak_microNm\":"<<v.peak_microNm<<",\"requested_peak_microNm\":"<<v.requested_peak_microNm<<",\"positive_work_J\":"<<w.body.work<<",\"joints\":[";
        for(int j=0;j<4;++j){if(j)out<<',';out<<"{\"min_rad\":"<<v.lo[j]<<",\"max_rad\":"<<v.hi[j]<<",\"control_end_tracking_RMS_rad\":"<<sqrtf(v.tracking_sum[j]/float(completed?completed:1))<<'}';}out<<"]}";
    }}out<<"],\"net_cycle_forward_m\":[";
    for(int backend=0;backend<2;++backend){if(backend)out<<',';out<<'[';for(int cycle=1;cycle<=4;++cycle){if(cycle>1)out<<',';out<<net[backend][cycle].x-net[backend][cycle-1].x;}out<<']';}
    out<<"],\"wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"closed\":true,\"retries\":0,\"PPO_updates\":0}\n";
    std::cout<<"Stroke diagnostic closed: controls="<<budget.total()<<"/640, net CPU="<<net[0][4].x*1000<<"mm, CUDA="<<net[1][4].x*1000<<"mm, measurable="<<measurable<<", fault="<<fault<<'\n';return fault.empty()?0:3;
}
}
int main(int argc,char** argv){try{require(argc==3,"Use --prepare/--run or --prepare-stroke/--stroke DIR");const fs::path root=fs::absolute(argv[2]);const std::string mode=argv[1];if(mode=="--prepare-stroke"){prepare_stroke(root);return 0;}if(mode=="--stroke")return run_stroke(root);if(mode=="--prepare"){prepare(root);return 0;}require(mode=="--run","Unknown mode");return run(root);}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
