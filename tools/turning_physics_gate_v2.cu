// New matched-command physics gate. Preparation performs no body integration.
#include "motor/turning_task_v2.h"
#include "motor/turning_policy_v2.h"
#include "motor/hold_assessment.h"
#include "rl/append_hold_checkpoint.h"
#include "hold_learned_amplitude_math.h"
#include "turning_study_authorization.h"
#include "nn/mlp_cpu.h"
#include "nn/mlp_gpu.h"
#include "cuda/cuda_memory.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace turning_gate_v2 {
namespace fs=std::filesystem;namespace p=cudarl::physics;namespace cp=cudarl::checkpoint;
namespace old=cudarl::straight_motor;namespace motor=cudarl::joint_motor;namespace model=cudarl::movement;
namespace task=cudarl::turning_task_v2;namespace policy=cudarl::turning_policy_v2;namespace nn=cudarl::nn;
constexpr uint64_t controls_cap=2480,substeps_cap=24800;
constexpr uint32_t prepared_magic=0x325a4754,passed_magic=0x32504754;
struct Step {motor::World world{};model::Telemetry telemetry{};p::BodyState before{},candidate{};int advanced=0;task::StepInfo info{};task::Task context{};};
struct Probe {motor::World world{};task::Task task{};float observation[41]{},raw[20]{};int valid=0;};
struct Decoded {float base[4]{},latent[4]{},command[4]{};};
struct GateCase {int group=0,id=0;};
struct Counts {uint64_t controls=0,substeps=0;};
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
nn::MlpSpec spec(){return {{41,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}};}
uint64_t protocol(const fs::path& root){return cp::checksum(cp::read((root/"protocol.bin").string()));}
std::vector<float> load_backbone(const fs::path& root) {
    const auto data=cudarl::rl::read_ppo_training_state(cp::read((root/"parent-checkpoint.bin").string()),
        cp::checksum(cp::read((root/"parent-protocol.bin").string())));
    require(data.actor.update==336,"Wrong fixed swimming parent");
    return cudarl::rl::append_zero_input_column(data.actor.parameters);
}
float load_hold(const fs::path& root,const std::vector<float>& backbone) {
    const auto bytes=cp::read((root/"holding-parent.bin").string());cp::Reader in{bytes};
    require(in.value<uint32_t>()==0x314d4148&&in.value<uint32_t>()==1,"Wrong holding parent schema");
    require(in.value<uint64_t>()!=0,"Missing holding protocol");
    require(in.value<uint64_t>()==cp::checksum(cp::read((root/"parent-checkpoint.bin").string())),"Wrong holding parent binding");
    for(uint32_t v:{6u,6u,3u,4u,41u,4u,336u})require(in.value<uint32_t>()==v,"Wrong holding parent contract");
    require(in.value<int>()==7&&in.value<int>()==112,"Wrong holding learning history");
    require(in.array<float>(2532)==backbone,"Holding backbone differs");
    const auto head=in.array<float>(2);const auto gain=in.value<float>(),sigma=in.value<float>();(void)in.array<uint8_t>();in.finish();
    require(head.size()==2&&gain==cudarl::learned_amplitude::sigmoid(head[0])&&sigma==cudarl::learned_amplitude::standard_deviation(head[1]),"Holding map differs");
    return gain;
}
old::ForwardConfig config(const GateCase& c) {
    auto out=old::config_for_speed(.008f);out.observation_schema=4;
    if(c.group==1){out.maximum_controls=1280;out.stationary_request=true;out.position_hold=true;out.target_speed_m_s=0;}
    return out;
}
Step initialize(const GateCase& c,old::ForwardTask& fixed,task::Task& free) {
    Step out;
    if(c.group==2){require(task::initialize(c.id,out.world,free),"Free gate initialization rejected");out.context=free;return out;}
    out.world=old::initial_world(c.group==0?c.id:c.id%2?3:0);out.world.arena=old::long_swim_arena();
    require(old::begin_forward(out.world,config(c),fixed),"Fixed gate task rejected");
    if(c.group==1){const float offset=c.id/2==0?0:c.id/2==1?-.0005f:.0005f;
        require(old::offset_episode_start(out.world,fixed,config(c),offset)&&old::update_velocity_request(out.world,config(c),fixed),"Holding fixture rejected");}
    // Fixed routes keep their original depth reference, including the original
    // pre-offset holding anchor. The context never replaces the old task.
    out.context.fixed_depth_model=fixed.start_com_model.y;
    return out;
}
std::vector<GateCase> cases(){std::vector<GateCase> out;for(int i=0;i<4;++i)out.push_back({0,i});for(int i=0;i<6;++i)out.push_back({1,i});for(int i=0;i<16;++i)out.push_back({2,i});return out;}
__global__ void pure_initial(Probe* output) {
    const int i=threadIdx.x;if(i>=24)return;auto& out=output[i];
    out.valid=task::initialize(i<16?i:i-16,out.world,out.task,1200,i>=16)&&task::observe(out.task,out.world,out.observation,out.raw);
}
__global__ void mapped(const float* means,const float* raw,const float* base,Decoded* output,int n,bool autonomous,bool holding,float g) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;auto& out=output[i];
    for(int j=0;j<4;++j)out.base[j]=base[i*4+j];
    policy::motor_latents(autonomous,holding,means,out.base,g,raw+i*20,out.latent);
    for(int j=0;j<4;++j)out.command[j]=tanhf(out.latent[j]);
}
__global__ void advance(Step* state,const float* command) {
    state->advanced=task::step_control(state->world,state->context,command,state->info);
    state->telemetry=state->info.telemetry;state->before=state->info.before;state->candidate=state->info.candidate;
}
Decoded decode(const std::array<float,80>& means,const float* raw,const float* base,bool autonomous,bool holding,float gain) {
    Decoded out;for(int j=0;j<4;++j)out.base[j]=base[j];
    policy::motor_latents(autonomous,holding,means.data(),out.base,gain,raw,out.latent);
    for(int j=0;j<4;++j)out.command[j]=tanhf(out.latent[j]);return out;
}
void pointwise(const Decoded& cpu,const Decoded& gpu,double& max_base,double& max_latent,double& max_command) {
    for(int j=0;j<4;++j){const double b=std::abs(double(cpu.base[j])-gpu.base[j]),l=std::abs(double(cpu.latent[j])-gpu.latent[j]),c=std::abs(double(cpu.command[j])-gpu.command[j]);
        require(std::isfinite(b)&&std::isfinite(l)&&std::isfinite(c),"Nonfinite same-observation inference");
        max_base=std::max(max_base,b);max_latent=std::max(max_latent,l);max_command=std::max(max_command,c);
        require(b<=2e-5&&l<=2e-5&&c<=2e-5,"Same-observation backbone/mapped command mismatch");}
}

std::vector<uint8_t> defined_state(const motor::World& w, const cudarl::movement::Telemetry& t) {
    cp::Writer o;const auto& s = w.physical;const auto& b = s.body;
    for (float v : {b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w}) o.value(v);
    for (float v : b.angle) o.value(v);for (float v : b.velocity) o.value(v);o.value(b.work);o.value(b.travel);
    for (float v : {s.fluid.density,s.fluid.viscosity,s.fluid.gravity,s.current.x,s.current.y,s.current.z}) o.value(v);o.value(s.actuation);
    for (float v : s.gains.kp) o.value(v);for (float v : s.gains.kd) o.value(v);
    for (float v : s.held_targets) o.value(v);for (float v : s.last_actual_torques) o.value(v);for (float v : s.raw_action) o.value(v);
    for (float v : s.actual_torques) o.value(v);for (float v : s.requested_torques) o.value(v);o.value(s.mean_squared_torque);
    o.value(s.control_steps);o.value(s.substeps);o.value(s.time_seconds);o.value(s.paused);o.value(s.unsafe);o.value(s.invalid);
    const auto& a = s.telemetry.inspection;
    for (float v : {a.floor_gap,a.min_other_gap,a.min_gap,a.max_floor_gap}) o.value(v);
    for (bool v : {a.contact,a.floor_contact,a.excessive,a.escape,a.invalid}) o.value(v);
    for (float v : {s.telemetry.minimum_floor_gap,s.telemetry.minimum_other_gap,s.telemetry.floor_spring_force_max,s.telemetry.total_spring_force_max}) o.value(v);
    o.value(s.telemetry.contact_samples);o.value(s.telemetry.executed_substeps);o.value(s.telemetry.first_unsafe_substep);
    o.value(w.arena.half_width_m);o.value(w.arena.half_depth_m);for (float v : w.reference_rate) o.value(v);o.value(w.attempted_substeps);
    for (double v : {w.signed_work,w.positive_work,w.fluid_loss,w.numerical_loss,w.external_work,w.last_energy_error}) o.value(v);o.value(w.failure_code);
    for (float v : t.torque) o.value(v);for (float v : t.requested) o.value(v);o.value(t.minimum_gap);o.value(t.solve_residual);o.value(t.failure_code);
    for (double v : {t.signed_work,t.positive_work,t.fluid_loss,t.external_work,t.numerical_loss}) o.value(v);
    return o.data;
}
void same_zero(const Step& old_state, const Step& new_state) {
    require(defined_state(old_state.world,old_state.telemetry) == defined_state(new_state.world,new_state.telemetry),"Zero-current defined world or base telemetry changed");
}
void cpu_cuda(const motor::World& cpu, const motor::World& gpu, double& maximum_pose) {
    const auto& a = cpu.physical.body;const auto& b = gpu.physical.body;
    for (double value : {double(std::abs(a.position.x-b.position.x)),double(std::abs(a.position.y-b.position.y)),double(std::abs(a.position.z-b.position.z))}) {
        maximum_pose = std::max(maximum_pose,value);require(value <= 2e-5,"CPU/CUDA root position mismatch");
    }
    for (float value : {std::abs(a.orientation.w-b.orientation.w),std::abs(a.orientation.x-b.orientation.x),std::abs(a.orientation.y-b.orientation.y),std::abs(a.orientation.z-b.orientation.z)}) require(value <= 2e-5f,"CPU/CUDA quaternion mismatch");
    for (int j = 0; j < 4; ++j) require(std::abs(a.angle[j]-b.angle[j]) <= 2e-5f,"CPU/CUDA joint angle mismatch");
    for (int j = 0; j < 10; ++j) require(std::abs(a.velocity[j]-b.velocity[j]) <= (j < 3 ? 2e-7f : 2e-5f),"CPU/CUDA generalized velocity mismatch");
    for(int j=0;j<40;++j)require(std::abs(cpu.physical.actual_torques[j]-gpu.physical.actual_torques[j])<=2e-5f,"CPU/CUDA actual normalized torque mismatch");
    for(int j=0;j<4;++j){
        require(std::abs(cpu.physical.last_actual_torques[j]-gpu.physical.last_actual_torques[j])<=2e-5f,"CPU/CUDA last actual torque mismatch");
        require(std::abs(cpu.physical.held_targets[j]-gpu.physical.held_targets[j])<=2e-5f,"CPU/CUDA held target mismatch");
        require(std::abs(cpu.reference_rate[j]-gpu.reference_rate[j])<=2e-5f,"CPU/CUDA reference rate mismatch");
    }
    const double ce[6] = {cpu.signed_work,cpu.positive_work,cpu.fluid_loss,cpu.external_work,cpu.numerical_loss,cpu.last_energy_error};
    const double ge[6] = {gpu.signed_work,gpu.positive_work,gpu.fluid_loss,gpu.external_work,gpu.numerical_loss,gpu.last_energy_error};
    for (int i = 0; i < 6; ++i) require(std::abs(ce[i]-ge[i]) <= 1e-12+1e-4*std::max(std::abs(ce[i]),std::abs(ge[i])),"CPU/CUDA cumulative energy mismatch");
}

void prepared_binding(cp::Writer& out,const fs::path& root) {
    out.value(prepared_magic);out.value(uint32_t{2});out.value(protocol(root));
    out.value(cp::checksum(cp::read((root/"parent-checkpoint.bin").string())));
    out.value(cp::checksum(cp::read((root/"holding-parent.bin").string())));
}
int pure_task_checks() {
    // Synthetic measurement records only: no step_control, body.substep,
    // integrator, optimizer or kernel is invoked by these checks.
    int checks=0;
    const auto check=[&](bool value,const char* why){require(value,why);++checks;};
    task::World world;task::Task state;
    check(task::initialize(0,world,state,0),"Synthetic task fixture rejected");
    float raw[20]{},features[20]{};
    check(task::raw_sensors(state,world,raw),"Synthetic twenty-sensor fixture rejected");
    policy::derived_features(raw,features);
    for(int k=8;k<20;++k)check(features[k]==tanhf(raw[k]),"Additional feature normalization changed");
    std::array<float,80> zero{};float base[4]={.1f,-.2f,.3f,-.4f},out[4]{};
    policy::motor_latents(true,false,zero.data(),base,.26211655139923096f,raw,out);
    check(std::equal(out,out+4,base),"Zero eighty-parameter head changed the backbone");
    auto varied=zero;for(int i=0;i<80;++i)varied[i]=.2f*sinf(float(i+1));
    policy::motor_latents(false,false,varied.data(),base,.26211655139923096f,raw,out);
    check(std::equal(out,out+4,base),"Explicit Swim changed");
    policy::motor_latents(true,true,varied.data(),base,.26211655139923096f,raw,out);
    for(int j=0;j<4;++j)check(out[j]==.26211655139923096f*base[j],"Saved Rest changed");
    policy::motor_latents(true,false,varied.data(),base,.26211655139923096f,raw,out);
    for(int j=0;j<4;++j)check(std::isfinite(out[j])&&std::abs(out[j]-base[j])<=.7500001f,"All-axis correction exceeded its bound");

    auto depth=world.physical.body;depth.position.y+=.0051f/p::scale;
    check(task::substep_behavior_reason(state,depth,task::bounds(depth))==task::Depth,"Five-millimeter task depth guard missing");
    auto tilted=world.physical.body;tilted.orientation=p::axis_rotation({0,0,1},25*model::pi/180.f);
    tilted.position.y+=(state.fixed_depth_model-cudarl::motor::motion(tilted).com_model.y);
    check(task::substep_behavior_reason(state,tilted,task::bounds(tilted))==task::Tilt,"Twenty-degree task tilt guard missing");
    auto boundary=task::bounds(depth);boundary.inset_radius_squared=1;
    check(task::substep_behavior_reason(state,depth,boundary)==task::Boundary,"Behavior guard priority changed");

    task::StepInfo info;info.valid=true;info.accepted_substeps=info.substeps=info.attempted_substeps=10;
    info.seconds=info.dt=10*.005f;
    world.physical.control_steps=1;
    auto transition=task::record(state,world,info);
    check(transition.accepted&&!transition.done&&std::abs(transition.reward+2*info.seconds)<1e-6,"Stationary reward no longer penalizes lost swimming");
    bool premature=false;
    for(int i=2;i<=80;++i){world.physical.control_steps=static_cast<uint64_t>(i);transition=task::record(state,world,info);if(i<80)premature=premature||!transition.accepted||transition.done;}
    check(!premature,"Stall warmup ended before nominal control80");
    check(transition.accepted&&transition.done&&transition.behavior_terminal&&transition.reason==task::Stalled&&state.controls==80,"Rolling two-second stall terminal missing");
    check(std::abs(transition.reward-(-2*info.seconds-4))<1e-6,"Stall penalty was not applied once");
    const auto reward=state.reward_sum;
    check(!task::record(state,world,info).accepted&&state.reward_sum==reward,"Terminal synthetic task accepted another control");

    check(task::initialize(0,world,state,0),"Partial synthetic fixture rejected");
    info.accepted_substeps=info.substeps=info.attempted_substeps=3;info.seconds=info.dt=3*.005f;
    info.behavior_terminal=true;info.terminal_reason=task::Depth;world.physical.body=depth;world.physical.control_steps=1;
    info.positive_work_delta=double(.000001f);
    transition=task::record(state,world,info);
    check(transition.accepted&&transition.done&&transition.reason==task::Depth&&transition.substeps==3&&state.elapsed_seconds==double(info.seconds),"Partial behavior terminal lost accepted time/counters");
    const double rate= -2-.05*std::pow(double(transition.metrics.depth_error_m/.001f),2);
    check(std::abs(transition.reward-(info.seconds*rate-6250*info.positive_work_delta-4))<2e-6,"Partial recorded-work reward reconstruction differs");
    return checks;
}
int prepare(const fs::path& root) {
    require(fs::is_directory(root),"Prepare learner inputs before gate preparation");
    require(fs::create_directory(root/"gate-preparation"),"Pure gate preparation identity exists");
    const int measurement_checks=pure_task_checks();
    const auto weights=load_backbone(root);const float gain=load_hold(root,weights);
    std::vector<Probe> cpu(24),gpu(24);
    for(int i=0;i<24;++i)cpu[i].valid=task::initialize(i<16?i:i-16,cpu[i].world,cpu[i].task,1200,i>=16)&&task::observe(cpu[i].task,cpu[i].world,cpu[i].observation,cpu[i].raw);
    device_ptr<Probe> probes(24);pure_initial<<<1,32>>>(probes.get());CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());probes.copy_to_host(gpu.data(),gpu.size());
    cp::Writer raw;raw.value(uint32_t{2});raw.value(uint32_t{sizeof(Probe)});raw.array(cpu);raw.array(gpu);
    cp::write_atomic((root/"gate-preparation/initial-probes.bin").string(),raw.data);
    double max_pose=0,max_observation=0,max_sensor=0;
    std::vector<float> observations(24*41),sensors(24*20);std::array<float,80> means{};
    for(int j=0;j<80;++j)means[j]=.18f*sinf(float(j+1));
    for(int i=0;i<24;++i){require(cpu[i].valid&&gpu[i].valid,"CPU/CUDA initial free fixture invalid");cpu_cuda(cpu[i].world,gpu[i].world,max_pose);
        require(task::bounds(cpu[i].world.physical.body).inset_radius_squared<1,"Fixture starts in behavioral buffer");
        for(int j=0;j<41;++j){max_observation=std::max(max_observation,std::abs(double(cpu[i].observation[j])-gpu[i].observation[j]));observations[i*41+j]=cpu[i].observation[j];}
        for(int j=0;j<20;++j){max_sensor=std::max(max_sensor,std::abs(double(cpu[i].raw[j])-gpu[i].raw[j]));sensors[i*20+j]=cpu[i].raw[j];}}
    require(max_observation<=2e-5&&max_sensor<=2e-5,"Pure CPU/CUDA observation/sensor mismatch");
    nn::CpuMlp<float> cpu_body(spec());cpu_body.params()=weights;nn::GpuMlp gpu_body(spec(),24,1);gpu_body.set_params(weights);
    device_ptr<float> d_obs(observations.size()),d_raw(sensors.size()),d_means(80);device_ptr<Decoded> d_decoded(24);
    d_obs.copy_from_host(observations.data(),observations.size());d_raw.copy_from_host(sensors.data(),sensors.size());d_means.copy_from_host(means.data(),80);
    gpu_body.forward(d_obs.get(),24);mapped<<<1,32>>>(d_means.get(),d_raw.get(),gpu_body.output(),d_decoded.get(),24,true,false,gain);
    CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());std::vector<Decoded> gpu_decode(24);d_decoded.copy_to_host(gpu_decode.data(),24);
    double max_base=0,max_latent=0,max_command=0;
    for(int i=0;i<24;++i){cpu_body.forward(observations.data()+i*41,1);pointwise(decode(means,sensors.data()+i*20,cpu_body.output(),true,false,gain),gpu_decode[i],max_base,max_latent,max_command);}
    std::ofstream result(root/"gate-preparation/result.json");result<<std::setprecision(17)
        <<"{\"passed\":true,\"CPU_initial_fixtures\":24,\"CUDA_initial_fixtures\":24,\"same_observation_NN_cases\":24,\"physical_controls\":0,\"physical_substeps\":0,\"optimizer_steps\":0"
        <<",\"maximum_initial_position_model\":"<<max_pose<<",\"maximum_observation_difference\":"<<max_observation<<",\"maximum_raw_sensor_difference\":"<<max_sensor
        <<",\"maximum_backbone_difference\":"<<max_base<<",\"maximum_latent_difference\":"<<max_latent<<",\"maximum_command_difference\":"<<max_command
        <<",\"synthetic_measurement_checks\":"<<measurement_checks<<",\"synthetic_measurements_are_physics\":false}\n";
    result.close();require(bool(result),"Cannot write pure gate result");
    cp::Writer marker;prepared_binding(marker,root);cp::write_atomic((root/"gate-prepared.bin").string(),marker.data);
    std::cout<<"Pure turning gate preparation passed; zero body controls\n";return 0;
}
void capture(cp::Writer& raw,std::ofstream& csv,const GateCase& c,int backend,int control,const Step& state) {
    raw.value(c.group);raw.value(c.id);raw.value(backend);raw.value(control);raw.value(state);
    const auto& w=state.world;const auto& b=w.physical.body;
    csv<<c.group<<','<<c.id<<','<<backend<<','<<control<<','<<b.position.x<<','<<b.position.y<<','<<b.position.z
       <<','<<b.orientation.x<<','<<b.orientation.y<<','<<b.orientation.z<<','<<b.orientation.w;
    for(float v:b.angle)csv<<','<<v;for(float v:b.velocity)csv<<','<<v;
    csv<<','<<w.attempted_substeps<<','<<w.physical.substeps<<','<<w.signed_work<<','<<w.positive_work<<','<<w.fluid_loss
       <<','<<w.external_work<<','<<w.numerical_loss<<','<<w.last_energy_error<<','<<state.advanced<<','<<state.info.substeps<<','<<state.info.boundary_terminal<<'\n';
}
int run(const fs::path& root) {
    const auto started=std::chrono::steady_clock::now();cp::Writer expected;prepared_binding(expected,root);
    cudarl::study_authorization::require_approved(root,protocol(root));
    require(cp::read((root/"gate-prepared.bin").string())==expected.data,"Pure gate binding missing or changed");
    require(!fs::exists(root/"gate-result.json")&&fs::create_directory(root/"gate-started"),"Physical gate already attempted; no retry");
    Counts counts;std::string fault;int completed=0;double max_pose=0,max_base=0,max_latent=0,max_command=0;
    Step cpu{},gpu{},historical{};GateCase active{};int active_control=0;cp::Writer raw;std::ofstream csv,commands;
    const auto output=root/"gate-started";
    try {
        raw.value(uint32_t{2});raw.value(uint32_t{sizeof(Step)});raw.value(uint32_t{sizeof(motor::World)});raw.value(uint32_t{sizeof(task::StepInfo)});
        csv.open(output/"states.csv");csv<<std::setprecision(17)<<"group,case,backend,control,x,y,z,qx,qy,qz,qw,j0,j1,j2,j3,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,attempted_substeps,accepted_substeps,signed_work,positive_work,fluid_loss,external_work,numerical_loss,energy_error,advanced,last_substeps,boundary_terminal\n";
        commands.open(output/"same-observation.csv");commands<<std::setprecision(17)<<"group,case,control";
        for(int j=0;j<41;++j)commands<<",observation"<<j;for(int j=0;j<20;++j)commands<<",raw"<<j;
        for(const char* prefix:{"cpu_base","gpu_base","cpu_latent","gpu_latent","cpu_command","gpu_command"})for(int j=0;j<4;++j)commands<<','<<prefix<<j;commands<<'\n';
        const auto weights=load_backbone(root);const float gain=load_hold(root,weights);
        nn::CpuMlp<float> cpu_body(spec());cpu_body.params()=weights;nn::GpuMlp gpu_body(spec(),1,1);gpu_body.set_params(weights);
        device_ptr<float> d_obs(41),d_raw(20),d_means(80),d_command(4);device_ptr<Decoded> d_decoded(1);device_ptr<Step> d_state(1);
        std::array<float,80> means{};d_means.copy_from_host(means.data(),80);
        for(const auto c:cases()) {
            active=c;old::ForwardTask fixed;task::Task free;cpu=initialize(c,fixed,free);gpu=historical=cpu;d_state.copy_from_host(&gpu,1);
            capture(raw,csv,c,1,0,cpu);capture(raw,csv,c,2,0,gpu);if(c.group<2)capture(raw,csv,c,0,0,historical);
            for(int control=1;control<=40;++control) {
                active_control=control;float obs[41]{},sensor[20]{};
                if(c.group==2)require(task::observe(free,cpu.world,obs,sensor),"Free gate observation failed");
                else {
                    if(c.group==1)require(old::update_velocity_request(cpu.world,config(c),fixed),"Hold gate request failed");
                    require(old::observe_appended(fixed,cpu.world,config(c),obs),"Fixed gate observation failed");
                }
                cpu_body.forward(obs,1);const auto chosen=decode(means,sensor,cpu_body.output(),c.group==2,c.group==1,gain);
                d_obs.copy_from_host(obs,41);d_raw.copy_from_host(sensor,20);gpu_body.forward(d_obs.get(),1);
                mapped<<<1,1>>>(d_means.get(),d_raw.get(),gpu_body.output(),d_decoded.get(),1,c.group==2,c.group==1,gain);
                CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());Decoded gpu_choice;d_decoded.copy_to_host(&gpu_choice,1);
                pointwise(chosen,gpu_choice,max_base,max_latent,max_command);
                commands<<c.group<<','<<c.id<<','<<control;for(float v:obs)commands<<','<<v;for(float v:sensor)commands<<','<<v;
                const float* row_values[6]={chosen.base,gpu_choice.base,chosen.latent,gpu_choice.latent,chosen.command,gpu_choice.command};
                for(const auto* values:row_values)for(int j=0;j<4;++j)commands<<','<<values[j];commands<<'\n';
                if(c.group<2) {
                    const auto before=historical.world.attempted_substeps,accepted_before=historical.world.physical.substeps;++counts.controls;
                    historical.advanced=motor::step_state(historical.world,chosen.command,&historical.telemetry,&historical.before,&historical.candidate);
                    historical.info={};historical.info.substeps=historical.info.attempted_substeps=static_cast<int>(historical.world.attempted_substeps-before);
                    historical.info.accepted_substeps=static_cast<int>(historical.world.physical.substeps-accepted_before);
                    historical.info.seconds=historical.info.dt=historical.info.accepted_substeps*.005f;
                    historical.info.valid=historical.advanced!=0;historical.info.hard_fault=historical.advanced==0;historical.info.failure_code=historical.world.failure_code;
                    historical.info.before=historical.before;historical.info.candidate=historical.candidate;historical.info.telemetry=historical.telemetry;
                    counts.substeps+=historical.world.attempted_substeps-before;capture(raw,csv,c,0,control,historical);
                    require(historical.advanced,"Legacy matched-command control failed");
                }
                const auto before=cpu.world.attempted_substeps;++counts.controls;
                cpu.advanced=task::step_control(cpu.world,cpu.context,chosen.command,cpu.info);cpu.telemetry=cpu.info.telemetry;cpu.before=cpu.info.before;cpu.candidate=cpu.info.candidate;
                counts.substeps+=cpu.world.attempted_substeps-before;capture(raw,csv,c,1,control,cpu);
                require(cpu.advanced&&cpu.info.accepted_substeps==10&&!cpu.info.behavior_terminal,"New CPU matched-command control failed/terminated");
                const auto gpu_before=gpu.world.attempted_substeps;++counts.controls;
                d_command.copy_from_host(chosen.command,4);advance<<<1,1>>>(d_state.get(),d_command.get());
                CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());d_state.copy_to_host(&gpu,1);
                counts.substeps+=gpu.world.attempted_substeps-gpu_before;capture(raw,csv,c,2,control,gpu);
                require(gpu.advanced&&gpu.info.accepted_substeps==10&&!gpu.info.behavior_terminal,"New CUDA matched-command control failed/terminated");
                if(c.group<2)same_zero(historical,cpu);cpu_cuda(cpu.world,gpu.world,max_pose);
                if(c.group==2)require(task::record(free,cpu.world,cpu.info).accepted,"Free gate task bookkeeping failed");
                else {const auto transition=c.group==1?old::record_hold_control(fixed,cpu.world,config(c)):old::record_control(fixed,cpu.world,config(c));require(transition.accepted,"Fixed gate task bookkeeping failed");}
                require(counts.controls<=controls_cap&&counts.substeps<=substeps_cap,"Gate physical budget exceeded");
            }++completed;
        }
        require(counts.controls==controls_cap&&counts.substeps==substeps_cap&&completed==26,"Gate final counters differ");
    } catch(const std::exception& e) {
        fault=e.what();cp::Writer failure;failure.value(uint32_t{2});failure.value(active.group);failure.value(active.id);failure.value(active_control);
        failure.value(counts.controls);failure.value(counts.substeps);failure.value(cpu);failure.value(gpu);failure.value(historical);
        cp::write_atomic((output/"first-failure.bin").string(),failure.data);
    }
    if(!raw.data.empty())cp::write_atomic((output/"states.bin").string(),raw.data);
    csv.close();commands.close();if(fault.empty()&&(!csv||!commands))fault="Gate CSV evidence write failed";
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::ofstream time(output/"active-seconds.txt");time<<std::setprecision(17)<<seconds<<'\n';time.close();
    std::ofstream result(root/"gate-result.json");result<<std::setprecision(17)
      <<"{\"schema\":2,\"closed\":true,\"passed\":"<<(fault.empty()?"true":"false")<<",\"fault\":"<<std::quoted(fault)
      <<",\"controls\":"<<counts.controls<<",\"substeps\":"<<counts.substeps<<",\"completed_cases\":"<<completed<<",\"active_seconds\":"<<seconds
      <<",\"maximum_position_difference_model\":"<<max_pose<<",\"maximum_same_observation_backbone_difference\":"<<max_base
      <<",\"maximum_same_observation_latent_difference\":"<<max_latent<<",\"maximum_same_observation_command_difference\":"<<max_command
      <<",\"world_bytes\":"<<sizeof(motor::World)<<",\"step_info_bytes\":"<<sizeof(task::StepInfo)<<",\"step_bytes\":"<<sizeof(Step)
      <<",\"optimizer_steps\":0,\"matched_commands\":true,\"closed_loop_trajectory_parity_claimed\":false}\n";result.close();
    require(bool(result),"Cannot write gate result");
    if(fault.empty()){cp::Writer marker;marker.value(passed_magic);marker.value(uint32_t{2});marker.value(uint32_t{1});marker.value(counts.controls);marker.value(counts.substeps);marker.value(protocol(root));
        cp::write_atomic((root/"gate-passed.bin").string(),marker.data);}
    std::cout<<"Turning matched-command gate "<<(fault.empty()?"passed":"FAILED")<<" controls="<<counts.controls<<" substeps="<<counts.substeps<<" "<<fault<<'\n';
    return fault.empty()?0:3;
}
}
int main(int argc,char** argv){try{
    if(argc==3&&std::string(argv[1])=="--prepare")return turning_gate_v2::prepare(std::filesystem::absolute(argv[2]));
    if(argc==2)return turning_gate_v2::run(std::filesystem::absolute(argv[1]));
    throw std::runtime_error("Use --prepare RUN for zero integration, or RUN for the one physical gate");
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
