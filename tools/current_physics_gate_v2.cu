// Version 2 current gate. Compilation/pure assembly never authorizes body integration.
#include "motor/current_motor_v2.h"
#include "motor/current_model_v1.h"
#include "motor/straight_task.h"
#include "motor/hold_assessment.h"
#include "rl/append_hold_checkpoint.h"
#include "hold_learned_amplitude_math.h"
#include "nn/mlp_cpu.h"
#include "cuda/cuda_memory.h"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace gate_v2 {
namespace fs = std::filesystem;
namespace p = cudarl::physics;
namespace cp = cudarl::checkpoint;
namespace old = cudarl::straight_motor;
namespace motor = cudarl::current_motor_v2;
namespace model = cudarl::current_model_v2;
static_assert(model::kVersion == 2 && motor::kVersion == 2,"Current gate v2 requires explicit model/motor v2");
namespace nn = cudarl::nn;
constexpr uint32_t protocol_magic = 0x32504743, passed_magic = 0x32474343;
constexpr uint64_t controls_cap = 3760, substeps_cap = 37600;
struct Counts {uint64_t controls = 0, substeps = 0, completed = 0;};
struct Case {int group = 0, id = 0, pose = 0, controls = 40;float offset = 0;p::V current{};};
struct Step {motor::World world;model::Telemetry telemetry;p::BodyState before{},candidate{};int advanced = 0;};
struct Probe {p::BodyState body{};p::V current{};p::System forced{},negative{},comoving{},zero{},zero_reference{};model::FlowTerms terms{};double direct_dissipation = 0,direct_supply = 0;};
// Explicit reader for preserved V1 diagnostic records. No V1 state is adopted.
struct RecordedV1Step {cudarl::joint_motor::World world;cudarl::current_model_v1::Telemetry telemetry;p::BodyState before{},candidate{};int advanced=0;};
static_assert(sizeof(RecordedV1Step)==1000,"Preserved V1 diagnostic ABI changed");
struct NeutralInput {p::BodyState body{};p::Parameters fluid{};p::V current{};int source=0;};
struct NeutralOutput {p::System old_current{},new_current{},new_zero{},legacy_zero{};cudarl::current_model_v1::FlowTerms old_flow{};model::FlowTerms new_flow{};};
void require(bool ok, const char* why) {if (!ok) throw std::runtime_error(why);}
old::ForwardConfig config(const Case& c) {
    auto out = old::config_for_speed(.008f);out.observation_schema = 4;
    if (c.group == 1) {out.maximum_controls = 1280;out.stationary_request = true;out.position_hold = true;out.target_speed_m_s = 0;}
    return out;
}
const char* group_name(int group) {return group == 0 ? "zero_swim" : group == 1 ? "zero_hold" : group == 2 ? "flow_saved_motor" : "flow_zero_target_servo";}
std::vector<Case> cases() {
    std::vector<Case> out;
    for (int id = 0; id < 4; ++id) out.push_back({0,id,id,40,0,{}});
    for (int id = 0; id < 6; ++id) out.push_back({1,id,id%2 ? 3 : 0,40,id/2 == 0 ? 0.f : id/2 == 1 ? -.0005f : .0005f,{}});
    const p::V directions[4] = {{.0025f,0,0},{-.0025f,0,0},{0,0,.0025f},{0,0,-.0025f}};
    for (int group = 2; group <= 3; ++group)
        for (int pose : {0,3}) for (int d = 0; d < 4; ++d) out.push_back({group,(pose == 0 ? 0 : 4)+d,pose,80,0,directions[d]});
    return out;
}
motor::World initialize(const Case& c, old::ForwardTask& task) {
    auto world = old::initial_world(c.pose);world.arena = old::long_swim_arena();
    if (c.group == 3) {
        for (int j = 0; j < 4; ++j) {world.physical.body.angle[j] = 0;world.physical.held_targets[j] = 0;world.reference_rate[j] = 0;}
        for (float& value : world.physical.body.velocity) value = 0;
    }
    require(old::begin_forward(world,config(c),task),"Gate fixture task rejected");
    if (c.group == 1) {
        require(old::offset_episode_start(world,task,config(c),c.offset),"Gate hold offset rejected");
        require(old::update_velocity_request(world,config(c),task),"Gate hold request rejected");
    }
    world.physical.current = c.current;
    require(motor::in_domain(world),"Gate initial physical state rejected");
    return world;
}
__global__ void advance(Step* state, const float* command) {
    state->before = state->candidate = state->world.physical.body;
    state->advanced = motor::step_state(state->world,command,&state->telemetry,&state->before,&state->candidate) ? 1 : 0;
}
CUDARL_HD void assemble_probe(Probe& result) {
    // Pure Galilean probes isolate hydrodynamics. Physical gate worlds below
    // keep their original default fluid.gravity=9.81 and all physical limits.
    const p::Parameters hydro_only{1000,.001f,0};
    result.forced = model::assemble(result.body,hydro_only,result.current);
    auto negative = result.body;
    negative.velocity[0] = -result.current.x;negative.velocity[1] = -result.current.y;negative.velocity[2] = -result.current.z;
    result.negative = cudarl::movement::assemble(negative,hydro_only,true);
    auto comoving = result.body;
    comoving.velocity[0] = result.current.x;comoving.velocity[1] = result.current.y;comoving.velocity[2] = result.current.z;
    result.comoving = model::assemble(comoving,hydro_only,result.current,&result.terms);
    const double velocity[10] = {result.current.x,result.current.y,result.current.z,0,0,0,0,0,0,0};
    const auto energy = model::water_energy(comoving,hydro_only,result.current,velocity);
    result.direct_dissipation = energy.dissipation;result.direct_supply = energy.supply;
    result.zero = model::assemble(result.body,hydro_only,{});
    result.zero_reference = cudarl::movement::assemble(result.body,hydro_only,true);
}
__global__ void probe_assembly(Probe* probe) {assemble_probe(probe[threadIdx.x]);}
CUDARL_HD void assemble_neutral(const NeutralInput& input,NeutralOutput& out) {
    out.old_current=cudarl::current_model_v1::assemble(input.body,input.fluid,input.current,&out.old_flow);
    out.new_current=model::assemble(input.body,input.fluid,input.current,&out.new_flow);
    out.new_zero=model::assemble(input.body,input.fluid,{});
    out.legacy_zero=cudarl::movement::assemble(input.body,input.fluid,true);
}
__global__ void neutral_assembly(const NeutralInput* inputs,NeutralOutput* outputs,int n) {
    const int i=threadIdx.x;if(i<n)assemble_neutral(inputs[i],outputs[i]);
}
std::vector<NeutralInput> neutral_fixtures(const fs::path& root) {
    std::vector<NeutralInput> out;
    const p::V directions[4]={{.0025f,0,0},{-.0025f,0,0},{0,0,.0025f},{0,0,-.0025f}};
    for(int pose:{0,3})for(int direction=0;direction<4;++direction) {
        NeutralInput input;input.body=old::initial_world(pose).physical.body;
        for(float& angle:input.body.angle)angle=0;for(float& velocity:input.body.velocity)velocity=0;
        input.current=directions[direction];input.source=(pose==0?0:4)+direction;out.push_back(input);
    }
    const auto bytes=cp::read((root/"preserved-v1-first-failure.bin").string());cp::Reader reader{bytes};
    require(reader.value<uint32_t>()==1,"Wrong preserved V1 diagnostic schema");
    const auto active=reader.value<Case>();require(reader.value<int>()==77,"Wrong preserved V1 failure control");
    const auto cpu=reader.value<RecordedV1Step>(),gpu=reader.value<RecordedV1Step>();(void)reader.value<RecordedV1Step>();reader.finish();
    require(active.group==3&&active.id==0,"Wrong preserved V1 failure fixture");
    for(int i=0;i<4;++i) {
        NeutralInput input;input.body=i==0?cpu.world.physical.body:i==1?gpu.world.physical.body:i==2?cpu.before:gpu.before;
        input.current=active.current;input.fluid=cpu.world.physical.fluid;input.source=8+i;out.push_back(input);
    }
    return out;
}
double check_probe(const Probe& x, double& maximum_dissipation, double& maximum_cancellation,
                   double& maximum_matrix_delta, double& maximum_matrix_ratio,
                   double& maximum_static_force, double& maximum_static_torque) {
    double maximum_force = 0, quadratic = 0, flow_power = 0;
    const double velocity[10] = {x.current.x,x.current.y,x.current.z,0,0,0,0,0,0,0};
    double resistance_scale = 0, resistance_delta = 0;
    for (int i = 0; i < 10; ++i) for (int j = 0; j < 10; ++j) {
        resistance_scale = std::max(resistance_scale,std::max(std::abs(double(x.forced.R[i][j])),std::abs(double(x.negative.R[i][j]))));
        resistance_delta = std::max(resistance_delta,std::abs(double(x.forced.R[i][j])-x.negative.R[i][j]));
    }
    const double resistance_bound = 64*std::numeric_limits<float>::epsilon()*resistance_scale;
    require(resistance_scale > 0 && resistance_delta <= resistance_bound,"Galilean drag matrix exceeds float32 roundoff bound");
    maximum_matrix_delta = std::max(maximum_matrix_delta,resistance_delta);
    maximum_matrix_ratio = std::max(maximum_matrix_ratio,resistance_delta/resistance_bound);
    for (int i = 0; i < 10; ++i) {
        require(x.zero.C[i] == x.zero_reference.C[i] && x.zero.force[i] == x.zero_reference.force[i],"Zero-current assembled vectors changed");
        require(x.forced.C[i] == x.negative.C[i],"Galilean inertial vector changed");
        double equivalent = x.negative.force[i];
        // Uniform comoving water has zero hydrodynamic force. Existing static
        // gravity/buoyancy is a separate term, including its CUDA FMA roundoff.
        double comoving_force = double(x.comoving.force[i])-x.zero_reference.force[i];
        if (i < 3) maximum_static_force = std::max(maximum_static_force,std::abs(double(x.zero_reference.force[i])));
        else maximum_static_torque = std::max(maximum_static_torque,std::abs(double(x.zero_reference.force[i])));
        for (int j = 0; j < 10; ++j) {
            require(x.zero.M[i][j] == x.zero_reference.M[i][j] && x.zero.R[i][j] == x.zero_reference.R[i][j],"Zero-current assembled matrices changed");
            require(x.forced.M[i][j] == x.negative.M[i][j],"Galilean mass matrix changed");
            equivalent += x.negative.R[i][j]*velocity[j];
            comoving_force -= x.comoving.R[i][j]*velocity[j];
            quadratic += velocity[i]*x.comoving.R[i][j]*velocity[j];
        }
        maximum_force = std::max(maximum_force,std::abs(equivalent-x.forced.force[i]));
        maximum_force = std::max(maximum_force,std::abs(comoving_force));
        flow_power += x.terms.force[i]*velocity[i];
    }
    maximum_cancellation = std::max(maximum_cancellation,std::abs(quadratic-2*flow_power+x.terms.water_quadratic));
    const double relative_dissipation = x.direct_dissipation;
    maximum_dissipation = std::max(maximum_dissipation,std::abs(relative_dissipation));
    require(maximum_force <= 1e-11,"Galilean/comoving generalized force defect");
    require(std::abs(relative_dissipation) <= 1e-16,"Comoving relative-water dissipation defect");
    require(std::abs(x.direct_supply) <= 1e-16,"Comoving water-supply work defect");
    return maximum_force;
}
std::vector<float> load_backbone(const fs::path& root) {
    const auto state = cudarl::rl::read_ppo_training_state(cp::read((root/"parent-checkpoint.bin").string()),cp::checksum(cp::read((root/"parent-protocol.bin").string())));
    require(state.actor.update == 336,"Gate requires preserved 336 parent");
    return cudarl::rl::append_zero_input_column(state.actor.parameters);
}
float load_hold_gain(const fs::path& root) {
    const auto bytes = cp::read((root/"qualified-hold-preview.bin").string());cp::Reader r{bytes};
    const uint32_t expected[12] = {0x31564c48,2,6,6,3,4,41,4,336,1,7,1};
    for (auto value : expected) require(r.value<uint32_t>() == value,"Wrong qualified holding package identity");
    require(r.value<uint64_t>() != 0,"Holding package lacks a source protocol");
    const auto parent = r.array<uint8_t>(64);
    require(std::string(parent.begin(),parent.end()) == "810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0","Wrong holding parent SHA");
    require(r.value<uint32_t>() == 12,"Wrong holding source contract count");
    for (int i = 0; i < 12; ++i) require(r.array<uint8_t>(64).size() == 64,"Wrong holding source hash shape");
    const auto parameters = r.array<float>(2532), head = r.array<float>(2);
    const float stored_gain = r.value<float>(), stored_sigma = r.value<float>();r.finish();
    require(parameters == load_backbone(root) && head.size() == 2,"Holding backbone differs from preserved 336");
    require(p::finite_number(head[0]) && p::finite_number(head[1]),"Nonfinite holding parameters");
    const float gain = cudarl::learned_amplitude::sigmoid(head[0]);
    require(std::abs(gain-stored_gain) <= 2e-7f && std::abs(cudarl::learned_amplitude::standard_deviation(head[1])-stored_sigma) <= 2e-6f,"Holding parameter mapping changed");
    return gain;
}
std::vector<uint8_t> protocol_bytes(const fs::path& root) {
    cp::Writer out;out.value(protocol_magic);out.value(uint32_t{2});out.value(controls_cap);out.value(substeps_cap);
    out.value(uint32_t{sizeof(motor::World)});out.value(uint32_t{sizeof(model::Telemetry)});
    for (const auto* name : {"parent-checkpoint.bin","parent-protocol.bin","qualified-hold-preview.bin","preserved-v1-first-failure.bin"}) out.value(cp::checksum(cp::read((root/name).string())));
    std::ifstream file(root/"approved-protocol.md",std::ios::binary);
    const std::vector<uint8_t> approved_bytes((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
    require(!approved_bytes.empty(),"Approved protocol is empty");out.value(cp::checksum(approved_bytes));out.value(load_hold_gain(root));
    return out.data;
}
uint64_t protocol(const fs::path& root) {
    const auto bytes = cp::read((root/"protocol.bin").string());require(bytes == protocol_bytes(root),"Gate protocol or input tuple changed");return cp::checksum(bytes);
}
int prepare(const fs::path& root, const fs::path& approved) {
    require(!fs::exists(root),"Gate identity already exists");require(fs::is_regular_file(approved),"Approved protocol missing");
    fs::create_directories(root);
    fs::copy_file(approved,root/"approved-protocol.md");
    const fs::path parent = "assets/ppo-four-axis-checkpoints/update-336-episode64";
    fs::copy_file(parent/"ppo-update-336.bin",root/"parent-checkpoint.bin");
    fs::copy_file(parent/"protocol.bin",root/"parent-protocol.bin");
    fs::copy_file("live/research_data/learned-amplitude-v1.bin",root/"qualified-hold-preview.bin");
    fs::copy_file("build/ppo/current-swim-2026-11-22-v1/physics-gate/run-started/first-failure.bin",root/"preserved-v1-first-failure.bin");
    cp::write_atomic((root/"protocol.bin").string(),protocol_bytes(root));
    (void)load_backbone(root);
    std::cout << "Prepared current-physics gate; physical controls=0, optimizer steps=0\n";
    return 0;
}
int zero(const fs::path& root) {
    (void)protocol(root);
    require(fs::create_directory(root/"zero-started"),"Zero gate already attempted");
    std::vector<Probe> cpu(8),gpu;
    const p::V directions[4] = {{.0025f,0,0},{-.0025f,0,0},{0,0,.0025f},{0,0,-.0025f}};
    for (int i = 0; i < 8; ++i) {
        cpu[i].body = old::initial_world(i/4 == 0 ? 0 : 3).physical.body;
        for (float& angle : cpu[i].body.angle) angle = 0;
        for (float& velocity : cpu[i].body.velocity) velocity = 0;
        cpu[i].current = directions[i%4];
    }
    gpu = cpu;for (auto& x : cpu) assemble_probe(x);
    device_ptr<Probe> device(8);device.copy_from_host(gpu.data(),8);
    probe_assembly<<<1,8>>>(device.get());CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());
    device.copy_to_host(gpu.data(),8);
    cp::Writer proof;proof.value(uint32_t{1});proof.value(uint32_t{sizeof(Probe)});proof.array(cpu);proof.array(gpu);
    cp::write_atomic((root/"zero-started"/"assembly-probes.bin").string(),proof.data);
    const auto neutral_inputs=neutral_fixtures(root);
    std::vector<NeutralOutput> neutral_cpu(neutral_inputs.size()),neutral_gpu(neutral_inputs.size());
    for(size_t i=0;i<neutral_inputs.size();++i)assemble_neutral(neutral_inputs[i],neutral_cpu[i]);
    device_ptr<NeutralInput> device_neutral_input(neutral_inputs.size());device_ptr<NeutralOutput> device_neutral_output(neutral_inputs.size());
    device_neutral_input.copy_from_host(neutral_inputs.data(),neutral_inputs.size());
    neutral_assembly<<<1,32>>>(device_neutral_input.get(),device_neutral_output.get(),int(neutral_inputs.size()));
    CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());device_neutral_output.copy_to_host(neutral_gpu.data(),neutral_gpu.size());
    cp::Writer neutral_proof;neutral_proof.value(uint32_t{1});neutral_proof.value(uint32_t{sizeof(NeutralInput)});neutral_proof.value(uint32_t{sizeof(NeutralOutput)});
    neutral_proof.array(neutral_inputs);neutral_proof.array(neutral_cpu);neutral_proof.array(neutral_gpu);
    cp::write_atomic((root/"zero-started"/"neutral-force-probes.bin").string(),neutral_proof.data);
    double old_neutral_max=0,new_neutral_max=0;
    for(const auto* group:{&neutral_cpu,&neutral_gpu})for(const auto& value:*group)for(int i=0;i<10;++i) {
        old_neutral_max=std::max(old_neutral_max,std::abs(double(value.old_current.force[i])-value.old_flow.force[i]));
        new_neutral_max=std::max(new_neutral_max,std::abs(double(value.new_current.force[i])-value.new_flow.force[i]));
        require(value.old_current.C[i]==value.new_current.C[i],"V2 default-gravity proof changed inertia vector");
        require(value.new_zero.C[i]==value.legacy_zero.C[i]&&value.new_zero.force[i]==value.legacy_zero.force[i],"V2 default-gravity zero delegation changed");
        for(int j=0;j<10;++j) {
            require(value.old_current.M[i][j]==value.new_current.M[i][j]&&value.old_current.R[i][j]==value.new_current.R[i][j],"V2 default-gravity proof changed mass or resistance");
            require(value.new_zero.M[i][j]==value.legacy_zero.M[i][j]&&value.new_zero.R[i][j]==value.legacy_zero.R[i][j],"V2 default-gravity zero matrix delegation changed");
        }
    }
    require(new_neutral_max==0,"V2 rounded default-gravity neutral force is not exactly zero");
    double maximum_force = 0, maximum_dissipation = 0, maximum_cancellation = 0;
    double maximum_matrix_delta = 0, maximum_matrix_ratio = 0, maximum_static_force = 0, maximum_static_torque = 0;
    for (const auto* group : {&cpu,&gpu}) for (const auto& x : *group)
        maximum_force = std::max(maximum_force,check_probe(x,maximum_dissipation,maximum_cancellation,
            maximum_matrix_delta,maximum_matrix_ratio,maximum_static_force,maximum_static_torque));
    std::ofstream report(root/"zero-started"/"result.json");report << std::setprecision(17)
        << "{\"passed\":true,\"CPU_probes\":8,\"actual_CUDA_probes\":8,\"physical_controls\":0,\"physical_substeps\":0,\"maximum_force_error\":"
        << maximum_force << ",\"maximum_comoving_relative_dissipation_W\":" << maximum_dissipation << ",\"diagnostic_aggregate_cancellation_W\":" << maximum_cancellation
        << ",\"Galilean_R_maximum_absolute_difference\":" << maximum_matrix_delta << ",\"Galilean_R_maximum_fraction_of_roundoff_bound\":" << maximum_matrix_ratio
        << ",\"diagnostic_maximum_static_force_N\":" << maximum_static_force << ",\"diagnostic_maximum_static_torque_Nm\":" << maximum_static_torque
        << ",\"hydrodynamic_assembly_gravity_m_s2\":0,\"physical_gate_default_gravity_m_s2\":9.81,\"default_gravity_CPU_probes\":12,\"default_gravity_CUDA_probes\":12"
        << ",\"default_gravity_old_neutral_force_max\":" << old_neutral_max << ",\"default_gravity_v2_neutral_force_max\":" << new_neutral_max << "}\n";
    cp::Writer marker;marker.value(uint32_t{2});marker.value(protocol(root));cp::write_atomic((root/"zero-passed.bin").string(),marker.data);
    std::cout << "Zero-integration current gate passed on CPU and CUDA\n";return 0;
}
void capture(cp::Writer& raw, std::ofstream& csv, const Case& c, int backend, int control,
             const motor::World& world, const model::Telemetry& telemetry) {
    raw.value(c.group);raw.value(c.id);raw.value(backend);raw.value(control);raw.value(world);raw.value(telemetry);
    const auto& b = world.physical.body;
    csv << group_name(c.group) << ',' << c.id << ',' << backend << ',' << control << ',' << b.position.x << ',' << b.position.y << ',' << b.position.z
        << ',' << b.orientation.w << ',' << b.orientation.x << ',' << b.orientation.y << ',' << b.orientation.z;
    for (float angle : b.angle) csv << ',' << angle;
    for (float velocity : b.velocity) csv << ',' << velocity;
    csv << ',' << world.attempted_substeps << ',' << world.physical.substeps << ',' << world.signed_work << ',' << world.positive_work << ',' << world.fluid_loss
        << ',' << world.external_work << ',' << world.numerical_loss << ',' << world.last_energy_error << ',' << telemetry.body_flow_work << ',' << telemetry.water_supply_work << '\n';
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
    const double ce[6] = {cpu.signed_work,cpu.positive_work,cpu.fluid_loss,cpu.external_work,cpu.numerical_loss,cpu.last_energy_error};
    const double ge[6] = {gpu.signed_work,gpu.positive_work,gpu.fluid_loss,gpu.external_work,gpu.numerical_loss,gpu.last_energy_error};
    for (int i = 0; i < 6; ++i) require(std::abs(ce[i]-ge[i]) <= 1e-12+1e-4*std::max(std::abs(ce[i]),std::abs(ge[i])),"CPU/CUDA cumulative energy mismatch");
}
int run(const fs::path& root) {
    const auto active_started = std::chrono::steady_clock::now();
    require(fs::is_regular_file(root/"zero-passed.bin"),"Zero gate has not passed");
    const auto zero_bytes = cp::read((root/"zero-passed.bin").string());cp::Reader zero_marker{zero_bytes};
    require(zero_marker.value<uint32_t>() == 2 && zero_marker.value<uint64_t>() == protocol(root),"Zero gate source identity changed");zero_marker.finish();
    require(fs::create_directory(root/"run-started"),"Physical gate already attempted; no retry");
    Counts count;std::string fault;double maximum_pose = 0;int finished_cases = 0;
    cp::Writer raw;std::ofstream csv;
    Step cpu{},gpu{},historical{};Case active{};int active_control = 0;
    try {
        raw.value(uint32_t{1});raw.value(uint32_t{sizeof(motor::World)});raw.value(uint32_t{sizeof(model::Telemetry)});
        csv.open(root/"run-started"/"states.csv");csv << std::setprecision(17)
            << "group,case,backend,control,x,y,z,qw,qx,qy,qz,j0,j1,j2,j3,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,attempted_substeps,substeps,joint_work,positive_work,relative_water_loss,external_work,numerical_loss,energy_error,last_body_flow_work,last_water_supply_work\n";
        require(bool(csv),"Gate state log unavailable");
        nn::CpuMlp<float> backbone({{41,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}});backbone.params() = load_backbone(root);
        const float hold_gain = load_hold_gain(root);
        device_ptr<Step> device(1);device_ptr<float> command_device(4);
        for (const auto& c : cases()) {
            active = c;old::ForwardTask task;cpu = {};cpu.world = initialize(c,task);gpu = cpu;historical = cpu;
            device.copy_from_host(&gpu,1);
            const auto start = cudarl::motor::motion(cpu.world.physical.body).com_model*p::scale;
            capture(raw,csv,c,1,0,cpu.world,cpu.telemetry);capture(raw,csv,c,2,0,gpu.world,gpu.telemetry);
            if (c.group < 2) capture(raw,csv,c,0,0,historical.world,historical.telemetry);
            for (int control = 1; control <= c.controls; ++control) {
                active_control = control;float command[4]{};
                auto view = cpu.world;view.physical.current = {};
                if (c.group == 1) require(old::update_velocity_request(view,config(c),task),"Gate hold request update failed");
                if (c.group != 3) {
                    float observation[41]{};require(old::observe_appended(task,view,config(c),observation),"Gate backbone observation rejected");
                    backbone.forward(observation,1);
                    for (int j = 0; j < 4; ++j) command[j] = tanhf(backbone.output()[j]*(c.group == 1 ? hold_gain : 1.f));
                }
                if (c.group < 2) {
                    const auto before = historical.world.attempted_substeps;++count.controls;
                    historical.advanced = cudarl::joint_motor::step_state(historical.world,command,&historical.telemetry,&historical.before,&historical.candidate);
                    count.substeps += historical.world.attempted_substeps-before;if (historical.advanced) ++count.completed;
                    capture(raw,csv,c,0,control,historical.world,historical.telemetry);
                    require(historical.advanced != 0,"Historical zero-current physical control failed");
                }
                const auto before = cpu.world.attempted_substeps;++count.controls;
                cpu.advanced = motor::step_state(cpu.world,command,&cpu.telemetry,&cpu.before,&cpu.candidate);
                count.substeps += cpu.world.attempted_substeps-before;if (cpu.advanced) ++count.completed;
                capture(raw,csv,c,1,control,cpu.world,cpu.telemetry);
                require(cpu.advanced != 0,"CPU current physical control failed");
                const auto gpu_before = gpu.world.attempted_substeps;++count.controls;
                command_device.copy_from_host(command,4);advance<<<1,1>>>(device.get(),command_device.get());
                CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());device.copy_to_host(&gpu,1);
                count.substeps += gpu.world.attempted_substeps-gpu_before;if (gpu.advanced) ++count.completed;
                capture(raw,csv,c,2,control,gpu.world,gpu.telemetry);
                require(gpu.advanced != 0,"CUDA current physical control failed");
                if (c.group < 2) same_zero(historical,cpu);
                cpu_cuda(cpu.world,gpu.world,maximum_pose);
                auto post_view = cpu.world;post_view.physical.current = {};
                const auto transition = c.group == 1 ? old::record_hold_control(task,post_view,config(c)) : old::record_control(task,post_view,config(c));
                require(transition.accepted,"Gate pure task record rejected");
                require(count.controls <= controls_cap && count.substeps <= substeps_cap,"Gate budget exceeded");
            }
            if (c.group == 3) {
                for (const auto* state : {&cpu,&gpu}) {
                    const auto finish = cudarl::motor::motion(state->world.physical.body).com_model*p::scale;
                    require(p::dot(finish-start,c.current) > 0,"Zero-target-servo advection sign failed");
                }
            }
            ++finished_cases;
        }
        require(count.controls == controls_cap && count.completed == controls_cap && count.substeps == substeps_cap,"Gate final accounting mismatch");
    } catch (const std::exception& e) {
        fault = e.what();cp::Writer failure;failure.value(uint32_t{1});failure.value(active);failure.value(active_control);
        failure.value(cpu);failure.value(gpu);failure.value(historical);
        cp::write_atomic((root/"run-started"/"first-failure.bin").string(),failure.data);
    }
    cp::write_atomic((root/"run-started"/"states.bin").string(),raw.data);csv.flush();
    const double active_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-active_started).count();
    std::ofstream elapsed(root/"run-started"/"active-seconds.txt");elapsed << std::setprecision(17) << active_seconds << '\n';elapsed.flush();
    std::ofstream result(root/"run-started"/"result.json");result << std::setprecision(17)
        << "{\"closed\":true,\"passed\":" << (fault.empty() ? "true" : "false") << ",\"fault\":" << std::quoted(fault)
        << ",\"attempted_controls\":" << count.controls << ",\"completed_controls\":" << count.completed << ",\"physical_substeps\":" << count.substeps
        << ",\"finished_cases\":" << finished_cases << ",\"maximum_position_error_model\":" << maximum_pose << ",\"active_seconds\":" << active_seconds << ",\"PPO_updates\":0,\"optimizer_steps\":0}\n";
    if (!fault.empty()) {std::cerr << fault << '\n';return 3;}
    cp::Writer marker;marker.value(passed_magic);marker.value(uint32_t{2});marker.value(uint32_t{1});
    marker.value(count.controls);marker.value(count.substeps);marker.value(protocol(root));
    cp::write_atomic((root/"gate-passed.bin").string(),marker.data);
    std::cout << "Current physics gate passed: controls=" << count.controls << ", substeps=" << count.substeps << ", maximum position error=" << maximum_pose << '\n';return 0;
}
} // namespace gate_v2
int main(int argc,char** argv) {
    try {
        gate_v2::require(argc == 3 || argc == 4,"Use --prepare RUN PROTOCOL, --zero RUN or --run RUN");
        const std::string mode = argv[1];const auto root = gate_v2::fs::absolute(argv[2]);
        if (mode == "--prepare") {gate_v2::require(argc == 4,"Prepare requires the approved protocol");return gate_v2::prepare(root,gate_v2::fs::absolute(argv[3]));}
        gate_v2::require(argc == 3,"Unexpected gate argument");
        if (mode == "--zero") return gate_v2::zero(root);if (mode == "--run") return gate_v2::run(root);
        throw std::runtime_error("Unknown current gate mode");
    } catch (const std::exception& e) {std::cerr << e.what() << '\n';return 2;}
}
