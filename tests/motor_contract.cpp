#if defined(CUDARL_WITH_CUDA) && CUDARL_WITH_CUDA
#error "The structural motor contract check must be CPU-only"
#endif
#ifndef CUDARL_WITH_CUDA
#define CUDARL_WITH_CUDA 0
#endif
#include "envs/swimmer/motor.h"
#include "envs/swimmer/motor_v2.h"
#include "rl/motor_policy.h"
#include "rl/motor_v2_policy.h"
#include "rl/boulder_source.h"
#include "motor_v2_benchmark.h"
#include <cstddef>
#include <cstring>
#include <iostream>
#include <type_traits>

// Structural/readback oracle only: no integration, model inference, learner,
// GPU allocation, candidate preparation or file writes.
namespace {
using namespace cudarl;
void require(bool condition,const char* message) {if(!condition) throw std::runtime_error(message);}
#define SAME_OFFSET(A,B,FIELD) static_assert(offsetof(A,FIELD)==offsetof(B,FIELD),"Offset changed: " #FIELD)
static_assert(sizeof(motor::Request)==sizeof(command_motor::Request));
static_assert(sizeof(motor::TaskState)==sizeof(command_motor::State));
static_assert(sizeof(motor::State)==sizeof(motor_v2::State));
static_assert(sizeof(motor::Gains)==sizeof(paired_motor::Gains));
static_assert(sizeof(motor::Result)==sizeof(command_motor::Result));
static_assert(sizeof(motor_history::State)==sizeof(paired_motor::State));
static_assert(sizeof(motor_history::RegulationCase)==sizeof(paired_motor::RegulationCase));
static_assert(std::is_trivially_copyable_v<motor::State>&&std::is_standard_layout_v<motor::State>);
SAME_OFFSET(motor::TaskState,command_motor::State,body);SAME_OFFSET(motor::TaskState,command_motor::State,request);
SAME_OFFSET(motor::TaskState,command_motor::State,previous_action);SAME_OFFSET(motor::TaskState,command_motor::State,seed);
SAME_OFFSET(motor::TaskState,command_motor::State,environment);SAME_OFFSET(motor::TaskState,command_motor::State,episode);SAME_OFFSET(motor::TaskState,command_motor::State,steps);
SAME_OFFSET(motor::State,motor_v2::State,task);SAME_OFFSET(motor::State,motor_v2::State,water_velocity);SAME_OFFSET(motor::State,motor_v2::State,lesson);
SAME_OFFSET(motor::State,motor_v2::State,integrated_substeps);SAME_OFFSET(motor::State,motor_v2::State,executed_substeps);
SAME_OFFSET(motor::State,motor_v2::State,held_target);SAME_OFFSET(motor::State,motor_v2::State,actual_torques);SAME_OFFSET(motor::State,motor_v2::State,unclamped_torques);
SAME_OFFSET(motor::State,motor_v2::State,mean_squared_torque);SAME_OFFSET(motor::State,motor_v2::State,first_unsafe_substep);SAME_OFFSET(motor::State,motor_v2::State,unsafe_substeps);
SAME_OFFSET(motor::State,motor_v2::State,first_unsafe_task);SAME_OFFSET(motor::State,motor_v2::State,substep_min_floor_gap);SAME_OFFSET(motor::State,motor_v2::State,substep_max_floor_gap);
SAME_OFFSET(motor::State,motor_v2::State,substep_min_other_gap);SAME_OFFSET(motor::State,motor_v2::State,floor_contact_samples);SAME_OFFSET(motor::State,motor_v2::State,substep_max_floor_force);
SAME_OFFSET(motor::State,motor_v2::State,substep_max_head_speed);SAME_OFFSET(motor::State,motor_v2::State,substep_max_com_speed);SAME_OFFSET(motor::State,motor_v2::State,substep_max_omega);
SAME_OFFSET(motor::State,motor_v2::State,substep_max_goal_error);SAME_OFFSET(motor::State,motor_v2::State,substep_max_heading_error);
SAME_OFFSET(motor_history::State,paired_motor::State,task);SAME_OFFSET(motor_history::State,paired_motor::State,held_target);
SAME_OFFSET(motor_history::State,paired_motor::State,actual_torques);SAME_OFFSET(motor_history::State,paired_motor::State,unclamped_torques);
SAME_OFFSET(motor_history::State,paired_motor::State,mean_squared_torque);SAME_OFFSET(motor_history::State,paired_motor::State,substep_max_target_error);
SAME_OFFSET(motor_history::State,paired_motor::State,substep_max_joint_rate);SAME_OFFSET(motor_history::State,paired_motor::State,first_unsafe_substep);
SAME_OFFSET(motor_history::State,paired_motor::State,unsafe_substeps);SAME_OFFSET(motor_history::State,paired_motor::State,first_unsafe_task);
template<class T> void write_task(checkpoint::Writer& out,const T& task) {
    out.value(task.body.position);out.value(task.body.orientation);for(float value:task.body.angle) out.value(value);for(float value:task.body.velocity) out.value(value);
    out.value(task.body.work);out.value(task.body.travel);out.value(task.request.position);out.value(task.request.velocity);out.value(task.request.heading);
    for(float value:task.previous_action) out.value(value);out.value(task.seed);out.value(task.environment);out.value(task.episode);out.value(task.steps);
}
template<class T> std::vector<uint8_t> defined_state(const T& state) {
    checkpoint::Writer out;write_task(out,state.task);out.value(state.water_velocity);out.value(state.lesson);out.value(state.integrated_substeps);out.value(state.executed_substeps);
    for(float value:state.held_target) out.value(value);for(float value:state.actual_torques) out.value(value);for(float value:state.unclamped_torques) out.value(value);out.value(state.mean_squared_torque);
    out.value(state.first_unsafe_substep);out.value(state.unsafe_substeps);write_task(out,state.first_unsafe_task);out.value(state.substep_min_floor_gap);out.value(state.substep_max_floor_gap);out.value(state.substep_min_other_gap);
    out.value(state.floor_contact_samples);out.value(state.substep_max_floor_force);out.value(state.substep_max_head_speed);out.value(state.substep_max_com_speed);out.value(state.substep_max_omega);out.value(state.substep_max_goal_error);out.value(state.substep_max_heading_error);return out.data;
}
struct CanonicalFixture {
    motor::State initial{};motor_v2_benchmark::Kind kind=motor_v2_benchmark::Kind::Move;uint32_t seed=rl::motor_final_seed;
    int id=0,steps=600,source_capture=-1;float actual_entry_speed=0;bool qualified=true;
};
static_assert(sizeof(CanonicalFixture)==sizeof(motor_v2_benchmark::Fixture));
SAME_OFFSET(CanonicalFixture,motor_v2_benchmark::Fixture,initial);SAME_OFFSET(CanonicalFixture,motor_v2_benchmark::Fixture,kind);
SAME_OFFSET(CanonicalFixture,motor_v2_benchmark::Fixture,seed);SAME_OFFSET(CanonicalFixture,motor_v2_benchmark::Fixture,id);SAME_OFFSET(CanonicalFixture,motor_v2_benchmark::Fixture,steps);
SAME_OFFSET(CanonicalFixture,motor_v2_benchmark::Fixture,source_capture);SAME_OFFSET(CanonicalFixture,motor_v2_benchmark::Fixture,actual_entry_speed);SAME_OFFSET(CanonicalFixture,motor_v2_benchmark::Fixture,qualified);
void fixtures_roundtrip(const std::vector<uint8_t>& bytes) {
    checkpoint::Reader in{bytes};checkpoint::Writer out;
    const auto magic=in.value<uint32_t>(),version=in.value<uint32_t>(),size=in.value<uint32_t>();require(magic==0x4D563246u&&version==2&&size==sizeof(CanonicalFixture),"Fixture metadata changed");out.value(magic);out.value(version);out.value(size);
    const auto source=in.value<uint64_t>(),physics=in.value<uint64_t>(),rules=in.value<uint64_t>();require(physics==motor::physics_fingerprint()&&rules==motor_v2_benchmark::rules_fingerprint(),"Fixture physics/rules hash differs");out.value(source);out.value(physics);out.value(rules);
    const auto fixtures=in.array<CanonicalFixture>(64);in.finish();require(fixtures.size()==64,"Fixture count changed");out.array(fixtures);require(out.data==bytes,"Canonical fixture layout did not roundtrip protected bytes");
    for(const auto& fixture:fixtures) require(motor::finite(fixture.initial),"Canonical fixture decode invalid");
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=5) {std::cerr<<"Usage: motor_contract PREPARED-POLICY FIXTURES HISTORICAL-GATE ORIGINAL-SOURCE\n";return 2;}
        require(motor_contact::physics_protocol()==motor_contact_v2::physics_protocol(),"Contact constants/serialized protocol changed");
        require(motor_contact::physics_fingerprint()==motor_contact_v2::physics_fingerprint(),"Contact physics fingerprint changed");
        require(motor::protocol()==motor_v2::protocol()&&motor::physics_fingerprint()==motor_v2::physics_fingerprint(),"Motor task/physics constants changed");
        require(rl::motor_training_protocol()==rl::motor_v2_training_protocol(),"Training constants changed");
        int initial_conditions=0;
        for(uint32_t episode:{0u,1u,7u}) for(uint32_t environment=0;environment<128;++environment) {
            const auto old=motor_v2::initial(rl::motor_v2_seed,environment,episode);const auto now=motor::initial(rl::motor_seed,environment,episode);
            require(defined_state(old)==defined_state(now),"Direct canonical initialization changed a defined state field");float a[59],b[59];motor_v2::observe(old,a);motor::observe(now,b);require(std::memcmp(a,b,sizeof(a))==0,"Canonical observation arithmetic changed");++initial_conditions;
        }
        const auto gate_bytes=checkpoint::read(argv[3]);const auto old_gate=paired_motor::RegulationGate::read(gate_bytes);const auto gate=motor_history::RegulationGate::read(gate_bytes);
        require(old_gate.serialize()==gate_bytes&&gate.serialize()==gate_bytes,"Historical evidence validation/readback changed");
        for(int j=0;j<4;++j) require(old_gate.gains.kp[j]==gate.gains.kp[j]&&old_gate.gains.kd[j]==gate.gains.kd[j]&&gate.gains.kp[j]==motor::validated_gains().kp[j]&&gate.gains.kd[j]==motor::validated_gains().kd[j],"Validated gain bits changed");
        const auto policy_bytes=checkpoint::read(argv[1]);const auto old_policy=rl::MotorV2Policy::read(policy_bytes);const auto policy=rl::MotorPolicy::read(policy_bytes);
        require(old_policy.serialize()==policy_bytes&&policy.serialize()==policy_bytes,"Canonical policy did not preserve protected serialized bytes");
        const auto fixture_bytes=checkpoint::read(argv[2]);fixtures_roundtrip(fixture_bytes);
        const auto source_bytes=checkpoint::read(argv[4]);const auto source=rl::read_boulder_source(source_bytes);
        require(rl::initialize_motor(source.motor)==rl::initialize_motor_v2(source.motor),"Canonical actor initialization changed");
        for(auto mode:{motor::Interface::Torque,motor::Interface::JointTarget}) {
            const auto old_mode=static_cast<paired_motor::Interface>(static_cast<int>(mode));require(rl::motor_environment_fingerprint(mode,gate.gains)==rl::motor_v2_environment_fingerprint(old_mode,old_gate.gains),"Canonical actuator/task fingerprint changed");
        }
        require(checkpoint::read(argv[1])==policy_bytes&&checkpoint::read(argv[2])==fixture_bytes&&checkpoint::read(argv[3])==gate_bytes&&checkpoint::read(argv[4])==source_bytes,"Contract check changed its protected inputs");
        std::cout<<"MOTOR_CONTRACT_PASSED initial_conditions="<<initial_conditions<<" state_bytes="<<sizeof(motor::State)<<" task_bytes="<<sizeof(motor::TaskState)<<" policy_fixture_history_roundtrip=exact physics_steps=0 inference_calls=0 training_updates=0 writes=0\n";return 0;
    }catch(const std::exception& error) {std::cerr<<"MOTOR_CONTRACT_FAILED "<<error.what()<<'\n';return 1;}
}
