#ifndef CUDARL_COMMAND_MOTOR_POLICY_H
#define CUDARL_COMMAND_MOTOR_POLICY_H
#include "envs/swimmer/command_motor.h"
#include "swimmer_spec.h"
#include <string>

// Pure C++ read contract shared by the diagnostic trainer and Godot viewer.
// An exported candidate is not a promotion into the working creature.
namespace cudarl::rl {
constexpr int command_motor_updates=128,command_motor_worlds=512;
inline nn::MlpSpec command_motor_specification(int outputs=4) {return swimmer_specification(outputs,command_motor::kObs);}
inline std::vector<float> command_motor_protocol() {
    // env/observation/reward version, dimensions, finite horizon, geometry,
    // cost terms, Gaussian/PPO/Adam settings, and the one registered budget.
    return {2,59,4,32,600,.05f,.005f,10,.04f,.072f,.02f,1e-5f,
        1,.25f,.1f,.05f,.02f,.02f,.002f,.5f,4.f,216.f,
        .99f,.95f,.2f,.002f,.5f,64,4,8,1e-3f,.9f,.999f,1e-5f,.5f,-.5f,.01f,512,128,
        1000.f,.001f,9.81f,4.5f,4.f,3.5f,1.f};
}
inline uint64_t command_motor_fingerprint() {
    const auto values=command_motor_protocol();uint64_t hash=14695981039346656037ull;
    const auto* bytes=reinterpret_cast<const uint8_t*>(values.data());
    for(size_t i=0;i<values.size()*sizeof(float);++i) {hash^=bytes[i];hash*=1099511628211ull;}
    const uint32_t seeds[]={command_motor::head_seed,command_motor::training_seed};
    bytes=reinterpret_cast<const uint8_t*>(seeds);
    for(size_t i=0;i<sizeof(seeds);++i) {hash^=bytes[i];hash*=1099511628211ull;}
    return hash;
}
struct CommandMotorPolicy {
    int update=0;
    uint32_t seed=command_motor::training_seed;
    uint64_t source_fingerprint=0,environment_fingerprint=command_motor_fingerprint();
    bool passed_final_gate=false;
    std::vector<float> actor;
    std::vector<uint8_t> serialize() const {
        if(update<0||update>command_motor_updates||seed!=command_motor::training_seed||!source_fingerprint||environment_fingerprint!=command_motor_fingerprint()||
            (passed_final_gate&&update!=command_motor_updates)||
            actor.size()!=static_cast<size_t>(command_motor_specification().num_params())) throw std::runtime_error("Invalid command motor policy");
        for(float x:actor) if(!std::isfinite(x)) throw std::runtime_error("Nonfinite command motor policy");
        checkpoint::Writer out;out.value(uint32_t{0x434D504Cu});out.value(uint32_t{1});out.value(environment_fingerprint);
        out.value(source_fingerprint);out.value(update);out.value(seed);out.value(passed_final_gate);out.array(actor);return out.data;
    }
    static CommandMotorPolicy read(const std::vector<uint8_t>& bytes) {
        checkpoint::Reader in{bytes};CommandMotorPolicy policy;
        if(in.value<uint32_t>()!=0x434D504Cu||in.value<uint32_t>()!=1) throw std::runtime_error("Incompatible command motor policy");
        policy.environment_fingerprint=in.value<uint64_t>();policy.source_fingerprint=in.value<uint64_t>();policy.update=in.value<int>();
        policy.seed=in.value<uint32_t>();policy.passed_final_gate=in.value<bool>();policy.actor=in.array<float>(command_motor_specification().num_params());in.finish();
        (void)policy.serialize();return policy;
    }
    static CommandMotorPolicy load(const std::string& path) {return read(checkpoint::read(path));}
};
struct CommandMotorGate {
    uint64_t environment_fingerprint=command_motor_fingerprint(),source_fingerprint=0,benchmark_fixture_fingerprint=0,benchmark_rules_fingerprint=0;
    bool numerical_passed=false,physical_reference_passed=false;
    std::vector<uint8_t> serialize() const {
        checkpoint::Writer out;out.value(uint32_t{0x434D4741u});out.value(uint32_t{1});out.value(environment_fingerprint);out.value(source_fingerprint);
        out.value(benchmark_fixture_fingerprint);out.value(benchmark_rules_fingerprint);out.value(numerical_passed);out.value(physical_reference_passed);return out.data;
    }
    static CommandMotorGate read(const std::vector<uint8_t>& bytes) {
        checkpoint::Reader in{bytes};CommandMotorGate gate;
        if(in.value<uint32_t>()!=0x434D4741u||in.value<uint32_t>()!=1) throw std::runtime_error("Incompatible motor preflight gate");
        gate.environment_fingerprint=in.value<uint64_t>();gate.source_fingerprint=in.value<uint64_t>();gate.benchmark_fixture_fingerprint=in.value<uint64_t>();gate.benchmark_rules_fingerprint=in.value<uint64_t>();
        gate.numerical_passed=in.value<bool>();gate.physical_reference_passed=in.value<bool>();in.finish();return gate;
    }
    void require_passed(uint64_t source,uint64_t expected_rules) const {
        if(!numerical_passed||!physical_reference_passed||environment_fingerprint!=command_motor_fingerprint()||source_fingerprint!=source||!benchmark_fixture_fingerprint||
            !benchmark_rules_fingerprint||benchmark_rules_fingerprint!=expected_rules)
            throw std::runtime_error("Motor pilot blocked: numerical and physical reference gates must both pass for this source/contract");
    }
};
}
#endif
