#ifndef CUDARL_PAIRED_MOTOR_POLICY_H
#define CUDARL_PAIRED_MOTOR_POLICY_H
#include "envs/swimmer/paired_motor.h"
#include "swimmer_spec.h"

namespace cudarl::rl {
constexpr uint32_t paired_motor_protocol_version=1,paired_motor_head_seed=0x4D072601u,paired_motor_seed=0x4D072602u,paired_motor_final_seed=0x4D072604u;
constexpr int paired_motor_updates=64,paired_motor_worlds=512,paired_motor_total_steps=4194304;
inline nn::MlpSpec paired_motor_specification(int outputs=4) {return swimmer_specification(outputs,59);}
inline const char* paired_motor_arm(paired_motor::Interface mode) {return mode==paired_motor::Interface::Torque?"torque":"joint_target";}
inline std::vector<float> paired_motor_protocol() {
    // A new experiment, not a changed loop bound in the preserved128-update task.
    // Effort is MEAN40 of actual normalized actuator commands in BOTH arms.
    return {1,59,4,32,600,.05f,.005f,10,.04f,.072f,.02f,1e-5f,
        1,.25f,.1f,.05f,.02f,.02f,.002f,.5f,4,216,40,
        .99f,.95f,.2f,.002f,.5f,64,4,8,.001f,.9f,.999f,.00001f,.5f,-.5f,.01f,
        512,64,2,4194304,1000,.001f,9.81f,4.5f,4,3.5f,1};
}
inline uint64_t paired_motor_fingerprint(paired_motor::Interface mode,const paired_motor::RegulationProtocol& regulation) {
    if(!paired_motor::valid_interface(mode)) throw std::runtime_error("Unknown paired motor interface");regulation.validate();
    checkpoint::Writer out;out.value(uint32_t{0x504D5052u});out.value(paired_motor_protocol_version);out.array(paired_motor_protocol());
    out.value(mode);out.value(regulation.payload_fingerprint());out.value(regulation.gains);out.value(paired_motor_head_seed);out.value(paired_motor_seed);out.value(paired_motor_final_seed);
    return paired_motor::fingerprint(out.data);
}
inline std::vector<float> initialize_paired_motor(const std::vector<float>& accepted55) {
    const auto old=swimmer_specification(4,55),next=paired_motor_specification();
    if(accepted55.size()!=static_cast<size_t>(old.num_params())) throw std::runtime_error("Paired motor requires accepted55 feature layers");
    for(float value:accepted55) if(!std::isfinite(value)) throw std::runtime_error("Nonfinite paired feature source");
    auto result=nn::orthogonal_params(next,paired_motor_head_seed,{std::sqrt(2.f),std::sqrt(2.f),.01f});
    for(int row=0;row<next.out(0);++row) {
        for(int col=0;col<next.in(0);++col) result[next.weight_offset(0)+row*next.in(0)+col]=col>=9&&col<=32?accepted55[old.weight_offset(0)+row*old.in(0)+col]:0;
        result[next.bias_offset(0)+row]=accepted55[old.bias_offset(0)+row];
    }
    for(int row=0;row<next.out(1);++row) {
        for(int col=0;col<next.in(1);++col) result[next.weight_offset(1)+row*next.in(1)+col]=accepted55[old.weight_offset(1)+row*old.in(1)+col];
        result[next.bias_offset(1)+row]=accepted55[old.bias_offset(1)+row];
    }
    return result;
}
struct PairedMotorPolicy {
    paired_motor::Interface interface_type=paired_motor::Interface::Torque;
    paired_motor::Gains gains{};
    int update=0;uint32_t seed=paired_motor_seed;
    uint64_t source_fingerprint=0,environment_fingerprint=0;
    bool passed_final_gate=false;
    std::vector<float> actor;
    std::vector<uint8_t> regulation_gate;
    bool has_trained_weights() const {return update>0;}
    std::vector<uint8_t> serialize() const {
        const auto gate=paired_motor::RegulationGate::read(regulation_gate);gate.validate_binding(source_fingerprint);
        if(!paired_motor::valid_interface(interface_type)||!paired_motor::same_gains(gains,gate.gains)||seed!=paired_motor_seed||update<0||update>paired_motor_updates||
            environment_fingerprint!=paired_motor_fingerprint(interface_type,gate.protocol)||(passed_final_gate&&update!=paired_motor_updates)||
            actor.size()!=static_cast<size_t>(paired_motor_specification().num_params())) throw std::runtime_error("Paired motor policy contract differs");
        if(update>0) gate.require_passed(source_fingerprint);
        for(float value:actor) if(!std::isfinite(value)) throw std::runtime_error("Nonfinite paired motor policy");
        checkpoint::Writer out;out.value(uint32_t{0x504D504Cu});out.value(paired_motor_protocol_version);out.value(interface_type);out.value(gains);
        out.value(update);out.value(seed);out.value(source_fingerprint);out.value(environment_fingerprint);out.value(passed_final_gate);out.array(regulation_gate);out.array(actor);return out.data;
    }
    static PairedMotorPolicy read(const std::vector<uint8_t>& bytes) {
        checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=0x504D504Cu||in.value<uint32_t>()!=paired_motor_protocol_version) throw std::runtime_error("Incompatible paired motor policy version");
        PairedMotorPolicy out;out.interface_type=in.value<paired_motor::Interface>();out.gains=in.value<paired_motor::Gains>();out.update=in.value<int>();out.seed=in.value<uint32_t>();
        out.source_fingerprint=in.value<uint64_t>();out.environment_fingerprint=in.value<uint64_t>();out.passed_final_gate=in.value<bool>();out.regulation_gate=in.array<uint8_t>(1000000);
        out.actor=in.array<float>(paired_motor_specification().num_params());in.finish();(void)out.serialize();return out;
    }
    static PairedMotorPolicy load(const std::string& path) {return read(checkpoint::read(path));}
};
struct PairedMotorReadiness {
    uint64_t source_fingerprint=0,protocol_payload_hash=0,regulation_gate_hash=0,recorded_frames_hash=0;
    uint64_t torque_task_fingerprint=0,target_task_fingerprint=0;
    int recorded_frames=0;bool observations=false,effort=false,latent_replay=false,complete_resume=false;
    bool passed() const {return recorded_frames==3200&&observations&&effort&&latent_replay&&complete_resume;}
    std::vector<uint8_t> serialize() const {
        checkpoint::Writer out;out.value(uint32_t{0x504D5244u});out.value(paired_motor_protocol_version);out.value(source_fingerprint);out.value(protocol_payload_hash);out.value(regulation_gate_hash);
        out.value(recorded_frames_hash);out.value(torque_task_fingerprint);out.value(target_task_fingerprint);out.value(recorded_frames);out.value(observations);out.value(effort);out.value(latent_replay);out.value(complete_resume);return out.data;
    }
    static PairedMotorReadiness read(const std::vector<uint8_t>& bytes) {
        checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=0x504D5244u||in.value<uint32_t>()!=paired_motor_protocol_version) throw std::runtime_error("Incompatible paired readiness format");
        PairedMotorReadiness out;out.source_fingerprint=in.value<uint64_t>();out.protocol_payload_hash=in.value<uint64_t>();out.regulation_gate_hash=in.value<uint64_t>();out.recorded_frames_hash=in.value<uint64_t>();
        out.torque_task_fingerprint=in.value<uint64_t>();out.target_task_fingerprint=in.value<uint64_t>();out.recorded_frames=in.value<int>();out.observations=in.value<bool>();out.effort=in.value<bool>();out.latent_replay=in.value<bool>();out.complete_resume=in.value<bool>();in.finish();return out;
    }
    void require_passed(const paired_motor::RegulationGate& gate) const {
        if(!passed()||source_fingerprint!=gate.source_fingerprint||protocol_payload_hash!=gate.protocol_payload_hash||regulation_gate_hash!=paired_motor::fingerprint(gate.serialize())||!recorded_frames_hash||
            torque_task_fingerprint!=paired_motor_fingerprint(paired_motor::Interface::Torque,gate.protocol)||target_task_fingerprint!=paired_motor_fingerprint(paired_motor::Interface::JointTarget,gate.protocol))
            throw std::runtime_error("Paired motor experiment blocked: focused recorded-data readiness did not pass for this regulation record");
    }
};
}
#endif
