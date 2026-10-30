#ifndef CUDARL_RL_MOTOR_V2_POLICY_H
#define CUDARL_RL_MOTOR_V2_POLICY_H
#include "envs/swimmer/motor_v2.h"
#include "swimmer_spec.h"

namespace cudarl::rl {
constexpr uint32_t motor_v2_protocol_version=2,motor_v2_head_seed=0x4D082601u,motor_v2_seed=0x4D082602u,motor_v2_final_seed=0x4D082604u;
constexpr int motor_v2_updates=64,motor_v2_worlds=512,motor_v2_total_steps=4194304;
inline nn::MlpSpec motor_v2_specification(int outputs=4) {return swimmer_specification(outputs,59);}
inline const char* motor_v2_arm(motor_v2::Interface mode) {return mode==motor_v2::Interface::Torque?"torque":"joint_target";}
inline uint64_t motor_v2_hash(const std::vector<uint8_t>& bytes) {return paired_motor::fingerprint(bytes);}
inline std::vector<float> motor_v2_training_protocol() {
    // New physical/task protocol; the historical stopped comparison is immutable.
    std::vector<float> values={2,59,4,32,512,64,64,4,8,.99f,.95f,.2f,.002f,.5f,.001f,.9f,.999f,.00001f,.5f,-.5f,.01f};
    const auto task=motor_v2::protocol();values.insert(values.end(),task.begin(),task.end());return values;
}
inline uint64_t motor_v2_environment_fingerprint(motor_v2::Interface mode,const motor_v2::Gains& gains) {
    if(!paired_motor::valid_gains(gains,mode)) throw std::runtime_error("Invalid motor-v2 actuator mapping");
    checkpoint::Writer out;out.value(uint32_t{0x4D563250u});out.value(motor_v2_protocol_version);out.value(motor_v2::physics_fingerprint());
    out.array(motor_v2_training_protocol());out.value(mode);out.value(gains);out.value(motor_v2_head_seed);out.value(motor_v2_seed);out.value(motor_v2_final_seed);return motor_v2_hash(out.data);
}
inline std::vector<float> initialize_motor_v2(const std::vector<float>& accepted55) {
    const auto old=swimmer_specification(4,55),next=motor_v2_specification();
    if(accepted55.size()!=static_cast<size_t>(old.num_params())) throw std::runtime_error("Motor-v2 requires the original accepted55 feature source");
    for(float value:accepted55) if(!std::isfinite(value)) throw std::runtime_error("Nonfinite motor-v2 feature source");
    auto result=nn::orthogonal_params(next,motor_v2_head_seed,{std::sqrt(2.f),std::sqrt(2.f),.01f});
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
struct MotorV2Policy {
    motor_v2::Interface interface_type=motor_v2::Interface::Torque;motor_v2::Gains gains{};
    int update=0;uint32_t seed=motor_v2_seed;uint64_t source_fingerprint=0,environment_fingerprint=0;
    bool completed_budget=false,passed_final_gate=false;
    std::vector<float> actor;std::vector<uint8_t> historical_regulation_gate,contact_gate;
    bool has_trained_weights() const {return update>0;}
    void validate() const {
        const auto history=paired_motor::RegulationGate::read(historical_regulation_gate);history.require_passed(source_fingerprint);
        const auto contact=motor_contact_v2::ContactGate::read(contact_gate);
        if(contact.physics_hash!=motor_contact_v2::physics_fingerprint()||!paired_motor::same_gains(gains,history.gains)||
            environment_fingerprint!=motor_v2_environment_fingerprint(interface_type,gains)||seed!=motor_v2_seed||update<0||update>motor_v2_updates||
            completed_budget!=(update==motor_v2_updates)||(passed_final_gate&&!completed_budget)||
            actor.size()!=static_cast<size_t>(motor_v2_specification().num_params())) throw std::runtime_error("Motor-v2 policy version/source/physics/task/actuator contract differs");
        if(update>0&&!contact.passed()) throw std::runtime_error("Trained motor-v2 policy has no passed corrected-contact gate");
        for(float value:actor) if(!std::isfinite(value)) throw std::runtime_error("Nonfinite motor-v2 policy");
    }
    std::vector<uint8_t> serialize() const {
        validate();checkpoint::Writer out;out.value(uint32_t{0x4D56324Cu});out.value(motor_v2_protocol_version);out.value(interface_type);out.value(gains);out.value(update);out.value(seed);
        out.value(source_fingerprint);out.value(environment_fingerprint);out.value(completed_budget);out.value(passed_final_gate);out.array(historical_regulation_gate);out.array(contact_gate);out.array(actor);return out.data;
    }
    static MotorV2Policy read(const std::vector<uint8_t>& bytes) {
        checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=0x4D56324Cu||in.value<uint32_t>()!=motor_v2_protocol_version) throw std::runtime_error("Incompatible motor-v2 policy");
        MotorV2Policy out;out.interface_type=in.value<motor_v2::Interface>();out.gains=in.value<motor_v2::Gains>();out.update=in.value<int>();out.seed=in.value<uint32_t>();
        out.source_fingerprint=in.value<uint64_t>();out.environment_fingerprint=in.value<uint64_t>();out.completed_budget=in.value<bool>();out.passed_final_gate=in.value<bool>();
        out.historical_regulation_gate=in.array<uint8_t>(1000000);out.contact_gate=in.array<uint8_t>(1000000);out.actor=in.array<float>(motor_v2_specification().num_params());in.finish();out.validate();return out;
    }
    static MotorV2Policy load(const std::string& path) {return read(checkpoint::read(path));}
};
struct MotorV2Readiness {
    uint64_t source_fingerprint=0,physics_fingerprint=0,historical_gate_hash=0,contact_gate_hash=0,contact_frames_hash=0,torque_task_hash=0,target_task_hash=0;
    int recorded_frames=0;bool observations=false,actual_torque=false,latent_replay=false,complete_resume=false;
    bool passed() const {return recorded_frames==32&&observations&&actual_torque&&latent_replay&&complete_resume;}
    std::vector<uint8_t> serialize() const {
        checkpoint::Writer out;out.value(uint32_t{0x4D563244u});out.value(motor_v2_protocol_version);out.value(source_fingerprint);out.value(physics_fingerprint);
        out.value(historical_gate_hash);out.value(contact_gate_hash);out.value(contact_frames_hash);out.value(torque_task_hash);out.value(target_task_hash);out.value(recorded_frames);
        out.value(observations);out.value(actual_torque);out.value(latent_replay);out.value(complete_resume);return out.data;
    }
    static MotorV2Readiness read(const std::vector<uint8_t>& bytes) {
        checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=0x4D563244u||in.value<uint32_t>()!=motor_v2_protocol_version) throw std::runtime_error("Incompatible motor-v2 readiness");
        MotorV2Readiness out;out.source_fingerprint=in.value<uint64_t>();out.physics_fingerprint=in.value<uint64_t>();out.historical_gate_hash=in.value<uint64_t>();out.contact_gate_hash=in.value<uint64_t>();
        out.contact_frames_hash=in.value<uint64_t>();out.torque_task_hash=in.value<uint64_t>();out.target_task_hash=in.value<uint64_t>();out.recorded_frames=in.value<int>();
        out.observations=in.value<bool>();out.actual_torque=in.value<bool>();out.latent_replay=in.value<bool>();out.complete_resume=in.value<bool>();in.finish();return out;
    }
    void require_passed(uint64_t source,const std::vector<uint8_t>& history,const std::vector<uint8_t>& contact) const {
        const auto h=paired_motor::RegulationGate::read(history);h.require_passed(source);const auto c=motor_contact_v2::ContactGate::read(contact);
        if(!passed()||!c.passed()||source_fingerprint!=source||physics_fingerprint!=motor_v2::physics_fingerprint()||historical_gate_hash!=motor_v2_hash(history)||
            contact_gate_hash!=motor_v2_hash(contact)||!contact_frames_hash||torque_task_hash!=motor_v2_environment_fingerprint(motor_v2::Interface::Torque,h.gains)||
            target_task_hash!=motor_v2_environment_fingerprint(motor_v2::Interface::JointTarget,h.gains)) throw std::runtime_error("Motor-v2 corrected-contact/data readiness gate did not pass for these inputs");
    }
};
}
#endif
