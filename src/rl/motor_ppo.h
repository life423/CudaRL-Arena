#ifndef CUDARL_RL_MOTOR_PPO_H
#define CUDARL_RL_MOTOR_PPO_H
#include "swimmer_ppo.h"
#include "motor_policy.h"

namespace cudarl::rl {
struct MotorPhysicalFailure:std::runtime_error {using std::runtime_error::runtime_error;};
struct MotorTask {
    using Batch=motor::Batch;using State=motor::State;
    static constexpr int inputs=59,actions=4,evaluation_steps=600;static constexpr float entropy=.002f;static constexpr bool research_checks=true;
    static int collected(const State&) {return 0;}
    static void validate_buffer(const float* pointer,int count,cudaStream_t stream,const char* label) {
        std::vector<float> values(count);CUDA_CHECK_THROW(cudaMemcpyAsync(values.data(),pointer,values.size()*sizeof(float),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        for(float value:values) if(!std::isfinite(value)) throw std::runtime_error(std::string("Motor-v2 shared nonfinite ")+label);
    }
    static void validate_collected(const Batch& env,const float* obs,const float* latent,const float* logp,const float* values,const float* rewards,const float* done,const float* truncated,int batch,cudaStream_t stream) {
        if(env.invalid_count(stream)) throw std::runtime_error("Motor-v2 shared invalid/numerical dynamics: rejected BEFORE GAE/Adam");
        // A simultaneous data defect must not be misclassified as arm-local contact.
        validate_buffer(obs,batch*inputs,stream,"observations");validate_buffer(latent,batch*actions,stream,"raw Gaussian latent");validate_buffer(logp,batch,stream,"old latent logp");
        validate_buffer(values,batch,stream,"values");validate_buffer(rewards,batch,stream,"actual-effort rewards");validate_buffer(done,batch,stream,"terminals");validate_buffer(truncated,batch,stream,"truncations");
        if(env.unsafe_count(stream)) throw MotorPhysicalFailure("Motor-v2 excessive penetration/escape: affected arm rejected BEFORE GAE/Adam");
    }
};
struct MotorTrainer:BasicSwimmerTrainer<MotorTask> {
    using Base=BasicSwimmerTrainer<MotorTask>;
    motor::Interface interface_type;motor::Gains gains{};uint64_t source_fingerprint=0;
    std::vector<uint8_t> historical_regulation_gate,contact_gate;
    MotorTrainer(const std::vector<float>& accepted55,uint64_t source,motor::Interface mode,const std::vector<uint8_t>& history,const std::vector<uint8_t>& contact):
        Base(motor_worlds,motor_seed),interface_type(mode),source_fingerprint(source),historical_regulation_gate(history),contact_gate(contact) {
        const auto h=motor_history::RegulationGate::read(history);h.require_passed(source);const auto c=motor_contact::ContactGate::read(contact);
        if(!c.passed()||c.physics_hash!=motor_contact::physics_fingerprint()) throw std::runtime_error("Motor-v2 blocked by corrected-contact gate");
        gains=h.gains;env.configure(mode,gains,stream.value);actor.set_params(initialize_motor(accepted55));CUDA_CHECK_THROW(cudaDeviceSynchronize());
        if(actor_opt.steps()||critic_opt.steps()||std_opt.steps()) throw std::runtime_error("Motor-v2 must initialize all Adam states freshly");
    }
    double update_seconds() {if(update_id>=motor_updates) throw std::runtime_error("Motor-v2 arm64-update budget exhausted");return Base::update_seconds();}
    Stats update() {throw std::runtime_error("Use bounded motor-v2 update_seconds and versioned snapshots");}
    void evaluate(uint32_t=0) {throw std::runtime_error("Motor-v2 evaluates its fixed64 terminal cases only");}
};
inline std::vector<uint8_t> motor_snapshot(const MotorTrainer& trainer) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(trainer.stream.value));checkpoint::Writer out;out.value(uint32_t{0x4D563254u});out.value(motor_protocol_version);out.value(uint32_t{sizeof(motor::State)});
    out.value(trainer.interface_type);out.value(motor_environment_fingerprint(trainer.interface_type,trainer.gains));out.value(trainer.source_fingerprint);out.array(trainer.historical_regulation_gate);out.array(trainer.contact_gate);
    out.value(trainer.n);out.value(trainer.seed);out.value(trainer.update_id);out.array(motor_training_protocol());out.array(trainer.actor.get_params());out.array(trainer.critic.get_params());
    std::vector<float> sigma(4);trainer.log_std.copy_to_host(sigma.data(),sigma.size());out.array(sigma);trainer.actor_opt.save(out);trainer.critic_opt.save(out);trainer.std_opt.save(out);trainer.env.save(out);return out.data;
}
inline void restore_motor(MotorTrainer& trainer,const std::vector<uint8_t>& bytes) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(trainer.stream.value));checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x4D563254u||in.value<uint32_t>()!=motor_protocol_version||in.value<uint32_t>()!=sizeof(motor::State)||in.value<motor::Interface>()!=trainer.interface_type||
        in.value<uint64_t>()!=motor_environment_fingerprint(trainer.interface_type,trainer.gains)||in.value<uint64_t>()!=trainer.source_fingerprint||in.array<uint8_t>(1000000)!=trainer.historical_regulation_gate||in.array<uint8_t>(1000000)!=trainer.contact_gate||
        in.value<int>()!=trainer.n||in.value<uint32_t>()!=trainer.seed) throw std::runtime_error("Incompatible motor-v2 full save (oldphysics/partialv1 prohibited)");
    const int update=in.value<int>();if(update<0||update>motor_updates||in.array<float>()!=motor_training_protocol()) throw std::runtime_error("Motor-v2 budget/task protocol differs");
    const auto actor=in.array<float>(trainer.actor.num_params()),critic=in.array<float>(trainer.critic.num_params()),sigma=in.array<float>(4);
    if(actor.size()!=static_cast<size_t>(trainer.actor.num_params())||critic.size()!=static_cast<size_t>(trainer.critic.num_params())||sigma.size()!=4) throw std::runtime_error("Motor-v2 parameter dimensions differ");
    for(const auto* values:{&actor,&critic,&sigma}) for(float value:*values) if(!std::isfinite(value)) throw std::runtime_error("Nonfinite motor-v2 saved parameters");
    trainer.actor.set_params(actor);trainer.critic.set_params(critic);trainer.log_std.copy_from_host(sigma.data(),sigma.size());trainer.actor_opt.load(in);trainer.critic_opt.load(in);trainer.std_opt.load(in);trainer.env.load(in);in.finish();CUDA_CHECK_THROW(cudaDeviceSynchronize());
    const auto expected=MotorTrainer::optimizer();for(const auto* optimizer:{&trainer.actor_opt,&trainer.critic_opt,&trainer.std_opt}) {const auto c=optimizer->config();
        if(optimizer->steps()!=update*epochs*minibatches||c.lr!=expected.lr||c.beta1!=expected.beta1||c.beta2!=expected.beta2||c.eps!=expected.eps||c.max_grad_norm!=expected.max_grad_norm) throw std::runtime_error("Motor-v2 saved Adam counters/settings differ");}
    for(const auto& state:trainer.env.host_states(trainer.stream.value)) if(state.task.steps!=(update*rollout)%600||state.task.episode!=static_cast<uint32_t>((update*rollout)/600)) throw std::runtime_error("Motor-v2 saved world clock differs from completed rollout history");
    trainer.update_id=update;
}
inline MotorPolicy motor_policy(const MotorTrainer& trainer,bool accepted=false) {
    MotorPolicy out;out.interface_type=trainer.interface_type;out.gains=trainer.gains;out.update=trainer.update_id;out.seed=trainer.seed;out.source_fingerprint=trainer.source_fingerprint;
    out.environment_fingerprint=motor_environment_fingerprint(trainer.interface_type,trainer.gains);out.completed_budget=trainer.update_id==motor_updates;out.passed_final_gate=accepted;
    out.actor=trainer.actor.get_params();out.historical_regulation_gate=trainer.historical_regulation_gate;out.contact_gate=trainer.contact_gate;return out;
}
}
#endif
