#ifndef CUDARL_PAIRED_MOTOR_PPO_H
#define CUDARL_PAIRED_MOTOR_PPO_H
#include "swimmer_ppo.h"
#include "paired_motor_policy.h"

namespace cudarl::rl {
struct PairedMotorTask {
    using Batch=paired_motor::Batch;using State=paired_motor::State;
    static constexpr int inputs=59,actions=4,evaluation_steps=600;static constexpr float entropy=.002f;static constexpr bool research_checks=true;
    static int collected(const State&) {return 0;}
    static void validate_buffer(const float* pointer,int count,cudaStream_t stream,const char* label) {
        std::vector<float> values(count);CUDA_CHECK_THROW(cudaMemcpyAsync(values.data(),pointer,values.size()*sizeof(float),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        for(float value:values) if(!std::isfinite(value)) throw std::runtime_error(std::string("Paired motor nonfinite ")+label);
    }
    static void validate_collected(const Batch& env,const float* obs,const float* latent,const float* logp,const float* values,const float* rewards,const float* done,const float* truncated,int batch,cudaStream_t stream) {
        if(env.invalid_count(stream)||env.unsafe_count(stream)) throw std::runtime_error("Paired rollout rejected BEFORE GAE/Adam: invalid dynamics or full-body safety breach");
        validate_buffer(obs,batch*inputs,stream,"observations");validate_buffer(latent,batch*actions,stream,"raw Gaussian latent actions");validate_buffer(logp,batch,stream,"old latent logp");
        validate_buffer(values,batch,stream,"values");validate_buffer(rewards,batch,stream,"actual-effort rewards");validate_buffer(done,batch,stream,"terminals");validate_buffer(truncated,batch,stream,"truncations");
    }
};
struct PairedMotorTrainer:BasicSwimmerTrainer<PairedMotorTask> {
    using Base=BasicSwimmerTrainer<PairedMotorTask>;
    paired_motor::Interface interface_type;paired_motor::RegulationGate regulation;
    explicit PairedMotorTrainer(const std::vector<float>& accepted55,paired_motor::Interface mode,const paired_motor::RegulationGate& gate):
        Base(paired_motor_worlds,paired_motor_seed),interface_type(mode),regulation(gate) {
        regulation.validate_binding(regulation.source_fingerprint);env.configure(mode,regulation.gains,stream.value);
        actor.set_params(initialize_paired_motor(accepted55));CUDA_CHECK_THROW(cudaDeviceSynchronize());
        if(actor_opt.steps()||critic_opt.steps()||std_opt.steps()) throw std::runtime_error("Paired arm must initialize all Adam states freshly");
    }
    double update_seconds() {
        regulation.require_passed(regulation.source_fingerprint);
        if(update_id>=paired_motor_updates) throw std::runtime_error("Paired arm64-update budget exhausted");
        return Base::update_seconds(); // original raw Gaussian samples/logp, no servo Jacobian
    }
    Stats update() {throw std::runtime_error("Use bounded paired update_seconds and its new snapshots");}
    void evaluate(uint32_t=0) {throw std::runtime_error("Paired comparison evaluates only fixed final64 cases");}
};
inline std::vector<uint8_t> paired_motor_snapshot(const PairedMotorTrainer& trainer) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(trainer.stream.value));checkpoint::Writer out;
    out.value(uint32_t{0x504D5452u});out.value(paired_motor_protocol_version);out.value(uint32_t{sizeof(paired_motor::State)});out.value(trainer.interface_type);
    out.value(paired_motor_fingerprint(trainer.interface_type,trainer.regulation.protocol));out.array(trainer.regulation.serialize());out.value(trainer.n);out.value(trainer.seed);out.value(trainer.update_id);out.array(paired_motor_protocol());
    out.array(trainer.actor.get_params());out.array(trainer.critic.get_params());std::vector<float> sigma(4);trainer.log_std.copy_to_host(sigma.data(),sigma.size());out.array(sigma);
    trainer.actor_opt.save(out);trainer.critic_opt.save(out);trainer.std_opt.save(out);trainer.env.save(out);return out.data;
}
inline void restore_paired_motor(PairedMotorTrainer& trainer,const std::vector<uint8_t>& bytes) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(trainer.stream.value));checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x504D5452u||in.value<uint32_t>()!=paired_motor_protocol_version||in.value<uint32_t>()!=sizeof(paired_motor::State)||in.value<paired_motor::Interface>()!=trainer.interface_type||
        in.value<uint64_t>()!=paired_motor_fingerprint(trainer.interface_type,trainer.regulation.protocol)||in.array<uint8_t>(1000000)!=trainer.regulation.serialize()||in.value<int>()!=trainer.n||in.value<uint32_t>()!=trainer.seed)
        throw std::runtime_error("Incompatible new paired motor full save");
    const int update=in.value<int>();if(update<0||update>paired_motor_updates||in.array<float>(paired_motor_protocol().size())!=paired_motor_protocol()) throw std::runtime_error("Paired motor budget/protocol differs");
    const auto actor=in.array<float>(trainer.actor.num_params()),critic=in.array<float>(trainer.critic.num_params()),sigma=in.array<float>(4);
    if(actor.size()!=static_cast<size_t>(trainer.actor.num_params())||critic.size()!=static_cast<size_t>(trainer.critic.num_params())||sigma.size()!=4) throw std::runtime_error("Paired learner dimensions differ");
    for(const auto* values:{&actor,&critic,&sigma}) for(float value:*values) if(!std::isfinite(value)) throw std::runtime_error("Nonfinite paired learner");
    trainer.actor.set_params(actor);trainer.critic.set_params(critic);trainer.log_std.copy_from_host(sigma.data(),sigma.size());
    trainer.actor_opt.load(in);trainer.critic_opt.load(in);trainer.std_opt.load(in);trainer.env.load(in);in.finish();CUDA_CHECK_THROW(cudaDeviceSynchronize());
    const auto expected=PairedMotorTrainer::optimizer();for(const auto* optimizer:{&trainer.actor_opt,&trainer.critic_opt,&trainer.std_opt}) {
        const auto c=optimizer->config();if(optimizer->steps()!=update*epochs*minibatches||c.lr!=expected.lr||c.beta1!=expected.beta1||c.beta2!=expected.beta2||c.eps!=expected.eps||c.max_grad_norm!=expected.max_grad_norm)
            throw std::runtime_error("Paired Adam counters/settings differ");
    }
    for(const auto& state:trainer.env.host_states(trainer.stream.value)) if(state.task.steps!=(update*rollout)%600||state.task.episode!=static_cast<uint32_t>((update*rollout)/600))
        throw std::runtime_error("Paired world clock differs from complete rollout history");
    trainer.update_id=update;
}
inline PairedMotorPolicy paired_motor_policy(const PairedMotorTrainer& trainer,bool passed=false) {
    PairedMotorPolicy out;out.interface_type=trainer.interface_type;out.gains=trainer.regulation.gains;out.update=trainer.update_id;out.seed=trainer.seed;
    out.source_fingerprint=trainer.regulation.source_fingerprint;out.environment_fingerprint=paired_motor_fingerprint(out.interface_type,trainer.regulation.protocol);
    out.passed_final_gate=passed;out.actor=trainer.actor.get_params();out.regulation_gate=trainer.regulation.serialize();return out;
}
}
#endif
