#ifndef CUDARL_COMMAND_MOTOR_PPO_H
#define CUDARL_COMMAND_MOTOR_PPO_H
#include "swimmer_ppo.h"
#include "command_motor_policy.h"

namespace cudarl::rl {
struct CommandMotorTask {
    using Batch=command_motor::Batch;using State=command_motor::State;
    static constexpr int inputs=command_motor::kObs,actions=command_motor::kAct,evaluation_steps=command_motor::kEpisodeSteps;
    static constexpr float entropy=.002f;
    static constexpr bool research_checks=true;
    static int collected(const State&) {return 0;} // no food in this task
    static void validate_buffer(const float* pointer,int count,cudaStream_t stream,const char* name) {
        std::vector<float> values(count);
        CUDA_CHECK_THROW(cudaMemcpyAsync(values.data(),pointer,values.size()*sizeof(float),cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        for(float value:values) if(!std::isfinite(value)) throw std::runtime_error(std::string("Motor research nonfinite ")+name+"; optimizer not advanced from this buffer");
    }
    static void validate_collected(const Batch& env,const float* obs,const float* sampled_actions,const float* logp,const float* values,
            const float* rewards,const float* done,const float* truncated,int batch,cudaStream_t stream) {
        if(env.invalid_count(stream)) throw std::runtime_error("Motor research stopped on invalid dynamics before GAE/Adam");
        if(env.unsafe_count(stream)) throw std::runtime_error("Motor research stopped on physical clearance breach before GAE/Adam; no rescue/reset/adoption");
        validate_buffer(obs,batch*inputs,stream,"rollout observations");validate_buffer(sampled_actions,batch*actions,stream,"sampled latent actions");
        validate_buffer(logp,batch,stream,"old action log probabilities");validate_buffer(values,batch,stream,"critic values");
        validate_buffer(rewards,batch,stream,"rewards");validate_buffer(done,batch,stream,"terminals");validate_buffer(truncated,batch,stream,"truncations");
    }
};
inline std::vector<float> initialize_command_motor(const std::vector<float>& accepted55) {
    const auto old=swimmer_specification(4,55),spec=command_motor_specification();
    if(accepted55.size()!=static_cast<size_t>(old.num_params())) throw std::invalid_argument("Command motor requires accepted55 actor feature layers");
    for(float x:accepted55) if(!std::isfinite(x)) throw std::invalid_argument("Nonfinite source motor feature layer");
    auto result=nn::orthogonal_params(spec,command_motor::head_seed,{std::sqrt(2.f),std::sqrt(2.f),.01f});
    // Preserve only exactly compatible physical body features. All new command
    // columns start at zero, not at unrelated old food/ray weights.
    for(int row=0;row<spec.out(0);++row) {
        for(int col=0;col<spec.in(0);++col)
            result[spec.weight_offset(0)+row*spec.in(0)+col]=(col>=9&&col<=32)?accepted55[old.weight_offset(0)+row*old.in(0)+col]:0.f;
        result[spec.bias_offset(0)+row]=accepted55[old.bias_offset(0)+row];
    }
    for(int row=0;row<spec.out(1);++row) {
        for(int col=0;col<spec.in(1);++col) result[spec.weight_offset(1)+row*spec.in(1)+col]=accepted55[old.weight_offset(1)+row*old.in(1)+col];
        result[spec.bias_offset(1)+row]=accepted55[old.bias_offset(1)+row];
    }
    return result;
}
struct CommandMotorTrainer:BasicSwimmerTrainer<CommandMotorTask> {
    using Base=BasicSwimmerTrainer<CommandMotorTask>;
    uint64_t source_fingerprint=0;
    explicit CommandMotorTrainer(const std::vector<float>& accepted55,uint64_t source):Base(command_motor_worlds,command_motor::training_seed),source_fingerprint(source) {
        actor.set_params(initialize_command_motor(accepted55));CUDA_CHECK_THROW(cudaDeviceSynchronize());
        if(actor_opt.steps()!=0||critic_opt.steps()!=0||std_opt.steps()!=0) throw std::runtime_error("Command motor requires fresh optimizers");
    }
    double update_seconds() {
        if(update_id>=command_motor_updates) throw std::runtime_error("Registered128-update motor budget exhausted");
        const double elapsed=Base::update_seconds();
        if(env.invalid_count(stream.value)) throw std::runtime_error("Motor pilot stopped on nonfinite dynamics/actions");
        return elapsed;
    }
    Stats update() {throw std::runtime_error("Command motor uses explicit bounded update_seconds and its own full snapshots");}
    void evaluate(uint32_t=0) {throw std::runtime_error("Use registered command motor physical benchmark");}
};
inline std::vector<uint8_t> command_motor_snapshot(const CommandMotorTrainer& trainer) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(trainer.stream.value));checkpoint::Writer out;
    out.value(uint32_t{0x434D5452u});out.value(uint32_t{2});out.value(uint32_t{sizeof(command_motor::State)});
    out.value(command_motor_fingerprint());out.value(trainer.source_fingerprint);out.value(trainer.n);out.value(trainer.seed);out.value(trainer.update_id);
    out.array(command_motor_protocol());out.array(trainer.actor.get_params());out.array(trainer.critic.get_params());
    std::vector<float> deviation(4);trainer.log_std.copy_to_host(deviation.data(),deviation.size());out.array(deviation);
    trainer.actor_opt.save(out);trainer.critic_opt.save(out);trainer.std_opt.save(out);trainer.env.save(out);return out.data;
}
inline void restore_command_motor(CommandMotorTrainer& trainer,const std::vector<uint8_t>& bytes) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(trainer.stream.value));checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x434D5452u||in.value<uint32_t>()!=2||in.value<uint32_t>()!=sizeof(command_motor::State)||
        in.value<uint64_t>()!=command_motor_fingerprint()||in.value<uint64_t>()!=trainer.source_fingerprint||
        in.value<int>()!=trainer.n||in.value<uint32_t>()!=trainer.seed) throw std::runtime_error("Incompatible command motor full save");
    const int update=in.value<int>();
    if(update<0||update>command_motor_updates||in.array<float>(command_motor_protocol().size())!=command_motor_protocol())
        throw std::runtime_error("Command motor save violates registered protocol/budget");
    const auto actor=in.array<float>(trainer.actor.num_params()),critic=in.array<float>(trainer.critic.num_params()),deviation=in.array<float>(4);
    if(actor.size()!=static_cast<size_t>(trainer.actor.num_params())||critic.size()!=static_cast<size_t>(trainer.critic.num_params())||deviation.size()!=4)
        throw std::runtime_error("Command motor learner dimensions mismatch");
    for(const auto* values:{&actor,&critic,&deviation}) for(float x:*values) if(!std::isfinite(x)) throw std::runtime_error("Nonfinite command motor learner");
    trainer.actor.set_params(actor);trainer.critic.set_params(critic);trainer.log_std.copy_from_host(deviation.data(),deviation.size());
    trainer.actor_opt.load(in);trainer.critic_opt.load(in);trainer.std_opt.load(in);trainer.env.load(in);in.finish();CUDA_CHECK_THROW(cudaDeviceSynchronize());
    if(trainer.actor_opt.steps()!=update*epochs*minibatches||trainer.critic_opt.steps()!=trainer.actor_opt.steps()||trainer.std_opt.steps()!=trainer.actor_opt.steps())
        throw std::runtime_error("Command motor saved Adam counters mismatch");
    const auto expected=CommandMotorTrainer::optimizer();
    for(const auto* optimizer:{&trainer.actor_opt,&trainer.critic_opt,&trainer.std_opt}) {
        const auto& c=optimizer->config();if(c.lr!=expected.lr||c.beta1!=expected.beta1||c.beta2!=expected.beta2||c.eps!=expected.eps||c.max_grad_norm!=expected.max_grad_norm)
            throw std::runtime_error("Command motor saved optimizer settings mismatch");
    }
    const int total_ticks=update*rollout;
    for(const auto& state:trainer.env.host_states(trainer.stream.value))
        if(state.steps!=total_ticks%command_motor::kEpisodeSteps||state.episode!=static_cast<uint32_t>(total_ticks/command_motor::kEpisodeSteps))
            throw std::runtime_error("Command motor world clock disagrees with its complete bounded rollout history");
    trainer.update_id=update;
}
}
#endif
