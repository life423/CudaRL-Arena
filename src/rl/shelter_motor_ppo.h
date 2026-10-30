#ifndef CUDARL_SHELTER_MOTOR_PPO_H
#define CUDARL_SHELTER_MOTOR_PPO_H
#include "swimmer_ppo.h"
#include "envs/swimmer/shelter_motor.h"

namespace cudarl::rl {
constexpr uint32_t shelter_motor_protocol=5;
struct ShelterMotorTask {
    using Batch=shelter_motor::Batch;using State=shelter_motor::State;
    static constexpr int inputs=shelter_motor::kObs,actions=shelter_motor::kAct,evaluation_steps=1200;
    static constexpr float entropy=.002f;
    static int collected(const State& state) {return state.state.pond.eaten;}
};
// The actor's last layer is expressed in cover actuator units. Food means
// recover their inherited gain; cover means need no post-network division.
// Gaussian samples, likelihoods and deviations all remain physical quantities.
// Keep the old tasks' BasicSwimmerTrainer path byte-for-byte untouched.
struct ShelterMotorTrainer:BasicSwimmerTrainer<ShelterMotorTask> {
    using Base=BasicSwimmerTrainer<ShelterMotorTask>;
    device_ptr<float> physical_means;
    ShelterMotorTrainer(int count,uint32_t key,sw::Body body={}):Base(count,key,body),
        physical_means(static_cast<size_t>(std::max({n,mb,evaluation_count}))*kActions) {}

    double update_seconds() {
        const auto start=Clock::now();const auto s=stream.value;
        // The transform is part of this policy's protocol, not mutable world
        // state. Freeze it for the complete rollout and every PPO replay epoch.
        const auto scales=env.action_scales();
        for(int t=0;t<rollout;++t) {
            const size_t row=static_cast<size_t>(t)*n;
            swimmer_copy(obs.get()+row*kInputs,env.obs(),static_cast<size_t>(n)*kInputs,s);
            const float* input=obs.get()+row*kInputs;
            actor.forward(input,n,s);
            shelter_motor::policy_means(actor.output(),input,physical_means.get(),n,scales,s);
            sample_gaussian(physical_means.get(),log_std.get(),n,kActions,seed,
                static_cast<uint32_t>(update_id*rollout+t),actions.get()+row*kActions,logp.get()+row,s);
            critic.forward(input,n,s);swimmer_copy(values.get()+row,critic.output(),n,s);
            env.step(actions.get()+row*kActions,{rewards.get()+row,done.get()+row,truncated.get()+row,final_obs.get()+row*kInputs},s);
        }
        critic.forward(env.obs(),n,s);swimmer_copy(last_values.get(),critic.output(),n,s);
        critic.forward(final_obs.get(),batch,s);swimmer_copy(final_values.get(),critic.output(),batch,s);
        compute_gae(rollout,n,rewards.get(),values.get(),done.get(),truncated.get(),final_values.get(),last_values.get(),
            .99f,.95f,adv.get(),returns.get(),s);
        CUDA_CHECK_THROW(cudaMemsetAsync(stats.get(),0,stats.size_bytes(),s));
        for(int epoch=0;epoch<epochs;++epoch) {
            const auto perm=make_permutation(static_cast<uint32_t>(batch),rng::bits(seed,static_cast<uint32_t>(update_id),static_cast<uint32_t>(epoch)));
            for(int m=0;m<minibatches;++m) {
                const int first=m*mb;
                gather_rows(perm,first,mb,kInputs,obs.get(),mb_obs.get(),s);
                gather_rows(perm,first,mb,kActions,actions.get(),mb_actions.get(),s);
                gather_rows(perm,first,mb,1,logp.get(),mb_logp.get(),s);
                gather_rows(perm,first,mb,1,adv.get(),mb_adv.get(),s);
                gather_rows(perm,first,mb,1,returns.get(),mb_ret.get(),s);
                mean_std(mb_adv.get(),mb,adv_stats.get(),s);
                actor.forward(mb_obs.get(),mb,s);critic.forward(mb_obs.get(),mb,s);
                // Use the captured on-policy mode flag. Some live practice
                // worlds may have reset to a different task since this row.
                shelter_motor::policy_means(actor.output(),mb_obs.get(),physical_means.get(),mb,scales,s);
                gaussian_ppo_loss_grad(physical_means.get(),log_std.get(),critic.output(),mb,kActions,
                    mb_actions.get(),mb_logp.get(),mb_adv.get(),mb_ret.get(),adv_stats.get(),{.2f,ShelterMotorTask::entropy,.5f},
                    dmean.get(),dstd.get(),dvalue.get(),row_stats.get(),stats.get(),s);
                // Cover gradients already have physical actor units. Food
                // gradients include the inherited output gain. This is still
                // a Gaussian mean transform, not an action sample transform.
                shelter_motor::policy_mean_grad(dmean.get(),mb_obs.get(),mb,scales,s);
                actor.zero_grad(s);actor.backward(dmean.get(),mb,nullptr,s);actor_opt.step(actor.params(),actor.grads(),s);
                CUDA_CHECK_THROW(cudaMemsetAsync(std_grad.get(),0,std_grad.size_bytes(),s));
                sum_columns(dstd.get(),mb,kActions,std_grad.get(),s);std_opt.step(log_std.get(),std_grad.get(),s);
                critic.zero_grad(s);critic.backward(dvalue.get(),mb,nullptr,s);critic_opt.step(critic.params(),critic.grads(),s);
            }
        }
        CUDA_CHECK_THROW(cudaStreamSynchronize(s));++update_id;
        return std::chrono::duration<double>(Clock::now()-start).count();
    }
    Stats update() {
        if(evaluation_every>0 || !checkpoint_path_.empty())
            throw std::runtime_error("Use matched shelter evaluation and full shelter snapshots for this task");
        const auto result=current_stats(update_seconds()*1000.0);
        training_history_.push_back(result);return result;
    }
    void evaluate(uint32_t=0x51A7E123u) {
        throw std::runtime_error("Use matched shelter evaluation with its saved actuator normalization");
    }
};
inline std::vector<float> transfer_shelter_motor(const std::vector<float>& old,int outputs=4) {
    const auto before=swimmer_specification(outputs,arch::kObs),after=swimmer_specification(outputs,shelter_motor::kObs);
    if(old.size()!=static_cast<size_t>(before.num_params())) throw std::invalid_argument("Shelter motor transfer requires the inherited 55-input network");
    std::vector<float> result(static_cast<size_t>(after.num_params()),0.f);
    for(int layer=0;layer<before.num_layers();++layer) {
        for(int row=0;row<before.out(layer);++row) for(int col=0;col<before.in(layer);++col)
            result[after.weight_offset(layer)+row*after.in(layer)+col]=old[before.weight_offset(layer)+row*before.in(layer)+col];
        for(int row=0;row<before.out(layer);++row) result[after.bias_offset(layer)+row]=old[before.bias_offset(layer)+row];
    }
    return result;
}
// Reparameterize only the final linear layer. With the saved gain restored for
// food, this is algebraically the same initial physical policy as protocol 4,
// but Adam now moves cover output parameters in actuator rather than inherited
// raw-output units. Power-of-two gains also preserve the transfer's float scale
// exactly in the ordinary range; runtime retention tests remain mandatory.
inline std::vector<float> normalized_shelter_actor(const std::vector<float>& legacy59,shelter_motor::ActuatorScale scales) {
    const auto spec=swimmer_specification(shelter_motor::kAct,shelter_motor::kObs);
    if(legacy59.size()!=static_cast<size_t>(spec.num_params()))
        throw std::invalid_argument("Shelter conditioning requires a 59-input four-output actor");
    for(float value:legacy59) if(!std::isfinite(value)) throw std::invalid_argument("Non-finite source shelter actor");
    for(float value:scales.value) if(!std::isfinite(value) || value<1.f || value>1048576.f)
        throw std::invalid_argument("Invalid shelter actor output calibration");
    auto actor=legacy59;const int last=spec.num_layers()-1;
    for(int row=0;row<spec.out(last);++row) {
        const float gain=scales.value[row];
        for(int column=0;column<spec.in(last);++column)
            actor[spec.weight_offset(last)+row*spec.in(last)+column]/=gain;
        actor[spec.bias_offset(last)+row]/=gain;
    }
    return actor;
}
inline std::vector<uint8_t> shelter_motor_snapshot(const ShelterMotorTrainer& trainer) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(trainer.stream.value));checkpoint::Writer out;
    out.value(uint32_t{0x534D5452u});out.value(shelter_motor_protocol);out.value(uint32_t{sizeof(shelter_motor::State)});
    out.value(trainer.n);out.value(trainer.seed);out.value(trainer.update_id);
    out.value(trainer.env.action_scales());
    out.array(trainer.actor.get_params());out.array(trainer.critic.get_params());
    std::vector<float> stds(4);trainer.log_std.copy_to_host(stds.data(),stds.size());out.array(stds);
    trainer.actor_opt.save(out);trainer.critic_opt.save(out);trainer.std_opt.save(out);trainer.env.save(out);return out.data;
}
inline void restore_shelter_motor(ShelterMotorTrainer& trainer,const std::vector<uint8_t>& bytes) {
    checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x534D5452u || in.value<uint32_t>()!=shelter_motor_protocol || in.value<uint32_t>()!=sizeof(shelter_motor::State) ||
       in.value<int>()!=trainer.n || in.value<uint32_t>()!=trainer.seed) throw std::runtime_error("Incompatible shelter motor snapshot");
    const int update=in.value<int>();const auto scales=in.value<shelter_motor::ActuatorScale>();
    trainer.env.set_action_scales(scales);
    const auto actor=in.array<float>(trainer.actor.num_params()),critic=in.array<float>(trainer.critic.num_params()),stds=in.array<float>(4);
    if(update<0 || actor.size()!=static_cast<size_t>(trainer.actor.num_params()) || critic.size()!=static_cast<size_t>(trainer.critic.num_params()) || stds.size()!=4)
        throw std::runtime_error("Invalid shelter motor learner dimensions");
    for(const auto* values:{&actor,&critic,&stds}) for(float v:*values) if(!std::isfinite(v)) throw std::runtime_error("Non-finite shelter motor learner");
    trainer.actor.set_params(actor);trainer.critic.set_params(critic);trainer.log_std.copy_from_host(stds.data(),stds.size());
    trainer.actor_opt.load(in);trainer.critic_opt.load(in);trainer.std_opt.load(in);trainer.env.load(in);in.finish();
    for(int joint=0;joint<4;++joint) if(trainer.env.action_scales().value[joint]!=scales.value[joint])
        throw std::runtime_error("Inconsistent saved shelter action calibration");
    if(trainer.actor_opt.steps()!=update*epochs*minibatches || trainer.critic_opt.steps()!=trainer.actor_opt.steps() || trainer.std_opt.steps()!=trainer.actor_opt.steps())
        throw std::runtime_error("Shelter motor optimizer counter mismatch");
    trainer.update_id=update;
}
}
#endif
