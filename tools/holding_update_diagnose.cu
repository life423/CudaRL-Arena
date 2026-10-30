#include "rl/shelter_motor_run.h"
#include "rl/shelter_hold_cases.h"
#include "rl/gaussian_math.h"
#include "cuda/device.h"
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>

// A disposable, fixed-data diagnostic for protocol5. Nothing here changes the
// production policy, optimizer, task, reward or physical implementation.
namespace {
using namespace cudarl;
namespace fs=std::filesystem;
using Trainer=rl::ShelterMotorTrainer;
constexpr int inputs=shelter_motor::kObs,actions=shelter_motor::kAct;
uint64_t hash_bytes(const std::vector<uint8_t>& bytes) {
    uint64_t hash=14695981039346656037ull;
    for(auto byte:bytes) {hash^=byte;hash*=1099511628211ull;}return hash;
}
std::string json_field(const std::string& json,const std::string& key) {
    const auto position=json.find('"'+key+'"');
    if(position==std::string::npos) throw std::runtime_error("Reuse summary lacks "+key);
    auto begin=json.find(':',position);
    if(begin==std::string::npos) throw std::runtime_error("Invalid reuse summary field "+key);
    ++begin;while(begin<json.size() && std::isspace(static_cast<unsigned char>(json[begin]))) ++begin;
    return json.substr(begin,json.find_first_of(",}\r\n",begin)-begin);
}
void validate_reuse_summary(const fs::path& directory,uint64_t fingerprint) {
    std::ifstream file(directory/"summary.json");
    if(!file) throw std::runtime_error("Reuse requires a completed valid audit summary");
    const std::string json((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
    if(std::stoull(json_field(json,"source_payload_fingerprint"))!=fingerprint)
        throw std::runtime_error("Reuse audit and protected source fingerprints differ");
    for(const auto* key:{"gradient_additivity_verified","gradient_input_immutability_verified","source_file_unchanged","resume_full_snapshot_bit_identical"})
        if(json_field(json,key).find("true")!=0) throw std::runtime_error(std::string("Reuse audit is not valid: ")+key);
}
void require(bool condition,const std::string& message) {
    if(!condition) throw std::runtime_error(message);
}
template<class T> std::vector<T> host_copy(const T* pointer,size_t count,cudaStream_t stream) {
    std::vector<T> values(count);
    CUDA_CHECK_THROW(cudaMemcpyAsync(values.data(),pointer,count*sizeof(T),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));return values;
}
std::vector<float> host_copy(const device_ptr<float>& pointer,cudaStream_t stream) {
    return host_copy(pointer.get(),pointer.size(),stream);
}
void upload(device_ptr<float>& destination,const std::vector<float>& values,cudaStream_t stream) {
    require(destination.size()==values.size(),"Fixed-rollout buffer dimensions differ");
    // A pageable synchronous/default-stream H2D copy may return after staging,
    // before its DMA is visible to a cudaStreamNonBlocking consumer. Keep both
    // copy and use on the same stream, and preserve this host buffer until DMA
    // completion. This diagnostic deliberately favors ordering over overlap.
    CUDA_CHECK_THROW(cudaMemcpyAsync(destination.get(),values.data(),values.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
void finite(const std::vector<float>& values,const char* label) {
    for(float value:values) require(std::isfinite(value),std::string("Non-finite ")+label);
}
double dot(const std::vector<float>& a,const std::vector<float>& b,size_t begin=0,size_t end=0) {
    require(a.size()==b.size(),"Gradient dimensions differ");if(end==0) end=a.size();double total=0;
    for(size_t i=begin;i<end;++i) total+=static_cast<double>(a[i])*b[i];return total;
}
double norm(const std::vector<float>& values,size_t begin=0,size_t end=0) {return std::sqrt(dot(values,values,begin,end));}
double cosine(const std::vector<float>& a,const std::vector<float>& b,size_t begin=0,size_t end=0) {
    const double denominator=norm(a,begin,end)*norm(b,begin,end);
    return denominator>0?dot(a,b,begin,end)/denominator:0;
}
double additivity(const std::vector<float>& food,const std::vector<float>& cover,const std::vector<float>& total,size_t begin=0,size_t end=0) {
    if(end==0) end=total.size();double maximum=0;
    for(size_t i=begin;i<end;++i) maximum=std::max(maximum,std::abs(static_cast<double>(food[i])+cover[i]-total[i]));return maximum;
}
void require_additive(const std::vector<float>& food,const std::vector<float>& cover,const std::vector<float>& total,const char* label,bool exact=false) {
    require(food.size()==cover.size() && food.size()==total.size(),std::string("Additivity dimensions differ: ")+label);
    double squared_residual=0;
    for(size_t i=0;i<total.size();++i) {const double residual=static_cast<double>(food[i])+cover[i]-total[i];squared_residual+=residual*residual;}
    const double scale=norm(food)+norm(cover),tolerance=exact?0:2e-5*scale+1e-7;
    require(std::sqrt(squared_residual)<=tolerance,std::string("INVALID AUDIT: FOOD+COVER gradient additivity failed for ")+label+
        "; residual_l2="+std::to_string(std::sqrt(squared_residual))+", tolerance="+std::to_string(tolerance));
}
struct FixedRollout {
    int worlds=0,update=0;uint32_t seed=0;
    std::vector<float> obs,means,action,logp,value,reward,done,truncated,final_obs,final_value,last_value,advantage,returns;
    std::vector<uint8_t> serialize() const {
        checkpoint::Writer out;out.value(uint32_t{0x48464735u});out.value(worlds);out.value(update);out.value(seed);
        for(const auto* array:{&obs,&means,&action,&logp,&value,&reward,&done,&truncated,&final_obs,&final_value,&last_value,&advantage,&returns}) out.array(*array);
        return out.data;
    }
    static FixedRollout read(const std::vector<uint8_t>& bytes) {
        checkpoint::Reader in{bytes};require(in.value<uint32_t>()==0x48464735u,"Invalid fixed-rollout diagnostic format");
        FixedRollout out;out.worlds=in.value<int>();out.update=in.value<int>();out.seed=in.value<uint32_t>();
        require(out.worlds>=16 && out.worlds<=2048 && out.update>=0,"Invalid fixed-rollout identity");
        const auto maximum=static_cast<size_t>(out.worlds)*rl::rollout*inputs;
        for(auto* array:{&out.obs,&out.means,&out.action,&out.logp,&out.value,&out.reward,&out.done,&out.truncated,&out.final_obs,&out.final_value,&out.last_value,&out.advantage,&out.returns}) *array=in.array<float>(maximum);
        in.finish();
        require(out.means.size()==static_cast<size_t>(out.worlds)*rl::rollout*actions,"Invalid reused physical mean dimensions");
        for(const auto* array:{&out.obs,&out.means,&out.action,&out.logp,&out.value,&out.reward,&out.done,&out.truncated,&out.final_obs,&out.final_value,&out.last_value,&out.advantage,&out.returns}) finite(*array,"reused fixed rollout");
        for(float value:out.done) require(value==0 || value==1,"Invalid fixed-rollout done flag");
        for(float value:out.truncated) require(value==0 || value==1,"Invalid fixed-rollout truncation flag");
        require(out.serialize()==bytes,"Fixed-rollout framing changed on read");return out;
    }
    void load(Trainer& trainer) const {
        require(trainer.n==worlds && trainer.update_id==update && trainer.seed==seed,"Fixed-rollout trainer identity differs");
        const auto stream=trainer.stream.value;
        upload(trainer.obs,obs,stream);upload(trainer.actions,action,stream);upload(trainer.logp,logp,stream);upload(trainer.values,value,stream);
        upload(trainer.rewards,reward,stream);upload(trainer.done,done,stream);upload(trainer.truncated,truncated,stream);
        upload(trainer.final_obs,final_obs,stream);upload(trainer.final_values,final_value,stream);upload(trainer.last_values,last_value,stream);
        upload(trainer.adv,advantage,stream);upload(trainer.returns,returns,stream);
    }
};
FixedRollout collect(Trainer& trainer,int logical_sampler_update=-1) {
    const auto stream=trainer.stream.value;const auto scales=trainer.env.action_scales();
    const int sampler_update=logical_sampler_update<0?trainer.update_id:logical_sampler_update;
    device_ptr<float> all_means(static_cast<size_t>(trainer.batch)*actions);
    // This is the production protocol5 rollout and GAE, with a read-only copy
    // of physical means. No additional random draw or environment step occurs.
    for(int tick=0;tick<rl::rollout;++tick) {
        const size_t row=static_cast<size_t>(tick)*trainer.n;
        rl::swimmer_copy(trainer.obs.get()+row*inputs,trainer.env.obs(),static_cast<size_t>(trainer.n)*inputs,stream);
        const auto observation=trainer.obs.get()+row*inputs;
        trainer.actor.forward(observation,trainer.n,stream);
        shelter_motor::policy_means(trainer.actor.output(),observation,trainer.physical_means.get(),trainer.n,scales,stream);
        rl::swimmer_copy(all_means.get()+row*actions,trainer.physical_means.get(),static_cast<size_t>(trainer.n)*actions,stream);
        rl::sample_gaussian(trainer.physical_means.get(),trainer.log_std.get(),trainer.n,actions,trainer.seed,
            static_cast<uint32_t>(sampler_update*rl::rollout+tick),trainer.actions.get()+row*actions,trainer.logp.get()+row,stream);
        trainer.critic.forward(observation,trainer.n,stream);rl::swimmer_copy(trainer.values.get()+row,trainer.critic.output(),trainer.n,stream);
        trainer.env.step(trainer.actions.get()+row*actions,{trainer.rewards.get()+row,trainer.done.get()+row,trainer.truncated.get()+row,trainer.final_obs.get()+row*inputs},stream);
    }
    trainer.critic.forward(trainer.env.obs(),trainer.n,stream);rl::swimmer_copy(trainer.last_values.get(),trainer.critic.output(),trainer.n,stream);
    trainer.critic.forward(trainer.final_obs.get(),trainer.batch,stream);rl::swimmer_copy(trainer.final_values.get(),trainer.critic.output(),trainer.batch,stream);
    rl::compute_gae(rl::rollout,trainer.n,trainer.rewards.get(),trainer.values.get(),trainer.done.get(),trainer.truncated.get(),trainer.final_values.get(),trainer.last_values.get(),
        .99f,.95f,trainer.adv.get(),trainer.returns.get(),stream);
    FixedRollout out;out.worlds=trainer.n;out.update=trainer.update_id;out.seed=trainer.seed;
    out.obs=host_copy(trainer.obs,stream);out.means=host_copy(all_means,stream);out.action=host_copy(trainer.actions,stream);out.logp=host_copy(trainer.logp,stream);
    out.value=host_copy(trainer.values,stream);out.reward=host_copy(trainer.rewards,stream);out.done=host_copy(trainer.done,stream);out.truncated=host_copy(trainer.truncated,stream);
    out.final_obs=host_copy(trainer.final_obs,stream);out.final_value=host_copy(trainer.final_values,stream);out.last_value=host_copy(trainer.last_values,stream);
    out.advantage=host_copy(trainer.adv,stream);out.returns=host_copy(trainer.returns,stream);
    for(const auto* values:{&out.obs,&out.means,&out.action,&out.logp,&out.value,&out.reward,&out.advantage,&out.returns}) finite(*values,"on-policy rollout");
    return out;
}
__global__ void replay_logps(const float* means,const float* deviation,const float* action,int count,float* result) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<count) result[row]=rl::gaussian_logp(means+row*actions,deviation,action+row*actions,actions);
}
struct LogpCheck {double maximum=0,mean_maximum=0,value_maximum=0;int different=0;};
LogpCheck check_logp(Trainer& trainer,const FixedRollout& batch) {
    LogpCheck result;const auto stream=trainer.stream.value;device_ptr<float> logs(trainer.mb);
    for(int first=0;first<trainer.batch;first+=trainer.mb) {
        const float* input=trainer.obs.get()+static_cast<size_t>(first)*inputs;trainer.actor.forward(input,trainer.mb,stream);
        shelter_motor::policy_means(trainer.actor.output(),input,trainer.physical_means.get(),trainer.mb,trainer.env.action_scales(),stream);
        const auto means=host_copy(trainer.physical_means.get(),static_cast<size_t>(trainer.mb)*actions,stream);
        for(size_t i=0;i<means.size();++i) {
            const auto expected=batch.means[static_cast<size_t>(first)*actions+i];const double error=std::abs(static_cast<double>(means[i])-expected);
            result.mean_maximum=std::max(result.mean_maximum,error);
            require(error<=1e-5*(1+std::abs(expected)),"Saved physical rollout means do not replay from the protected actor");
        }
        trainer.critic.forward(input,trainer.mb,stream);const auto values=host_copy(trainer.critic.output(),trainer.mb,stream);
        for(int i=0;i<trainer.mb;++i) {
            const auto expected=batch.value[first+i];const double error=std::abs(static_cast<double>(values[i])-expected);
            result.value_maximum=std::max(result.value_maximum,error);
            require(error<=1e-5*(1+std::abs(expected)),"Saved rollout values do not replay from the protected critic");
        }
        replay_logps<<<(trainer.mb+255)/256,256,0,stream>>>(trainer.physical_means.get(),trainer.log_std.get(),trainer.actions.get()+static_cast<size_t>(first)*actions,trainer.mb,logs.get());
        CUDA_CHECK_THROW(cudaGetLastError());const auto replay=host_copy(logs,stream);
        for(int i=0;i<trainer.mb;++i) {const float old=batch.logp[first+i];result.maximum=std::max(result.maximum,std::abs(static_cast<double>(replay[i])-old));result.different+=replay[i]!=old;}
    }
    require(result.maximum<=1e-4,"On-policy physical log-probabilities do not replay before any update");return result;
}
void validate_fixed_gae(Trainer& trainer,const FixedRollout& batch) {
    const auto stream=trainer.stream.value;
    trainer.critic.forward(trainer.env.obs(),trainer.n,stream);
    const auto last=host_copy(trainer.critic.output(),trainer.n,stream);
    require(last==batch.last_value,"Reused post-rollout worlds do not reproduce their saved bootstrap values");
    trainer.critic.forward(trainer.final_obs.get(),trainer.batch,stream);
    const auto final=host_copy(trainer.critic.output(),trainer.batch,stream);
    require(final==batch.final_value,"Reused terminal observations do not reproduce their saved bootstrap values");
    rl::compute_gae(rl::rollout,trainer.n,trainer.rewards.get(),trainer.values.get(),trainer.done.get(),trainer.truncated.get(),trainer.final_values.get(),trainer.last_values.get(),
        .99f,.95f,trainer.adv.get(),trainer.returns.get(),stream);
    require(host_copy(trainer.adv,stream)==batch.advantage && host_copy(trainer.returns,stream)==batch.returns,"Reused GAE differs from the fixed on-policy rewards/values");
}
void gather(Trainer& trainer,int epoch,int minibatch) {
    const auto stream=trainer.stream.value;
    const auto permutation=rl::make_permutation(static_cast<uint32_t>(trainer.batch),rng::bits(trainer.seed,static_cast<uint32_t>(trainer.update_id),static_cast<uint32_t>(epoch)));
    const int first=minibatch*trainer.mb;
    rl::gather_rows(permutation,first,trainer.mb,inputs,trainer.obs.get(),trainer.mb_obs.get(),stream);
    rl::gather_rows(permutation,first,trainer.mb,actions,trainer.actions.get(),trainer.mb_actions.get(),stream);
    rl::gather_rows(permutation,first,trainer.mb,1,trainer.logp.get(),trainer.mb_logp.get(),stream);
    rl::gather_rows(permutation,first,trainer.mb,1,trainer.adv.get(),trainer.mb_adv.get(),stream);
    rl::gather_rows(permutation,first,trainer.mb,1,trainer.returns.get(),trainer.mb_ret.get(),stream);
    rl::mean_std(trainer.mb_adv.get(),trainer.mb,trainer.adv_stats.get(),stream);
}
struct Gradients {
    std::vector<float> actor,critic,deviation,physical_mean,actor_mean,std_rows,value;
};
struct SplitGradients {
    Gradients food,cover,total,food_unit;
    std::array<float,2> advantage_stats{};int food_rows=0,cover_rows=0;
};
SplitGradients gradients(Trainer& trainer) {
    const auto stream=trainer.stream.value;gather(trainer,0,0);
    trainer.actor.forward(trainer.mb_obs.get(),trainer.mb,stream);trainer.critic.forward(trainer.mb_obs.get(),trainer.mb,stream);
    shelter_motor::policy_means(trainer.actor.output(),trainer.mb_obs.get(),trainer.physical_means.get(),trainer.mb,trainer.env.action_scales(),stream);
    CUDA_CHECK_THROW(cudaMemsetAsync(trainer.stats.get(),0,trainer.stats.size_bytes(),stream));
    rl::gaussian_ppo_loss_grad(trainer.physical_means.get(),trainer.log_std.get(),trainer.critic.output(),trainer.mb,actions,
        trainer.mb_actions.get(),trainer.mb_logp.get(),trainer.mb_adv.get(),trainer.mb_ret.get(),trainer.adv_stats.get(),{.2f,rl::ShelterMotorTask::entropy,.5f},
        trainer.dmean.get(),trainer.dstd.get(),trainer.dvalue.get(),trainer.row_stats.get(),trainer.stats.get(),stream);
    const auto observations=host_copy(trainer.mb_obs,stream),mean=host_copy(trainer.dmean,stream),std_rows=host_copy(trainer.dstd,stream),value=host_copy(trainer.dvalue,stream);
    SplitGradients out;const auto advantage=host_copy(trainer.adv_stats,stream);std::copy_n(advantage.data(),2,out.advantage_stats.data());
    for(int i=0;i<trainer.mb;++i) {if(observations[static_cast<size_t>(i)*inputs+55]>=.5f) ++out.cover_rows;else ++out.food_rows;}
    auto contribution=[&](int mode,bool unit_jacobian) {
        Gradients result;result.physical_mean=mean;result.std_rows=std_rows;result.value=value;
        for(int row=0;row<trainer.mb;++row) {
            const bool cover=observations[static_cast<size_t>(row)*inputs+55]>=.5f;
            if(mode>=0 && cover!=(mode==1)) {
                for(int joint=0;joint<actions;++joint) {result.physical_mean[row*actions+joint]=0;result.std_rows[row*actions+joint]=0;}
                result.value[row]=0;
            }
        }
        // Mask AFTER the one shared production normalization and1/mb loss.
        // FOOD and COVER are additive contributions, not differently scaled
        // subset objectives. Gain=1 is only a Jacobian counterfactual.
        upload(trainer.dmean,result.physical_mean,stream);
        require(host_copy(trainer.dmean,stream)==result.physical_mean,"INVALID AUDIT: physical mean rows did not upload exactly");
        if(!unit_jacobian) shelter_motor::policy_mean_grad(trainer.dmean.get(),trainer.mb_obs.get(),trainer.mb,trainer.env.action_scales(),stream);
        result.actor_mean=host_copy(trainer.dmean,stream);
        for(int row=0;row<trainer.mb;++row) for(int joint=0;joint<actions;++joint) {
            const int index=row*actions+joint;const bool cover=observations[static_cast<size_t>(row)*inputs+55]>=.5f;
            const float gain=unit_jacobian || cover?1.f:trainer.env.action_scales().value[joint];
            require(result.actor_mean[index]==result.physical_mean[index]*gain,"INVALID AUDIT: uploaded physical-to-actor gradient transform differs");
        }
        trainer.actor.forward(trainer.mb_obs.get(),trainer.mb,stream);trainer.actor.zero_grad(stream);
        trainer.actor.backward(trainer.dmean.get(),trainer.mb,nullptr,stream);CUDA_CHECK_THROW(cudaStreamSynchronize(stream));result.actor=trainer.actor.get_grads();
        require(host_copy(trainer.dmean,stream)==result.actor_mean,"INVALID AUDIT: actor backward mutated its input gradient");
        upload(trainer.dstd,result.std_rows,stream);CUDA_CHECK_THROW(cudaMemsetAsync(trainer.std_grad.get(),0,trainer.std_grad.size_bytes(),stream));
        require(host_copy(trainer.dstd,stream)==result.std_rows,"INVALID AUDIT: logstd rows did not upload exactly");
        rl::sum_columns(trainer.dstd.get(),trainer.mb,actions,trainer.std_grad.get(),stream);result.deviation=host_copy(trainer.std_grad,stream);
        require(host_copy(trainer.dstd,stream)==result.std_rows,"INVALID AUDIT: std reduction input differs from uploaded rows");
        for(int joint=0;joint<actions;++joint) {
            double expected=0,magnitude=0;for(int row=0;row<trainer.mb;++row) {expected+=result.std_rows[row*actions+joint];magnitude+=std::abs(result.std_rows[row*actions+joint]);}
            require(std::abs(result.deviation[joint]-expected)<=2e-5*magnitude+1e-7,"INVALID AUDIT: std reduction differs from immutable row sum");
        }
        upload(trainer.dvalue,result.value,stream);trainer.critic.forward(trainer.mb_obs.get(),trainer.mb,stream);trainer.critic.zero_grad(stream);
        trainer.critic.backward(trainer.dvalue.get(),trainer.mb,nullptr,stream);CUDA_CHECK_THROW(cudaStreamSynchronize(stream));result.critic=trainer.critic.get_grads();
        require(host_copy(trainer.dvalue,stream)==result.value,"INVALID AUDIT: critic backward mutated its input gradient");
        finite(result.actor,"actor gradient");finite(result.critic,"critic gradient");finite(result.deviation,"std gradient");return result;
    };
    out.food=contribution(0,false);out.cover=contribution(1,false);out.total=contribution(-1,false);out.food_unit=contribution(0,true);
    require_additive(out.food.physical_mean,out.cover.physical_mean,out.total.physical_mean,"physical mean rows",true);
    require_additive(out.food.actor_mean,out.cover.actor_mean,out.total.actor_mean,"chained actor mean rows",true);
    require_additive(out.food.std_rows,out.cover.std_rows,out.total.std_rows,"logstd rows",true);
    require_additive(out.food.value,out.cover.value,out.total.value,"critic value rows",true);
    require_additive(out.food.actor,out.cover.actor,out.total.actor,"actor parameters");
    require_additive(out.food.critic,out.cover.critic,out.total.critic,"critic parameters");
    require_additive(out.food.deviation,out.cover.deviation,out.total.deviation,"shared logstd parameters");return out;
}
std::vector<uint8_t> optimizer_bytes(const nn::GpuAdam& optimizer) {checkpoint::Writer out;optimizer.save(out);return out.data;}
struct UpdateCopy {
    std::string name;std::vector<float> actor,deviation;float actor_norm=0,std_norm=0,actor_clip=1,std_clip=1;int steps=0;
};
float clip_scale(const nn::AdamConfig& config,float gradient_norm) {return config.max_grad_norm>0?fminf(1.f,config.max_grad_norm/(gradient_norm+1e-6f)):1.f;}
UpdateCopy one_step(const std::string& name,const Trainer& original,const Gradients& gradient) {
    const auto stream=original.stream.value;nn::GpuMlp actor(original.actor.spec(),original.mb,7);actor.set_params(original.actor.get_params());
    nn::GpuAdam actor_opt(actor.num_params(),original.actor_opt.config()),std_opt(actions,original.std_opt.config());
    const auto actor_snapshot=optimizer_bytes(original.actor_opt),std_snapshot=optimizer_bytes(original.std_opt);
    checkpoint::Reader actor_input{actor_snapshot},std_input{std_snapshot};actor_opt.load(actor_input);std_opt.load(std_input);actor_input.finish();std_input.finish();
    CUDA_CHECK_THROW(cudaDeviceSynchronize()); // Fence default-stream parameter/moment restoration.
    device_ptr<float> actor_gradient(gradient.actor.size()),std_gradient(actions),deviation(actions);
    upload(actor_gradient,gradient.actor,stream);upload(std_gradient,gradient.deviation,stream);
    const auto old_std=host_copy(original.log_std,stream);upload(deviation,old_std,stream);
    actor_opt.step(actor.params(),actor_gradient.get(),stream);std_opt.step(deviation.get(),std_gradient.get(),stream);CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    UpdateCopy result;result.name=name;result.actor=actor.get_params();result.deviation=host_copy(deviation,stream);
    result.actor_norm=actor_opt.last_grad_norm();result.std_norm=std_opt.last_grad_norm();
    result.actor_clip=clip_scale(actor_opt.config(),result.actor_norm);result.std_clip=clip_scale(std_opt.config(),result.std_norm);result.steps=actor_opt.steps();return result;
}
std::vector<float> physical_means(Trainer& buffers,const std::vector<float>& parameters,const FixedRollout& batch) {
    const auto stream=buffers.stream.value;nn::GpuMlp actor(buffers.actor.spec(),buffers.mb,7);actor.set_params(parameters);
    CUDA_CHECK_THROW(cudaDeviceSynchronize()); // set_params uses the default stream internally.
    std::vector<float> out(batch.means.size());
    for(int first=0;first<buffers.batch;first+=buffers.mb) {
        const float* input=buffers.obs.get()+static_cast<size_t>(first)*inputs;actor.forward(input,buffers.mb,stream);
        shelter_motor::policy_means(actor.output(),input,buffers.physical_means.get(),buffers.mb,buffers.env.action_scales(),stream);
        const auto values=host_copy(buffers.physical_means.get(),static_cast<size_t>(buffers.mb)*actions,stream);
        std::copy(values.begin(),values.end(),out.begin()+static_cast<size_t>(first)*actions);
    }
    finite(out,"updated physical means");return out;
}
std::vector<float> physical_logps(const std::vector<float>& means,const std::vector<float>& deviation,const std::vector<float>& action) {
    require(means.size()==action.size() && deviation.size()==actions,"Invalid likelihood diagnostic dimensions");
    const auto rows=means.size()/actions;device_ptr<float> native_means(means.size()),native_action(action.size()),native_std(actions),logs(rows);
    upload(native_means,means,nullptr);upload(native_action,action,nullptr);upload(native_std,deviation,nullptr);
    replay_logps<<<(static_cast<int>(rows)+255)/256,256>>>(native_means.get(),native_std.get(),native_action.get(),static_cast<int>(rows),logs.get());
    CUDA_CHECK_THROW(cudaGetLastError());return host_copy(logs,nullptr);
}
struct ModeMetrics {
    int rows=0,clipped=0;
    double kl=0,approximate_kl=0;
    std::array<double,actions> delta{},clipped_delta{},sample_delta{},clipped_sample_delta{},old_saturated{},new_saturated{},old_sample_saturated{},new_sample_saturated{};
};
void write_effects(std::ofstream& out,const std::string& name,const FixedRollout& batch,const std::vector<float>& next,
                   const std::vector<float>& old_std,const std::vector<float>& new_std) {
    ModeMetrics modes[2];const auto new_logps=physical_logps(next,new_std,batch.action);
    for(size_t row=0;row<batch.logp.size();++row) {
        const int mode=batch.obs[row*inputs+55]>=.5f?1:0;auto& result=modes[mode];++result.rows;
        for(int j=0;j<actions;++j) {
            const size_t i=row*actions+j;const double before=batch.means[i],after=next[i],old_sigma=std::exp(old_std[j]),sigma=std::exp(new_std[j]);
            const double epsilon=(batch.action[i]-before)/old_sigma,new_sample=after+sigma*epsilon;
            const auto clipped=[](double x) {return std::max(-1.,std::min(1.,x));};
            const double delta=after-before,cdelta=clipped(after)-clipped(before),sdelta=new_sample-batch.action[i],csdelta=clipped(new_sample)-clipped(batch.action[i]);
            result.delta[j]+=delta*delta;result.clipped_delta[j]+=cdelta*cdelta;result.sample_delta[j]+=sdelta*sdelta;result.clipped_sample_delta[j]+=csdelta*csdelta;
            result.old_saturated[j]+=std::abs(before)>=1;result.new_saturated[j]+=std::abs(after)>=1;
            result.old_sample_saturated[j]+=std::abs(batch.action[i])>=1;result.new_sample_saturated[j]+=std::abs(new_sample)>=1;
            result.kl+=new_std[j]-old_std[j]+(old_sigma*old_sigma+delta*delta)/(2*sigma*sigma)-.5;
        }
        // Use the actual native latent-Gaussian likelihood and stored old_logp;
        // torque clipping is measured separately and never substituted in PPO.
        const double log_ratio=static_cast<double>(new_logps[row])-batch.logp[row];
        require(log_ratio<700,"Diagnostic sample likelihood ratio overflow");
        result.approximate_kl+=std::expm1(log_ratio)-log_ratio;
        result.clipped+=log_ratio<std::log(.8) || log_ratio>std::log(1.2);
    }
    for(int mode=0;mode<2;++mode) {const auto& r=modes[mode];require(r.rows>0,"Missing FOOD or COVER observations in rollout");
        for(int joint=0;joint<actions;++joint) out<<name<<','<<(mode?"cover":"food")<<','<<joint<<','<<r.rows<<','
            <<std::sqrt(r.delta[joint]/r.rows)<<','<<std::sqrt(r.clipped_delta[joint]/r.rows)<<','<<std::sqrt(r.sample_delta[joint]/r.rows)<<','<<std::sqrt(r.clipped_sample_delta[joint]/r.rows)<<','
            <<std::exp(old_std[joint])<<','<<std::exp(new_std[joint])<<','<<r.old_saturated[joint]/r.rows<<','<<r.new_saturated[joint]/r.rows<<','
            <<r.old_sample_saturated[joint]/r.rows<<','<<r.new_sample_saturated[joint]/r.rows<<','<<r.kl/r.rows<<','<<r.approximate_kl/r.rows<<','<<double(r.clipped)/r.rows<<'\n';
    }
}
constexpr uint32_t nominal_seed=0x481D2026u,fresh_near_seed=0x48D1F7A9u,food_comparison_seed=0x571EF001u;
std::vector<shelter_motor::State> nominal_worlds(const rl::BoulderSource& source) {
    std::vector<shelter_motor::State> worlds;const deep::V eye{-8,4,-1.25f};
    for(int goal=0;goal<predator::kCoverCount;++goal) {
        const auto cover=predator::cover(goal,eye,source.frame.habitat,{2});require(cover.valid && cover.exposure<=.05f,"Nominal cover pose is not physically safe");
        // Match holding_diagnose exactly. Copying the personal frame here
        // would retain unrelated food/history observations and old clocks.
        shelter_motor::State state;state.state=arch::initial(nominal_seed,0,static_cast<uint32_t>(goal),false,{2});
        state.state.habitat=source.frame.habitat;state.state.pond.body=predator::cover_pose(cover);
        state.state.pond.target=-1;state.state.pond.decision=false;state.state.pond.steps=0;state.state.pond.since_food=0;
        state.state.pond.anchor=state.state.pond.body.position;state.held_ticks=state.peak_hold=0;
        state.request.cover=true;state.request.eye=eye;state.request.goal.position=cover.position;state.request.goal.cover=goal;state.request.goal.valid=true;
        state.request.heading=deep::axis_rotation({0,1,0},cover.yaw).rotate({1,0,0});
        state.exposure=predator::exposure(state.state.pond.body,eye,state.state.habitat,{2});worlds.push_back(state);
    }
    return worlds;
}
std::vector<float> nominal_observations(const rl::BoulderSource& source) {
    std::vector<float> observations(predator::kCoverCount*inputs);const auto worlds=nominal_worlds(source);
    for(int i=0;i<predator::kCoverCount;++i) shelter_motor::observe(worlds[i],observations.data()+i*inputs,{2});
    return observations;
}
std::vector<float> nominal_means(const std::vector<float>& parameters,const std::vector<float>& observations,shelter_motor::ActuatorScale scales) {
    rl::SwimmerStream stream;nn::GpuMlp actor(rl::swimmer_specification(actions,inputs),predator::kCoverCount,7);actor.set_params(parameters);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    device_ptr<float> obs(observations.size()),mean(predator::kCoverCount*actions);upload(obs,observations,stream.value);
    actor.forward(obs.get(),predator::kCoverCount,stream.value);shelter_motor::policy_means(actor.output(),obs.get(),mean.get(),predator::kCoverCount,scales,stream.value);
    return host_copy(mean,stream.value);
}
void write_nominal(std::ofstream& out,const std::string& name,const std::vector<float>& before,const std::vector<float>& after) {
    for(int goal=0;goal<predator::kCoverCount;++goal) {
        double projection=0,length_delta=0,length_safe=0;
        for(int j=0;j<actions;++j) {const int index=goal*actions+j;const double delta=after[index]-before[index];projection+=delta*(-before[index]);length_delta+=delta*delta;length_safe+=double(before[index])*before[index];}
        const double alignment=length_delta>0 && length_safe>0?projection/std::sqrt(length_delta*length_safe):0;
        for(int j=0;j<actions;++j) {const int index=goal*actions+j;out<<name<<','<<goal<<','<<j<<','<<before[index]<<','<<after[index]<<','<<after[index]-before[index]<<','<<projection<<','<<alignment<<'\n';}
    }
}
std::vector<rl::ShelterProbeCase> frozen_holding_cases(const rl::BoulderSource& source) {
    std::vector<rl::ShelterProbeCase> cases;
    for(const auto& world:nominal_worlds(source)) cases.push_back({world.state,world.request.goal,world.request.eye,{},{2}});
    const auto near=rl::shelter_hold_cases(source.frame,fresh_near_seed);cases.insert(cases.end(),near.begin(),near.end());return cases;
}
struct FrozenEvaluation {int nominal_holds=0,near_holds=0;double calm_rate=0,flow_rate=0;};
FrozenEvaluation evaluate_frozen(const std::string& name,const std::vector<float>& actor,const rl::BoulderSource& source,
                                shelter_motor::ActuatorScale scales,const std::vector<rl::ShelterProbeCase>& cases,
                                std::ofstream& detailed,std::ofstream& summary,std::vector<uint8_t>* numeric_capture=nullptr) {
    std::cout<<"HOLDING_AUDIT_FROZEN_EVAL variant="<<name<<" nominal=4 near=24 food_worlds=16 food_seconds=30 training=0\n"<<std::flush;
    const auto parameters=actor;const auto reports=rl::probe_shelter_motor(actor,cases,1,200,scales);
    require(reports.size()==28,"Frozen holding suite dimensions changed");FrozenEvaluation result;
    double longest[2]{},mean_exposure[2]{};float minimum_clearance[2]={1e10f,1e10f};
    for(size_t i=0;i<reports.size();++i) {
        const auto& report=reports[i];const int suite=i<predator::kCoverCount?0:1;const bool held=report.longest_safe_dwell_ticks>=100;
        require(report.finite && std::isfinite(report.minimum_clearance) && std::isfinite(report.mean_exposure),"Non-finite frozen holding result");
        const auto& final_body=report.final_state.pond.body;const auto q=final_body.orientation;
        require(!report.final_state.pond.invalid && std::isfinite(deep::norm(final_body.position)) && std::isfinite(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w),"Invalid frozen final physical body");
        for(float value:final_body.angle) require(std::isfinite(value),"Non-finite frozen final joint");
        for(float value:final_body.velocity) require(std::isfinite(value),"Non-finite frozen final velocity");
        if(suite==0) result.nominal_holds+=held;else result.near_holds+=held;
        longest[suite]+=report.longest_safe_dwell_ticks*deep::dt;mean_exposure[suite]+=report.mean_exposure;
        minimum_clearance[suite]=fminf(minimum_clearance[suite],report.minimum_clearance);
        detailed<<name<<','<<(suite?"fresh_near":"matched_nominal")<<','<<i<<','<<cases[i].goal.cover<<','<<report.longest_safe_dwell_ticks*deep::dt<<','
            <<held<<','<<report.mean_exposure<<','<<report.last_exposure<<','<<report.minimum_clearance<<','<<report.final_goal_distance<<','<<report.contacts<<",1\n";
    }
    result.calm_rate=rl::shelter_motor_food_rate(actor,source,food_comparison_seed,16,600,{},scales);
    result.flow_rate=rl::shelter_motor_food_rate(actor,source,food_comparison_seed,16,600,{.0075f,0,0},scales);
    require(actor==parameters,"Frozen evaluation changed diagnostic policy parameters");
    if(numeric_capture) {
        // Serialize named outputs, not struct padding, for an exact repeated
        // inference check on the no-update control.
        checkpoint::Writer capture;
        for(const auto& r:reports) {
            for(float value:{r.minimum_goal_distance,r.mean_exposure,r.last_exposure,r.final_goal_distance,r.minimum_clearance}) capture.value(value);
            for(int value:{r.arrival_step,r.safe_dwell_ticks,r.longest_safe_dwell_ticks,r.contacts}) capture.value(value);capture.value(r.finite);
            const auto& body=r.final_state.pond.body;capture.value(body.position);capture.value(body.orientation);
            for(float value:body.angle) capture.value(value);for(float value:body.velocity) capture.value(value);capture.value(body.work);capture.value(body.travel);
            for(const auto& food:r.final_state.pond.pellets) {capture.value(food.active);if(food.active) {capture.value(food.id);capture.value(food.position);capture.value(food.velocity);}}
        }
        capture.value(result.nominal_holds);capture.value(result.near_holds);capture.value(result.calm_rate);capture.value(result.flow_rate);*numeric_capture=std::move(capture.data);
    }
    summary<<name<<','<<result.nominal_holds<<",4,"<<result.near_holds<<",24,"<<longest[0]/4<<','<<longest[1]/24<<','
        <<mean_exposure[0]/4<<','<<mean_exposure[1]/24<<','<<minimum_clearance[0]<<','<<minimum_clearance[1]<<','<<result.calm_rate<<','<<result.flow_rate<<",1,0\n";
    detailed.flush();summary.flush();require(bool(detailed) && bool(summary),"Could not write frozen evaluation results");
    std::cout<<"HOLDING_AUDIT_FROZEN_DONE variant="<<name<<" nominal_holds="<<result.nominal_holds<<"/4 fresh_near_holds="<<result.near_holds
        <<"/24 calm_food_min="<<result.calm_rate<<" flow075_food_min="<<result.flow_rate<<" trained_during_evaluation=0\n"<<std::flush;
    return result;
}
void save_audit_policy(const fs::path& output,const std::string& name,const std::vector<float>& actor,const std::vector<float>& deviation,
                       const rl::ShelterMotorRun& source,int actor_steps) {
    // Deliberately not a resumable trainer: isolated one-step Adam copies have
    // partial optimizer counters. The distinct artifact magic prevents them
    // from being mistaken for an approved policy or complete motor run.
    checkpoint::Writer out;out.value(uint32_t{0x48415035u});out.value(uint32_t{1});out.value(uint32_t{5});
    out.value(source.source.fingerprint);out.value(source.update);out.value(actor_steps);out.value(source.scales);out.array(actor);out.array(deviation);
    checkpoint::write_atomic((output/(name+"-audit-policy.bin")).u8string(),out.data);
}
void replay_update(Trainer& trainer,std::ofstream* trace) {
    const auto stream=trainer.stream.value;const auto scales=trainer.env.action_scales();
    CUDA_CHECK_THROW(cudaMemsetAsync(trainer.stats.get(),0,trainer.stats.size_bytes(),stream));
    // Exact production PPO replay loop; only data collection is replaced by
    // the one serialized on-policy rollout. The critic updates normally here.
    for(int epoch=0;epoch<rl::epochs;++epoch) for(int minibatch=0;minibatch<rl::minibatches;++minibatch) {
        gather(trainer,epoch,minibatch);trainer.actor.forward(trainer.mb_obs.get(),trainer.mb,stream);trainer.critic.forward(trainer.mb_obs.get(),trainer.mb,stream);
        shelter_motor::policy_means(trainer.actor.output(),trainer.mb_obs.get(),trainer.physical_means.get(),trainer.mb,scales,stream);
        rl::gaussian_ppo_loss_grad(trainer.physical_means.get(),trainer.log_std.get(),trainer.critic.output(),trainer.mb,actions,
            trainer.mb_actions.get(),trainer.mb_logp.get(),trainer.mb_adv.get(),trainer.mb_ret.get(),trainer.adv_stats.get(),{.2f,rl::ShelterMotorTask::entropy,.5f},
            trainer.dmean.get(),trainer.dstd.get(),trainer.dvalue.get(),trainer.row_stats.get(),trainer.stats.get(),stream);
        shelter_motor::policy_mean_grad(trainer.dmean.get(),trainer.mb_obs.get(),trainer.mb,scales,stream);
        trainer.actor.zero_grad(stream);trainer.actor.backward(trainer.dmean.get(),trainer.mb,nullptr,stream);trainer.actor_opt.step(trainer.actor.params(),trainer.actor.grads(),stream);
        CUDA_CHECK_THROW(cudaMemsetAsync(trainer.std_grad.get(),0,trainer.std_grad.size_bytes(),stream));rl::sum_columns(trainer.dstd.get(),trainer.mb,actions,trainer.std_grad.get(),stream);
        trainer.std_opt.step(trainer.log_std.get(),trainer.std_grad.get(),stream);
        trainer.critic.zero_grad(stream);trainer.critic.backward(trainer.dvalue.get(),trainer.mb,nullptr,stream);trainer.critic_opt.step(trainer.critic.params(),trainer.critic.grads(),stream);
        if(trace) {const auto a=trainer.actor_opt.last_grad_norm(),s=trainer.std_opt.last_grad_norm(),c=trainer.critic_opt.last_grad_norm();
            *trace<<epoch<<','<<minibatch<<','<<a<<','<<clip_scale(trainer.actor_opt.config(),a)<<','<<s<<','<<clip_scale(trainer.std_opt.config(),s)<<','<<c<<','<<clip_scale(trainer.critic_opt.config(),c)<<'\n';}
    }
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));++trainer.update_id;
}
void write_alignment(std::ofstream& out,const char* kind,int layer,const Gradients& food,const Gradients& cover,const Gradients& total,const Gradients& unit,size_t first,size_t end,bool critic=false) {
    const auto& f=critic?food.critic:food.actor;const auto& c=critic?cover.critic:cover.actor;const auto& t=critic?total.critic:total.actor;const auto& u=critic?unit.critic:unit.actor;
    const double fn=norm(f,first,end),cn=norm(c,first,end),tn=norm(t,first,end),un=norm(u,first,end);
    out<<kind<<','<<layer<<','<<fn<<','<<cn<<','<<tn<<','<<un<<','<<(un>0?fn/un:0)<<','<<dot(f,c,first,end)<<','<<cosine(f,c,first,end)<<','
       <<(fn+cn>0?tn/(fn+cn):0)<<','<<additivity(f,c,t,first,end)<<'\n';
}
void optimizer_json(std::ofstream& out,const nn::GpuAdam& optimizer) {
    const auto c=optimizer.config();out<<"{\"steps\":"<<optimizer.steps()<<",\"lr\":"<<c.lr<<",\"beta1\":"<<c.beta1<<",\"beta2\":"<<c.beta2<<",\"eps\":"<<c.eps<<",\"max_grad_norm\":"<<c.max_grad_norm<<'}';
}
enum class FollowupArm:int {NoUpdate=0,HoldingOnly=1,FeedingOnly=2,Combined=3};
constexpr int followup_updates=16;
constexpr uint32_t development_near_seed=0x48D1E117u,blind_near_seed=0x48D1AB37u,fresh_food_seed=0x571E2027u;
const char* arm_name(FollowupArm arm) {
    switch(arm) {case FollowupArm::NoUpdate:return "no_update";case FollowupArm::HoldingOnly:return "holding_only";
        case FollowupArm::FeedingOnly:return "feeding_only";default:return "current_combined";}
}
__global__ void followup_task_mask(float* mean,float* std_rows,const float* observations,int count,int arm) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count || arm==3) return;
    const bool cover=observations[static_cast<size_t>(row)*inputs+55]>=.5f;
    if((arm==1 && !cover) || (arm==2 && cover)) for(int joint=0;joint<actions;++joint) {mean[row*actions+joint]=0;std_rows[row*actions+joint]=0;}
}
struct FollowupGradientStats {
    double first_actor=0,first_std=0,maximum_actor=0,maximum_std=0,minimum_actor_clip=1,minimum_std_clip=1;
    bool applied=false;
};
FollowupGradientStats followup_update(Trainer& trainer,FollowupArm arm,std::ofstream& trace,int iteration) {
    require(arm!=FollowupArm::NoUpdate,"No-update arm must never execute Adam");
    FollowupGradientStats out;out.applied=true;const auto stream=trainer.stream.value;const auto scales=trainer.env.action_scales();
    CUDA_CHECK_THROW(cudaMemsetAsync(trainer.stats.get(),0,trainer.stats.size_bytes(),stream));
    for(int epoch=0;epoch<rl::epochs;++epoch) for(int minibatch=0;minibatch<rl::minibatches;++minibatch) {
        gather(trainer,epoch,minibatch);trainer.actor.forward(trainer.mb_obs.get(),trainer.mb,stream);trainer.critic.forward(trainer.mb_obs.get(),trainer.mb,stream);
        shelter_motor::policy_means(trainer.actor.output(),trainer.mb_obs.get(),trainer.physical_means.get(),trainer.mb,scales,stream);
        rl::gaussian_ppo_loss_grad(trainer.physical_means.get(),trainer.log_std.get(),trainer.critic.output(),trainer.mb,actions,
            trainer.mb_actions.get(),trainer.mb_logp.get(),trainer.mb_adv.get(),trainer.mb_ret.get(),trainer.adv_stats.get(),{.2f,rl::ShelterMotorTask::entropy,.5f},
            trainer.dmean.get(),trainer.dstd.get(),trainer.dvalue.get(),trainer.row_stats.get(),trainer.stats.get(),stream);
        // Shared production advantage normalization,1/total-minibatch loss,
        // food Jacobian and entropy coefficients are unchanged. Only these
        // actor/std rows are masked; critic learning still uses every row.
        shelter_motor::policy_mean_grad(trainer.dmean.get(),trainer.mb_obs.get(),trainer.mb,scales,stream);
        followup_task_mask<<<(trainer.mb+255)/256,256,0,stream>>>(trainer.dmean.get(),trainer.dstd.get(),trainer.mb_obs.get(),trainer.mb,static_cast<int>(arm));
        CUDA_CHECK_THROW(cudaGetLastError());
        trainer.actor.zero_grad(stream);trainer.actor.backward(trainer.dmean.get(),trainer.mb,nullptr,stream);trainer.actor_opt.step(trainer.actor.params(),trainer.actor.grads(),stream);
        CUDA_CHECK_THROW(cudaMemsetAsync(trainer.std_grad.get(),0,trainer.std_grad.size_bytes(),stream));rl::sum_columns(trainer.dstd.get(),trainer.mb,actions,trainer.std_grad.get(),stream);
        trainer.std_opt.step(trainer.log_std.get(),trainer.std_grad.get(),stream);
        trainer.critic.zero_grad(stream);trainer.critic.backward(trainer.dvalue.get(),trainer.mb,nullptr,stream);trainer.critic_opt.step(trainer.critic.params(),trainer.critic.grads(),stream);
        const float actor_norm=trainer.actor_opt.last_grad_norm(),std_norm=trainer.std_opt.last_grad_norm(),critic_norm=trainer.critic_opt.last_grad_norm();
        const float actor_clip=clip_scale(trainer.actor_opt.config(),actor_norm),std_clip=clip_scale(trainer.std_opt.config(),std_norm);
        if(epoch==0 && minibatch==0) {out.first_actor=actor_norm;out.first_std=std_norm;}
        out.maximum_actor=std::max(out.maximum_actor,double(actor_norm));out.maximum_std=std::max(out.maximum_std,double(std_norm));
        out.minimum_actor_clip=std::min(out.minimum_actor_clip,double(actor_clip));out.minimum_std_clip=std::min(out.minimum_std_clip,double(std_clip));
        trace<<arm_name(arm)<<','<<iteration<<','<<epoch<<','<<minibatch<<','<<actor_norm<<','<<actor_clip<<','<<std_norm<<','<<std_clip<<','<<critic_norm<<','<<clip_scale(trainer.critic_opt.config(),critic_norm)<<'\n';
    }
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));++trainer.update_id;return out;
}
std::vector<rl::ShelterProbeCase> followup_cases(const rl::BoulderSource& source,uint32_t seed) {
    std::vector<rl::ShelterProbeCase> result;
    for(const auto& world:nominal_worlds(source)) result.push_back({world.state,world.request.goal,world.request.eye,{},{2}});
    const auto near=rl::shelter_hold_cases(source.frame,seed);result.insert(result.end(),near.begin(),near.end());return result;
}
void followup_checkpoint(std::ofstream& file,const char* name,int completed,const Trainer& trainer,const FixedRollout& batch,
                         const FollowupGradientStats& gradient,const FrozenEvaluation& evaluation,bool prospective) {
    int food=0,cover=0;double reward[2]{};
    for(size_t row=0;row<batch.reward.size();++row) {const int mode=batch.obs[row*inputs+55]>=.5f?1:0;if(mode) ++cover;else ++food;reward[mode]+=batch.reward[row];}
    const auto deviation=host_copy(trainer.log_std,trainer.stream.value);
    file<<name<<','<<completed<<','<<trainer.update_id<<','<<trainer.actor_opt.steps()<<','<<trainer.critic_opt.steps()<<','<<trainer.std_opt.steps()<<','<<food<<','<<cover<<','
        <<reward[0]/std::max(1,food)<<','<<reward[1]/std::max(1,cover)<<','<<gradient.first_actor<<','<<gradient.first_std<<','<<gradient.maximum_actor<<','<<gradient.maximum_std<<','
        <<gradient.minimum_actor_clip<<','<<gradient.minimum_std_clip<<','<<gradient.applied<<','<<prospective<<','<<deviation[0]<<','<<deviation[1]<<','<<deviation[2]<<','<<deviation[3]<<','
        <<evaluation.nominal_holds<<','<<evaluation.near_holds<<','<<evaluation.calm_rate<<','<<evaluation.flow_rate<<'\n';file.flush();
}
int run_followup(const std::string& source_path,const std::string& output_path,const std::string& reference_path) {
    const auto output=fs::weakly_canonical(fs::absolute(output_path)),build=fs::weakly_canonical(fs::current_path()/"build");const auto relative=output.lexically_relative(build);
    require(!relative.empty() && relative!=fs::path(".") && *relative.begin()!=fs::path("..") && !fs::exists(output),"Follow-up requires a NEW ignored directory under repository build/");
    const auto source_bytes=checkpoint::read(source_path);const auto saved=rl::read_shelter_motor_run(source_bytes);
    require(rl::shelter_motor_protocol==5,"Follow-up requires unchanged protocol5");
    const auto reference_bytes=checkpoint::read(reference_path);const auto reference=rl::read_shelter_motor_run(reference_bytes);
    require(reference.source.parent==saved.source.parent && reference.source.fingerprint==saved.source.fingerprint && reference.worlds==saved.worlds && reference.seed==saved.seed && reference.update==saved.update+1,
        "Combined first-update reference does not match this source and one-update budget");
    fs::create_directories(output);
    {
        std::ofstream plan(output/"predeclared-protocol.json");plan<<"{\"protocol\":5,\"source_update\":"<<saved.update<<",\"source_payload_fingerprint\":"<<hash_bytes(source_bytes)
            <<",\"worlds_per_arm\":"<<saved.worlds<<",\"iterations_per_arm\":16,\"rollout_steps_per_iteration\":64,\"learning_arms\":[\"holding_only\",\"feeding_only\",\"current_combined\"],\"control\":\"no_update\",\"total_PPO_updates\":48,\"initial_full_state_and_RNG_identical\":true,\"later_rollouts\":\"on-policy per arm; matched logical sampler keys, not forced identical trajectories\",\"mask\":\"actor+std rows after unchanged all-row advantage normalization, total-minibatch denominator and food gain Jacobian; critic all rows\",\"development_checkpoints\":[0,8,16],\"development_near_seed\":"<<development_near_seed
            <<",\"final_blind_near_seed\":"<<blind_near_seed<<",\"old_food_seed\":"<<food_comparison_seed<<",\"fresh_final_food_seed\":"<<fresh_food_seed
            <<",\"combined_first_update_reference_fingerprint\":"<<hash_bytes(reference_bytes)<<",\"combined_first_update_complete_state_must_match_reference\":true"
            <<",\"holding_gate\":\"4/4 exact nominal and >=18/24 final fresh near five-second whole-body holds (exposure<=.05,clearance>=.08)\",\"food_gate\":\">=90% ORIGINAL inherited calm and0.75 food rates for BOTH seeds\",\"finite_required\":true,\"auto_adoption\":false,\"budget_extension\":false,\"final_blind_evaluation\":\"only after every arm finishes its fixed budget\",\"no_update\":\"parameters+optimizers+trainer.update_id fixed; separate logical sampler clock advances. Advanced worlds not exported as a resumable p5 learner\"}\n";
        require(bool(plan),"Could not write predeclared follow-up protocol");
    }
    const auto device=probe_device();require(device.available,"CUDA device/kernel probe failed: "+device.error);
    std::vector<uint8_t> restored_initial;
    {
        Trainer first(saved.worlds,saved.seed);rl::restore_shelter_motor(first,saved.training);restored_initial=rl::shelter_motor_snapshot(first);
        require(restored_initial==saved.training,"First complete initial restore differs from source");
    }
    {
        Trainer second(saved.worlds,saved.seed);rl::restore_shelter_motor(second,restored_initial);
        require(rl::shelter_motor_snapshot(second)==restored_initial,"Second independent complete initial restore differs");
    }
    std::ofstream detailed(output/"frozen-holding-cases.csv"),evaluation_file(output/"frozen-evaluations.csv"),checkpoint_file(output/"checkpoints.csv"),trace(output/"optimizer-trace.csv"),effects(output/"physical-effects.csv");
    for(auto* file:{&detailed,&evaluation_file,&checkpoint_file,&trace,&effects}) *file<<std::setprecision(12);
    detailed<<"variant,suite,case,goal,longest_strict_hold_seconds,five_second_hold,mean_exposure,last_exposure,minimum_full_body_clearance,final_goal_distance,contacts,finite\n";
    evaluation_file<<"variant,nominal_five_second_holds,nominal_cases,near_five_second_holds,near_cases,nominal_mean_longest_hold_seconds,near_mean_longest_hold_seconds,nominal_mean_exposure,near_mean_exposure,nominal_minimum_clearance,near_minimum_clearance,calm_food_per_minute,flow075_food_per_minute,frozen_inference,training_during_evaluation\n";
    checkpoint_file<<"arm,completed_updates,trainer_update_id,actor_adam_steps,critic_adam_steps,std_adam_steps,last_rollout_food_rows,last_rollout_cover_rows,food_mean_reward,cover_mean_reward,first_mb_actor_norm,first_mb_std_norm,max_actor_norm,max_std_norm,min_actor_clip_scale,min_std_clip_scale,gradient_applied,gradient_prospective_only,log_std0,log_std1,log_std2,log_std3,nominal_holds,development_near_holds,calm_food_per_minute,flow075_food_per_minute\n";
    trace<<"arm,iteration,epoch,minibatch,actor_preclip_norm,actor_clip_scale,std_preclip_norm,std_clip_scale,critic_preclip_norm,critic_clip_scale\n";
    effects<<"variant,mode,joint,rows,physical_mean_delta_rms,clipped_mean_delta_rms,coupled_sample_delta_rms,clipped_coupled_sample_delta_rms,old_sigma,new_sigma,old_mean_saturated_fraction,new_mean_saturated_fraction,old_sample_saturated_fraction,new_sample_saturated_fraction,mode_diagonal_gaussian_kl,mode_sample_approx_kl,mode_sample_ppo_clip_fraction\n";
    const auto development=followup_cases(saved.source,development_near_seed);
    const auto inherited=rl::normalized_shelter_actor(rl::transfer_shelter_motor(saved.source.motor),saved.scales);
    const auto original_reference=evaluate_frozen("original_inherited_reference_dev",inherited,saved.source,saved.scales,development,detailed,evaluation_file);
    std::vector<uint8_t> initial_numeric;
    const auto initial=evaluate_frozen("initial_v5_checkpoint0_dev",saved.actor,saved.source,saved.scales,development,detailed,evaluation_file,&initial_numeric);
    struct FinalArm {std::string name;std::vector<float> actor,deviation;FrozenEvaluation development;};std::vector<FinalArm> finals;
    uint64_t first_rollout_hash=0;
    for(int arm_index=0;arm_index<4;++arm_index) {
        const auto arm=static_cast<FollowupArm>(arm_index);const std::string name=arm_name(arm);fs::create_directories(output/name);
        Trainer trainer(saved.worlds,saved.seed);rl::restore_shelter_motor(trainer,saved.training);
        require(rl::shelter_motor_snapshot(trainer)==saved.training,"Follow-up arm did not restore identical complete initial state");
        const auto actor_start=trainer.actor.get_params(),critic_start=trainer.critic.get_params(),std_start=host_copy(trainer.log_std,trainer.stream.value);
        const auto actor_opt_start=optimizer_bytes(trainer.actor_opt),critic_opt_start=optimizer_bytes(trainer.critic_opt),std_opt_start=optimizer_bytes(trainer.std_opt);
        FrozenEvaluation latest=initial;
        for(int iteration=0;iteration<followup_updates;++iteration) {
            const int logical_update=saved.update+iteration;const auto old_std=host_copy(trainer.log_std,trainer.stream.value);
            const auto batch=collect(trainer,logical_update);const auto replay=check_logp(trainer,batch);(void)replay;
            if(iteration==0) {
                const auto fingerprint=hash_bytes(batch.serialize());if(arm_index==0) first_rollout_hash=fingerprint;
                require(fingerprint==first_rollout_hash,"Follow-up first rollout differs across restored arms/RNG");
                const auto snapshot=rl::shelter_motor_snapshot(trainer);const auto split=gradients(trainer);
                require(rl::shelter_motor_snapshot(trainer)==snapshot,"Prospective gradient probe changed a follow-up arm");
                const auto& selected=arm==FollowupArm::HoldingOnly?split.cover:(arm==FollowupArm::FeedingOnly?split.food:split.total);
                FollowupGradientStats prospective;prospective.first_actor=prospective.maximum_actor=norm(selected.actor);prospective.first_std=prospective.maximum_std=norm(selected.deviation);
                prospective.minimum_actor_clip=clip_scale(trainer.actor_opt.config(),static_cast<float>(prospective.first_actor));prospective.minimum_std_clip=clip_scale(trainer.std_opt.config(),static_cast<float>(prospective.first_std));
                followup_checkpoint(checkpoint_file,name.c_str(),0,trainer,batch,prospective,initial,true);
                write_effects(effects,name+"_checkpoint0",batch,batch.means,old_std,old_std);
            }
            FollowupGradientStats update;
            if(arm!=FollowupArm::NoUpdate) update=followup_update(trainer,arm,trace,iteration+1);
            else require(trainer.update_id==saved.update && trainer.actor.get_params()==actor_start && trainer.critic.get_params()==critic_start && host_copy(trainer.log_std,trainer.stream.value)==std_start &&
                optimizer_bytes(trainer.actor_opt)==actor_opt_start && optimizer_bytes(trainer.critic_opt)==critic_opt_start && optimizer_bytes(trainer.std_opt)==std_opt_start,"No-update control changed frozen parameters or optimizer state");
            if(arm==FollowupArm::Combined && iteration==0) {
                require(rl::shelter_motor_snapshot(trainer)==reference.training,"Combined first update differs from the previously verified full fixed-update snapshot");
                std::ofstream proof(output/name/"first-update-reference-match.json");proof<<"{\"complete_snapshot_bit_identical\":true,\"update\":"<<trainer.update_id<<",\"reference_payload_fingerprint\":"<<hash_bytes(reference_bytes)<<"}\n";
                require(bool(proof),"Could not record verified combined first-update match");
            }
            const int completed=iteration+1;
            if(completed==8 || completed==16) {
                const auto before_evaluation=rl::shelter_motor_snapshot(trainer);std::vector<uint8_t> numeric;
                latest=evaluate_frozen(name+"_checkpoint"+std::to_string(completed)+"_dev",trainer.actor.get_params(),saved.source,saved.scales,development,detailed,evaluation_file,&numeric);
                require(rl::shelter_motor_snapshot(trainer)==before_evaluation,"Frozen evaluation changed learner worlds/parameters/optimizer/RNG counters");
                if(arm==FollowupArm::NoUpdate) require(numeric==initial_numeric,"No-update deterministic development results changed at checkpoint8/16");
                followup_checkpoint(checkpoint_file,name.c_str(),completed,trainer,batch,update,latest,false);
                write_effects(effects,name+"_checkpoint"+std::to_string(completed),batch,physical_means(trainer,trainer.actor.get_params(),batch),old_std,host_copy(trainer.log_std,trainer.stream.value));
            }
            std::cout<<"HOLDING_FOLLOWUP arm="<<name<<" iteration="<<completed<<"/16 trainer_update="<<trainer.update_id<<" logical_sampler="<<logical_update<<" training="<<(arm!=FollowupArm::NoUpdate)<<'\n'<<std::flush;
        }
        if(arm!=FollowupArm::NoUpdate) {
            require(trainer.update_id==saved.update+16 && trainer.actor_opt.steps()==trainer.update_id*rl::epochs*rl::minibatches && trainer.critic_opt.steps()==trainer.actor_opt.steps() && trainer.std_opt.steps()==trainer.actor_opt.steps(),"Follow-up final counters are inconsistent");
            const auto expected=rl::shelter_motor_snapshot(trainer);const auto path=output/name/"final-full-disposable-run.bin";
            rl::save_shelter_motor_run(path.u8string(),saved.source,trainer);const auto written=rl::load_shelter_motor_run(path.u8string());
            require(written.source.parent==saved.source.parent && written.training==expected && written.update==trainer.update_id,"Final full-run readback differs from the completed arm");
            Trainer restored(saved.worlds,saved.seed);rl::restore_shelter_motor(restored,written.training);
            require(rl::shelter_motor_snapshot(restored)==expected,"Final full-run restored learner differs from saved actor/Adam/worlds");
            std::ofstream proof(output/name/"resume-readback-verification.json");proof<<"{\"full_training_payload_identical\":true,\"restored_complete_snapshot_identical\":true,\"update\":"<<trainer.update_id<<",\"optimizer_steps\":"<<trainer.actor_opt.steps()<<"}\n";
            require(bool(proof),"Could not record final full-run continuation verification");
        } else {
            checkpoint::write_atomic((output/name/"original-unchanged-full-run.bin").u8string(),source_bytes);
            require(checkpoint::read((output/name/"original-unchanged-full-run.bin").u8string())==source_bytes,"No-update original-source copy differs");
            std::ofstream control(output/name/"logical-rng-control.json");control<<"{\"trainer_update_id\":"<<saved.update<<",\"last_logical_sampler_update\":"<<saved.update+15<<",\"next_logical_sampler_update\":"<<saved.update+16
                <<",\"rollouts\":16,\"physical_steps_per_world\":1024,\"optimizer_steps_added\":0,\"parameters_unchanged\":true,\"advanced_worlds_exported_as_resumable_learner\":false,\"saved_full_run\":\"original unchanged source, not advanced diagnostic worlds\"}\n";
            require(bool(control),"Could not write no-update logical RNG metadata");
        }
        finals.push_back({name,trainer.actor.get_params(),host_copy(trainer.log_std,trainer.stream.value),latest});
        require(checkpoint::read(source_path)==source_bytes,"Protected source changed during follow-up");
    }
    // Blind cases are constructed/evaluated only now, after all48 predeclared
    // PPO updates have completed. No selection, budget extension or adoption.
    const auto blind=followup_cases(saved.source,blind_near_seed);
    const auto inherited_blind=evaluate_frozen("original_inherited_reference_final_blind",inherited,saved.source,saved.scales,blind,detailed,evaluation_file);
    const double original_fresh_calm=rl::shelter_motor_food_rate(inherited,saved.source,fresh_food_seed,16,600,{},saved.scales);
    const double original_fresh_flow=rl::shelter_motor_food_rate(inherited,saved.source,fresh_food_seed,16,600,{.0075f,0,0},saved.scales);
    std::ofstream final_file(output/"final-heldout.csv");final_file<<std::setprecision(12)<<"arm,nominal_holds,blind_near_holds,old_seed_calm_rate,old_seed_flow_rate,fresh_seed_calm_rate,fresh_seed_flow_rate,old_calm_retained_ratio,old_flow_retained_ratio,fresh_calm_retained_ratio,fresh_flow_retained_ratio,holding_gate,food_gate,promising_diagnostic,adopted\n";
    for(const auto& arm:finals) {
        const auto result=evaluate_frozen(arm.name+"_final_blind",arm.actor,saved.source,saved.scales,blind,detailed,evaluation_file);
        const double fresh_calm=rl::shelter_motor_food_rate(arm.actor,saved.source,fresh_food_seed,16,600,{},saved.scales),fresh_flow=rl::shelter_motor_food_rate(arm.actor,saved.source,fresh_food_seed,16,600,{.0075f,0,0},saved.scales);
        const double ratios[4]={result.calm_rate/std::max(1e-12,original_reference.calm_rate),result.flow_rate/std::max(1e-12,original_reference.flow_rate),fresh_calm/std::max(1e-12,original_fresh_calm),fresh_flow/std::max(1e-12,original_fresh_flow)};
        const bool holding=result.nominal_holds==4 && result.near_holds>=18,retained=ratios[0]>=.9 && ratios[1]>=.9 && ratios[2]>=.9 && ratios[3]>=.9;
        final_file<<arm.name<<','<<result.nominal_holds<<','<<result.near_holds<<','<<result.calm_rate<<','<<result.flow_rate<<','<<fresh_calm<<','<<fresh_flow;
        for(double ratio:ratios) final_file<<','<<ratio;final_file<<','<<holding<<','<<retained<<','<<(holding && retained)<<",0\n";
    }
    std::ofstream references(output/"inherited-reference.json");references<<std::setprecision(17)<<"{\"normalized_original_inherited_actor\":true,\"nominal_holds\":"<<inherited_blind.nominal_holds<<",\"final_blind_near_holds\":"<<inherited_blind.near_holds<<",\"old_seed_calm_food\":"<<original_reference.calm_rate<<",\"old_seed_flow_food\":"<<original_reference.flow_rate<<",\"fresh_seed_calm_food\":"<<original_fresh_calm<<",\"fresh_seed_flow_food\":"<<original_fresh_flow<<"}\n";
    require(checkpoint::read(source_path)==source_bytes,"Protected source changed during final held-out evaluation");
    require(checkpoint::read(reference_path)==reference_bytes,"Verified one-update reference changed during follow-up");
    std::ofstream verification(output/"verification.json");verification<<"{\"completed\":true,\"iterations_per_arm\":16,\"PPO_updates\":48,\"first_rollout_payload_fingerprint\":"<<first_rollout_hash<<",\"dual_initial_complete_restore_identical\":true,\"all_first_rollouts_identical\":true,\"every_new_rollout_preupdate_logp_checked\":true,\"later_rollouts_on_policy\":true,\"source_unchanged\":true,\"evaluation_learner_worlds_and_rng_unchanged\":true,\"no_update_dev_outputs_bit_identical_at0_8_16\":true,\"no_update_parameters_and_optimizer_unchanged\":true,\"production_files_modified\":false,\"adopted\":false,\"budget_extended\":false}\n";
    for(auto* file:{&detailed,&evaluation_file,&checkpoint_file,&trace,&effects,&final_file,&references,&verification}) {file->flush();require(bool(*file),"Could not finish follow-up artifacts");}
    std::cout<<"HOLDING_FOLLOWUP_DONE arms=4 PPO_updates=48 source_unchanged=1 adopted=0 budget_extended=0 output="<<output.u8string()<<'\n';return 0;
}
}

int main(int argc,char** argv) {
    using namespace cudarl;
    try {
        if(argc>=4 && std::string(argv[3])=="--followup=16") {
            require(argc==5,"Usage: holding_update_diagnose <full-v5-run> <NEW-build-directory> --followup=16 --reference=<verified-one-update-full-run>");
            const std::string option=argv[4];require(option.rfind("--reference=",0)==0 && option.size()>12,"Follow-up requires --reference=<verified-one-update-full-run>");
            return run_followup(argv[1],argv[2],option.substr(12));
        }
        require(argc==3 || argc==4,"Usage: holding_update_diagnose <full-protocol5-motor-run> <new-directory-under-build> [--reuse=<completed-valid-audit-directory>]");
        fs::path reuse;
        if(argc==4) {
            const std::string option=argv[3];require(option.rfind("--reuse=",0)==0 && option.size()>8,"Optional argument must be --reuse=<completed-valid-audit-directory>");
            reuse=fs::weakly_canonical(fs::absolute(option.substr(8)));
        }
        const fs::path output=fs::weakly_canonical(fs::absolute(argv[2]));const auto build=fs::weakly_canonical(fs::current_path()/"build");
        const auto relative=output.lexically_relative(build);
        require(!relative.empty() && relative!=fs::path(".") && *relative.begin()!=fs::path(".."),"Diagnostic output must be a new ignored directory under this repository's build/");
        require(!fs::exists(output),"Refusing to overwrite an existing diagnostic directory");
        const auto source_bytes=checkpoint::read(argv[1]);const auto saved=rl::read_shelter_motor_run(source_bytes);
        require(rl::shelter_motor_protocol==5,"This diagnostic requires the existing protocol5 physical action units");
        const auto device=probe_device();require(device.available,"CUDA device/kernel probe failed: "+device.error);
        Trainer trainer(saved.worlds,saved.seed);rl::restore_shelter_motor(trainer,saved.training);
        require(rl::shelter_motor_snapshot(trainer)==saved.training,"Full saved optimizer/world snapshot did not restore byte-identically");
        const auto original_actor=trainer.actor.get_params(),original_critic=trainer.critic.get_params(),original_std=host_copy(trainer.log_std,trainer.stream.value);
        const auto original_actor_optimizer=optimizer_bytes(trainer.actor_opt),original_critic_optimizer=optimizer_bytes(trainer.critic_opt),original_std_optimizer=optimizer_bytes(trainer.std_opt);
        const auto original_context=trainer.env.context();
        std::cout<<"HOLDING_AUDIT source_update="<<saved.update<<" worlds="<<saved.worlds<<" protocol=5 fixed_rollout=1 reused="<<(!reuse.empty())<<" adoption=0\n"<<std::flush;
        FixedRollout batch;std::vector<uint8_t> post_rollout;
        if(reuse.empty()) {batch=collect(trainer);post_rollout=rl::shelter_motor_snapshot(trainer);}
        else {
            validate_reuse_summary(reuse,hash_bytes(source_bytes));
            batch=FixedRollout::read(checkpoint::read((reuse/"fixed-rollout.bin").u8string()));
            post_rollout=checkpoint::read((reuse/"post-rollout-disposable-trainer.bin").u8string());
            rl::restore_shelter_motor(trainer,post_rollout);batch.load(trainer);
            require(trainer.actor.get_params()==original_actor && trainer.critic.get_params()==original_critic && host_copy(trainer.log_std,trainer.stream.value)==original_std,
                "Reused post-rollout actor/critic/std differ from protected source parameters");
            require(optimizer_bytes(trainer.actor_opt)==original_actor_optimizer && optimizer_bytes(trainer.critic_opt)==original_critic_optimizer && optimizer_bytes(trainer.std_opt)==original_std_optimizer,
                "Reused post-rollout optimizer configuration/moments/counters differ from the protected source");
            const auto context=trainer.env.context();
            require(context.basin.layout==2 && context.basin.layout==original_context.basin.layout && context.version==original_context.version &&
                context.current.x==original_context.current.x && context.current.y==original_context.current.y && context.current.z==original_context.current.z,
                "Reused fixed rollout has a different physical context");
            for(int j=0;j<actions;++j) require(trainer.env.action_scales().value[j]==saved.scales.value[j],"Reused policy gains differ from source");
            require(rl::shelter_motor_snapshot(trainer)==post_rollout,"Reused post-rollout full state did not restore exactly");
        }
        const auto replay=check_logp(trainer,batch);validate_fixed_gae(trainer,batch);
        fs::create_directories(output);checkpoint::write_atomic((output/"fixed-rollout.bin").u8string(),batch.serialize());
        checkpoint::write_atomic((output/"post-rollout-disposable-trainer.bin").u8string(),post_rollout);
        const auto split=gradients(trainer);require(split.food_rows>0 && split.cover_rows>0,"First production minibatch must contain both goal types");
        std::ofstream alignment(output/"gradient-alignment.csv");alignment<<std::setprecision(12)<<"kind,layer,food_l2,cover_l2,combined_l2,food_unit_jacobian_l2,food_gain_to_unit_ratio,food_cover_dot,food_cover_cosine,combined_over_sum_norms,additivity_max_abs\n";
        write_alignment(alignment,"actor_all",-1,split.food,split.cover,split.total,split.food_unit,0,split.total.actor.size());
        write_alignment(alignment,"critic_all",-1,split.food,split.cover,split.total,split.food_unit,0,split.total.critic.size(),true);
        for(int layer=0;layer<trainer.actor.spec().num_layers();++layer) write_alignment(alignment,"actor_layer",layer,split.food,split.cover,split.total,split.food_unit,
            trainer.actor.spec().weight_offset(layer),trainer.actor.spec().weight_offset(layer+1));
        for(int layer=0;layer<trainer.critic.spec().num_layers();++layer) write_alignment(alignment,"critic_layer",layer,split.food,split.cover,split.total,split.food_unit,
            trainer.critic.spec().weight_offset(layer),trainer.critic.spec().weight_offset(layer+1),true);
        std::ofstream joints(output/"joint-gradients.csv");joints<<std::setprecision(12)<<"joint,saved_food_gain,mode,rows,physical_mean_grad_l2,actor_mean_grad_l2,last_linear_row_parameter_grad_l2,last_linear_row_unit_jacobian_grad_l2,shared_logstd_gradient,shared_logstd_combined,shared_logstd_additivity_abs\n";
        for(int joint=0;joint<actions;++joint) for(int mode=0;mode<2;++mode) {
            const auto& part=mode?split.cover:split.food;double physical=0,actor=0;
            for(int row=0;row<trainer.mb;++row) {physical+=double(part.physical_mean[row*actions+joint])*part.physical_mean[row*actions+joint];actor+=double(part.actor_mean[row*actions+joint])*part.actor_mean[row*actions+joint];}
            const auto& spec=trainer.actor.spec();const int layer=spec.num_layers()-1;
            const size_t first=spec.weight_offset(layer)+joint*spec.in(layer),end=first+spec.in(layer),bias=spec.bias_offset(layer)+joint;
            const auto& unit=mode?split.cover.actor:split.food_unit.actor;
            const double row_norm=std::sqrt(dot(part.actor,part.actor,first,end)+double(part.actor[bias])*part.actor[bias]);
            const double unit_norm=std::sqrt(dot(unit,unit,first,end)+double(unit[bias])*unit[bias]);
            joints<<joint<<','<<saved.scales.value[joint]<<','<<(mode?"cover":"food")<<','<<(mode?split.cover_rows:split.food_rows)<<','<<std::sqrt(physical)<<','<<std::sqrt(actor)<<','
                <<row_norm<<','<<unit_norm<<','<<part.deviation[joint]<<','<<split.total.deviation[joint]<<','<<std::abs(double(split.food.deviation[joint])+split.cover.deviation[joint]-split.total.deviation[joint])<<'\n';
        }
        const auto nominal=nominal_observations(saved.source);const auto nominal_before=nominal_means(original_actor,nominal,saved.scales);
        std::ofstream effects(output/"physical-effects.csv");effects<<std::setprecision(12)<<"variant,mode,joint,rows,physical_mean_delta_rms,clipped_mean_delta_rms,coupled_sample_delta_rms,clipped_coupled_sample_delta_rms,old_sigma,new_sigma,old_mean_saturated_fraction,new_mean_saturated_fraction,old_sample_saturated_fraction,new_sample_saturated_fraction,mode_diagonal_gaussian_kl,mode_sample_approx_kl,mode_sample_ppo_clip_fraction\n";
        std::ofstream pose(output/"nominal-zero-alignment.csv");pose<<std::setprecision(12)<<"variant,goal,joint,before_physical_mean,after_physical_mean,delta,toward_zero_dot,toward_zero_cosine\n";
        write_nominal(pose,"frozen_v5_baseline",nominal_before,nominal_before);
        std::ofstream frozen_cases(output/"frozen-holding-cases.csv");frozen_cases<<std::setprecision(12)<<"variant,suite,case,goal,longest_strict_hold_seconds,five_second_hold,mean_exposure,last_exposure,minimum_full_body_clearance,final_goal_distance,contacts,finite\n";
        std::ofstream frozen_summary(output/"frozen-evaluations.csv");frozen_summary<<std::setprecision(12)<<"variant,nominal_five_second_holds,nominal_cases,fresh_near_five_second_holds,fresh_near_cases,nominal_mean_longest_hold_seconds,near_mean_longest_hold_seconds,nominal_mean_exposure,near_mean_exposure,nominal_minimum_clearance,near_minimum_clearance,calm_food_per_minute,flow075_food_per_minute,frozen_inference,training_during_evaluation\n";
        const auto holding_cases=frozen_holding_cases(saved.source);
        const auto baseline_evaluation=evaluate_frozen("frozen_v5_baseline",original_actor,saved.source,saved.scales,holding_cases,frozen_cases,frozen_summary);
        save_audit_policy(output,"frozen_v5_baseline",original_actor,original_std,saved,trainer.actor_opt.steps());
        std::vector<UpdateCopy> copies;
        copies.push_back(one_step("food_only_one_adam_step",trainer,split.food));copies.push_back(one_step("cover_only_one_adam_step",trainer,split.cover));copies.push_back(one_step("combined_one_adam_step",trainer,split.total));
        Gradients moments_only;moments_only.actor.assign(split.total.actor.size(),0.f);moments_only.deviation.assign(actions,0.f);
        copies.push_back(one_step("moments_only_zero_current_gradient",trainer,moments_only));
        const double food_actor_norm=norm(split.food.actor),cover_actor_norm=norm(split.cover.actor);
        const float actor_balance_lambda=static_cast<float>(food_actor_norm>0?std::min(1.,cover_actor_norm/food_actor_norm):1.);
        Gradients actor_balanced;actor_balanced.actor.resize(split.total.actor.size());actor_balanced.deviation=split.total.deviation;
        for(size_t i=0;i<actor_balanced.actor.size();++i) actor_balanced.actor[i]=actor_balance_lambda*split.food.actor[i]+split.cover.actor[i];
        require(actor_balanced.deviation==split.total.deviation,"Actor balance must retain the original combined std gradient");
        copies.push_back(one_step("actor_task_balanced_one_adam_step",trainer,actor_balanced));
        require(copies.back().deviation==copies[2].deviation,"Actor-only balancing changed the combined step's std update");
        FrozenEvaluation balanced_evaluation;
        std::ofstream steps(output/"isolated-adam.csv");steps<<std::setprecision(12)<<"variant,actor_steps_after,actor_preclip_global_norm,actor_clip_scale,std_preclip_global_norm,std_clip_scale,critic_updated\n";
        for(const auto& copy:copies) {
            write_effects(effects,copy.name,batch,physical_means(trainer,copy.actor,batch),original_std,copy.deviation);
            write_nominal(pose,copy.name,nominal_before,nominal_means(copy.actor,nominal,saved.scales));
            steps<<copy.name<<','<<copy.steps<<','<<copy.actor_norm<<','<<copy.actor_clip<<','<<copy.std_norm<<','<<copy.std_clip<<",0\n";
            const auto evaluation=evaluate_frozen(copy.name,copy.actor,saved.source,saved.scales,holding_cases,frozen_cases,frozen_summary);
            if(copy.name=="actor_task_balanced_one_adam_step") balanced_evaluation=evaluation;
            save_audit_policy(output,copy.name,copy.actor,copy.deviation,saved,copy.steps);
        }
        require(trainer.actor.get_params()==original_actor && trainer.critic.get_params()==original_critic && host_copy(trainer.log_std,trainer.stream.value)==original_std,
            "Isolated gradient audit mutated its frozen reference parameters");
        require(rl::shelter_motor_snapshot(trainer)==post_rollout,"Gradient audit mutated the disposable optimizer/world snapshot");
        std::vector<uint8_t> after_first;int full_steps=0;FrozenEvaluation full_evaluation;
        {
            Trainer copy(saved.worlds,saved.seed);rl::restore_shelter_motor(copy,post_rollout);batch.load(copy);
            require(rl::shelter_motor_snapshot(copy)==post_rollout,"Fixed-data full-update clone did not restore exactly");
            std::ofstream trace(output/"full-update-optimizer-trace.csv");trace<<std::setprecision(12)<<"epoch,minibatch,actor_preclip_norm,actor_clip_scale,std_preclip_norm,std_clip_scale,critic_preclip_norm,critic_clip_scale\n";
            replay_update(copy,&trace);after_first=rl::shelter_motor_snapshot(copy);full_steps=copy.actor_opt.steps();
            write_effects(effects,"full_fixed_rollout_ppo_update",batch,physical_means(trainer,copy.actor.get_params(),batch),original_std,host_copy(copy.log_std,copy.stream.value));
            write_nominal(pose,"full_fixed_rollout_ppo_update",nominal_before,nominal_means(copy.actor.get_params(),nominal,saved.scales));
            full_evaluation=evaluate_frozen("full_fixed_rollout_ppo_update",copy.actor.get_params(),saved.source,saved.scales,holding_cases,frozen_cases,frozen_summary);
            save_audit_policy(output,"full_fixed_rollout_ppo_update",copy.actor.get_params(),host_copy(copy.log_std,copy.stream.value),saved,copy.actor_opt.steps());
            rl::save_shelter_motor_run((output/"full-fixed-update-disposable-run.bin").u8string(),saved.source,copy);
        }
        bool bit_identical=false;
        {
            // Restore both the trainer and fixed rollout through their actual
            // checksummed disk formats, then execute the same single update.
            const auto restored_post=checkpoint::read((output/"post-rollout-disposable-trainer.bin").u8string());
            const auto restored_batch=FixedRollout::read(checkpoint::read((output/"fixed-rollout.bin").u8string()));
            require(restored_batch.serialize()==batch.serialize(),"Fixed rollout disk round trip changed data");
            Trainer resumed(saved.worlds,saved.seed);rl::restore_shelter_motor(resumed,restored_post);restored_batch.load(resumed);replay_update(resumed,nullptr);
            bit_identical=rl::shelter_motor_snapshot(resumed)==after_first;
        }
        require(bit_identical,"Restored fixed-data PPO continuation differs from its identical clone");
        require(checkpoint::read(argv[1])==source_bytes,"Protected source file changed during diagnosis");
        int food_rows=0,cover_rows=0;
        for(size_t row=0;row<batch.logp.size();++row) {if(batch.obs[row*inputs+55]>=.5f) ++cover_rows;else ++food_rows;}
        std::ofstream summary(output/"summary.json");summary<<std::setprecision(17)<<"{\n\"protocol\":5,\"source_update\":"<<saved.update<<",\"source_payload_fingerprint\":"<<hash_bytes(source_bytes)
            <<",\"worlds\":"<<saved.worlds<<",\"rollout_steps\":"<<rl::rollout<<",\"rollout_food_rows\":"<<food_rows<<",\"rollout_cover_rows\":"<<cover_rows
            <<",\"gradient_scope\":\"first production epoch0/minibatch0; same shared advantage normalization and denominator\",\"gradient_food_rows\":"<<split.food_rows<<",\"gradient_cover_rows\":"<<split.cover_rows
            <<",\"advantage_mean\":"<<split.advantage_stats[0]<<",\"advantage_std\":"<<split.advantage_stats[1]<<",\"preupdate_replay_logp_max_abs\":"<<replay.maximum<<",\"preupdate_replay_logp_nonidentical_rows\":"<<replay.different
            <<",\"saved_food_gains\":["<<saved.scales.value[0]<<','<<saved.scales.value[1]<<','<<saved.scales.value[2]<<','<<saved.scales.value[3]<<"],\"gain1_comparison\":\"Jacobian-only counterfactual: fixed forward physical policy, samples and logp; no variant trained\",\"single_step_attribution_control\":\"moments_only_zero_current_gradient retains saved Adam moments and step counter\",\n\"actor_optimizer\":";
        optimizer_json(summary,trainer.actor_opt);summary<<",\"std_optimizer\":";optimizer_json(summary,trainer.std_opt);summary<<",\"critic_optimizer\":";optimizer_json(summary,trainer.critic_opt);
        summary<<",\"full_update_optimizer_steps_after\":"<<full_steps<<",\"full_update_minibatches\":"<<rl::epochs*rl::minibatches
            <<",\"reused_completed_valid_rollout\":"<<(reuse.empty()?"false":"true")<<",\"fixed_means_replay_max_abs\":"<<replay.mean_maximum<<",\"fixed_values_replay_max_abs\":"<<replay.value_maximum<<",\"fixed_gae_replay_identical\":true"
            <<",\"actor_task_balance_lambda\":"<<actor_balance_lambda<<",\"actor_task_balance_food_gradient_norm\":"<<food_actor_norm<<",\"actor_task_balance_cover_gradient_norm\":"<<cover_actor_norm
            <<",\"actor_task_balance_combined_gradient_norm\":"<<norm(actor_balanced.actor)<<",\"actor_task_balance_std_gradient\":\"original combined; unchanged\",\"actor_task_balance_scope\":\"one copied Adam step on the same frozen batch; no changed forward policy, Jacobian, rewards, physical rules or production trainer\""
            <<",\"gradient_additivity_verified\":true,\"gradient_input_immutability_verified\":true,\"host_uploads_ordered_on_consumer_stream\":true,\"resume_full_snapshot_bit_identical\":true,\"source_file_unchanged\":true,\"frozen_reference_parameters_unchanged\":true,\"actor_adopted\":false,\"production_files_modified\":false,\"nominal_fixture_seed\":"<<nominal_seed
            <<",\"fresh_near_seed\":"<<fresh_near_seed<<",\"food_comparison_seed\":"<<food_comparison_seed<<",\"frozen_evaluation_policy_count\":7,\"frozen_evaluation_actions\":\"deterministic physical means; no sampling or weight updates\",\"frozen_holding_approach_ticks\":1,\"frozen_holding_dwell_ticks\":200,\"food_evaluation_worlds\":16,\"food_evaluation_steps\":600"
            <<",\"baseline_frozen_results\":{\"nominal_holds\":"<<baseline_evaluation.nominal_holds<<",\"fresh_near_holds\":"<<baseline_evaluation.near_holds<<",\"calm_food_per_minute\":"<<baseline_evaluation.calm_rate<<",\"flow075_food_per_minute\":"<<baseline_evaluation.flow_rate<<'}'
            <<",\"full_update_frozen_results\":{\"nominal_holds\":"<<full_evaluation.nominal_holds<<",\"fresh_near_holds\":"<<full_evaluation.near_holds<<",\"calm_food_per_minute\":"<<full_evaluation.calm_rate<<",\"flow075_food_per_minute\":"<<full_evaluation.flow_rate<<'}'
            <<",\"actor_task_balanced_frozen_results\":{\"nominal_holds\":"<<balanced_evaluation.nominal_holds<<",\"fresh_near_holds\":"<<balanced_evaluation.near_holds<<",\"calm_food_per_minute\":"<<balanced_evaluation.calm_rate<<",\"flow075_food_per_minute\":"<<balanced_evaluation.flow_rate<<'}'
            <<",\"audit_policy_format\":\"Checksummed payload: uint32 magic0x48415035,format1,protocol5; uint64 boulder fingerprint; int source_update,actor_adam_steps; ActuatorScale; float arrays actor,log_std. Not a resumable trainer or adopted policy. Full32-step run also saved separately.\",\"nominal_zero_scope\":\"four fresh matched exact calm zero-velocity cover poses; directional diagnostic is supplemented by frozen whole-body holding evaluation, not an active-braking mastery claim\"\n}\n";
        for(auto* file:{&alignment,&joints,&effects,&pose,&steps,&frozen_cases,&frozen_summary,&summary}) {file->flush();require(bool(*file),"Could not finish diagnostic report");}
        std::cout<<"HOLDING_AUDIT_DONE replay_logp_max="<<replay.maximum<<" first_mb_food="<<split.food_rows<<" first_mb_cover="<<split.cover_rows
            <<" actor_food_cover_cosine="<<cosine(split.food.actor,split.cover.actor)<<" food_gain_to_unit_norm_ratio="<<norm(split.food.actor)/std::max(1e-30,norm(split.food_unit.actor))
            <<" full_update_resume_bit_identical=1 source_unchanged=1 adopted=0 output="<<output.u8string()<<'\n';return 0;
    } catch(const std::exception& error) {std::cerr<<"HOLDING_AUDIT_ERROR "<<error.what()<<'\n';return 1;}
}
