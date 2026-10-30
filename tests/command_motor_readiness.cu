#include "rl/command_motor_ppo.h"
#include "rl/boulder_source.h"
#include "motor_benchmark.h"
#include "nn/mlp_cpu.h"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

// Registered readiness check, not a learning/skill-success experiment.
// One CUDA nonblocking stream, eight worlds x64 controls, no backward/Adam/PPO
// update. Rows0/1 use the immutable original reference captures4/5; row7 begins
// at599 to exercise the declared600-tick terminal (done1, truncated0). A separate
// one-world breach is a negative safety test, never admitted to rollout data.
// Independent CPU inference uses CpuMlp<float>; CPU physics replays precisely
// the downloaded pre-state and sampled commands for ONE step, not a drifting
// second trajectory. Log-likelihood and GAE formulas are handwritten double.
// Tolerances are fixed here before execution; a failure is reported, not tuned:
// observation2e-4+2e-4|reference|; MLP3e-5+3e-5|reference|;
// logp3e-5+3e-5|reference|; physical state/reward2e-4+2e-3|reference|;
// GAE2e-4+2e-4|reference|; normalized minibatch statistics2e-5+2e-5|reference|.
// Copies, discrete flags, sampled RNG repeats and full-save restoration are
// exact. Save/restore includes512 fresh worlds and ALL three Adam clocks at0.
namespace {
using namespace cudarl;
namespace cm=command_motor;
namespace bench=motor_benchmark;
namespace fs=std::filesystem;
constexpr int world_count=8,steps=64,rows=world_count*steps,inputs=59,actions=4;
constexpr uint64_t frozen_fixture_hash=16079702944816815034ull;
constexpr uint64_t frozen_fixture_environment=10511261149222959219ull;
constexpr uint64_t frozen_source=2073117933376950176ull;
void need(bool okay,const std::string& message) {if(!okay) throw std::runtime_error(message);}
uint64_t hash(const std::vector<uint8_t>& bytes) {
    uint64_t h=14695981039346656037ull;for(uint8_t b:bytes) {h^=b;h*=1099511628211ull;}return h;
}
struct Comparison {
    std::string name;double absolute,relative,max_error=0,max_ratio=0;size_t checked=0,failed=0;
    void value(double actual,double expected) {
        ++checked;const double error=std::abs(actual-expected),limit=absolute+relative*std::abs(expected);
        if(!std::isfinite(actual)||!std::isfinite(expected)||error>limit) ++failed;
        if(std::isfinite(error)) {max_error=std::max(max_error,error);max_ratio=std::max(max_ratio,error/limit);}
        else {max_error=1e300;max_ratio=1e300;}
    }
};
template<class T> std::vector<T> download(const T* pointer,size_t count,cudaStream_t stream) {
    std::vector<T> out(count);CUDA_CHECK_THROW(cudaMemcpyAsync(out.data(),pointer,count*sizeof(T),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));return out;
}
template<class T> void upload(device_ptr<T>& pointer,const std::vector<T>& values,cudaStream_t stream) {
    need(pointer.size()==values.size(),"Readiness upload dimensions differ");
    CUDA_CHECK_THROW(cudaMemcpyAsync(pointer.get(),values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
void vector_check(Comparison& c,deep::V actual,deep::V expected) {
    c.value(actual.x,expected.x);c.value(actual.y,expected.y);c.value(actual.z,expected.z);
}
void state_check(Comparison& c,const cm::State& actual,const cm::State& expected) {
    vector_check(c,actual.body.position,expected.body.position);
    for(int j=0;j<deep::dofs;++j) c.value(actual.body.velocity[j],expected.body.velocity[j]);
    for(int j=0;j<4;++j) {c.value(actual.body.angle[j],expected.body.angle[j]);c.value(actual.previous_action[j],expected.previous_action[j]);}
    const auto a=actual.body.orientation,b=expected.body.orientation;
    c.value(a.x,b.x);c.value(a.y,b.y);c.value(a.z,b.z);c.value(a.w,b.w);
    c.value(actual.body.work,expected.body.work);c.value(actual.body.travel,expected.body.travel);
    vector_check(c,actual.request.position,expected.request.position);vector_check(c,actual.request.velocity,expected.request.velocity);
    vector_check(c,actual.request.heading,expected.request.heading);
    need(actual.steps==expected.steps&&actual.episode==expected.episode&&actual.seed==expected.seed&&actual.environment==expected.environment,
         "CPU/GPU discrete world/reset clocks disagree");
}
bool exact_state(const cm::State& actual,const cm::State& expected) {
    const auto same_vector=[](deep::V a,deep::V b) {return a.x==b.x&&a.y==b.y&&a.z==b.z;};
    if(!same_vector(actual.body.position,expected.body.position)||!same_vector(actual.request.position,expected.request.position)||
       !same_vector(actual.request.velocity,expected.request.velocity)||!same_vector(actual.request.heading,expected.request.heading)) return false;
    const auto a=actual.body.orientation,b=expected.body.orientation;
    if(a.x!=b.x||a.y!=b.y||a.z!=b.z||a.w!=b.w||actual.body.work!=expected.body.work||actual.body.travel!=expected.body.travel) return false;
    for(int j=0;j<deep::dofs;++j) if(actual.body.velocity[j]!=expected.body.velocity[j]) return false;
    for(int j=0;j<actions;++j) if(actual.body.angle[j]!=expected.body.angle[j]||actual.previous_action[j]!=expected.previous_action[j]) return false;
    return actual.steps==expected.steps&&actual.episode==expected.episode&&actual.seed==expected.seed&&actual.environment==expected.environment;
}
std::vector<bench::Fixture> read_captures(const std::vector<uint8_t>& bytes,uint64_t source) {
    need(hash(bytes)==frozen_fixture_hash,"Readiness requires the exact immutable reference-v1 capture file");
    checkpoint::Reader in{bytes};need(in.value<uint32_t>()==0x434D4649u&&in.value<uint32_t>()==1,"Wrong reference fixture schema");
    need(in.value<uint64_t>()==frozen_fixture_environment&&in.value<uint64_t>()==source,"Historical reference fixture identity differs");
    const auto all=in.array<bench::Fixture>(16);in.finish();need(all.size()==16,"Expected16 reference captures");
    for(int id:{4,5}) need(all[id].id==id&&all[id].kind==bench::Kind::Brake&&all[id].qualified&&all[id].warmup_ticks>0&&
         cm::finite(all[id].initial)&&all[id].actual_entry_speed>=.01f&&all[id].actual_entry_speed<=.03f,
         "Captured braking fixture is not qualified actual swimming motion");
    return all;
}
struct Rollout {
    std::vector<float> observations,means,critic_values,commands,logp,rewards,done,truncated,final_observations,advantages,returns;
    std::vector<cm::State> pre_states;
};
void check_finite(const std::vector<float>& values,const char* description) {
    for(float x:values) need(std::isfinite(x),std::string("Nonfinite ")+description);
}
void write_report(const fs::path& output,bool passed,uint64_t source_hash,uint64_t source_fingerprint,uint64_t fixture_hash,
                  const std::vector<Comparison>& comparisons,size_t terminal_rows,size_t collected_rows,int negative_ticks,
                  bool safety_rejection_verified,bool preserved,const std::string& error="") {
    std::ofstream file(output/"summary.json");need(static_cast<bool>(file),"Cannot write readiness report");file<<std::setprecision(17);
    file<<"{\n\"protocol\":1,\"passed\":"<<(passed?"true":"false")<<",\"research_readiness_only\":true,\"physical_reference_passed\":false,"
        <<"\n\"source_payload_fnv1a64\":\""<<source_hash<<"\",\"source_fingerprint\":\""<<source_fingerprint
        <<"\",\"capture_payload_fnv1a64\":\""<<fixture_hash<<"\",\"historical_capture_environment_fingerprint\":\""<<frozen_fixture_environment
        <<"\",\"environment_fingerprint\":\""<<rl::command_motor_fingerprint()<<"\",\"benchmark_rules_fingerprint\":\""<<bench::rules_fingerprint()<<"\","
        <<"\n\"worlds\":8,\"control_ticks_per_world\":64,\"registered_sampled_transitions\":512,\"sampled_transitions\":"<<collected_rows
        <<",\"separate_negative_safety_ticks\":"<<negative_ticks<<",\"training_seed\":"<<cm::training_seed
        <<",\"captured_braking_fixture_ids\":[4,5],\"horizon_fixture_initial_tick\":599,\"initial_log_std\":-0.5,"
        <<"\"PPO_updates\":0,\"Adam_steps\":0,\"backward_calls\":0,\"nonblocking_stream\":true,\"finite_horizon_terminal_rows\":"<<terminal_rows
        <<",\"pre_optimizer_safety_rejection_verified\":"<<(passed&&safety_rejection_verified?"true":"false")
        <<",\"source_inputs_unchanged\":"<<(preserved?"true":"false")<<",\"tolerances_were_not_changed\":true,"
        <<"\n\"scope\":\"inference/collection/logp/physics/terminal/GAE/gather/normalization/persistence; not gradient or learning success\","
        <<"\n\"comparisons\":[";
    for(size_t i=0;i<comparisons.size();++i) {const auto& c=comparisons[i];if(i) file<<',';
        file<<"{\"kind\":\""<<c.name<<"\",\"absolute_tolerance\":"<<c.absolute<<",\"relative_tolerance\":"<<c.relative
            <<",\"checked\":"<<c.checked<<",\"failed\":"<<c.failed<<",\"maximum_absolute_error\":"<<c.max_error
            <<",\"maximum_error_over_tolerance\":"<<c.max_ratio<<'}';}
    file<<"],\n\"error\":\"";for(char ch:error) {if(ch=='"'||ch=='\\') file<<'\\';if(ch=='\n'||ch=='\r') file<<' ';else file<<ch;}
    file<<"\"\n}\n";file.flush();need(static_cast<bool>(file),"Readiness report write failed");
}
}

int main(int argc,char** argv) {
    fs::path output;bool owned_output=false;uint64_t source_hash=0,source_fingerprint=0,fixture_hash=0;
    size_t terminal_rows=0,collected_rows=0;int negative_ticks=0;bool safety_rejection_verified=false;
    std::vector<Comparison> checks={{"observations",2e-4,2e-4},{"actor_and_critic_forward",3e-5,3e-5},
        {"stored_latent_logp",3e-5,3e-5},{"one_step_physics_reward",2e-4,2e-3},{"independent_double_GAE",2e-4,2e-4},
        {"minibatch_statistics",2e-5,2e-5},{"restored_observations",2e-4,2e-4}};
    try {
        if(argc!=4) {std::cerr<<"Usage: command_motor_readiness source-boulder.bin reference-fixtures.bin NEW-build-output-dir\n";return 2;}
        const fs::path source_path=fs::absolute(argv[1]),capture_path=fs::absolute(argv[2]);output=fs::absolute(argv[3]);
        need(!fs::exists(output),"Readiness output must be a new directory");
        const auto build=fs::weakly_canonical(fs::current_path()/"build"),target=fs::weakly_canonical(output);
        const auto relative=target.lexically_relative(build);need(!relative.empty()&&!relative.is_absolute()&&*relative.begin()!=".."&&relative!=".","Readiness output must be below this workspace build directory");
        const auto source_bytes=checkpoint::read(source_path.string()),capture_bytes=checkpoint::read(capture_path.string());
        source_hash=hash(source_bytes);fixture_hash=hash(capture_bytes);const auto source=rl::read_boulder_source(source_bytes);source_fingerprint=source.fingerprint;
        need(source.fingerprint==frozen_source,"Readiness source is not the immutable original swimmer");const auto fixtures=read_captures(capture_bytes,source.fingerprint);
        fs::create_directories(output);owned_output=true;
        rl::CommandMotorTrainer learner(source.motor,source.fingerprint);const auto initial_snapshot=rl::command_motor_snapshot(learner);
        need(learner.update_id==0&&learner.actor_opt.steps()==0&&learner.critic_opt.steps()==0&&learner.std_opt.steps()==0,"Readiness must begin with fresh complete learner state");
        const auto actor_parameters=learner.actor.get_params(),critic_parameters=learner.critic.get_params();
        nn::CpuMlp<float> cpu_actor(rl::command_motor_specification()),cpu_critic(rl::command_motor_specification(1));
        cpu_actor.params()=actor_parameters;cpu_critic.params()=critic_parameters;const auto stream=learner.stream.value;
        unsigned flags=0;CUDA_CHECK_THROW(cudaStreamGetFlags(stream,&flags));need((flags&cudaStreamNonBlocking)!=0,"Expected native nonblocking training stream");
        const auto deviation=download(learner.log_std.get(),4,stream);need(deviation==std::vector<float>(4,-.5f),"Expected fresh fixed Gaussian deviations");
        std::vector<cm::State> starts;for(int i=0;i<world_count;++i) {auto state=i<2?fixtures[4+i].initial:cm::initial(cm::training_seed,static_cast<uint32_t>(i));
            state.seed=cm::training_seed;state.environment=static_cast<uint32_t>(i);state.episode=0;state.steps=i==7?599:0;
            need(bench::measure(state).clearance>=0,"A preregistered positive fixture starts outside the physical volume");starts.push_back(state);}
        cm::Batch worlds(world_count,cm::training_seed);worlds.set_states(starts,stream);
        device_ptr<float> observations(rows*inputs),means(rows*actions),values(rows),commands(rows*actions),logp(rows),rewards(rows),done(rows),truncated(rows),final_obs(rows*inputs);
        device_ptr<float> repeat_commands(world_count*actions),repeat_logp(world_count),final_values(rows),last_values(world_count),adv(rows),returns(rows);
        CUDA_CHECK_THROW(cudaMemsetAsync(final_obs.get(),0,final_obs.size_bytes(),stream));
        Rollout record;record.pre_states.reserve(rows);
        for(int tick=0;tick<steps;++tick) {
            const auto before=worlds.host_states(stream);
            const auto gpu_observation=download(worlds.obs(),world_count*inputs,stream);check_finite(gpu_observation,"observations");
            for(int i=0;i<world_count;++i) {float expected[inputs];cm::observe(before[i],expected);for(int j=0;j<inputs;++j) checks[0].value(gpu_observation[i*inputs+j],expected[j]);record.pre_states.push_back(before[i]);}
            const size_t row=static_cast<size_t>(tick)*world_count;
            rl::swimmer_copy(observations.get()+row*inputs,worlds.obs(),world_count*inputs,stream);
            learner.actor.forward(worlds.obs(),world_count,stream);rl::swimmer_copy(means.get()+row*actions,learner.actor.output(),world_count*actions,stream);
            const auto gpu_means=download(learner.actor.output(),world_count*actions,stream);cpu_actor.forward(gpu_observation.data(),world_count);
            for(int i=0;i<world_count*actions;++i) checks[1].value(gpu_means[i],cpu_actor.output()[i]);
            rl::sample_gaussian(learner.actor.output(),learner.log_std.get(),world_count,actions,cm::training_seed,static_cast<uint32_t>(tick),commands.get()+row*actions,logp.get()+row,stream);
            rl::sample_gaussian(learner.actor.output(),learner.log_std.get(),world_count,actions,cm::training_seed,static_cast<uint32_t>(tick),repeat_commands.get(),repeat_logp.get(),stream);
            const auto sampled=download(commands.get()+row*actions,world_count*actions,stream),old_logp=download(logp.get()+row,world_count,stream);
            need(sampled==download(repeat_commands.get(),world_count*actions,stream)&&old_logp==download(repeat_logp.get(),world_count,stream),"Fixed sampled RNG replay differs");
            check_finite(sampled,"latent actions");check_finite(old_logp,"stored old logp");
            for(int i=0;i<world_count;++i) {double expected=0;for(int j=0;j<actions;++j) {const double sigma=std::exp(static_cast<double>(deviation[j]));
                const double z=(static_cast<double>(sampled[i*actions+j])-gpu_means[i*actions+j])/sigma;expected+=-.5*z*z-deviation[j]-.5*std::log(2.*std::acos(-1.));}checks[2].value(old_logp[i],expected);}
            learner.critic.forward(worlds.obs(),world_count,stream);rl::swimmer_copy(values.get()+row,learner.critic.output(),world_count,stream);
            const auto gpu_values=download(learner.critic.output(),world_count,stream);cpu_critic.forward(gpu_observation.data(),world_count);
            for(int i=0;i<world_count;++i) checks[1].value(gpu_values[i],cpu_critic.output()[i]);
            worlds.step(commands.get()+row*actions,{rewards.get()+row,done.get()+row,truncated.get()+row,final_obs.get()+row*inputs},stream);
            const auto after=worlds.host_states(stream);
            collected_rows+=world_count;
            const auto reward=download(rewards.get()+row,world_count,stream),ended=download(done.get()+row,world_count,stream),trunc=download(truncated.get()+row,world_count,stream);
            const auto terminal_observation=download(final_obs.get()+row*inputs,world_count*inputs,stream);
            for(int i=0;i<world_count;++i) {
                auto expected=before[i];const auto result=cm::advance(expected,sampled.data()+i*actions);need(!result.invalid,"Positive CPU replay became nonfinite");
                checks[3].value(reward[i],result.reward);need(ended[i]==(result.timeout?1.f:0.f),"Native done flag differs from finite horizon");
                need(trunc[i]==0.f,"Declared600-tick finite horizon must NOT bootstrap as truncation");
                if(result.timeout) {++terminal_rows;float expected_final[inputs];cm::observe(expected,expected_final);
                    for(int j=0;j<inputs;++j) checks[0].value(terminal_observation[i*inputs+j],expected_final[j]);
                    expected=cm::initial(cm::training_seed,static_cast<uint32_t>(i),expected.episode+1);}
                state_check(checks[3],after[i],expected);
            }
        }
        need(terminal_rows==1,"Expected exactly one explicit terminal row");need(worlds.invalid_count(stream)==0,"Positive collection recorded invalid dynamics");
        need(worlds.unsafe_count(stream)==0,"Positive scratch collection encountered a physical breach; readiness does not pass");
        learner.critic.forward(worlds.obs(),world_count,stream);rl::swimmer_copy(last_values.get(),learner.critic.output(),world_count,stream);
        learner.critic.forward(final_obs.get(),rows,stream);rl::swimmer_copy(final_values.get(),learner.critic.output(),rows,stream);
        rl::compute_gae(steps,world_count,rewards.get(),values.get(),done.get(),truncated.get(),final_values.get(),last_values.get(),.99f,.95f,adv.get(),returns.get(),stream);
        record.observations=download(observations.get(),rows*inputs,stream);record.means=download(means.get(),rows*actions,stream);
        record.commands=download(commands.get(),rows*actions,stream);record.logp=download(logp.get(),rows,stream);record.critic_values=download(values.get(),rows,stream);
        record.rewards=download(rewards.get(),rows,stream);record.done=download(done.get(),rows,stream);record.truncated=download(truncated.get(),rows,stream);
        record.final_observations=download(final_obs.get(),rows*inputs,stream);record.advantages=download(adv.get(),rows,stream);record.returns=download(returns.get(),rows,stream);
        for(const auto* array:{&record.observations,&record.means,&record.commands,&record.logp,&record.critic_values,&record.rewards,&record.done,&record.truncated,&record.final_observations,&record.advantages,&record.returns}) check_finite(*array,"collected PPO data");
        const auto last=download(last_values.get(),world_count,stream),terminal_values=download(final_values.get(),rows,stream);
        const double gamma=static_cast<double>(.99f),lambda=static_cast<double>(.95f);
        for(int i=0;i<world_count;++i) {double carry=0;for(int tick=steps-1;tick>=0;--tick) {const int row=tick*world_count+i;const bool terminal=record.done[row]!=0;
            const double next=terminal?(record.truncated[row]!=0?terminal_values[row]:0.):(tick==steps-1?last[i]:record.critic_values[row+world_count]);
            const double delta=record.rewards[row]+gamma*next-record.critic_values[row];carry=delta+gamma*lambda*(terminal?0.:carry);
            checks[4].value(record.advantages[row],carry);checks[4].value(record.returns[row],carry+record.critic_values[row]);}}
        const auto permutation=rl::make_permutation(rows,rng::bits(cm::training_seed,0,0));std::vector<uint8_t> visited(rows,0);
        constexpr int mb=rows/8;device_ptr<float> mb_obs(mb*inputs),mb_actions(mb*actions),mb_logp(mb),mb_adv(mb),mb_ret(mb),statistics(2);
        for(int batch=0;batch<8;++batch) {
            rl::gather_rows(permutation,batch*mb,mb,inputs,observations.get(),mb_obs.get(),stream);rl::gather_rows(permutation,batch*mb,mb,actions,commands.get(),mb_actions.get(),stream);
            rl::gather_rows(permutation,batch*mb,mb,1,logp.get(),mb_logp.get(),stream);rl::gather_rows(permutation,batch*mb,mb,1,adv.get(),mb_adv.get(),stream);
            rl::gather_rows(permutation,batch*mb,mb,1,returns.get(),mb_ret.get(),stream);rl::mean_std(mb_adv.get(),mb,statistics.get(),stream);
            const auto o=download(mb_obs.get(),mb*inputs,stream),a=download(mb_actions.get(),mb*actions,stream),l=download(mb_logp.get(),mb,stream),v=download(mb_adv.get(),mb,stream),r=download(mb_ret.get(),mb,stream),stats=download(statistics.get(),2,stream);
            double mean=0;for(int j=0;j<mb;++j) {const auto source_row=rl::permute(permutation,static_cast<uint32_t>(batch*mb+j));need(source_row<rows&&visited[source_row]++==0,"Minibatch permutation is not bijective");
                for(int col=0;col<inputs;++col) need(o[j*inputs+col]==record.observations[source_row*inputs+col],"Gathered observation row differs");
                for(int col=0;col<actions;++col) need(a[j*actions+col]==record.commands[source_row*actions+col],"Gathered action row differs");
                need(l[j]==record.logp[source_row]&&v[j]==record.advantages[source_row]&&r[j]==record.returns[source_row],"Gathered scalar tuple differs");mean+=v[j];}
            mean/=mb;double variance=0;for(float value:v) variance+=(value-mean)*(value-mean);const double sd=std::sqrt(variance/mb);
            checks[5].value(stats[0],mean);checks[5].value(stats[1],sd);
            for(float value:v) checks[5].value((value-stats[0])/(stats[1]+1e-8f),(value-mean)/(sd+1e-8));
        }
        for(uint8_t count:visited) need(count==1,"A rollout row was omitted from minibatches");
        // A deliberately breached body tests fail-closed detection, not an
        // early reset, negative terminal reward, policy sample or PPO data row.
        cm::Batch negative(1,cm::training_seed);auto breached=cm::initial(cm::training_seed,0);breached.body={};breached.body.position={0,.2f,0};
        breached.request.position=breached.body.position;breached.request.heading={1,0,0};negative.set_states({breached},stream);
        device_ptr<float> negative_action(4);const std::vector<float> fixed_command={.1f,-.2f,.3f,-.4f};upload(negative_action,fixed_command,stream);
        device_ptr<float> negative_reward(1),negative_done(1),negative_trunc(1),negative_final(inputs);
        negative.step(negative_action.get(),{negative_reward.get(),negative_done.get(),negative_trunc.get(),negative_final.get()},stream);
        const auto negative_state=negative.host_states(stream);negative_ticks=1;
        need(negative.unsafe_count(stream)==1&&negative.unsafe_states(stream).size()==1,"Negative physical breach was not retained as safety evidence");
        const auto unsafe_evidence=negative.unsafe_states(stream);
        need(unsafe_evidence[0].steps==0&&exact_state(unsafe_evidence[0],breached),"Safety evidence must preserve the exact breached PRE-state, not the corrected solver endpoint");
        bool rejected_before_buffers=false;
        try {rl::CommandMotorTask::validate_collected(negative,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0,stream);}
        catch(const std::runtime_error& error) {
            const std::string message=error.what();rejected_before_buffers=message.find("physical clearance breach")!=std::string::npos&&message.find("before GAE/Adam")!=std::string::npos;
        }
        need(rejected_before_buffers,"The production task callback must reject physical breach before consuming buffers or reaching GAE/Adam");
        safety_rejection_verified=true;
        need(negative_state[0].steps==1&&negative_state[0].episode==0&&download(negative_done.get(),1,stream)[0]==0&&download(negative_trunc.get(),1,stream)[0]==0,"Unsafe state was silently terminated or reset");
        need(download(negative_action.get(),4,stream)==fixed_command,"Safety handling rewrote actuator commands");
        // No collection above touches learner.env; snapshot/restore must retain
        // the original512 worlds and all fresh actor/critic/std Adam moments.
        need(rl::command_motor_snapshot(learner)==initial_snapshot,"Readiness inference modified the complete learner");
        rl::CommandMotorTrainer restored(source.motor,source.fingerprint);
        need(restored.actor.get_params()==actor_parameters&&restored.critic.get_params()==critic_parameters&&
             download(restored.log_std.get(),4,restored.stream.value)==deviation,"Fresh model initialization is not repeatable");
        need(download(restored.env.obs(),rl::command_motor_worlds*inputs,restored.stream.value)==
             download(learner.env.obs(),rl::command_motor_worlds*inputs,stream),"Fresh physical initialization is not repeatable");
        rl::restore_command_motor(restored,initial_snapshot);need(rl::command_motor_snapshot(restored)==initial_snapshot,"Full learner/Adam/world restore is not exact");
        need(restored.update_id==0&&restored.actor_opt.steps()==0&&restored.critic_opt.steps()==0&&restored.std_opt.steps()==0,"Readiness performed learning");
        const auto restored_states=restored.env.host_states(restored.stream.value);
        const auto restored_obs=download(restored.env.obs(),rl::command_motor_worlds*inputs,restored.stream.value);
        for(int i=0;i<rl::command_motor_worlds;++i) {float expected[inputs];cm::observe(restored_states[i],expected);for(int j=0;j<inputs;++j) checks[6].value(restored_obs[i*inputs+j],expected[j]);}
        checkpoint::write_atomic((output/"fresh-complete-training.bin").string(),initial_snapshot);need(checkpoint::read((output/"fresh-complete-training.bin").string())==initial_snapshot,"Fresh snapshot file readback differs");
        checkpoint::Writer evidence;evidence.value(uint32_t{0x434D5231u});evidence.value(uint32_t{1});evidence.value(rl::command_motor_fingerprint());evidence.value(source.fingerprint);
        evidence.array(record.pre_states);evidence.array(record.observations);evidence.array(record.means);evidence.array(record.commands);evidence.array(record.logp);evidence.array(record.critic_values);
        evidence.array(record.rewards);evidence.array(record.done);evidence.array(record.truncated);evidence.array(record.final_observations);evidence.array(record.advantages);evidence.array(record.returns);
        checkpoint::write_atomic((output/"scratch-rollout.bin").string(),evidence.data);
        const bool preserved=checkpoint::read(source_path.string())==source_bytes&&checkpoint::read(capture_path.string())==capture_bytes;need(preserved,"Protected source/capture changed");
        bool passed=true;for(const auto& comparison:checks) passed=passed&&comparison.failed==0;
        write_report(output,passed,source_hash,source.fingerprint,fixture_hash,checks,terminal_rows,collected_rows,negative_ticks,safety_rejection_verified,preserved);
        std::cout<<"COMMAND_MOTOR_READINESS passed="<<passed<<" worlds=8 ticks=64 PPO_updates=0 Adam_steps=0\n";return passed?0:1;
    } catch(const std::exception& error) {
        std::cerr<<"COMMAND_MOTOR_READINESS failed: "<<error.what()<<'\n';
        if(owned_output) try {write_report(output,false,source_hash,source_fingerprint,fixture_hash,checks,terminal_rows,collected_rows,negative_ticks,safety_rejection_verified,false,error.what());} catch(...) {}
        return 1;
    }
}
