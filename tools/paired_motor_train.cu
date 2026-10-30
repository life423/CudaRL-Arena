#include "rl/paired_motor_ppo.h"
#include "rl/boulder_source.h"
#include "rl/gaussian_math.h"
#include "nn/mlp_cpu.h"
#include "cuda/device.h"
#include "motor_benchmark.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>

// New, bounded paired protocol. No old128-update gate, loop or file is changed.
namespace {
using namespace cudarl;
namespace pm=paired_motor;namespace cm=command_motor;namespace bench=motor_benchmark;namespace fs=std::filesystem;
void need(bool okay,const std::string& text) {if(!okay) throw std::runtime_error(text);}
std::string read_text(const fs::path& path) {std::ifstream file(path,std::ios::binary);need(bool(file),"Cannot read "+path.string());return {std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};}
void immutable_text(const fs::path& path,const std::string& value) {
    if(fs::exists(path)) {need(read_text(path)==value,"Immutable paired record differs: "+path.string());return;}
    std::ofstream file(path,std::ios::binary);need(bool(file),"Cannot create "+path.string());file<<value;file.flush();need(bool(file),"Cannot write "+path.string());
}
void immutable_payload(const fs::path& path,const std::vector<uint8_t>& bytes) {
    if(fs::exists(path)) need(checkpoint::read(path.string())==bytes,"Immutable paired payload differs: "+path.string());else checkpoint::write_atomic(path.string(),bytes);
}
template<class T> std::vector<T> download(const T* pointer,size_t count,cudaStream_t stream) {
    std::vector<T> values(count);CUDA_CHECK_THROW(cudaMemcpyAsync(values.data(),pointer,count*sizeof(T),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));return values;
}
template<class T> void upload(device_ptr<T>& pointer,const std::vector<T>& values,cudaStream_t stream) {
    need(pointer.size()==values.size(),"Paired upload dimensions differ");CUDA_CHECK_THROW(cudaMemcpyAsync(pointer.get(),values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
pm::RegulationGate prepared_gate(const pm::RegulationProtocol& protocol) {
    pm::RegulationGate gate;gate.protocol=protocol;gate.gains=protocol.gains;gate.source_fingerprint=protocol.source_fingerprint;
    gate.catalog_fingerprint=protocol.catalog_fingerprint();gate.protocol_payload_hash=protocol.payload_fingerprint();gate.validate_binding(protocol.source_fingerprint);return gate;
}
rl::PairedMotorPolicy initial_policy(const rl::BoulderSource& source,const pm::RegulationProtocol& protocol,pm::Interface mode) {
    rl::PairedMotorPolicy out;out.interface_type=mode;out.gains=protocol.gains;out.source_fingerprint=source.fingerprint;out.environment_fingerprint=rl::paired_motor_fingerprint(mode,protocol);
    out.actor=rl::initialize_paired_motor(source.motor);out.regulation_gate=prepared_gate(protocol).serialize();return out;
}
std::string manifest(const rl::BoulderSource& source,const pm::RegulationProtocol& protocol) {
    std::ostringstream out;out<<std::setprecision(17);
    out<<"{\n\"protocol\":\"paired-motor-v1\",\n\"source_fingerprint\":\""<<source.fingerprint<<"\",\n\"regulation_protocol_hash\":\""<<protocol.payload_fingerprint()
       <<"\",\n\"regulation_catalog_hash\":\""<<protocol.catalog_fingerprint()<<"\",\n\"benchmark_rules_hash\":\""<<bench::rules_fingerprint()<<"\",\n"
       <<"\"arms\":[\"torque\",\"joint_target\"],\n\"updates_per_arm\":64,\n\"worlds\":512,\n\"rollout\":64,\n\"steps_per_arm\":2097152,\n\"maximum_training_steps_total\":4194304,\n"
       <<"\"actor\":\"same59-32-32-4 tanh/tanh/linear raw initialization; accepted55 columns9..32 plus hidden layers; fresh small gain0.01 head; no food imitation\",\n"
       <<"\"seeds\":{\"head\":"<<rl::paired_motor_head_seed<<",\"training\":"<<rl::paired_motor_seed<<",\"final\":"<<rl::paired_motor_final_seed<<"},\n"
       <<"\"actuation\":\"torque:clip(z,-1,1) normalized internal torque; joint_target:q*=limit*clip(z,-1,1), SI PD recomputed EACH5ms substep, clamp to same +/-10uNm; no rate limiter\",\n"
       <<"\"policy_likelihood\":\"unclipped raw4D Gaussian latent z; stored actions/logp replay z, never torque or clipped target; no post-servo Jacobian\",\n"
       <<"\"observation\":\"same59 command/body fields;29..32 last ACTUAL5ms normalized torque in both arms; held_target is saved actuator metadata and overwritten by each new50ms command\",\n"
       <<"\"cost\":\"-0.05*(||position_error/.072||^2+.25||terminal_velocity_error/.02||^2+.1*(1-heading_dot)+.05*MEAN216(point_speed^2)/.02^2+.02||angular_velocity/.5||^2+.02*MEAN4((joint_rate/4)^2)+.002*MEAN40(actual_normalized_torque^2))\",\n"
       <<"\"task\":\"calm rectangular tank; persistent p*, zero terminal velocity, LEVEL heading; same articulated mass/drag/buoyancy/contact/substep;600tick true terminal, remaining time observed; goal chooser excluded\",\n"
       <<"\"ppo\":\"gamma.99 lambda.95 clip.2 entropy.002 vf.5;4epochs8minibatches; fresh actor/critic/std Adam lr.001 beta.9/.999 eps1e-5 clipnorm.5; initial logstd-.5\",\n"
       <<"\"gates\":\"one predeclared3200-control-tick regulation test(1600CPU+1600CUDA), focused recorded-data observation/effort/rawGaussian replay and zero-update fullstate resume; old16trial conventional gate not used or altered\",\n"
       <<"\"readiness_budget\":\"NO additional physical integration or PPO updates; reuse regulation frames; synthetic Gaussian actions are not training data\",\n"
       <<"\"evaluation\":\"no midrun evaluations; fixedupdate64 both arms and original55 once on identical64 final starts; captured original actual-motion braking warmups ONCE before learning; total final eval+capture controlticks<=115200; no selection/sweep/adoption\",\n"
       <<"\"failure\":\"sticky unsafe/nonfinite worlds rejected before GAE/Adam; preserve rejected rollout and lastgood fullsave; global stop forbids restart after failure; any exported partial candidate is its accurately labelled last completed update\",\n\"SI_kp\":[";
    for(int j=0;j<4;++j) {if(j) out<<',';out<<protocol.gains.kp[j];}out<<"],\n\"SI_kd\":[";for(int j=0;j<4;++j) {if(j) out<<',';out<<protocol.gains.kd[j];}out<<"],\n\"protocol_values\":[";
    const auto values=rl::paired_motor_protocol();for(size_t i=0;i<values.size();++i) {if(i) out<<',';out<<values[i];}out<<"]\n}\n";return out.str();
}
__global__ void recorded_observations(const pm::State* states,float* out,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count) pm::observe(states[i],out+i*pm::kObs);
}
void readiness(const rl::BoulderSource& source,const pm::RegulationProtocol& protocol,const pm::RegulationGate& gate,
        const std::vector<uint8_t>& frame_bytes,const fs::path& root) {
    const auto frames=pm::read_regulation_frames(frame_bytes,protocol);need(!frames.empty(),"No regulation frames were recorded");
    rl::PairedMotorReadiness result;result.source_fingerprint=source.fingerprint;result.protocol_payload_hash=protocol.payload_fingerprint();result.regulation_gate_hash=pm::fingerprint(gate.serialize());
    result.recorded_frames_hash=pm::fingerprint(frame_bytes);result.recorded_frames=static_cast<int>(frames.size());
    result.torque_task_fingerprint=rl::paired_motor_fingerprint(pm::Interface::Torque,protocol);result.target_task_fingerprint=rl::paired_motor_fingerprint(pm::Interface::JointTarget,protocol);
    std::vector<pm::State> states;std::vector<float> cpu;
    double effort_error=0,reward_error=0,observation_error=0,logp_error=0;int outside_latents=0;
    bool seen[2][8][200]{};
    for(const auto& frame:frames) {
        need(!seen[frame.backend][frame.case_id][frame.tick-1],"Regulation records contain duplicate backend/case/tick");seen[frame.backend][frame.case_id][frame.tick-1]=true;
        need(!frame.result.invalid&&pm::finite(frame.after),"Regulation invalid result prevents claiming data readiness");
        states.push_back(frame.before);states.push_back(frame.after);
        double actual_mean=0;for(int i=0;i<40;++i) {const double value=frame.after.actual_torques[i];actual_mean+=value*value/40.;}
        const double error=fabs(actual_mean-frame.after.mean_squared_torque);effort_error=std::max(effort_error,error);need(error<=2e-6,"Recorded effort is not mean40 actual torque squared");
        for(int j=0;j<4;++j) {
            need(frame.after.held_target[j]==deep::limit*deep::clamp(frame.latent[j],-1,1),"Recorded servo held target differs from clipped latent");
            need(frame.after.task.previous_action[j]==frame.after.actual_torques[36+j],"Motor observation history is not last actual substep torque");
            for(int tick=0;tick<10;++tick) need(fabsf(deep::clamp(frame.after.unclamped_torques[tick*4+j]/deep::torque,-1,1)-frame.after.actual_torques[tick*4+j])<=3e-6f,"Recorded SI torque and actuator clamp disagree");
        }
        auto cost=cm::tracking_cost(frame.after.task,frame.latent,cm::motion(frame.after.task.body));cost.torque=.002f*frame.after.mean_squared_torque;
        const double expected=-deep::dt*cost.total();reward_error=std::max(reward_error,fabs(expected-frame.result.reward));
        need(fabs(expected-frame.result.reward)<=2e-4+2e-3*fabs(expected),"Recorded reward does not charge actual applied effort");
    }
    cpu.resize(states.size()*pm::kObs);for(size_t i=0;i<states.size();++i) pm::observe(states[i],cpu.data()+i*pm::kObs);
    rl::SwimmerStream stream;const auto s=stream.value;device_ptr<pm::State> d_states(states.size());device_ptr<float> d_obs(cpu.size());upload(d_states,states,s);
    recorded_observations<<<(static_cast<int>(states.size())+63)/64,64,0,s>>>(d_states.get(),d_obs.get(),static_cast<int>(states.size()));CUDA_CHECK_THROW(cudaGetLastError());
    const auto gpu=download(d_obs.get(),cpu.size(),s);for(size_t i=0;i<cpu.size();++i) {const double error=fabs(static_cast<double>(cpu[i])-gpu[i]);observation_error=std::max(observation_error,error);need(error<=2e-5+2e-5*fabs(cpu[i]),"Recorded CPU/CUDA observations differ");}
    result.observations=true;result.effort=true;
    std::vector<float> observations(512*59);for(int i=0;i<512;++i) {
        const size_t row=rng::bits(rl::paired_motor_seed,0x52454144u,static_cast<uint32_t>(i))%states.size();std::copy_n(cpu.data()+row*59,59,observations.data()+i*59);
    }
    device_ptr<float> inputs(observations.size()),sigma(4),actions(512*4),old_logp(512),repeat_actions(512*4),repeat_logp(512);upload(inputs,observations,s);upload(sigma,std::vector<float>(4,-.5f),s);
    const auto weights=rl::initialize_paired_motor(source.motor);nn::GpuMlp actor(rl::paired_motor_specification(),512,rl::paired_motor_seed);actor.set_params(weights);CUDA_CHECK_THROW(cudaDeviceSynchronize());
    actor.forward(inputs.get(),512,s);const auto means=download(actor.output(),512*4,s);
    rl::sample_gaussian(actor.output(),sigma.get(),512,4,rl::paired_motor_seed,0,actions.get(),old_logp.get(),s);
    const auto latent=download(actions.get(),512*4,s),saved_logp=download(old_logp.get(),512,s);
    actor.forward(inputs.get(),512,s);rl::sample_gaussian(actor.output(),sigma.get(),512,4,rl::paired_motor_seed,0,repeat_actions.get(),repeat_logp.get(),s);
    need(download(repeat_actions.get(),512*4,s)==latent&&download(repeat_logp.get(),512,s)==saved_logp,"Counter-seeded raw Gaussian replay differs");
    for(int i=0;i<512;++i) {
        double mu[4],a[4],deviation[4]={-.5,-.5,-.5,-.5};for(int j=0;j<4;++j) {mu[j]=means[i*4+j];a[j]=latent[i*4+j];outside_latents+=fabs(a[j])>1?1:0;}
        const double expected=rl::gaussian_logp(mu,deviation,a,4);logp_error=std::max(logp_error,fabs(expected-saved_logp[i]));need(fabs(expected-saved_logp[i])<=2e-5+2e-5*fabs(expected),"Raw latent log probability replay differs");
    }
    need(outside_latents>0,"Raw Gaussian readiness did not exercise values outside actuator/target bounds");
    device_ptr<float> zero(512),one(512),adv_stats(2),dmean(512*4),dstd(512*4),dvalue(512),row_stats(512*rl::kGaussianStats),stats(rl::kGaussianStats);
    upload(zero,std::vector<float>(512,0),s);upload(one,std::vector<float>(512,1),s);upload(adv_stats,std::vector<float>{0,1},s);CUDA_CHECK_THROW(cudaMemsetAsync(stats.get(),0,stats.size_bytes(),s));
    rl::gaussian_ppo_loss_grad(actor.output(),sigma.get(),zero.get(),512,4,actions.get(),old_logp.get(),one.get(),zero.get(),adv_stats.get(),{.2f,.002f,.5f},dmean.get(),dstd.get(),dvalue.get(),row_stats.get(),stats.get(),s);
    const auto terms=download(stats.get(),rl::kGaussianStats,s);need(fabsf(terms[3])<=1e-6f&&terms[4]==0,"Unchanged raw latent policy has nonzero replay KL/clipping");
    need(download(actions.get(),512*4,s)==latent&&download(old_logp.get(),512,s)==saved_logp&&actor.get_params()==weights,"Likelihood readiness mutated raw actions/logp/actor");result.latent_replay=true;
    std::vector<float> initial_actor,initial_critic,initial_sigma;std::vector<uint8_t> initial_optimizers;
    for(auto mode:{pm::Interface::Torque,pm::Interface::JointTarget}) {
        rl::PairedMotorTrainer trainer(source.motor,mode,gate);const auto before=rl::paired_motor_snapshot(trainer);
        const auto a=trainer.actor.get_params(),c=trainer.critic.get_params(),d=download(trainer.log_std.get(),4,trainer.stream.value);
        checkpoint::Writer optimizers;trainer.actor_opt.save(optimizers);trainer.critic_opt.save(optimizers);trainer.std_opt.save(optimizers);
        if(initial_actor.empty()) {initial_actor=a;initial_critic=c;initial_sigma=d;}else need(a==initial_actor&&c==initial_critic&&d==initial_sigma,"Paired arms do not share identical fresh model/std initialization");
        if(initial_optimizers.empty()) initial_optimizers=optimizers.data;else need(initial_optimizers==optimizers.data,"Paired arms do not share identical fresh zero Adam states");
        rl::PairedMotorTrainer restored(source.motor,mode,gate);rl::restore_paired_motor(restored,before);need(rl::paired_motor_snapshot(restored)==before,"Paired complete zero-update snapshot/held target/Adam replay differs");
        need(trainer.update_id==0&&trainer.actor_opt.steps()==0&&trainer.critic_opt.steps()==0&&trainer.std_opt.steps()==0,"Readiness accidentally advanced PPO/Adam");
        immutable_payload(root/"run"/rl::paired_motor_arm(mode)/"readiness-initial-training.bin",before);
    }
    result.complete_resume=true;immutable_payload(root/"readiness-gate.bin",result.serialize());
    std::ostringstream report;report<<std::setprecision(17)<<"{\"recorded_frames\":"<<frames.size()<<",\"observations_max_abs\":"<<observation_error<<",\"actual_effort_max_abs\":"<<effort_error
        <<",\"reward_max_abs\":"<<reward_error<<",\"raw_logp_max_abs\":"<<logp_error<<",\"unclipped_latents_outside_bounds\":"<<outside_latents<<",\"replay_kl_sum\":"<<terms[3]
        <<",\"regulation_passed\":"<<(gate.passed?"true":"false")<<",\"data_readiness_passed\":"<<(result.passed()?"true":"false")<<",\"additional_physical_steps\":0,\"ppo_updates\":0,\"adopted\":false}\n";
    immutable_text(root/"readiness-summary.json",report.str());std::cout<<"PAIRED_READINESS "<<report.str();
}
std::vector<bench::Fixture> final_fixtures(const rl::BoulderSource& source,const fs::path& path) {
    if(fs::exists(path)) {
        const auto bytes=checkpoint::read(path.string());checkpoint::Reader in{bytes};need(in.value<uint32_t>()==0x504D4649u&&in.value<uint32_t>()==1&&in.value<uint64_t>()==source.fingerprint&&in.value<uint64_t>()==bench::rules_fingerprint(),"Paired final fixture binding differs");
        auto cases=in.array<bench::Fixture>(64);in.finish();need(cases.size()==64,"Paired final suite must have64 cases");return cases;
    }
    std::vector<bench::Fixture> cases;bench::FrozenMotor original(source.motor,bench::TargetMode::Original55);
    for(int i=0;i<64;++i) {auto fixture=bench::make_fixture(rl::paired_motor_final_seed,bench::Suite::Final,i);bench::prepare_braking(fixture,original);cases.push_back(fixture);}
    checkpoint::Writer out;out.value(uint32_t{0x504D4649u});out.value(uint32_t{1});out.value(source.fingerprint);out.value(bench::rules_fingerprint());out.array(cases);immutable_payload(path,out.data);return cases;
}
void save_training(const rl::PairedMotorTrainer& trainer,const rl::BoulderSource& source,const fs::path& path,bool full_verify) {
    const auto bytes=rl::paired_motor_snapshot(trainer);checkpoint::write_atomic(path.string(),bytes);need(checkpoint::read(path.string())==bytes,"Paired fullsave checksum readback differs");
    if(full_verify) {rl::PairedMotorTrainer clone(source.motor,trainer.interface_type,trainer.regulation);rl::restore_paired_motor(clone,bytes);need(rl::paired_motor_snapshot(clone)==bytes,"Paired fullsave restoration differs");}
}
bool train_arm(const rl::BoulderSource& source,const pm::RegulationGate& gate,pm::Interface mode,const fs::path& root) {
    const auto path=root/"run"/rl::paired_motor_arm(mode);rl::PairedMotorTrainer trainer(source.motor,mode,gate);
    if(checkpoint::exists((path/"latest-training.bin").string())) {const auto bytes=checkpoint::read((path/"latest-training.bin").string());rl::restore_paired_motor(trainer,bytes);need(rl::paired_motor_snapshot(trainer)==bytes,"Paired resumed fullstate differs");}
    else save_training(trainer,source,path/"latest-training.bin",true);
    std::ofstream log(path/"training.csv",std::ios::app);need(bool(log),"Cannot write paired training history");log<<std::setprecision(9);
    if(trainer.update_id==0) log<<"update,physical_training_steps,seconds,actor_adam,critic_adam,std_adam,reward_mean,actor_gradient_norm,logstd0,logstd1,logstd2,logstd3\n";
    while(trainer.update_id<64) {
        const auto before=rl::paired_motor_snapshot(trainer);const int old_update=trainer.update_id,old_adam=trainer.actor_opt.steps();double elapsed=0;
        try {elapsed=trainer.update_seconds();}
        catch(const std::exception& error) {
            checkpoint::Writer failure;failure.value(uint32_t{0x504D4641u});failure.value(uint32_t{1});failure.value(mode);failure.value(old_update);failure.value(old_adam);failure.value(trainer.actor_opt.steps());failure.array(before);
            failure.value(trainer.env.invalid_count(trainer.stream.value));failure.value(trainer.env.unsafe_count(trainer.stream.value));failure.array(trainer.env.unsafe_states(trainer.stream.value));failure.array(trainer.env.host_states(trainer.stream.value));
            failure.array(download(trainer.obs.get(),trainer.obs.size(),trainer.stream.value));failure.array(download(trainer.actions.get(),trainer.actions.size(),trainer.stream.value));failure.array(download(trainer.logp.get(),trainer.logp.size(),trainer.stream.value));failure.array(download(trainer.rewards.get(),trainer.rewards.size(),trainer.stream.value));
            checkpoint::write_atomic((path/"rejected-rollout.bin").string(),failure.data);
            // Export only the accurately labelled LAST completed safe update.
            // The failed trajectory remains in its separate immutable record.
            rl::restore_paired_motor(trainer,before);if(old_update>0) checkpoint::write_atomic((path/"candidate-policy.bin").string(),rl::paired_motor_policy(trainer).serialize());
            std::ostringstream report;report<<"{\"arm\":\""<<rl::paired_motor_arm(mode)<<"\",\"last_completed_update\":"<<old_update<<",\"attempted_update\":"<<old_update+1
                <<",\"adam_before\":"<<old_adam<<",\"source_unchanged\":true,\"automatic_restart\":false,\"adopted\":false}\n";immutable_text(root/"global-stop.json",report.str());
            std::cerr<<"PAIRED_STOP arm="<<rl::paired_motor_arm(mode)<<" completed_update="<<old_update<<" reason="<<error.what()<<'\n';return false;
        }
        const auto rewards=download(trainer.rewards.get(),trainer.batch,trainer.stream.value),stds=download(trainer.log_std.get(),4,trainer.stream.value);double sum=0;
        for(float value:rewards) {need(std::isfinite(value),"Nonfinite paired reward after successful validation");sum+=value;}
        log<<trainer.update_id<<','<<static_cast<long long>(trainer.update_id)*trainer.batch<<','<<elapsed<<','<<trainer.actor_opt.steps()<<','<<trainer.critic_opt.steps()<<','<<trainer.std_opt.steps()<<','<<sum/rewards.size()<<','<<trainer.actor_opt.last_grad_norm();for(float value:stds) log<<','<<value;log<<'\n';log.flush();
        save_training(trainer,source,path/"latest-training.bin",trainer.update_id==64);
        checkpoint::write_atomic((path/"candidate-policy.bin").string(),rl::paired_motor_policy(trainer).serialize());
        std::cout<<"PAIRED_TRAIN arm="<<rl::paired_motor_arm(mode)<<" update="<<trainer.update_id<<"/64 steps="<<static_cast<long long>(trainer.update_id)*trainer.batch<<" adam="<<trainer.actor_opt.steps()<<'\n';
    }
    need(trainer.update_id==64&&trainer.actor_opt.steps()==2048&&trainer.critic_opt.steps()==2048&&trainer.std_opt.steps()==2048,"Paired final counters differ");return true;
}
struct Evaluation {int cases=64,qualified=0,safe=0,motion=0,settled=0,held=0,passed=0;uint64_t physical_ticks=0;};
Evaluation evaluate(const std::vector<float>& actor,pm::Interface mode,const pm::Gains& gains,const std::vector<bench::Fixture>& fixtures,const fs::path& output,bool original=false) {
    rl::SwimmerStream stream;const auto s=stream.value;pm::Batch worlds(64,0xE71A0026u);worlds.configure(mode,gains,s);
    std::vector<pm::State> states(64);for(int i=0;i<64;++i) states[i].task=fixtures[i].initial;worlds.set_states(states,s);
    nn::CpuMlp<float> policy(rl::paired_motor_specification());if(!original) policy.params()=actor;bench::FrozenMotor legacy(actor,original?bench::TargetMode::Original55:bench::TargetMode::Candidate59);
    std::vector<bench::Progress> progress(64);std::vector<deep::Physics> physical(64);std::vector<uint8_t> active(64);std::vector<float> latent(64*4);
    device_ptr<float> d_latent(latent.size());device_ptr<deep::Physics> d_physical(64);device_ptr<uint8_t> d_active(64);
    std::ofstream trace(output.string()+"-trace.csv");need(bool(trace),"Cannot write paired physical trace");trace<<std::setprecision(9);
    trace<<"case,tick,head_x_model,head_y_model,head_z_model,raw0,raw1,raw2,raw3,held_target0,held_target1,held_target2,held_target3,last_actual0,last_actual1,last_actual2,last_actual3,mean40_torque2,head_m_s,com_m_s,max_surface_m_s,goal_error_m,heading_error_rad,clearance_m,first_unsafe_substep\n";
    auto predict=[&]() {for(int i=0;i<64;++i) {
        if(original) {const auto a=legacy.action(states[i].task);for(int j=0;j<4;++j) latent[i*4+j]=a[j];}
        else {float obs[59];pm::observe(states[i],obs);policy.forward(obs,1);for(int j=0;j<4;++j) latent[i*4+j]=policy.output()[j];}
    }};
    auto record=[&](int i) {const auto sample=bench::measure(states[i].task);bench::assess(progress[i],states[i].task,sample,fixtures[i]);
        if(states[i].first_unsafe_substep>=0) {progress[i].penetrated=true;progress[i].minimum_clearance=std::min(progress[i].minimum_clearance,cm::physical_clearance(states[i].first_unsafe_task.body));if(progress[i].first_failure_step<0) progress[i].first_failure_step=states[i].task.steps;}
        trace<<i<<','<<states[i].task.steps<<','<<states[i].task.body.position.x<<','<<states[i].task.body.position.y<<','<<states[i].task.body.position.z;
        for(int j=0;j<4;++j) trace<<','<<latent[i*4+j];for(float value:states[i].held_target) trace<<','<<value;for(float value:states[i].task.previous_action) trace<<','<<value;
        trace<<','<<states[i].mean_squared_torque<<','<<sample.head_speed<<','<<sample.com_speed<<','<<sample.point_speed<<','<<sample.goal_error<<','<<sample.heading_error<<','<<sample.clearance<<','<<states[i].first_unsafe_substep<<'\n';};
    predict();for(int i=0;i<64;++i) record(i);Evaluation total;
    for(int tick=0;tick<600;++tick) {
        bool any=false;for(int i=0;i<64;++i) {active[i]=static_cast<uint8_t>(fixtures[i].qualified&&tick<fixtures[i].evaluation_ticks&&progress[i].finite&&!progress[i].penetrated);any=any||active[i];physical[i]=bench::physics_for(fixtures[i],states[i].task.steps);}
        if(!any) break;predict();for(float value:latent) need(std::isfinite(value),"Frozen paired comparison generated nonfinite latent action");upload(d_latent,latent,s);upload(d_physical,physical,s);upload(d_active,active,s);
        worlds.step_physics(d_latent.get(),d_physical.get(),d_active.get(),s);states=worlds.host_states(s);
        for(int i=0;i<64;++i) if(active[i]) {++total.physical_ticks;record(i);}need(worlds.invalid_count(s)==0,"Frozen paired final encountered nonfinite dynamics");
    }
    std::ofstream cases(output.string()+"-cases.csv");need(bool(cases),"Cannot write paired case outcomes");cases<<std::setprecision(9);
    cases<<"case,kind,qualified,safety,motion,settle,hold,passed,half_s,reach_s,stop_s,stop_distance_m,hold_seconds,path_m,path_efficiency,reversals,min_clearance_m,first_safety_tick\n";
    for(int i=0;i<64;++i) {const auto o=bench::finish(progress[i],fixtures[i]);total.qualified+=o.qualified;total.safe+=o.safety;total.motion+=o.motion;total.settled+=o.settle;total.held+=o.hold;total.passed+=o.passed;
        cases<<i<<','<<bench::name(fixtures[i].kind)<<','<<o.qualified<<','<<o.safety<<','<<o.motion<<','<<o.settle<<','<<o.hold<<','<<o.passed<<','<<o.half_time<<','<<o.reach_time<<','<<o.stop_time<<','<<o.stop_distance<<','<<o.hold_seconds<<','<<progress[i].path<<','<<bench::path_efficiency(progress[i])<<','<<progress[i].reversals<<','<<progress[i].minimum_clearance<<','<<progress[i].first_failure_step<<'\n';}
    return total;
}
std::string evaluation_json(const Evaluation& value) {std::ostringstream out;out<<"{\"cases\":64,\"qualified\":"<<value.qualified<<",\"safe\":"<<value.safe<<",\"motion\":"<<value.motion<<",\"settled\":"<<value.settled<<",\"held\":"<<value.held<<",\"passed\":"<<value.passed<<",\"physical_ticks\":"<<value.physical_ticks<<'}';return out.str();}
}
int main(int argc,char** argv) {
    try {
        if(argc!=5&&argc!=7) {std::cerr<<"Usage: paired_motor_train SOURCE REGULATION-PROTOCOL STUDYROOT --prepare\n       paired_motor_train SOURCE REGULATION-PROTOCOL STUDYROOT --readiness REGULATION-GATE RECORDED-FRAMES\n       paired_motor_train SOURCE REGULATION-PROTOCOL STUDYROOT --run REGULATION-GATE READINESS-GATE\n";return 2;}
        const std::string mode=argv[4];need((mode=="--prepare"&&argc==5)||((mode=="--readiness"||mode=="--run")&&argc==7),"Explicit paired prepare/readiness/run mode required");
        const auto source_bytes=checkpoint::read(argv[1]);const auto source=rl::read_boulder_source(source_bytes);const auto protocol_bytes=checkpoint::read(argv[2]);const auto protocol=pm::RegulationProtocol::read(protocol_bytes);
        need(protocol.source_fingerprint==source.fingerprint,"Regulation protocol has another inherited source");const fs::path root=fs::absolute(argv[3]);
        fs::create_directories(root/"sources");for(auto arm:{"torque","joint_target"}) fs::create_directories(root/"run"/arm);
        immutable_payload(root/"sources"/"source-boulder.bin",source_bytes);immutable_payload(root/"sources"/"regulation-protocol.bin",protocol_bytes);immutable_text(root/"paired-protocol.json",manifest(source,protocol));
        for(auto arm:{pm::Interface::Torque,pm::Interface::JointTarget}) immutable_payload(root/"run"/rl::paired_motor_arm(arm)/"initial-policy.bin",initial_policy(source,protocol,arm).serialize());
        if(mode=="--prepare") {std::cout<<"PAIRED_PREPARED protocol=paired-motor-v1 physical_steps=0 training_updates=0\n";return 0;}
        const auto gate_bytes=checkpoint::read(argv[5]);const auto gate=pm::RegulationGate::read(gate_bytes);gate.validate_binding(source.fingerprint);need(gate.protocol.serialize()==protocol_bytes,"Regulation gate changed the registered protocol");
        need(probe_device().available,"A tested CUDA device is required for paired data readiness/training");
        if(mode=="--readiness") {const auto frames=checkpoint::read(argv[6]);immutable_payload(root/"sources"/"recorded-regulation-frames.bin",frames);immutable_payload(root/"sources"/"regulation-gate.bin",gate_bytes);
            readiness(source,protocol,gate,frames,root);need(checkpoint::read(argv[1])==source_bytes,"Original source changed during readiness");return 0;}
        gate.require_passed(source.fingerprint);const auto ready=rl::PairedMotorReadiness::read(checkpoint::read(argv[6]));ready.require_passed(gate);
        need(!fs::exists(root/"global-stop.json")&&!fs::exists(root/"final-summary.json")&&!fs::exists(root/"final-evaluation-started.json"),"This paired study already stopped/completed or began its single final suite; no restart or repeat allowed");
        checkpoint::Lease lease((root/"run"/"paired-study").string());immutable_payload(root/"sources"/"passed-regulation-gate.bin",gate_bytes);
        const auto fixtures=final_fixtures(source,root/"sources"/"final-fixtures.bin");uint64_t capture_ticks=0;for(const auto& fixture:fixtures) capture_ticks+=fixture.warmup_ticks;
        need(capture_ticks<=1600,"Paired final capture exceeds8x200 warmup bound");
        for(auto arm:{pm::Interface::Torque,pm::Interface::JointTarget}) if(!train_arm(source,gate,arm,root)) {need(checkpoint::read(argv[1])==source_bytes,"Original source changed during stopped paired study");return 3;}
        const auto torque=rl::PairedMotorPolicy::load((root/"run"/"torque"/"candidate-policy.bin").string()),target=rl::PairedMotorPolicy::load((root/"run"/"joint_target"/"candidate-policy.bin").string());
        need(torque.update==64&&target.update==64,"Both paired arms must finish fixed64 updates before final evaluation");
        immutable_text(root/"final-evaluation-started.json","{\"fixed_update\":64,\"no_selection\":true,\"policies\":3,\"cases_each\":64}\n");
        const auto torque_result=evaluate(torque.actor,pm::Interface::Torque,protocol.gains,fixtures,root/"run"/"torque"/"final"),target_result=evaluate(target.actor,pm::Interface::JointTarget,protocol.gains,fixtures,root/"run"/"joint_target"/"final");
        const auto original_result=evaluate(source.motor,pm::Interface::Torque,protocol.gains,fixtures,root/"run"/"original-final",true);
        const uint64_t total_ticks=capture_ticks+torque_result.physical_ticks+target_result.physical_ticks+original_result.physical_ticks;need(total_ticks<=115200,"Paired frozen comparison exceeds registered physical evaluation budget");
        auto final_torque=torque,final_target=target;final_torque.passed_final_gate=torque_result.passed==64;final_target.passed_final_gate=target_result.passed==64;
        checkpoint::write_atomic((root/"run"/"torque"/"candidate-policy.bin").string(),final_torque.serialize());checkpoint::write_atomic((root/"run"/"joint_target"/"candidate-policy.bin").string(),final_target.serialize());
        need(checkpoint::read(argv[1])==source_bytes&&checkpoint::read(argv[2])==protocol_bytes,"Protected source/protocol changed during paired comparison");
        std::ostringstream summary;summary<<"{\"protocol\":\"paired-motor-v1\",\"updates_each\":64,\"adam_steps_each\":2048,\"training_steps_total\":4194304,\"torque\":"<<evaluation_json(torque_result)<<",\"joint_target\":"<<evaluation_json(target_result)<<",\"original55\":"<<evaluation_json(original_result)
            <<",\"capture_ticks\":"<<capture_ticks<<",\"total_final_ticks\":"<<total_ticks<<",\"source_unchanged\":true,\"adopted\":false,\"legacy_unsupported_fields\":\"Original55 virtual target has no explicit terminal velocity/heading/hold interface\",\"trace_alignment\":\"tick0 action preview; later rows driving previous-state latent and resulting actual substep telemetry\"}\n";
        immutable_text(root/"final-summary.json",summary.str());std::cout<<"PAIRED_FINISHED "<<summary.str();return 0;
    } catch(const std::exception& error) {std::cerr<<"PAIRED_STOPPED "<<error.what()<<'\n';return 1;}
}
