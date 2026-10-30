#include "rl/motor_v2_ppo.h"
#include "rl/boulder_source.h"
#include "rl/gaussian_math.h"
#include "nn/mlp_cpu.h"
#include "cuda/device.h"
#include "motor_v2_benchmark.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <cstring>

// One newly registered study. Historical v1 checkpoints are never continued.
namespace {
using namespace cudarl;namespace mv=motor_v2;namespace mc=motor_contact_v2;namespace pm=paired_motor;namespace mb=motor_v2_benchmark;namespace fs=std::filesystem;
void need(bool okay,const std::string& message) {if(!okay) throw std::runtime_error(message);}
std::string text_file(const fs::path& path) {std::ifstream in(path,std::ios::binary);need(bool(in),"Cannot read "+path.string());return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};}
void immutable_text(const fs::path& path,const std::string& value) {
    if(fs::exists(path)) {need(text_file(path)==value,"Immutable motor-v2 record differs: "+path.string());return;}
    std::ofstream out(path,std::ios::binary);need(bool(out),"Cannot create "+path.string());out<<value;out.flush();need(bool(out),"Cannot write "+path.string());
}
void immutable_payload(const fs::path& path,const std::vector<uint8_t>& bytes) {
    if(fs::exists(path)) need(checkpoint::read(path.string())==bytes,"Immutable motor-v2 payload differs: "+path.string());else checkpoint::write_atomic(path.string(),bytes);
}
template<class T> std::vector<T> download(const T* pointer,size_t count,cudaStream_t stream) {
    std::vector<T> values(count);CUDA_CHECK_THROW(cudaMemcpyAsync(values.data(),pointer,count*sizeof(T),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));return values;
}
template<class T> void upload(device_ptr<T>& pointer,const std::vector<T>& values,cudaStream_t stream) {
    need(pointer.size()==values.size(),"Motor-v2 upload dimensions differ");CUDA_CHECK_THROW(cudaMemcpyAsync(pointer.get(),values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
struct Inputs {
    rl::BoulderSource source;pm::RegulationGate history;mc::ContactGate contact;
    std::vector<uint8_t> source_bytes,history_bytes,contact_bytes,reference_bytes;
};
rl::MotorV2Policy initial_policy(const Inputs& in,mv::Interface mode) {
    rl::MotorV2Policy policy;policy.interface_type=mode;policy.gains=in.history.gains;policy.source_fingerprint=in.source.fingerprint;
    policy.environment_fingerprint=rl::motor_v2_environment_fingerprint(mode,policy.gains);policy.actor=rl::initialize_motor_v2(in.source.motor);
    policy.historical_regulation_gate=in.history_bytes;policy.contact_gate=in.contact_bytes;return policy;
}
std::string manifest(const Inputs& in) {
    std::ostringstream out;out<<std::setprecision(17)<<"{\n\"protocol\":\"motor-v2\",\n\"physics_version\":2,\n\"physics_hash\":\""<<mv::physics_fingerprint()<<"\",\n\"contact_physics_hash\":\""<<mc::physics_fingerprint()
        <<"\",\n\"source_fingerprint\":\""<<in.source.fingerprint<<"\",\n\"historical_servo_gate_hash\":\""<<rl::motor_v2_hash(in.history_bytes)<<"\",\n\"contact_gate_hash\":\""<<rl::motor_v2_hash(in.contact_bytes)
        <<"\",\n\"reference_capture_hash\":\""<<rl::motor_v2_hash(in.reference_bytes)<<"\",\n\"evaluation_rules_hash\":\""<<mb::rules_fingerprint()<<"\",\n"
        <<"\"seeds\":{\"head\":"<<rl::motor_v2_head_seed<<",\"training\":"<<rl::motor_v2_seed<<",\"final\":"<<rl::motor_v2_final_seed<<"},\n"
        <<"\"arms\":[\"torque\",\"joint_target\"],\n\"updates_per_arm\":64,\n\"worlds\":512,\n\"rollout\":64,\n\"maximum_steps_per_arm\":2097152,\n\"maximum_training_steps_total\":4194304,\n"
        <<"\"architecture\":\"one59-32-32-4 motor actor; two universal joints with two controlled axes each; goal chooser excluded; PPO critics are training auxiliaries\",\n"
        <<"\"initialization\":\"both arms same fresh actor/head/critic/logstd/zeroAdam and worlds; compatible original accepted55 body features only; historical six-update actor never loaded\",\n"
        <<"\"actuation\":\"raw4D Gaussian z stored/logged unchanged; torque clamp(z); joint targets limit*clamp(z), deterministic5ms SI PD with retained historical gains and same10uNm cap; likelihood never evaluated on resulting torque/clipped target\",\n"
        <<"\"observation\":\"59: original body9..32; previous_action29..32=actual final5ms torque;4..6 bodylocal current/.02;7 signed floor gap/.072;8 contact;36..53 reservedzero,54 geometry unavailable0; no obstacle perception claim\",\n"
        <<"\"cost\":\"reward=-.05*(||position_error/.072||^2+.25||terminal_velocity_error/.02||^2+.1*(1-heading_dot)+.02||body_omega/.5||^2+.002MEAN40(actual_normalized_torque^2)); positive actuator work recorded separately\",\n"
        <<"\"task\":\"seven matched lessons move/turn/brake/slow_arrival/unsupportedcalmhold/unsupportedsteady5mm_s_currenthold/bottomrest; persistent destination/zero terminal velocity/level heading;600 true terminal remainingtimeobserved\",\n"
        <<"\"contact\":\"versioned directional head/fin/end support; allowed floor contact/overlap<=.5mm; excessive penetration/escape/numerical invalidity rejected, not every negative clearance; fixed beforelearning\",\n"
        <<"\"readiness_budget\":\"exactly32 physical controls:8 fixed safe starts x2interfaces xCPU/CUDA one50ms each; pure512Gaussian and fullfresh512snapshot tests addno dynamics/optimizer steps\",\n"
        <<"\"readiness_tolerances\":{\"state_abs\":0.0002,\"state_relative\":0.002,\"observation_abs\":0.00002,\"observation_relative\":0.00002,\"SI_torque_abs_Nm\":1e-9,\"SI_torque_relative\":0.002,\"work_abs_J\":1e-10,\"work_relative\":0.002,\"effort_abs\":0.000002,\"raw_logp_abs\":0.00002,\"raw_logp_relative\":0.00002},\n"
        <<"\"evaluation\":\"same64 withheld complete initial states beforelearning,600controlsmaxeacharm, terminal actor once evenif honestlypartial; strata8move8turn8brake8arrival12calmhold12currenthold8rest;10s holding separate; no midrun selection; no new capture dynamics\",\n"
        <<"\"failure\":\"finite physical failure stops affectedarm beforeGAE/Adam; other arm remains independent; shared physics/numerical/data/evaluation defect globallystops; no retries/sweeps/gainchanges/extensions/adoption\",\n\"gains_kp\":[";
    for(int j=0;j<4;++j) {if(j) out<<',';out<<in.history.gains.kp[j];}out<<"],\n\"gains_kd\":[";for(int j=0;j<4;++j) {if(j) out<<',';out<<in.history.gains.kd[j];}
    out<<"],\n\"protocol_values\":[";const auto values=rl::motor_v2_training_protocol();for(size_t i=0;i<values.size();++i) {if(i) out<<',';out<<values[i];}out<<"]\n}\n";return out.str();
}
std::vector<mb::Fixture> final_fixtures(const Inputs& in,const fs::path& root) {
    const auto path=root/"sources"/"final-fixtures.bin";
    if(fs::exists(path)) {const auto bytes=checkpoint::read(path.string());checkpoint::Reader saved{bytes};
        need(saved.value<uint32_t>()==0x4D563246u&&saved.value<uint32_t>()==rl::motor_v2_protocol_version&&saved.value<uint32_t>()==sizeof(mb::Fixture)&&saved.value<uint64_t>()==in.source.fingerprint&&saved.value<uint64_t>()==mv::physics_fingerprint()&&saved.value<uint64_t>()==mb::rules_fingerprint(),"Saved motor-v2 withheld fixture binding differs");
        auto result=saved.array<mb::Fixture>(64);saved.finish();need(result.size()==64,"Saved motor-v2 withheld episode count differs");for(int i=0;i<64;++i) need(result[i].id==i&&result[i].seed==rl::motor_v2_final_seed&&result[i].steps==600&&mv::finite(result[i].initial),"Saved motor-v2 fixture identity differs");return result;
    }
    const auto generated=mb::make_fixtures(rl::motor_v2_final_seed,in.reference_bytes);std::vector<mb::Fixture> result(generated.begin(),generated.end());
    checkpoint::Writer out;out.value(uint32_t{0x4D563246u});out.value(rl::motor_v2_protocol_version);out.value(uint32_t{sizeof(mb::Fixture)});out.value(in.source.fingerprint);out.value(mv::physics_fingerprint());out.value(mb::rules_fingerprint());out.array(result);
    immutable_payload(path,out.data);return result;
}
__global__ void recorded_observations(const mv::State* states,float* out,int count) {const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count) mv::observe(states[i],out+i*59);}
struct DataFrame {int backend=0,mode=0,id=0;mv::State before{},after{};float latent[4]{};command_motor::Result result{};};
std::vector<uint8_t> canonical_worlds(const std::vector<mv::State>& states) {
    checkpoint::Writer out;auto body=[&](const deep::BodyState& b) {out.value(b.position);out.value(b.orientation);for(float value:b.angle) out.value(value);for(float value:b.velocity) out.value(value);out.value(b.work);out.value(b.travel);};
    auto task=[&](const command_motor::State& s) {body(s.body);out.value(s.request.position);out.value(s.request.velocity);out.value(s.request.heading);for(float value:s.previous_action) out.value(value);out.value(s.seed);out.value(s.environment);out.value(s.episode);out.value(s.steps);};
    for(const auto& state:states) {task(state.task);out.value(state.water_velocity);out.value(state.lesson);out.value(state.integrated_substeps);out.value(state.executed_substeps);for(float value:state.held_target) out.value(value);for(float value:state.actual_torques) out.value(value);for(float value:state.unclamped_torques) out.value(value);out.value(state.mean_squared_torque);out.value(state.first_unsafe_substep);out.value(state.unsafe_substeps);task(state.first_unsafe_task);
        out.value(state.substep_min_floor_gap);out.value(state.substep_max_floor_gap);out.value(state.substep_min_other_gap);out.value(state.floor_contact_samples);out.value(state.substep_max_floor_force);out.value(state.substep_max_head_speed);out.value(state.substep_max_com_speed);out.value(state.substep_max_omega);out.value(state.substep_max_goal_error);out.value(state.substep_max_heading_error);}
    return out.data; // excludes non-semantic struct padding, includes every field
}
void readiness(const Inputs& in,const fs::path& root) {
    in.contact.require_passed();need(!fs::exists(root/"readiness-started.json"),"This fixed motor-v2 readiness has already started; no automatic repeat");
    immutable_text(root/"readiness-started.json","{\"physical_controls_cap\":32,\"ppo_updates\":0}\n");
    rl::SwimmerStream stream;const auto s=stream.value;std::vector<DataFrame> frames;std::vector<mv::State> observations;
    double max_state_error=0,max_obs_error=0,max_effort_error=0,max_reward_error=0,max_logp_error=0;int outside=0;
    std::vector<mv::State> base(8);for(int i=0;i<8;++i) {
        base[i]=mv::initial(rl::motor_v2_seed,static_cast<uint32_t>(i));need(mv::finite(base[i]),"Readiness initial state nonfinite");const auto contact=mc::inspect(base[i].task.body);
        need(!contact.invalid&&!contact.excessive&&!contact.escape,"Readiness start physically unsafe");
    }
    std::vector<float> latent(8*4);for(int i=0;i<8;++i) for(int j=0;j<4;++j) latent[i*4+j]=static_cast<float>(((i*7+j*3)%17)-8)*.2f;
    auto compare=[&](float a,float b,const char* label) {need(std::isfinite(a)&&std::isfinite(b),std::string("Readiness nonfinite ")+label);const double error=fabs(static_cast<double>(a)-b);max_state_error=std::max(max_state_error,error);need(error<=2e-4+2e-3*fabs(a),std::string("Readiness CPU/CUDA ")+label+" mismatch");};
    for(auto mode:{mv::Interface::Torque,mv::Interface::JointTarget}) {
        std::vector<mv::State> cpu=base;std::vector<command_motor::Result> cpu_result(8);
        for(int i=0;i<8;++i) {cpu_result[i]=mv::advance(cpu[i],latent.data()+i*4,mode,in.history.gains);need(!cpu_result[i].invalid&&!cpu_result[i].unsafe,"Fixed CPU motor-v2 data control failed");}
        mv::Batch worlds(8,rl::motor_v2_seed);worlds.configure(mode,in.history.gains,s);worlds.set_states(base,s);device_ptr<float> input(latent.size()),reward(8),done(8),truncated(8),final_obs(8*59);upload(input,latent,s);
        worlds.step(input.get(),{reward.get(),done.get(),truncated.get(),final_obs.get()},s,false);auto gpu=worlds.host_states(s);const auto rewards=download(reward.get(),8,s),terminals=download(done.get(),8,s),truncations=download(truncated.get(),8,s);
        need(worlds.invalid_count(s)==0&&worlds.unsafe_count(s)==0,"Fixed CUDA motor-v2 data control failed");
        for(int i=0;i<8;++i) {
            for(int backend=0;backend<2;++backend) {DataFrame frame;frame.backend=backend;frame.mode=static_cast<int>(mode);frame.id=i;frame.before=base[i];frame.after=backend==0?cpu[i]:gpu[i];for(int j=0;j<4;++j) frame.latent[j]=latent[i*4+j];frame.result=backend==0?cpu_result[i]:command_motor::Result{rewards[i],false,false,false};frames.push_back(frame);observations.push_back(frame.before);observations.push_back(frame.after);}
            const auto& a=cpu[i];const auto& b=gpu[i];compare(a.task.body.position.x,b.task.body.position.x,"positionX");compare(a.task.body.position.y,b.task.body.position.y,"positionY");compare(a.task.body.position.z,b.task.body.position.z,"positionZ");
            compare(a.task.body.orientation.w,b.task.body.orientation.w,"orientationW");compare(a.task.body.orientation.x,b.task.body.orientation.x,"orientationX");compare(a.task.body.orientation.y,b.task.body.orientation.y,"orientationY");compare(a.task.body.orientation.z,b.task.body.orientation.z,"orientationZ");
            for(int j=0;j<10;++j) compare(a.task.body.velocity[j],b.task.body.velocity[j],"velocity");for(int j=0;j<4;++j) {compare(a.task.body.angle[j],b.task.body.angle[j],"angle");compare(a.held_target[j],b.held_target[j],"heldtarget");need(b.task.previous_action[j]==b.actual_torques[36+j],"Previous action is not final actual substep torque");if(mode==mv::Interface::JointTarget) need(b.held_target[j]==deep::limit*deep::clamp(latent[i*4+j],-1,1),"Held joint target transformation differs");}
            for(int j=0;j<40;++j) {compare(a.actual_torques[j],b.actual_torques[j],"actualtorque");const double requested_error=fabs(static_cast<double>(a.unclamped_torques[j])-b.unclamped_torques[j]);need(requested_error<=1e-9+2e-3*fabs(a.unclamped_torques[j]),"Requested SI torque parity mismatch");
                need(fabsf(deep::clamp(b.unclamped_torques[j]/deep::torque,-1,1)-b.actual_torques[j])<=3e-6f,"Applied torque differs from requested SI torque clamp");if(mode==mv::Interface::Torque) need(b.actual_torques[j]==deep::clamp(latent[i*4+j%4],-1,1),"Direct torque latent transform differs");}
            double effort=0;for(float torque:b.actual_torques) effort+=torque*torque/40.;max_effort_error=std::max(max_effort_error,fabs(effort-b.mean_squared_torque));need(fabs(effort-b.mean_squared_torque)<=2e-6,"Actual effort is not mean40 torque squared");
            need(fabs(static_cast<double>(a.task.body.work)-b.task.body.work)<=1e-10+2e-3*fabs(a.task.body.work),"Positive actuator work parity mismatch");max_reward_error=std::max(max_reward_error,fabs(static_cast<double>(cpu_result[i].reward)-rewards[i]));compare(cpu_result[i].reward,rewards[i],"reward");need(a.task.steps==1&&b.task.steps==1&&a.executed_substeps==10&&b.executed_substeps==10&&a.integrated_substeps==10&&b.integrated_substeps==10,"Readiness did not consume exactly one complete control");need(terminals[i]==0&&truncations[i]==0,"One-tick readiness incorrectly terminated or truncated");
        }
    }
    need(frames.size()==32,"Motor-v2 readiness physical control budget differs");checkpoint::Writer recorded;recorded.value(uint32_t{0x4D563252u});recorded.value(rl::motor_v2_protocol_version);recorded.value(uint32_t{sizeof(DataFrame)});recorded.value(mv::physics_fingerprint());recorded.array(frames);immutable_payload(root/"readiness-controls.bin",recorded.data);
    std::vector<float> cpu_obs(observations.size()*59);for(size_t i=0;i<observations.size();++i) mv::observe(observations[i],cpu_obs.data()+i*59);
    device_ptr<mv::State> d_states(observations.size());device_ptr<float> d_obs(cpu_obs.size());upload(d_states,observations,s);recorded_observations<<<(static_cast<int>(observations.size())+63)/64,64,0,s>>>(d_states.get(),d_obs.get(),static_cast<int>(observations.size()));CUDA_CHECK_THROW(cudaGetLastError());
    const auto gpu_obs=download(d_obs.get(),cpu_obs.size(),s);for(size_t i=0;i<cpu_obs.size();++i) {const double error=fabs(static_cast<double>(cpu_obs[i])-gpu_obs[i]);max_obs_error=std::max(max_obs_error,error);need(error<=2e-5+2e-5*fabs(cpu_obs[i]),"Motor-v2 CPU/CUDA observation mismatch");}
    std::vector<float> inputs(512*59);for(int i=0;i<512;++i) {const size_t row=rng::bits(rl::motor_v2_seed,0x52454144u,static_cast<uint32_t>(i))%observations.size();std::copy_n(cpu_obs.data()+row*59,59,inputs.data()+i*59);}
    device_ptr<float> d_inputs(inputs.size()),sigma(4),actions(512*4),logp(512),again(512*4),again_logp(512);upload(d_inputs,inputs,s);upload(sigma,std::vector<float>(4,-.5f),s);
    nn::GpuMlp actor(rl::motor_v2_specification(),512,rl::motor_v2_seed);const auto weights=rl::initialize_motor_v2(in.source.motor);actor.set_params(weights);CUDA_CHECK_THROW(cudaDeviceSynchronize());actor.forward(d_inputs.get(),512,s);
    const auto means=download(actor.output(),512*4,s);rl::sample_gaussian(actor.output(),sigma.get(),512,4,rl::motor_v2_seed,0,actions.get(),logp.get(),s);const auto sampled=download(actions.get(),512*4,s),old_logp=download(logp.get(),512,s);
    rl::sample_gaussian(actor.output(),sigma.get(),512,4,rl::motor_v2_seed,0,again.get(),again_logp.get(),s);need(download(again.get(),512*4,s)==sampled&&download(again_logp.get(),512,s)==old_logp,"Raw Gaussian action/logp replay differs");
    for(int i=0;i<512;++i) {double mu[4],a[4],sd[4]={-.5,-.5,-.5,-.5};for(int j=0;j<4;++j) {mu[j]=means[i*4+j];a[j]=sampled[i*4+j];outside+=fabs(a[j])>1?1:0;}const double expected=rl::gaussian_logp(mu,sd,a,4);max_logp_error=std::max(max_logp_error,fabs(expected-old_logp[i]));need(fabs(expected-old_logp[i])<=2e-5+2e-5*fabs(expected),"Raw Gaussian likelihood changed after actuator clipping");}
    need(outside>0,"Readiness did not exercise raw latents outside actuator bounds");device_ptr<float> zero(512),one(512),adv(2),dmean(512*4),dstd(512*4),dvalue(512),rows(512*rl::kGaussianStats),stats(rl::kGaussianStats);
    upload(zero,std::vector<float>(512,0),s);upload(one,std::vector<float>(512,1),s);upload(adv,std::vector<float>{0,1},s);CUDA_CHECK_THROW(cudaMemsetAsync(stats.get(),0,stats.size_bytes(),s));
    rl::gaussian_ppo_loss_grad(actor.output(),sigma.get(),zero.get(),512,4,actions.get(),logp.get(),one.get(),zero.get(),adv.get(),{.2f,.002f,.5f},dmean.get(),dstd.get(),dvalue.get(),rows.get(),stats.get(),s);
    const auto terms=download(stats.get(),rl::kGaussianStats,s);need(fabsf(terms[3])<=1e-6f&&terms[4]==0,"Unchanged motor-v2 policy has nonzero KL/clipfraction");need(download(actions.get(),512*4,s)==sampled&&download(logp.get(),512,s)==old_logp&&actor.get_params()==weights,"Likelihood check mutated stored raw data/actor");
    std::vector<float> initial_actor,initial_critic,initial_sigma;std::vector<uint8_t> initial_optimizers;std::vector<mv::State> initial_worlds;
    for(auto mode:{mv::Interface::Torque,mv::Interface::JointTarget}) {
        rl::MotorV2Trainer trainer(in.source.motor,in.source.fingerprint,mode,in.history_bytes,in.contact_bytes);const auto snapshot=rl::motor_v2_snapshot(trainer);const auto a=trainer.actor.get_params(),c=trainer.critic.get_params(),d=download(trainer.log_std.get(),4,trainer.stream.value);const auto worlds=trainer.env.host_states(trainer.stream.value);
        checkpoint::Writer optim;trainer.actor_opt.save(optim);trainer.critic_opt.save(optim);trainer.std_opt.save(optim);
        if(initial_actor.empty()) {initial_actor=a;initial_critic=c;initial_sigma=d;initial_optimizers=optim.data;initial_worlds=worlds;}else {need(a==initial_actor&&c==initial_critic&&d==initial_sigma&&optim.data==initial_optimizers,"V2 arms do not share fresh actor/critic/std/Adam initialization");need(canonical_worlds(worlds)==canonical_worlds(initial_worlds),"V2 initial complete world fields differ between arms");}
        immutable_payload(root/"run"/rl::motor_v2_arm(mode)/"initial-training.bin",snapshot);rl::MotorV2Trainer restored(in.source.motor,in.source.fingerprint,mode,in.history_bytes,in.contact_bytes);rl::restore_motor_v2(restored,snapshot);need(rl::motor_v2_snapshot(restored)==snapshot,"V2 complete zero-update snapshot replay differs");
        auto stressed=worlds;for(size_t i=0;i<stressed.size();++i) {float mean=0;for(int j=0;j<40;++j) {const float x=static_cast<float>((static_cast<int>(i)+j)%9-4)*.1f;stressed[i].actual_torques[j]=x;stressed[i].unclamped_torques[j]=x*deep::torque;mean+=x*x/40.f;}stressed[i].mean_squared_torque=mean;for(int j=0;j<4;++j) {stressed[i].held_target[j]=.1f*(j+1);stressed[i].task.previous_action[j]=stressed[i].actual_torques[36+j];}}
        trainer.env.set_states(stressed,trainer.stream.value);const auto stress_snapshot=rl::motor_v2_snapshot(trainer);rl::restore_motor_v2(restored,stress_snapshot);need(rl::motor_v2_snapshot(restored)==stress_snapshot,"Held-target/actual-action metadata snapshot replay differs");
        need(trainer.update_id==0&&trainer.actor_opt.steps()==0&&trainer.critic_opt.steps()==0&&trainer.std_opt.steps()==0,"Readiness changed PPO/Adam counters");
    }
    rl::MotorV2Readiness ready;ready.source_fingerprint=in.source.fingerprint;ready.physics_fingerprint=mv::physics_fingerprint();ready.historical_gate_hash=rl::motor_v2_hash(in.history_bytes);ready.contact_gate_hash=rl::motor_v2_hash(in.contact_bytes);ready.contact_frames_hash=rl::motor_v2_hash(recorded.data);
    ready.torque_task_hash=rl::motor_v2_environment_fingerprint(mv::Interface::Torque,in.history.gains);ready.target_task_hash=rl::motor_v2_environment_fingerprint(mv::Interface::JointTarget,in.history.gains);ready.recorded_frames=32;ready.observations=ready.actual_torque=ready.latent_replay=ready.complete_resume=true;immutable_payload(root/"readiness-gate.bin",ready.serialize());
    std::ostringstream report;report<<std::setprecision(17)<<"{\"passed\":true,\"physical_controls\":32,\"ppo_updates\":0,\"state_max_abs\":"<<max_state_error<<",\"observation_max_abs\":"<<max_obs_error<<",\"actual_effort_max_abs\":"<<max_effort_error<<",\"reward_max_abs\":"<<max_reward_error<<",\"raw_logp_max_abs\":"<<max_logp_error<<",\"unclipped_latent_components\":"<<outside<<",\"noop_kl_sum\":"<<terms[3]<<",\"fresh512_complete_snapshot_exact\":true,\"adopted\":false}\n";immutable_text(root/"readiness-summary.json",report.str());std::cout<<"MOTOR_V2_READINESS "<<report.str();
}

std::vector<uint8_t> parameters_and_optimizers(const rl::MotorV2Trainer& trainer) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(trainer.stream.value));checkpoint::Writer out;out.array(trainer.actor.get_params());out.array(trainer.critic.get_params());out.array(download(trainer.log_std.get(),4,trainer.stream.value));
    trainer.actor_opt.save(out);trainer.critic_opt.save(out);trainer.std_opt.save(out);return out.data;
}
void save_training(const rl::MotorV2Trainer& trainer,const Inputs& in,const fs::path& path) {
    const auto bytes=rl::motor_v2_snapshot(trainer);checkpoint::write_atomic(path.string(),bytes);need(checkpoint::read(path.string())==bytes,"Motor-v2 fullsave checksum/readback differs");
    rl::MotorV2Trainer clone(in.source.motor,in.source.fingerprint,trainer.interface_type,in.history_bytes,in.contact_bytes);rl::restore_motor_v2(clone,bytes);need(rl::motor_v2_snapshot(clone)==bytes,"Motor-v2 complete actor/critic/std/Adam/world/held-target continuation differs");
}
struct ArmResult {int updates=0,attempted_updates=0;bool physical_stop=false;uint64_t integrated_substeps=0,rollout_slots=0,frozen_slots=0;};
uint64_t substeps(const std::vector<mv::State>& states) {uint64_t value=0;for(const auto& state:states) value+=state.integrated_substeps;return value;}
uint64_t clock(const command_motor::State& state) {return static_cast<uint64_t>(state.episode)*600u+static_cast<uint64_t>(state.steps);}
ArmResult train_arm(const Inputs& in,mv::Interface mode,const fs::path& root) {
    const auto directory=root/"run"/rl::motor_v2_arm(mode);need(!fs::exists(directory/"arm-stop.json"),"Motor-v2 arm already stopped; no automatic retry");
    const auto initial=checkpoint::read((directory/"initial-training.bin").string());rl::MotorV2Trainer trainer(in.source.motor,in.source.fingerprint,mode,in.history_bytes,in.contact_bytes);rl::restore_motor_v2(trainer,initial);
    need(trainer.update_id==0&&trainer.actor_opt.steps()==0&&trainer.critic_opt.steps()==0&&trainer.std_opt.steps()==0,"Each v2 arm must start from matched fresh zero-update history");save_training(trainer,in,directory/"latest-training.bin");
    std::ofstream log(directory/"training.csv");need(bool(log),"Cannot create motor-v2 training history");log<<std::setprecision(17)<<"completed_update,attempted_updates,integrated_substeps,physical_controls,actor_adam,critic_adam,std_adam,elapsed_s,mean_reward,actor_gradient_norm,logstd0,logstd1,logstd2,logstd3\n";
    ArmResult result;
    for(int attempt=0;attempt<rl::motor_v2_updates;++attempt) {
        const auto before=rl::motor_v2_snapshot(trainer),before_parameters=parameters_and_optimizers(trainer);const auto before_worlds=trainer.env.host_states(trainer.stream.value);const int old_update=trainer.update_id,old_adam=trainer.actor_opt.steps();
        result.attempted_updates=attempt+1;result.rollout_slots+=static_cast<uint64_t>(trainer.batch);double elapsed=0;
        try {elapsed=trainer.update_seconds();}
        catch(const rl::MotorV2PhysicalFailure& error) {
            const auto after_parameters=parameters_and_optimizers(trainer);need(after_parameters==before_parameters&&trainer.actor_opt.steps()==old_adam,"Rejected physical rollout changed learner/Adam before guard");
            const auto after_worlds=trainer.env.host_states(trainer.stream.value);const auto first=trainer.env.unsafe_states(trainer.stream.value);
            checkpoint::Writer failure;failure.value(uint32_t{0x4D563258u});failure.value(rl::motor_v2_protocol_version);failure.value(uint32_t{sizeof(mv::State)});failure.value(mode);failure.value(old_update);failure.value(old_adam);failure.value(trainer.actor_opt.steps());
            failure.array(before);failure.array(before_parameters);failure.array(after_parameters);failure.value(trainer.env.invalid_count(trainer.stream.value));failure.value(trainer.env.unsafe_count(trainer.stream.value));failure.array(first);failure.array(after_worlds);
            failure.array(download(trainer.obs.get(),trainer.obs.size(),trainer.stream.value));failure.array(download(trainer.actions.get(),trainer.actions.size(),trainer.stream.value));failure.array(download(trainer.logp.get(),trainer.logp.size(),trainer.stream.value));failure.array(download(trainer.rewards.get(),trainer.rewards.size(),trainer.stream.value));
            immutable_payload(directory/"rejected-rollout.bin",failure.data);result.integrated_substeps=substeps(after_worlds);result.physical_stop=true;result.updates=old_update;
            uint64_t integrated_controls=0;for(size_t i=0;i<after_worlds.size();++i) {need(clock(after_worlds[i].task)>=clock(before_worlds[i].task),"Failed motor-v2 clock reversed");integrated_controls+=clock(after_worlds[i].task)-clock(before_worlds[i].task);}
            result.frozen_slots=trainer.batch-integrated_controls;rl::restore_motor_v2(trainer,before);need(rl::motor_v2_snapshot(trainer)==before,"Restored last-good motor-v2 snapshot differs");save_training(trainer,in,directory/"latest-training.bin");checkpoint::write_atomic((directory/"candidate-policy.bin").string(),rl::motor_v2_policy(trainer).serialize());
            std::ostringstream stop;stop<<"{\"arm\":\""<<rl::motor_v2_arm(mode)<<"\",\"physical_safety_stop\":true,\"shared_defect\":false,\"last_completed_update\":"<<old_update<<",\"attempted_update\":"<<attempt+1<<",\"adam_before\":"<<old_adam<<",\"adam_after\":"<<trainer.actor_opt.steps()<<",\"parameters_and_moments_unchanged\":true,\"integrated_substeps\":"<<result.integrated_substeps<<",\"frozen_rollout_slots\":"<<result.frozen_slots<<",\"other_arm_may_continue\":true,\"automatic_retry\":false,\"adopted\":false}\n";
            immutable_text(directory/"arm-stop.json",stop.str());std::cerr<<"MOTOR_V2_ARM_STOP "<<stop.str()<<"reason="<<error.what()<<'\n';return result;
        }
        result.updates=trainer.update_id;result.integrated_substeps=substeps(trainer.env.host_states(trainer.stream.value));
        need(trainer.update_id==attempt+1&&trainer.actor_opt.steps()==trainer.update_id*32&&trainer.critic_opt.steps()==trainer.update_id*32&&trainer.std_opt.steps()==trainer.update_id*32,"Motor-v2 completed update/optimizer counters differ");
        need(result.integrated_substeps==static_cast<uint64_t>(trainer.update_id)*512u*64u*10u,"Safe motor-v2 completed rollout did not integrate exact registered controls");
        const auto reward=download(trainer.rewards.get(),trainer.batch,trainer.stream.value),sigma=download(trainer.log_std.get(),4,trainer.stream.value);double total=0;for(float value:reward) {need(std::isfinite(value),"Nonfinite reward after successful validation");total+=value;}
        log<<trainer.update_id<<','<<attempt+1<<','<<result.integrated_substeps<<','<<result.integrated_substeps/10<<','<<trainer.actor_opt.steps()<<','<<trainer.critic_opt.steps()<<','<<trainer.std_opt.steps()<<','<<elapsed<<','<<total/reward.size()<<','<<trainer.actor_opt.last_grad_norm();for(float value:sigma) log<<','<<value;log<<'\n';log.flush();need(bool(log),"Cannot flush motor-v2 training evidence");
        save_training(trainer,in,directory/"latest-training.bin");checkpoint::write_atomic((directory/"candidate-policy.bin").string(),rl::motor_v2_policy(trainer).serialize());std::cout<<"MOTOR_V2_TRAIN arm="<<rl::motor_v2_arm(mode)<<" update="<<trainer.update_id<<"/64 integrated_substeps="<<result.integrated_substeps<<" adam="<<trainer.actor_opt.steps()<<'\n';
    }
    need(result.updates==64,"Motor-v2 arm stopped without a physical-stop record");return result;
}
struct Evaluation {int episodes=64,qualified=0,safe=0,motion=0,settled=0,held=0,passed=0;int stratum_count[7]{},stratum_safe[7]{},stratum_held[7]{},stratum_passed[7]{};uint64_t integrated_substeps=0,control_calls=0;};
Evaluation evaluate(const rl::MotorV2Policy& policy,const std::vector<mb::Fixture>& fixtures,const fs::path& directory) {
    need(fixtures.size()==64,"Motor-v2 terminal evaluation must contain exactly64 withheld cases");need(!fs::exists(directory/"evaluation-started.json"),"Motor-v2 terminal suite already started; no automatic repeat");
    std::ostringstream start;start<<"{\"saved_update\":"<<policy.update<<",\"completed_budget\":"<<(policy.completed_budget?"true":"false")<<",\"episodes\":64,\"controls_cap_each\":600,\"training_updates_during_evaluation\":0}\n";immutable_text(directory/"evaluation-started.json",start.str());
    rl::SwimmerStream stream;const auto s=stream.value;mv::Batch worlds(64,rl::motor_v2_final_seed);worlds.configure(policy.interface_type,policy.gains,s);std::vector<mv::State> states(64);for(int i=0;i<64;++i) states[i]=fixtures[i].initial;worlds.set_states(states,s);
    nn::CpuMlp<float> actor(rl::motor_v2_specification());actor.params()=policy.actor;std::vector<mb::Progress> progress(64);std::vector<float> latent(64*4);std::vector<uint8_t> active(64);std::vector<deep::Physics> physical(64);device_ptr<float> d_latent(latent.size());device_ptr<uint8_t> d_active(64);device_ptr<deep::Physics> d_physical(64);
    std::ofstream trace(directory/"final-trace.csv");need(bool(trace),"Cannot create motor-v2 frozen trace");trace<<std::setprecision(17)<<"case,kind,control_tick,integrated_substeps,head_x_model,head_y_model,head_z_model,goal_error_m,heading_error_rad,head_m_s,com_m_s,body_omega_rad_s,point_m_s,max_joint_rad_s,floor_gap_m,other_gap_m,floor_contact,excessive,escape,torque_rms_uNm,positive_work_J,interval_floor_samples,interval_floor_force_N,first_unsafe_substep";
    for(int j=0;j<4;++j) trace<<",driving_raw"<<j;for(int j=0;j<4;++j) trace<<",held_target_rad"<<j;for(int j=0;j<4;++j) trace<<",previous_actual_normalized"<<j;for(int j=0;j<40;++j) trace<<",actual_normalized_substep"<<j;trace<<'\n';
    auto predict=[&]() {for(int i=0;i<64;++i) {float obs[59];mv::observe(states[i],obs);actor.forward(obs,1);for(int j=0;j<4;++j) {latent[i*4+j]=actor.output()[j];need(std::isfinite(latent[i*4+j]),"Frozen motor-v2 inference generated nonfinite raw latent");}}};
    auto record=[&](int i) {const auto sample=mb::measure(states[i]);mb::assess(progress[i],states[i],sample,fixtures[i]);const auto& state=states[i];trace<<i<<','<<mb::name(fixtures[i].kind)<<','<<state.task.steps<<','<<state.integrated_substeps<<','<<state.task.body.position.x<<','<<state.task.body.position.y<<','<<state.task.body.position.z
        <<','<<sample.goal_error<<','<<sample.heading_error<<','<<sample.head_speed<<','<<sample.com_speed<<','<<sample.angular_speed<<','<<sample.point_speed<<','<<sample.joint_rate<<','<<sample.floor_gap<<','<<sample.other_gap<<','<<sample.floor_contact<<','<<sample.excessive<<','<<sample.escape<<','<<sample.torque_rms_uNm<<','<<sample.positive_work_J<<','<<state.floor_contact_samples<<','<<state.substep_max_floor_force<<','<<state.first_unsafe_substep;
        for(int j=0;j<4;++j) trace<<','<<latent[i*4+j];for(float value:state.held_target) trace<<','<<value;for(float value:state.task.previous_action) trace<<','<<value;for(float value:state.actual_torques) trace<<','<<value;trace<<'\n';};
    predict();for(int i=0;i<64;++i) record(i);Evaluation total;
    for(int tick=0;tick<600;++tick) {
        bool any=false;for(int i=0;i<64;++i) {active[i]=static_cast<uint8_t>(fixtures[i].qualified&&tick<fixtures[i].steps&&progress[i].finite&&!progress[i].unsafe);any=any||active[i];physical[i]=deep::Physics{};}
        if(!any) break;predict();upload(d_latent,latent,s);upload(d_active,active,s);upload(d_physical,physical,s);worlds.step_physics(d_latent.get(),d_physical.get(),d_active.get(),s);states=worlds.host_states(s);
        need(worlds.invalid_count(s)==0,"Shared nonfinite dynamics during terminal motor-v2 evaluation");for(int i=0;i<64;++i) if(active[i]) {++total.control_calls;record(i);}
    }
    total.integrated_substeps=substeps(states);need(total.control_calls<=64u*600u&&total.integrated_substeps<=64u*600u*10u,"Motor-v2 terminal physical budget exceeded");
    std::ofstream cases(directory/"final-cases.csv");need(bool(cases),"Cannot create motor-v2 per-task outcomes");cases<<std::setprecision(17)<<"case,kind,source_capture,entry_m_s,qualified,safety,motion,settled,held10s,passed,half_s,reach_s,heading_s,slow_arrival_s,stop_s,stop_distance_m,hold_seconds,geometric_hold_seconds,support_force_fraction,hold_start_s,final_error_m,path_m,path_efficiency,min_floor_gap_m,min_other_gap_m,max_head_m_s,max_com_m_s,max_body_omega,max_point_m_s,max_joint_rad_s,max_torque_rms_uNm,torque_squared_integral,positive_work_J,floor_contact_intervals,unsupported_contact_intervals,first_safety_tick,integrated_substeps\n";
    for(int i=0;i<64;++i) {const auto outcome=mb::finish(progress[i],fixtures[i]);const auto& p=progress[i];const int kind=static_cast<int>(fixtures[i].kind);++total.stratum_count[kind];total.stratum_safe[kind]+=outcome.safety;total.stratum_held[kind]+=outcome.hold;total.stratum_passed[kind]+=outcome.passed;total.qualified+=outcome.qualified;total.safe+=outcome.safety;total.motion+=outcome.motion;total.settled+=outcome.settle;total.held+=outcome.hold;total.passed+=outcome.passed;
        cases<<i<<','<<mb::name(fixtures[i].kind)<<','<<fixtures[i].source_capture<<','<<fixtures[i].actual_entry_speed<<','<<outcome.qualified<<','<<outcome.safety<<','<<outcome.motion<<','<<outcome.settle<<','<<outcome.hold<<','<<outcome.passed<<','<<outcome.half_time<<','<<outcome.reach_time<<','<<outcome.heading_time<<','<<outcome.slow_arrival_time<<','<<outcome.stop_time<<','<<outcome.stop_distance<<','<<outcome.hold_seconds<<','<<p.geometric_hold_seconds<<','<<outcome.support_force_fraction<<','<<outcome.settle_time<<','<<p.final_error<<','<<p.path<<','<<mb::path_efficiency(p)<<','<<p.minimum_floor_gap<<','<<p.minimum_other_gap<<','<<p.maximum_head_speed<<','<<p.maximum_com_speed<<','<<p.maximum_angular_speed<<','<<p.maximum_point_speed<<','<<p.maximum_joint_rate<<','<<p.maximum_torque_rms_uNm<<','<<p.torque_squared_integral<<','<<p.positive_work_J<<','<<p.floor_contact_intervals<<','<<p.unsupported_contact_intervals<<','<<p.first_failure_step<<','<<states[i].integrated_substeps<<'\n';}
    return total;
}
std::string evaluation_json(const Evaluation& value) {
    std::ostringstream out;out<<"{\"episodes\":64,\"qualified\":"<<value.qualified<<",\"safe\":"<<value.safe<<",\"motion\":"<<value.motion<<",\"settled\":"<<value.settled<<",\"held10s\":"<<value.held<<",\"passed\":"<<value.passed<<",\"physical_control_calls\":"<<value.control_calls<<",\"integrated_substeps\":"<<value.integrated_substeps<<",\"strata\":[";
    for(int kind=0;kind<7;++kind) {if(kind) out<<',';out<<"{\"kind\":\""<<mb::name(static_cast<mb::Kind>(kind))<<"\",\"count\":"<<value.stratum_count[kind]<<",\"safe\":"<<value.stratum_safe[kind]<<",\"held10s\":"<<value.stratum_held[kind]<<",\"passed\":"<<value.stratum_passed[kind]<<'}';}out<<"]}";return out.str();
}
std::string arm_json(const ArmResult& value) {std::ostringstream out;out<<"{\"completed_updates\":"<<value.updates<<",\"attempted_updates\":"<<value.attempted_updates<<",\"physical_safety_stop\":"<<(value.physical_stop?"true":"false")<<",\"rollout_slots\":"<<value.rollout_slots<<",\"integrated_substeps\":"<<value.integrated_substeps<<",\"physical_control_equivalents\":"<<std::setprecision(17)<<value.integrated_substeps/10.<<",\"frozen_slots\":"<<value.frozen_slots<<'}';return out.str();}
}
int main(int argc,char** argv) {
    fs::path root;bool shared_failure_boundary=false;
    try {
        if(argc!=7) {std::cerr<<"Usage: motor_v2_train SOURCE HISTORICAL-REGULATION-GATE CONTACT-GATE REFERENCE-FIXTURES STUDYROOT --prepare|--readiness|--run\n";return 2;}
        const std::string command=argv[6];need(command=="--prepare"||command=="--readiness"||command=="--run","Explicit motor-v2 command required");Inputs in;
        in.source_bytes=checkpoint::read(argv[1]);in.source=rl::read_boulder_source(in.source_bytes);in.history_bytes=checkpoint::read(argv[2]);in.history=pm::RegulationGate::read(in.history_bytes);in.history.require_passed(in.source.fingerprint);
        in.contact_bytes=checkpoint::read(argv[3]);in.contact=mc::ContactGate::read(in.contact_bytes);in.reference_bytes=checkpoint::read(argv[4]);root=fs::absolute(argv[5]);
        fs::create_directories(root/"sources");for(const char* arm:{"torque","joint_target"}) fs::create_directories(root/"run"/arm);
        immutable_payload(root/"sources"/"source-boulder.bin",in.source_bytes);immutable_payload(root/"sources"/"historical-regulation-gate.bin",in.history_bytes);immutable_payload(root/"sources"/"contact-gate.bin",in.contact_bytes);immutable_payload(root/"sources"/"reference-fixtures.bin",in.reference_bytes);immutable_text(root/"motor-v2-protocol.json",manifest(in));
        const auto fixtures=final_fixtures(in,root);for(auto mode:{mv::Interface::Torque,mv::Interface::JointTarget}) immutable_payload(root/"run"/rl::motor_v2_arm(mode)/"initial-policy.bin",initial_policy(in,mode).serialize());
        if(command=="--prepare") {std::cout<<"MOTOR_V2_PREPARED physical_steps=0 training_updates=0 fixtures=64 contact_passed="<<in.contact.passed()<<'\n';return 0;}
        in.contact.require_passed();need(probe_device().available,"Motor-v2 native data readiness/training requires a tested CUDA device");
        need(!fs::exists(root/"global-stop.json"),"This motor-v2 study stopped on a shared defect; no retry/extension authorized");shared_failure_boundary=true;
        if(command=="--readiness") {readiness(in,root);need(checkpoint::read(argv[1])==in.source_bytes&&checkpoint::read(argv[2])==in.history_bytes&&checkpoint::read(argv[3])==in.contact_bytes&&checkpoint::read(argv[4])==in.reference_bytes,"Protected inputs changed during data readiness");return 0;}
        const auto ready=rl::MotorV2Readiness::read(checkpoint::read((root/"readiness-gate.bin").string()));ready.require_passed(in.source.fingerprint,in.history_bytes,in.contact_bytes);
        need(!fs::exists(root/"run-started.json")&&!fs::exists(root/"final-summary.json"),"Motor-v2 bounded run already started/completed; no automatic restart/retry");checkpoint::Lease lease((root/"motor-v2-study").string());immutable_text(root/"run-started.json","{\"maximum_updates_each\":64,\"maximum_training_slots_total\":4194304,\"separate_arm_physical_stops\":true}\n");
        const auto torque=train_arm(in,mv::Interface::Torque,root);need(checkpoint::read(argv[1])==in.source_bytes,"Protected original source changed after torque arm");const auto target=train_arm(in,mv::Interface::JointTarget,root);
        need(torque.rollout_slots<=2097152&&target.rollout_slots<=2097152&&torque.integrated_substeps+target.integrated_substeps<=41943040,"Motor-v2 paired training exceeded hard cap");
        auto torque_policy=rl::MotorV2Policy::load((root/"run"/"torque"/"candidate-policy.bin").string()),target_policy=rl::MotorV2Policy::load((root/"run"/"joint_target"/"candidate-policy.bin").string());
        need(torque_policy.update==torque.updates&&target_policy.update==target.updates,"Motor-v2 terminal actor labels differ from completed histories");
        const auto torque_eval=evaluate(torque_policy,fixtures,root/"run"/"torque"),target_eval=evaluate(target_policy,fixtures,root/"run"/"joint_target");need(torque_eval.control_calls+target_eval.control_calls<=76800,"Motor-v2 withheld evaluation cap exceeded");
        torque_policy.passed_final_gate=torque_policy.completed_budget&&torque_eval.passed==64;target_policy.passed_final_gate=target_policy.completed_budget&&target_eval.passed==64;
        checkpoint::write_atomic((root/"run"/"torque"/"candidate-policy.bin").string(),torque_policy.serialize());checkpoint::write_atomic((root/"run"/"joint_target"/"candidate-policy.bin").string(),target_policy.serialize());
        need(checkpoint::read(argv[1])==in.source_bytes&&checkpoint::read(argv[2])==in.history_bytes&&checkpoint::read(argv[3])==in.contact_bytes&&checkpoint::read(argv[4])==in.reference_bytes,"Protected sources/gates/captures changed during bounded comparison");
        std::ostringstream summary;summary<<"{\"protocol\":\"motor-v2\",\"physics_version\":2,\"source_unchanged\":true,\"torque_training\":"<<arm_json(torque)<<",\"joint_target_training\":"<<arm_json(target)<<",\"torque_evaluation\":"<<evaluation_json(torque_eval)<<",\"joint_target_evaluation\":"<<evaluation_json(target_eval)<<",\"torque_accepted\":"<<(torque_policy.passed_final_gate?"true":"false")<<",\"joint_target_accepted\":"<<(target_policy.passed_final_gate?"true":"false")<<",\"new_fixture_capture_controls\":0,\"readiness_controls\":32,\"adopted\":false,\"trace_alignment\":\"tick0 action preview; subsequent raw actions drive preceding state and are recorded beside resulting state/actual torques\"}\n";
        immutable_text(root/"final-summary.json",summary.str());std::cout<<"MOTOR_V2_FINISHED "<<summary.str();return 0;
    } catch(const std::exception& error) {
        if(shared_failure_boundary&&!root.empty()) {try {immutable_text(root/"global-stop.json","{\"shared_defect_or_gate_failure\":true,\"all_affected_work_stopped\":true,\"automatic_retry\":false,\"adopted\":false}\n");}catch(...) {}}
        std::cerr<<"MOTOR_V2_SHARED_STOP "<<error.what()<<'\n';return 1;
    }
}
