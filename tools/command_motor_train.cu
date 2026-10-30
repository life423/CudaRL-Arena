#include "rl/command_motor_ppo.h"
#include "rl/boulder_source.h"
#include "rl/shelter_motor_run.h"
#include "motor_benchmark.h"
#include "cuda/device.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>

// One explicitly gated, registered128-update pilot. Preparation never executes
// physics or training. Failed candidates stay diagnostic files in this child
// directory and are never installed into the working pond.
namespace {
using namespace cudarl;
namespace fs=std::filesystem;
namespace cm=command_motor;
namespace bench=motor_benchmark;
void need(bool value,const std::string& text) {if(!value) throw std::runtime_error(text);}
uint64_t hash_bytes(const std::vector<uint8_t>& bytes) {
    uint64_t hash=14695981039346656037ull;for(uint8_t value:bytes) {hash^=value;hash*=1099511628211ull;}return hash;
}
std::string text_file(const fs::path& path) {
    std::ifstream in(path,std::ios::binary);need(static_cast<bool>(in),"Cannot read "+path.string());
    return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
void immutable_text(const fs::path& path,const std::string& value) {
    if(fs::exists(path)) {need(text_file(path)==value,"Existing immutable protocol differs: "+path.string());return;}
    std::ofstream out(path,std::ios::binary);need(static_cast<bool>(out),"Cannot create "+path.string());out<<value;out.flush();need(static_cast<bool>(out),"Cannot write "+path.string());
}
void immutable_payload(const fs::path& path,const std::vector<uint8_t>& value) {
    if(fs::exists(path)) need(checkpoint::read(path.string())==value,"Existing immutable payload differs: "+path.string());
    else checkpoint::write_atomic(path.string(),value);
}
std::string protocol_json(const rl::BoulderSource& source,const std::vector<uint8_t>& calibrated_bytes) {
    std::ostringstream out;out<<std::setprecision(17);
    out<<"{\n  \"protocol\":2,\n  \"environment_fingerprint\":\""<<rl::command_motor_fingerprint()<<"\",\n"
       <<"  \"source_fingerprint\":\""<<source.fingerprint<<"\",\n  \"source_applied_actor_update\":"<<source.motor_update<<",\n"
       <<"  \"calibrated_reference_update\":300,\n  \"calibrated_reference_payload_hash\":\""<<hash_bytes(calibrated_bytes)<<"\",\n  \"benchmark_rules_fingerprint\":\""<<bench::rules_fingerprint()<<"\",\n"
       <<"  \"worlds\":512,\n  \"rollout\":64,\n  \"updates\":128,\n  \"maximum_training_steps\":4194304,\n"
       <<"  \"seeds\":{\"reference\":1292314112,\"head\":1292314113,\"training\":1292314114,\"development\":1292314115,\"final\":1292314116},\n"
       <<"  \"episode_ticks\":600,\n  \"control_dt\":0.05,\n  \"physical_substeps\":10,\n  \"physical_substep_dt\":0.005,\n"
       <<"  \"physics\":\"unchanged deep::advance_physics; default rectangular bounds; calm water; same mass, drag, buoyancy, inertia and four +/-10uNm internal actuators\",\n"
       <<"  \"request\":\"persistent position(model units), terminal velocity(SI zero), LEVEL world heading; no consumption, respawn, scripted phase stop or target switch\",\n"
       <<"  \"observation\":{\"inputs\":59,\"position_error_0_2\":\"body-local SI error /0.072m\",\"time_remaining_3\":\"max(0,1-steps/600)\",\"compatible_body_9_32\":\"exact original55 indices/scalers\",\"velocity_error_33_35\":\"body-local terminal minus actual velocity /0.02m/s\",\"55\":1,\"56_58\":\"body-local unit desired heading\",\"others\":0},\n"
       <<"  \"initialization\":\"one59-32-32-4 tanh/tanh/linear motor; copy only first-layer columns9..32 and hidden biases plus whole second layer from accepted55 actor; zero other first-layer columns; fresh orthogonal output gain0.01 seed0x4D072601; fresh critic seed training+1 and ALL fresh Adam/std\",\n"
       <<"  \"training_starts\":\"fixed counter-RNG six equal classes: rest-to-goal, moving-to-goal, braking momentum1..3cm/s, hold, turn, mild dynamic recovery; start yaw full circle/pitch+/-0.35rad, joints+/-0.25rad; terminal heading level, depth goal varies; see version1 initial() for exact frozen draws\",\n"
       <<"  \"cost\":\"reward=-0.05*(||head_position_error/0.072||^2 +0.25*||terminal_velocity_error/0.02||^2 +0.1*(1-clamp(forward_dot_heading,-1,1)) +0.05*MEAN216(surface_point_speed^2)/0.02^2 +0.02*||body_angular_velocity/0.5||^2 +0.02*MEAN4((joint_rate/4)^2) +0.002*MEAN4(clamp(action,-1,1)^2))\",\n"
       <<"  \"motion_sampling\":\"three articulated links x9 axial slices x8 ellipse rim samples; exact head profile and fin width; body_point_velocity=link_center_velocity+omega cross offset; RMS means square-root of arithmetic mean squares, cost uses mean squares\",\n"
       <<"  \"termination\":\"true600tick finite-horizon terminal done1/truncated0, remaining time observed; no contact-dependent early reset or reward clipping; any finite full-body breach (including illegal pre-state) or invalid action/dynamics rejects collection before GAE/Adam; retain last good save and first unsafe states; frozen trials never reset\",\n"
       <<"  \"ppo\":{\"gamma\":0.99,\"gae_lambda\":0.95,\"ratio_clip\":0.2,\"entropy\":0.002,\"value_weight\":0.5,\"epochs\":4,\"minibatches\":8,\"advantage_normalization\":\"existing per-minibatch shared mean/std\",\"initial_log_std\":-0.5,\"latent_gaussian\":\"four raw physical-unit means/samples/logp; clamp ONLY at the unchanged torque actuator\"},\n"
       <<"  \"adam\":{\"lr\":0.001,\"beta1\":0.9,\"beta2\":0.999,\"eps\":0.00001,\"max_global_grad_norm\":0.5},\n"
       <<"  \"evaluation\":\"frozen deterministic means;16 development at0/32/64/96/128;64 final only128 plus original55/calibratedv5update300 on identical complete fixtures; all complete starts registered before PPO; actual inherited55 physical braking warmup; every final case must pass full motion, safety, settling and10s whole-body hold; no best selection\",\n"
       <<"  \"protocol_float_values\":[";
    const auto values=rl::command_motor_protocol();for(size_t i=0;i<values.size();++i) {if(i) out<<',';out<<values[i];}
    out<<"]\n}\n";return out.str();
}
template<class T> void upload(device_ptr<T>& pointer,const std::vector<T>& values,cudaStream_t stream) {
    need(pointer.size()==values.size(),"Motor diagnostic upload dimension mismatch");
    CUDA_CHECK_THROW(cudaMemcpyAsync(pointer.get(),values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
template<class T> std::vector<T> download(const T* pointer,size_t count,cudaStream_t stream) {
    std::vector<T> values(count);CUDA_CHECK_THROW(cudaMemcpyAsync(values.data(),pointer,count*sizeof(T),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));return values;
}
std::vector<bench::Fixture> fixtures(const rl::BoulderSource& source,bench::Suite suite,uint32_t seed,const fs::path& path) {
    if(fs::exists(path)) {
        const auto payload=checkpoint::read(path.string());checkpoint::Reader in{payload};
        need(in.value<uint32_t>()==0x434D4649u&&in.value<uint32_t>()==1&&in.value<uint64_t>()==rl::command_motor_fingerprint()&&
            in.value<uint64_t>()==source.fingerprint,"Saved motor fixture protocol/source differs");
        const auto saved=in.array<bench::Fixture>(bench::suite_count(suite));in.finish();
        need(saved.size()==static_cast<size_t>(bench::suite_count(suite)),"Saved motor fixture count differs");
        for(size_t i=0;i<saved.size();++i) need(saved[i].id==static_cast<int>(i)&&saved[i].seed==seed&&cm::finite(saved[i].initial),"Saved motor fixture identity/state differs");
        return saved;
    }
    std::vector<bench::Fixture> result;bench::FrozenMotor original(source.motor,bench::TargetMode::Original55);
    checkpoint::Writer record;record.value(uint32_t{0x434D4649u});record.value(uint32_t{1});record.value(rl::command_motor_fingerprint());record.value(source.fingerprint);
    for(int id=0;id<bench::suite_count(suite);++id) {
        auto fixture=bench::make_fixture(seed,suite,id);bench::prepare_braking(fixture,original);result.push_back(fixture);
    }
    record.array(result);immutable_payload(path,record.data);return result;
}
struct Evaluation {int cases=0,qualified=0,safe=0,motion=0,settle=0,hold=0,passed=0;};
void trace_row(std::ofstream& out,int id,const cm::State& state,const bench::Sample& sample,const float* raw) {
    out<<id<<','<<state.steps<<','<<state.steps*deep::dt<<','<<state.body.position.x<<','<<state.body.position.y<<','<<state.body.position.z;
    for(int j=0;j<4;++j) out<<','<<raw[j];for(int j=0;j<4;++j) out<<','<<deep::clamp(raw[j],-1,1);
    out<<','<<sample.head_speed<<','<<sample.com_speed<<','<<sample.point_speed<<','<<sample.point_rms_speed<<','<<sample.angular_speed<<','<<sample.joint_rate
       <<','<<sample.goal_error<<','<<sample.heading_error<<','<<sample.clearance<<','<<sample.finite<<'\n';
}
Evaluation evaluate(const std::vector<float>& actor,const std::vector<bench::Fixture>& cases,const fs::path& path,bench::TargetMode mode=bench::TargetMode::Candidate59) {
    const int n=static_cast<int>(cases.size());rl::SwimmerStream stream;const auto s=stream.value;
    cm::Batch worlds(n,0xE71A0026u);bench::FrozenMotor policy(actor,mode);
    std::vector<cm::State> states;for(const auto& fixture:cases) states.push_back(fixture.initial);worlds.set_states(states,s);
    std::vector<bench::Progress> progress(n);std::vector<deep::Physics> physical(n);std::vector<uint8_t> active(n);
    device_ptr<deep::Physics> physical_device(n);device_ptr<uint8_t> active_device(n);device_ptr<float> action_device(static_cast<size_t>(n)*4);
    std::ofstream trace(path.string()+"-trace.csv");need(static_cast<bool>(trace),"Cannot write motor physical trace");trace<<std::setprecision(9);
    trace<<"case,tick,time_s,head_x_model,head_y_model,head_z_model,driving_raw0,driving_raw1,driving_raw2,driving_raw3,applied0,applied1,applied2,applied3,head_m_s,com_m_s,max_surface_m_s,rms_surface_m_s,angular_rad_s,max_joint_rad_s,goal_error_m,heading_error_rad,clearance_m,finite\n";
    auto predict=[&]() {std::vector<float> commands(static_cast<size_t>(n)*4);for(int i=0;i<n;++i) {
        const auto values=policy.action(states[i]);for(int j=0;j<4;++j) {need(cm::finite_number(values[j]),"Frozen motor produced nonfinite action");commands[i*4+j]=values[j];}}return commands;};
    auto actions=predict();
    for(int i=0;i<n;++i) {const auto sample=bench::measure(states[i]);bench::assess(progress[i],states[i],sample,cases[i]);trace_row(trace,i,states[i],sample,actions.data()+i*4);}
    // At tick0 action is an initial-state preview. Every tick>=1 row aligns the
    // preceding-state policy command with the resulting physical state.
    for(int tick=0;tick<600;++tick) {
        bool any=false;for(int i=0;i<n;++i) {
            active[i]=static_cast<uint8_t>(cases[i].qualified&&tick<cases[i].evaluation_ticks&&progress[i].finite);
            any=any||active[i];physical[i]=bench::physics_for(cases[i],states[i].steps);
        }
        if(!any) break;
        actions=predict();upload(action_device,actions,s);
        upload(physical_device,physical,s);upload(active_device,active,s);worlds.step_physics(action_device.get(),physical_device.get(),active_device.get(),s);states=worlds.host_states(s);
        need(worlds.invalid_count(s)==0,"Frozen motor physical trial became nonfinite; no recovery/reset allowed");
        for(int i=0;i<n;++i) if(active[i]) {const auto sample=bench::measure(states[i]);bench::assess(progress[i],states[i],sample,cases[i]);trace_row(trace,i,states[i],sample,actions.data()+i*4);}
    }
    std::ofstream result(path.string()+"-cases.csv");need(static_cast<bool>(result),"Cannot write motor case results");result<<std::setprecision(9);
    result<<"case,kind,qualified,safety,motion,settle,hold,passed,warmup_ticks,entry_speed_m_s,half_s,reach_s,heading_s,stop_s,stop_distance_m,settle_s,longest_hold_s,minimum_clearance_m,path_m,path_efficiency,error_slope_reversals,heading_error_slope_reversals,max_translation_m,overshoot_m,cross_track_m,first_safety_failure_tick\n";
    Evaluation total;total.cases=n;
    for(int i=0;i<n;++i) {
        const auto o=bench::finish(progress[i],cases[i]);total.qualified+=o.qualified;total.safe+=o.safety;total.motion+=o.motion;total.settle+=o.settle;total.hold+=o.hold;total.passed+=o.passed;
        result<<i<<','<<bench::name(cases[i].kind)<<','<<o.qualified<<','<<o.safety<<','<<o.motion<<','<<o.settle<<','<<o.hold<<','<<o.passed<<','<<cases[i].warmup_ticks<<','<<cases[i].actual_entry_speed
              <<','<<o.half_time<<','<<o.reach_time<<','<<o.heading_time<<','<<o.stop_time<<','<<o.stop_distance<<','<<o.settle_time<<','<<o.hold_seconds<<','<<progress[i].minimum_clearance
              <<','<<progress[i].path<<','<<bench::path_efficiency(progress[i])<<','<<progress[i].reversals<<','<<progress[i].heading_reversals<<','<<progress[i].max_translation
              <<','<<progress[i].max_overshoot<<','<<progress[i].max_cross_track<<','<<progress[i].first_failure_step<<'\n';
    }
    return total;
}
std::string evaluation_json(const Evaluation& e) {
    std::ostringstream out;out<<"{\"cases\":"<<e.cases<<",\"qualified\":"<<e.qualified<<",\"safe\":"<<e.safe<<",\"motion\":"<<e.motion
        <<",\"settle\":"<<e.settle<<",\"hold\":"<<e.hold<<",\"passed\":"<<e.passed<<'}';return out.str();
}
void verified_save(const rl::CommandMotorTrainer& trainer,const rl::BoulderSource& source,const fs::path& path,bool verify_readback) {
    const auto payload=rl::command_motor_snapshot(trainer);checkpoint::write_atomic(path.string(),payload);
    need(checkpoint::read(path.string())==payload,"Command motor checksum readback differs");
    if(verify_readback) {rl::CommandMotorTrainer restored(source.motor,source.fingerprint);rl::restore_command_motor(restored,payload);
        need(rl::command_motor_snapshot(restored)==payload,"Command motor complete learner/Adam/world resume differs");}
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=5&&argc!=6) {std::cerr<<"Usage: command_motor_train source-boulder.bin calibrated-v5-fullrun.bin NEW-output-dir --prepare\n       command_motor_train source-boulder.bin calibrated-v5-fullrun.bin output-dir --run passed-gate.bin\n";return 2;}
        const bool prepare=std::string(argv[4])=="--prepare",run=std::string(argv[4])=="--run";
        need((prepare&&argc==5)||(run&&argc==6),"Explicit --prepare or --run passed-gate.bin required");
        const auto source_bytes=checkpoint::read(argv[1]);const auto source=rl::read_boulder_source(source_bytes);
        const auto calibrated_bytes=checkpoint::read(argv[2]);const auto calibrated=rl::read_shelter_motor_run(calibrated_bytes);
        need(calibrated.source.parent==source_bytes&&calibrated.source.fingerprint==source.fingerprint&&calibrated.update==300,"Expected fixed calibratedv5 update300 with original parent");
        const fs::path output=fs::absolute(argv[3]);
        fs::create_directories(output);immutable_text(output/"pilot-protocol.json",protocol_json(source,calibrated_bytes));immutable_payload(output/"source-parent.bin",source_bytes);
        immutable_payload(output/"calibrated-reference.bin",calibrated_bytes);
        const auto initial_actor=rl::initialize_command_motor(source.motor);
        rl::CommandMotorPolicy initial;initial.source_fingerprint=source.fingerprint;initial.actor=initial_actor;immutable_payload(output/"initial-policy.bin",initial.serialize());
        if(prepare) {std::cout<<"COMMAND_MOTOR_PREPARED no_physics=1 no_training=1 source="<<source.fingerprint<<" environment="<<rl::command_motor_fingerprint()<<'\n';return 0;}
        const auto gate_bytes=checkpoint::read(argv[5]);const auto gate=rl::CommandMotorGate::read(gate_bytes);gate.require_passed(source.fingerprint,bench::rules_fingerprint());immutable_payload(output/"passed-preflight.bin",gate_bytes);
        need(!fs::exists(output/"final-summary.json"),"Final evaluation already exists; do not repeat the withheld suite");
        need(probe_device().available,"A tested CUDA device is required for the isolated motor pilot");
        checkpoint::Lease lease((output/"latest-training.bin").string());
        // Register complete physical starting conditions before any gradient.
        const auto development=fixtures(source,bench::Suite::Development,bench::development_seed,output/"development-fixtures.bin");
        const auto withheld=fixtures(source,bench::Suite::Final,bench::final_seed,output/"final-fixtures.bin");
        rl::CommandMotorTrainer trainer(source.motor,source.fingerprint);
        if(checkpoint::exists((output/"latest-training.bin").string())) {
            const auto bytes=checkpoint::read((output/"latest-training.bin").string());rl::restore_command_motor(trainer,bytes);
            need(rl::command_motor_snapshot(trainer)==bytes,"Resumed command motor full state differs");
        } else verified_save(trainer,source,output/"latest-training.bin",true);
        std::ofstream history(output/"training.csv",std::ios::app);need(static_cast<bool>(history),"Cannot write bounded motor training history");history<<std::setprecision(9);
        if(trainer.update_id==0) history<<"update,env_steps,seconds,reward_mean,reward_min,reward_max,actor_gradient_norm,critic_gradient_norm,std_gradient_norm,logstd0,logstd1,logstd2,logstd3\n";
        auto development_check=[&]() {
            const std::string label="development-"+std::to_string(trainer.update_id);const auto result=evaluate(trainer.actor.get_params(),development,output/label);
            immutable_text(output/(label+".json"),evaluation_json(result)+"\n");std::cout<<"COMMAND_MOTOR_DEVELOPMENT update="<<trainer.update_id<<" passed="<<result.passed<<'/'<<result.cases<<" selection=none\n";
        };
        if(trainer.update_id%32==0&&!fs::exists(output/("development-"+std::to_string(trainer.update_id)+".json"))) development_check();
        while(trainer.update_id<rl::command_motor_updates) {
            double seconds=0;const int adam_before=trainer.actor_opt.steps();
            try {seconds=trainer.update_seconds();}
            catch(const std::exception& error) {
                // Never overwrite latest-training.bin with a rejected rollout.
                checkpoint::Writer failure;failure.value(uint32_t{0x434D4641u});failure.value(uint32_t{2});
                failure.value(trainer.update_id);failure.value(adam_before);failure.value(trainer.actor_opt.steps());
                failure.value(trainer.env.invalid_count(trainer.stream.value));failure.value(trainer.env.unsafe_count(trainer.stream.value));
                failure.array(trainer.env.unsafe_states(trainer.stream.value));failure.array(trainer.env.host_states(trainer.stream.value));
                failure.array(download(trainer.obs.get(),trainer.obs.size(),trainer.stream.value));
                failure.array(download(trainer.actions.get(),trainer.actions.size(),trainer.stream.value));
                failure.array(download(trainer.rewards.get(),trainer.rewards.size(),trainer.stream.value));
                failure.array(download(trainer.logp.get(),trainer.logp.size(),trainer.stream.value));
                checkpoint::write_atomic((output/("rejected-rollout-"+std::to_string(trainer.update_id)+".bin")).string(),failure.data);
                std::cerr<<"COMMAND_MOTOR_REJECTED update="<<trainer.update_id<<" actor_adam_before="<<adam_before<<" actor_adam_after="<<trainer.actor_opt.steps()
                         <<" last_good_checkpoint_preserved=1 reason="<<error.what()<<'\n';throw;
            }
            const auto reward=download(trainer.rewards.get(),trainer.batch,trainer.stream.value);
            double sum=0;float minimum=1e30f,maximum=-1e30f;for(float value:reward) {need(std::isfinite(value),"Nonfinite motor reward");sum+=value;minimum=fminf(minimum,value);maximum=fmaxf(maximum,value);}
            const auto deviation=download(trainer.log_std.get(),4,trainer.stream.value);
            for(const auto* values:{&deviation}) for(float value:*values) need(std::isfinite(value),"Nonfinite motor Gaussian deviation");
            for(float value:trainer.actor.get_params()) need(std::isfinite(value),"Nonfinite motor actor");
            history<<trainer.update_id<<','<<static_cast<long long>(trainer.update_id)*trainer.batch<<','<<seconds<<','<<sum/reward.size()<<','<<minimum<<','<<maximum
                   <<','<<trainer.actor_opt.last_grad_norm()<<','<<trainer.critic_opt.last_grad_norm()<<','<<trainer.std_opt.last_grad_norm();for(float value:deviation) history<<','<<value;history<<'\n';history.flush();
            verified_save(trainer,source,output/"latest-training.bin",false);
            if(trainer.update_id%32==0) {
                verified_save(trainer,source,output/("training-update-"+std::to_string(trainer.update_id)+".bin"),true);development_check();
            }
            std::cout<<"COMMAND_MOTOR_TRAIN update="<<trainer.update_id<<"/128 steps="<<static_cast<long long>(trainer.update_id)*trainer.batch<<" actor_adam="<<trainer.actor_opt.steps()<<'\n';
        }
        need(trainer.update_id==128&&trainer.actor_opt.steps()==4096&&trainer.critic_opt.steps()==4096&&trainer.std_opt.steps()==4096,"Final bounded motor counters disagree");
        need(!fs::exists(output/"final-summary.json"),"Final evaluation already exists; do not repeat the withheld suite");
        const auto final=evaluate(trainer.actor.get_params(),withheld,output/"final");
        const auto original_final=evaluate(source.motor,withheld,output/"original-final",bench::TargetMode::Original55);
        const auto calibrated_final=evaluate(calibrated.actor,withheld,output/"calibrated-final",bench::TargetMode::Calibrated59);
        const bool source_preserved=checkpoint::read(argv[1])==source_bytes&&checkpoint::read(argv[2])==calibrated_bytes;need(source_preserved,"Protected motor references changed during pilot");
        rl::CommandMotorPolicy candidate;candidate.update=128;candidate.source_fingerprint=source.fingerprint;candidate.actor=trainer.actor.get_params();candidate.passed_final_gate=final.passed==64;
        checkpoint::write_atomic((output/"candidate-policy.bin").string(),candidate.serialize());
        need(rl::CommandMotorPolicy::load((output/"candidate-policy.bin").string()).serialize()==candidate.serialize(),"Candidate policy readback differs");
        std::ostringstream summary;summary<<"{\"protocol\":1,\"update\":128,\"training_steps\":4194304,\"adam_steps\":4096,\"final\":"<<evaluation_json(final)
            <<",\"original_final\":"<<evaluation_json(original_final)<<",\"calibrated_final\":"<<evaluation_json(calibrated_final)
            <<",\"calibrated_reference_update\":"<<calibrated.update<<",\"calibrated_reference_payload_hash\":\""<<hash_bytes(calibrated_bytes)<<"\""
            <<",\"passed_final_gate\":"<<(candidate.passed_final_gate?"true":"false")<<",\"source_preserved\":true,\"automatically_adopted\":false,\"final_actor_selection\":\"fixed update128\",\"frozen_inference\":\"all three native CPU float MLPs/adapters drive same CUDA physical worlds\",\"legacy_unsupported_fields\":\"Original55 has no explicit heading/stop request; virtual target is not consumed. Calibrated59 uses its old cover command and unchanged identity physical means.\",\"trace_action_alignment\":\"tick0 preview; tick>=1 driving preceding-state action with resulting state\"}\n";
        immutable_text(output/"final-summary.json",summary.str());
        std::cout<<"COMMAND_MOTOR_FINISHED final="<<final.passed<<"/64 source_preserved=1 adopted=0\n";return candidate.passed_final_gate?0:3;
    } catch(const std::exception& error) {std::cerr<<"COMMAND_MOTOR_STOPPED "<<error.what()<<'\n';return 1;}
}
