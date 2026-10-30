#include "envs/swimmer/paired_motor.h"
#include "motor_benchmark.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
using namespace cudarl;
namespace pm=paired_motor;namespace fs=std::filesystem;
constexpr uint32_t regulation_seed=0x4D072620u;
void need(bool yes,const std::string& message) {if(!yes) throw std::runtime_error(message);}
std::string text(const fs::path& path) {std::ifstream in(path,std::ios::binary);need(static_cast<bool>(in),"Cannot read "+path.string());return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};}
void write_new(const fs::path& path,const std::string& value) {need(!fs::exists(path),"Refusing to replace regulation evidence");std::ofstream out(path,std::ios::binary);out<<value;out.flush();need(static_cast<bool>(out),"Cannot write regulation evidence");}
struct Source {std::vector<uint8_t> bytes;std::vector<motor_benchmark::Fixture> fixtures;uint64_t source=0,environment=0;};
Source source(const fs::path& path) {
    Source result;result.bytes=checkpoint::read(path.string());checkpoint::Reader in{result.bytes};need(in.value<uint32_t>()==0x434D4649u&&in.value<uint32_t>()==1,"Expected archived reference fixture format1");
    result.environment=in.value<uint64_t>();result.source=in.value<uint64_t>();result.fixtures=in.array<motor_benchmark::Fixture>(16);in.finish();
    need(result.fixtures.size()==16&&result.source==2073117933376950176ull&&pm::fingerprint(result.bytes)==16079702944816815034ull,"This fixed study requires the exact protected reference-v1 fixture payload");return result;
}
pm::RegulationProtocol prepare(const Source& input) {
    pm::RegulationProtocol out;out.source_fingerprint=input.source;out.fixture_fingerprint=pm::fingerprint(input.bytes);
    for(int i=0;i<6;++i) {
        auto& item=out.catalog[i];auto& task=item.initial.task;task.seed=regulation_seed;task.environment=static_cast<uint32_t>(i);task.body=deep::BodyState{};task.body.position={0,-2,0};task.request.position=task.body.position;task.request.heading={1,0,0};
        if(i<4) item.target[i]=.2f;
        else {const float sign=i==4?1.f:-1.f;task.body.angle[0]=sign*.2f;task.body.angle[1]=sign*.1f;task.body.angle[2]=-sign*.2f;task.body.angle[3]=-sign*.1f;for(int j=0;j<4;++j) item.target[j]=task.body.angle[j];}
    }
    for(int i=6;i<8;++i) {
        auto& item=out.catalog[i];const auto& archived=input.fixtures[static_cast<size_t>(i-2)];need(archived.id==i-2&&archived.qualified&&archived.initial.steps==0,"Captured swimming start identity differs");
        item.initial.task=archived.initial;item.source_fixture=archived.id;
        for(int j=0;j<4;++j) {item.target[j]=deep::clamp(archived.initial.body.angle[j],-deep::limit,deep::limit);item.target_clipped=item.target_clipped||item.target[j]!=archived.initial.body.angle[j];}
    }
    out.gains=pm::derive_gains(out.catalog);out.validate();return out;
}
std::string manifest(const pm::RegulationProtocol& protocol,uint64_t archived_environment) {
    std::ostringstream out;out<<std::setprecision(17);
    out<<"{\"protocol\":1,\"scope\":\"nonlearning coupled joint regulation only; no claim of swimming or braking\",\"source_fingerprint\":\""<<protocol.source_fingerprint<<"\",\"archived_fixture_environment_fingerprint\":\""<<archived_environment<<"\",\"fixture_payload_fingerprint\":\""<<protocol.fixture_fingerprint<<"\",\"catalog_fingerprint\":\""<<protocol.catalog_fingerprint()<<"\",\"protocol_payload_hash\":\""<<protocol.payload_fingerprint()
       <<"\",\"state_bytes\":"<<sizeof(pm::State)<<",\"frame_bytes\":"<<sizeof(pm::RegulationFrame)<<",\"seed\":"<<regulation_seed<<",\"CPU_cases\":8,\"CUDA_cases\":8,\"ticks_per_case\":200,\"dt_s\":0.05,\"substeps_per_tick\":10,\"substep_s\":0.005,\"maximum_physical_control_ticks\":3200,\"interface\":\"joint target radians=limit*clip(latent,-1,1), held50ms; SI PD recalculated each5ms; no rate limiter\",\"physical_integrator\":\"unchanged deep::substep and all physical forces/limits/work\",\"gain_derivation\":\"one fixed2Hz,zeta1 proposal: minimum Ieff=1/(M^-1)_jj over the eight exact initial poses; full10x10 mass matrix from unchanged deep::assemble, double Cholesky solves; kp=Imin*(2*pi*2)^2,kd=2*Imin*(2*pi*2). Coupling/drag/saturation mean this is not proof of critical damping\",\"kp_Nm_per_rad\":[";
    for(int j=0;j<4;++j) {if(j) out<<',';out<<protocol.gains.kp[j];}out<<"],\"kd_Nm_per_rad_s\":[";for(int j=0;j<4;++j) {if(j) out<<',';out<<protocol.gains.kd[j];}
    out<<"],\"Ieffective_min_kg_m2_from_registered_kp\":[";const double omega=2.*3.14159265358979323846*2.;for(int j=0;j<4;++j) {if(j) out<<',';out<<protocol.gains.kp[j]/(omega*omega);}
    out<<"],\"gate\":\"all initial and5ms states finite/nonpenetrating, actualtorque<=10uNm; final continuous joint-settling run starts by5s AND fixed41 control observations t8..10s all four errors<=.02rad and rates<=.05rad/s, with t8endpoint and every5ms sample in the following40intervals; no target recentering/rescue\",\"CPU_CUDA_parity\":\"each recorded current-state/telemetry/reward float: abs(error)<=2e-4+2e-3*abs(CPUreference); NOT a relaxation of joint acceptance bands\",\"cost\":\"command_motor tracking terms unchanged, effort=.002*mean40(ACTUAL normalized torque^2), previous-action observation is LAST actual5ms normalizedtorque\",\"cases\":[";
    for(int i=0;i<8;++i) {if(i) out<<',';const auto& item=protocol.catalog[i];out<<"{\"id\":"<<i<<",\"source_fixture\":"<<item.source_fixture<<",\"target_clipped\":"<<(item.target_clipped?"true":"false")<<",\"targets_rad\":[";for(int j=0;j<4;++j) {if(j) out<<',';out<<item.target[j];}out<<"]}";}
    out<<"],\"extra_physics_for_numerical_checks\":0,\"PPO_updates\":0,\"production_modified\":false}\n";return out.str();
}
struct Metrics {int ticks=0,quiet_run_start=-1,last_window_samples=0;float max_error=0,max_rate=0,max_point=0,min_clearance=1e6f,max_torque=0;bool finite=true,safe=true,caps=true,last_window=true;};
void inspect(Metrics& m,const pm::State& state,const pm::RegulationCase& item) {
    float error=0,rate=0;for(int j=0;j<4;++j) {error=fmaxf(error,fabsf(state.task.body.angle[j]-item.target[j]));rate=fmaxf(rate,fabsf(state.task.body.velocity[6+j]));}
    const bool endpoint_quiet=error<=.02f&&rate<=.05f;
    const float interval_error=state.task.steps>0?state.substep_max_target_error:error,interval_rate=state.task.steps>0?state.substep_max_joint_rate:rate;
    const bool quiet=interval_error<=.02f&&interval_rate<=.05f;
    if(quiet) {if(m.quiet_run_start<0) m.quiet_run_start=state.task.steps>0?state.task.steps-1:0;} else m.quiet_run_start=-1;
    m.finite=m.finite&&pm::finite(state);m.safe=m.safe&&state.first_unsafe_substep<0;m.max_error=fmaxf(m.max_error,interval_error);m.max_rate=fmaxf(m.max_rate,interval_rate);
    const auto sample=motor_benchmark::measure(state.task);m.min_clearance=fminf(m.min_clearance,sample.clearance);m.max_point=fmaxf(m.max_point,sample.point_speed);m.safe=m.safe&&sample.clearance>=0;
    for(float a:state.actual_torques) {m.caps=m.caps&&fabsf(a)<=1&&command_motor::finite_number(a);m.max_torque=fmaxf(m.max_torque,fabsf(a)*deep::torque);}
    if(state.task.steps>=160) {++m.last_window_samples;m.last_window=m.last_window&&(state.task.steps==160?endpoint_quiet:quiet);}
}
bool passed(const Metrics& m) {return m.ticks==200&&m.finite&&m.safe&&m.caps&&m.quiet_run_start>=0&&m.quiet_run_start<=100&&m.last_window&&m.last_window_samples==41;}
void trace_header(std::ostream& out) {
    out<<"backend,case,tick,time_s,head_x_model,head_y_model,head_z_model,head_speed_m_s,COM_speed_m_s,max_point_m_s,clearance_m,reward,unsafe_substep,mean_normalized_torque_squared,substep_max_target_error_rad,substep_max_joint_rate_rad_s";
    for(int j=0;j<4;++j) out<<",target"<<j;for(int j=0;j<4;++j) out<<",joint"<<j;for(int j=0;j<10;++j) out<<",velocity"<<j;for(int j=0;j<40;++j) out<<",actual_normalized_torque"<<j;for(int j=0;j<40;++j) out<<",unclamped_SI_torque"<<j;out<<'\n';
}
void trace_row(std::ostream& out,int backend,int id,const pm::State& state,const pm::RegulationCase& item,float reward) {
    const auto m=motor_benchmark::measure(state.task);out<<backend<<','<<id<<','<<state.task.steps<<','<<state.task.steps*deep::dt<<','<<state.task.body.position.x<<','<<state.task.body.position.y<<','<<state.task.body.position.z<<','<<m.head_speed<<','<<m.com_speed<<','<<m.point_speed<<','<<m.clearance<<','<<reward<<','<<state.first_unsafe_substep<<','<<state.mean_squared_torque<<','<<state.substep_max_target_error<<','<<state.substep_max_joint_rate;
    for(float q:item.target) out<<','<<q;for(float q:state.task.body.angle) out<<','<<q;for(float v:state.task.body.velocity) out<<','<<v;for(float a:state.actual_torques) out<<','<<a;for(float a:state.unclamped_torques) out<<','<<a;out<<'\n';
}
struct Parity {uint64_t count=0,failed=0;double worst=0;};
void parity(Parity& p,float cpu,float gpu) {++p.count;const double ratio=std::abs(double(cpu)-gpu)/(2e-4+2e-3*std::abs(double(cpu)));if(!std::isfinite(ratio)||ratio>1) ++p.failed;p.worst=std::fmax(p.worst,ratio);}
void compare(Parity& p,const pm::State& a,const pm::State& b,float ar,float br) {
    const auto& x=a.task.body;const auto& y=b.task.body;for(auto v:{std::array<float,2>{x.position.x,y.position.x},{x.position.y,y.position.y},{x.position.z,y.position.z},{x.orientation.x,y.orientation.x},{x.orientation.y,y.orientation.y},{x.orientation.z,y.orientation.z},{x.orientation.w,y.orientation.w},{x.work,y.work},{x.travel,y.travel}}) parity(p,v[0],v[1]);
    for(int j=0;j<4;++j) {parity(p,x.angle[j],y.angle[j]);parity(p,a.held_target[j],b.held_target[j]);parity(p,a.task.previous_action[j],b.task.previous_action[j]);}for(int j=0;j<10;++j) parity(p,x.velocity[j],y.velocity[j]);for(int j=0;j<40;++j) {parity(p,a.actual_torques[j],b.actual_torques[j]);parity(p,a.unclamped_torques[j],b.unclamped_torques[j]);}parity(p,a.mean_squared_torque,b.mean_squared_torque);parity(p,a.substep_max_target_error,b.substep_max_target_error);parity(p,a.substep_max_joint_rate,b.substep_max_joint_rate);parity(p,ar,br);
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=4) {std::cerr<<"Usage: paired_motor_regulation archived-reference-fixtures.bin NEW-output-dir --prepare\n       paired_motor_regulation same-reference-fixtures.bin prepared-output-dir --run\n";return 2;}
        const fs::path path=fs::absolute(argv[1]),output=fs::absolute(argv[2]);const std::string mode=argv[3];need(mode=="--prepare"||mode=="--run","Explicit prepare/run required");const auto input=source(path);
        if(mode=="--prepare") {
            need(!fs::exists(output),"Regulation preparation requires a new directory");const auto protocol=prepare(input);fs::create_directories(output);checkpoint::write_atomic((output/"input-reference-fixtures.bin").string(),input.bytes);checkpoint::write_atomic((output/"regulation-protocol.bin").string(),protocol.serialize());const auto json=manifest(protocol,input.environment);write_new(output/"protocol.json",json);std::cout<<json<<"PAIRED_REGULATION_PREPARED no_cuda=1 physical_steps=0\n";return 0;
        }
        const auto bytes=checkpoint::read((output/"regulation-protocol.bin").string());const auto protocol=pm::RegulationProtocol::read(bytes);need(protocol.source_fingerprint==input.source&&protocol.fixture_fingerprint==pm::fingerprint(input.bytes)&&checkpoint::read((output/"input-reference-fixtures.bin").string())==input.bytes,"Prepared regulation source changed");need(text(output/"protocol.json")==manifest(protocol,input.environment),"Prepared regulation protocol text differs");
        need(!fs::exists(output/"run-started.json"),"Regulation run already attempted; no repetition or gain changes permitted");write_new(output/"run-started.json","{\"maximum_control_ticks\":3200,\"gain_search\":false,\"PPO_updates\":0}\n");
        pm::RegulationGate gate;gate.protocol=protocol;gate.gains=protocol.gains;gate.source_fingerprint=protocol.source_fingerprint;gate.catalog_fingerprint=protocol.catalog_fingerprint();gate.protocol_payload_hash=protocol.payload_fingerprint();
        std::vector<pm::State> cpu,gpu;for(const auto& item:protocol.catalog) cpu.push_back(item.initial);gpu=cpu;std::array<Metrics,8> cpu_metric{},gpu_metric{};for(int i=0;i<8;++i) {inspect(cpu_metric[i],cpu[i],protocol.catalog[i]);inspect(gpu_metric[i],gpu[i],protocol.catalog[i]);}
        pm::Batch worlds(8,regulation_seed);worlds.configure(pm::Interface::JointTarget,protocol.gains,nullptr);worlds.set_states(gpu,nullptr);std::vector<float> latent(32);for(int i=0;i<8;++i) for(int j=0;j<4;++j) latent[i*4+j]=protocol.catalog[i].target[j]/deep::limit;
        device_ptr<float> actions(32),rewards(8),done(8),truncated(8);actions.copy_from_host(latent.data(),latent.size());CUDA_CHECK_THROW(cudaDeviceSynchronize());std::ofstream trace(output/"regulation-trace.csv");need(static_cast<bool>(trace),"Cannot create regulation trace");trace<<std::setprecision(9);trace_header(trace);
        for(int i=0;i<8;++i) {trace_row(trace,0,i,cpu[i],protocol.catalog[i],0);trace_row(trace,1,i,gpu[i],protocol.catalog[i],0);}
        std::vector<pm::RegulationFrame> frames;frames.reserve(3200);Parity equality;bool aborted=false;
        for(int tick=1;tick<=200&&!aborted;++tick) {
            float cpu_reward[8]{};
            for(int i=0;i<8;++i) {
                pm::RegulationFrame frame;frame.backend=0;frame.case_id=i;frame.tick=tick;frame.before=cpu[i];for(int j=0;j<4;++j) frame.latent[j]=latent[i*4+j];frame.result=pm::advance(cpu[i],frame.latent,pm::Interface::JointTarget,protocol.gains);frame.after=cpu[i];frames.push_back(frame);cpu_reward[i]=frame.result.reward;++gate.cpu_steps;++cpu_metric[i].ticks;inspect(cpu_metric[i],cpu[i],protocol.catalog[i]);trace_row(trace,0,i,cpu[i],protocol.catalog[i],frame.result.reward);
                if(frame.result.invalid||frame.result.unsafe) {aborted=true;break;}
            }
            if(aborted) break;
            const auto before=gpu;worlds.step(actions.get(),{rewards.get(),done.get(),truncated.get(),nullptr},nullptr,false);gpu=worlds.host_states(nullptr);std::vector<float> actual_reward(8);rewards.copy_to_host(actual_reward.data(),8);
            for(int i=0;i<8;++i) {
                pm::RegulationFrame frame;frame.backend=1;frame.case_id=i;frame.tick=tick;frame.before=before[i];frame.after=gpu[i];for(int j=0;j<4;++j) frame.latent[j]=latent[i*4+j];frame.result.reward=actual_reward[i];frame.result.invalid=!pm::finite(gpu[i]);frame.result.unsafe=gpu[i].first_unsafe_substep>=0;frame.result.timeout=gpu[i].task.steps>=pm::kEpisodeSteps;frames.push_back(frame);++gate.cuda_steps;++gpu_metric[i].ticks;inspect(gpu_metric[i],gpu[i],protocol.catalog[i]);trace_row(trace,1,i,gpu[i],protocol.catalog[i],actual_reward[i]);compare(equality,cpu[i],gpu[i],cpu_reward[i],actual_reward[i]);
            }
            if(worlds.invalid_count(nullptr)||worlds.unsafe_count(nullptr)) aborted=true;
        }
        gate.cpu_cases=gate.gpu_cases=8;for(int i=0;i<8;++i) {gate.cpu_passed+=passed(cpu_metric[i]);gate.gpu_passed+=passed(gpu_metric[i]);}gate.completed=gate.cpu_steps==1600&&gate.cuda_steps==1600;gate.parity_passed=equality.failed==0&&gate.completed;gate.passed=gate.completed&&gate.cpu_passed==8&&gate.gpu_passed==8&&gate.parity_passed;
        checkpoint::Writer record;record.value(uint32_t{0x504D5246u});record.value(pm::kProtocol);record.value(uint32_t{sizeof(pm::RegulationFrame)});record.value(protocol.payload_fingerprint());record.array(frames);checkpoint::write_atomic((output/"regulation-frames.bin").string(),record.data);checkpoint::write_atomic((output/"regulation-gate.bin").string(),gate.serialize());
        need(checkpoint::read(path.string())==input.bytes&&checkpoint::read((output/"regulation-protocol.bin").string())==bytes,"Protected regulation source changed");trace.flush();need(static_cast<bool>(trace),"Regulation trace write failed");
        std::ostringstream report;report<<std::setprecision(17)<<"{\"completed\":"<<(gate.completed?"true":"false")<<",\"passed\":"<<(gate.passed?"true":"false")<<",\"aborted_physical_failure\":"<<(aborted?"true":"false")<<",\"cpu_steps\":"<<gate.cpu_steps<<",\"cuda_steps\":"<<gate.cuda_steps<<",\"cpu_passed\":"<<gate.cpu_passed<<",\"cuda_passed\":"<<gate.gpu_passed<<",\"parity_fields\":"<<equality.count<<",\"parity_failed\":"<<equality.failed<<",\"maximum_parity_error_over_tolerance\":"<<equality.worst<<",\"frames\":"<<frames.size()<<",\"catalog_fingerprint\":\""<<protocol.catalog_fingerprint()<<"\",\"protocol_payload_hash\":\""<<protocol.payload_fingerprint()<<"\",\"source_unchanged\":true,\"PPO_updates\":0,\"not_a_swimming_or_braking_claim\":true,\"cases\":[";
        for(int backend=0;backend<2;++backend) for(int i=0;i<8;++i) {if(backend||i) report<<',';const auto& m=backend==0?cpu_metric[i]:gpu_metric[i];report<<"{\"backend\":"<<backend<<",\"id\":"<<i<<",\"ticks\":"<<m.ticks<<",\"passed\":"<<(passed(m)?"true":"false")<<",\"final_continuous_settle_start_s\":"<<(m.quiet_run_start<0?-1.f:m.quiet_run_start*deep::dt)<<",\"fixed_last_window_samples\":"<<m.last_window_samples<<",\"last_window_passed\":"<<(m.last_window?"true":"false")<<",\"finite\":"<<(m.finite?"true":"false")<<",\"safe\":"<<(m.safe?"true":"false")<<",\"max_joint_error_rad\":"<<m.max_error<<",\"max_joint_rate_rad_s\":"<<m.max_rate<<",\"max_body_point_speed_m_s\":"<<m.max_point<<",\"minimum_endpoint_clearance_m\":"<<m.min_clearance<<",\"max_actual_torque_Nm\":"<<m.max_torque<<'}';}
        report<<"]}\n";write_new(output/"summary.json",report.str());std::cout<<"PAIRED_REGULATION_FINISHED passed="<<gate.passed<<" cpu="<<gate.cpu_passed<<"/8 cuda="<<gate.gpu_passed<<"/8 steps="<<gate.cpu_steps+gate.cuda_steps<<" no_learning=1\n";return gate.passed?0:3;
    } catch(const std::exception& error) {std::cerr<<"Paired motor regulation stopped: "<<error.what()<<'\n';return 1;}
}
