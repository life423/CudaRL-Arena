#ifndef CUDARL_PAIRED_MOTOR_H
#define CUDARL_PAIRED_MOTOR_H
#include "command_motor.h"
#include <array>
#include <cstring>
#include <stdexcept>

// Isolated experiment: the physical substep itself is never changed.
namespace cudarl::paired_motor {
constexpr int kObs=command_motor::kObs,kAct=4,kEpisodeSteps=600,kSubsteps=10;
constexpr uint32_t kProtocol=1;
enum class Interface:int {Torque=0,JointTarget=1};
struct Gains {float kp[4]{},kd[4]{};}; // SI N*m/rad and N*m/(rad/s).
struct State {
    command_motor::State task{};
    float held_target[4]{}; // rad; overwritten by every new50ms joint-target action
    float actual_torques[40]{}; // normalized actuator command, substep*4+joint
    float unclamped_torques[40]{}; // requested SI N*m, before physical actuator clamp
    float mean_squared_torque=0;
    float substep_max_target_error=0,substep_max_joint_rate=0; // prestate plus all ten5ms endpoints
    int first_unsafe_substep=-1; //0=before this tick,1..10=after corresponding5ms step
    int unsafe_substeps=0; // unsafe geometric samples among prestate plus10substeps
    command_motor::State first_unsafe_task{};
};
CUDARL_HD inline bool valid_interface(Interface mode) {return mode==Interface::Torque||mode==Interface::JointTarget;}
CUDARL_HD inline bool valid_gains(const Gains& gains,Interface mode) {
    for(int j=0;j<4;++j) if(!command_motor::finite_number(gains.kp[j])||!command_motor::finite_number(gains.kd[j])||
        gains.kp[j]<0||gains.kd[j]<0||(mode==Interface::JointTarget&&(gains.kp[j]<=0||gains.kd[j]<=0))) return false;
    return valid_interface(mode);
}
CUDARL_HD inline bool finite(const State& state) {
    if(!command_motor::finite(state.task)||!command_motor::finite_number(state.mean_squared_torque)||state.mean_squared_torque<0||state.mean_squared_torque>1.0001f||!command_motor::finite_number(state.substep_max_target_error)||!command_motor::finite_number(state.substep_max_joint_rate)||state.substep_max_target_error<0||state.substep_max_joint_rate<0||state.first_unsafe_substep< -1||state.first_unsafe_substep>10||state.unsafe_substeps<0||state.unsafe_substeps>11) return false;
    for(int j=0;j<4;++j) if(!command_motor::finite_number(state.held_target[j])||fabsf(state.held_target[j])>deep::limit) return false;
    for(int j=0;j<40;++j) if(!command_motor::finite_number(state.actual_torques[j])||fabsf(state.actual_torques[j])>1||!command_motor::finite_number(state.unclamped_torques[j])) return false;
    return state.first_unsafe_substep<0||command_motor::finite(state.first_unsafe_task);
}
CUDARL_HD inline State initial(uint32_t seed,uint32_t environment,uint32_t episode=0,swimmer::Body body={}) {
    State result;result.task=command_motor::initial(seed,environment,episode,body);return result;
}
CUDARL_HD inline void observe(const State& state,float* out) {command_motor::observe(state.task,out);}
CUDARL_HD inline command_motor::Result advance(State& state,const float* latent,Interface mode,const Gains& gains,deep::Physics physical=command_motor::physics()) {
    command_motor::Result result;
    if(!finite(state)||!valid_gains(gains,mode)) {result.invalid=true;result.reward=-100;return result;}
    for(int j=0;j<4;++j) if(!command_motor::finite_number(latent[j])) {result.invalid=true;result.reward=-100;return result;}
    state.first_unsafe_substep=-1;state.unsafe_substeps=0;state.mean_squared_torque=0;state.substep_max_target_error=state.substep_max_joint_rate=0;
    for(int j=0;j<40;++j) {state.actual_torques[j]=0;state.unclamped_torques[j]=0;}
    for(int j=0;j<4;++j) {
        state.held_target[j]=mode==Interface::JointTarget?deep::limit*deep::clamp(latent[j],-1,1):0;
        if(mode==Interface::JointTarget) state.substep_max_target_error=fmaxf(state.substep_max_target_error,fabsf(state.held_target[j]-state.task.body.angle[j]));
        state.substep_max_joint_rate=fmaxf(state.substep_max_joint_rate,fabsf(state.task.body.velocity[6+j]));
    }
    if(command_motor::physical_clearance(state.task.body)<0) {result.unsafe=true;state.first_unsafe_substep=0;state.unsafe_substeps=1;state.first_unsafe_task=state.task;}
    for(int step=0;step<10;++step) {
        float actual[4];
        for(int j=0;j<4;++j) {
            const float requested=mode==Interface::Torque?deep::torque*latent[j]:gains.kp[j]*(state.held_target[j]-state.task.body.angle[j])-gains.kd[j]*state.task.body.velocity[6+j];
            if(!command_motor::finite_number(requested)) {result.invalid=true;result.reward=-100;return result;}
            // For direct torque preserve exactly its previous normalized clamp
            // arithmetic; multiplying/dividing by torque would add rounding.
            actual[j]=mode==Interface::Torque?deep::clamp(latent[j],-1,1):deep::clamp(requested/deep::torque,-1,1);
            state.unclamped_torques[step*4+j]=requested;state.actual_torques[step*4+j]=actual[j];state.mean_squared_torque+=actual[j]*actual[j]/40.f;
        }
        deep::substep(state.task.body,actual,.005f,physical);
        for(int j=0;j<4;++j) state.task.previous_action[j]=actual[j];
        if(!command_motor::finite(state.task)) {result.invalid=true;result.reward=-100;return result;}
        for(int j=0;j<4;++j) {
            if(mode==Interface::JointTarget) state.substep_max_target_error=fmaxf(state.substep_max_target_error,fabsf(state.held_target[j]-state.task.body.angle[j]));
            state.substep_max_joint_rate=fmaxf(state.substep_max_joint_rate,fabsf(state.task.body.velocity[6+j]));
        }
        if(command_motor::physical_clearance(state.task.body)<0) {
            result.unsafe=true;++state.unsafe_substeps;if(state.first_unsafe_substep<0) {state.first_unsafe_substep=step+1;state.first_unsafe_task=state.task;}
        }
    }
    ++state.task.steps;result.timeout=state.task.steps>=kEpisodeSteps;
    const auto motion=command_motor::motion(state.task.body);auto cost=command_motor::tracking_cost(state.task,latent,motion);
    cost.torque=.002f*state.mean_squared_torque;result.reward=-deep::dt*cost.total();
    result.invalid=!finite(state)||!command_motor::finite_number(result.reward);if(result.invalid) result.reward=-100;return result;
}
inline uint64_t fingerprint(const std::vector<uint8_t>& bytes) {uint64_t h=14695981039346656037ull;for(uint8_t x:bytes) {h^=x;h*=1099511628211ull;}return h;}
struct RegulationCase {State initial{};float target[4]{};int source_fixture=-1;bool target_clipped=false;};
inline Gains derive_gains(const std::array<RegulationCase,8>& catalog) {
    double minimum[4]={1e30,1e30,1e30,1e30};
    for(const auto& item:catalog) {
        if(!finite(item.initial)) throw std::runtime_error("Nonfinite servo inertia catalog");
        const auto system=deep::assemble(item.initial.task.body);double L[10][10]{};
        for(int i=0;i<10;++i) for(int j=0;j<=i;++j) {
            double x=system.M[i][j];for(int k=0;k<j;++k) x-=L[i][k]*L[j][k];
            if(i==j) {if(!(x>0)||!std::isfinite(x)) throw std::runtime_error("Servo mass matrix not positive definite");L[i][j]=std::sqrt(x);} else L[i][j]=x/L[j][j];
        }
        for(int joint=0;joint<4;++joint) {
            double x[10]{};for(int i=0;i<10;++i) {double value=i==6+joint?1.:0.;for(int k=0;k<i;++k) value-=L[i][k]*x[k];x[i]=value/L[i][i];}
            for(int i=9;i>=0;--i) {double value=x[i];for(int k=i+1;k<10;++k) value-=L[k][i]*x[k];x[i]=value/L[i][i];}
            const double inertia=1./x[6+joint];if(!(inertia>0)||!std::isfinite(inertia)) throw std::runtime_error("Invalid coupled effective joint inertia");minimum[joint]=std::fmin(minimum[joint],inertia);
        }
    }
    const double omega=2.*3.14159265358979323846*2.;Gains result;
    for(int j=0;j<4;++j) {result.kp[j]=static_cast<float>(minimum[j]*omega*omega);result.kd[j]=static_cast<float>(2.*minimum[j]*omega);}
    if(!valid_gains(result,Interface::JointTarget)) throw std::runtime_error("Invalid derived joint gains");return result;
}
inline bool same_gains(const Gains& a,const Gains& b) {for(int j=0;j<4;++j) if(a.kp[j]!=b.kp[j]||a.kd[j]!=b.kd[j]) return false;return true;}
struct RegulationProtocol {
    std::array<RegulationCase,8> catalog{};Gains gains{};uint64_t source_fingerprint=0,fixture_fingerprint=0;
    void write_catalog(checkpoint::Writer& out) const {for(const auto& item:catalog) {out.value(item.initial);for(float q:item.target) out.value(q);out.value(item.source_fixture);out.value(item.target_clipped);}}
    uint64_t catalog_fingerprint() const {checkpoint::Writer out;write_catalog(out);return fingerprint(out.data);}
    void validate() const {
        if(!source_fingerprint||!fixture_fingerprint||!same_gains(gains,derive_gains(catalog))) throw std::runtime_error("Servo protocol source or derived gains differ");
        for(const auto& item:catalog) {
            if(!finite(item.initial)||item.initial.task.steps!=0||command_motor::physical_clearance(item.initial.task.body)<0) throw std::runtime_error("Unsafe servo regulation catalog");
            for(float target:item.target) if(!command_motor::finite_number(target)||fabsf(target)>deep::limit) throw std::runtime_error("Invalid servo regulation target");
        }
    }
    std::vector<uint8_t> serialize() const {
        validate();checkpoint::Writer out;out.value(uint32_t{0x504D5250u});out.value(kProtocol);out.value(uint32_t{sizeof(State)});out.value(source_fingerprint);out.value(fixture_fingerprint);write_catalog(out);out.value(gains);return out.data;
    }
    uint64_t payload_fingerprint() const {return fingerprint(serialize());}
    static RegulationProtocol read(const std::vector<uint8_t>& bytes) {
        checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=0x504D5250u||in.value<uint32_t>()!=kProtocol||in.value<uint32_t>()!=sizeof(State)) throw std::runtime_error("Incompatible servo regulation protocol");
        RegulationProtocol out;out.source_fingerprint=in.value<uint64_t>();out.fixture_fingerprint=in.value<uint64_t>();for(auto& item:out.catalog) {item.initial=in.value<State>();for(float& q:item.target) q=in.value<float>();item.source_fixture=in.value<int>();item.target_clipped=in.value<bool>();}out.gains=in.value<Gains>();in.finish();out.validate();return out;
    }
};
struct RegulationGate {
    RegulationProtocol protocol{};Gains gains{};uint64_t source_fingerprint=0,catalog_fingerprint=0,protocol_payload_hash=0;
    int cpu_steps=0,cuda_steps=0,cpu_cases=0,gpu_cases=0,cpu_passed=0,gpu_passed=0;bool completed=false,passed=false,parity_passed=false;
    void validate_binding(uint64_t source) const {
        protocol.validate();if(!source||source_fingerprint!=source||protocol.source_fingerprint!=source||catalog_fingerprint!=protocol.catalog_fingerprint()||protocol_payload_hash!=protocol.payload_fingerprint()||!same_gains(gains,protocol.gains)) throw std::runtime_error("Paired servo gate source/catalog/gain binding differs");
        if(cpu_steps<0||cpu_steps>1600||cuda_steps<0||cuda_steps>1600||cpu_cases<0||cpu_cases>8||gpu_cases<0||gpu_cases>8||cpu_passed<0||cpu_passed>cpu_cases||gpu_passed<0||gpu_passed>gpu_cases||completed!=(cpu_steps==1600&&cuda_steps==1600)||passed!=(completed&&cpu_cases==8&&gpu_cases==8&&cpu_passed==8&&gpu_passed==8&&parity_passed)) throw std::runtime_error("Invalid or overstated servo regulation gate");
    }
    void require_passed(uint64_t source) const {validate_binding(source);if(!passed) throw std::runtime_error("Paired motor experiment blocked: servo regulation gate did not pass");}
    std::vector<uint8_t> serialize() const {
        validate_binding(source_fingerprint);checkpoint::Writer out;out.value(uint32_t{0x504D5247u});out.value(kProtocol);out.array(protocol.serialize());out.value(gains);out.value(source_fingerprint);out.value(catalog_fingerprint);out.value(protocol_payload_hash);out.value(cpu_steps);out.value(cuda_steps);out.value(cpu_cases);out.value(gpu_cases);out.value(cpu_passed);out.value(gpu_passed);out.value(completed);out.value(passed);out.value(parity_passed);return out.data;
    }
    static RegulationGate read(const std::vector<uint8_t>& bytes) {
        checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=0x504D5247u||in.value<uint32_t>()!=kProtocol) throw std::runtime_error("Incompatible servo regulation gate");
        RegulationGate out;out.protocol=RegulationProtocol::read(in.array<uint8_t>(1000000));out.gains=in.value<Gains>();out.source_fingerprint=in.value<uint64_t>();out.catalog_fingerprint=in.value<uint64_t>();out.protocol_payload_hash=in.value<uint64_t>();out.cpu_steps=in.value<int>();out.cuda_steps=in.value<int>();out.cpu_cases=in.value<int>();out.gpu_cases=in.value<int>();out.cpu_passed=in.value<int>();out.gpu_passed=in.value<int>();out.completed=in.value<bool>();out.passed=in.value<bool>();out.parity_passed=in.value<bool>();in.finish();out.validate_binding(out.source_fingerprint);return out;
    }
};
struct RegulationFrame {int backend=0,case_id=0,tick=0;State before{},after{};float latent[4]{};command_motor::Result result{};};
inline std::vector<RegulationFrame> read_regulation_frames(const std::vector<uint8_t>& bytes,const RegulationProtocol& protocol) {
    checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=0x504D5246u||in.value<uint32_t>()!=kProtocol||in.value<uint32_t>()!=sizeof(RegulationFrame)||in.value<uint64_t>()!=protocol.payload_fingerprint()) throw std::runtime_error("Incompatible paired regulation frames");
    auto frames=in.array<RegulationFrame>(3200);in.finish();
    for(const auto& frame:frames) {
        if(frame.backend<0||frame.backend>1||frame.case_id<0||frame.case_id>=8||frame.tick<1||frame.tick>200||!finite(frame.before)||(!frame.result.invalid&&!finite(frame.after))) throw std::runtime_error("Invalid recorded paired regulation frame");
        if(frame.before.task.steps!=frame.tick-1||(!frame.result.invalid&&frame.after.task.steps!=frame.tick)) throw std::runtime_error("Recorded regulation clock differs");
        for(int j=0;j<4;++j) if(frame.latent[j]!=protocol.catalog[static_cast<size_t>(frame.case_id)].target[j]/deep::limit) throw std::runtime_error("Recorded regulation target differs");
    }
    return frames;
}
#if CUDARL_WITH_CUDA
class Batch {
    int n_;uint32_t seed_;Interface interface_=Interface::Torque;Gains gains_{};
    device_ptr<State> states_,first_unsafe_;device_ptr<float> observations_;device_ptr<uint32_t> invalid_,safety_;
public:
    Batch(int n,uint32_t seed,swimmer::Body body={});
    void configure(Interface,const Gains&,cudaStream_t);
    Interface mode() const {return interface_;} Gains gains() const {return gains_;}
    const float* obs() const {return observations_.get();} const State* states() const {return states_.get();}
    void step(const float*,swimmer::Outputs,cudaStream_t,bool autoreset=true);
    void step_physics(const float*,const deep::Physics*,const uint8_t*,cudaStream_t);
    std::vector<State> host_states(cudaStream_t) const;void set_states(const std::vector<State>&,cudaStream_t);
    uint64_t invalid_count(cudaStream_t) const;uint64_t unsafe_count(cudaStream_t) const;std::vector<State> unsafe_states(cudaStream_t) const;
    void save(checkpoint::Writer&) const;void load(checkpoint::Reader&);
};
#endif
}
#endif
