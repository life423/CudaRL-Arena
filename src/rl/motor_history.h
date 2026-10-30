#ifndef CUDARL_MOTOR_HISTORY_H
#define CUDARL_MOTOR_HISTORY_H
#include "envs/swimmer/motor_types.h"
#include <array>
#include <stdexcept>
// Pure POD evidence schema and original validation. No old motor integration,
// GPU world, neural network or training implementation is retained here.
namespace cudarl::motor_history {
using Interface=motor::Interface;using Gains=motor::Gains;
constexpr uint32_t kProtocol=1;
struct State {
    motor::TaskState task{};
    float held_target[4]{}; // rad; overwritten by every new50ms joint-target action
    float actual_torques[40]{}; // normalized actuator command, substep*4+joint
    float unclamped_torques[40]{}; // requested SI N*m, before physical actuator clamp
    float mean_squared_torque=0;
    float substep_max_target_error=0,substep_max_joint_rate=0; // prestate plus all ten5ms endpoints
    int first_unsafe_substep=-1; //0=before this tick,1..10=after corresponding5ms step
    int unsafe_substeps=0; // unsafe geometric samples among prestate plus10substeps
    motor::TaskState first_unsafe_task{};
};
CUDARL_HD inline bool valid_interface(Interface mode) {return mode==Interface::Torque||mode==Interface::JointTarget;}
CUDARL_HD inline bool valid_gains(const Gains& gains,Interface mode) {
    for(int j=0;j<4;++j) if(!motor::finite_number(gains.kp[j])||!motor::finite_number(gains.kd[j])||
        gains.kp[j]<0||gains.kd[j]<0||(mode==Interface::JointTarget&&(gains.kp[j]<=0||gains.kd[j]<=0))) return false;
    return motor_history::valid_interface(mode);
}
CUDARL_HD inline bool finite(const State& state) {
    if(!motor::finite(state.task)||!motor::finite_number(state.mean_squared_torque)||state.mean_squared_torque<0||state.mean_squared_torque>1.0001f||!motor::finite_number(state.substep_max_target_error)||!motor::finite_number(state.substep_max_joint_rate)||state.substep_max_target_error<0||state.substep_max_joint_rate<0||state.first_unsafe_substep< -1||state.first_unsafe_substep>10||state.unsafe_substeps<0||state.unsafe_substeps>11) return false;
    for(int j=0;j<4;++j) if(!motor::finite_number(state.held_target[j])||fabsf(state.held_target[j])>deep::limit) return false;
    for(int j=0;j<40;++j) if(!motor::finite_number(state.actual_torques[j])||fabsf(state.actual_torques[j])>1||!motor::finite_number(state.unclamped_torques[j])) return false;
    return state.first_unsafe_substep<0||motor::finite(state.first_unsafe_task);
}
CUDARL_HD inline float physical_clearance(const deep::BodyState& body) {
    const auto geometry=deep::shape(body);float gap=1e6f;
    const deep::V normals[6]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    for(int link=0;link<3;++link) {
        const auto& l=geometry.link[link];
        if(link==0) {
            const float planes[6]={l.center.x+4.5f*deep::scale,4.5f*deep::scale-l.center.x,l.center.y+4.f*deep::scale,
                -l.center.y,l.center.z+3.5f*deep::scale,3.5f*deep::scale-l.center.z};
            for(int face=0;face<6;++face) {
                const auto n=l.q.inverse(normals[face]);const float x=n.x*l.length*.5f,y=n.y*.0064f,z=n.z*.008f;
                gap=fminf(gap,planes[face]-sqrtf(x*x+y*y+z*z));
            }
        }
        for(int axial=0;axial<9;++axial) {
            const float f=(axial-4)*.25f;const auto center=l.center+l.q.rotate({f*l.length*.5f,0,0});
            const float planes[6]={center.x+4.5f*deep::scale,4.5f*deep::scale-center.x,center.y+4.f*deep::scale,
                -center.y,center.z+3.5f*deep::scale,3.5f*deep::scale-center.z};
            for(int face=0;face<6;++face) gap=fminf(gap,planes[face]-deep::arch_slice_support(l,link,f,normals[face]));
        }
    }
    return gap;
}

inline uint64_t fingerprint(const std::vector<uint8_t>& bytes) {return motor::fingerprint(bytes);}
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
    if(!motor_history::valid_gains(result,Interface::JointTarget)) throw std::runtime_error("Invalid derived joint gains");return result;
}
inline bool same_gains(const Gains& a,const Gains& b) {for(int j=0;j<4;++j) if(a.kp[j]!=b.kp[j]||a.kd[j]!=b.kd[j]) return false;return true;}
struct RegulationProtocol {
    std::array<RegulationCase,8> catalog{};Gains gains{};uint64_t source_fingerprint=0,fixture_fingerprint=0;
    void write_catalog(checkpoint::Writer& out) const {for(const auto& item:catalog) {out.value(item.initial);for(float q:item.target) out.value(q);out.value(item.source_fixture);out.value(item.target_clipped);}}
    uint64_t catalog_fingerprint() const {checkpoint::Writer out;write_catalog(out);return fingerprint(out.data);}
    void validate() const {
        if(!source_fingerprint||!fixture_fingerprint||!motor_history::same_gains(gains,motor_history::derive_gains(catalog))) throw std::runtime_error("Servo protocol source or derived gains differ");
        for(const auto& item:catalog) {
            if(!finite(item.initial)||item.initial.task.steps!=0||physical_clearance(item.initial.task.body)<0) throw std::runtime_error("Unsafe servo regulation catalog");
            for(float target:item.target) if(!motor::finite_number(target)||fabsf(target)>deep::limit) throw std::runtime_error("Invalid servo regulation target");
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
        protocol.validate();if(!source||source_fingerprint!=source||protocol.source_fingerprint!=source||catalog_fingerprint!=protocol.catalog_fingerprint()||protocol_payload_hash!=protocol.payload_fingerprint()||!motor_history::same_gains(gains,protocol.gains)) throw std::runtime_error("Paired servo gate source/catalog/gain binding differs");
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

}
#endif
