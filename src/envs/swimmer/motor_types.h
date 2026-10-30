#ifndef CUDARL_MOTOR_TYPES_H
#define CUDARL_MOTOR_TYPES_H
#include "deep_physics.h"
#include "swimmer.h"
#include "core/checkpoint.h"
#include <cstring>

namespace cudarl::motor {
constexpr int kObs=59,kAct=4,kEpisodeSteps=600;
constexpr float body_length=.072f,velocity_scale=.02f;
struct Request {deep::V position{},velocity{},heading{1,0,0};};
using Command=Request;
// Stable request/body layout used inside complete motor checkpoints.
struct TaskState {
    deep::BodyState body;Request request;float previous_action[kAct]{};
    uint32_t seed=1,environment=0,episode=0;int steps=0;
};
struct Result {float reward=0;bool timeout=false,invalid=false,unsafe=false;};
enum class Interface:int {Torque=0,JointTarget=1};
struct Gains {float kp[4]{},kd[4]{};};
CUDARL_HD inline bool finite_number(float x) {return x==x&&fabsf(x)<1e30f;}
CUDARL_HD inline bool finite_vector(deep::V v) {return finite_number(v.x)&&finite_number(v.y)&&finite_number(v.z);}
CUDARL_HD inline bool finite(const TaskState& s) {
    if(!finite_vector(s.body.position)||!finite_vector(s.request.position)||!finite_vector(s.request.velocity)||!finite_vector(s.request.heading)) return false;
    const auto q=s.body.orientation;const float q2=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
    if(!finite_number(q2)||fabsf(q2-1.f)>.01f||fabsf(deep::dot(s.request.heading,s.request.heading)-1.f)>.01f) return false;
    for(float v:s.body.velocity) if(!finite_number(v)) return false;
    for(int j=0;j<kAct;++j) if(!finite_number(s.body.angle[j])||!finite_number(s.previous_action[j])) return false;
    return finite_number(s.body.work)&&finite_number(s.body.travel)&&s.steps>=0;
}
CUDARL_HD inline void observe(const TaskState& s,float* out) {
    for(int i=0;i<kObs;++i) out[i]=0;const auto q=s.body.orientation;
    const deep::V p=q.inverse((s.request.position-s.body.position)*deep::scale)/body_length;
    const deep::V v=q.inverse({s.body.velocity[0],s.body.velocity[1],s.body.velocity[2]});
    const deep::V w=q.inverse({s.body.velocity[3],s.body.velocity[4],s.body.velocity[5]});
    const deep::V up=q.inverse({0,1,0}),h=q.inverse(s.request.heading);
    const deep::V velocity_error=q.inverse(s.request.velocity-deep::V{s.body.velocity[0],s.body.velocity[1],s.body.velocity[2]})/velocity_scale;
    out[0]=p.x;out[1]=p.y;out[2]=p.z;out[3]=deep::clamp(1.f-static_cast<float>(s.steps)/kEpisodeSteps,0,1);
    out[9]=v.x/.04f;out[10]=v.y/.04f;out[11]=v.z/.04f;out[12]=w.x/4;out[13]=w.y/4;out[14]=w.z/4;
    out[15]=up.x;out[16]=up.y;out[17]=up.z;
    for(int j=0;j<kAct;++j) {out[18+j]=s.body.angle[j]/deep::limit;out[22+j]=s.body.velocity[6+j]/12.f;out[29+j]=s.previous_action[j];}
    out[26]=s.body.position.x/4.5f;out[27]=(s.body.position.y+2)/2;out[28]=s.body.position.z/3.5f;
    out[33]=velocity_error.x;out[34]=velocity_error.y;out[35]=velocity_error.z;out[55]=1;out[56]=h.x;out[57]=h.y;out[58]=h.z;
}
CUDARL_HD inline bool valid_interface(Interface mode) {return mode==Interface::Torque||mode==Interface::JointTarget;}
CUDARL_HD inline bool valid_gains(const Gains& gains,Interface mode) {
    for(int j=0;j<4;++j) if(!finite_number(gains.kp[j])||!finite_number(gains.kd[j])||gains.kp[j]<0||gains.kd[j]<0||
        (mode==Interface::JointTarget&&(gains.kp[j]<=0||gains.kd[j]<=0))) return false;return valid_interface(mode);
}
inline bool same_gains(const Gains& a,const Gains& b) {for(int j=0;j<4;++j) if(a.kp[j]!=b.kp[j]||a.kd[j]!=b.kd[j]) return false;return true;}
// Exact IEEE754 values from the completed eight-case CPU/CUDA regulation proof.
inline Gains validated_gains() {
    constexpr uint32_t bits[8]={0x36772E0Bu,0x366077A1u,0x357478F5u,0x35C0F964u,0x351D5C0Eu,0x350EE67Bu,0x341BA2D2u,0x3475B3CDu};Gains out;
    for(int j=0;j<4;++j) {std::memcpy(&out.kp[j],bits+j,sizeof(float));std::memcpy(&out.kd[j],bits+4+j,sizeof(float));}return out;
}
inline uint64_t fingerprint(const std::vector<uint8_t>& bytes) {uint64_t h=14695981039346656037ull;for(uint8_t x:bytes) {h^=x;h*=1099511628211ull;}return h;}
}
#endif
