#ifndef CUDARL_MOTOR_CONTACT_V2_H
#define CUDARL_MOTOR_CONTACT_V2_H
#include "command_motor.h"
#include <array>

// An opt-in rectangular motor laboratory. Legacy pond physics is untouched.
namespace cudarl::motor_contact_v2 {
constexpr uint32_t kPhysicsVersion=2;
constexpr float kOverlapTolerance=.0005f,kContactStiffness=.8f;
constexpr float kContactDampingRatio=1.f,kSubstep=.005f;
struct Inspection {
    float floor_gap=1e6f,min_other_gap=1e6f,min_gap=1e6f,max_floor_gap=-1e6f;
    bool contact=false,floor_contact=false,excessive=false,escape=false,invalid=false;
};
struct Telemetry {float floor_force=0,total_force=0;int points=0;};
CUDARL_HD inline bool supported_physics(const deep::Physics& p) {
    return !p.pond&&!p.habitat.arch&&p.basin.layout==0;
}
CUDARL_HD inline bool finite_body(const deep::BodyState& b) {
    if(!command_motor::finite_vector(b.position)) return false;
    const auto q=b.orientation;const float n=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
    if(!command_motor::finite_number(n)||fabsf(n-1.f)>.01f) return false;
    for(float v:b.velocity) if(!command_motor::finite_number(v)) return false;
    for(float qj:b.angle) if(!command_motor::finite_number(qj)) return false;
    return command_motor::finite_number(b.work)&&command_motor::finite_number(b.travel);
}
CUDARL_HD inline deep::V normal(int face) {
    switch(face) {case 0:return {1,0,0};case 1:return {-1,0,0};case 2:return {0,1,0};case 3:return {0,-1,0};case 4:return {0,0,1};default:return {0,0,-1};}
}
CUDARL_HD inline float offset(int face) {return face<2?4.5f*deep::scale:(face==2?4.f*deep::scale:(face==3?0.f:3.5f*deep::scale));}
// Minimum-support point of the existing ellipsoid / elliptical cylinder.
// Tail ends are not interior axial samples, and radial support is directional.
CUDARL_HD inline deep::V support_offset(const deep::Link& link,int body,deep::V n,int endpoint=0) {
    const auto u=link.q.inverse(n);deep::V local{};
    if(body==0) {
        const float a=link.length*.5f,b=.0064f,c=.008f;
        const float d=sqrtf(a*a*u.x*u.x+b*b*u.y*u.y+c*c*u.z*u.z);
        if(d>0) local={-a*a*u.x/d,-b*b*u.y/d,-c*c*u.z/d};
    } else {
        const float ry=body==2?.006f:.0015f,rz=.0015f;
        const float d=sqrtf(ry*ry*u.y*u.y+rz*rz*u.z*u.z);
        local.x=(endpoint==0?-.5f:.5f)*link.length;
        if(d>0) {local.y=-ry*ry*u.y/d;local.z=-rz*rz*u.z/d;}
    }
    return link.q.rotate(local);
}
CUDARL_HD inline Inspection inspect(const deep::BodyState& state) {
    Inspection out;out.invalid=!finite_body(state);if(out.invalid) {out.excessive=true;return out;}
    const auto shape=deep::shape(state);float maximum[6]={-1e6f,-1e6f,-1e6f,-1e6f,-1e6f,-1e6f};
    for(int body=0;body<3;++body) {
        const auto& link=shape.link[body];
        for(int face=0;face<6;++face) {
            const auto n=normal(face);const int points=body==0?1:2;
            for(int end=0;end<points;++end) {
                const float gap=deep::dot(n,link.center+support_offset(link,body,n,end))+offset(face);
                const float high=deep::dot(n,link.center+support_offset(link,body,n*(-1.f),end))+offset(face);
                out.min_gap=fminf(out.min_gap,gap);maximum[face]=fmaxf(maximum[face],high);
                if(face==2) {out.floor_gap=fminf(out.floor_gap,gap);out.max_floor_gap=fmaxf(out.max_floor_gap,high);}else out.min_other_gap=fminf(out.min_other_gap,gap);
                if(gap<=0) {out.contact=true;if(face==2) out.floor_contact=true;}
            }
        }
    }
    out.excessive=out.min_gap< -kOverlapTolerance;
    for(float high:maximum) if(high< -kOverlapTolerance) out.escape=true;
    return out;
}
CUDARL_HD inline deep::System assemble(const deep::BodyState& state,deep::Physics physical,Telemetry* telemetry=nullptr) {
    auto free_physics=physical;free_physics.boundaries=false;auto system=deep::assemble(state,free_physics);
    if(telemetry) *telemetry={};if(!physical.boundaries) return system;
    const auto shape=deep::shape(state);
    for(int body=0;body<3;++body) {
        const auto& link=shape.link[body];const int points=body==0?1:2;const float weight=1.f/points;
        const float stiffness=kContactStiffness*weight;
        const float damping=2.f*kContactDampingRatio*sqrtf(kContactStiffness*link.mass)*weight;
        for(int face=0;face<6;++face) for(int end=0;end<points;++end) {
            const auto n=normal(face),r=support_offset(link,body,n,end),point=link.center+r;
            const float gap=deep::dot(n,point)+offset(face);
            if(gap<0) {
                deep::V J[deep::dofs];for(int i=0;i<deep::dofs;++i) J[i]=link.linear[i]+deep::cross(link.angular[i],r);
                const float force=-stiffness*gap;deep::load(system,J,n*force);
                if(deep::dot(deep::multiply(J,state.velocity),n)<0) deep::resist(system,J,n,damping);
                if(telemetry) {telemetry->total_force+=force;++telemetry->points;if(face==2) telemetry->floor_force+=force;}
            }
        }
    }
    return system;
}
// Identical legacy semi-implicit integration algebra; only assemble() differs.
// No projection, position correction, target overwrite or non-contact force.
CUDARL_HD inline void substep(deep::BodyState& state,const float* action,float h,deep::Physics physical={},Telemetry* telemetry=nullptr) {
    auto a=assemble(state,physical,telemetry);
    for(int j=0;j<4;++j) {
        const float displacement=state.angle[j]-deep::clamp(state.angle[j],-deep::limit,deep::limit);
        a.force[6+j]+=deep::torque*deep::clamp(action[j],-1,1)-1e-4f*displacement;
        a.R[6+j][6+j]+=deep::torque/12.f+(displacement!=0?3e-6f:0.f);
    }
    auto& L=a.R;float rhs[deep::dofs]{},v[deep::dofs]{};
    for(int i=0;i<deep::dofs;++i) {
        rhs[i]=h*(a.force[i]-a.C[i]);for(int j=0;j<deep::dofs;++j) rhs[i]+=a.M[i][j]*state.velocity[j];
        for(int j=0;j<=i;++j) {float x=a.M[i][j]+h*a.R[i][j];for(int k=0;k<j;++k) x-=L[i][k]*L[j][k];L[i][j]=i==j?sqrtf(fmaxf(x,1e-18f)):x/L[j][j];}
    }
    for(int i=0;i<deep::dofs;++i) {float x=rhs[i];for(int j=0;j<i;++j) x-=L[i][j]*v[j];v[i]=x/L[i][i];}
    for(int i=deep::dofs-1;i>=0;--i) {float x=v[i];for(int j=i+1;j<deep::dofs;++j) x-=L[j][i]*v[j];v[i]=x/L[i][i];}
    state.position+=deep::V{v[0],v[1],v[2]}*(h/deep::scale);const deep::V omega{v[3],v[4],v[5]};const float speed=deep::norm(omega);
    if(speed>1e-10f) state.orientation=deep::axis_rotation(omega/speed,h*speed)*state.orientation;
    state.orientation.normalize();for(int j=0;j<4;++j) {state.angle[j]+=h*v[6+j];state.work+=h*fmaxf(0,deep::torque*deep::clamp(action[j],-1,1)*v[6+j]);}
    for(int i=0;i<deep::dofs;++i) state.velocity[i]=v[i];state.travel+=h*deep::norm({v[0],v[1],v[2]})/deep::scale;
}
inline uint64_t fingerprint(const std::vector<uint8_t>& bytes) {uint64_t h=14695981039346656037ull;for(uint8_t x:bytes) {h^=x;h*=1099511628211ull;}return h;}
inline std::vector<uint8_t> physics_protocol() {
    checkpoint::Writer out;out.value(kPhysicsVersion);out.value(kOverlapTolerance);out.value(kContactStiffness);out.value(kContactDampingRatio);out.value(kSubstep);
    out.array(std::vector<float>{.020f,.026f,.026f,.0064f,.008f,.0015f,.006f,4.5f*deep::scale,4.f*deep::scale,3.5f*deep::scale,deep::torque,deep::limit,1.f,.5f,.5f});
    return out.data;
}
inline uint64_t physics_fingerprint() {return fingerprint(physics_protocol());}
struct ContactGate {
    uint64_t physics_hash=physics_fingerprint(),source_failure_hash=0,protocol_hash=0;
    int cpu_substeps=0,cuda_substeps=0,static_checks=0;
    bool completed=false,geometry=false,response=false,support=false,timestep=false,parity=false,freewater=false;
    bool passed() const {return completed&&geometry&&response&&support&&timestep&&parity&&freewater&&physics_hash==physics_fingerprint()&&source_failure_hash&&protocol_hash;}
    void require_passed() const {if(!passed()) throw std::runtime_error("Contact-v2 physical readiness did not pass");}
    std::vector<uint8_t> serialize() const {
        checkpoint::Writer out;out.value(uint32_t{0x4D434732u});out.value(kPhysicsVersion);out.value(physics_hash);out.value(source_failure_hash);out.value(protocol_hash);out.value(cpu_substeps);out.value(cuda_substeps);out.value(static_checks);
        for(bool flag:{completed,geometry,response,support,timestep,parity,freewater}) out.value(flag);return out.data;
    }
    static ContactGate read(const std::vector<uint8_t>& bytes) {
        checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=0x4D434732u||in.value<uint32_t>()!=kPhysicsVersion) throw std::runtime_error("Incompatible contact-v2 gate");ContactGate out;
        out.physics_hash=in.value<uint64_t>();out.source_failure_hash=in.value<uint64_t>();out.protocol_hash=in.value<uint64_t>();out.cpu_substeps=in.value<int>();out.cuda_substeps=in.value<int>();out.static_checks=in.value<int>();out.completed=in.value<bool>();out.geometry=in.value<bool>();out.response=in.value<bool>();out.support=in.value<bool>();out.timestep=in.value<bool>();out.parity=in.value<bool>();out.freewater=in.value<bool>();in.finish();
        if(out.physics_hash!=physics_fingerprint()||out.cpu_substeps<0||out.cuda_substeps<0||out.cpu_substeps+out.cuda_substeps>113600||out.static_checks<0) throw std::runtime_error("Contact-v2 gate binding differs");return out;
    }
};
struct ContactFrame {
    int backend=0,case_id=0,step=0,substeps=0;float h=.005f,time_s=0;
    deep::BodyState before{},after{};deep::Physics physical{};float action[4]{};
    Inspection inspection{};Telemetry contact{};
};
inline std::vector<ContactFrame> read_frames(const std::vector<uint8_t>& bytes,const ContactGate& gate) {
    checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=0x4D434632u||in.value<uint32_t>()!=kPhysicsVersion||in.value<uint32_t>()!=sizeof(ContactFrame)||in.value<uint64_t>()!=gate.physics_hash||in.value<uint64_t>()!=gate.protocol_hash) throw std::runtime_error("Incompatible contact-v2 recorded frames");
    auto frames=in.array<ContactFrame>(10000);in.finish();for(const auto& frame:frames) if(frame.backend<0||frame.backend>1||frame.case_id<0||!finite_body(frame.before)||!finite_body(frame.after)||frame.h<=0||frame.substeps<=0) throw std::runtime_error("Invalid contact-v2 recorded frame");return frames;
}
}
#endif
