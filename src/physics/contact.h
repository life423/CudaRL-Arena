#ifndef CUDARL_PHYSICS_CONTACT_H
#define CUDARL_PHYSICS_CONTACT_H
#include <cstdint>
#include "dynamics.h"

// Same full-envelope spring/damping response as baseline e5e9bd1 motor_contact.h.
// KNOWN UNRESOLVED: rolled negative-buoyancy support breached 0.5 mm in the
// registered contact study. Copying this core does not turn that failure into a pass.
namespace cudarl::contact {
constexpr bool kRolledSupportKnownUnresolved=true;
constexpr uint32_t kPhysicsVersion=2;
constexpr float kOverlapTolerance=.0005f,kContactStiffness=.8f;
constexpr float kContactDampingRatio=1.f,kSubstep=.005f;
struct Inspection {
    float floor_gap=1e6f,min_other_gap=1e6f,min_gap=1e6f,max_floor_gap=-1e6f;
    bool contact=false,floor_contact=false,excessive=false,escape=false,invalid=false;
};
struct Telemetry {float floor_force=0,total_force=0;int points=0;};
CUDARL_HD inline physics::V normal(int face) {
    switch(face) {case 0:return {1,0,0};case 1:return {-1,0,0};case 2:return {0,1,0};case 3:return {0,-1,0};case 4:return {0,0,1};default:return {0,0,-1};}
}
CUDARL_HD inline float offset(int face) {return face<2?4.5f*physics::scale:(face==2?4.f*physics::scale:(face==3?0.f:3.5f*physics::scale));}
// Minimum-support point of the existing ellipsoid / elliptical cylinder.
// Tail ends are not interior axial samples, and radial support is directional.
CUDARL_HD inline physics::V support_offset(const physics::Link& link,int body,physics::V n,int endpoint=0) {
    const auto u=link.q.inverse(n);physics::V local{};
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
CUDARL_HD inline Inspection inspect(const physics::BodyState& state) {
    Inspection out;out.invalid=!physics::finite_body(state);if(out.invalid) {out.excessive=true;return out;}
    const auto shape=physics::shape(state);float maximum[6]={-1e6f,-1e6f,-1e6f,-1e6f,-1e6f,-1e6f};
    for(int body=0;body<3;++body) {
        const auto& link=shape.link[body];
        for(int face=0;face<6;++face) {
            const auto n=normal(face);const int points=body==0?1:2;
            for(int end=0;end<points;++end) {
                const float gap=physics::dot(n,link.center+support_offset(link,body,n,end))+offset(face);
                const float high=physics::dot(n,link.center+support_offset(link,body,n*(-1.f),end))+offset(face);
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
CUDARL_HD inline physics::System assemble(const physics::BodyState& state,physics::Parameters physical={},physics::V current={},Telemetry* telemetry=nullptr) {
    auto system=physics::assemble(state,physical,current);
    if(telemetry) *telemetry={};
    const auto shape=physics::shape(state);
    for(int body=0;body<3;++body) {
        const auto& link=shape.link[body];const int points=body==0?1:2;const float weight=1.f/points;
        const float stiffness=kContactStiffness*weight;
        const float damping=2.f*kContactDampingRatio*sqrtf(kContactStiffness*link.mass)*weight;
        for(int face=0;face<6;++face) for(int end=0;end<points;++end) {
            const auto n=normal(face),r=support_offset(link,body,n,end),point=link.center+r;
            const float gap=physics::dot(n,point)+offset(face);
            if(gap<0) {
                physics::V J[physics::dofs];for(int i=0;i<physics::dofs;++i) J[i]=link.linear[i]+physics::cross(link.angular[i],r);
                const float force=-stiffness*gap;physics::load(system,J,n*force);
                if(physics::dot(physics::multiply(J,state.velocity),n)<0) physics::resist(system,J,n,damping);
                if(telemetry) {telemetry->total_force+=force;++telemetry->points;if(face==2) telemetry->floor_force+=force;}
            }
        }
    }
    return system;
}
// Identical legacy semi-implicit integration algebra; only assemble() differs.
// No projection, position correction, target overwrite or non-contact force.
CUDARL_HD inline void substep(physics::BodyState& state,const float* action,float h,physics::Parameters physical={},physics::V current={},Telemetry* telemetry=nullptr) {
    auto a=assemble(state,physical,current,telemetry);
    for(int j=0;j<4;++j) {
        const float displacement=state.angle[j]-physics::clamp(state.angle[j],-physics::limit,physics::limit);
        a.force[6+j]+=physics::torque*physics::clamp(action[j],-1,1)-1e-4f*displacement;
        a.R[6+j][6+j]+=physics::torque/12.f+(displacement!=0?3e-6f:0.f);
    }
    auto& L=a.R;float rhs[physics::dofs]{},v[physics::dofs]{};
    for(int i=0;i<physics::dofs;++i) {
        rhs[i]=h*(a.force[i]-a.C[i]);for(int j=0;j<physics::dofs;++j) rhs[i]+=a.M[i][j]*state.velocity[j];
        for(int j=0;j<=i;++j) {float x=a.M[i][j]+h*a.R[i][j];for(int k=0;k<j;++k) x-=L[i][k]*L[j][k];L[i][j]=i==j?sqrtf(fmaxf(x,1e-18f)):x/L[j][j];}
    }
    for(int i=0;i<physics::dofs;++i) {float x=rhs[i];for(int j=0;j<i;++j) x-=L[i][j]*v[j];v[i]=x/L[i][i];}
    for(int i=physics::dofs-1;i>=0;--i) {float x=v[i];for(int j=i+1;j<physics::dofs;++j) x-=L[j][i]*v[j];v[i]=x/L[i][i];}
    state.position+=physics::V{v[0],v[1],v[2]}*(h/physics::scale);const physics::V omega{v[3],v[4],v[5]};const float speed=physics::norm(omega);
    if(speed>1e-10f) state.orientation=physics::axis_rotation(omega/speed,h*speed)*state.orientation;
    state.orientation.normalize();for(int j=0;j<4;++j) {state.angle[j]+=h*v[6+j];state.work+=h*fmaxf(0,physics::torque*physics::clamp(action[j],-1,1)*v[6+j]);}
    for(int i=0;i<physics::dofs;++i) state.velocity[i]=v[i];state.travel+=h*physics::norm({v[0],v[1],v[2]})/physics::scale;
}
}
#endif
