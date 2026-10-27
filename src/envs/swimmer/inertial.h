#ifndef CUDARL_ENVS_SWIMMER_INERTIAL_H
#define CUDARL_ENVS_SWIMMER_INERTIAL_H
#include <cmath>
#include "core/hd.h"

// Planar articulated rigid-body dynamics in SI units. Three rigid links have
// five generalized DOFs: translation, heading and two relative joint angles.
// Joint torques are internal. Passive viscous/quadratic water resistance is
// integrated implicitly; prescribed velocity and direct thrust are absent.
// This is a reduced hydrodynamic model, not CFD or a biological calibration.
namespace cudarl::swimmer::inertial {
constexpr float kScale=0.04f; // 1.8 model units = 7.2 cm total body length
constexpr float kTorque=1e-5f; // N m per joint; an explicit actuator assumption
struct Point { float x[5]{},y[5]{}; };
struct System { float M[5][5]{},R[5][5]{},C[5]{},v[5]{}; };

CUDARL_HD inline void add_mass(System& d,const Point& J,const float* angular,
                               float mass,float inertia,float ax,float ay) {
    for(int i=0;i<5;++i) {
        d.C[i]+=mass*(J.x[i]*ax+J.y[i]*ay);
        for(int j=0;j<5;++j)
            d.M[i][j]+=mass*(J.x[i]*J.x[j]+J.y[i]*J.y[j])+inertia*angular[i]*angular[j];
    }
}
CUDARL_HD inline void add_drag(System& d,const Point& J,float tx,float ty,
                               float area_t,float area_n,float viscous,float density) {
    float vx=0,vy=0;
    for(int i=0;i<5;++i) {vx+=J.x[i]*d.v[i];vy+=J.y[i]*d.v[i];}
    const float vt=vx*tx+vy*ty,vn=-vx*ty+vy*tx;
    const float ct=viscous+0.5f*density*area_t*fabsf(vt);
    const float cn=viscous+0.5f*density*area_n*fabsf(vn);
    for(int i=0;i<5;++i) for(int j=0;j<5;++j) {
        const float it=J.x[i]*tx+J.y[i]*ty,in=-J.x[i]*ty+J.y[i]*tx;
        const float jt=J.x[j]*tx+J.y[j]*ty,jn=-J.x[j]*ty+J.y[j]*tx;
        d.R[i][j]+=ct*it*jt+cn*in*jn;
    }
}

template<class State>
CUDARL_HD inline System assemble(const State& s,float density=1000.f,float viscosity=0.001f) {
    System d;
    d.v[0]=s.velocity[0]*kScale;d.v[1]=s.velocity[1]*kScale;d.v[2]=s.velocity[2];
    d.v[3]=s.rate[0];d.v[4]=s.rate[1];
    const float length[3]={s.body.head*kScale,s.body.proximal*kScale,s.body.distal*kScale};
    const float phi[3]={0,s.angle[0],s.angle[0]+s.angle[1]};
    const float speed[3]={d.v[2],d.v[2]+d.v[3],d.v[2]+d.v[3]+d.v[4]};
    const float pi=3.141592653589793f;
    const float head_a=length[0]*0.5f,head_b=0.0064f,head_c=0.008f,core_radius=0.0015f;
    const float head_mass=1000.f*(4.f*pi/3.f)*head_a*head_b*head_c;
    for(int link=0;link<3;++link) {
        Point J;J.x[0]=1;J.y[1]=1;
        float angular[5]={0,0,1,link>0?1.f:0.f,link>1?1.f:0.f};
        float ax=0,ay=0;
        // Center position = sum of negative link-direction offsets. Its second
        // derivative at zero generalized acceleration supplies Coriolis terms.
        for(int segment=0;segment<=link;++segment) {
            if(link==0) break;
            const float offset=segment==0?length[0]*0.5f:(segment==link?length[segment]*0.5f:length[segment]);
            const float c=cosf(phi[segment]),n=sinf(phi[segment]);
            for(int k=2;k<5;++k) if(k<=2+segment) {
                J.x[k]+=offset*n;J.y[k]-=offset*c;
            }
            ax+=offset*c*speed[segment]*speed[segment];ay+=offset*n*speed[segment]*speed[segment];
        }
        const float fin_height=link==2?fmaxf(0.f,s.body.fin_normal-2.f)*0.00485f:0.f;
        const float volume=pi*core_radius*core_radius*length[link]+fin_height*length[link]*0.0001f;
        const float mass=link==0?head_mass:1000.f*volume;
        const float inertia=link==0?mass*(head_a*head_a+head_c*head_c)/5.f:mass*(length[link]*length[link]+4.f*core_radius*core_radius)/12.f;
        add_mass(d,J,angular,mass,inertia,ax,ay);
        const float c=cosf(phi[link]),n=sinf(phi[link]);
        const float area_t=link==0?0.3f*pi*head_b*head_c:0.1f*(pi*core_radius*core_radius+fin_height*0.0001f);
        const float area_n=link==0?0.8f*pi*head_a*head_b:1.2f*length[link]*(2.f*core_radius+fin_height);
        const float viscous=link==0?6.f*pi*viscosity*head_c:4.f*pi*viscosity*length[link]/(logf(2.f*length[link]/core_radius)-0.5f);
        for(int q=0;q<3;++q) {
            const float offset=(static_cast<float>(q)+0.5f)/3.f*length[link]-0.5f*length[link];
            Point point=J;
            for(int k=2;k<5;++k) {point.x[k]-=offset*n*angular[k];point.y[k]+=offset*c*angular[k];}
            add_drag(d,point,c,n,area_t/3.f,area_n/3.f,viscous/3.f,density);
        }
    }
    return d;
}
template<class State>
CUDARL_HD inline float kinetic_energy(const State& s) {
    const auto d=assemble(s,0,0);float result=0;
    for(int i=0;i<5;++i) for(int j=0;j<5;++j) result+=0.5f*d.v[i]*d.M[i][j]*d.v[j];
    return result;
}
template<class State>
CUDARL_HD inline void step(State& s,const float* action,float dt,float limit,float density=1000.f,float viscosity=0.001f) {
    auto d=assemble(s,density,viscosity);
    float force[5]{},drive[2]{};
    for(int joint=0;joint<2;++joint) {
        const float command=fminf(1.f,fmaxf(-1.f,action[joint]));
        const float displacement=s.angle[joint]-fminf(limit,fmaxf(-limit,s.angle[joint]));
        drive[joint]=kTorque*command;
        force[3+joint]=drive[joint]-1e-4f*displacement;
        // Passive joint/motor resistance; it cannot add mechanical energy.
        d.R[3+joint][3+joint]+=kTorque/12.f+(displacement!=0?3e-6f:0.f);
    }
    float A[5][5]{},b[5]{},L[5][5]{},v[5]{};
    for(int i=0;i<5;++i) {
        b[i]=dt*(force[i]-d.C[i]);
        for(int j=0;j<5;++j) {A[i][j]=d.M[i][j]+dt*d.R[i][j];b[i]+=d.M[i][j]*d.v[j];}
    }
    for(int i=0;i<5;++i) for(int j=0;j<=i;++j) {
        float value=A[i][j];for(int k=0;k<j;++k) value-=L[i][k]*L[j][k];
        L[i][j]=i==j?sqrtf(fmaxf(value,1e-18f)):value/L[j][j];
    }
    for(int i=0;i<5;++i) {float value=b[i];for(int j=0;j<i;++j) value-=L[i][j]*v[j];v[i]=value/L[i][i];}
    for(int i=4;i>=0;--i) {float value=v[i];for(int j=i+1;j<5;++j) value-=L[j][i]*v[j];v[i]=value/L[i][i];}
    const float c=cosf(s.heading),n=sinf(s.heading),rotation=dt*v[2];
    s.x+=dt*(c*v[0]-n*v[1])/kScale;s.y+=dt*(n*v[0]+c*v[1])/kScale;
    s.heading+=rotation;s.heading=atan2f(sinf(s.heading),cosf(s.heading));
    s.velocity[0]=(cosf(rotation)*v[0]+sinf(rotation)*v[1])/kScale;
    s.velocity[1]=(-sinf(rotation)*v[0]+cosf(rotation)*v[1])/kScale;s.velocity[2]=v[2];
    for(int joint=0;joint<2;++joint) {s.rate[joint]=v[3+joint];s.angle[joint]+=dt*s.rate[joint];}
    s.work+=dt*(fmaxf(0.f,drive[0]*v[3])+fmaxf(0.f,drive[1]*v[4]));
    s.travel+=dt*sqrtf(v[0]*v[0]+v[1]*v[1])/kScale;s.strokes+=dt*fabsf(v[4])/(4.f*limit);
}
}
#endif
